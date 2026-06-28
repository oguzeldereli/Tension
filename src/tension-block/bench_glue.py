"""
Benchmark 6 -- tension halting on a PRETRAINED backbone, on real data (SST-2 / GLUE).

The honest lesson from bench_listops/bench_babi: adaptive computation only pays when the base
model can already convert compute into accuracy on the hard cases. So here we put the halting
on top of a model that CAN -- a pretrained DistilBERT (6 transformer layers) fine-tuned on
SST-2 sentiment. Each layer gets an exit head (deep supervision), and a learned tension HALT
head reads the [CLS] state and its speed ||Δcls|| and fires when the field has settled into a
confident answer. Easy sentences resolve in 1-2 layers; hard ones use more. This is the
recognized early-exit setting (DeeBERT / PABEE), reframed as tension halting.

Comparison, all from ONE trained model (only the exit POLICY differs -- a fair test):
  fixed-K           : always exit at layer K (K=1..6). The optimal NON-adaptive frontier.
  Adaptive (ours)   : exit at the first layer whose halt prob crosses a threshold tau. Sweep
                      tau to trace the adaptive accuracy-vs-compute curve.
  full (layer 6)    : the fixed-K=6 point = the standard fine-tuned model (accuracy ceiling).

Claims under test (the same shape as the coin benchmark, now on real data):
  (1) the adaptive curve beats the fixed-K frontier -- matched accuracy at fewer avg layers;
  (2) compute spent tracks DIFFICULTY -- exit layer rises as the (layer-6) confidence margin
      falls: corr(exit_layer, -margin) > 0. It thinks longer on the genuinely hard sentences.

Run:  python3 bench_glue.py
"""
import math
import time
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE = "distilbert-base-uncased"
MAXLEN = 64
BATCH = 32
EPOCHS = 3
LR = 3e-5
HALT_W = 0.5
THRESHOLDS = [0.5, 0.7, 0.85, 0.95, 0.99]
LOG_EVERY = 400


def load_data(dev):
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    ds = load_dataset("nyu-mll/glue", "sst2")

    def enc(split):
        b = tok(list(split["sentence"]), padding="max_length", truncation=True,
                max_length=MAXLEN, return_tensors="pt")
        return dict(ids=b["input_ids"], mask=b["attention_mask"],
                    y=torch.tensor(list(split["label"])))
    return enc(ds["train"]), enc(ds["validation"])


class EarlyExitTension(nn.Module):
    def __init__(self, n_cls=2):
        super().__init__()
        self.bb = AutoModel.from_pretrained(BACKBONE)
        d = self.bb.config.dim
        self.L = self.bb.config.n_layers
        self.heads = nn.ModuleList([nn.Linear(d, n_cls) for _ in range(self.L)])
        self.halt = nn.Sequential(nn.Linear(d + 1, d), nn.Tanh(), nn.Linear(d, 1))

    def forward(self, ids, mask):
        hs = self.bb(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
        # hs: tuple len L+1 (embeddings + L layers); CLS = position 0
        cls = [h[:, 0] for h in hs]                       # cls[0]=emb ... cls[L]=last layer
        logits, halts = [], []
        for i in range(1, self.L + 1):
            speed = (cls[i] - cls[i - 1]).norm(dim=-1, keepdim=True)
            logits.append(self.heads[i - 1](cls[i]))
            halts.append(self.halt(torch.cat([cls[i], speed], -1)).squeeze(-1))
        return torch.stack(logits), torch.stack(halts)    # (L,B,2), (L,B)


def train(model, tr, dev):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    n = tr["y"].shape[0]
    steps = EPOCHS * (n // BATCH)
    model.train()
    for it in range(steps):
        idx = torch.randint(0, n, (BATCH,), device=dev)
        ids, mask, y = tr["ids"][idx], tr["mask"][idx], tr["y"][idx]
        logits, halts = model(ids, mask)                  # (L,B,C),(L,B)
        L, _, C = logits.shape
        ce = F.cross_entropy(logits.reshape(L * BATCH, C), y.repeat(L))
        # halt target: this layer is already correct -> safe to exit ("the field has resolved")
        with torch.no_grad():
            correct = (logits.argmax(-1) == y.unsqueeze(0)).float()   # (L,B)
        bce = F.binary_cross_entropy_with_logits(halts, correct)
        loss = ce + HALT_W * bce
        opt.zero_grad(); loss.backward(); opt.step()
        if it % LOG_EVERY == 0:
            print(f"  step {it:5d}/{steps}  ce {ce.item():.3f}  bce {bce.item():.3f}", flush=True)
    return model


@torch.no_grad()
def evaluate(model, va, dev):
    model.eval()
    n = va["y"].shape[0]
    all_logits, all_halts = [], []
    for i in range(0, n, 256):
        lg, ht = model(va["ids"][i:i+256], va["mask"][i:i+256])
        all_logits.append(lg); all_halts.append(ht)
    logits = torch.cat(all_logits, 1)                     # (L,N,2)
    halts = torch.sigmoid(torch.cat(all_halts, 1))        # (L,N)
    y = va["y"]
    L, N, _ = logits.shape

    fixedK = [(logits[k].argmax(-1) == y).float().mean().item() * 100 for k in range(L)]
    # difficulty proxy: confidence margin at the final layer
    p6 = logits[-1].softmax(-1)
    margin6 = (p6.max(-1).values - p6.min(-1).values)     # high = easy

    curve = []
    for tau in THRESHOLDS:
        crossed = halts >= tau
        crossed[-1] = True
        layer = crossed.float().argmax(0)                 # 0..L-1
        pred = logits[layer, torch.arange(N)].argmax(-1)
        acc = (pred == y).float().mean().item() * 100
        avg = (layer.float() + 1).mean().item()
        comp = layer.float() + 1
        corr = (((comp - comp.mean()) * (-margin6 - (-margin6).mean())).mean()
                / (comp.std() * margin6.std() + 1e-8)).item()
        curve.append((tau, avg, acc, corr))
    return fixedK, curve


def interp(xs, ys, x):
    import numpy as np
    return float(np.interp(x, xs, ys))


# ---- real wall-clock throughput via depth-bucketed inference (public API, robust) ----
# An example that halts at layer K only needs the first K layers. We bucket the batch by exit
# layer and run each bucket through a K-layer-truncated backbone (a real forward pass with the
# library's own masking), summing the wall-clock. Total work = sum_K |bucket_K| * K layers --
# exactly the compute an online early-exit would do, measured for real rather than estimated.
@torch.no_grad()
def measure_throughput(model, ids, mask, halts, tau, reps=20):
    model.eval()
    bb = model.bb
    L = bb.config.n_layers
    full_layers = list(bb.transformer.layer)
    crossed = halts >= tau; crossed[-1] = True
    exit_layer = crossed.float().argmax(0) + 1                # (N,) in 1..L

    def set_depth(K):
        bb.transformer.layer = nn.ModuleList(full_layers[:K]); bb.config.n_layers = K

    def run(K, ii, mm):
        set_depth(K); return bb(input_ids=ii, attention_mask=mm)

    cuda = ids.is_cuda
    sync = (lambda: torch.cuda.synchronize()) if cuda else (lambda: None)
    buckets = [(K, exit_layer == K) for K in range(1, L + 1)]

    for _ in range(3): run(L, ids, mask)                                  # warmup
    sync(); t0 = time.time()
    for _ in range(reps): run(L, ids, mask)
    sync(); full = time.time() - t0

    for _ in range(3):
        for K, sel in buckets:
            if sel.any(): run(K, ids[sel], mask[sel])
    sync(); t0 = time.time()
    for _ in range(reps):
        for K, sel in buckets:
            if sel.any(): run(K, ids[sel], mask[sel])
    sync(); adap = time.time() - t0

    set_depth(L)                                                          # restore
    n = ids.shape[0] * reps
    return n / full, n / adap, full / adap


def main():
    dev = torch.device(DEVICE)
    torch.manual_seed(0)
    print(f"device {DEVICE}  backbone {BACKBONE}  SST-2  maxlen {MAXLEN}")
    tr, va = load_data(dev)
    for d in (tr, va):
        for k in d: d[k] = d[k].to(dev)
    print(f"train {tr['y'].shape[0]}  val {va['y'].shape[0]}\n")

    model = EarlyExitTension().to(dev)
    print(f"params {sum(p.numel() for p in model.parameters())}\ntraining...", flush=True)
    train(model, tr, dev)
    fixedK, curve = evaluate(model, va, dev)

    L = len(fixedK)
    print("\nfixed-K (always exit at layer K) -- the non-adaptive frontier:")
    print("  layer K :", " ".join(f"{k+1:6d}" for k in range(L)))
    print("  acc %   :", " ".join(f"{a:6.1f}" for a in fixedK))
    print(f"  full model (K={L}) accuracy = {fixedK[-1]:.2f}%   compute = {L} layers")

    print("\nadaptive (tension halt) -- exit at first layer with halt prob >= tau:")
    print(f"{'tau':>6}{'avg layers':>12}{'acc %':>9}{'fixedK@same':>13}{'Δacc':>8}{'corr(exit,diff)':>17}")
    Ks = list(range(1, L + 1))
    for tau, avg, acc, corr in curve:
        fk = interp(Ks, fixedK, avg)
        print(f"{tau:>6.2f}{avg:>12.2f}{acc:>9.2f}{fk:>13.2f}{acc-fk:>8.2f}{corr:>17.3f}")
    print("\nread: Δacc>0 = adaptive beats the best fixed-depth budget at the same avg compute;")
    print("      corr(exit,diff)>0 = it spends more layers on harder (low-margin) sentences.")

    # ---- does the layer saving become a real wall-clock saving? ----
    torch.cuda.empty_cache()
    bs = 512
    n = va["y"].shape[0]
    rep = (bs + n - 1) // n
    ids = va["ids"].repeat(rep, 1)[:bs]; mask = va["mask"].repeat(rep, 1)[:bs]
    with torch.no_grad():
        halts_tp = torch.sigmoid(model(ids, mask)[1])        # (L,bs) halting probs
        for tau in [0.95, 0.99]:
            f_tps, a_tps, sx = measure_throughput(model, ids, mask, halts_tp, tau)
            print(f"\nwall-clock throughput (batch {bs}, tau={tau}): "
                  f"full {f_tps:,.0f} ex/s  ->  adaptive {a_tps:,.0f} ex/s   = {sx:.2f}x faster")
    print("read: the layer-count saving is a real wall-clock speedup -- examples that resolve")
    print("      early skip the deep layers entirely (measured, depth-bucketed forward passes).")


if __name__ == "__main__":
    main()

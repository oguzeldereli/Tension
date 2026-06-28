"""
Experiment A -- ABLATION: is the settling signal ||Δcls|| doing anything, or is confidence
doing all the work?

One fine-tuned DistilBERT/SST-2 model with per-layer exit heads (deep supervision). ONLY the
exit POLICY varies; every policy reads the SAME heads of the SAME model, so training budget is
identical by construction:

  1. confidence-threshold (DeeBERT-style; Xin et al. 2020): exit when softmax max-prob >= tau.
  2. PABEE patience (Zhou et al. 2020): exit once the predicted class is unchanged for p
     consecutive layers.
  3. tension-halt (mine): a learned halt head reading [cls, ||Δcls||] -- confidence + settling.
  4. tension-halt ABLATED (confidence-only version of MY head): a SECOND halt head with the
     SAME architecture and SAME training (joint, same BCE-to-correctness target, same budget)
     but WITHOUT the ||Δcls|| feature. This is the fair ablation. We ALSO report (4z): head (3)
     with ||Δcls|| zeroed at inference -- but that is out-of-distribution for a head trained
     with the feature and would *unfairly favour* (3), so it is a flagged sanity point, not the
     comparison.

Fairness: (3) and (4) are trained jointly in the same run with identical targets and budget --
(3) is NOT tuned harder. (1),(2) are parameter-free. Each policy is swept across its knob to
trace the full speed-accuracy frontier (avg layers vs accuracy).

Key comparison: does (3) beat (1),(2),(4) at matched compute?
  (3) ~ (4)  -> the ||Δcls|| term adds nothing; the contribution is the framing, not the signal.
  (3) > (4) and > (1),(2) -> the settling signal is real; we quantify Δacc at matched layers.
A null result is a legitimate result and is reported as such.

Run:  python3 bench_glue_ablation.py
"""
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from bench_glue import load_data, BACKBONE, BATCH, LR
from transformers import AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 3500            # shorter than the headline run (thermal budget); ~1.7 epochs
HALT_W = 0.5
COOLDOWN_EVERY = 350    # let the GPU idle briefly to keep temperatures down
COOLDOWN_SECS = 3
LOG_EVERY = 350


class EarlyExitAblation(nn.Module):
    """Per-layer exit heads + TWO halt heads: halt_full reads [cls, ||Δcls||]; halt_conf reads
    [cls] only (the fair confidence-only ablation of the same head, trained jointly)."""
    def __init__(self, n_cls=2):
        super().__init__()
        self.bb = AutoModel.from_pretrained(BACKBONE)
        d = self.bb.config.dim
        self.L = self.bb.config.n_layers
        self.heads = nn.ModuleList([nn.Linear(d, n_cls) for _ in range(self.L)])
        self.halt_full = nn.Sequential(nn.Linear(d + 1, d), nn.Tanh(), nn.Linear(d, 1))
        self.halt_conf = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))

    def forward(self, ids, mask):
        hs = self.bb(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
        cls = [h[:, 0] for h in hs]                       # cls[0]=emb ... cls[L]=last
        logits, hfull, hfz, hconf = [], [], [], []
        for i in range(1, self.L + 1):
            c = cls[i]
            speed = (c - cls[i - 1]).norm(dim=-1, keepdim=True)
            logits.append(self.heads[i - 1](c))
            hfull.append(self.halt_full(torch.cat([c, speed], -1)).squeeze(-1))
            hfz.append(self.halt_full(torch.cat([c, torch.zeros_like(speed)], -1)).squeeze(-1))
            hconf.append(self.halt_conf(c).squeeze(-1))
        return (torch.stack(logits), torch.stack(hfull),
                torch.stack(hfz), torch.stack(hconf))      # (L,B,2),(L,B),(L,B),(L,B)


def train(model, tr, dev):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    n = tr["y"].shape[0]
    model.train()
    for it in range(STEPS):
        idx = torch.randint(0, n, (BATCH,), device=dev)
        ids, mask, y = tr["ids"][idx], tr["mask"][idx], tr["y"][idx]
        logits, hfull, _, hconf = model(ids, mask)
        L, B, C = logits.shape
        ce = F.cross_entropy(logits.reshape(L * B, C), y.repeat(L))
        with torch.no_grad():
            correct = (logits.argmax(-1) == y.unsqueeze(0)).float()
        bce = (F.binary_cross_entropy_with_logits(hfull, correct)
               + F.binary_cross_entropy_with_logits(hconf, correct))   # both heads, same target
        loss = ce + HALT_W * bce
        opt.zero_grad(); loss.backward(); opt.step()
        if it % LOG_EVERY == 0:
            print(f"  step {it:5d}/{STEPS}  ce {ce.item():.3f}  bce {bce.item():.3f}", flush=True)
        if COOLDOWN_EVERY and it and it % COOLDOWN_EVERY == 0:
            torch.cuda.synchronize(); time.sleep(COOLDOWN_SECS)         # thermal cooldown
    return model


@torch.no_grad()
def gather(model, va, dev):
    model.eval()
    Lg, Hf, Hz, Hc = [], [], [], []
    n = va["y"].shape[0]
    for i in range(0, n, 256):
        lg, hf, hz, hc = model(va["ids"][i:i+256], va["mask"][i:i+256])
        Lg.append(lg); Hf.append(hf); Hz.append(hz); Hc.append(hc)
    return (torch.cat(Lg, 1), torch.sigmoid(torch.cat(Hf, 1)),
            torch.sigmoid(torch.cat(Hz, 1)), torch.sigmoid(torch.cat(Hc, 1)))


def frontier_threshold(score, preds, y, taus):
    """Exit at first layer with score>=tau; sweep tau -> [(avg_layers, acc), ...]."""
    L, N = score.shape
    dev = score.device
    pts = []
    for tau in taus:
        crossed = score >= tau
        crossed[-1] = True
        layer = crossed.float().argmax(0)
        pred = preds[layer, torch.arange(N, device=dev)]
        pts.append((layer.float().mean().item() + 1, (pred == y).float().mean().item() * 100))
    return pts


def frontier_pabee(preds, y):
    """PABEE: exit once the prediction is unchanged for p consecutive layers; sweep p=1..L."""
    L, N = preds.shape
    dev = preds.device
    pts = []
    for p in range(1, L + 1):
        cnt = torch.ones(N, device=dev)
        done = torch.zeros(N, dtype=torch.bool, device=dev)
        layer = torch.full((N,), L - 1, dtype=torch.long, device=dev)
        for i in range(L):
            if i > 0:
                same = preds[i] == preds[i - 1]
                cnt = torch.where(same, cnt + 1, torch.ones_like(cnt))
            fire = (~done) & (cnt >= p)
            layer[fire] = i; done |= fire
        pred = preds[layer, torch.arange(N, device=dev)]
        pts.append((layer.float().mean().item() + 1, (pred == y).float().mean().item() * 100))
    return pts


def at(pts, x):
    a = sorted(pts); xs = [p[0] for p in a]; ys = [p[1] for p in a]
    return float(np.interp(x, xs, ys))


def main():
    dev = torch.device(DEVICE)
    torch.manual_seed(0)
    print(f"device {DEVICE}  Experiment A (ablation)  SST-2  steps {STEPS}")
    tr, va = load_data(dev)
    for d in (tr, va):
        for k in d: d[k] = d[k].to(dev)
    model = EarlyExitAblation().to(dev)
    print("training (one model; policies differ only at inference)...", flush=True)
    train(model, tr, dev)

    logits, hfull, hzero, hconf = gather(model, va, dev)
    preds = logits.argmax(-1); y = va["y"]; L = logits.shape[0]
    conf = logits.softmax(-1).max(-1).values                     # (L,N)
    fixedK = [(preds[k] == y).float().mean().item() * 100 for k in range(L)]
    print("\nprecondition -- fixed-K accuracy (rises with depth?):")
    print("  K:", " ".join(f"{k+1}:{fixedK[k]:.1f}" for k in range(L)))

    taus = list(np.linspace(0.5, 0.999, 16))
    F1 = frontier_threshold(conf, preds, y, list(np.linspace(0.5, 0.9999, 16)))   # confidence
    F2 = frontier_pabee(preds, y)                                                  # PABEE
    F3 = frontier_threshold(hfull, preds, y, taus)                                 # tension-full
    F4 = frontier_threshold(hconf, preds, y, taus)                                 # tension-ablated (fair)
    Fz = frontier_threshold(hzero, preds, y, taus)                                 # tension-zeroed (flagged)

    print("\nspeed-accuracy at matched compute (acc % interpolated at fixed avg-layers):")
    print(f"{'policy':<26}{'@2.0L':>8}{'@3.0L':>8}{'@4.0L':>8}")
    rows = [("1 confidence (DeeBERT)", F1), ("2 PABEE patience", F2),
            ("3 tension-halt (mine)", F3), ("4 tension ablated (no speed)", F4),
            ("4z tension speed-zeroed", Fz)]
    for name, fr in rows:
        print(f"{name:<26}" + "".join(f"{at(fr, x):>8.2f}" for x in (2.0, 3.0, 4.0)))

    # honest verdict
    print("\n--- verdict (Experiment A) ---")
    for x in (2.0, 3.0):
        mine = at(F3, x)
        best_other = max(at(F1, x), at(F2, x), at(F4, x))
        abl = at(F4, x)
        print(f"  @{x:.0f} layers: tension-full {mine:.2f} | best of others {best_other:.2f} "
              f"| fair-ablated(no speed) {abl:.2f}  -> Δ vs ablated {mine-abl:+.2f}")
    print("  read: Δ(tension-full vs fair-ablated) ~0  => the ||Δcls|| signal adds nothing "
          "(contribution is framing).")
    print("        Δ clearly >0 AND tension-full >= best-of-others  => the settling signal is real.")

    try:
        plot([("1 confidence (DeeBERT)", F1, "tab:gray", "s-"),
              ("2 PABEE patience", F2, "tab:green", "^-"),
              ("3 tension-halt (mine)", F3, "tab:blue", "o-"),
              ("4 tension ablated (no speed)", F4, "tab:red", "d-"),
              ("4z tension speed-zeroed", Fz, "tab:purple", "x--")], fixedK)
    except Exception as e:
        print(f"(plot skipped: {e})")


def plot(series, fixedK):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(6.4, 4.4))
    for name, fr, color, style in series:
        a = sorted(fr); xs = [p[0] for p in a]; ys = [p[1] for p in a]
        plt.plot(xs, ys, style, color=color, ms=5, label=name, alpha=0.9)
    plt.axhline(fixedK[-1], ls=":", color="k", lw=0.8)
    plt.text(4.2, fixedK[-1] + 0.1, "full model", fontsize=8)
    plt.xlabel("average layers used (compute)"); plt.ylabel("SST-2 accuracy %")
    plt.title("Experiment A: exit-policy frontiers (is ||Δcls|| doing anything?)")
    plt.legend(fontsize=7.5, loc="lower right"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("figures/glue_ablation_frontier.png", dpi=130)
    print("\nsaved figures/glue_ablation_frontier.png")


if __name__ == "__main__":
    main()

"""
Benchmark 5 -- the tension ponder layer on bAbI, a real, standard benchmark.

bAbI (Weston et al. 2015) is the canonical test bed for adaptive computation: the
Universal Transformer + ACT result showed a recurrent model allocates MORE ponder steps to
tasks that need more reasoning hops. We reproduce that with tension-halting and race it
against the fixed-depth options. We use the three multi-hop QA tasks, which share the same
6 location answers (a clean 6-way classification):
    qa1 single-supporting-fact  -> 1 hop
    qa2 two-supporting-facts     -> 2 hops
    qa3 three-supporting-facts   -> 3 hops
The number of supporting facts is the labeled DIFFICULTY. Each example is
[CLS] story... [SEP] question; the model predicts the answer location from the [CLS] state.

Why bAbI and not ListOps: a small Transformer can actually SOLVE bAbI's hops (it converts
more recurrent steps into more accuracy), so adaptive compute has something real to allocate
-- whereas deep ListOps is unsolved at this scale, so halting just collapses (see
bench_listops.py, an honest negative).

Models (all read the answer from [CLS]):
  Plain Transformer  : T_MAX distinct layers, always full depth.       (T_MAX x params)
  Universal (fixed)  : ONE shared layer applied T_MAX times, always.   (1x params)
  Pondering UT (ours): shared layer + per-example tension halting.      (1x params)

Claims under test:
  (1) Pondering UT matches the fixed-depth Universal Transformer accuracy at LOWER avg
      compute (recurrent steps), same params;
  (2) compute spent rises with the number of reasoning hops -- corr(steps, #facts) > 0 and
      mean steps(qa1) < steps(qa2) < steps(qa3): it thinks longer when more hops are needed.

Run:  python3 bench_babi.py     (expects data/ from the bAbI tar; see README)
"""
import re
import math
import random
import torch
from torch import nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE = "data/tasks_1-20_v1-2/en-10k/"
FILES = {1: "qa1_single-supporting-fact", 2: "qa2_two-supporting-facts", 3: "qa3_three-supporting-facts"}
MAXLEN = 320
T_MAX = 8
D_MODEL = 128
HEADS = 4
FFN = 256
BATCH = 64
STEPS = 6000
LR = 1e-3
LAMBDA_C = 0.01
LOG_EVERY = 1000


def _tok(s):
    return re.findall(r"[a-z]+", s.lower())


def parse(path):
    out, story = [], []
    for line in open(path):
        nid, rest = line.split(" ", 1)
        if int(nid) == 1:
            story = []
        if "\t" in rest:
            q, ans, sup = rest.split("\t")
            ctx = [w for sent in story for w in sent]
            out.append((ctx, _tok(q), ans.strip(), len(sup.split())))
        else:
            story.append(_tok(rest))
    return out


def build_vocab(splits):
    words, answers = set(), set()
    for d in splits:
        for ctx, q, a, n in d:
            words |= set(ctx) | set(q); answers.add(a)
    itos = ["[PAD]", "[CLS]", "[SEP]"] + sorted(words)
    stoi = {w: i for i, w in enumerate(itos)}
    ans_list = sorted(answers)
    a2i = {a: i for i, a in enumerate(ans_list)}
    return stoi, a2i, len(itos), len(ans_list)


def encode(data, stoi, a2i, dev):
    n = len(data)
    ids = torch.zeros(n, MAXLEN, dtype=torch.long)
    y = torch.zeros(n, dtype=torch.long)
    hops = torch.zeros(n, dtype=torch.long)
    for i, (ctx, q, a, nsup) in enumerate(data):
        budget = MAXLEN - 2 - len(q)            # room for [CLS], [SEP], question
        c = ctx[-budget:] if len(ctx) > budget else ctx
        seq = [stoi["[CLS]"]] + [stoi[w] for w in c] + [stoi["[SEP]"]] + [stoi[w] for w in q]
        seq = seq[:MAXLEN]
        ids[i, :len(seq)] = torch.tensor(seq)
        y[i] = a2i[a]; hops[i] = nsup
    return dict(ids=ids.to(dev), y=y.to(dev), hops=hops.to(dev))


# ----------------------------- modules -----------------------------
class SharedLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, HEADS, batch_first=True)
        self.n2 = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(nn.Linear(D_MODEL, FFN), nn.GELU(), nn.Linear(FFN, D_MODEL))

    def forward(self, x, kpm):
        a = self.n1(x)
        x = x + self.attn(a, a, a, key_padding_mask=kpm, need_weights=False)[0]
        return x + self.ff(self.n2(x))


class Embed(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, D_MODEL, padding_idx=0)
        self.pos = nn.Embedding(MAXLEN, D_MODEL)

    def forward(self, ids):
        pos = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
        return self.tok(ids) + self.pos(pos)


class PlainTransformer(nn.Module):
    def __init__(self, vocab, n_cls, depth=T_MAX):
        super().__init__()
        self.emb = Embed(vocab); self.layers = nn.ModuleList([SharedLayer() for _ in range(depth)])
        self.norm = nn.LayerNorm(D_MODEL); self.read = nn.Linear(D_MODEL, n_cls)

    def forward(self, b):
        kpm = b["ids"] == 0; x = self.emb(b["ids"])
        for layer in self.layers:
            x = layer(x, kpm)
        return self.read(self.norm(x[:, 0]))


class UniversalTransformer(nn.Module):
    def __init__(self, vocab, n_cls):
        super().__init__()
        self.emb = Embed(vocab); self.layer = SharedLayer()
        self.norm = nn.LayerNorm(D_MODEL); self.read = nn.Linear(D_MODEL, n_cls)

    def forward(self, b):
        kpm = b["ids"] == 0; x = self.emb(b["ids"])
        for _ in range(T_MAX):
            x = self.layer(x, kpm)
        return self.read(self.norm(x[:, 0]))


class PonderingUT(nn.Module):
    def __init__(self, vocab, n_cls):
        super().__init__()
        self.emb = Embed(vocab); self.layer = SharedLayer()
        self.norm = nn.LayerNorm(D_MODEL); self.read = nn.Linear(D_MODEL, n_cls)
        self.halt = nn.Sequential(nn.Linear(D_MODEL + 1, D_MODEL), nn.Tanh(), nn.Linear(D_MODEL, 1))

    def unroll(self, b):
        kpm = b["ids"] == 0; x = self.emb(b["ids"])
        lam_list, logit_list = [], []
        for t in range(T_MAX):
            x_new = self.layer(x, kpm); cls = x_new[:, 0]
            speed = (cls - x[:, 0]).norm(dim=-1, keepdim=True)
            lam_list.append(torch.sigmoid(self.halt(torch.cat([cls, speed], -1)).squeeze(-1)))
            logit_list.append(self.read(self.norm(cls))); x = x_new
        lam = torch.stack(lam_list); lam = lam.clone(); lam[-1] = 1.0
        logits = torch.stack(logit_list)
        oneminus = (1 - lam).clamp(1e-6, 1.0)
        carry = torch.cat([torch.ones_like(lam[:1]), torch.cumprod(oneminus, 0)[:-1]], 0)
        return logits, lam * carry

    @torch.no_grad()
    def infer(self, b, thresh=0.5):
        kpm = b["ids"] == 0; x = self.emb(b["ids"]); B = x.shape[0]
        carry = torch.ones(B, device=x.device); chalt = torch.zeros(B, device=x.device)
        done = torch.zeros(B, dtype=torch.bool, device=x.device)
        step = torch.full((B,), T_MAX - 1, dtype=torch.long, device=x.device)
        out = self.read(self.norm(x[:, 0]))
        for t in range(T_MAX):
            x_new = self.layer(x, kpm); cls = x_new[:, 0]
            speed = (cls - x[:, 0]).norm(dim=-1, keepdim=True)
            lam = torch.sigmoid(self.halt(torch.cat([cls, speed], -1)).squeeze(-1))
            if t == T_MAX - 1:
                lam = torch.ones_like(lam)
            chalt = chalt + lam * carry
            newly = (~done) & (chalt >= thresh)
            logits_now = self.read(self.norm(cls))
            out[newly] = logits_now[newly]; step[newly] = t; done |= newly
            carry = carry * (1 - lam); x = x_new
        return out.argmax(-1), step


def ponder_loss(logits, p_halt, y, lambda_c):
    T, B, C = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, C), y.unsqueeze(0).expand(T, B).reshape(-1),
                         reduction="none").reshape(T, B)
    exp_ce = (p_halt * ce).sum(0).mean()
    steps = (torch.arange(T, device=logits.device).float() + 1).unsqueeze(-1)
    return exp_ce + lambda_c * (p_halt * steps).sum(0).mean()


def n_params(m): return sum(p.numel() for p in m.parameters())


def sample(pool, B, dev):
    i = torch.randint(0, pool["y"].shape[0], (B,), device=dev)
    return {k: v[i] for k, v in pool.items()}


def train_ce(model, pool, dev, tag=""):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for it in range(STEPS):
        b = sample(pool, BATCH, dev)
        loss = F.cross_entropy(model(b), b["y"])
        opt.zero_grad(); loss.backward(); opt.step()
        if it % LOG_EVERY == 0:
            print(f"  [{tag}] step {it:5d}/{STEPS}  loss {loss.item():.3f}", flush=True)
    return model


def train_ponder(model, pool, dev, tag="ponder"):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for it in range(STEPS):
        lam_c = LAMBDA_C * min(1.0, max(0.0, (it / STEPS - 0.5) / 0.2))
        b = sample(pool, BATCH, dev)
        logits, p_halt = model.unroll(b)
        loss = ponder_loss(logits, p_halt, b["y"], lam_c)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % LOG_EVERY == 0:
            print(f"  [{tag}] step {it:5d}/{STEPS}  loss {loss.item():.3f}  lam_c {lam_c:.4f}", flush=True)
    return model


@torch.no_grad()
def eval_plain(model, test):
    # batched to bound memory at MAXLEN=400
    preds = []
    for i in range(0, test["y"].shape[0], 512):
        b = {k: v[i:i + 512] for k, v in test.items()}
        preds.append(model(b).argmax(-1))
    pred = torch.cat(preds)
    return pred


def per_task(pred, test):
    return {h: (pred[test["hops"] == h] == test["y"][test["hops"] == h]).float().mean().item() * 100
            for h in [1, 2, 3]}


def main():
    dev = torch.device(DEVICE)
    torch.manual_seed(0); random.seed(0)
    tr = [parse(BASE + f + "_train.txt") for f in FILES.values()]
    te = [parse(BASE + f + "_test.txt") for f in FILES.values()]
    stoi, a2i, vocab, n_cls = build_vocab(tr)
    train_pool = encode([x for d in tr for x in d], stoi, a2i, dev)
    test = encode([x for d in te for x in d], stoi, a2i, dev)
    print(f"device {DEVICE}  bAbI qa1/2/3  vocab={vocab} classes={n_cls} maxlen={MAXLEN} T_max={T_MAX}")
    print(f"train {train_pool['y'].shape[0]}  test {test['y'].shape[0]}  "
          f"chance {100/n_cls:.1f}%\n")

    print("training Plain Transformer...", flush=True)
    plain = train_ce(PlainTransformer(vocab, n_cls).to(dev), train_pool, dev, tag="plain")
    print("training Universal Transformer...", flush=True)
    uni = train_ce(UniversalTransformer(vocab, n_cls).to(dev), train_pool, dev, tag="uni")
    print("training Pondering UT...", flush=True)
    pon = train_ponder(PonderingUT(vocab, n_cls).to(dev), train_pool, dev)

    p_plain = eval_plain(plain, test); a_plain = (p_plain == test["y"]).float().mean().item() * 100
    p_uni = eval_plain(uni, test); a_uni = (p_uni == test["y"]).float().mean().item() * 100
    # pondering inference (batched)
    preds, steps = [], []
    for i in range(0, test["y"].shape[0], 512):
        b = {k: v[i:i + 512] for k, v in test.items()}
        pr, st = pon.infer(b); preds.append(pr); steps.append(st)
    p_pon = torch.cat(preds); step = torch.cat(steps).float()
    a_pon = (p_pon == test["y"]).float().mean().item() * 100
    comp = step + 1; avg = comp.mean().item()
    h = test["hops"].float()
    corr = (((comp - comp.mean()) * (h - h.mean())).mean() / (comp.std() * h.std() + 1e-8)).item()

    print(f"{'model':<24}{'params':>10}{'acc %':>9}{'avg steps':>11}{'corr(steps,hops)':>18}")
    print("-" * 72)
    print(f"{'Plain Transformer':<24}{n_params(plain):>10}{a_plain:>9.2f}{float(T_MAX):>11.2f}{'n/a':>18}")
    print(f"{'Universal (fixed)':<24}{n_params(uni):>10}{a_uni:>9.2f}{float(T_MAX):>11.2f}{'n/a':>18}")
    print(f"{'Pondering UT (ours)':<24}{n_params(pon):>10}{a_pon:>9.2f}{avg:>11.2f}{corr:>18.3f}")
    print(f"\ncompute vs Universal: {100*(1-avg/T_MAX):.1f}% fewer recurrent steps "
          f"({'matched' if abs(a_pon-a_uni)<2 else 'see'} accuracy).")

    print("\nper-task accuracy (acc / mean ponder steps), difficulty = #hops:")
    accs = {"Plain": per_task(p_plain, test), "Universal": per_task(p_uni, test), "Pondering": per_task(p_pon, test)}
    print(f"  {'hops':<10}{'qa1 (1)':>12}{'qa2 (2)':>12}{'qa3 (3)':>12}")
    for name in ["Plain", "Universal", "Pondering"]:
        print(f"  {name:<10}" + "".join(f"{accs[name][hh]:>11.1f}%" for hh in [1, 2, 3]))
    print(f"  {'steps':<10}" + "".join(f"{comp[test['hops']==hh].mean().item():>12.2f}" for hh in [1, 2, 3]))
    print("\nread: steps should rise qa1<qa2<qa3 -> it thinks longer when more hops are needed.")


if __name__ == "__main__":
    main()

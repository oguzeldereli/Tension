"""
Benchmark 4 -- the tension ponder layer on ListOps, a standard public benchmark.

ListOps (Nangia & Bowman 2018; the Long Range Arena classification task) -- nested list
operations over digits 0-9 with operators MAX, MIN, MED, SM(=sum mod 10), e.g.
    [MAX 3 [MIN 2 4 ] [SM 1 9 ] 0 ]  -> 4
The model must predict the single result digit (10-way classification). Crucially the
DIFFICULTY is the nesting DEPTH and it varies per example -- a Transformer resolves ~one
nesting level per recurrent step, so a deep expression needs more steps than a shallow one.
This is the canonical setting for per-example adaptive computation, on a recognized task
(generated with the standard ListOps procedure, not a toy of our own).

Models (all read the answer from a prepended [CLS] token):
  Plain Transformer  : T_MAX distinct layers, always full depth. (T_MAX x params)
  Universal (fixed)  : ONE shared layer applied T_MAX times, always. (1x params)
  Pondering UT (ours): the shared layer + per-EXAMPLE tension halting -- it stops recurring
                       once the [CLS] field comes to rest. (1x params)

Claims under test:
  (1) Pondering UT matches the fixed-depth Universal Transformer accuracy at LOWER average
      compute (recurrent steps), same params -- only halting added;
  (2) compute spent tracks expression DEPTH (corr(steps, depth) -> +1): it thinks longer on
      deeper expressions, with nothing telling it the depth.

Run:  python3 bench_listops.py
"""
import math
import random
import statistics
import torch
from torch import nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAXD = 5           # max nesting depth sampled
MAX_ARGS = 4
MAXLEN = 100       # cap token length (+1 for CLS)
T_MAX = 8          # recurrent steps / stack depth (>= MAXD so deep exprs are resolvable)
D_MODEL = 128
HEADS = 4
FFN = 256
TRAIN_POOL = 24000
TEST_POOL = 4000
BATCH = 64
STEPS = 12000
LR = 1e-3
LAMBDA_C = 0.004   # per-example compute penalty (annealed); small -- accuracy first

OPS = ["[MAX", "[MIN", "[MED", "[SM"]
TOKENS = ["[PAD]", "[CLS]", "]"] + OPS + [str(d) for d in range(10)]
TOK2ID = {t: i for i, t in enumerate(TOKENS)}
VOCAB = len(TOKENS)
PAD = TOK2ID["[PAD]"]


def _apply(op, vals):
    if op == "[MAX": return max(vals)
    if op == "[MIN": return min(vals)
    if op == "[MED": return int(statistics.median(vals))
    return sum(vals) % 10                      # [SM


def _gen_tree(max_depth, p_leaf=0.27):
    if max_depth <= 1:
        v = random.randint(0, 9)
        return [str(v)], v, 1
    op = random.choice(OPS)
    nargs = random.randint(2, MAX_ARGS)
    toks, vals, cds = [op], [], []
    for _ in range(nargs):
        if random.random() > p_leaf:
            ct, cv, cd = _gen_tree(max_depth - 1, p_leaf)
        else:
            cv = random.randint(0, 9); ct, cd = [str(cv)], 1
        toks += ct; vals.append(cv); cds.append(cd)
    toks.append("]")
    return toks, _apply(op, vals), 1 + max(cds)


def gen_pool(n, dev):
    ids = torch.full((n, MAXLEN + 1), PAD, dtype=torch.long)
    y = torch.zeros(n, dtype=torch.long)
    depth = torch.zeros(n, dtype=torch.long)
    i = 0
    while i < n:
        md = random.randint(2, MAXD)
        toks, val, d = _gen_tree(md)
        if len(toks) > MAXLEN:
            continue
        seq = ["[CLS]"] + toks
        ids[i, :len(seq)] = torch.tensor([TOK2ID[t] for t in seq])
        y[i] = val; depth[i] = d
        i += 1
    return dict(ids=ids.to(dev), y=y.to(dev), depth=depth.to(dev))


def sample(pool, B, dev):
    idx = torch.randint(0, pool["y"].shape[0], (B,), device=dev)
    return {k: v[idx] for k, v in pool.items()}


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
        x = x + self.ff(self.n2(x))
        return x


class Embed(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D_MODEL, padding_idx=PAD)
        self.pos = nn.Embedding(MAXLEN + 1, D_MODEL)

    def forward(self, ids):
        pos = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
        return self.tok(ids) + self.pos(pos)


class PlainTransformer(nn.Module):
    def __init__(self, depth=T_MAX):
        super().__init__()
        self.emb = Embed()
        self.layers = nn.ModuleList([SharedLayer() for _ in range(depth)])
        self.norm = nn.LayerNorm(D_MODEL); self.read = nn.Linear(D_MODEL, 10)

    def forward(self, b):
        kpm = b["ids"] == PAD
        x = self.emb(b["ids"])
        for layer in self.layers:
            x = layer(x, kpm)
        return self.read(self.norm(x[:, 0]))


class UniversalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = Embed(); self.layer = SharedLayer()
        self.norm = nn.LayerNorm(D_MODEL); self.read = nn.Linear(D_MODEL, 10)

    def forward(self, b):
        kpm = b["ids"] == PAD
        x = self.emb(b["ids"])
        for _ in range(T_MAX):
            x = self.layer(x, kpm)
        return self.read(self.norm(x[:, 0]))


class PonderingUT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = Embed(); self.layer = SharedLayer()
        self.norm = nn.LayerNorm(D_MODEL); self.read = nn.Linear(D_MODEL, 10)
        self.halt = nn.Sequential(nn.Linear(D_MODEL + 1, D_MODEL), nn.Tanh(),
                                  nn.Linear(D_MODEL, 1))

    def unroll(self, b):
        kpm = b["ids"] == PAD
        x = self.emb(b["ids"])
        lam_list, logit_list = [], []
        for t in range(T_MAX):
            x_new = self.layer(x, kpm)
            cls = x_new[:, 0]
            speed = (x_new[:, 0] - x[:, 0]).norm(dim=-1, keepdim=True)
            lam_list.append(torch.sigmoid(self.halt(torch.cat([cls, speed], -1)).squeeze(-1)))
            logit_list.append(self.read(self.norm(cls)))
            x = x_new
        lam = torch.stack(lam_list); lam = lam.clone(); lam[-1] = 1.0     # (T,B)
        logits = torch.stack(logit_list)                                 # (T,B,10)
        oneminus = (1 - lam).clamp(1e-6, 1.0)
        carry = torch.cat([torch.ones_like(lam[:1]), torch.cumprod(oneminus, 0)[:-1]], 0)
        return logits, lam * carry

    @torch.no_grad()
    def infer(self, b, thresh=0.5):
        kpm = b["ids"] == PAD
        x = self.emb(b["ids"]); B = x.shape[0]
        carry = torch.ones(B, device=x.device); chalt = torch.zeros(B, device=x.device)
        done = torch.zeros(B, dtype=torch.bool, device=x.device)
        step = torch.full((B,), T_MAX - 1, dtype=torch.long, device=x.device)
        out = self.read(self.norm(x[:, 0]))
        for t in range(T_MAX):
            x_new = self.layer(x, kpm)
            cls = x_new[:, 0]
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


def train_ce(model, pool, dev):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(STEPS):
        b = sample(pool, BATCH, dev)
        loss = F.cross_entropy(model(b), b["y"])
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def train_ponder(model, pool, dev):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for it in range(STEPS):
        lam_c = LAMBDA_C * min(1.0, max(0.0, (it / STEPS - 0.5) / 0.2))
        b = sample(pool, BATCH, dev)
        logits, p_halt = model.unroll(b)
        loss = ponder_loss(logits, p_halt, b["y"], lam_c)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def acc(model, test): return (model(test).argmax(-1) == test["y"]).float().mean().item() * 100


def main():
    dev = torch.device(DEVICE)
    torch.manual_seed(0); random.seed(0)
    print(f"device {DEVICE}  ListOps (canonical)  vocab={VOCAB} maxlen={MAXLEN} T_max={T_MAX}")
    train_pool = gen_pool(TRAIN_POOL, dev); test = gen_pool(TEST_POOL, dev)
    print(f"pools built; test mean depth {test['depth'].float().mean():.2f} "
          f"range [{test['depth'].min()},{test['depth'].max()}], "
          f"majority-class acc {torch.bincount(test['y']).max().item()/TEST_POOL*100:.1f}%\n")

    plain = train_ce(PlainTransformer().to(dev), train_pool, dev)
    uni = train_ce(UniversalTransformer().to(dev), train_pool, dev)
    pon = train_ponder(PonderingUT().to(dev), train_pool, dev)

    a_plain, a_uni = acc(plain, test), acc(uni, test)
    pred, step = pon.infer(test)
    a_pon = (pred == test["y"]).float().mean().item() * 100
    comp = step.float() + 1; avg = comp.mean().item()
    d = test["depth"].float()
    corr = (((comp - comp.mean()) * (d - d.mean())).mean() / (comp.std() * d.std() + 1e-8)).item()

    print(f"{'model':<24}{'params':>10}{'acc %':>9}{'avg steps':>12}{'corr(steps,depth)':>20}")
    print("-" * 75)
    print(f"{'Plain Transformer':<24}{n_params(plain):>10}{a_plain:>9.2f}{float(T_MAX):>12.2f}{'n/a':>20}")
    print(f"{'Universal (fixed)':<24}{n_params(uni):>10}{a_uni:>9.2f}{float(T_MAX):>12.2f}{'n/a':>20}")
    print(f"{'Pondering UT (ours)':<24}{n_params(pon):>10}{a_pon:>9.2f}{avg:>12.2f}{corr:>20.3f}")
    print(f"\ncompute vs Universal: {100*(1-avg/T_MAX):.1f}% fewer recurrent steps "
          f"({'matched' if abs(a_pon-a_uni)<2 else 'see'} accuracy).")

    # honesty diagnostics: does the base UT convert steps -> accuracy, and per depth?
    @torch.no_grad()
    def uni_acc_at(k):
        kpm = test["ids"] == PAD; x = uni.emb(test["ids"])
        for _ in range(k):
            x = uni.layer(x, kpm)
        return (uni.read(uni.norm(x[:, 0])).argmax(-1) == test["y"]).float().mean().item() * 100
    print("\nUniversal acc vs #steps (does recurrence convert to accuracy?):")
    print("  " + " ".join(f"{k}:{uni_acc_at(k):.1f}" for k in range(1, T_MAX + 1)))

    print("\nthink-time & accuracy vs depth (Pondering UT):")
    print("  depth :", " ".join(f"{dv:6d}" for dv in range(2, MAXD + 1)))
    print("  steps :", " ".join(
        f"{comp[d.long()==dv].mean().item():6.1f}" if (d.long()==dv).any() else "   n/a"
        for dv in range(2, MAXD + 1)))
    ok = (pred == test["y"]).float()
    print("  acc % :", " ".join(
        f"{ok[d.long()==dv].mean().item()*100:6.1f}" if (d.long()==dv).any() else "   n/a"
        for dv in range(2, MAXD + 1)))


if __name__ == "__main__":
    main()

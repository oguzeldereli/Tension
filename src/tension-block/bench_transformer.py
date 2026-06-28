"""
Benchmark 3 -- the TensionBlock as a per-token PONDER layer inside a Transformer.

Does the operator compose at scale? We drop per-token tension-halting into a Universal
Transformer (one shared layer applied recurrently) and let each token decide for itself how
many layer-applications it needs. Compared against the conventional fixed-depth options.

Task: pointer chasing (variable per-token difficulty). N nodes; each node points to its
parent (roots point to themselves). The answer at node i is the VALUE OF ITS ROOT -- reached
by following parents depth(i) times. depth varies a lot across tokens in the same sequence.
One attention step can follow one pointer (copy the parent's current estimate), so resolving
a depth-d node needs ~d recurrent steps. A fixed-depth stack must pay max-depth for EVERY
token; a per-token ponder layer pays depth(i) for token i and stops.

Models (all see the same inputs):
  Plain Transformer  : T_MAX DISTINCT layers, always full depth. (T_MAX x params)
  Universal (fixed)  : ONE shared layer applied T_MAX times, always. (1x params)
  Pondering UT (ours): the shared layer + per-token tension halting. (1x params) At inference
                       a token freezes once it has resolved (its field comes to rest); deep
                       tokens iterate more, shallow tokens stop early.

Claims under test:
  (1) Pondering UT matches the fixed-depth Universal Transformer's accuracy at LOWER average
      per-token compute (it halts resolved tokens) -- same params, only halting added.
  (2) Per-token think-time tracks chain depth (corr(halt, depth) -> +1): deliberation is
      allocated where the problem is actually hard, with nothing telling it the depths.

Run:  python3 bench_transformer.py
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = 16              # nodes per sequence
V = 16             # value vocabulary (classes)
D_MODEL = 96
HEADS = 4
FFN = 256
T_MAX = 8          # max recurrent steps / max stack depth / max resolvable chain depth
TRAIN_POOL = 8192
TEST_POOL = 4096
BATCH = 128
STEPS = 7000
LR = 3e-3
LAMBDA_C = 0.02    # per-token compute penalty (annealed)


# ----------------------------- task -----------------------------
def gen_pool(n_seq, dev):
    """Random forests of self-loop-rooted chains; depths spread in [0, T_MAX-1]."""
    parent = torch.zeros(n_seq, N, dtype=torch.long, device=dev)
    for b in range(n_seq):
        perm = torch.randperm(N, device=dev)
        idx = 0
        while idx < N:
            L = int(torch.randint(1, T_MAX + 1, (1,)).item())
            L = min(L, N - idx)
            chain = perm[idx:idx + L]
            parent[b, chain[0]] = chain[0]                  # root: self-loop
            for j in range(1, L):
                parent[b, chain[j]] = chain[j - 1]          # point toward the root
            idx += L
    values = torch.randint(0, V, (n_seq, N), device=dev)
    cur = torch.arange(N, device=dev).expand(n_seq, N).clone()
    depth = torch.zeros(n_seq, N, dtype=torch.long, device=dev)
    for _ in range(T_MAX):
        nxt = parent.gather(1, cur)
        depth += (nxt != cur).long()
        cur = nxt
    y = values.gather(1, cur)                               # value at the root
    return dict(parent=parent, values=values, y=y, depth=depth)


def sample(pool, B, dev):
    i = torch.randint(0, pool["y"].shape[0], (B,), device=dev)
    return {k: v[i] for k, v in pool.items()}


# ----------------------------- modules -----------------------------
class SharedLayer(nn.Module):
    """Pre-norm self-attention + FFN. The recurrent core."""
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, HEADS, batch_first=True)
        self.n2 = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(nn.Linear(D_MODEL, FFN), nn.GELU(), nn.Linear(FFN, D_MODEL))

    def forward(self, x):
        a = self.n1(x)
        x = x + self.attn(a, a, a, need_weights=False)[0]
        x = x + self.ff(self.n2(x))
        return x


class Embed(nn.Module):
    def __init__(self):
        super().__init__()
        self.val = nn.Embedding(V, D_MODEL)
        self.pos = nn.Embedding(N, D_MODEL)
        self.ptr = nn.Embedding(N, D_MODEL)        # which position I point to

    def forward(self, batch):
        B = batch["values"].shape[0]
        pos = torch.arange(N, device=batch["values"].device).expand(B, N)
        return self.val(batch["values"]) + self.pos(pos) + self.ptr(batch["parent"])


class PlainTransformer(nn.Module):
    def __init__(self, depth=T_MAX):
        super().__init__()
        self.emb = Embed()
        self.layers = nn.ModuleList([SharedLayer() for _ in range(depth)])
        self.norm = nn.LayerNorm(D_MODEL)
        self.read = nn.Linear(D_MODEL, V)

    def forward(self, batch):
        x = self.emb(batch)
        for layer in self.layers:
            x = layer(x)
        return self.read(self.norm(x))


class UniversalTransformer(nn.Module):
    """Shared layer applied T_MAX times, always (no halting)."""
    def __init__(self):
        super().__init__()
        self.emb = Embed()
        self.layer = SharedLayer()
        self.norm = nn.LayerNorm(D_MODEL)
        self.read = nn.Linear(D_MODEL, V)

    def forward(self, batch):
        x = self.emb(batch)
        for _ in range(T_MAX):
            x = self.layer(x)
        return self.read(self.norm(x))


class PonderingUT(nn.Module):
    """Shared layer + per-token tension halting (reads the field's speed)."""
    def __init__(self):
        super().__init__()
        self.emb = Embed()
        self.layer = SharedLayer()
        self.norm = nn.LayerNorm(D_MODEL)
        self.read = nn.Linear(D_MODEL, V)
        self.halt = nn.Sequential(nn.Linear(D_MODEL + 1, D_MODEL), nn.Tanh(),
                                  nn.Linear(D_MODEL, 1))

    def unroll(self, batch):
        """Full unroll -> per-token logits & halt probs for every step (training)."""
        x = self.emb(batch)
        lam_list, logit_list = [], []
        for t in range(T_MAX):
            x_new = self.layer(x)
            speed = (x_new - x).norm(dim=-1, keepdim=True)            # (B,N,1)
            lam = torch.sigmoid(self.halt(torch.cat([x_new, speed], dim=-1)).squeeze(-1))
            lam_list.append(lam)
            logit_list.append(self.read(self.norm(x_new)))
            x = x_new
        lam = torch.stack(lam_list)                                  # (T,B,N)
        lam = lam.clone(); lam[-1] = 1.0
        logits = torch.stack(logit_list)                            # (T,B,N,V)
        oneminus = (1 - lam).clamp(1e-6, 1.0)
        carry = torch.cat([torch.ones_like(lam[:1]), torch.cumprod(oneminus, 0)[:-1]], 0)
        p_halt = lam * carry                                        # (T,B,N)
        return logits, p_halt

    @torch.no_grad()
    def infer(self, batch, thresh=0.5):
        """Per-token adaptive inference: a token FREEZES (stops computing, stays a stable
        lookup target) once its cumulative halt prob crosses thresh. Returns pred, halt_step."""
        x = self.emb(batch)
        B = x.shape[0]
        carry = torch.ones(B, N, device=x.device)
        chalt = torch.zeros(B, N, device=x.device)
        frozen = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        halt_step = torch.full((B, N), T_MAX - 1, dtype=torch.long, device=x.device)
        out = self.read(self.norm(x))
        for t in range(T_MAX):
            x_cand = self.layer(x)
            x_new = torch.where(frozen.unsqueeze(-1), x, x_cand)     # frozen tokens hold
            speed = (x_new - x).norm(dim=-1, keepdim=True)
            lam = torch.sigmoid(self.halt(torch.cat([x_new, speed], dim=-1)).squeeze(-1))
            if t == T_MAX - 1:
                lam = torch.ones_like(lam)
            p_halt = lam * carry
            chalt = chalt + p_halt
            newly = (~frozen) & (chalt >= thresh)
            logits_now = self.read(self.norm(x_new))
            out[newly] = logits_now[newly]
            halt_step[newly] = t
            frozen = frozen | newly
            carry = carry * (1 - lam)
            x = x_new
        return out.argmax(-1), halt_step


def ponder_loss(logits, p_halt, y, lambda_c):
    T, B, n, C = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, C), y.unsqueeze(0).expand(T, B, n).reshape(-1),
                         reduction="none").reshape(T, B, n)
    exp_ce = (p_halt * ce).sum(0).mean()
    steps = (torch.arange(T, device=logits.device).float() + 1).view(T, 1, 1)
    exp_steps = (p_halt * steps).sum(0).mean()
    return exp_ce + lambda_c * exp_steps


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ----------------------------- train / eval -----------------------------
def train_plain(model, pool, dev, ponder=False):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(STEPS):
        b = sample(pool, BATCH, dev)
        logits = model(b)
        loss = F.cross_entropy(logits.reshape(-1, V), b["y"].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def train_ponder(model, pool, dev):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for it in range(STEPS):
        frac = max(0.0, (it / STEPS - 0.4) / 0.3)           # anneal compute penalty
        lam_c = LAMBDA_C * min(1.0, frac)
        b = sample(pool, BATCH, dev)
        logits, p_halt = model.unroll(b)
        loss = ponder_loss(logits, p_halt, b["y"], lam_c)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def acc_plain(model, test, dev):
    b = test
    pred = model(b).argmax(-1)
    return (pred == b["y"]).float().mean().item() * 100


def main():
    dev = torch.device(DEVICE)
    torch.manual_seed(0)
    print(f"device {DEVICE}  pointer-chasing  N={N} V={V} d_model={D_MODEL} T_max={T_MAX}")
    train_pool = gen_pool(TRAIN_POOL, dev)
    test = gen_pool(TEST_POOL, dev)
    dmean = test["depth"].float().mean().item()
    print(f"train/test pools built; mean chain depth = {dmean:.2f} (so >= {dmean:.1f} steps "
          f"are genuinely needed on average)\n")

    plain = train_plain(PlainTransformer().to(dev), train_pool, dev)
    uni = train_plain(UniversalTransformer().to(dev), train_pool, dev)
    pon = train_ponder(PonderingUT().to(dev), train_pool, dev)

    a_plain = acc_plain(plain, test, dev)
    a_uni = acc_plain(uni, test, dev)
    pred, halt = pon.infer(test)
    a_pon = (pred == test["y"]).float().mean().item() * 100
    compute = (halt.float() + 1)
    avg_compute = compute.mean().item()
    d = test["depth"].float()
    corr = (((compute - compute.mean()) * (d - d.mean())).mean()
            / (compute.std() * d.std() + 1e-8)).item()

    print(f"{'model':<24}{'params':>10}{'acc %':>9}{'avg depth/token':>18}{'corr(halt,depth)':>18}")
    print("-" * 79)
    print(f"{'Plain Transformer':<24}{n_params(plain):>10}{a_plain:>9.2f}{float(T_MAX):>18.2f}{'n/a':>18}")
    print(f"{'Universal (fixed)':<24}{n_params(uni):>10}{a_uni:>9.2f}{float(T_MAX):>18.2f}{'n/a':>18}")
    print(f"{'Pondering UT (ours)':<24}{n_params(pon):>10}{a_pon:>9.2f}{avg_compute:>18.2f}{corr:>18.3f}")

    print(f"\ncompute vs Universal: {100*(1 - avg_compute / T_MAX):.1f}% fewer layer-applications "
          f"per token at {'matched' if abs(a_pon-a_uni)<1.5 else 'see'} accuracy.")
    print("\nthink-time vs problem difficulty (mean halt step per chain depth):")
    print("  depth :", " ".join(f"{dv:5d}" for dv in range(T_MAX)))
    print("  halt  :", " ".join(
        f"{compute[d.long() == dv].mean().item():5.1f}" if (d.long() == dv).any() else "  n/a"
        for dv in range(T_MAX)))


if __name__ == "__main__":
    main()

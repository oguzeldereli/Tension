"""
budget_dynamic.py -- make the global budget ELASTIC and BINDING: it starts at a small fixed pool
and operators can "request" more shared budget when their tension needs more deliberation -- but
requesting is destabilising (a convex discomfort term raises the loss), and there is a hard CAP on
the reserve. Goal: squeeze more capability than a fixed pool by expanding adaptively.

WHY BINDING. A first version made the pool a soft PRICE feature; the head just learned to ignore a
high price, so the pool never constrained anything and making it elastic changed nothing (requests
~0). For the budget to mean something, it must BIND: once the pool is dry an operator is forced to
commit -- unless extra budget has been requested. We implement that with an availability gate
    avail = clamp((pool - spent)/E0, 0, 1)       lam_effective = lam + (1-lam)*(1-avail)
so as the pool empties the operator is pushed to commit; requesting raises the pool and keeps it
deliberating. A FIXED pool therefore hard-caps compute at ~its size; the ELASTIC one expands a
small base toward the cap by paying convex discomfort.

Loss (unchanged in spirit -- synthesis correct + global budget, now elastic/convex/capped):
    L = E[CE]  +  mu * E[(granted extra budget / E0)^2]          (lambda*used dropped: the gate caps compute)

Experiment. FIXED frontier: sweep the pool size E0 (each a hard cap) -> accuracy vs compute. ELASTIC
frontier: fix a SMALL base E0 and sweep the discomfort mu (high mu -> ask for little -> low compute;
low mu -> ask freely -> high compute) -> accuracy vs compute, plus how much it asked for (req G/E0).
The question: starting from a small base, does the elastic budget reach the accuracy of large fixed
pools, and does it sit ABOVE the fixed-pool / Fixed-N frontier at matched mean compute (because it
gives the reserve to the instances that need it instead of everyone)? Readout trained once, FROZEN.

Run:  python3 budget_dynamic.py
"""
import os, sys
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_opposites import gen, D, C
from energy_budget import (train_readout, observe, make_instances, loguniform,
                           T_MAX, M, SIGMA_LO, DISTRACT_SIGMA, DISTRACT_FRAC, DEVICE)
from compare_baselines import trajectories, f_fixed_n, f_conf_threshold, f_oracle, TAUS

H = 64
R_PER_TICK = 1.5
PRICE_GAIN = 1.0
STEPS_HALT = 800
BATCH_INST = 96
LR = 3e-3

E0_GRID = [M * b for b in [1, 2, 3, 4, 6, 9, 13]]      # fixed pool sizes (hard caps)
E0_BASE = M * 2.0                                       # small starting pool for the elastic budget
R_MAX = M * 14.0                                        # cap on total requestable reserve
MU_GRID = [3.0, 1.2, 0.5, 0.2, 0.08, 0.02, 0.0]        # discomfort: high -> ask little, low -> ask freely


class Head(nn.Module):
    def __init__(self, dynamic=True):
        super().__init__()
        self.dynamic = dynamic
        self.trunk = nn.Sequential(nn.Linear(4, H), nn.Tanh())   # s, tmean, conf, price
        self.halt = nn.Linear(H, 1)
        self.req = nn.Linear(H, 1)

    def forward(self, s, tmean, conf, price):
        h = self.trunk(torch.stack([s, tmean, conf, price], -1))
        lam = torch.sigmoid(self.halt(h)).squeeze(-1)
        req = torch.sigmoid(self.req(h)).squeeze(-1) if self.dynamic else torch.zeros_like(lam)
        return lam, req


def run_system(read, head, poles, y, sigma, inst_id, n_inst, dev, e0):
    """Differentiable rollout with an elastic, BINDING shared pool (availability-gated commit)."""
    B = poles.shape[0]
    zbar = torch.zeros(B, D, device=dev); prev = torch.zeros(B, D, device=dev)
    carry = torch.ones(B, device=dev)
    S = torch.zeros(n_inst, device=dev)        # expected compute used
    G = torch.zeros(n_inst, device=dev)        # cumulative granted extra budget
    ce_t, corr_t, step_t = [], [], []
    for t in range(T_MAX):
        z, tmean = observe(poles, sigma)
        zbar = (zbar * t + z) / (t + 1)
        s = (zbar - prev).norm(dim=-1); prev = zbar
        logits = read(zbar)
        conf = F.softmax(logits, -1).amax(-1)
        remaining = e0 + G - S
        avail = (remaining / e0).clamp(0.0, 1.0)               # 1 full -> 0 empty
        price = (e0 / remaining.clamp(min=1e-3)).clamp(min=1.0) ** PRICE_GAIN
        lam, req = head(s, tmean, conf, price[inst_id])
        lam = lam + (1 - lam) * (1 - avail[inst_id])           # BINDING: empty pool forces commit
        if t == T_MAX - 1:
            lam = torch.ones_like(lam)
        p_halt = lam * carry
        if head.dynamic:                                        # petition for more budget (preemptive)
            demand = torch.zeros(n_inst, device=dev).index_add_(0, inst_id, req * carry * R_PER_TICK)
            G = G + torch.minimum(demand, (R_MAX - G).clamp(min=0))
        S = S + torch.zeros(n_inst, device=dev).index_add_(0, inst_id, carry)
        ce_t.append(p_halt * F.cross_entropy(logits, y, reduction="none"))
        corr_t.append(p_halt * (logits.argmax(-1) == y).float())
        step_t.append(carry)
        carry = carry * (1 - lam)
    return (torch.stack(ce_t).sum(0), torch.stack(corr_t).sum(0),
            torch.stack(step_t).sum(0), G)


def make_batch(dev, mode):
    poles, y, _ = gen(BATCH_INST * M, (1.0, 2.0), dev)
    if mode == "distractor":
        sigma = loguniform(BATCH_INST * M, SIGMA_LO, 0.4, dev)
        hop = torch.rand(BATCH_INST * M, device=dev) < DISTRACT_FRAC
        sigma[hop] = DISTRACT_SIGMA
    else:
        sigma = loguniform(BATCH_INST * M, SIGMA_LO, 0.9, dev)
    return poles, y, sigma


def train_head(read, dev, dynamic, mode, e0, mu):
    head = Head(dynamic=dynamic).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    for p in read.parameters():
        p.requires_grad_(False)
    read.eval(); head.train()
    inst_id = torch.arange(BATCH_INST, device=dev).repeat_interleave(M)
    for it in range(STEPS_HALT):
        poles, y, sigma = make_batch(dev, mode)
        ce, _, _, G = run_system(read, head, poles, y, sigma, inst_id, BATCH_INST, dev, e0)
        loss = ce.mean() + mu * ((G / e0) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return head


@torch.no_grad()
def evaluate(read, head, data, n_inst, dev, e0):
    poles, y, sigma, inst_id = data
    _, correct, steps, G = run_system(read, head, poles, y, sigma, inst_id, n_inst, dev, e0)
    return steps.mean().item(), correct.mean().item() * 100, (G / e0).mean().item()


def _table(title, rows, has_g=False):
    print(f"\n{title}")
    if has_g:
        print(f"  {'steps/op':>9}{'acc %':>8}{'req G/E0':>10}")
        for s, a, g in rows:
            print(f"  {s:>9.2f}{a:>8.2f}{g:>10.2f}")
    else:
        print(f"  {'steps/op':>9}{'acc %':>8}")
        for s, a in rows:
            print(f"  {s:>9.2f}{a:>8.2f}")


def run_regime(read, dev, mode):
    n_inst = 1024
    data = make_instances(n_inst, dev, mode=mode)
    poles, y, sigma, inst_id = data
    conf, corr = trajectories(read, poles, y, sigma, dev)
    budgets = [M * b for b in [1, 1.5, 2, 3, 4, 6, 8, 12]]
    print("\n" + "=" * 72)
    print(f"{mode.upper()} regime  (elastic base E0={E0_BASE}, R_MAX={R_MAX})")

    print("training FIXED pools (sweep size, hard cap)...", flush=True)
    fixed = []
    for e0 in E0_GRID:
        h = train_head(read, dev, dynamic=False, mode=mode, e0=e0, mu=0.0)
        s, a, _ = evaluate(read, h, data, n_inst, dev, e0)
        fixed.append((s, a))
    print("training ELASTIC budget (small base, sweep discomfort mu)...", flush=True)
    dyn = []
    for mu in MU_GRID:
        h = train_head(read, dev, dynamic=True, mode=mode, e0=E0_BASE, mu=mu)
        s, a, g = evaluate(read, h, data, n_inst, dev, E0_BASE)
        dyn.append((s, a, g))

    _table("Fixed-N (uniform)", f_fixed_n(corr))
    _table("Confidence>=tau (SPRT/DeeBERT)", f_conf_threshold(conf, corr, TAUS, dev))
    _table("Oracle knapsack (UPPER BOUND)", f_oracle(corr, n_inst, budgets, dev))
    _table("FIXED pool (sweep size)", fixed)
    _table("ELASTIC budget (small base + requests + convex discomfort + cap)", dyn, has_g=True)
    _plot(mode, fixed, dyn, f_fixed_n(corr), f_conf_threshold(conf, corr, TAUS, dev),
          f_oracle(corr, n_inst, budgets, dev))


def _plot(mode, fixed, dyn, fN, cf, orc):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    series = [(orc, "oracle (upper bound)", "k--"), (cf, "confidence>=tau (SPRT)", "o-"),
              (fN, "fixed-N", "x-"), (fixed, "fixed pool (sized)", "^-"),
              ([(s, a) for s, a, _ in dyn], "elastic budget (ours)", "s-")]
    for rows, lab, st in series:
        ax.plot([r[0] for r in rows], [r[1] for r in rows], st, label=lab)
    ax.set_xlabel("compute (mean inner steps / operator)"); ax.set_ylabel("accuracy (%)")
    ax.set_title(f"Elastic (binding) global budget -- {mode}")
    ax.grid(alpha=0.3); ax.legend(fontsize=8); fig.tight_layout()
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(p, exist_ok=True)
    fn = os.path.join(p, f"budget_dynamic_{mode}.png")
    fig.savefig(fn, dpi=120); print(f"  saved {fn}")


def main():
    dev = torch.device(DEVICE)
    print(f"device {DEVICE}  M={M}  T_MAX={T_MAX}   binding pool, loss = E[CE] + mu*E[(granted/E0)^2]")
    print("training shared readout (frozen)...", flush=True)
    read = train_readout(dev)
    for mode in ["distractor", "hetero"]:
        run_regime(read, dev, mode)
    print("\nread: does the ELASTIC budget (small base) climb the fixed-pool frontier as mu drops,")
    print("      asking for more (req G/E0 up) only where it pays -- and sit at/above Fixed-N?")


if __name__ == "__main__":
    main()

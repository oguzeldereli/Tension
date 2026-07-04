"""
compare_baselines.py -- the energy-budget composed system vs KNOWN algorithms (not vs our own
settling variants). Same task, readout, observations as energy_budget.py; every method emits an
answer for EVERY operator and we trace each one's accuracy-vs-compute frontier by sweeping its
own knob.

The problem -- M sub-problems per instance, each needing an adaptive amount of sequential
evidence, under finite shared compute -- is the classic "adaptive stopping + budgeted compute
allocation" setting. Standard algorithms for it:

  - Fixed-N        : give every operator the same N steps. Non-adaptive (the floor). Sweep N.
  - Confidence>=tau: stop an operator once its readout max-prob >= tau (Wald's SPRT / the
                     deep-net early-exit rule, DeeBERT/entropy-exit). Adaptive, INDEPENDENT. Sweep tau.
  - Greedy VoC     : metareasoning / value-of-computation -- repeatedly spend the next compute
                     unit on the LEAST-confident still-running operator until a shared per-instance
                     budget is exhausted. The principled shared-budget allocator. Sweep budget.
  - Oracle knapsack: label-aware upper bound -- fund each operator to its cheapest correct depth,
                     cheapest first, until the budget runs out. The ceiling. Sweep budget.

  - ENERGY BUDGET  : ours -- a shared scalar pool, dynamic shadow price, refund on commit; commit
                     when settledness s <= theta*price(E). Sweep pool E0.

Because i.i.d. observations make an operator's k-th step exchangeable, we precompute, once with a
fixed noise seed, every operator's confidence/correctness TRAJECTORY over depth 1..T_MAX; all
depth-allocation policies (Fixed-N, tau, VoC, Oracle) are then read off the same trajectories --
a perfectly paired comparison. (The energy budget runs its own settling dynamics via
energy_budget.simulate, on the same readout/observations.)

Run:  python3 compare_baselines.py
"""
import os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_opposites import D, C
from energy_budget import (train_readout, observe, make_instances, frontier_coupled,
                           T_MAX, M, DEVICE)

THETA_BASE, PRICE_GAIN, REFUND = 0.02, 1.0, 0.3
E0S = [10.0, 14.0, 20.0, 28.0, 40.0, 56.0, 80.0, 120.0, 200.0]
TAUS = [0.45, 0.55, 0.65, 0.75, 0.85, 0.92, 0.96, 0.99, 0.999]


@torch.no_grad()
def trajectories(read, poles, y, sigma, dev, seed=0):
    """confidence (max softmax) and correctness at every accumulation depth 1..T_MAX, fixed noise."""
    torch.manual_seed(seed)
    B = poles.shape[0]
    zbar = torch.zeros(B, D, device=dev)
    confs, corrects = [], []
    for k in range(T_MAX):
        z, _ = observe(poles, sigma)
        zbar = (zbar * k + z) / (k + 1)
        lg = read(zbar)
        confs.append(F.softmax(lg, -1).amax(-1))
        corrects.append(lg.argmax(-1) == y)
    return torch.stack(confs), torch.stack(corrects)            # (T,B), (T,B) bool


def f_fixed_n(corr):
    T, B = corr.shape
    return [(float(n), corr[n - 1].float().mean().item() * 100) for n in range(1, T + 1)]


def f_conf_threshold(conf, corr, taus, dev):
    T, B = conf.shape
    ar = torch.arange(B, device=dev)
    out = []
    for tau in taus:
        reach = conf >= tau
        reach[-1] = True
        depth = reach.float().argmax(0)                          # 0..T-1
        acc = corr[depth, ar].float().mean().item() * 100
        out.append(((depth.float() + 1).mean().item(), acc))
    return out


def f_greedy_voc(conf, corr, n_inst, budgets, dev):
    """Spend each next step on the least-confident still-running operator (per instance) until the
    instance's step budget is gone. Value-of-computation / uncertainty-prioritised allocation."""
    T, B = conf.shape
    ar = torch.arange(B, device=dev)
    inst = torch.arange(n_inst, device=dev)
    out = []
    for Bsteps in budgets:
        depth = torch.ones(B, dtype=torch.long, device=dev)     # one observation each to start
        rem = torch.full((n_inst,), float(Bsteps - M), device=dev)
        for _ in range(M * (T_MAX - 1)):
            if (rem <= 0).all():
                break
            cur = conf[(depth - 1).clamp(max=T - 1), ar]
            cur = cur.masked_fill(depth >= T_MAX, float("inf"))  # capped ops can't be chosen
            curm = cur.view(n_inst, M)
            choice = curm.argmin(1)
            chosen_conf = curm.gather(1, choice[:, None]).squeeze(1)
            valid = (rem > 0) & torch.isfinite(chosen_conf)
            flat = (inst * M + choice)[valid]
            depth[flat] += 1
            rem[valid] -= 1
        acc = corr[(depth - 1), ar].float().mean().item() * 100
        out.append((depth.float().mean().item(), acc))
    return out


def f_oracle(corr, n_inst, budgets, dev):
    """Label-aware ceiling: every op gets 1 step (free, correct iff already right); spend the rest
    upgrading ops to their cheapest correct depth, cheapest upgrade first."""
    T, B = corr.shape
    ever = corr.any(0)
    first = corr.float().argmax(0) + 1                            # first correct depth (1..T)
    free = corr[0]                                                # correct at depth 1 already
    upg_cost = torch.where(ever & (~free), (first - 1).float(),
                           torch.full((B,), float("inf"), device=dev))
    base = free.view(n_inst, M).float().sum(1)                    # already correct, cost M total
    sc, _ = upg_cost.view(n_inst, M).sort(1)                      # ascending upgrade costs
    cum = sc.cumsum(1)
    out = []
    for Bsteps in budgets:
        rem = float(Bsteps - M)
        funded = (cum <= rem).sum(1).float()                     # how many upgrades fit
        acc = ((base + funded).sum() / B).item() * 100
        steps = (M + torch.minimum(cum[:, -1].clamp(max=rem), torch.full((n_inst,), rem, device=dev))
                 ).mean().item() / M
        out.append((steps, acc))
    return out


def _table(title, rows):
    print(f"\n{title}")
    print(f"  {'steps/op':>9}{'acc %':>8}")
    for s, a in rows:
        print(f"  {s:>9.2f}{a:>8.2f}")


def run(read, data, n_inst, dev, tag):
    poles, y, sigma, inst_id = data
    conf, corr = trajectories(read, poles, y, sigma, dev)
    budgets = [M * b for b in [1, 1.5, 2, 3, 4, 6, 8, 12, 16]]
    print("\n" + "=" * 72)
    print(f"{tag}: ours (energy budget) vs known algorithms  (n={n_inst} instances, M={M} ops each)")
    fixed = f_fixed_n(corr)
    conf_t = f_conf_threshold(conf, corr, TAUS, dev)
    voc = f_greedy_voc(conf, corr, n_inst, budgets, dev)
    orac = f_oracle(corr, n_inst, budgets, dev)
    ours = frontier_coupled(read, data, n_inst, dev, THETA_BASE, E0S, PRICE_GAIN, REFUND)
    _table("Fixed-N (uniform, non-adaptive)", fixed)
    _table("Confidence>=tau (SPRT / DeeBERT early-exit, independent)", conf_t)
    _table("Greedy VoC (metareasoning shared-budget allocation)", voc)
    _table("Oracle knapsack (label-aware UPPER BOUND)", orac)
    _table("ENERGY BUDGET (ours: shared pool + dynamic price + refund)", ours)
    _plot(tag, fixed, conf_t, voc, orac, ours)


def _plot(tag, fixed, conf_t, voc, orac, ours):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for rows, lab, st in [(orac, "oracle knapsack (upper bound)", "k--"),
                          (conf_t, "confidence>=tau (SPRT/DeeBERT)", "o-"),
                          (voc, "greedy VoC (metareasoning)", "^-"),
                          (fixed, "fixed-N (uniform)", "x-"),
                          (ours, "energy budget (ours)", "s-")]:
        ax.plot([s for s, _ in rows], [v for _, v in rows], st, label=lab)
    ax.set_xlabel("compute (mean inner steps / operator)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title(f"Composed system vs known algorithms -- {tag}")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(p, exist_ok=True)
    fn = os.path.join(p, f"baselines_{tag.lower().replace(' ', '_')}.png")
    fig.savefig(fn, dpi=120)
    print(f"\n  saved {fn}")


def main():
    dev = torch.device(DEVICE)
    print(f"device {DEVICE}  M={M} ops/instance  max ticks={T_MAX}")
    print("training shared readout...", flush=True)
    read = train_readout(dev)
    n_inst = 1024
    run(read, make_instances(n_inst, dev, mode="hetero"), n_inst, dev, "heterogeneous")
    run(read, make_instances(n_inst, dev, mode="distractor"), n_inst, dev, "distractors")
    print("\nread: compare each curve to the oracle ceiling and to confidence>=tau, the standard")
    print("      adaptive-stopping baseline. The energy budget's claim must hold against THESE.")


if __name__ == "__main__":
    main()

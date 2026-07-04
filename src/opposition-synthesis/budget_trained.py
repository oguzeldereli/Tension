"""
budget_trained.py -- replace the PonderNet objective with a loss tied to the GLOBAL energy budget.

energy_budget.py treated the shared pool as an INFERENCE-TIME control over a frozen settling
signal, and it merely tied naive Fixed-N (compare_baselines.py). The reason: nothing ever *learned*
to allocate the shared resource. Here we change the loss.

PonderNet's objective is  E[CE]  +  lambda * E[steps_per_operator]  (a per-operator compute term,
or a KL to a geometric prior). We drop that. The new objective ties the loss to exactly the two
things asked for:

    L  =  E[CE]                      # (a) the synthesis is correct
       +  lambda * E[global_energy]  # (b) the TOTAL energy the shared pool spent on this instance

The compute term is no longer per-operator and separable -- it is the single shared-pool
consumption summed over all M operators of an instance. And the coupling is real: each operator's
commit head reads the dynamic PRICE of the shared pool (price = (E0/E_t)^g, rising as it drains),
which depends on what every other operator is doing. So minimizing L forces the operators to
co-adapt -- to TRIAGE: spend the shared energy where continuing actually lowers CE, and cut the
operators (e.g. hopeless distractors) whose continuation only burns budget. That is precisely the
value-of-continuation signal confidence/settledness lacked.

We keep a PonderNet-style stopping DISTRIBUTION only as the differentiable machinery to take
expectations through the discrete commit -- the OBJECTIVE is the new one above.

Fair test: the readout is trained once and FROZEN; only the commit head is trained by the new loss,
so every method (ours, confidence>=tau, Fixed-N, oracle) shares one readout. We sweep lambda to
trace the accuracy-vs-compute frontier and overlay the known baselines + the oracle ceiling, on the
distractor regime where triage is the whole game. An ablation (no price input) isolates whether the
GLOBAL coupling matters or it is just learned halting.

Run:  python3 budget_trained.py
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
E0 = M * 5.0            # shared pool per instance during training/eval (moderate scarcity)
PRICE_GAIN = 1.0
STEPS_HALT = 900
BATCH_INST = 96         # instances per training step
LR = 3e-3
LAMBDAS = [0.0, 0.02, 0.05, 0.10, 0.25]


class HaltHead(nn.Module):
    """Commit head. Reads value-of-continuation signals + (optionally) the shared-pool price."""
    def __init__(self, use_price=True):
        super().__init__()
        self.use_price = use_price
        in_dim = 3 + (1 if use_price else 0)              # settledness, tension, confidence [, price]
        self.net = nn.Sequential(nn.Linear(in_dim, H), nn.Tanh(), nn.Linear(H, 1))

    def forward(self, s, tmean, conf, price):
        feats = [s, tmean, conf] + ([price] if self.use_price else [])
        return torch.sigmoid(self.net(torch.stack(feats, -1))).squeeze(-1)


def run_system(read, halt, poles, y, sigma, inst_id, n_inst, dev, e0=E0, price_gain=PRICE_GAIN):
    """One differentiable rollout of the composed system with a price-coupled shared pool.
    Returns per-operator expected CE / expected correctness / expected steps, and per-instance
    expected global energy. Uses a PonderNet stopping distribution only to take the expectations."""
    B = poles.shape[0]
    zbar = torch.zeros(B, D, device=dev)
    prev = torch.zeros(B, D, device=dev)
    carry = torch.ones(B, device=dev)                    # prob operator still active (not committed)
    S = torch.zeros(n_inst, device=dev)                  # expected global energy spent so far
    ce_t, corr_t, step_t = [], [], []
    for t in range(T_MAX):
        z, tmean = observe(poles, sigma)
        zbar = (zbar * t + z) / (t + 1)                  # running estimate of the third thing
        s = (zbar - prev).norm(dim=-1); prev = zbar      # settledness
        logits = read(zbar)
        conf = F.softmax(logits, -1).amax(-1)
        price = (e0 / (e0 - S).clamp(min=1e-3)).clamp(min=1.0) ** price_gain   # (n_inst,)
        lam = halt(s, tmean, conf, price[inst_id])
        if t == T_MAX - 1:
            lam = torch.ones_like(lam)                    # must commit by the last tick
        p_halt = lam * carry
        # energy spent THIS tick = operators still active (prob = carry), summed per instance
        S = S + torch.zeros(n_inst, device=dev).index_add_(0, inst_id, carry)
        ce_t.append(p_halt * F.cross_entropy(logits, y, reduction="none"))
        corr_t.append(p_halt * (logits.argmax(-1) == y).float())
        step_t.append(carry)
        carry = carry * (1 - lam)
    exp_ce = torch.stack(ce_t).sum(0)                    # (B,)
    exp_correct = torch.stack(corr_t).sum(0)             # (B,)
    exp_steps = torch.stack(step_t).sum(0)               # (B,) expected steps per operator
    return exp_ce, exp_correct, exp_steps, S             # S: (n_inst,) expected global energy


def train_halt(read, dev, lam_budget, use_price, mode="distractor"):
    """Train ONLY the commit head with L = E[CE] + lam_budget * E[global_energy]/M. Readout frozen."""
    halt = HaltHead(use_price=use_price).to(dev)
    opt = torch.optim.Adam(halt.parameters(), lr=LR)
    for p in read.parameters():
        p.requires_grad_(False)
    read.eval(); halt.train()
    inst_id = torch.arange(BATCH_INST, device=dev).repeat_interleave(M)
    for it in range(STEPS_HALT):
        poles, y, _ = gen(BATCH_INST * M, (1.0, 2.0), dev)
        if mode == "distractor":
            sigma = loguniform(BATCH_INST * M, SIGMA_LO, 0.4, dev)
            hop = torch.rand(BATCH_INST * M, device=dev) < DISTRACT_FRAC
            sigma[hop] = DISTRACT_SIGMA
        else:
            sigma = loguniform(BATCH_INST * M, SIGMA_LO, 0.9, dev)
        exp_ce, _, _, S = run_system(read, halt, poles, y, sigma, inst_id, BATCH_INST, dev)
        loss = exp_ce.mean() + lam_budget * (S.mean() / M)        # (a) correct + (b) GLOBAL budget
        opt.zero_grad(); loss.backward(); opt.step()
    return halt


def eval_frontier(read, dev, data, n_inst, use_price, lambdas):
    poles, y, sigma, inst_id = data
    out = []
    for lam in lambdas:
        halt = train_halt(read, dev, lam, use_price)            # needs grad
        with torch.no_grad():
            _, exp_correct, exp_steps, _ = run_system(read, halt, poles, y, sigma, inst_id, n_inst, dev)
        out.append((exp_steps.mean().item(), exp_correct.mean().item() * 100))
    return out


def _table(title, rows):
    print(f"\n{title}\n  {'steps/op':>9}{'acc %':>8}")
    for s, a in rows:
        print(f"  {s:>9.2f}{a:>8.2f}")


def main():
    dev = torch.device(DEVICE)
    print(f"device {DEVICE}  M={M}  T_MAX={T_MAX}  pool E0={E0}  loss = E[CE] + lambda*E[global_energy]")
    print("training shared readout (frozen for all methods)...", flush=True)
    read = train_readout(dev)

    n_inst = 1024
    dis = make_instances(n_inst, dev, mode="distractor")
    poles, y, sigma, inst_id = dis
    conf, corr = trajectories(read, poles, y, sigma, dev)
    budgets = [M * b for b in [1, 1.5, 2, 3, 4, 6, 8, 12]]

    print("\ntraining budget-tied commit heads (this sweeps lambda; each is a fresh train)...", flush=True)
    ours = eval_frontier(read, dev, dis, n_inst, use_price=True, lambdas=LAMBDAS)
    nopr = eval_frontier(read, dev, dis, n_inst, use_price=False, lambdas=LAMBDAS)

    print("\n" + "=" * 72)
    print("DISTRACTOR regime -- trained global-budget loss vs known algorithms (shared frozen readout)")
    _table("Fixed-N (uniform)", f_fixed_n(corr))
    _table("Confidence>=tau (SPRT / DeeBERT)", f_conf_threshold(conf, corr, TAUS, dev))
    _table("Oracle knapsack (label-aware UPPER BOUND)", f_oracle(corr, n_inst, budgets, dev))
    _table("TRAINED budget loss, NO price (ablation: just learned halting)", nopr)
    _table("TRAINED budget loss + price (ours: global coupling)", ours)

    _plot(f_fixed_n(corr), f_conf_threshold(conf, corr, TAUS, dev),
          f_oracle(corr, n_inst, budgets, dev), nopr, ours)
    print("\nread: does training the loss against the GLOBAL budget (not per-op steps) let the")
    print("      commit head triage distractors -- beating confidence>=tau and closing the oracle")
    print("      gap? And does the price (global coupling) beat the no-price ablation?")


def _plot(fixed, conf_t, orac, nopr, ours):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for rows, lab, st in [(orac, "oracle (upper bound)", "k--"),
                          (conf_t, "confidence>=tau (SPRT)", "o-"),
                          (fixed, "fixed-N", "x-"),
                          (nopr, "trained budget, no price", "^-"),
                          (ours, "trained budget + price (ours)", "s-")]:
        ax.plot([s for s, _ in rows], [v for _, v in rows], st, label=lab)
    ax.set_xlabel("compute (mean inner steps / operator)"); ax.set_ylabel("accuracy (%)")
    ax.set_title("Trained global-budget loss vs known algorithms (distractors)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8); fig.tight_layout()
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(p, exist_ok=True)
    fn = os.path.join(p, "budget_trained_distractor.png")
    fig.savefig(fn, dpi=120); print(f"\n  saved {fn}")


if __name__ == "__main__":
    main()

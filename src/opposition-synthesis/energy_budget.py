"""
energy_budget.py -- a global "psychic energy budget" that couples MANY tension operators.

synth_opposites.py made the synthesis geometry literal (a third thing orthogonal to the
opposition axes). tension_synth_operator.py made ONE operator hold/deliberate/commit. This file
adds the missing third layer: a COMPOSED system -- a "mind" of M operators that all draw on a
single, finite, shared scalar pool of energy. The pool is what turns a static, per-operator
commit penalty into a global, dynamic pressure to commit.

The physics (the string metaphor, taken literally):
  - Holding a stretched string against the loss-landscape pull is METABOLICALLY expensive -- like
    a muscle holding a weight burns ATP doing zero mechanical work. So every operator still
    holding tension drains the shared pool each tick, at a rate proportional to the tension it
    holds (drain_i = t_i).
  - Committing ("cutting the string") RELEASES the stored energy back to the pool: refund = +gamma*t_i.
    A clean resolution subsidizes the operators still deliberating -- the economy is self-funding.
  - The pool may also slowly REPLENISH (refill rho per tick): a sustained-rate budget rather than
    a one-shot one ("dynamic budget").
  - The shared scarcity shows up to each operator as a PRICE -- a dynamic shadow price lambda(E)
    that rises as the pool drains. The old static compute penalty lambda becomes a STATE.

Commit rule, identical structure under every policy, only the price differs:
    commit_i  when  settledness s_i <= theta * price(E)
  Higher price => higher effective threshold => commit while still moving => commit SOONER.
  As the pool empties, price rises, and the whole mind is pushed toward commitment -- exactly the
  "global psychic energy budget pushes all tension operators toward commitment" idea.

Three policies, fair-by-construction (same readout, same settledness signal, same refund/pool --
ONLY the price response differs, so any gap is attributable to the coupling alone):
  - settling : price == 1, no pool. The strong ADAPTIVE baseline (commit when settled). Already
               difficulty-aware: noisy sub-problems settle slower and get more steps on their own.
  - hardstop : same finite pool + refund, but price == 1 -- it ignores the pool until it's empty,
               then force-commits everything still running (the guillotine). = independent halting
               under a hard per-instance budget.
  - coupled  : the same pool + refund, but price(E) rises as it drains, so operators commit a
               little early under scarcity instead of being guillotined (graceful reallocation).

Claim under test (and we report it honestly either way, cf. Experiment B):
  A shared budget only EARNS its keep when (a) sub-problem difficulty is heterogeneous AND (b)
  there is a per-instance budget/deadline (a finite energy for THIS act of thinking). Then the
  coupled price reallocates the fixed energy to where marginal accuracy is highest and beats the
  guillotine at matched compute. With abundant energy, or homogeneous difficulty, coupled should
  collapse onto the plain settling frontier -- coupling buys allocation, not free accuracy.

Run:  python3 energy_budget.py
"""
import os, sys
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_opposites import gen, D, C                       # opposition task + geometry constants
from tension_synth_operator import _project_perp            # the U^perp (settled-equilibrium) projection

DEVICE = "cpu"          # tiny; the GPU hard-powers-off this machine under load
H = 64
T_MAX = 16              # max inner deliberation ticks
TENSION = (1.0, 2.0)    # keep geometry well-conditioned; difficulty comes from OBSERVATION NOISE
SIGMA_LO, SIGMA_HI = 0.05, 0.9      # per-sub-problem observation-noise band (the difficulty axis)
M = 8                   # operators per composed instance ("a mind")
STEPS = 2500
BATCH = 256
LR = 2e-3
SEED = 0


# ----------------------------- task: noisy sequential observation of a synthesis -----------------------------
def loguniform(n, lo, hi, dev):
    return lo * (hi / lo) ** torch.rand(n, device=dev)


def observe(poles, sigma):
    """One noisy observation of the poles -> a single-shot synthesis estimate (in U^perp) and the
    mean string tension. Higher sigma = noisier opposition axes = a noisier, slower-settling z."""
    obs = poles + sigma[:, None, None] * torch.randn_like(poles)
    p_a, p_b = obs[:, 0::2, :], obs[:, 1::2, :]
    diff = p_a - p_b
    t = diff.norm(dim=-1)                                    # (B,K) string tension
    u = diff / (t.unsqueeze(-1) + 1e-6)
    content = (0.5 * (p_a + p_b)).mean(1)                    # aggregated agreement
    z = _project_perp(content, u.transpose(1, 2))           # third thing in U^perp
    return z, t.mean(1)


class Readout(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, H), nn.GELU(), nn.Linear(H, C))

    def forward(self, z):
        return self.net(z)


def train_readout(dev):
    """Learn to read the class (quadrant of the synthesis) from the running estimate zbar, at a
    RANDOM accumulation depth so the readout is valid whenever an operator decides to commit."""
    read = Readout().to(dev)
    opt = torch.optim.Adam(read.parameters(), lr=LR)
    read.train()
    for it in range(STEPS):
        poles, y, _ = gen(BATCH, TENSION, dev)
        sigma = loguniform(BATCH, SIGMA_LO, SIGMA_HI, dev)
        ksteps = int(torch.randint(2, T_MAX + 1, (1,)).item())
        zbar = torch.zeros(BATCH, D, device=dev)
        for k in range(ksteps):
            z, _ = observe(poles, sigma)
            zbar = (zbar * k + z) / (k + 1)                  # running mean of the estimate
        loss = F.cross_entropy(read(zbar), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return read


# ----------------------------- the composed system + the three policies -----------------------------
@torch.no_grad()
def simulate(read, poles, y, sigma, inst_id, n_inst, dev,
             theta, use_pool=False, use_price=False, E0=0.0, price_gain=1.0,
             refund=0.0, refill=0.0, deadline=None, seed=0):
    """Run M*n_inst operators in lockstep; per-instance shared pool when use_pool. Returns metrics.
    Reseed so the observation-noise stream is identical across policies (paired comparison)."""
    torch.manual_seed(seed)
    B = poles.shape[0]
    zbar = torch.zeros(B, D, device=dev)
    cnt = torch.zeros(B, device=dev)
    committed = torch.zeros(B, dtype=torch.bool, device=dev)
    forced = torch.zeros(B, dtype=torch.bool, device=dev)    # committed by pool/deadline, not by settling
    answer = torch.zeros(B, C, device=dev)
    compute = torch.zeros(n_inst, device=dev)                # op-ticks spent per instance
    E = torch.full((n_inst,), float(E0), device=dev)

    for t in range(1, T_MAX + 1):
        active = ~committed
        if not active.any():
            break
        z, tmean = observe(poles, sigma)
        a = active
        cnt = cnt + a.float()
        zprev = zbar.clone()
        cn = cnt.clamp(min=1).unsqueeze(-1)
        zbar = torch.where(a.unsqueeze(-1), (zbar * (cnt - 1).clamp(min=0).unsqueeze(-1) + z) / cn, zbar)
        s = (zbar - zprev).norm(dim=-1)                      # settledness: movement of the estimate

        compute.index_add_(0, inst_id[a], torch.ones(int(a.sum()), device=dev))

        broke = torch.zeros(B, dtype=torch.bool, device=dev)
        price_op = torch.ones(B, device=dev)
        if use_pool:
            # holding cost in COMPUTE units (1 per active op per tick) so E0 is literally the
            # step-budget for this thought and is directly comparable to the x-axis. (The general
            # metabolic form drain ~ tension reduces to this when tension is homogeneous, as here.)
            drain = torch.zeros(n_inst, device=dev)
            drain.index_add_(0, inst_id[a], torch.ones(int(a.sum()), device=dev))
            E = E - drain + refill
            broke = (E <= 0)[inst_id]                        # pool empty -> forced commit
            if use_price:
                price = (E0 / E.clamp(min=1e-3)).clamp(min=1.0) ** price_gain
                price_op = price[inst_id]

        theta_eff = theta * price_op
        force = broke.clone()
        force |= (t == T_MAX)
        if deadline is not None:
            force |= (compute >= deadline)[inst_id]
        settled = s <= theta_eff
        commit_now = active & (settled | force)
        # released energy refunds the pool (only on REAL commits this tick)
        if use_pool and refund and commit_now.any():
            ref = torch.zeros(n_inst, device=dev)
            ref.index_add_(0, inst_id[commit_now],
                           torch.full((int(commit_now.sum()),), float(refund), device=dev))
            E = E + ref
        answer[commit_now] = read(zbar[commit_now])
        forced |= commit_now & (~settled) & force
        committed |= commit_now

    rem = ~committed
    if rem.any():
        answer[rem] = read(zbar[rem]); committed |= rem

    correct = answer.argmax(-1) == y
    acc = correct.float().mean().item() * 100
    steps_per_op = compute.sum().item() / B
    # hard tail: instances whose mean difficulty is in the top quartile
    inst_sig = torch.zeros(n_inst, device=dev).index_add_(0, inst_id, sigma) / M
    hard = inst_sig >= inst_sig.quantile(0.75)
    hard_ops = hard[inst_id]
    acc_hard = correct[hard_ops].float().mean().item() * 100
    trunc = forced.float().mean().item() * 100
    return dict(acc=acc, steps=steps_per_op, acc_hard=acc_hard, trunc=trunc)


DISTRACT_SIGMA = 5.0    # "hopeless" operators: noise so high they never settle within the budget
DISTRACT_FRAC = 0.4


def make_instances(n_inst, dev, mode="hetero"):
    B = n_inst * M
    poles, y, _ = gen(B, TENSION, dev)
    if mode == "hetero":
        sigma = loguniform(B, SIGMA_LO, SIGMA_HI, dev)
    elif mode == "homo":
        sigma = torch.full((B,), (SIGMA_LO * SIGMA_HI) ** 0.5, device=dev)   # matched geo-mean difficulty
    elif mode == "distractor":
        # a fraction of operators are hopeless drains: huge noise -> never settle, ~chance accuracy.
        # A good shared budget should triage them OUT and reallocate the freed energy to the solvable
        # ones; independent settling can only chase them to max compute.
        sigma = loguniform(B, SIGMA_LO, 0.4, dev)
        hop = torch.rand(B, device=dev) < DISTRACT_FRAC
        sigma[hop] = DISTRACT_SIGMA
    else:
        raise ValueError(mode)
    inst_id = torch.arange(n_inst, device=dev).repeat_interleave(M)
    return poles, y, sigma, inst_id


# ----------------------------- experiments -----------------------------
def frontier_settling(read, data, n_inst, dev, thetas):
    poles, y, sigma, inst_id = data
    out = []
    for th in thetas:
        r = simulate(read, poles, y, sigma, inst_id, n_inst, dev, theta=th)
        out.append((r["steps"], r["acc"]))
    return out


def frontier_coupled(read, data, n_inst, dev, theta_base, E0s, price_gain, refund, refill=0.0):
    poles, y, sigma, inst_id = data
    out = []
    for e in E0s:
        r = simulate(read, poles, y, sigma, inst_id, n_inst, dev, theta=theta_base,
                     use_pool=True, use_price=True, E0=e, price_gain=price_gain,
                     refund=refund, refill=refill)
        out.append((r["steps"], r["acc"]))
    return out


def main():
    dev = torch.device(DEVICE)
    torch.manual_seed(SEED)
    print(f"device {DEVICE}  D={D} classes={C}  operators/instance M={M}  max ticks={T_MAX}")
    print(f"difficulty = observation noise sigma in [{SIGMA_LO},{SIGMA_HI}] (log-uniform)\n")

    print("training shared readout (z -> class) at random accumulation depth...", flush=True)
    read = train_readout(dev)

    n_inst = 1024
    het = make_instances(n_inst, dev, mode="hetero")
    hom = make_instances(n_inst, dev, mode="homo")
    dis = make_instances(n_inst, dev, mode="distractor")

    THETAS = [0.30, 0.22, 0.16, 0.12, 0.09, 0.07, 0.05, 0.035, 0.02]
    E0S = [10.0, 14.0, 20.0, 28.0, 40.0, 56.0, 80.0, 120.0, 200.0]   # pool ~ M*steps (M=8)
    THETA_BASE = 0.02       # think long unless the price says otherwise
    PRICE_GAIN = 1.0
    REFUND = 0.3

    # ---- Exp 1: accuracy-vs-compute frontier, HETEROGENEOUS difficulty ----
    fs = frontier_settling(read, het, n_inst, dev, THETAS)
    fc = frontier_coupled(read, het, n_inst, dev, THETA_BASE, E0S, PRICE_GAIN, REFUND)
    print("=" * 72)
    print("Exp 1  accuracy vs compute frontier  (HETEROGENEOUS difficulty)")
    print("  settling (adaptive baseline): commit when settled (sweep theta)")
    print(f"  coupled  (energy budget):     theta={THETA_BASE}, refund={REFUND} (sweep pool E0)\n")
    _print_two(fs, fc)

    # ---- Exp 2: same finite pool, price response vs guillotine (hardstop) ----
    poles, y, sigma, inst_id = het
    print("\n" + "=" * 72)
    print("Exp 2  same finite pool + refund; ONLY the price response differs (HETEROGENEOUS)")
    print(f"{'pool E0':>8}{'policy':>10}{'steps/op':>10}{'acc':>8}{'acc(hard)':>11}{'forced%':>9}")
    for e in [20.0, 28.0, 40.0, 56.0]:
        hs = simulate(read, poles, y, sigma, inst_id, n_inst, dev, theta=THETA_BASE,
                      use_pool=True, use_price=False, E0=e, refund=REFUND)
        cp = simulate(read, poles, y, sigma, inst_id, n_inst, dev, theta=THETA_BASE,
                      use_pool=True, use_price=True, E0=e, price_gain=PRICE_GAIN, refund=REFUND)
        print(f"{e:>8.1f}{'hardstop':>10}{hs['steps']:>10.2f}{hs['acc']:>8.2f}"
              f"{hs['acc_hard']:>11.2f}{hs['trunc']:>9.1f}")
        print(f"{'':>8}{'coupled':>10}{cp['steps']:>10.2f}{cp['acc']:>8.2f}"
              f"{cp['acc_hard']:>11.2f}{cp['trunc']:>9.1f}")

    # ---- Exp 3: HOMOGENEOUS control -- coupling should add ~nothing ----
    fsh = frontier_settling(read, hom, n_inst, dev, THETAS)
    fch = frontier_coupled(read, hom, n_inst, dev, THETA_BASE, E0S, PRICE_GAIN, REFUND)
    print("\n" + "=" * 72)
    print("Exp 3  same frontier, HOMOGENEOUS difficulty (control: expect coupled ~ settling)\n")
    _print_two(fsh, fch)

    # ---- Exp 4: DISTRACTORS -- the regime a shared pool should actually win ----
    fsd = frontier_settling(read, dis, n_inst, dev, THETAS)
    fcd = frontier_coupled(read, dis, n_inst, dev, THETA_BASE, E0S, PRICE_GAIN, REFUND)
    print("\n" + "=" * 72)
    print(f"Exp 4  {int(DISTRACT_FRAC*100)}% hopeless DISTRACTOR operators (sigma={DISTRACT_SIGMA}) per instance")
    print("  settling chases non-settling drains to max compute; the price should triage them out\n")
    _print_two(fsd, fcd)

    _maybe_plot(fs, fc, fsd, fcd)
    print("\nread (the honest verdict): at MATCHED compute, per-op settling ties or beats the")
    print("      coupled budget everywhere (Exp 1-3) -- as an ACCURACY mechanism the global budget")
    print("      is falsified on settling-style composition. Its one real benefit (Exp 4): when")
    print("      hopeless operators drain compute, settling has a hard compute FLOOR it can't go")
    print("      below, while the budget bounds/controls compute down to any level. The budget is a")
    print("      compute-robustness mechanism, not an accuracy one -- here.")


def _print_two(fs, fc):
    print(f"  {'settling':<28}{'coupled (budget)':<28}")
    print(f"  {'steps/op':>9}{'acc':>8}    {'steps/op':>9}{'acc':>8}")
    for (s1, a1), (s2, a2) in zip(fs, fc):
        print(f"  {s1:>9.2f}{a1:>8.2f}    {s2:>9.2f}{a2:>8.2f}")


def _maybe_plot(fs, fc, fsd, fcd):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, fsx, fcx, title in [(ax[0], fs, fc, "Heterogeneous difficulty (budget inert)"),
                               (ax[1], fsd, fcd, "40% distractors (budget bounds compute)")]:
        a.plot([s for s, _ in fsx], [v for _, v in fsx], "o-", label="settling (adaptive)")
        a.plot([s for s, _ in fcx], [v for _, v in fcx], "s-", label="coupled (energy budget)")
        a.set_xlabel("compute (mean inner steps / operator)")
        a.set_title(title); a.grid(alpha=0.3); a.legend()
    ax[0].set_ylabel("accuracy at commit (%)")
    fig.suptitle("Global psychic-energy budget vs independent settling")
    fig.tight_layout()
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(p, exist_ok=True)
    fig.savefig(os.path.join(p, "energy_budget_frontier.png"), dpi=120)
    print(f"\nsaved figures/energy_budget_frontier.png")


if __name__ == "__main__":
    main()

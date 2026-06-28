"""
Benchmark 1 -- "when to commit to a coin flip" (sequential evidence under uncertainty).

The regime your goals named and the old tasks lacked: irreducible stochastic evidence, no
synthesis available, must commit to a SIDE, and the real question is WHEN -- spend more
samples only when it's actually hard.

Setup. Per episode draw a class y in {0,1} and a difficulty delta ~ U[lo,hi]; the coin's
bias is p = 0.5 + (2y-1)*delta. The model sees a stream of +-1 samples (Bernoulli) and
must name y using as FEW samples as it can. delta varies per episode, so a single fixed
sample budget is wasteful on easy coins and too short on hard ones -- exactly where
deciding-when-to-stop pays.

Baselines (both see only the samples, neither is told delta -- fair):
  fixed-N majority : read N samples, take the sign of the sum. This is the OPTIMAL
                     non-adaptive rule (the count is a sufficient statistic). Sweeping N
                     traces the best accuracy-vs-samples frontier without adaptivity.
  Wald SPRT (d0)   : the classic sequential test, hand-derived for a single reference
                     delta d0=mean. Adaptive, but mis-specified for our mixture of deltas.

Claim under test:
  (1) the TensionBlock's learned adaptive policy beats the fixed-N frontier -- higher
      accuracy at equal AVERAGE samples (it reallocates compute to hard coins);
  (2) its commit time tracks difficulty (corr(halt, |delta|) << 0): a SENSE OF TIME;
  (3) it is NOT a stats table -- trained on one delta band it still commits later on
      unseen-harder coins and earlier on unseen-easier ones (the policy extrapolates).

Run:  python3 bench_coin.py
"""
import math
import numpy as np
import torch
from tension_block import TensionBlock, ponder_ce_loss, halt_infer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T_MAX = 40
HIDDEN = 48
BATCH = 512
STEPS = 3000
LR = 3e-3
TRAIN_LO, TRAIN_HI = 0.05, 0.35
LAMBDAS = [0.020, 0.008, 0.003, 0.001]      # compute penalty: high -> fast, low -> deliberate


def gen(B, dev, lo=TRAIN_LO, hi=TRAIN_HI):
    y = (torch.rand(B, device=dev) < 0.5).long()
    delta = torch.rand(B, device=dev) * (hi - lo) + lo
    p = 0.5 + (2 * y - 1).float() * delta
    bits = (torch.rand(T_MAX, B, device=dev) < p.unsqueeze(0)).float()
    evidence = (2 * bits - 1).unsqueeze(-1)     # +-1 stream, (T,B,1)
    return evidence, y, delta


def train(lambda_c, dev):
    blk = TensionBlock(1, HIDDEN, 2, T_MAX).to(dev)
    opt = torch.optim.Adam(blk.parameters(), lr=LR)
    for _ in range(STEPS):
        evidence, y, _ = gen(BATCH, dev)
        out = blk(evidence)
        loss, _, _ = ponder_ce_loss(out, y, lambda_c=lambda_c)
        opt.zero_grad(); loss.backward(); opt.step()
    return blk


@torch.no_grad()
def eval_block(blk, dev, lo=TRAIN_LO, hi=TRAIN_HI, n=16384):
    evidence, y, delta = gen(n, dev, lo, hi)
    out = blk(evidence)
    pred, steps, _ = halt_infer(out)
    acc = (pred == y).float().mean().item() * 100
    avg = steps.float().mean().item()
    # sense of time: correlation between commit step and difficulty (|delta|)
    s, d = steps.float(), delta
    corr = (((s - s.mean()) * (d - d.mean())).mean() / (s.std() * d.std() + 1e-8)).item()
    return acc, avg, corr, steps.float(), delta


@torch.no_grad()
def fixed_N_majority(dev, n=16384, lo=TRAIN_LO, hi=TRAIN_HI):
    """Optimal non-adaptive frontier: accuracy(N) for N = 1..T_MAX."""
    evidence, y, _ = gen(n, dev, lo, hi)
    s = evidence.squeeze(-1)                      # (T,B) in +-1
    csum = torch.cumsum(s, dim=0)                 # (T,B)
    sign = torch.sign(csum)
    tie = sign == 0
    sign[tie] = (torch.rand_like(sign[tie]) < 0.5).float() * 2 - 1
    pred = (sign > 0).long()                      # +sum -> class 1
    acc = (pred == y.unsqueeze(0)).float().mean(1).cpu().numpy() * 100   # (T,)
    return np.arange(1, T_MAX + 1), acc


@torch.no_grad()
def sprt_curve(dev, n=16384, lo=TRAIN_LO, hi=TRAIN_HI):
    """Wald SPRT with reference delta d0=mean(band); sweep the boundary -> (avg N, acc)."""
    evidence, y, _ = gen(n, dev, lo, hi)
    s = evidence.squeeze(-1)
    d0 = 0.5 * (lo + hi)
    llr_step = torch.where(s > 0, math.log((0.5 + d0) / (0.5 - d0)),
                                  math.log((0.5 - d0) / (0.5 + d0)))
    llr = torch.cumsum(llr_step, dim=0)          # (T,B)
    pts = []
    for A in [0.3, 0.6, 1.0, 1.6, 2.4, 3.5, 5.0, 7.0]:
        crossed = llr.abs() >= A
        crossed[-1] = True
        step = crossed.float().argmax(0)
        dec = (llr[step, torch.arange(llr.shape[1], device=dev)] > 0).long()
        acc = (dec == y).float().mean().item() * 100
        pts.append((float((step + 1).float().mean()), acc))
    return pts


def interp_acc(Ns, accs, x):
    return float(np.interp(x, Ns, accs))


def main():
    dev = torch.device(DEVICE)
    torch.manual_seed(0)
    print(f"device {DEVICE}  T_max {T_MAX}  train delta in [{TRAIN_LO},{TRAIN_HI}]\n")

    Ns, fixed_acc = fixed_N_majority(dev)
    sprt = sprt_curve(dev)

    print("=== speed-accuracy: TensionBlock vs optimal non-adaptive (fixed-N majority) ===")
    print(f"{'lambda_c':>10}{'avg samples':>13}{'acc %':>9}{'fixedN@same N acc%':>20}{'Δacc':>8}{'corr(halt,|d|)':>16}")
    blocks = {}
    pts_tb = []
    for lc in LAMBDAS:
        blk = train(lc, dev); blocks[lc] = blk
        acc, avg, corr, _, _ = eval_block(blk, dev)
        fixed_at = interp_acc(Ns, fixed_acc, avg)
        pts_tb.append((avg, acc))
        print(f"{lc:>10.3f}{avg:>13.2f}{acc:>9.2f}{fixed_at:>20.2f}{acc - fixed_at:>8.2f}{corr:>16.3f}")

    print("\n  (Δacc>0 means: at the SAME average sample budget, adaptivity wins.)")
    print("\n  Wald SPRT(d0=mean) reference points (avg samples, acc%):")
    print("   " + "  ".join(f"({a:.1f},{b:.1f})" for a, b in sprt))

    # not-a-stats-table: take a mid operating point, test on unseen delta bands
    blk = blocks[0.008]
    print("\n=== not a stats table: policy extrapolates to UNSEEN difficulty bands ===")
    print(f"  trained band     [{TRAIN_LO},{TRAIN_HI}]")
    for tag, (lo, hi) in [("unseen-HARDER [0.02,0.05]", (0.02, 0.05)),
                          ("in-band       [0.05,0.35]", (TRAIN_LO, TRAIN_HI)),
                          ("unseen-EASIER [0.35,0.48]", (0.35, 0.48))]:
        acc, avg, corr, _, _ = eval_block(blk, dev, lo, hi)
        print(f"  {tag:<28} avg samples {avg:5.2f}   acc {acc:5.2f}%")
    print("  expect: avg samples DECREASES from harder->easier (per-instance timing, not a table).")

    try:
        plot(Ns, fixed_acc, pts_tb, sprt)
    except Exception as e:
        print(f"\n(plot skipped: {e})")


def plot(Ns, fixed_acc, pts_tb, sprt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4.2))
    plt.plot(Ns, fixed_acc, "-", color="gray", label="fixed-N majority (optimal non-adaptive)")
    sx, sy = zip(*sprt)
    plt.plot(sx, sy, "s--", color="tab:orange", ms=4, label="Wald SPRT (d0=mean)")
    tx, ty = zip(*pts_tb)
    plt.plot(tx, ty, "o-", color="tab:blue", ms=7, label="TensionBlock (adaptive)")
    plt.xlabel("average samples used (compute)"); plt.ylabel("accuracy %")
    plt.title("When to commit: adaptive deliberation vs fixed budget")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    out = "figures/coin_speed_accuracy.png"
    plt.savefig(out, dpi=130); print(f"\nsaved {out}")


if __name__ == "__main__":
    main()

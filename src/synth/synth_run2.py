"""
v2 test bed: intrinsic release + forced holding under information-insufficiency.

Trains IntrinsicField and InterpLeaner2 on the staggered, end-flag-free task, in two
regimes of evidence quality:
    clean        : votes are exact.
    uncertainty  : votes are noisy -> regime only recoverable by integrating over time
                   ("complete synthesis under uncertainty").

Reports, per operator:
  decisive / BALANCED acc %  : resolves to the pole / to the orthogonal synthesis.
  balanced off-axis (deg)    : ~90 = the third thing; ~0 = sat on the bisector (leaning).
  determinism std (deg)      : same poles, different cancelling votes -> spread of answer.

  -- the two things v2 adds --
  HOLD |z| before t*         : mean magnitude while info is insufficient. ~0 = genuinely
                               holding at the null (emitting nothing) until it has the
                               poles+votes. >0 = committing before it can know.
  commit==final agree %      : does reading the answer at the INTRINSIC rest point (speed
                               -> 0, detected, not told) give the same answer as the last
                               step? high = the operator's own commit picks the resolution.
  commit-step vs tB corr     : correlation between pole-B arrival and the detected commit
                               step. ~1 = timing is owned by the dynamics tracking when
                               information actually arrives, NOT a fixed clock.
  commit step | tB=lo/hi     : mean commit step when B arrives early vs late. It should
                               slide later as B arrives later -> the release is intrinsic.

Run:  python3 synth_run2.py
"""
import math
import torch
from synth_task2 import sample_episode, angle_deg, T, T_VOTE, TB_MIN, TB_MAX
from synth_models2 import build, detect_commit

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 4000
BATCH = 512
LR = 5e-3
TOL = 15.0
SETTLE_W = 0.05
HOLD_W = 0.5
EPS = 0.02          # rest threshold for intrinsic commit
MAG = 0.5          # committed-magnitude threshold


def train(name, noise, dev):
    op = build(name).to(dev)
    opt = torch.optim.Adam(op.parameters(), lr=LR)
    step_idx = torch.arange(T, device=dev).unsqueeze(-1)        # (T,1)
    for _ in range(STEPS):
        obs, target, reg, _, _, _, _, tB, tstar = sample_episode(BATCH, dev, noise=noise)
        z_path, speeds = op.rollout(obs)
        # COMMIT loss on z itself (not just its direction): z must REACH the unit target
        # vector. Scoring direction alone admits a degenerate optimum -- hold z
        # infinitesimally in the right direction forever (|z|~0, never move) -- which the
        # settle penalty actively rewards. Demanding |z|->1 makes committing mandatory, so
        # "come to rest after committing" becomes a real event instead of "never start".
        commit = ((z_path[-1] - target) ** 2).sum(-1).mean()
        # settle over the post-sufficiency tail: rest ASAP once information has arrived, so
        # the rest point is a detectable snap-then-stop, not a creep to the final step.
        tail_mask = (step_idx >= tstar.unsqueeze(0)).float()    # (T,B)
        settle = (tail_mask * speeds).sum() / tail_mask.sum().clamp(min=1)
        # hold: stay at the null while information is insufficient (t < t*)
        hold_mask = (step_idx < tstar.unsqueeze(0)).float()     # (T,B)
        mags = z_path.norm(dim=-1)
        hold = (hold_mask * mags ** 2).sum() / hold_mask.sum().clamp(min=1)
        loss = commit + SETTLE_W * settle + HOLD_W * hold
        opt.zero_grad(); loss.backward(); opt.step()
    return op


@torch.no_grad()
def evaluate(op, noise, dev):
    out = {}
    step_idx = torch.arange(T, device=dev).unsqueeze(-1)
    for reg_id, key in [(0, "decisive_A"), (1, "decisive_B"), (2, "balanced")]:
        obs, target, reg, uA, uB, mean, perp, tB, tstar = sample_episode(4096, dev, regime=reg_id, noise=noise)
        z_path, speeds = op.rollout(obs)
        ans = z_path[-1] / (z_path[-1].norm(dim=-1, keepdim=True) + 1e-8)
        err = angle_deg(ans, target)
        out[key] = (err.mean().item(), (err < TOL).float().mean().item() * 100)
        if reg_id == 2:
            out["balanced_offaxis_deg"] = angle_deg(ans, mean).mean().item()
            # holding magnitude before info-sufficiency
            hold_mask = (step_idx < tstar.unsqueeze(0)).float()
            mags = z_path.norm(dim=-1)
            out["hold_mag"] = ((hold_mask * mags).sum() / hold_mask.sum().clamp(min=1)).item()
            # intrinsic commit vs final-step agreement
            cidx, zc = detect_commit(z_path, speeds, eps=EPS, mag=MAG)
            zc_u = zc / (zc.norm(dim=-1, keepdim=True) + 1e-8)
            agree = (angle_deg(zc_u, ans) < TOL).float().mean().item() * 100
            out["commit_agree"] = agree

    # intrinsic timing: vary tB widely, correlate detected commit step with tB.
    obs, target, reg, uA, uB, mean, perp, tB, tstar = sample_episode(4096, dev, regime=2, noise=noise)
    z_path, speeds = op.rollout(obs)
    cidx, _ = detect_commit(z_path, speeds, eps=EPS, mag=MAG)
    tBf, cf = tB.float(), cidx.float()
    vt, vc = tBf - tBf.mean(), cf - cf.mean()
    corr = (vt * vc).mean() / (vt.std() * vc.std() + 1e-8)
    out["commit_tB_corr"] = corr.item()
    lo = cf[tB <= TB_MIN + 2].mean().item()
    hi = cf[tB >= TB_MAX - 2].mean().item()
    out["commit_lo"], out["commit_hi"] = lo, hi

    # determinism: one pole pair, many cancelling-vote balanced episodes
    o1, *_ = sample_episode(1, dev, regime=2, noise=noise, tB_fixed=TB_MIN)
    o2, _, _, _, _, _, _, _, _ = sample_episode(256, dev, regime=2, noise=noise, tB_fixed=TB_MIN)
    o2[:, :, 0:2] = o1[0, 0, 0:2]
    # rebuild B presence for the shared poles (B present from TB_MIN on)
    uB1 = o1[T - 1, 0, 2:4]
    pres = (torch.arange(T, device=dev) >= TB_MIN).float().unsqueeze(-1).unsqueeze(-1)
    o2[:, :, 2:4] = uB1 * pres
    z2, sp2 = op.rollout(o2)
    a2 = z2[-1]
    ang = torch.atan2(a2[:, 1], a2[:, 0])
    c, s = torch.cos(ang).mean(), torch.sin(ang).mean()
    R = torch.sqrt(c * c + s * s).clamp(1e-8, 1.0)
    out["determinism_std_deg"] = math.degrees(math.sqrt(-2.0 * math.log(R.item())))
    return out


def report(title, rows):
    hdr = f"{'metric':<28}" + "".join(f"{n:>14}" for n in rows)
    print(f"\n===== {title} =====")
    print(hdr); print("-" * len(hdr))
    metrics = [
        ("decisive_A acc %", lambda r: f"{r['decisive_A'][1]:.1f}"),
        ("decisive_B acc %", lambda r: f"{r['decisive_B'][1]:.1f}"),
        ("BALANCED acc %", lambda r: f"{r['balanced'][1]:.1f}"),
        ("balanced err (deg)", lambda r: f"{r['balanced'][0]:.1f}"),
        ("balanced off-axis (deg)", lambda r: f"{r['balanced_offaxis_deg']:.1f}"),
        ("determinism std (deg)", lambda r: f"{r['determinism_std_deg']:.2f}"),
        ("HOLD |z| before t*", lambda r: f"{r['hold_mag']:.3f}"),
        ("commit==final agree %", lambda r: f"{r['commit_agree']:.1f}"),
        ("commit-step vs tB corr", lambda r: f"{r['commit_tB_corr']:.2f}"),
        ("commit step | tB lo", lambda r: f"{r['commit_lo']:.1f}"),
        ("commit step | tB hi", lambda r: f"{r['commit_hi']:.1f}"),
    ]
    for label, fn in metrics:
        print(f"{label:<28}" + "".join(f"{fn(rows[n]):>14}" for n in rows))


def main():
    dev = torch.device(DEVICE)
    torch.manual_seed(0)
    print(f"device {DEVICE}  steps {STEPS}  T {T}  votes[0,{T_VOTE})  tB in [{TB_MIN},{TB_MAX}]")
    for noise, title in [(0.0, "CLEAN evidence"), (0.6, "UNCERTAINTY (noisy votes, sigma=0.6)")]:
        rows = {}
        for name in ["intrinsic", "interp"]:
            op = train(name, noise, dev)
            rows[name] = evaluate(op, noise, dev)
        report(title, rows)
    print("\nread: intrinsic should HOLD ~0 before t*, commit agree ~100, corr ~1,")
    print("      commit step slide later as tB grows (timing owned by the dynamics).")
    print("      interp leans (off-axis ~0, BALANCED low) and never truly holds/rests.")


if __name__ == "__main__":
    main()

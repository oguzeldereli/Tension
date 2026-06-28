"""
Build both (all three), test both, see how it goes.

Trains StructuredField, LearnedField, InterpLeaner on the angular synthesis task and
prints the comparison that actually matters:

  - decisive accuracy   : can it resolve to a pole when evidence is clear? (all should)
  - BALANCED accuracy    : does it produce the orthogonal synthesis under insufficiency?
                           (the whole question -- interp should fail here)
  - balanced off-axis ang: angle of the balanced answer from the bisector. ~90 = the
                           third thing (orthogonal). ~0 = sat on the axis (leaning).
  - determinism std      : same poles, different cancelling vote patterns -> std of the
                           predicted angle. ~0 = the synthesis is determinate (depends
                           on poles+balance, not on the vote details).
  - mean settle speed    : did the dynamics come to rest (tension dissolved)?

Run:  python3 synth_run.py
"""
import math
import torch
from synth_task import sample_episode, angle_deg
from synth_models import build

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 3000
BATCH = 512
LR = 5e-3
TOL = 15.0          # angular-error tolerance for "correct" (deg)
SETTLE_W = 0.1      # encourage the field to come to rest


def train(name):
    dev = torch.device(DEVICE)
    op = build(name).to(dev)
    opt = torch.optim.Adam(op.parameters(), lr=LR)
    for it in range(STEPS):
        obs, target, reg, *_ = sample_episode(BATCH, dev)
        ans, speeds = op.rollout(obs)
        dir_loss = (1.0 - (ans * target).sum(-1)).mean()
        settle = speeds[-1].mean()
        loss = dir_loss + SETTLE_W * settle
        opt.zero_grad()
        loss.backward()
        opt.step()
    return op


@torch.no_grad()
def evaluate(op, dev):
    out = {}
    # per-regime accuracy + error
    for reg_id, key in [(0, "decisive_A"), (1, "decisive_B"), (2, "balanced")]:
        obs, target, reg, uA, uB, mean, perp = sample_episode(4096, dev, regime=reg_id)
        ans, speeds = op.rollout(obs)
        err = angle_deg(ans, target)
        out[key] = (err.mean().item(), (err < TOL).float().mean().item() * 100,
                    speeds[-1].mean().item())
        if reg_id == 2:
            offaxis = angle_deg(ans, mean)        # angle from bisector; 90 = orthogonal
            out["balanced_offaxis_deg"] = offaxis.mean().item()

    # determinism: ONE fixed pole pair, many balanced episodes (varying vote patterns)
    obs1, *_ , uA1, uB1, _, _ = sample_episode(1, dev, regime=2)
    obs2, _, _, _, _, _, _ = sample_episode(256, dev, regime=2)
    obs2[:, :, 0:2] = uA1[0]          # force all 256 to share the same poles
    obs2[:, :, 2:4] = uB1[0]
    ans2, _ = op.rollout(obs2)
    ang = torch.atan2(ans2[:, 1], ans2[:, 0])
    c, s = torch.cos(ang).mean(), torch.sin(ang).mean()
    R = torch.sqrt(c * c + s * s).clamp(1e-8, 1.0)
    out["determinism_std_deg"] = math.degrees(math.sqrt(-2.0 * math.log(R.item())))
    return out


def main():
    dev = torch.device(DEVICE)
    print(f"device {DEVICE}  steps {STEPS}\n")
    rows = {}
    for name in ["structured", "learned", "interp"]:
        op = train(name)
        rows[name] = evaluate(op, dev)

    def cell(r, k):
        v = r[k]
        return v

    hdr = f"{'metric':<26}" + "".join(f"{n:>14}" for n in rows)
    print(hdr)
    print("-" * len(hdr))
    metrics = [
        ("decisive_A acc %", lambda r: f"{r['decisive_A'][1]:.1f}"),
        ("decisive_B acc %", lambda r: f"{r['decisive_B'][1]:.1f}"),
        ("BALANCED acc %", lambda r: f"{r['balanced'][1]:.1f}"),
        ("balanced err (deg)", lambda r: f"{r['balanced'][0]:.1f}"),
        ("balanced off-axis (deg)", lambda r: f"{r['balanced_offaxis_deg']:.1f}"),
        ("determinism std (deg)", lambda r: f"{r['determinism_std_deg']:.2f}"),
        ("settle speed (balanced)", lambda r: f"{r['balanced'][2]:.4f}"),
    ]
    for label, fn in metrics:
        print(f"{label:<26}" + "".join(f"{fn(rows[n]):>14}" for n in rows))

    print("\nread: BALANCED acc + off-axis ~90 = produced the orthogonal third thing.")
    print("      interp should ace decisive, fail balanced, sit at off-axis ~0 (the bisector).")


if __name__ == "__main__":
    main()

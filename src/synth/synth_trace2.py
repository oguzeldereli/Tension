"""
v2 trajectory trace -- the visual proof of intrinsic, information-driven commit.

For balanced episodes with DIFFERENT pole-B arrival times tB, print per step:
    |z|   : distance from the null (0 = holding/emitting nothing, 1 = committed)
    speed : how fast it's moving (the discomfort readout)
    off   : current angle from the bisector (~90 = it has reached the synthesis)
and mark the detected commit step (first step with |z|>0.5 AND speed<eps -- the field at
rest on a committed symbol; nobody told it when).

What you should see: |z| pinned at ~0 until step tB (it literally cannot synthesize
without the second pole), then a snap up to ~1 with off swinging to ~90, then rest. The
snap ONSET moves with tB, and the commit marker moves with it -- the timing is owned by
the dynamics tracking when information arrives, not by any clock in the input.

Run:  python3 synth_trace2.py
"""
import torch
from synth_task2 import sample_episode, angle_deg, T
from synth_models2 import build, detect_commit
from synth_run2 import train, EPS, MAG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def trace(op, dev, tBs=(3, 7, 11)):
    print(f"\n=== IntrinsicField: hold -> snap(at tB) -> rest, across pole-B arrivals ===")
    print("(|z| 0=holding 1=committed | speed=discomfort | off=angle from bisector, 90=synthesis)\n")
    for tB in tBs:
        obs, target, reg, uA, uB, mean, perp, tBv, tstar = sample_episode(1, dev, regime=2, tB_fixed=tB)
        z_path, speeds = op.rollout(obs)
        cidx, zc = detect_commit(z_path, speeds, eps=EPS, mag=MAG)
        mags = z_path[:, 0].norm(dim=-1)
        zu = z_path[:, 0] / (mags.unsqueeze(-1) + 1e-8)
        off = angle_deg(zu, mean.expand(T, 2))
        print(f"pole-B arrives tB={tB}   detected commit step = {cidx.item()}   (no end flag exists)")
        print("  step:  " + " ".join(f"{t:5d}" for t in range(T)))
        print("  |z|:   " + " ".join(f"{mags[t].item():5.2f}" for t in range(T)))
        print("  speed: " + " ".join(f"{speeds[t,0].item():5.2f}" for t in range(T)))
        print("  off:   " + " ".join(f"{off[t].item():5.0f}" for t in range(T)))
        print()


def main():
    dev = torch.device(DEVICE)
    torch.manual_seed(0)
    op = train("intrinsic", 0.0, dev)
    trace(op, dev)


if __name__ == "__main__":
    main()

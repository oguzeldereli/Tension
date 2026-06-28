"""
Trajectory trace: does the field HOLD (sit near the null while undecided) and then
COLLAPSE into the perpendicular, or does it slide straight to the answer?

The metrics can't tell these apart -- same endpoint, same accuracy. Only the path can.
For a few BALANCED episodes we print, per step:
    |z|   : distance from the null (0 = holding/emitting nothing, 1 = committed)
    speed : how fast it's moving (the discomfort readout)
    off   : current angle from the bisector (-> ~90 means it has reached the synthesis)

Holding-then-collapse looks like: |z| stays low for several steps (a plateau near the
null), then rises to ~1 while off swings to ~90. A disguised feedforward map looks
like: |z| rises immediately from step 0 with no plateau.

Run:  python3 synth_trace.py structured
      python3 synth_trace.py learned
"""
import sys
import math
import torch
from synth_task import sample_episode, angle_deg
from synth_models import build

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 3000
BATCH = 512
LR = 5e-3
SETTLE_W = 0.1


def train(name, dev):
    op = build(name).to(dev)
    opt = torch.optim.Adam(op.parameters(), lr=LR)
    for _ in range(STEPS):
        obs, target, reg, *_ = sample_episode(BATCH, dev)
        ans, speeds = op.rollout(obs)
        loss = (1.0 - (ans * target).sum(-1)).mean() + SETTLE_W * speeds[-1].mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return op


@torch.no_grad()
def trace(op, dev, n=4):
    """Re-run the rollout step by step, recording |z|, speed, off-axis angle."""
    obs, target, reg, uA, uB, mean, perp = sample_episode(n, dev, regime=2)

    # generic per-step replay: works for any model by reading its internal answer point.
    # We reconstruct the answer-point path by calling rollout but capturing speeds, and
    # recomputing the point via a hook-free re-run for the two field models.
    T = obs.shape[0]

    # Re-run structured/learned/interp dynamics here mirroring synth_models, capturing z_t.
    import torch.nn.functional as Fn
    name = type(op).__name__

    paths = []  # list over episodes of list of (|z|, speed, off)
    for i in range(n):
        o = obs[:, i:i+1, :]
        zs = _rollout_capture(op, o)               # (T,2)
        m_dir = mean[i:i+1]
        row = []
        prev = torch.zeros(1, 2, device=dev)
        for t in range(T):
            zt = zs[t:t+1]
            sp = (zt - prev).norm(dim=-1).item()
            nz = zt.norm(dim=-1).item()
            off = angle_deg(zt / (zt.norm(dim=-1, keepdim=True) + 1e-8), m_dir).item()
            row.append((nz, sp, off))
            prev = zt
        paths.append(row)

    print(f"\n=== {name}: per-step trajectory on balanced episodes ===")
    print("(|z| 0=holding 1=committed | speed=discomfort | off=angle from bisector, 90=synthesis)\n")
    for i, row in enumerate(paths):
        print(f"episode {i}:")
        print("  step:  " + " ".join(f"{t:5d}" for t in range(T)))
        print("  |z|:   " + " ".join(f"{r[0]:5.2f}" for r in row))
        print("  speed: " + " ".join(f"{r[1]:5.2f}" for r in row))
        print("  off:   " + " ".join(f"{r[2]:5.0f}" for r in row))
        print()


@torch.no_grad()
def _rollout_capture(op, obs):
    """Return the (T,2) path of the model's answer point for a single episode."""
    import torch.nn.functional as Fn
    T, B, _ = obs.shape
    dev = obs.device
    name = type(op).__name__
    uA = obs[0, :, 0:2]; uB = obs[0, :, 2:4]

    def unit(v): return v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    def R90(v): return torch.stack([-v[..., 1], v[..., 0]], dim=-1)

    if name == "StructuredField":
        sp = Fn.softplus
        mean = unit(uA + uB); perp = R90(mean)
        z = torch.zeros(B, 2, device=dev); m = torch.zeros(B, device=dev)
        out = []
        for t in range(T):
            m = m + sp(op.w_v) * obs[t, :, 4]
            s = torch.exp(-(m / (sp(op.tau) + 1e-6)) ** 2)
            gate = (1 - s).unsqueeze(-1)
            sel = torch.sigmoid(sp(op.k) * m).unsqueeze(-1)
            pole = sel * uA + (1 - sel) * uB
            conf = (1 - (z * z).sum(-1, keepdim=True)) * z
            force = sp(op.alpha) * gate * (pole - z) + sp(op.beta) * s.unsqueeze(-1) * perp + sp(op.gamma) * conf
            z = z + 0.25 * torch.sigmoid(op.eta) * torch.tanh(force)
            out.append(z.squeeze(0))
        return torch.stack(out)

    if name == "LearnedField":
        h = torch.zeros(B, op.hidden, device=dev); out = []
        for t in range(T):
            e = torch.relu(op.enc(obs[t]))
            dh = op.f2(torch.tanh(op.f1(torch.cat([h, e], dim=-1))))
            h = h + Fn.softplus(op.eta) * torch.tanh(dh)
            out.append(op.read(h).squeeze(0))
        return torch.stack(out)

    # InterpLeaner
    sp = Fn.softplus; m = torch.zeros(B, device=dev); out = []
    for t in range(T):
        m = m + sp(op.w_v) * obs[t, :, 4]
        sel = torch.sigmoid(sp(op.k) * m).unsqueeze(-1)
        out.append((sel * uA + (1 - sel) * uB).squeeze(0))
    return torch.stack(out)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "learned"
    dev = torch.device(DEVICE)
    op = train(name, dev)
    trace(op, dev)


if __name__ == "__main__":
    main()
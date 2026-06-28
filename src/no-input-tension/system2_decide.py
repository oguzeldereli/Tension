"""
SYSTEM 2 -- the atom: a single decision under a thinking-cost.

The model's own weights descend the tension loss. Every step it stays indecisive,
the 'thinking time' t accumulates (in proportion to how torn it is). That rising t
lifts the balanced valley until, at a predictable critical time, balance becomes a
hilltop and the system snaps to a commitment -- which side it picks is decided by
the tiniest asymmetry (spontaneous symmetry breaking).

No data. No input. Just weights rolling on a landscape that tilts because of how
long they've hesitated. Run it and watch the snap.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import config as C
from tension import TensionModel, tension_loss, indecision, base_tension


def run():
    torch.manual_seed(C.SEED)
    os.makedirs(C.FIG_DIR, exist_ok=True)

    model = TensionModel(C.LATENT_DIM, C.HIDDEN)
    opt = torch.optim.SGD(model.parameters(), lr=C.LR)

    t = 0.0
    ps, ts = [], []
    for step in range(C.STEPS_DECIDE):
        p = model.p()
        loss = tension_loss(p, t, C.PENALTY_STRENGTH)   # t is a constant coefficient here
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            # tiny symmetry-breaking noise on every weight
            for prm in model.parameters():
                prm.add_(torch.randn_like(prm) * C.NOISE)
            indec = indecision(p).item()

        # thinking time: accrues while indecisive (indec ~ 1), leaks away while committed
        t = t * C.DECAY + C.DT * indec
        ps.append(p.item())
        ts.append(t)

    # locate the snap: first step the decision leaves the indecision band
    snap = next((i for i, v in enumerate(ps) if abs(v - 0.5) > 0.3), None)
    print(f"predicted bifurcation at  t = 2*pi^2 / s = {C.T_CRIT:.2f}")
    if snap is not None:
        side = "output 1" if ps[-1] > 0.5 else "output 2"
        print(f"observed snap at step {snap},  t there = {ts[snap]:.2f},  committed to {side}")
    else:
        print("no snap within STEPS_DECIDE -- raise steps or DT, or lower DECAY")

    # ---- plots ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].plot(ps, lw=1.2)
    ax[0].axhline(0.5, ls="--", c="gray", lw=0.8)
    if snap is not None:
        ax[0].axvline(snap, c="crimson", ls=":", label="snap")
        ax[0].legend()
    ax[0].set_ylim(-0.02, 1.02)
    ax[0].set_title("decision p over thinking steps")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("p(output 1)")

    ax[1].plot(ts, lw=1.2)
    ax[1].axhline(C.T_CRIT, c="crimson", ls="--", lw=1, label="bifurcation t")
    ax[1].set_title("accumulated thinking-time t")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("t"); ax[1].legend()

    pp = torch.linspace(0, 1, 400)
    for frac, lbl in [(0.0, "t=0 (fresh)"), (0.5, "t=T_crit/2"),
                      (1.0, "t=T_crit (flat)"), (1.6, "t>T_crit (hilltop)")]:
        tv = frac * C.T_CRIT
        L = base_tension(pp) + tv * C.PENALTY_STRENGTH * (1 - (2 * pp - 1) ** 2)
        ax[2].plot(pp.numpy(), L.numpy(), label=lbl)
    ax[2].set_title("the landscape tilting as t grows")
    ax[2].set_xlabel("p"); ax[2].set_ylabel("tension"); ax[2].legend(fontsize=8)

    plt.tight_layout()
    out = f"{C.FIG_DIR}/system2_decide.png"
    plt.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    run()

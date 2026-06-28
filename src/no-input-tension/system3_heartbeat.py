"""
SYSTEM 3 -- the heartbeat: ongoing autonomous life.

System 2 deliberates once and then freezes in a committed valley. To get *ongoing*
dynamics, we add a cyclic perturbation: every PERTURB_PERIOD steps the system is
'kicked' back toward the balanced state (its output weights are shrunk toward zero,
plus noise), and its thinking-time resets. So it deliberates afresh, commits, gets
knocked back, deliberates again -- a rhythm of decisions.

Still inputless: the only thing entering is the periodic self-perturbation (the
'heartbeat'). Each beat can break either way, because the kick's noise reseeds the
symmetry. Run it and watch repeated deliberate -> snap -> reset cycles.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import config as C
from tension import TensionModel, tension_loss, indecision


def run():
    torch.manual_seed(C.SEED)
    os.makedirs(C.FIG_DIR, exist_ok=True)

    model = TensionModel(C.LATENT_DIM, C.HIDDEN)
    opt = torch.optim.SGD(model.parameters(), lr=C.LR)

    t = 0.0
    ps, ts, kicks = [], [], []
    for step in range(C.STEPS_HEARTBEAT):
        # the heartbeat: knock it back toward indecision and reset thinking-time
        if step > 0 and step % C.PERTURB_PERIOD == 0:
            with torch.no_grad():
                model.fc2.weight.mul_(C.KICK_SCALE)        # logits -> ~equal -> p -> ~0.5
                model.fc2.bias.mul_(C.KICK_SCALE)
                for prm in model.parameters():
                    prm.add_(torch.randn_like(prm) * C.KICK_NOISE)
            t = 0.0
            kicks.append(step)

        p = model.p()
        loss = tension_loss(p, t, C.PENALTY_STRENGTH)
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            for prm in model.parameters():
                prm.add_(torch.randn_like(prm) * C.NOISE)
            indec = indecision(p).item()

        t = t * C.HEART_DECAY + C.HEART_DT * indec
        ps.append(p.item())
        ts.append(t)

    n_commits = sum(1 for k in kicks)
    print(f"ran {C.STEPS_HEARTBEAT} steps, {len(kicks)} heartbeats")
    print("each beat: deliberate near 0.5, then snap to a side; the side can differ per beat")

    # ---- plots ----
    fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

    ax[0].plot(ps, lw=1.0)
    ax[0].axhline(0.5, ls="--", c="gray", lw=0.8)
    for k in kicks:
        ax[0].axvline(k, c="crimson", ls=":", lw=0.8)
    ax[0].set_ylim(-0.02, 1.02)
    ax[0].set_title("decision p -- a rhythm of deliberate -> commit -> reset (red = heartbeat)")
    ax[0].set_ylabel("p(output 1)")

    ax[1].plot(ts, lw=1.0, c="darkorange")
    ax[1].axhline(C.T_CRIT, c="crimson", ls="--", lw=1, label="bifurcation t")
    ax[1].set_title("thinking-time t: ramps each beat, forces a commit at the line, resets on the kick")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("t"); ax[1].legend()

    plt.tight_layout()
    out = f"{C.FIG_DIR}/system3_heartbeat.png"
    plt.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    run()

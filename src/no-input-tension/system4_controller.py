"""
SYSTEM 4 -- one model changes another's weights.

Two networks:
  * TARGET     -- a TensionModel that self-descends its tension loss (it 'wants' to
                  resolve into a decision, exactly like System 2).
  * CONTROLLER -- a small network that READS the target's state [p, t, indecision]
                  and WRITES a delta into the target's output weights every step.

This is the literal "one model changes the weights of the other" you asked for --
a hypernetwork / fast-weights setup. The controller's output is bounded (tanh), so
the coupled system stays stable, and the two drives -- the target trying to commit,
the controller perturbing it -- produce ongoing autonomous dynamics with no input
and no data.

NOTE ON THE OBJECTIVE (honest):
Here the controller is FROZEN at random init. That already demonstrates the
mechanism and gives a self-contained autonomous dynamical system (different SEEDs
give fixed points, cycles, or wandering). The deeper version -- *training* the
controller on an intrinsic objective (e.g. 'keep the target at the edge of
decision', reward novelty / avoid freezing) -- is the real research extension and
is sketched in the README. That needs a differentiable inner loop and is its own
project; this file proves the wiring and lets you watch the coupled behavior.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import config as C
from tension import TensionModel, tension_loss, indecision


class Controller(nn.Module):
    """Reads target state -> writes a bounded delta for the target's fc2 weights."""

    def __init__(self, target_fc2_numel, hidden):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden)                 # input: [p, t_norm, indecision]
        self.fc2 = nn.Linear(hidden, target_fc2_numel)

    def forward(self, state):
        h = torch.tanh(self.fc1(state))
        return torch.tanh(self.fc2(h))                  # delta in [-1, 1] per weight


def run():
    torch.manual_seed(C.SEED)
    os.makedirs(C.FIG_DIR, exist_ok=True)

    target = TensionModel(C.LATENT_DIM, C.HIDDEN)
    t_opt = torch.optim.SGD(target.parameters(), lr=C.LR)

    fc2_numel = target.fc2.weight.numel() + target.fc2.bias.numel()
    controller = Controller(fc2_numel, C.CTRL_HIDDEN)
    for prm in controller.parameters():
        prm.requires_grad_(False)                       # frozen: we run it, we don't train it

    t = 0.0
    ps, ts = [], []
    w_n = target.fc2.weight.numel()
    for step in range(C.STEPS_CONTROLLER):
        p = target.p()

        # --- target's own drive: descend its tension loss ---
        loss = tension_loss(p, t, C.PENALTY_STRENGTH)
        t_opt.zero_grad()
        loss.backward()
        t_opt.step()

        # --- controller reads the target's state and writes its weights ---
        with torch.no_grad():
            indec = indecision(p)
            state = torch.tensor([p.item(), t / C.T_CRIT, indec.item()])
            delta = controller(state) * C.CTRL_DELTA_SCALE
            target.fc2.weight.add_(delta[:w_n].view_as(target.fc2.weight))
            target.fc2.bias.add_(delta[w_n:].view_as(target.fc2.bias))

        t = t * C.HEART_DECAY + C.HEART_DT * indec.item()
        ps.append(p.item())
        ts.append(t)

    print(f"ran {C.STEPS_CONTROLLER} steps of the coupled (controller -> target) system")
    print("the controller (a separate network) wrote the target's weights every step")
    print("try different SEED values in config.py: you'll see fixed points, cycles, or wandering")

    # ---- plots ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))

    ax[0].plot(ps, lw=1.0)
    ax[0].axhline(0.5, ls="--", c="gray", lw=0.8)
    ax[0].set_ylim(-0.02, 1.02)
    ax[0].set_title("target decision p, driven by the controller writing its weights")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("p(output 1)")

    # phase portrait: the autonomous trajectory in (p, indecision-pressure) space
    pp = torch.tensor(ps)
    indec_series = 1 - (2 * pp - 1) ** 2
    ax[1].plot(pp.numpy(), indec_series.numpy(), lw=0.5, alpha=0.7)
    ax[1].scatter([ps[0]], [(1 - (2 * ps[0] - 1) ** 2)], c="green", s=40, label="start", zorder=5)
    ax[1].scatter([ps[-1]], [(1 - (2 * ps[-1] - 1) ** 2)], c="crimson", s=40, label="end", zorder=5)
    ax[1].set_title("phase portrait of the coupled dynamics")
    ax[1].set_xlabel("p"); ax[1].set_ylabel("indecision"); ax[1].legend()

    plt.tight_layout()
    out = f"{C.FIG_DIR}/system4_controller.png"
    plt.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    run()

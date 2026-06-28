"""
The core: an inputless model that produces a single decision p = P(output 1),
and a tension loss with three valleys (commit-to-1, commit-to-2, stay-balanced)
that *tilts* as 'thinking time' t accumulates.

Nothing here takes any input. The model's output is a pure function of its own
weights. "Inference" is just letting those weights roll downhill on the tension
landscape.
"""
import math
import torch
import torch.nn as nn


def base_tension(p):
    """Three valleys at p = 0, 0.5, 1 ; two peaks at p = 0.25, 0.75.

    -cos(4*pi*p):  p=0 -> -1,  p=0.25 -> +1,  p=0.5 -> -1,  p=0.75 -> +1,  p=1 -> -1.
    So 'committed to 1', 'committed to 2', and 'perfectly balanced' are all minima;
    everything in between costs more. This is the static 'tension'.
    """
    return -torch.cos(4 * math.pi * p)


def indecision(p):
    """1 at p=0.5 (maximally torn), 0 at p=0 or p=1 (fully committed)."""
    return 1 - (2 * p - 1) ** 2


def tension_loss(p, t, strength):
    """Full tension at the current 'thinking time' t.

    t is a plain float (the accumulated cost of having stayed indecisive). It acts
    as a *coefficient*, not something we differentiate through. As t grows, the
    penalty `t * indecision(p)` lifts the middle of the landscape: the balanced
    valley gets shallower and shallower, while the committed valleys (where
    indecision ~ 0) are untouched -- so commitment becomes relatively deeper.

    Past a critical t the balanced minimum flips into a *hilltop* (a bifurcation),
    and the system is forced to fall to one side.
    """
    return base_tension(p) + (t * strength) * indecision(p)


class TensionModel(nn.Module):
    """Inputless. Internal weights -> 2 logits. The 'decision' is p = P(output 1).

    Initialized to START balanced (logits ~ 0 -> p ~ 0.5), i.e. maximally
    indecisive, so we can watch it deliberate from the edge.
    """

    def __init__(self, latent_dim, hidden):
        super().__init__()
        self.latent = nn.Parameter(torch.randn(latent_dim) * 0.1)
        self.fc1 = nn.Linear(latent_dim, hidden)
        self.fc2 = nn.Linear(hidden, 2)
        # start near the balanced state: shrink the output layer so logits ~ 0
        with torch.no_grad():
            self.fc2.weight.mul_(0.05)
            self.fc2.bias.zero_()

    def forward(self):
        h = torch.tanh(self.fc1(self.latent))
        return self.fc2(h)                                  # (2,) logits

    def p(self):
        return torch.softmax(self.forward(), dim=-1)[0]     # scalar prob of output 1

"""
The policy: a runaway accumulator.

A single decision variable d evolves by ADDING a per-turn increment:
    d_{t+1} = d_t + g_t
p = sigmoid(d) is the decision (P heads). d=0 -> p=0.5 = indecision/waiting.
Because increments accumulate, under steady evidence d keeps growing and p runs
away to a corner -- a genuine collapse, not a single sampled bet. (A contractive
cell like a vanilla GRU would settle to a fixed point and never run away; an
integrator is what makes the runaway real.)

The network outputs only the *mean* increment mu given the current evidence and d.
The actual increment is sampled g ~ Normal(mu, SIGMA): the increments are the
stochastic ACTIONS for policy-gradient (REINFORCE), so we never have to backprop
through the unrolled dynamics -- we just reinforce the increments that led to good bets.

Note: p = sigmoid(d) is exactly softmax over two logits whose difference is d, so this
is the two-logit / "wait = balanced" mechanism, with d the logit difference.
"""
import torch
import torch.nn as nn
import config as C

class AccumulatorPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # Pristine, unconstrained multi-layer perceptron
        self.net = nn.Sequential(
            nn.Linear(3, C.HIDDEN), nn.Tanh(),
            nn.Linear(C.HIDDEN, C.HIDDEN), nn.Tanh(),
            nn.Linear(C.HIDDEN, 2),        # Back to 2 explicit outputs
            nn.Softmax(dim=-1)             # Smooth, normalized probability mapping
        )

    def forward(self, state):
        """state: [B, 3] -> Returns mu: [B, 2] where column 0 is p."""
        return self.net(state)
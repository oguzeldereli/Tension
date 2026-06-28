"""
TensionOperator -- a deliberation operator that breaks "one forward pass = one output".

It carries an internal latent state z across forward passes. Each pass it integrates the
current evidence into z (this is the deliberation -- it refines internally, emitting
nothing). While the tension is unresolved it outputs the ZERO VECTOR: not a symbol, not
a confidence -- literally nothing, so a downstream MLP receives no signal yet. When the
latent SETTLES (the competing pulls reconcile into a stable point), a latch flips and the
operator SNAPS to a clean one-hot symbol decoded from z -- a synthesis that may be neither
of the opposites it was holding.

Key properties:
  * Holding = zero output. Deliberation happens in z across passes, invisible downstream.
  * The snap is discrete (one-hot via straight-through); holding is the literal 0 vector.
  * No noise: evidence + weights decide the symbol; settledness decides the moment.
  * Backprop trains the continuous deliberation (z dynamics) and the decoder; the latch
    reads off settledness. The clock just tells it time is passing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def hard_step(x):
    """Forward: 1 if x>0 else 0. Backward: sigmoid gradient (straight-through)."""
    hard = (x > 0).float()
    soft = torch.sigmoid(4.0 * x)
    return hard + soft - soft.detach()


def st_one_hot(logits):
    """Straight-through one-hot of argmax: digital forward, soft gradient backward."""
    soft = F.softmax(logits, dim=-1)
    idx = logits.argmax(-1)
    hard = F.one_hot(idx, logits.shape[-1]).float()
    return hard + soft - soft.detach()


class TensionOperator(nn.Module):
    def __init__(self, in_dim, n_symbols, latent_dim=32, hidden=64,
                 eta=0.3, settle_scale=0.02, settle_thresh=0.95):
        super().__init__()
        self.drive = nn.Sequential(                       # how z updates given evidence + current z
            nn.Linear(in_dim + latent_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Linear(latent_dim, n_symbols)   # z -> symbol logits (read only when settled)
        self.latent_dim = latent_dim
        self.n_symbols = n_symbols
        self.eta = eta
        self.settle_scale = settle_scale
        self.settle_thresh = settle_thresh

    def init_state(self, B, device):
        z = torch.zeros(B, self.latent_dim, device=device)
        latch = torch.zeros(B, device=device)
        return z, latch

    def forward(self, x, clock, z, latch):
        if not torch.is_tensor(clock):
            clock = torch.full((x.shape[0], 1), float(clock), device=x.device)
        elif clock.dim() == 1:
            clock = clock.unsqueeze(-1)

        dz = torch.tanh(self.drive(torch.cat([x, z], dim=-1)))   # the pull from evidence
        z_new = z + self.eta * dz                                # integrate (deliberate)

        # settledness: how stationary z is now. competing pulls reconciled -> z stops moving -> settled
        speed = (z_new - z).pow(2).sum(-1)
        settled = torch.exp(-speed / self.settle_scale)          # ~1 when settled, ~0 when still moving

        # latch: fires (once, monotonic) when settled crosses threshold OR clock hits the deadline
        fire = hard_step(settled - self.settle_thresh)
        deadline = hard_step(clock.squeeze(-1) - 0.999)
        new_latch = torch.maximum(latch, torch.maximum(fire, deadline))

        logits = self.decoder(z_new)
        symbol = st_one_hot(logits)
        output = new_latch.unsqueeze(-1) * symbol                # ZERO while holding; symbol when committed

        return dict(output=output, z=z_new, latch=new_latch,
                    logits=logits, settled=settled)

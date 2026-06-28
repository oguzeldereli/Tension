"""
Synthesis task: the "third thing".

A G x G grid. The hidden answer is a CELL (r, c), encoded as index r*G + c.

Two streams give noisy evidence each step, but neither is about a row or a column:
    stream A : noisy evidence about  s = r + c        (an anti-diagonal of the grid)
    stream B : noisy evidence about  d = r - c        (a diagonal of the grid)

A single stream pins the answer down only to a diagonal line of G candidates.
The cell exists only at the INTERSECTION of the two diagonals -- so the operator
must hold both partial constraints in its latent and synthesize a point that is
on neither input's axis. That point (the cell) is the third thing.

Each step's hint is correct with prob P_SIGNAL, else uniform noise, so one pass is
ambiguous and integrating across passes (deliberating) is what sharpens it.
"""
import torch
import torch.nn.functional as F
from config import GRID, P_SIGNAL, MAX_STEPS, DEVICE

S_CARD = 2 * GRID - 1   # s = r + c          in [0, 2G-2]
D_CARD = 2 * GRID - 1   # d = (r - c)+(G-1)  in [0, 2G-2]
OBS_DIM = S_CARD + D_CARD
N_SYM = GRID * GRID


def sample_episode(batch, device=DEVICE, steps=MAX_STEPS, p_signal=P_SIGNAL):
    """Returns obs (steps, batch, OBS_DIM) and y (batch,) cell indices."""
    G = GRID
    r = torch.randint(0, G, (batch,), device=device)
    c = torch.randint(0, G, (batch,), device=device)
    y = r * G + c                       # the cell (third thing)

    s_true = (r + c)                    # in [0, 2G-2]
    d_true = (r - c) + (G - 1)          # shift to [0, 2G-2]

    rand_s = torch.randint(0, S_CARD, (steps, batch), device=device)
    rand_d = torch.randint(0, D_CARD, (steps, batch), device=device)
    keep_s = torch.rand(steps, batch, device=device) < p_signal
    keep_d = torch.rand(steps, batch, device=device) < p_signal

    obs_s = torch.where(keep_s, s_true.expand(steps, -1), rand_s)
    obs_d = torch.where(keep_d, d_true.expand(steps, -1), rand_d)

    oh_s = F.one_hot(obs_s, S_CARD).float()   # (steps, batch, S_CARD)
    oh_d = F.one_hot(obs_d, D_CARD).float()   # (steps, batch, D_CARD)
    obs = torch.cat([oh_s, oh_d], dim=-1)     # (steps, batch, OBS_DIM)
    return obs, y

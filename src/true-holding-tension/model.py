"""
TensionOperator.

A small recurrent operator that carries an internal latent z (the held tension),
evolves it across forward passes under the incoming evidence + a clock, and at each
step exposes:
    lam   : commit probability  (the latch -- "have I resolved?")
    logits: a symbol distribution decoded from z (the synthesized cell)

While holding it emits the ZERO VECTOR (a literal null, not a softmax sitting at
0.5). It emits a symbol only when the latch fires. The recurrent core is a GRUCell
for this first build; it can later be swapped for explicit two-attractor dynamics
without changing the training objective or the latch interface.
"""
import torch
from torch import nn
from task import OBS_DIM, N_SYM
from config import HIDDEN, INPUT_DIM, TAU


class TensionOperator(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_sym=N_SYM, hidden=HIDDEN, input_dim=INPUT_DIM):
        super().__init__()
        self.hidden = hidden
        self.n_sym = n_sym
        self.in_proj = nn.Linear(obs_dim + 1, input_dim)   # +1 for the clock
        self.cell = nn.GRUCell(input_dim, hidden)
        self.commit = nn.Linear(hidden, 1)                 # latch logit  ->  lam
        nn.init.constant_(self.commit.bias, -2.0)          # start by holding (lam ~ 0.12)
        self.decode = nn.Linear(hidden, n_sym)             # the synthesized symbol

    def init_state(self, batch, device):
        return torch.zeros(batch, self.hidden, device=device)

    def step(self, h, obs_t, clock_t):
        """One forward pass. obs_t (B,obs_dim), clock_t (B,1)."""
        inp = torch.relu(self.in_proj(torch.cat([obs_t, clock_t], dim=-1)))
        h = self.cell(inp, h)
        lam = torch.sigmoid(self.commit(h)).squeeze(-1)    # (B,)
        logits = self.decode(h)                            # (B, n_sym)
        return h, lam, logits


@torch.no_grad()
def hard_rollout(op, obs, tau=TAU):
    """
    Inference behaviour: HARD latch. Hold (emit zero) until commit prob crosses tau,
    then emit argmax(decode) once and freeze. No runway in the emitted signal -- the
    output is literally zero, then a single symbol.

    Returns:
        committed (B,) bool   -- did it resolve at all (else: silent / held forever)
        pred      (B,) long   -- predicted cell (-1 if never committed)
        cstep     (B,) long   -- step it committed at  (-1 if never)
    """
    T, B, _ = obs.shape
    device = obs.device
    h = op.init_state(B, device)
    committed = torch.zeros(B, dtype=torch.bool, device=device)
    pred = torch.full((B,), -1, dtype=torch.long, device=device)
    cstep = torch.full((B,), -1, dtype=torch.long, device=device)
    for t in range(T):
        clock = torch.full((B, 1), t / (T - 1), device=device)
        h, lam, logits = op.step(h, obs[t], clock)
        fire = (~committed) & (lam >= tau)
        if fire.any():
            pred[fire] = logits.argmax(-1)[fire]
            cstep[fire] = t
            committed |= fire
    return committed, pred, cstep
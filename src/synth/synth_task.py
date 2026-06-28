"""
Angular synthesis task -- the test bed for "holding resolves into a third thing".

Two poles A, B (unit vectors, given every step). Votes arrive over time:
  - decisive_A : net votes strongly +  -> answer is pole A     (ON the A-B arc)
  - decisive_B : net votes strongly -  -> answer is pole B     (ON the A-B arc)
  - balanced   : votes cancel / are sparse (net ~0) -> answer is PERP(bisector),
                 the synthesis, OFF the arc, orthogonal to the opposition.

Why orthogonal: with no noise, a symmetric (balanced) tension cannot resolve ALONG
the A-B axis -- there's no tie-breaker -- so deterministic resolution is only possible
perpendicular to it. The third thing literally cannot live between the poles.

Test logic:
  - An interpolator ("leaning") can only land between the poles, so at balance it sits
    at the bisector -> ~90 deg wrong. Decisive cases it gets right.
  - A synthesis mechanism squeezes off-axis to the perp -> right at balance too.
  - Determinism: same poles + different cancelling vote patterns must give the SAME
    perp answer (it depends on poles+balance, not on the vote details).

obs per step (6 dims): [uA(2), uB(2), vote(1), end(1)].
"""
import math
import torch

T_VOTE = 6
T_RESP = 6
MAX_STEPS = T_VOTE + T_RESP
OBS_DIM = 6
DMIN, DMAX = 40.0, 140.0     # pole separation range (deg); <180 so never antipodal


def _R90(v):                 # +90 deg rotation: (x,y) -> (-y,x)
    return torch.stack([-v[..., 1], v[..., 0]], dim=-1)


def _unit(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def sample_episode(batch, device, regime=None):
    """regime: None -> uniform over {0:dec_A, 1:dec_B, 2:balanced}; or a fixed int."""
    thA = torch.rand(batch, device=device) * 2 * math.pi
    sep = (torch.rand(batch, device=device) * (DMAX - DMIN) + DMIN) * math.pi / 180.0
    sgn = torch.where(torch.rand(batch, device=device) < 0.5, 1.0, -1.0)
    thB = thA + sgn * sep
    uA = torch.stack([torch.cos(thA), torch.sin(thA)], dim=-1)
    uB = torch.stack([torch.cos(thB), torch.sin(thB)], dim=-1)
    mean = _unit(uA + uB)
    perp = _R90(mean)

    if regime is None:
        reg = torch.randint(0, 3, (batch,), device=device)
    else:
        reg = torch.full((batch,), regime, device=device, dtype=torch.long)

    # votes: (T_VOTE, batch) in {-1,0,+1}
    votes = torch.zeros(T_VOTE, batch, device=device)
    # decisive: all +1 (A) or all -1 (B)
    votes[:, reg == 0] = 1.0
    votes[:, reg == 1] = -1.0
    # balanced: a random even count of votes that exactly cancel (and sometimes zero)
    bmask = (reg == 2).nonzero(as_tuple=True)[0]
    for j in bmask.tolist():
        nv = int(torch.randint(0, T_VOTE // 2 + 1, (1,)).item()) * 2   # 0,2,4,6
        if nv > 0:
            pattern = torch.cat([torch.ones(nv // 2), -torch.ones(nv // 2)])
            pattern = pattern[torch.randperm(nv)]
            slots = torch.randperm(T_VOTE)[:nv]
            votes[slots, j] = pattern.to(device)

    target = torch.where((reg == 0).unsqueeze(-1), uA,
             torch.where((reg == 1).unsqueeze(-1), uB, perp))

    obs = torch.zeros(MAX_STEPS, batch, OBS_DIM, device=device)
    obs[:, :, 0:2] = uA.unsqueeze(0)
    obs[:, :, 2:4] = uB.unsqueeze(0)
    obs[:T_VOTE, :, 4] = votes
    obs[T_VOTE:, :, 5] = 1.0
    return obs, target, reg, uA, uB, mean, perp


def angle_deg(u, v):
    """angle between unit vectors, in degrees (batch,)."""
    d = (u * v).sum(-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(d))

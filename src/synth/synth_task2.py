"""
Angular synthesis, v2 -- the two scaffolds removed.

What changed vs synth_task:
  (1) NO end flag. The observation no longer contains a "time's up" bit. Nothing in
      the input tells the operator when to commit. Release has to come from inside the
      dynamics (the field coming to rest), not from a clock the task hands over.
  (2) STAGGERED pole arrival. Pole A is present from step 0; pole B arrives at a random
      later step tB (the obs slots for B are ZERO until then). Before B arrives the
      perpendicular/synthesis direction literally cannot be computed -- both poles are
      required to define it. So early holding is forced by genuine information-
      insufficiency, not merely by regime ambiguity. The answer is unknowable early.
  (3) UNCERTAINTY mode (noise>0). Each step's vote is corrupted by Gaussian noise, so
      the regime (decisive vs balanced) is only recoverable by integrating evidence over
      time. The operator must accumulate until its running read stabilizes, then rest.

obs per step (5 dims): [uA(2), uB(2), vote(1)].   <- no end bit.

Info-sufficiency time t* (per episode) = max(tB, last_vote_step): the first step at which
both poles are present AND all votes have landed. Before t*, the correct behaviour is to
HOLD at the null. The intrinsic-timing claim is that the field's commit step tracks t*
(specifically tB when tB is the binding constraint) -- timing owned by the dynamics
responding to information, not by any provided signal.
"""
import math
import torch

T = 24                       # total steps (room for variable, late commits to settle)
T_VOTE = 8                   # votes land on steps 0..T_VOTE-1
TB_MIN, TB_MAX = 2, 12       # pole B arrives at a random step in [TB_MIN, TB_MAX]
OBS_DIM = 5
DMIN, DMAX = 40.0, 140.0     # pole separation range (deg); <180 so never antipodal


def _R90(v):                 # +90 deg rotation: (x,y) -> (-y,x)
    return torch.stack([-v[..., 1], v[..., 0]], dim=-1)


def _unit(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def sample_episode(batch, device, regime=None, noise=0.0, tB_fixed=None):
    """regime: None -> uniform over {0:dec_A, 1:dec_B, 2:balanced}; or a fixed int.
    noise: std of per-step vote corruption (0 = clean).
    tB_fixed: force pole-B arrival step (else random per episode).
    Returns obs, target, reg, uA, uB, mean, perp, tB, tstar.
    """
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

    # ---- clean votes over [0, T_VOTE) ----
    votes = torch.zeros(T_VOTE, batch, device=device)
    votes[:, reg == 0] = 1.0
    votes[:, reg == 1] = -1.0
    bmask = (reg == 2).nonzero(as_tuple=True)[0]
    for j in bmask.tolist():
        nv = int(torch.randint(0, T_VOTE // 2 + 1, (1,)).item()) * 2   # 0,2,4,..
        if nv > 0:
            pattern = torch.cat([torch.ones(nv // 2), -torch.ones(nv // 2)])
            pattern = pattern[torch.randperm(nv)]
            slots = torch.randperm(T_VOTE)[:nv]
            votes[slots, j] = pattern.to(device)

    target = torch.where((reg == 0).unsqueeze(-1), uA,
             torch.where((reg == 1).unsqueeze(-1), uB, perp))

    # ---- staggered pole-B arrival ----
    if tB_fixed is None:
        tB = torch.randint(TB_MIN, TB_MAX + 1, (batch,), device=device)
    else:
        tB = torch.full((batch,), tB_fixed, device=device, dtype=torch.long)

    obs = torch.zeros(T, batch, OBS_DIM, device=device)
    obs[:, :, 0:2] = uA.unsqueeze(0)                         # A present throughout
    step_idx = torch.arange(T, device=device).unsqueeze(-1)  # (T,1)
    present_B = (step_idx >= tB.unsqueeze(0)).float()        # (T,batch)
    obs[:, :, 2:4] = uB.unsqueeze(0) * present_B.unsqueeze(-1)
    obs[:T_VOTE, :, 4] = votes
    if noise > 0:
        obs[:T_VOTE, :, 4] += noise * torch.randn(T_VOTE, batch, device=device)

    tstar = torch.clamp(tB, min=T_VOTE - 1)                  # info-sufficiency step
    return obs, target, reg, uA, uB, mean, perp, tB, tstar


def angle_deg(u, v):
    """angle between unit vectors, in degrees (batch,)."""
    d = (u * v).sum(-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(d))

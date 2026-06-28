"""
Operators for the v2 task (no end flag, staggered poles, optional noise).

All expose: rollout(obs) -> (z_path, speeds)
    z_path : (T,B,2) the answer point at every step. |z|~0 = holding at the null
             (emitting nothing); |z|~1 = a committed symbol.
    speeds : (T,B)   per-step movement = the discomfort readout.

Commit is NOT a step the task names. It is detected from the dynamics: the first step
where the field has reached a committed magnitude AND come to rest (speed<eps). See
detect_commit(). That is the whole point of v2 -- the operator owns its commit timing.

IntrinsicField -- structured force-balance, pole-presence gated. While pole B is absent
    the driving forces are switched OFF by the data itself (uB=0 -> both=0), so the only
    force is unit-confinement, which is zero at the null: the field SITS at the null with
    nothing to do. It cannot compute the synthesis it doesn't yet have the poles for.
    When B arrives the squeeze/pull engage; balanced -> off-axis perp, decisive -> pole;
    then it rests. Holding is forced by missing information; release is the field resting.

InterpLeaner2 -- evidence-weighted blend of the (currently visible) poles. Reference for
    "leaning": confined to the A-B arc, ~90 deg from the synthesis at balance. Also it has
    no notion of resting -- it just tracks the running blend.
"""
import torch
from torch import nn
import torch.nn.functional as F
from synth_task2 import OBS_DIM


def _R90(v):
    return torch.stack([-v[..., 1], v[..., 0]], dim=-1)


def _unit(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True) + eps)


class IntrinsicField(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_v   = nn.Parameter(torch.tensor(1.0))   # vote -> tilt gain
        self.k     = nn.Parameter(torch.tensor(1.0))   # tilt -> pole selection sharpness
        self.tau   = nn.Parameter(torch.tensor(1.0))   # balance width (squeeze)
        self.alpha = nn.Parameter(torch.tensor(1.0))   # pole-pull strength
        self.beta  = nn.Parameter(torch.tensor(1.0))   # perp-squeeze strength
        self.gamma = nn.Parameter(torch.tensor(1.0))   # unit-circle confinement
        self.eta   = nn.Parameter(torch.tensor(0.5))   # integration step

    def rollout(self, obs):
        T, B, _ = obs.shape
        sp = F.softplus
        z = torch.zeros(B, 2, device=obs.device)
        m = torch.zeros(B, device=obs.device)
        prev = z
        zs, speeds = [], []
        for t in range(T):
            uA = obs[t, :, 0:2]
            uB = obs[t, :, 2:4]
            present_A = (uA.norm(dim=-1) > 1e-6).float()
            present_B = (uB.norm(dim=-1) > 1e-6).float()
            both = (present_A * present_B).unsqueeze(-1)        # gate: need both poles
            mean = _unit(uA + uB)
            perp = _R90(mean)

            m = m + sp(self.w_v) * obs[t, :, 4]                 # integrate evidence
            s = torch.exp(-(m / (sp(self.tau) + 1e-6)) ** 2)    # squeeze: peaks at m=0
            gate = (1.0 - s).unsqueeze(-1)                      # pole pull vanishes at balance
            sel = torch.sigmoid(sp(self.k) * m).unsqueeze(-1)
            pole = sel * uA + (1.0 - sel) * uB
            conf = (1.0 - (z * z).sum(-1, keepdim=True)) * z
            drive = (sp(self.alpha) * gate * (pole - z)
                     + sp(self.beta) * s.unsqueeze(-1) * perp)
            force = both * drive + sp(self.gamma) * conf        # no poles -> no drive -> hold
            z = z + 0.25 * torch.sigmoid(self.eta) * torch.tanh(force)
            speeds.append((z - prev).norm(dim=-1))
            zs.append(z)
            prev = z
        return torch.stack(zs), torch.stack(speeds)


class InterpLeaner2(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_v = nn.Parameter(torch.tensor(1.0))
        self.k = nn.Parameter(torch.tensor(1.0))

    def rollout(self, obs):
        T, B, _ = obs.shape
        sp = F.softplus
        m = torch.zeros(B, device=obs.device)
        prev = torch.zeros(B, 2, device=obs.device)
        zs, speeds = [], []
        for t in range(T):
            uA = obs[t, :, 0:2]
            uB = obs[t, :, 2:4]
            m = m + sp(self.w_v) * obs[t, :, 4]
            sel = torch.sigmoid(sp(self.k) * m).unsqueeze(-1)
            ans = sel * uA + (1.0 - sel) * uB
            speeds.append((ans - prev).norm(dim=-1))
            zs.append(ans)
            prev = ans
        return torch.stack(zs), torch.stack(speeds)


def detect_commit(z_path, speeds, eps=0.02, mag=0.5):
    """First step where the field is committed (|z|>mag) AND at rest (speed<eps).
    Falls back to the last step if it never settles committed. Returns (idx (B,), z_commit (B,2))."""
    T, B, _ = z_path.shape
    mags = z_path.norm(dim=-1)                       # (T,B)
    committed = (mags > mag) & (speeds < eps)        # (T,B)
    any_c = committed.any(0)
    idx = torch.where(any_c, committed.float().argmax(0),
                      torch.full((B,), T - 1, device=z_path.device, dtype=torch.long))
    z_commit = z_path[idx, torch.arange(B, device=z_path.device)]
    return idx, z_commit


def build(name):
    return {"intrinsic": IntrinsicField, "interp": InterpLeaner2}[name]()

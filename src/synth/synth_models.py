"""
Three resolution mechanisms, one interface: rollout(obs) -> (answer_unit, speeds).

answer_unit : (B,2) the settled direction (the committed symbol; while holding, z is
              near 0 = the null/zero output).
speeds      : (T,B) per-step movement of the answer point = discomfort V_t, a READOUT
              of how unresolved it still is. Commit (at inference) = when speed falls
              below eps (the tension has dissolved). No external discomfort knob: the
              pressure is intrinsic to the dynamics.

StructuredField -- the force-balance is imposed; content is learned.
    pole pull (vanishes at balance) + perp squeeze (peaks at balance) + unit confinement.
    Balanced evidence cannot pull along the axis, so the squeeze drives z OFF-axis to
    the perpendicular synthesis. Decisive evidence pulls z to a pole.

LearnedField -- a freeform learned vector field on a hidden state (Euler residual
    dynamics). Same latch (settle) and same loss. We see whether it DISCOVERS the
    off-axis squeeze on its own, or collapses to interpolation.

InterpLeaner -- the "leaning" hypothesis: answer = evidence-weighted blend of the two
    poles. Structurally confined to the A-B arc; at balance it sits at the bisector,
    ~90 deg from the synthesis. Reference line, not a trained-to-fail strawman.
"""
import torch
from torch import nn
import torch.nn.functional as F
from synth_task import OBS_DIM


def _R90(v):
    return torch.stack([-v[..., 1], v[..., 0]], dim=-1)


def _unit(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True) + eps)


class StructuredField(nn.Module):
    def __init__(self):
        super().__init__()
        # all forced positive via softplus
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
        uA = obs[0, :, 0:2]
        uB = obs[0, :, 2:4]
        mean = _unit(uA + uB)
        perp = _R90(mean)

        z = torch.zeros(B, 2, device=obs.device)
        m = torch.zeros(B, device=obs.device)
        prev = z
        speeds = []
        for t in range(T):
            vote = obs[t, :, 4]
            m = m + sp(self.w_v) * vote
            s = torch.exp(-(m / (sp(self.tau) + 1e-6)) ** 2)        # squeeze: peaks at m=0
            gate = (1.0 - s).unsqueeze(-1)                          # pole pull vanishes at balance
            sel = torch.sigmoid(sp(self.k) * m).unsqueeze(-1)
            pole = sel * uA + (1.0 - sel) * uB                      # favored pole
            conf = (1.0 - (z * z).sum(-1, keepdim=True)) * z
            force = (sp(self.alpha) * gate * (pole - z)
                     + sp(self.beta) * s.unsqueeze(-1) * perp
                     + sp(self.gamma) * conf)
            # bounded Euler step: caps |dz| per step so the stiff confinement can't blow up
            z = z + 0.25 * torch.sigmoid(self.eta) * torch.tanh(force)
            speeds.append((z - prev).norm(dim=-1))
            prev = z
        return _unit(z), torch.stack(speeds)


class LearnedField(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.enc = nn.Linear(OBS_DIM, hidden)
        self.f1 = nn.Linear(2 * hidden, hidden)
        self.f2 = nn.Linear(hidden, hidden)
        self.read = nn.Linear(hidden, 2)
        self.eta = nn.Parameter(torch.tensor(0.5))
        self.hidden = hidden

    def rollout(self, obs):
        T, B, _ = obs.shape
        h = torch.zeros(B, self.hidden, device=obs.device)
        prev = torch.zeros(B, 2, device=obs.device)
        speeds = []
        for t in range(T):
            e = torch.relu(self.enc(obs[t]))
            dh = self.f2(torch.tanh(self.f1(torch.cat([h, e], dim=-1))))
            h = h + F.softplus(self.eta) * torch.tanh(dh)
            ans = self.read(h)
            speeds.append((ans - prev).norm(dim=-1))
            prev = ans
        return _unit(ans), torch.stack(speeds)


class InterpLeaner(nn.Module):
    """answer = evidence-weighted blend of poles. Cannot leave the A-B arc."""
    def __init__(self):
        super().__init__()
        self.w_v = nn.Parameter(torch.tensor(1.0))
        self.k = nn.Parameter(torch.tensor(1.0))

    def rollout(self, obs):
        T, B, _ = obs.shape
        sp = F.softplus
        uA = obs[0, :, 0:2]
        uB = obs[0, :, 2:4]
        m = torch.zeros(B, device=obs.device)
        prev = torch.zeros(B, 2, device=obs.device)
        speeds = []
        ans = torch.zeros(B, 2, device=obs.device)
        for t in range(T):
            m = m + sp(self.w_v) * obs[t, :, 4]
            sel = torch.sigmoid(sp(self.k) * m).unsqueeze(-1)
            ans = sel * uA + (1.0 - sel) * uB
            speeds.append((ans - prev).norm(dim=-1))
            prev = ans
        return _unit(ans), torch.stack(speeds)


def build(name):
    return {"structured": StructuredField,
            "learned": LearnedField,
            "interp": InterpLeaner}[name]()
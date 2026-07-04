"""
Tension-Synthesis OPERATOR -- the opposition-synthesis geometry put back inside the actual
tension contract that synth_opposites.py left out.

synth_opposites.py proved the GEOMETRY: opposing pole pairs -> a third thing orthogonal to the
opposition axes, with a real string-tension signal ||p_a - p_b|| (corr 0.999 with true tension).
But it was a single feed-forward that always emits a class. That is NOT the tension operator.
This file restores the three things that make it one:

  (1) HOLDING. While still deciding it emits the ZERO VECTOR (no committed answer). The answer
      appears once -- and only once -- synthesis has settled and the block commits.
  (2) DELIBERATION over many inner steps. No "one forward = one answer". The internal estimate
      of the third thing rolls toward the spring equilibrium across steps; the block decides
      ITSELF when the strings have settled enough to cut them (commit).
  (3) PERSISTENT STATE across separate inference runs. The block carries its internal state
      between calls (reset_state / step), so repeatedly invoking it CONTINUES the same
      deliberation rather than restarting -- a single call cannot produce the answer.

Commit ("cut the string") is a learned function of (a) external evidence -- the current poles
via the projected synthesis estimate -- and (b) intrinsic tension characteristics: the real
string tension ||p_a-p_b|| and how much the orthogonal synthesis is still moving (settledness
||Δz||). We also train a no-tension ablation of the halt head, so we can check -- in the style
of Experiment B -- whether THIS (structural) tension signal actually carries weight, unlike the
old inert ||Δh||.

Run:  python3 tension_synth_operator.py
"""
import os, sys
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_opposites import gen, D, K, N, C, TRAIN_TENSION  # reuse the opposition task + geometry

DEVICE = "cpu"          # tiny; the GPU hard-powers-off this machine under load
HID = 64
T_MAX = 12              # max inner deliberation steps per thinking episode
STEPS = 4000
BATCH = 256
LR = 2e-3
LAMBDA_C = 0.02         # compute penalty (annealed) -- one knob sweeps speed/accuracy
COMMIT_THRESH = 0.5     # cumulative halt mass at which the block commits / cuts the string
SEED = 0


def _project_perp(x, U):
    """Component of x orthogonal to the columns of U (the opposition subspace). This is the
    settled equilibrium of unit springs along the opposition axes -- the 'third thing'."""
    G = torch.bmm(U.transpose(1, 2), U)                              # (B,k,k)
    rhs = torch.bmm(U.transpose(1, 2), x.unsqueeze(-1))             # (B,k,1)
    coeff = torch.linalg.solve(G + 1e-4 * torch.eye(U.shape[-1], device=x.device), rhs)
    par = torch.bmm(U, coeff).squeeze(-1)
    return x - par


class TensionSynthOperator(nn.Module):
    """Holds, deliberates, commits when settled, and persists state across calls."""
    def __init__(self, use_tension=True):
        super().__init__()
        self.use_tension = use_tension
        self.enc = nn.Linear(D + 2, D)                              # [content, tmean, tmax] -> input
        self.cell = nn.GRUCell(D, D)                                # state z is the third-thing estimate
        halt_in = D + (3 if use_tension else 1)                    # [z, ||Δz||, (tmean, tmax)]
        self.halt = nn.Sequential(nn.Linear(halt_in, HID), nn.Tanh(), nn.Linear(HID, 1))
        self.read = nn.Sequential(nn.Linear(D, HID), nn.GELU(), nn.Linear(HID, C))
        self._z = None                                             # persistent state across calls

    # ---- the opposition geometry (external evidence -> tension + agreement) ----
    def geom(self, poles):
        p_a, p_b = poles[:, 0::2, :], poles[:, 1::2, :]            # (B,K,D)
        diff = p_a - p_b
        t = diff.norm(dim=-1)                                      # (B,K) real string tension
        u = diff / (t.unsqueeze(-1) + 1e-6)
        content = (0.5 * (p_a + p_b)).mean(1)                      # (B,D) aggregated agreement
        U = u.transpose(1, 2)                                      # (B,D,K) opposition axes as cols
        tmean, tmax = t.mean(1, keepdim=True), t.amax(1, keepdim=True)
        return content, U, tmean, tmax, t

    # ---- one settling step: roll z toward the orthogonal equilibrium, decide whether to commit ----
    def settle(self, poles, z):
        content, U, tmean, tmax, t = self.geom(poles)
        feat = torch.tanh(self.enc(torch.cat([content, tmean, tmax], -1)))
        z_new = self.cell(feat, z)
        z_new = _project_perp(z_new, U)                           # keep the estimate in U^perp
        s = (z_new - z).norm(dim=-1, keepdim=True)               # settledness: how much it moved
        if self.use_tension:
            h_in = torch.cat([z_new, s, tmean, tmax], -1)        # intrinsic tension feeds commit
        else:
            h_in = torch.cat([z_new, s], -1)                     # ablation: no tension signal
        lam = torch.sigmoid(self.halt(h_in)).squeeze(-1)        # halt prob (given not yet halted)
        return z_new, lam, self.read(z_new), t

    # ---- training: unroll the whole episode (PonderNet stopping distribution) ----
    def unroll(self, poles):
        B = poles.shape[0]
        z = torch.zeros(B, D, device=poles.device)
        lam_l, logit_l = [], []
        for _ in range(T_MAX):
            z, lam, logit, _ = self.settle(poles, z)
            lam_l.append(lam); logit_l.append(logit)
        lam = torch.stack(lam_l); lam = lam.clone(); lam[-1] = 1.0  # force commit by last step
        logits = torch.stack(logit_l)                             # (T,B,C)
        oneminus = (1 - lam).clamp(1e-6, 1.0)
        carry = torch.cat([torch.ones_like(lam[:1]), torch.cumprod(oneminus, 0)[:-1]], 0)
        return logits, lam * carry                                # logits, p_halt (T,B)

    # ---- persistent, holding inference across SEPARATE calls (no single forward = answer) ----
    def reset_state(self, batch, dev):
        self._z = torch.zeros(batch, D, device=dev)
        self._chalt = torch.zeros(batch, device=dev)
        self._carry = torch.ones(batch, device=dev)
        self._committed = torch.zeros(batch, dtype=torch.bool, device=dev)
        self._answer = torch.zeros(batch, C, device=dev)
        self._step = 0

    @torch.no_grad()
    def step(self, poles, thresh=COMMIT_THRESH):
        """Advance the SAME persistent deliberation by one step. Emit the ZERO vector while
        holding; latch and emit the third thing once committed. Returns (emitted, committed)."""
        z, lam, logit, _ = self.settle(poles, self._z)
        self._z = z                                              # state persists to the next call
        self._chalt = self._chalt + lam * self._carry
        self._carry = self._carry * (1 - lam)
        self._step += 1
        force = self._step >= T_MAX
        newly = (~self._committed) & ((self._chalt >= thresh) | force)
        self._answer[newly] = logit[newly]
        self._committed |= newly
        emitted = torch.where(self._committed.unsqueeze(-1), self._answer,
                              torch.zeros_like(self._answer))     # HOLD => zero vector
        return emitted, self._committed.clone()


def ponder_loss(logits, p_halt, y, lam_c):
    T, B, Cc = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, Cc), y.unsqueeze(0).expand(T, B).reshape(-1),
                         reduction="none").reshape(T, B)
    exp_ce = (p_halt * ce).sum(0).mean()
    steps = (torch.arange(T, device=logits.device).float() + 1).unsqueeze(-1)
    exp_steps = (p_halt * steps).sum(0).mean()
    return exp_ce + lam_c * exp_steps


def train(model, dev):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for it in range(STEPS):
        lam_c = LAMBDA_C * min(1.0, max(0.0, (it / STEPS - 0.4) / 0.3))   # anneal compute penalty
        poles, y, _ = gen(BATCH, TRAIN_TENSION, dev)
        logits, p_halt = model.unroll(poles)
        loss = ponder_loss(logits, p_halt, y, lam_c)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def evaluate(model, dev, n=4096):
    """Run the persistent/holding inference loop; measure accuracy at commit, commit timing,
    whether one forward can answer, and corr(commit step, tension)."""
    model.eval()
    poles, y, tau = gen (n, TRAIN_TENSION, dev)
    model.reset_state(n, dev)
    held_zero_at_1 = None
    commit_step = torch.zeros(n, device=dev)
    for k in range(T_MAX):
        emitted, committed = model.step(poles)
        if k == 0:
            held_zero_at_1 = (emitted.norm(dim=-1) == 0)          # holding => exact zero vector
        newly_done = committed & (commit_step == 0)
        commit_step[newly_done] = k + 1
    pred = emitted.argmax(-1)
    acc = (pred == y).float().mean().item() * 100
    frac_hold1 = held_zero_at_1.float().mean().item() * 100
    avg_step = commit_step.mean().item()
    tt = tau.mean(1)
    corr = (((commit_step - commit_step.mean()) * (tt - tt.mean())).mean()
            / (commit_step.std() * tt.std() + 1e-8)).item()
    return acc, frac_hold1, avg_step, corr


def main():
    dev = torch.device(DEVICE)
    torch.manual_seed(SEED)
    print(f"device {DEVICE}  D={D} pairs K={K} (N={N} poles) classes={C}  max inner steps={T_MAX}\n")

    print("training tension-synth operator (with intrinsic tension in the commit head)...")
    op = train(TensionSynthOperator(use_tension=True).to(dev), dev)
    print("training no-tension ablation (commit head cannot see string tension)...")
    abl = train(TensionSynthOperator(use_tension=False).to(dev), dev)

    acc, hold1, step, corr = evaluate(op, dev)
    accA, hold1A, stepA, corrA = evaluate(abl, dev)

    print("\n--- holding + deliberation + commit (persistent, multi-call inference) ---")
    print(f"{'operator':<24}{'acc@commit':>11}{'avg steps':>11}{'corr(step,tension)':>20}"
          f"{'% holding (0) @step1':>22}")
    print(f"{'tension-synth (ours)':<24}{acc:>11.2f}{step:>11.2f}{corr:>20.3f}{hold1:>22.1f}")
    print(f"{'no-tension ablation':<24}{accA:>11.2f}{stepA:>11.2f}{corrA:>20.3f}{hold1A:>22.1f}")

    # explicit demonstration that one feed-forward is NOT enough, and state persists across calls
    print("\n--- proof: no single feed-forward = answer; state carries across calls ---")
    torch.manual_seed(1)
    poles, y, tau = gen(6, TRAIN_TENSION, dev)
    op.reset_state(6, dev)
    for k in range(T_MAX):
        emitted, committed = op.step(poles)                       # SAME poles re-fed each call
        norms = emitted.norm(dim=-1)
        print(f"  call {k+1:2d}: emitted-norm per item = "
              f"[{' '.join(f'{v:4.1f}' for v in norms.tolist())}]  committed={committed.tolist()}")
        if committed.all():
            break
    print("  (norm 0.0 = still HOLDING / emitting the zero vector; nonzero = committed answer.")
    print("   A single call commits nothing for the harder items -- the operator must be run")
    print("   repeatedly, and its internal z persists between calls to make that progress.)")


if __name__ == "__main__":
    main()

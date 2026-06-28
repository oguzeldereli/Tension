"""
TensionBlock -- a composable deliberation operator (like a conv/attention block, but it
buys *time* instead of receptive field).

The contract that makes it composable AND faithful to the "holding the tension" idea:

  * It carries a small internal state h (the held tension), separate from any sequence
    context. Deliberation happens *inside* the block across inner micro-steps -- it is
    pulled OUT of the surrounding model and into this tiny, parallelizable component.
  * Each inner step it integrates evidence into h. While unresolved it emits NOTHING
    downstream (holding) -- you simply don't read a committed output until it halts.
  * It decides ITSELF when to stop. The halting signal is a function of how settled the
    field is (its speed ||dh|| ) plus the state -- not an external clock or deadline. When
    the competing pulls reconcile, the field stops moving and the block commits. This is
    the "sense of time": easy inputs commit fast, hard inputs deliberate longer.
  * Training is PonderNet-style: a proper stopping distribution over inner steps, so the
    expected output and the expected compute are both differentiable. The compute/accuracy
    operating point is set by one scalar (halt_prior) -- so the SAME architecture sweeps
    the whole speed-accuracy frontier.

Two usage modes, one class:
  - STREAMING : evidence is (T,B,in_dim) with a genuinely new sample each step (the coin:
                accumulate noisy evidence, commit when sure enough). Holding = wait for
                more data.
  - PONDERING : feed the SAME x at every step (evidence = x repeated). The block iterates
                internally on a fixed input and halts when the computation has converged.
                Holding = think longer. This is the drop-in "tension layer".

forward(evidence) -> dict(logits, lam, p_halt, speeds, h_seq)
    logits  (T,B,out_dim) : per-inner-step readout (read only the halted one at inference)
    lam     (T,B)         : per-step halt probability (given not yet halted)
    p_halt  (T,B)         : the proper stopping distribution (sums to 1 over T)
    speeds  (T,B)         : ||dh|| each step -- the discomfort / settledness readout
"""
import torch
from torch import nn
import torch.nn.functional as F


class TensionBlock(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, max_steps, halt_prior=0.2):
        super().__init__()
        self.hidden = hidden
        self.max_steps = max_steps
        self.halt_prior = halt_prior            # geometric prior mean ~ 1/halt_prior steps
        self.enc = nn.Linear(in_dim, hidden)
        self.cell = nn.GRUCell(hidden, hidden)
        # halting reads the field's SPEED (||dh||) and state -> commit when it comes to rest
        self.halt = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.Tanh(),
                                   nn.Linear(hidden, 1))
        self.read = nn.Linear(hidden, out_dim)

    def forward(self, evidence):
        T, B, _ = evidence.shape
        T = min(T, self.max_steps)
        dev = evidence.device
        h = torch.zeros(B, self.hidden, device=dev)
        lam_list, logit_list, speed_list, h_list = [], [], [], []
        for t in range(T):
            e = torch.relu(self.enc(evidence[t]))
            h_new = self.cell(e, h)
            speed = (h_new - h).norm(dim=-1, keepdim=True)          # how much the field moved
            lam_logit = self.halt(torch.cat([h_new, speed], dim=-1)).squeeze(-1)
            lam = torch.sigmoid(lam_logit)
            lam_list.append(lam)
            speed_list.append(speed.squeeze(-1))
            logit_list.append(self.read(h_new))
            h_list.append(h_new)
            h = h_new
        lam = torch.stack(lam_list)                                 # (T,B)
        lam = lam.clone()
        lam[-1] = 1.0                                               # force halt by the last step
        logits = torch.stack(logit_list)
        speeds = torch.stack(speed_list)
        # proper stopping distribution: p_halt_t = lam_t * prod_{k<t}(1-lam_k)
        oneminus = (1.0 - lam).clamp(1e-6, 1.0)
        cp = torch.cumprod(oneminus, dim=0)
        carry = torch.cat([torch.ones(1, B, device=dev), cp[:-1]], dim=0)
        p_halt = lam * carry
        return dict(logits=logits, lam=lam, p_halt=p_halt, speeds=speeds,
                    h_seq=torch.stack(h_list))


def ponder_ce_loss(out, target, lambda_c=0.01, task_loss=None):
    """Expected task loss over the stopping distribution + an expected-COMPUTE penalty.

    reg = lambda_c * E[steps] (mean over batch). Penalising the AVERAGE number of steps --
    not matching a fixed per-instance schedule -- is what lets the block be adaptive: it can
    freely spend more steps on hard inputs and fewer on easy ones, because only the mean is
    paid for. (A KL-to-prior, by contrast, pushes every instance to the same schedule and
    kills adaptivity.) lambda_c is the single knob that sweeps the speed-accuracy frontier.

    task_loss(logits_2d, target_2d)->(N,) overrides cross-entropy for non-classification heads.
    """
    logits, p_halt = out["logits"], out["p_halt"]              # (T,B,C),(T,B)
    T, B, C = logits.shape
    if task_loss is None:
        per = F.cross_entropy(logits.reshape(T * B, C),
                              target.unsqueeze(0).expand(T, B).reshape(T * B),
                              reduction="none").reshape(T, B)
    else:
        per = task_loss(logits.reshape(T * B, C),
                        target.unsqueeze(0).expand(T, *target.shape).reshape(T * B, *target.shape[1:])
                        ).reshape(T, B)
    exp_loss = (p_halt * per).sum(0).mean()
    steps = (torch.arange(T, device=logits.device).float() + 1).unsqueeze(-1)   # (T,1)
    exp_steps = (p_halt * steps).sum(0).mean()
    return exp_loss + lambda_c * exp_steps, exp_loss.detach(), exp_steps.detach()


@torch.no_grad()
def halt_infer(out, thresh=0.5):
    """Inference halting: stop at the first step where the CUMULATIVE halt probability crosses
    thresh (the median stopping time of the learned distribution). This is the decisive snap;
    it is consistent with how training scores the stopping distribution, and -- unlike a raw
    lam>=0.5 test -- it fires even when the field commits via a run of moderate halt probs.
    Returns pred (B,), steps_used (B,) [=t+1], halted_logits (B,C)."""
    p_halt, logits = out["p_halt"], out["logits"]             # (T,B),(T,B,C)
    T, B = p_halt.shape
    chalt = torch.cumsum(p_halt, dim=0)
    crossed = chalt >= thresh
    crossed[-1] = True
    step = crossed.float().argmax(0)                          # first crossing
    halted_logits = logits[step, torch.arange(B, device=p_halt.device)]
    return halted_logits.argmax(-1), step + 1, halted_logits

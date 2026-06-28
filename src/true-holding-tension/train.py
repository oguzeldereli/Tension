"""
Training: expected loss over halting time (so gradients flow through a discrete,
hold-or-commit process without REINFORCE).

For each step t the operator emits a commit prob lam_t. The probability of FIRST
committing at step t is   p_t = lam_t * prod_{j<t}(1 - lam_j).  The episode loss is
the expected cross-entropy under that halting distribution:

    L_resolve   = sum_t  p_t * CE(decode(z_t), y)
    L_discomfort= sum_t  H_t          where H_t = prob still holding entering step t
    L           = L_resolve + DISCOMFORT_W * L_discomfort

Knob B = "none": the residual "never committed" mass (prod over all t of (1-lam_t))
carries NO loss. Nothing forces a commit; the pull to commit comes only from
discomfort vs. the CE you'd pay -- so it holds while uncertain and snaps once the
cell is synthesized.
"""
import torch
import torch.nn.functional as F
from model import TensionOperator, hard_rollout
from task import sample_episode
from config import (DEVICE, MAX_STEPS, BATCH, STEPS, LR,
                    DISCOMFORT_W, W_DECODE, P_SIGNAL)


def train():
    dev = torch.device(DEVICE)
    op = TensionOperator().to(dev)
    opt = torch.optim.Adam(op.parameters(), lr=LR)
    print(f"device: {DEVICE}  discomfort_w: {DISCOMFORT_W}  deadline: none")

    for it in range(STEPS):
        obs, y = sample_episode(BATCH, dev, MAX_STEPS, P_SIGNAL)
        h = op.init_state(BATCH, dev)
        still = torch.ones(BATCH, device=dev)        # H_t: prob still holding entering t
        L_res = torch.zeros(BATCH, device=dev)
        L_dec = torch.zeros(BATCH, device=dev)       # halt-independent readout supervision
        hold = torch.zeros(BATCH, device=dev)        # expected steps held

        for t in range(MAX_STEPS):
            clock = torch.full((BATCH, 1), t / (MAX_STEPS - 1), device=dev)
            h, lam, logits = op.step(h, obs[t], clock)
            p_halt = still * lam                                  # first-halt at t
            ce = F.cross_entropy(logits, y, reduction="none")
            L_res = L_res + p_halt * ce
            L_dec = L_dec + ce                                    # readout, every step
            hold = hold + still                                   # pay to enter step t holding
            still = still * (1 - lam)

        loss = (L_res.mean()
                + W_DECODE * (L_dec / MAX_STEPS).mean()
                + DISCOMFORT_W * hold.mean())
        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % 200 == 0:
            with torch.no_grad():
                commit_mass = (1 - still).mean().item()          # soft: prob resolved by end
                read_acc = (logits.argmax(-1) == y).float().mean().item()  # final-step readout
                obs_e, y_e = sample_episode(2048, dev, MAX_STEPS, P_SIGNAL)
                comm, pred, cstep = hard_rollout(op, obs_e)
                resolved = comm.float().mean().item()
                acc = (pred[comm] == y_e[comm]).float().mean().item() if comm.any() else float("nan")
                pond = cstep[comm].float().mean().item() if comm.any() else float("nan")
            print(f"it {it:4d}  loss {loss.item():.4f}  read_acc {read_acc*100:4.1f}%  "
                  f"commit_mass {commit_mass:.3f} | hard: resolved {resolved*100:4.1f}%  "
                  f"acc {acc*100:5.1f}%  E[t] {pond:4.1f}")

    torch.save(op.state_dict(), "tension.pt")
    print("saved tension.pt")


if __name__ == "__main__":
    train()
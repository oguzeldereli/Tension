"""
Train the coin-bettor with REINFORCE (policy gradient over the trajectory).
Rollout (vectorized over a batch of episodes):
  - theta ~ U(0,1) per episode (fresh, unknown coin)
  - each turn: maybe reveal a flip (every X turns), then the policy proposes a mean
    increment mu; sample g ~ Normal(mu, SIGMA); accumulate d += g; p = sigmoid(d)
  - commit when p crosses COMMIT_P (or 1-COMMIT_P); forced commit at the deadline
  - reward = (+1 if committed side matches theta>0.5 else -1) - WAIT_COST * turns_waited
REINFORCE: loss = -(R - baseline) * sum_t logprob(g_t).  The increments are the
actions; states are detached, so we do NOT backprop through the dynamics (Approach B).
We compare against the benchmark: betting the running majority at the deadline (the
Bayes-optimal call given the flips). The agent can't beat it on accuracy, but the
interesting question is whether it learns to (a) bet the right way and (b) collapse
*early* when the evidence is already clear, trading a little accuracy for less wait-cost.
"""

import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import config as C
from policy import AccumulatorPolicy

def base_tension(p):
    """Three valleys at p = 0, 0.5, 1 ; two peaks at p = 0.25, 0.75.
    -cos(4*pi*p):  p=0 -> -1,  p=0.25 -> +1,  p=0.5 -> -1,  p=0.75 -> +1,  p=1 -> -1.
    So 'committed to 1', 'committed to 2', and 'perfectly balanced' are all minima;
    everything in between costs more. This is the static 'tension'.
    """
    return -torch.cos(4 * math.pi * p) + 1.0

def base_tension1(p):
    """A sharp triangular ridge landscape.
    0.5 is a clean valley. 0.0 and 1.0 are clean valleys.
    Everything in between is an active, constant gradient slope 
    that forces a strict binary switch.
    """
    # Absolute distance from the neutral center (0.0 at p=0.5, 0.5 at edges)
    d = torch.abs(p - 0.5)

    # Left slope (d < 0.25): Slopes UP away from 0.5, pulling p back to center
    # Right slope (d >= 0.25): Slopes DOWN toward the edges, pushing p to snap to 0 or 1
    ridge = torch.where(d < 0.25, d * 4.0, (0.5 - d) * 4.0)
    return ridge

def indecision(p):
    """1 at p=0.5 (maximally torn), 0 at p=0 or p=1 (fully committed)."""
    return 1 - (2 * p - 1) ** 2


def tension_loss(p, t, strength):
    """Full tension at the current 'thinking time' t.
    
    FIX: Passing the time variable through a cubic power (t^3) keeps the 
    early deliberation rounds completely free, allowing the network to wait 
    for multiple coin flips before the deadline pressure forces a choice.
    """
    return base_tension(p) + (t * strength) * indecision(p)

def rollout(policy, B, sigma, record=False):
    dev = next(policy.parameters()).device
    theta = torch.rand(B, device=dev)                  
    target_side = (theta > 0.5).float() 
    h      = torch.zeros(B, device=dev)
    t_cnt  = torch.zeros(B, device=dev) 
    active = torch.ones(B, device=dev, dtype=torch.bool)
    commit_turn = torch.full((B,), C.DEADLINE - 1, device=dev, dtype=torch.long)
    committed_side = torch.zeros(B, device=dev)        
    p_at_commit = torch.zeros(B, device=dev)
    running_tension = torch.zeros(B, device=dev)
    traj = [] 
    for turn in range(C.DEADLINE):
        if turn % C.X_TURNS_PER_FLIP == 0 and turn != 0:
            flip = (torch.rand(B, device=dev) < theta).float()
            h = h + flip * active.float()
            t_cnt = t_cnt + (1 - flip) * active.float()
        clock = torch.full((B,), turn, device=dev, dtype=torch.float) / C.DEADLINE
        state = torch.stack([h / C.INPUT_SCALE, t_cnt / C.INPUT_SCALE, clock], dim=1).detach() 
        mu = policy(state)                                
        p = mu[:, 0] 

        # 1. Blind turn-by-turn tracking: ONLY accumulate your boxcar tension loss
        normalized_time = float(turn) / C.DEADLINE
        running_tension = running_tension + tension_loss(p, normalized_time, C.STRENGTH) * active.float()
        if record:
            traj.append(p.detach().cpu().clone())
        newly = active & ((p > C.COMMIT_P) | (p < 1 - C.COMMIT_P))

        # Lock in the continuous p value at the exact moment of commitment
        p_at_commit = torch.where(newly, p, p_at_commit)
        committed_side = torch.where(newly, (p > 0.5).float(), committed_side)
        commit_turn = torch.where(newly, torch.full_like(commit_turn, turn), commit_turn)
        active = active & ~newly
        if not active.any():
            break

    # Balance the final weight parameters
    REWARD_STRENGTH = 30.0
    DEADLINE_PUNISHMENT = 15.0

    if record:
        while len(traj) < C.DEADLINE:
            traj.append(traj[-1])
    failed_to_commit = active.clone()

    # Capture the final positions for anyone who timed out at the deadline
    p_at_commit = torch.where(failed_to_commit, p, p_at_commit)
    committed_side = torch.where(failed_to_commit, (p > 0.5).float(), committed_side)

    # 1. Base error for early committers (standard square error)
    early_reveal_penalty = torch.square(p_at_commit - target_side)
    early_reveal_penalty = torch.where((p_at_commit > C.COMMIT_P) | (p_at_commit < 1 - C.COMMIT_P), torch.zeros_like(early_reveal_penalty), torch.ones_like(early_reveal_penalty))  # Only apply to early committers
    # 2. FIX: Differentiable deadline penalty that equals 1.0 at p=0.5 
    # but provides a continuous, live gradient pulling toward target_side
    deadline_reveal_penalty = 4.0 * torch.square(p - target_side) # Using final turn's 'p'
    deadline_reveal_penalty = torch.where(failed_to_commit, torch.ones_like(deadline_reveal_penalty), torch.zeros_like(deadline_reveal_penalty))  # Only apply to those who failed to commit
    
    # 3. Combine them using torch.where to keep the entire tracking graph active
    terminal_penalty = torch.where(failed_to_commit, deadline_reveal_penalty, early_reveal_penalty)
    
    # Balance the final weight parameters
    REWARD_STRENGTH = 30.0
   
    # We no longer need an independent deadline_penalty constant because 
    # deadline_reveal_penalty automatically scales up to enforce the full 1.0 tax!
    total_loss = running_tension + (terminal_penalty * REWARD_STRENGTH)

    # Standard metric logging
    correct = (committed_side == target_side).float()
    correct = torch.where(failed_to_commit, torch.zeros_like(correct), correct)
    waited = commit_turn.float()
    reward = torch.where(waited < C.DEADLINE, correct * 2 - 1, float(C.WAIT_COST))
    out = dict(loss=total_loss, reward=reward, correct=correct,
               waited=waited, theta=theta, h=h, t=t_cnt, committed_side=committed_side)
    if record:
        out["traj"] = torch.stack(traj, dim=1)          
    return out

def benchmark(B):
    """Bet the running majority after ALL flips (Bayes-optimal given the data)."""
    theta = torch.rand(B)
    nflips = C.MAX_FLIPS
    flips = (torch.rand(B, nflips) < theta[:, None]).float()
    h = flips.sum(1); t = nflips - h
    bet = (h > t).float()

    # ties -> coin flip
    ties = (h == t)
    bet[ties] = (torch.rand(ties.sum()) < 0.5).float()
    acc = (bet == (theta > 0.5).float()).float().mean().item()
    return acc

def main():
    torch.manual_seed(C.SEED)
    os.makedirs(C.FIG_DIR, exist_ok=True)
    policy = AccumulatorPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=C.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.5)
    bench = benchmark(20000)
    print(f"benchmark (majority-at-deadline) accuracy: {bench:.3f}")
    print(f"chance accuracy: 0.500\n")

    hist = []
    for update in range(1, C.UPDATES + 1):
        out = rollout(policy, C.BATCH, C.SIGMA)

        # Calculate the batch average of your combined landscape + reveal loss
        loss = out["loss"].mean()
        opt.zero_grad()
        loss.backward()  # Backpropagates through all turns simultaneously
        opt.step()
        scheduler.step()

        if update % 100 == 0 or update == 1:
            acc = out["correct"].mean().item()
            flips_at_commit = (out["waited"] / C.X_TURNS_PER_FLIP + 1).mean().item()
            hist.append((update, acc, flips_at_commit))
            print(f"update {update:4d}  acc {acc:.3f}  "
                  f"avg flips seen at commit {flips_at_commit:4.1f}  ")

    # ---- final eval + plots ----
    with torch.no_grad():
        ev = rollout(policy, 20000, C.SIGMA)
        final_acc = ev["correct"].mean().item()
        final_flips = (ev["waited"] / C.X_TURNS_PER_FLIP + 1).mean().item()
    print(f"\nfinal: acc {final_acc:.3f}  (benchmark {bench:.3f})  "
          f"avg flips seen at commit {final_flips:.1f} / {C.MAX_FLIPS}")

    # learning curve
    ups, accs, fl = zip(*hist)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ups, accs, label="agent")
    ax[0].axhline(bench, c="green", ls="--", label="majority benchmark")
    ax[0].axhline(0.5, c="gray", ls=":", label="chance")
    ax[0].set_title("accuracy over training"); ax[0].set_xlabel("update")
    ax[0].set_ylabel("accuracy"); ax[0].legend(); ax[0].set_ylim(0.45, 1.0)
    ax[1].plot(ups, fl, c="darkorange")
    ax[1].set_title("avg flips seen before commit"); ax[1].set_xlabel("update")
    ax[1].set_ylabel("flips"); ax[1].set_ylim(0, C.MAX_FLIPS + 1)
    plt.tight_layout(); plt.savefig(f"{C.FIG_DIR}/learning.png", dpi=110)
    print(f"saved {C.FIG_DIR}/learning.png")

    # runaway trajectories: watch p collapse, with low noise so the learned drift shows
    with torch.no_grad():
        rec = rollout(policy, 12, sigma=0.05, record=True)

    plt.figure(figsize=(10, 5))
    for i in range(12):
        col = "tab:blue" if rec["theta"][i] > 0.5 else "tab:red"
        plt.plot(rec["traj"][i].numpy(), color=col, alpha=0.7, lw=1.2)

    plt.axhline(0.5, c="gray", ls=":")
    plt.axhline(C.COMMIT_P, c="k", ls="--", lw=0.7)
    plt.axhline(1 - C.COMMIT_P, c="k", ls="--", lw=0.7)
    plt.title("p collapsing (runaway). blue = coin truly heads-biased, red = tails-biased")
    plt.xlabel("turn"); plt.ylabel("p (decision)"); plt.ylim(-0.02, 1.02)
    plt.tight_layout(); plt.savefig(f"{C.FIG_DIR}/runaway.png", dpi=110)
    print(f"saved {C.FIG_DIR}/runaway.png")

if __name__ == "__main__":
    main()
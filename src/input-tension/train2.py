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
    """Flat-topped boxcar barrier landscape."""
    d = torch.abs(p - 0.5)
    threshold_boundary = C.COMMIT_P - 0.5
    steepness = 200.0 
    
    inside_wall = 1.0 - torch.sigmoid((0.005 - d) * steepness)
    outside_wall = torch.sigmoid((threshold_boundary - d) * steepness)
    return inside_wall * outside_wall

def tension_loss(p, t, strength):
    return base_tension(p)


def rollout(policy, B, sigma, record=False):
    dev = next(policy.parameters()).device
    theta = torch.rand(B, device=dev)                  
    target_side = (theta > 0.5).float() 
    
    h      = torch.zeros(B, device=dev)
    t_cnt  = torch.zeros(B, device=dev) 
    active = torch.ones(B, device=dev, dtype=torch.bool)
    commit_turn = torch.full((B,), C.DEADLINE - 1, device=dev, dtype=torch.long)
    committed_side = torch.zeros(B, device=dev)        
    
    running_tension = torch.zeros(B, device=dev)
    running_reveal  = torch.zeros(B, device=dev) 
    traj = [] 

    for turn in range(C.DEADLINE):
        if turn % C.X_TURNS_PER_FLIP == 0 and turn != 0:
            flip = (torch.rand(B, device=dev) < theta).float()
            h = h + flip * active.float()
            t_cnt = t_cnt + (1 - flip) * active.float()

        clock = torch.full((B,), turn, device=dev, dtype=torch.float) / C.DEADLINE

        state = torch.stack([h, t_cnt, clock], dim=1).detach() / C.INPUT_SCALE
        mu = policy(state)                                
        p = mu[:, 0] 

        normalized_time = float(turn) / C.DEADLINE
        running_tension = running_tension + tension_loss(p, normalized_time, C.STRENGTH) * active.float()
        
        # Dense turn-by-turn tracking keeps the gradient lines wide open
        turn_reveal_penalty = torch.square(p - target_side) - 0.25
        running_reveal = running_reveal + turn_reveal_penalty * active.float()

        newly = active & ((p > C.COMMIT_P) | (p < 1 - C.COMMIT_P))
        committed_side = torch.where(newly, (p > 0.5).float(), committed_side)
        commit_turn = torch.where(newly, torch.full_like(commit_turn, turn), commit_turn)

        if record:
            # FIX: If an episode is about to become inactive this turn, 
            # snap its visual plot coordinate instantly to 1.0 or 0.0
            still_active = active & ~newly
            visual_p = torch.where(still_active, p, (p > 0.5).float())
            traj.append(visual_p.detach().cpu().clone())

        active = active & ~newly
        if not active.any():
            break

    if record:
        while len(traj) < C.DEADLINE:
            traj.append(traj[-1])

    failed_to_commit = active.clone()
    committed_side = torch.where(failed_to_commit, (p > 0.5).float(), committed_side)
    
    deadline_penalty = torch.where(failed_to_commit, torch.ones_like(running_tension), torch.zeros_like(running_tension))

    REWARD_STRENGTH = 25.0 
    DEADLINE_PUNISHMENT = 15.0
    total_loss = running_tension + (running_reveal * REWARD_STRENGTH) + (deadline_penalty * DEADLINE_PUNISHMENT)

    # Standard metrics
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
    theta = torch.rand(B)
    nflips = C.MAX_FLIPS
    flips = (torch.rand(B, nflips) < theta[:, None]).float()
    h = flips.sum(1); t = nflips - h
    bet = (h > t).float()
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
        loss = out["loss"].mean()

        opt.zero_grad()
        loss.backward()  
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


if __name__ == "__main__":
    main()
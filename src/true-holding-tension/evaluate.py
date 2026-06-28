"""
Evaluation.

1. Hard-latch metrics: how often it resolves vs. holds forever (silent), accuracy
   over the episodes it DOES resolve, and the ponder-time distribution.
2. Synthesis ablation: blind one stream. If the answer were decomposable into
   independent per-stream accumulators, hiding a stream would only cost that stream;
   instead accuracy collapses toward chance (~1/G), because the cell only exists at
   the intersection. This is the test that the operator is synthesizing a third
   thing, not running two stats tables.
3. A few episodes printed step by step: HOLD (zero) ... until it COMMITs to a cell.
"""
import torch
from model import TensionOperator, hard_rollout
from task import sample_episode, S_CARD
from config import DEVICE, MAX_STEPS, P_SIGNAL, GRID, TAU


def load(dev):
    op = TensionOperator().to(dev)
    op.load_state_dict(torch.load("tension.pt", map_location=dev))
    op.eval()
    return op


def main():
    dev = torch.device(DEVICE)
    op = load(dev)

    obs, y = sample_episode(8192, dev, MAX_STEPS, P_SIGNAL)
    comm, pred, cstep = hard_rollout(op, obs)

    silent = ~comm
    acc = (pred[comm] == y[comm]).float().mean().item() if comm.any() else float("nan")
    print(f"resolved (committed): {comm.float().mean()*100:5.1f}%")
    print(f"held forever (silent): {silent.float().mean()*100:5.1f}%")
    print(f"accuracy over resolved episodes: {acc*100:.2f}%")
    if comm.any():
        ct = cstep[comm].float()
        print(f"ponder time | mean {ct.mean():.1f}  min {int(ct.min())}  max {int(ct.max())}")

    # ---- synthesis ablation: hide stream B (the d / diagonal evidence) ----
    obs_blind = obs.clone()
    obs_blind[..., S_CARD:] = 0.0
    cb, pb, _ = hard_rollout(op, obs_blind)
    accb = (pb[cb] == y[cb]).float().mean().item() if cb.any() else float("nan")
    print(f"\n[ablation: stream B hidden] resolved {cb.float().mean()*100:.1f}%  "
          f"acc {accb*100:.2f}%   (chance ~ {100/GRID:.1f}%)")
    print("  -> if synthesis is real, accuracy collapses toward chance with one stream gone.")

    demo(op, dev, n=6)


@torch.no_grad()
def demo(op, dev, n=6):
    print("\n--- per-episode trace (HOLD emits zero; COMMIT emits the cell) ---")
    obs, y = sample_episode(n, dev, MAX_STEPS, P_SIGNAL)
    h = op.init_state(n, dev)
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    trace = [[] for _ in range(n)]
    pred = torch.full((n,), -1, dtype=torch.long, device=dev)
    for t in range(MAX_STEPS):
        clock = torch.full((n, 1), t / (MAX_STEPS - 1), device=dev)
        h, lam, logits = op.step(h, obs[t], clock)
        for i in range(n):
            if done[i]:
                continue
            if lam[i] >= TAU:
                pred[i] = logits[i].argmax()
                trace[i].append(f"COMMIT@{t}")
                done[i] = True
            else:
                trace[i].append("0")
    for i in range(n):
        r, c = int(y[i]) // GRID, int(y[i]) % GRID
        pr = int(pred[i])
        prc = f"({pr//GRID},{pr%GRID})" if pr >= 0 else "(silent)"
        ok = "ok " if pr == int(y[i]) else "MISS" if pr >= 0 else "----"
        print(f"  true ({r},{c}) -> pred {prc} {ok} | {' '.join(trace[i])}")


if __name__ == "__main__":
    main()

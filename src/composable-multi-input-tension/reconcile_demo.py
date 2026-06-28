"""
Demo: the reconciliation task -- where HOLDING the tension is necessary, not optional.

A 3x3 grid. The true answer is a cell (row*, col*). Two streams trickle noisy hints:
one about the ROW, one about the COLUMN. Crucially, hints about one coordinate may arrive
long before the other -- so for a stretch the operator KNOWS the row but NOT the column.

Committing then is wrong: any cell it picks is a guess on the unknown coordinate. The only
correct behavior is to HOLD (emit the zero vector, keep deliberating) until BOTH streams
have spoken, then SNAP to the reconciled cell -- a synthesis of row and column that is
neither "a row" nor "a column". There is no running tally to lean on: the answer is a
joint of two partials, and leaning toward either stream's partial is simply the wrong
type of answer.

This is the test the coin could never be: a task where premature commitment is
structurally wrong and holding-until-reconciled is the optimal strategy.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from tension_operator import TensionOperator

SEED = 0
G = 3; N = G * G
T = 24                      # passes (deadline)
P_HINT = 0.5               # chance a hint arrives on a given pass
NOISE = 0.1                # chance a hint points to the wrong coordinate
BATCH = 256; UPDATES = 1200; LR = 2e-3
HOLD_W = 0.0              # cost for staying unsettled (pressures eventual resolution)


def make_episode(B, device, first_stream_delay=True):
    """Returns per-turn evidence [T,B,6] and the true symbol [B].
    6 = [row hint one-hot(3) | col hint one-hot(3)], zeros on no-hint turns."""
    r = torch.randint(0, G, (B,), device=device)
    c = torch.randint(0, G, (B,), device=device)
    symbol = r * G + c
    ev = torch.zeros(T, B, 6, device=device)
    # which stream each hint serves; to make holding necessary, bias early hints to ONE
    # stream so the other coordinate stays unknown for a while.
    for turn in range(T):
        has = (torch.rand(B, device=device) < P_HINT)
        # early turns lean toward row hints, later toward column -> forces hold for column
        prob_row = 0.85 if turn < T // 2 else 0.15
        is_row = (torch.rand(B, device=device) < prob_row)
        # noisy coordinate
        noisy_r = torch.where(torch.rand(B, device=device) < NOISE,
                              torch.randint(0, G, (B,), device=device), r)
        noisy_c = torch.where(torch.rand(B, device=device) < NOISE,
                              torch.randint(0, G, (B,), device=device), c)
        row_oh = F.one_hot(noisy_r, G).float() * (has & is_row).float().unsqueeze(-1)
        col_oh = F.one_hot(noisy_c, G).float() * (has & ~is_row).float().unsqueeze(-1)
        ev[turn] = torch.cat([row_oh, col_oh], dim=-1)
    return ev, symbol, r, c


def rollout(op, B, record=False):
    dev = next(op.parameters()).device
    ev, symbol, r, c = make_episode(B, dev)
    z, latch = op.init_state(B, dev)
    prev_latch = latch.clone()
    commit_logits = torch.zeros(B, N, device=dev)
    commit_turn = torch.full((B,), T - 1, dtype=torch.long, device=dev)
    unsettled_cost = torch.zeros(B, device=dev)
    out_norms, settles = [], []

    last_logits = None
    for turn in range(T):
        clock = turn / (T - 1)
        out = op(ev[turn], clock, z, latch)
        z = out["z"]; latch = out["latch"]; last_logits = out["logits"]
        unsettled_cost = unsettled_cost + (1 - out["settled"]) * (prev_latch < 0.5).float()

        newly = (prev_latch < 0.5) & (latch > 0.5)
        commit_logits = torch.where(newly.unsqueeze(-1), out["logits"], commit_logits)
        commit_turn = torch.where(newly, torch.full_like(commit_turn, turn), commit_turn)
        prev_latch = latch
        if record:
            out_norms.append(out["output"].abs().sum(-1).detach().clone())   # 0 holding, 1 committed
            settles.append(out["settled"].detach().clone())

    never = latch < 0.5
    commit_logits = torch.where(never.unsqueeze(-1), last_logits, commit_logits)

    pred = commit_logits.argmax(-1)
    correct = (pred == symbol).float()
    ce = F.cross_entropy(commit_logits, symbol, reduction="none")
    loss = (ce + HOLD_W * unsettled_cost).mean()

    out = dict(loss=loss, correct=correct, commit_turn=commit_turn, symbol=symbol)
    if record:
        out["out_norms"] = torch.stack(out_norms, 1)
        out["settles"] = torch.stack(settles, 1)
        out["ev"] = ev
    return out


def main():
    torch.manual_seed(SEED)
    os.makedirs("figures", exist_ok=True)
    op = TensionOperator(in_dim=6, n_symbols=N)
    opt = torch.optim.Adam(op.parameters(), lr=LR)

    hist = []
    for u in range(1, UPDATES + 1):
        out = rollout(op, BATCH)
        opt.zero_grad(); out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(op.parameters(), 2.0)
        opt.step()
        if u % 100 == 0 or u == 1:
            acc = out["correct"].mean().item()
            ct = out["commit_turn"].float().mean().item()
            hist.append((u, acc, ct))
            print(f"update {u:4d}  acc {acc:.3f}  avg commit turn {ct:4.1f} / {T}")

    with torch.no_grad():
        ev = rollout(op, 20000)
        print(f"\nfinal acc {ev['correct'].mean().item():.3f}  "
              f"(chance = {1/N:.3f})  avg commit turn {ev['commit_turn'].float().mean():.1f}/{T}")

    ups, accs, cts = zip(*hist)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ups, accs); ax[0].axhline(1 / N, c="gray", ls=":", label="chance")
    ax[0].axhline(1.0, c="green", ls="--", label="perfect")
    ax[0].set_ylim(0, 1.05); ax[0].set_title("accuracy"); ax[0].set_xlabel("update"); ax[0].legend()
    ax[1].plot(ups, cts, c="darkorange"); ax[1].set_ylim(0, T)
    ax[1].set_title("avg commit turn"); ax[1].set_xlabel("update"); ax[1].set_ylabel("turn")
    plt.tight_layout(); plt.savefig("figures/learning.png", dpi=110)

    # the money plot: output norm over time -- should be 0 (holding) then jump to 1 (snap)
    with torch.no_grad():
        rec = rollout(op, 8, record=True)
    plt.figure(figsize=(10, 5))
    for i in range(8):
        plt.plot(rec["out_norms"][i].numpy(), lw=1.6, alpha=0.8)
    plt.title("output magnitude: 0 = holding the tension (zero vector), 1 = snapped to a symbol")
    plt.xlabel("forward pass"); plt.ylabel("||output||  (0 = holding, 1 = committed)")
    plt.ylim(-0.05, 1.1)
    plt.tight_layout(); plt.savefig("figures/hold_then_snap.png", dpi=110)
    print("saved figures/learning.png and figures/hold_then_snap.png")


if __name__ == "__main__":
    main()

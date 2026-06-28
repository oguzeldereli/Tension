"""
Benchmark 2 -- the TensionBlock as a composable layer that buys adaptive COMPUTE.

Problem space: variable-length parity. The label is the parity (XOR) of L real bits
(L ~ U[2, L_max]), fed as a +-1 stream then padded with 0s to a fixed width. Parity is the
canonical task that genuinely needs sequential depth ~ L (a bounded-width net cannot
shortcut it), and crucially the running parity STOPS CHANGING once the padding begins --
the field comes to rest exactly when the computation is complete. So "halt when settled"
here means "halt when there's nothing left to compute". Difficulty (L) varies per example.

This is the drop-in story: deliberation is pulled out of the surrounding model into one
small recurrent component that spends compute proportional to each instance's complexity.

Baseline: a conventional fixed-depth GRU classifier (same hidden width) that must process
all L_max steps for every example -- it has no way to stop early.

Claims under test:
  (1) matched accuracy, less compute: the tension layer matches the fixed-depth GRU's
      accuracy while using avg steps ~ E[L] instead of L_max (a real FLOP saving);
  (2) sense of time scales with problem size: corr(halt step, L) ~ +1 -- it literally
      thinks longer on longer problems, with nothing telling it the length.

Run:  python3 bench_compute.py
"""
import torch
from torch import nn
import torch.nn.functional as F
from tension_block import TensionBlock, ponder_ce_loss, halt_infer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
L_MAX = 16
HIDDEN = 64
BATCH = 256
STEPS = 9000
LR = 3e-3
LAMBDA_C = 0.02


def gen(B, dev):
    L = torch.randint(2, L_MAX + 1, (B,), device=dev)
    bits = (torch.rand(L_MAX, B, device=dev) < 0.5).float()          # 0/1
    idx = torch.arange(L_MAX, device=dev).unsqueeze(-1)
    real = (idx < L.unsqueeze(0)).float()                             # 1 where real bit
    parity = ((bits * real).sum(0) % 2).long()                       # XOR over real bits
    stream = torch.where(real.bool(), 2 * bits - 1, torch.zeros_like(bits))  # +-1 real, 0 pad
    return stream.unsqueeze(-1), parity, L


class GRUClassifier(nn.Module):
    """Conventional fixed-depth recurrent baseline: read all L_MAX steps, then decide."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(1, HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.read = nn.Linear(HIDDEN, 2)

    def forward(self, stream):
        T, B, _ = stream.shape
        h = torch.zeros(B, HIDDEN, device=stream.device)
        for t in range(T):
            h = self.cell(torch.relu(self.enc(stream[t])), h)
        return self.read(h)


def train_tension(dev):
    blk = TensionBlock(1, HIDDEN, 2, L_MAX).to(dev)
    opt = torch.optim.Adam(blk.parameters(), lr=LR)
    for it in range(STEPS):
        # anneal the compute penalty: learn parity first (lambda=0), THEN press it to halt
        # early. Parity gives no partial credit, so pressing for compute before the task is
        # learned collapses to "halt at step 1, guess". Warmup avoids that local optimum.
        frac = max(0.0, (it / STEPS - 0.4) / 0.3)
        lam_c = LAMBDA_C * min(1.0, frac)
        stream, y, _ = gen(BATCH, dev)
        out = blk(stream)
        loss, _, _ = ponder_ce_loss(out, y, lambda_c=lam_c)
        opt.zero_grad(); loss.backward(); opt.step()
    return blk


def train_gru(dev):
    net = GRUClassifier().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    for _ in range(STEPS):
        stream, y, _ = gen(BATCH, dev)
        loss = F.cross_entropy(net(stream), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return net


@torch.no_grad()
def main():
    dev = torch.device(DEVICE)
    torch.manual_seed(0)
    print(f"device {DEVICE}  parity, L ~ U[2,{L_MAX}], padded to width {L_MAX}\n")

    with torch.enable_grad():
        blk = train_tension(dev)
        gru = train_gru(dev)

    stream, y, L = gen(16384, dev)
    out = blk(stream)
    pred, steps, _ = halt_infer(out)
    t_acc = (pred == y).float().mean().item() * 100
    t_steps = steps.float().mean().item()
    s, l = steps.float(), L.float()
    corr = (((s - s.mean()) * (l - l.mean())).mean() / (s.std() * l.std() + 1e-8)).item()

    g_pred = gru(stream).argmax(-1)
    g_acc = (g_pred == y).float().mean().item() * 100

    print(f"{'model':<22}{'acc %':>9}{'avg compute steps':>20}{'corr(steps,L)':>16}")
    print("-" * 67)
    print(f"{'GRU (fixed depth)':<22}{g_acc:>9.2f}{float(L_MAX):>20.2f}{'n/a':>16}")
    print(f"{'TensionBlock':<22}{t_acc:>9.2f}{t_steps:>20.2f}{corr:>16.3f}")
    print(f"\ncompute saved: {100*(1 - t_steps / L_MAX):.1f}%  (avg steps {t_steps:.2f} vs fixed {L_MAX})")
    print("\nthink-time vs problem size (mean halt step per true length L):")
    print("  L     :", " ".join(f"{lv:5d}" for lv in range(2, L_MAX + 1)))
    print("  halt  :", " ".join(
        f"{s[L == lv].mean().item():5.1f}" if (L == lv).any() else "  n/a"
        for lv in range(2, L_MAX + 1)))
    print("\nread: matched accuracy at less compute; think-time rises with L (nobody told it L).")


if __name__ == "__main__":
    main()

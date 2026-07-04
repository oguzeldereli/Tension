"""
Opposition-Synthesis prototype v0 -- making "holding the tension of opposites" LITERAL.

The metaphor (user's words): the poles are not potential wells themselves; they are held by a
STRING, and the tension is literally the tension in that string. The poles roll down the loss
function; because they are opposites, the bottom along the axis of opposition is 0 -- they
cancel -- and what survives is an ORTHOGONAL new state, a third thing (the synthesis). The
block also learns when to CUT THE STRING (commit) from the external facts plus an intrinsic
characteristic of its own tension.

This file tests part (A): synthesize N poles that are opposites on N/2 axes into a third thing
that lives ORTHOGONAL to the opposition axes -- and show that doing this *structurally* (a real
spring-equilibrium projection) is what the metaphor demands, with a falsifiable payoff the
black-box baselines cannot have: invariance/extrapolation to unseen tension magnitudes.

Why the geometry is literal, not decorative:
  - K opposing pairs. Pair i = (p_a, p_b). Their disagreement vector d_i = p_a - p_b points
    along the opposition axis u_i = d_i/||d_i||; the string tension t_i = ||d_i|| is exactly how
    far the string is stretched. (This is a REAL tension signal -- unlike ||Δh|| in the old
    TensionBlock, which Experiment B showed was inert.)
  - The pair's agreement is the midpoint m_i = (p_a+p_b)/2.
  - The synthesis must be the part of the aggregated agreement that is ORTHOGONAL to every
    opposition axis: z = P_{U^perp} (mean_i m_i),  U = span{u_1..u_K}. That projection is
    exactly the minimum-energy (settled) state of unit springs pulling along the u_i -- the
    bottom of the loss the strings roll down. The opposition-axis components are cancelled to 0;
    the third thing is what is left.

The task is built so the cancellation is NOT free: opposing poles are IMBALANCED (|alpha|!=|beta|),
so the naive mean of the poles keeps a residual along u_i that grows with tension. Only a model
that explicitly identifies u_i and projects it out is invariant to tension magnitude. Train on
tension in a band; test IN-band and on a HIGHER, unseen band. Prediction:
  - projection model (ours): ~flat accuracy across tension (it cancels u_i regardless of size);
  - mean / MLP baselines: degrade as tension grows, and collapse out-of-band.

(Part B -- learning WHEN to cut the string, an iterative settling with an intrinsic+extrinsic
halt and a tension-ablation in the style of Experiment B -- is the next file. See README.)

Run:  python3 synth_opposites.py
"""
import torch
from torch import nn
import torch.nn.functional as F

DEVICE = "cpu"          # tiny; keep the GPU free for the DistilBERT benchmarks
D = 24                  # ambient dimension
K = 4                   # number of opposing pairs -> N = 2K poles, opposition on K axes
N = 2 * K
C = 4                   # classes (two sign-bits of the synthesis -> a quadrant)
NOISE = 0.05
TRAIN_TENSION = (0.5, 1.5)
TEST_TENSION = (4.0, 6.0)   # unseen, much larger -> stresses tension-invariance
HID = 128
STEPS = 4000
BATCH = 256
LR = 2e-3
SEED = 0

# fixed global readout directions defining the target (the "meaning" of the third thing)
_g = torch.Generator().manual_seed(123)
W1 = F.normalize(torch.randn(D, generator=_g), dim=0)
W2 = F.normalize(torch.randn(D, generator=_g), dim=0)


# ----------------------------- task -----------------------------
def gen(batch, tension_range, dev):
    """N poles in K imbalanced opposing pairs around a hidden synthesis g; label = quadrant of g."""
    # per-example random orthonormal basis: first K cols = opposition axes, rest = synthesis space
    Q, _ = torch.linalg.qr(torch.randn(batch, D, D, device=dev))     # (B,D,D)
    opp = Q[:, :, :K]                                                # (B,D,K) opposition axes
    syn = Q[:, :, K:]                                                # (B,D,D-K) synthesis space
    c = torch.randn(batch, D - K, device=dev)                       # synthesis content coeffs
    g = torch.bmm(syn, c.unsqueeze(-1)).squeeze(-1)                 # (B,D) the third thing, in U^perp

    lo, hi = tension_range
    tau = torch.empty(batch, K, device=dev).uniform_(lo, hi)        # string tension per pair
    delta = torch.empty(batch, K, device=dev).uniform_(-0.6, 0.6)   # imbalance of the opposites
    alpha = tau * (1 + delta)                                       # pole a coefficient (>0)
    beta = -tau * (1 - delta)                                       # pole b coefficient (<0, opposite)
    # each pair's opposition direction r_i = the i-th opposition axis
    r = opp.transpose(1, 2)                                         # (B,K,D) unit axes as rows
    p_a = g.unsqueeze(1) + alpha.unsqueeze(-1) * r                  # (B,K,D)
    p_b = g.unsqueeze(1) + beta.unsqueeze(-1) * r
    poles = torch.stack([p_a, p_b], dim=2).reshape(batch, N, D)     # interleaved a,b,a,b,...
    poles = poles + NOISE * torch.randn_like(poles)

    s1 = (g @ W1.to(dev) > 0).long()
    s2 = (g @ W2.to(dev) > 0).long()
    y = 2 * s1 + s2                                                 # quadrant in {0,1,2,3}
    return poles, y, tau


# ----------------------------- models -----------------------------
class MLPBaseline(nn.Module):
    """Black box: concat all poles -> MLP. Has the capacity to learn projection, but no reason
    to make it tension-invariant beyond the trained band."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N * D, HID), nn.GELU(),
                                 nn.Linear(HID, HID), nn.GELU(), nn.Linear(HID, C))

    def forward(self, poles):
        return self.net(poles.reshape(poles.shape[0], -1))


class OppositionSynth(nn.Module):
    """Ours: identify each opposition axis from the pole pair, measure string tension, and
    project the aggregated agreement onto the orthogonal complement of the opposition subspace
    (the settled spring equilibrium). Read the third thing out of U^perp.

    The no-projection ablation (project=False) is IDENTICAL in every way -- same 2-layer readout
    head, same aggregated-agreement input -- except it skips the orthogonal projection. So any
    gap between the two is attributable to the projection alone, nothing else."""
    def __init__(self, project=True):
        super().__init__()
        self.project = project
        self.read = nn.Sequential(nn.Linear(D, HID), nn.GELU(), nn.Linear(HID, C))

    def synthesize(self, poles):
        B = poles.shape[0]
        p_a = poles[:, 0::2, :]                                     # (B,K,D)
        p_b = poles[:, 1::2, :]
        diff = p_a - p_b                                            # (B,K,D) string vector
        t = diff.norm(dim=-1)                                       # (B,K) tension = string stretch
        u = diff / (t.unsqueeze(-1) + 1e-6)                        # (B,K,D) opposition axes
        m = 0.5 * (p_a + p_b)                                       # (B,K,D) agreement midpoints
        content = m.mean(1)                                         # (B,D) aggregated agreement
        if not self.project:
            return content, t
        U = u.transpose(1, 2)                                       # (B,D,K) axes as columns
        G = torch.bmm(U.transpose(1, 2), U)                        # (B,K,K) Gram (~I, well-cond.)
        rhs = torch.bmm(U.transpose(1, 2), content.unsqueeze(-1))  # (B,K,1) U^T content
        coeff = torch.linalg.solve(G + 1e-4 * torch.eye(K, device=poles.device), rhs)
        par = torch.bmm(U, coeff).squeeze(-1)                       # (B,D) component IN U
        z = content - par                                          # (B,D) third thing in U^perp
        return z, t

    def forward(self, poles):
        z, _ = self.synthesize(poles)
        return self.read(z)


# ----------------------------- train / eval -----------------------------
def train(model, dev):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for it in range(STEPS):
        poles, y, _ = gen(BATCH, TRAIN_TENSION, dev)
        loss = F.cross_entropy(model(poles), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def acc(model, dev, tension_range, n=4096):
    model.eval()
    poles, y, _ = gen(n, tension_range, dev)
    return (model(poles).argmax(-1) == y).float().mean().item() * 100


@torch.no_grad()
def acc_by_tension(model, dev, bins, n=2048):
    out = []
    for lo, hi in bins:
        out.append(acc(model, dev, (lo, hi), n))
    return out


def main():
    dev = torch.device(DEVICE)
    torch.manual_seed(SEED)
    print(f"device {DEVICE}  D={D} pairs K={K} (N={N} poles) classes={C}")
    print(f"train tension {TRAIN_TENSION}  |  test extrapolation tension {TEST_TENSION}\n")

    models = {
        "MLP (black box)": MLPBaseline(),
        "OppositionSynth NOPROJ (abl)": OppositionSynth(project=False),
        "OppositionSynth (ours, project)": OppositionSynth(project=True),
    }
    for name, m in models.items():
        train(m.to(dev), dev)

    print(f"{'model':<34}{'acc @train band':>16}{'acc @unseen band':>18}{'drop':>8}")
    print("-" * 76)
    for name, m in models.items():
        a_in = acc(m, dev, TRAIN_TENSION)
        a_out = acc(m, dev, TEST_TENSION)
        print(f"{name:<34}{a_in:>16.2f}{a_out:>18.2f}{a_in-a_out:>8.2f}")

    bins = [(0.5, 1.0), (1.0, 1.5), (2.0, 2.5), (3.0, 3.5), (4.0, 5.0), (5.0, 6.0), (6.0, 8.0)]
    print("\naccuracy vs tension magnitude (train band = first two bins):")
    print("  band :", " ".join(f"{lo:.1f}-{hi:.1f}" for lo, hi in bins))
    for name, m in models.items():
        row = acc_by_tension(m, dev, bins)
        print(f"  {name[:28]:<28}", " ".join(f"{v:6.1f}" for v in row))

    # sanity: does the block's measured tension track the true string stretch?
    ours = models["OppositionSynth (ours, project)"]
    poles, _, tau = gen(2048, (0.5, 3.5), dev)
    _, t = ours.synthesize(poles)
    tt, tn = tau.flatten(), t.flatten()
    corr = (((tt - tt.mean()) * (tn - tn.mean())).mean() / (tt.std() * tn.std() + 1e-8)).item()
    print(f"\nsanity: corr(measured string tension ||p_a-p_b||, true tension tau) = {corr:.3f} "
          f"(should be ~1: the block reads real tension)")
    print("\nread: ours should stay ~flat across tension (it cancels the opposition axis "
          "regardless of\n      stretch); mean/MLP should fall as tension grows and collapse "
          "out-of-band.")


if __name__ == "__main__":
    main()

"""
Experiment B -- ABLATION in the PONDERING regime: does the settling signal ||Δh|| do anything
when "the computation converged" is a genuinely different event from "I'm confident"?

Experiment A ran this null-test on SST-2 (single-pass classification) and found the settling
signal adds ~0 -- but that is the regime where [CLS] confidence is already a near-perfect halt
signal, so settling has no room to help. The honest open question (flagged at the end of
Experiment A) is whether the null holds in the PONDERING regime, where a token can be
transiently confident mid-chase, or converge before confidence saturates. This is the one
place the novel signal has a real shot.

So we reuse Benchmark 3's pointer-chasing task and Pondering-UT, and run the SAME structure as
Experiment A: ONE trained model, exit POLICY is the only thing that varies, every policy reads
the SAME per-step states of the SAME model (training budget identical by construction). Five
policies, swept across their knob to trace the full per-token speed-accuracy frontier:

  1. confidence threshold (DeeBERT-style, parameter-free): exit when softmax max-prob >= tau.
  2. pure settling (parameter-free): exit when the field speed ||Δh|| <= eps -- the raw novel
     signal, with NO learned head and NO confidence. If settling carries signal on its own,
     this frontier is competitive; if it is dominated by (1), the signal is weak.
  3. tension-halt FULL (mine): learned halt head reading [h, ||Δh||] -- confidence + settling.
  4. tension-halt ABLATED (fair): a SECOND learned head, same architecture / same joint
     PonderNet training / same budget, but WITHOUT the ||Δh|| feature ([h] only).
  5. PABEE patience: exit once the argmax prediction is unchanged for p consecutive steps.

Key comparisons (at matched per-token compute):
  (3) vs (4)  -> marginal value of the ||Δh|| feature inside a learned head.
  (2) vs (1)  -> does the RAW settling signal carry information confidence does not.
  best-learned vs (1) -> does learning to halt beat a plain confidence threshold here.
A null (settling ~ confidence everywhere) is a legitimate result and localizes the project to
"PonderNet-style halting where confidence is the operative feature". A win for (2)/(3) over (1)
localizes the novel contribution to iterative-computation halting.

Checkpoint/resume + thermal cooldowns are baked in: an interrupted run continues from the last
checkpoint instead of restarting.

Run:  python3 bench_transformer_ablation.py
"""
import os
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# reuse the Benchmark-3 task and core modules verbatim (no duplication / no drift)
from bench_transformer import (
    gen_pool, sample, SharedLayer, Embed, n_params,
    N, V, D_MODEL, T_MAX, TRAIN_POOL, TEST_POOL, BATCH, STEPS, LR, LAMBDA_C,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "runs/exp_b_ablation.pt"
CKPT_EVERY = 500            # save a resumable checkpoint this often
COOLDOWN_EVERY = 500        # let the GPU idle briefly to keep temperatures down
COOLDOWN_SECS = 3
LOG_EVERY = 500


# ----------------------------- model -----------------------------
class PonderingUTAblation(nn.Module):
    """Shared-layer Universal Transformer with TWO jointly-trained per-token halt heads:
    halt_full reads [h, ||Δh||]; halt_conf reads [h] only (the fair no-speed ablation)."""
    def __init__(self):
        super().__init__()
        self.emb = Embed()
        self.layer = SharedLayer()
        self.norm = nn.LayerNorm(D_MODEL)
        self.read = nn.Linear(D_MODEL, V)
        self.halt_full = nn.Sequential(nn.Linear(D_MODEL + 1, D_MODEL), nn.Tanh(),
                                       nn.Linear(D_MODEL, 1))
        self.halt_conf = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.Tanh(),
                                       nn.Linear(D_MODEL, 1))

    def unroll(self, batch):
        """Full unroll (no early stop) -> per-step logits, speeds, and BOTH heads' lam.
        logits (T,B,N,V), speed/lam_* (T,B,N)."""
        x = self.emb(batch)
        logit_l, speed_l, lf_l, lc_l = [], [], [], []
        for _ in range(T_MAX):
            x_new = self.layer(x)
            speed = (x_new - x).norm(dim=-1, keepdim=True)            # (B,N,1)
            logit_l.append(self.read(self.norm(x_new)))
            lf_l.append(torch.sigmoid(self.halt_full(torch.cat([x_new, speed], -1)).squeeze(-1)))
            lc_l.append(torch.sigmoid(self.halt_conf(x_new).squeeze(-1)))
            speed_l.append(speed.squeeze(-1))
            x = x_new
        return (torch.stack(logit_l), torch.stack(speed_l),
                torch.stack(lf_l), torch.stack(lc_l))


def ponder_terms(logits, lam, y, lambda_c):
    """PonderNet expected CE over the stopping distribution + expected-compute penalty,
    for ONE halt head. logits (T,B,N,V), lam (T,B,N)."""
    T, B, n, C = logits.shape
    lam = lam.clone(); lam[-1] = 1.0
    oneminus = (1.0 - lam).clamp(1e-6, 1.0)
    carry = torch.cat([torch.ones_like(lam[:1]), torch.cumprod(oneminus, 0)[:-1]], 0)
    p_halt = lam * carry                                             # (T,B,N)
    ce = F.cross_entropy(logits.reshape(-1, C),
                         y.unsqueeze(0).expand(T, B, n).reshape(-1),
                         reduction="none").reshape(T, B, n)
    exp_ce = (p_halt * ce).sum(0).mean()
    steps = (torch.arange(T, device=logits.device).float() + 1).view(T, 1, 1)
    exp_steps = (p_halt * steps).sum(0).mean()
    return exp_ce + lambda_c * exp_steps


# ----------------------------- train (checkpoint/resume) -----------------------------
def train(model, pool, dev):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    start = 0
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        torch.set_rng_state(ck["rng"].cpu())
        if dev.type == "cuda" and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
        start = ck["it"] + 1
        print(f"  resumed from {CKPT} at step {start}/{STEPS}", flush=True)

    def save(it):
        os.makedirs(os.path.dirname(CKPT), exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "it": it,
                    "rng": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state() if dev.type == "cuda" else None},
                   CKPT)

    model.train()
    for it in range(start, STEPS):
        frac = max(0.0, (it / STEPS - 0.4) / 0.3)                   # anneal compute penalty
        lam_c = LAMBDA_C * min(1.0, frac)
        b = sample(pool, BATCH, dev)
        logits, _, lf, lc = model.unroll(b)
        # both heads trained jointly to the same task, identical budget (fair ablation)
        loss = (ponder_terms(logits, lf, b["y"], lam_c)
                + ponder_terms(logits, lc, b["y"], lam_c))
        opt.zero_grad(); loss.backward(); opt.step()
        if it % LOG_EVERY == 0:
            print(f"  step {it:5d}/{STEPS}  loss {loss.item():.4f}  lam_c {lam_c:.4f}", flush=True)
        if CKPT_EVERY and it and it % CKPT_EVERY == 0:
            save(it)
        if COOLDOWN_EVERY and it and it % COOLDOWN_EVERY == 0:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            time.sleep(COOLDOWN_SECS)
    save(STEPS - 1)
    return model


@torch.no_grad()
def gather(model, pool, dev):
    """Full unroll over the whole test pool -> flattened per-TOKEN tensors (T, M)."""
    model.eval()
    Lg, Sp, Lf, Lc = [], [], [], []
    n = pool["y"].shape[0]
    for i in range(0, n, 512):
        b = {k: v[i:i+512] for k, v in pool.items()}
        lg, sp, lf, lc = model.unroll(b)
        Lg.append(lg); Sp.append(sp); Lf.append(lf); Lc.append(lc)
    logits = torch.cat(Lg, 1)                                       # (T,B,N,V)
    T = logits.shape[0]
    logits = logits.reshape(T, -1, V)                              # (T,M,V)
    speed = torch.cat(Sp, 1).reshape(T, -1)                        # (T,M)
    lf = torch.cat(Lf, 1).reshape(T, -1)
    lc = torch.cat(Lc, 1).reshape(T, -1)
    return logits, speed, lf, lc


# ----------------------------- frontier policies (all per-token, on T,M) -----------------------------
def _eval_at(layer, preds, y):
    M = preds.shape[1]
    idx = torch.arange(M, device=preds.device)
    pred = preds[layer, idx]
    return layer.float().mean().item() + 1, (pred == y).float().mean().item() * 100, layer


def frontier_ge(score, preds, y, threshs):
    """Exit at first step with score >= thresh (confidence / cumulative-halt)."""
    pts = []
    for th in threshs:
        crossed = score >= th
        crossed[-1] = True
        layer = crossed.float().argmax(0)
        a, acc, _ = _eval_at(layer, preds, y)
        pts.append((a, acc))
    return pts


def frontier_le(score, preds, y, threshs):
    """Exit at first step with score <= thresh (pure settling: field speed has dropped)."""
    pts = []
    for th in threshs:
        crossed = score <= th
        crossed[-1] = True
        layer = crossed.float().argmax(0)
        a, acc, _ = _eval_at(layer, preds, y)
        pts.append((a, acc))
    return pts


def lam_to_chalt(lam):
    """Learned head lam (T,M) -> cumulative stopping mass (T,M), PonderNet-consistent."""
    lam = lam.clone(); lam[-1] = 1.0
    oneminus = (1.0 - lam).clamp(1e-6, 1.0)
    carry = torch.cat([torch.ones_like(lam[:1]), torch.cumprod(oneminus, 0)[:-1]], 0)
    return torch.cumsum(lam * carry, 0)


def frontier_pabee(preds, y):
    """Exit once the argmax is unchanged for p consecutive steps; sweep p = 1..T."""
    T, M = preds.shape
    dev = preds.device
    pts = []
    for p in range(1, T + 1):
        cnt = torch.ones(M, device=dev)
        done = torch.zeros(M, dtype=torch.bool, device=dev)
        layer = torch.full((M,), T - 1, dtype=torch.long, device=dev)
        for i in range(T):
            if i > 0:
                same = preds[i] == preds[i - 1]
                cnt = torch.where(same, cnt + 1, torch.ones_like(cnt))
            fire = (~done) & (cnt >= p)
            layer[fire] = i; done |= fire
        a, acc, _ = _eval_at(layer, preds, y)
        pts.append((a, acc))
    return pts


def at(pts, x):
    a = sorted(pts); xs = [p[0] for p in a]; ys = [p[1] for p in a]
    return float(np.interp(x, xs, ys))


def corr_at(score, preds, depth, target_steps, threshs, mode="ge"):
    """corr(exit-step, depth) at the knob whose avg compute is closest to target_steps."""
    best = None
    for th in threshs:
        if mode == "ge":
            crossed = score >= th
        else:
            crossed = score <= th
        crossed[-1] = True
        layer = crossed.float().argmax(0).float()
        avg = layer.mean().item() + 1
        if best is None or abs(avg - target_steps) < abs(best[0] - target_steps):
            d = depth.float()
            c = (((layer - layer.mean()) * (d - d.mean())).mean()
                 / (layer.std() * d.std() + 1e-8)).item()
            best = (avg, c)
    return best[1]


# ----------------------------- main -----------------------------
def main():
    dev = torch.device(DEVICE)
    torch.manual_seed(0)
    print(f"device {DEVICE}  Experiment B (pondering-regime ablation)  pointer-chasing  "
          f"N={N} V={V} T_max={T_MAX} steps={STEPS}")
    train_pool = gen_pool(TRAIN_POOL, dev)
    test = gen_pool(TEST_POOL, dev)
    print(f"pools built; mean chain depth = {test['depth'].float().mean().item():.2f}\n")

    model = PonderingUTAblation().to(dev)
    print(f"training one model ({n_params(model)} params); policies differ only at inference...",
          flush=True)
    train(model, train_pool, dev)

    logits, speed, lf, lc = gather(model, test, dev)
    preds = logits.argmax(-1)                                       # (T,M)
    y = test["y"].reshape(-1)                                       # (M,)
    depth = test["depth"].reshape(-1)                               # (M,)
    conf = logits.softmax(-1).max(-1).values                       # (T,M)

    # precondition: does more compute buy accuracy at all? (fixed-K, full unroll)
    fixedK = [(preds[k] == y).float().mean().item() * 100 for k in range(T_MAX)]
    print("precondition -- fixed-K accuracy (must rise with depth or nothing can halt-adapt):")
    print("  K:", " ".join(f"{k+1}:{fixedK[k]:.1f}" for k in range(T_MAX)), "\n")

    th_conf = list(np.linspace(0.5, 0.9999, 16))
    th_halt = list(np.linspace(0.1, 0.95, 16))
    # settling thresholds from the observed speed distribution (low eps = wait for rest)
    qs = torch.quantile(speed.flatten().float(),
                        torch.linspace(0.02, 0.98, 16, device=speed.device)).tolist()

    F1 = frontier_ge(conf, preds, y, th_conf)                      # confidence
    F2 = frontier_le(speed, preds, y, qs)                          # pure settling ||Δh||
    F3 = frontier_ge(lam_to_chalt(lf), preds, y, th_halt)         # tension-full (mine)
    F4 = frontier_ge(lam_to_chalt(lc), preds, y, th_halt)         # tension-ablated (no speed)
    F5 = frontier_pabee(preds, y)                                  # PABEE

    print("speed-accuracy at matched compute (acc % interpolated at fixed avg steps/token):")
    print(f"{'policy':<30}{'@2.0':>8}{'@3.0':>8}{'@4.0':>8}{'corr(halt,depth)@~3':>22}")
    rows = [
        ("1 confidence (DeeBERT)", F1, conf, "ge"),
        ("2 pure settling ||Δh||", F2, speed, "le"),
        ("3 tension-full [h,||Δh||] (mine)", F3, lam_to_chalt(lf), "ge"),
        ("4 tension-ablated [h] (no speed)", F4, lam_to_chalt(lc), "ge"),
        ("5 PABEE patience", F5, None, None),
    ]
    for name, fr, sc, mode in rows:
        cstr = ""
        if sc is not None:
            ths = qs if mode == "le" else (th_conf if "confidence" in name else th_halt)
            cstr = f"{corr_at(sc, preds, depth, 3.0, ths, mode):>22.3f}"
        else:
            cstr = f"{'n/a':>22}"
        print(f"{name:<30}" + "".join(f"{at(fr, x):>8.2f}" for x in (2.0, 3.0, 4.0)) + cstr)

    print("\n--- verdict (Experiment B) ---")
    for x in (2.0, 3.0):
        full, abl, conf_a, sett = at(F3, x), at(F4, x), at(F1, x), at(F2, x)
        print(f"  @{x:.0f} steps: full {full:.2f} | ablated(no speed) {abl:.2f} "
              f"(Δ {full-abl:+.2f}) | confidence {conf_a:.2f} | pure-settling {sett:.2f}")
    print("  read: Δ(full vs ablated) ~0 AND settling<=confidence  => ||Δh|| adds nothing even"
          " here; project = confidence-driven halting.")
    print("        Δ>0 OR pure-settling competitive with/over confidence  => the settling signal"
          " is real in the pondering regime.")

    try:
        plot([("1 confidence (DeeBERT)", F1, "tab:gray", "s-"),
              ("2 pure settling ||Δh||", F2, "tab:orange", "v-"),
              ("3 tension-full (mine)", F3, "tab:blue", "o-"),
              ("4 tension-ablated (no speed)", F4, "tab:red", "d-"),
              ("5 PABEE patience", F5, "tab:green", "^-")], fixedK)
    except Exception as e:
        print(f"(plot skipped: {e})")


def plot(series, fixedK):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(6.4, 4.4))
    for name, fr, color, style in series:
        a = sorted(fr); xs = [p[0] for p in a]; ys = [p[1] for p in a]
        plt.plot(xs, ys, style, color=color, ms=5, label=name, alpha=0.9)
    plt.axhline(fixedK[-1], ls=":", color="k", lw=0.8)
    plt.text(T_MAX * 0.55, fixedK[-1] + 0.1, "full unroll", fontsize=8)
    plt.xlabel("average steps/token (compute)"); plt.ylabel("pointer-chasing accuracy %")
    plt.title("Experiment B: exit-policy frontiers in the PONDERING regime")
    plt.legend(fontsize=7.5, loc="lower right"); plt.grid(alpha=0.3); plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/transformer_ablation_frontier.png", dpi=130)
    print("\nsaved figures/transformer_ablation_frontier.png")


if __name__ == "__main__":
    main()

"""
budget_jointreadout.py -- the next lever: stop freezing the readout. Co-train the synthesis readout
WITH the elastic budget (commit + request) end to end, so the readout can (a) become more accurate
under the halting it will actually face and (b) produce a confidence signal the triage can use.

Everywhere above, the readout was trained once and FROZEN, and the residual ~10pt "oracle gap"
turned out to be a readout problem, not a budget problem (budget elasticity can stop wasting compute
but cannot manufacture signal). Caveat on that oracle: it is label-aware and can pick a *luckily*
correct depth, so part of the gap is unreachable by ANY method -- the honest target is to beat the
frozen-readout elastic frontier and the standard early-exit baselines, not to hit the oracle.

This file: warm-start the readout (so joint training is stable), then for each discomfort mu train
TWO systems from the same warm start -- FROZEN readout (= Part C") and JOINT (readout unfrozen,
trained with halt+request under the elastic binding pool). Same loss as Part C":
    L = E[CE]  +  mu * E[(granted extra budget / E0)^2]
Baselines (Fixed-N, confidence>=tau, oracle) are computed on the warm-start readout -- i.e. what a
standard early-exit system would have. Question: does co-training the readout lift the elastic
frontier above the frozen one (and above the baselines), i.e. is the readout really the lever?

Run:  python3 budget_jointreadout.py
"""
import os, sys, copy
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_opposites import D, C
from energy_budget import train_readout, make_instances, T_MAX, M, DEVICE
from compare_baselines import trajectories, f_fixed_n, f_conf_threshold, f_oracle, TAUS
from budget_dynamic import (Head, run_system, make_batch, evaluate,
                            E0_BASE, MU_GRID, BATCH_INST, LR)

STEPS_JOINT = 1000


def train_run(read, dev, mode, e0, mu, train_read):
    head = Head(dynamic=True).to(dev)
    for p in read.parameters():
        p.requires_grad_(train_read)
    params = list(head.parameters()) + (list(read.parameters()) if train_read else [])
    opt = torch.optim.Adam(params, lr=LR)
    read.train() if train_read else read.eval()
    head.train()
    inst_id = torch.arange(BATCH_INST, device=dev).repeat_interleave(M)
    for it in range(STEPS_JOINT):
        poles, y, sigma = make_batch(dev, mode)
        ce, _, _, G = run_system(read, head, poles, y, sigma, inst_id, BATCH_INST, dev, e0)
        loss = ce.mean() + mu * ((G / e0) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return read, head


def frontier(read0, dev, mode, train_read):
    out = []
    for mu in MU_GRID:
        read = copy.deepcopy(read0)
        read, head = train_run(read, dev, mode, E0_BASE, mu, train_read)
        s, a, g = evaluate(read, head, _DATA[mode], 1024, dev, E0_BASE)
        out.append((s, a, g))
    return out


def _table(title, rows, has_g=False):
    print(f"\n{title}")
    if has_g:
        print(f"  {'steps/op':>9}{'acc %':>8}{'req G/E0':>10}")
        for s, a, g in rows:
            print(f"  {s:>9.2f}{a:>8.2f}{g:>10.2f}")
    else:
        print(f"  {'steps/op':>9}{'acc %':>8}")
        for s, a in rows:
            print(f"  {s:>9.2f}{a:>8.2f}")


_DATA = {}


def run_regime(read0, dev, mode):
    n_inst = 1024
    _DATA[mode] = make_instances(n_inst, dev, mode=mode)
    poles, y, sigma, inst_id = _DATA[mode]
    conf, corr = trajectories(read0, poles, y, sigma, dev)
    budgets = [M * b for b in [1, 1.5, 2, 3, 4, 6, 8, 12]]
    print("\n" + "=" * 72)
    print(f"{mode.upper()} regime  (elastic base E0={E0_BASE})")
    print("training FROZEN-readout elastic frontier...", flush=True)
    froz = frontier(read0, dev, mode, train_read=False)
    print("training JOINT-readout elastic frontier (readout unfrozen)...", flush=True)
    joint = frontier(read0, dev, mode, train_read=True)
    _table("Fixed-N (uniform, warm-start readout)", f_fixed_n(corr))
    _table("Confidence>=tau (SPRT/DeeBERT, warm-start readout)", f_conf_threshold(conf, corr, TAUS, dev))
    _table("Oracle knapsack (label-aware UPPER BOUND)", f_oracle(corr, n_inst, budgets, dev))
    _table("ELASTIC, FROZEN readout (Part C\")", froz, has_g=True)
    _table("ELASTIC, JOINT readout (ours: unfrozen)", joint, has_g=True)
    _plot(mode, froz, joint, f_fixed_n(corr), f_conf_threshold(conf, corr, TAUS, dev),
          f_oracle(corr, n_inst, budgets, dev))


def _plot(mode, froz, joint, fN, cf, orc):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    series = [(orc, "oracle (label-aware bound)", "k--"), (cf, "confidence>=tau (SPRT)", "o-"),
              (fN, "fixed-N", "x-"), ([(s, a) for s, a, _ in froz], "elastic, frozen readout", "^-"),
              ([(s, a) for s, a, _ in joint], "elastic, JOINT readout (ours)", "s-")]
    for rows, lab, st in series:
        ax.plot([r[0] for r in rows], [r[1] for r in rows], st, label=lab)
    ax.set_xlabel("compute (mean inner steps / operator)"); ax.set_ylabel("accuracy (%)")
    ax.set_title(f"Co-training the readout with the elastic budget -- {mode}")
    ax.grid(alpha=0.3); ax.legend(fontsize=8); fig.tight_layout()
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(p, exist_ok=True)
    fn = os.path.join(p, f"budget_jointreadout_{mode}.png")
    fig.savefig(fn, dpi=120); print(f"  saved {fn}")


def main():
    dev = torch.device(DEVICE)
    print(f"device {DEVICE}  M={M}  T_MAX={T_MAX}  co-train readout + elastic budget")
    print("warm-starting the readout...", flush=True)
    read0 = train_readout(dev)
    for mode in ["distractor", "hetero"]:
        run_regime(read0, dev, mode)
    print("\nread: does the JOINT-readout elastic frontier sit ABOVE the frozen one and the")
    print("      baselines -- i.e. is co-training the readout the lever that squeezes past?")


if __name__ == "__main__":
    main()

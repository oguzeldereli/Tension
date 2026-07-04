# TODO / status

Cleanup pass done (this session): the root [README.md](README.md) is now the **map** — every
iteration (stages 0–6, plus the per-bench and per-part sub-iterations) has a plain-English ELI5,
a "what's genuinely new", and an honest "is it useful / any advantage". Each subproject README has
a matching ELI5 box at the top.

## Done recently
- **Stage 6 `opposition-synthesis`** (new this arc):
  - Part A `synth_opposites` — orthogonal-projection synthesis. **The genuinely-new win**:
    extrapolates 3–4× outside the training tension range (drop −2.3 vs baselines ~9).
  - Part B `tension_synth_operator` — Part A inside the hold/deliberate/commit contract.
  - Part C `energy_budget` + `compare_baselines` — shared "psychic energy budget". As an
    *inference-time control* it is **inert** (ties Fixed-N; only buys compute-robustness). Tested
    against known algorithms (fixed-N, SPRT/DeeBERT, value-of-computation, oracle).
  - Part C′ `budget_trained` — **budget tied into the loss** → learned triage. **Small real win**:
    beats Fixed-N (+3) and confidence-exit in the hopeless-distractor case.
  - Part C″ `budget_dynamic` — elastic/requestable budget (convex discomfort + cap). Works; reaches
    C′ adaptively from a tiny base but does not exceed it.
  - Part C‴ `budget_jointreadout` — unfreeze/co-train the readout. **No gain**; the residual gap is
    irreducible noise + a label-peeking oracle. Closes the budget arc.
- **Bench 7** (`bench_babi_pretrained.py`) — DONE on CPU: exit depth tracks reasoning hops
  (corr 0.778), full accuracy at ~78% depth.
- **item b** (`bench_glue.py`) — DONE: wall-clock throughput 1.88× @ τ=0.99, 2.63× @ τ=0.95.
- **Experiments A & B** — the `‖Δh‖` settling signal falsified twice; the real engine is learned
  halting (PonderNet/ACT).

## Next ideas (not blocked by hardware)
- **A task with genuine triage structure** — sub-problems that differ *sharply* in
  value-of-compute (e.g. "confidently wrong fast" vs "slow but solvable"), where reallocating
  effort actually changes outcomes. The budget arc tied baselines because the current toy task has
  near-uniform value-of-compute; this is the change that could turn the budget into a real win.
- **Push Part A's extrapolation off synthetic data** — the orthogonal-synthesis advantage is the
  most promising thread; test whether it survives in a non-toy setting.
- **Composition** — nested deliberation that has to pay (e.g. nested pointer-chasing where the
  outer chase can't settle until inner sub-chases do). Note: the signal being composed on is the
  learned halt readout, not `‖Δh‖`.

## Housekeeping
- `src/tension-block/runs/*.pt` are large (~1.5 GB) completed-run checkpoints, gitignored and
  regenerable; safe to delete to reclaim disk (results are recorded in the READMEs).

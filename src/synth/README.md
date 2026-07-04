# synth — holding that resolves into a third thing

> **ELI5:** two opposite votes cancel out, so the answer can't sit between them — it swings **90°
> off to the side** to a perpendicular third thing. v2 also hides *when* to stop and makes one clue
> arrive late, so the operator is forced to wait for missing information.
> **Genuinely new:** it **owns its own timing with no clock** — it commits when its internal field
> comes to *rest*, and that moment tracks when the late clue arrives (corr **0.98**).
> **Useful? / advantage:** The cleanest *concept* result — the strongest evidence it genuinely
> holds and self-times. Still synthetic, no benchmark advantage. (Stage 4.)

Angular synthesis task. Two poles A, B (unit vectors). Evidence ("votes") arrives over
time. Three regimes:

- **decisive_A / decisive_B** — votes net strongly one way → answer is that pole (on the A–B arc).
- **balanced** — votes cancel → answer is the **perpendicular** to the bisector: a *third
  thing*, off the arc, orthogonal to the opposition. With no noise a symmetric tension
  cannot resolve *along* the axis (no tie-breaker), so deterministic resolution is only
  possible perpendicular to it. The synthesis literally cannot live between the poles.

The interpolator ("leaning") baseline can only land on the A–B arc, so at balance it sits
on the bisector ~90° from the synthesis. It's a reference line, not a strawman.

## v1 — the synthesis half (`synth_*.py`)

`synth_models.py` / `synth_task.py` / `synth_run.py` / `synth_trace.py`.

Result: the `StructuredField` produces a genuine third thing (BALANCED ~100%, off-axis
~90°, determinate), not a stats-table blend — the off-axis answer rules that out. The
`LearnedField` is the free-form control. `InterpLeaner` fails balanced (off-axis ~0).

**Two scaffolds remained, though:**

1. **Commit timing was task-clocked.** The observation carried an `end` bit (votes over →
   commit). The operator was *told* "time's up" rather than deciding from an internal
   sense that the tension had dissolved.
2. **The answer was knowable from step 0.** Both poles were present the whole time, so the
   perpendicular was computable immediately. Only the *regime* unfolded — a weaker form of
   "holding is necessary" than holding because the information hasn't arrived yet.

## v2 — intrinsic release + forced holding (`synth_*2.py`)

`synth_models2.py` / `synth_task2.py` / `synth_run2.py` / `synth_trace2.py`. Both scaffolds removed:

1. **No `end` flag.** Nothing in the input says when to commit. Commit is *detected* from
   the dynamics: the first step where the field has reached a committed magnitude (|z|>0.5)
   **and** come to rest (speed < eps). The operator owns its commit timing.
2. **Staggered pole arrival.** Pole A is present from step 0; pole B arrives at a random
   later step `tB` (its obs slots are zero until then). The perpendicular needs *both*
   poles, so early holding is forced by genuine information-insufficiency — the answer is
   unknowable early.
3. **Uncertainty mode.** Per-step votes are corrupted by Gaussian noise, so the regime is
   only recoverable by integrating evidence over time ("complete synthesis under
   uncertainty").

`IntrinsicField` holds because the data switches the driving forces off (no pole B → no
drive → it sits at the null with nothing to do), then synthesizes and rests when B arrives.

### Results (`python3 synth_run2.py`)

| metric (balanced)        | intrinsic (clean) | intrinsic (noisy σ=0.6) | interp |
|--------------------------|------------------:|------------------------:|-------:|
| BALANCED acc %           | 99.7              | 84.5                    | 0.0    |
| off-axis (deg, 90=synth) | 89.2              | 81.9                    | ~0–11  |
| HOLD \|z\| before t\*    | 0.011             | 0.019                   | 0.53   |
| commit==final agree %    | 100               | 100                     | ~56    |
| **commit-step vs tB corr** | **0.98**        | **0.89**                | ~0     |
| commit step (tB lo→hi)   | 16.2 → 23.0       | 18.2 → 23.0             | flat   |

Decisive accuracy ~100% (clean) / ~95% (noisy) for the field.

**Read:** the field genuinely holds at the null (|z|≈0) while information is missing; the
intrinsic rest-point readout matches the resolved answer (100%); and the commit step
correlates ~0.98 with when the second pole arrives and slides later as it arrives later —
**timing owned by the dynamics tracking information, not a clock.** The interpolator leans
(off-axis ~0), never holds, and its commit step doesn't track `tB`.

### The trajectory (`python3 synth_trace2.py`)

For balanced episodes with different `tB`, `|z|` is pinned at ~0 until exactly step `tB`,
then snaps to ~1 (off-axis swinging to ~90, the synthesis), then rests — and the detected
commit step moves with `tB`. Hold → snap → rest, released from inside.

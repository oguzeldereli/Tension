# Tension

An operator that **holds the tension of opposites** — it deliberates internally across several
cheap passes, stays silent (emits nothing) while it's unsure, and commits *on its own* to an
answer, instead of the usual **one forward pass = one answer**.

This repo is a **research log**. Each folder in [`src/`](src/) is one experiment ("iteration"),
and they form a progression — each fixes a specific limitation of the one before. This README is
the **map**: for every iteration it says, in plain words, *what it does*, *what's genuinely new*,
and *whether it's actually useful*.

---

## The whole idea in one breath (ELI5)

Normally an AI reads something and instantly blurts one answer. Here we build a part that can
**hold two opposite ideas at the same time**, **think quietly for a while without saying
anything**, and only speak when it has worked out a **third answer** that combines them — and it
**decides for itself when it's ready** to speak. Instead of guessing the moment it sees a
question, it sits with it, and the harder the question, the longer it sits.

---

## The honest bottom line (read this first)

After all the iterations, here is what is *actually* useful versus what is just a concept demo:

- ✅ **Genuinely useful & measured:** spending **more thinking only when the problem is harder**.
  On four benchmarks it matched a full model using much less compute, and the thinking-time tracks
  difficulty (coin evidence, chain length, sentiment confidence, number of reasoning hops). See
  Stage 5, Bench 1/3/6/7.
- ⚠️ **Useful but not new:** that win is essentially **PonderNet / Adaptive-Computation-Time**, a
  known idea — done carefully and validated, but not our invention. Our *own* named signal (the
  "settling speed" `‖Δh‖`) was **tested and found to do nothing** (Stage 5, Experiments A & B).
- ✅ **The one genuinely-new positive:** the **orthogonal-synthesis geometry** (Stage 6, Part A).
  It builds the answer as a *third thing* perpendicular to the disagreement, and this makes it
  **extrapolate** to situations 3–4× outside its training range where ordinary models fall apart.
  This is the result that is both *new* and *better*.
- ⚠️ **Small niche win:** tying the loss to a **shared "energy budget"** lets a group of operators
  **triage** — stop wasting effort on hopeless sub-problems (Stage 6, Part C′/C″). It beats simple
  baselines only when some sub-problems are genuinely hopeless; otherwise it ties them.
- 🧪 **Concept demos (no practical advantage, and that's fine):** Stages 0–4 exist to *prove the
  phenomenon is real* (thinking has a price; indecision forces a choice; the answer can be a
  synthesis). They win no benchmarks and aren't meant to.
- ❌ **Things we tried that didn't help (kept on purpose):** from-scratch hard tasks (Stage 5,
  Bench 4/5), the energy budget as an *inference-time* control (Stage 6, Part C), and unfreezing
  the readout to close the last gap (Stage 6, Part C‴). Negative results are labeled, not hidden.

---

## The progression — every iteration at a glance

| # | folder | ELI5: what it does | genuinely new | useful? / advantage |
|---|---|---|---|---|
| 0 | [`no-input-tension`](src/no-input-tension/) | A tiny model with **no input** sits on a "tension landscape"; the longer it stays undecided the more the ground tilts, until it's *forced* to tip to one side at a moment you can predict with a formula. | Decision-under-a-deadline as **pure physics** (a pitchfork bifurcation); predicts the snap time (~19.7) and it matches. | Concept demo only. No task, no baseline — proves "indecision can be made to force a choice." |
| 1 | [`input-tension`](src/input-tension/) | An agent watches a biased coin and learns *when* and *which way* to bet; its decision variable **runs away** to a corner once evidence piles up. | Makes the tension idea **trainable on a task** (REINFORCE). | Reaches ~Bayes-optimal — but so does a plain tally. **No advantage**; a coin only needs "leaning," not holding. |
| 2 | [`composable-multi-input-tension`](src/composable-multi-input-tension/) | Two clues (a row clue and a column clue) arrive at *different times*; the operator must **stay silent (output all-zeros) until both arrive**, then snap to the one correct grid cell. | First time **holding is mandatory** and the operator emits a literal **zero vector** while waiting. | Concept win (~0.99). Shows the hold-then-snap mechanism composes; no benchmark advantage yet. |
| 3 | [`true-holding-tension`](src/true-holding-tension/) | Two clues each pin the answer to a *diagonal line*; the answer is the **crossing point** — a cell on neither clue's line. | The answer is a **third thing** (a synthesis); an ablation proves blinding one clue → chance. | Concept win (~0.99). Proves it's real synthesis, not a lookup table. |
| 4 | [`synth`](src/synth/) | Two opposing "votes" cancel out; the answer swings **90° off** to a perpendicular third thing. v2 removes all hints about *when* to stop and **staggers** the clues so it's forced to wait for missing info. | **Owns its own timing** with no clock: it commits when the internal field comes to *rest*, and that moment tracks when the missing clue arrives (corr **0.98**). | The cleanest *concept* result — strong evidence it genuinely holds and self-times. Still no benchmark. |
| 5 | [`tension-block`](src/tension-block/) | Turns it into a **drop-in block** and runs **head-to-head benchmarks** vs normal models: when to stop, adaptive compute inside a Transformer, and real datasets. | A reusable operator + the first **measured wins** and the key insight about *when* it pays off. | **The payoff.** Real compute wins (see sub-table) — but they're the known PonderNet/ACT idea; our own "settling" signal was falsified here. |
| 6 | [`opposition-synthesis`](src/opposition-synthesis/) | Makes "holding opposites" **literal**: opposites are held by a *string*, the answer is what's left **perpendicular** to the disagreement; then explores a shared **energy budget** over many such operators. | The **orthogonal-projection synthesis** (Part A) — the one genuinely new *and* better result — plus a **budget-as-loss** that learns to triage. | **Mixed, and honestly labeled.** Part A extrapolates where baselines can't (real advantage). The budget only helps in a narrow "hopeless-distractor" case. |

---

## Stage 5 in detail — `tension-block` (the benchmarks)

This is where "is it actually useful?" gets *measured* against real baselines.

| iteration | ELI5: what it does | genuinely new | useful? / advantage |
|---|---|---|---|
| **Bench 1** coin/SPRT | Bet on a coin's bias; decide how many flips to watch first. | Stops adaptively *without being told* the coin's difficulty. | ✅ **Real win:** beats the *optimal* fixed-number-of-flips rule by **+2 to +6 pts**; extrapolates its timing to unseen difficulties. The headline. |
| **Bench 2** parity | Decide if a variable-length sequence has even/odd ones; think longer for longer ones. | Adaptive depth on a sequence task. | ✅ Modest win: 100% at ~**19% less compute**; think-time rises with length. |
| **Bench 3** pointer-chasing | Follow a chain of "go to the next box"; longer chains need more steps. | Adaptive compute as a **Transformer layer**, per token. | ✅ **Clean win:** matches a full Transformer at **66% less compute**; steps tracks chain length (corr **0.95**). Composes at scale. |
| **Bench 4/5** ListOps / bAbI (from scratch) | Hard reasoning tasks with a small model trained from zero. | — | ❌ **Negative (kept):** it just commits instantly. **Insight:** adaptive compute only helps if the base model *can* turn extra compute into accuracy. This insight is what makes 6 & 7 work. |
| **Bench 6** DistilBERT/SST-2 | Real sentiment task on a pretrained model; exit early when sure. | Early-exit on a strong backbone, real data. | ✅ **Real-data win:** full accuracy at ~half the layers; **+3.8 pts** over the best fixed-depth at matched compute. |
| **Bench 7** bAbI multi-hop | Real QA needing 1, 2, or 3 chained facts; spend more layers for more hops. | Difficulty = a *labeled reasoning-hop count*. | ✅ **Real-data win:** exit depth rises with hops (corr **0.778**), full accuracy at ~78% of the depth. |
| **Exp A & B** ablations | Remove our special "settling speed" signal and see if anything changes. | A fair, null-seeking test of *our own* idea. | ❌ **Falsified (twice):** the settling signal adds ~0. The real engine is **learned halting (PonderNet/ACT)**, a known method. Honest, important. |

**Verdict for Stage 5:** the adaptive-compute wins are real and on real data — but they are a
*known* technique executed well, not a new mechanism.

---

## Stage 6 in detail — `opposition-synthesis` (the newest work)

| iteration | ELI5: what it does | genuinely new | useful? / advantage |
|---|---|---|---|
| **Part A** `synth_opposites` | Opposite inputs pull on a "string"; the answer is what survives **perpendicular** to the pull (they cancel, a third thing is left). | A real **orthogonal-projection** synthesis with a *real* tension signal `‖p_a−p_b‖`. | ✅ **The genuinely-new win.** Stays accurate (drop **−2.3**) when tested 3–4× outside its training range, where black-box baselines drop ~9 and collapse. A structural ability they lack. |
| **Part B** `tension_synth_operator` | Wraps Part A in the full contract: hold (emit zero), deliberate over steps, commit when settled, remember state between calls. | Part A made to genuinely *hold and self-commit*. | 🧪 Concept: confirms the geometry works inside a hold/deliberate/commit loop. |
| **Part C** `energy_budget` | Many operators share one pool of "mental energy"; a rising "price" pushes everyone to commit as it drains. | A shared, dynamic commit pressure (vs a fixed per-operator penalty). | ❌ **Inert as a control:** as an *inference-time* knob it just ties simple uniform allocation. Only buys *compute-robustness*, not accuracy. |
| `compare_baselines` | The measuring stick: pits everything against **known algorithms** (fixed-N, SPRT/DeeBERT early-exit, value-of-computation, an oracle ceiling). | — (it's the honest test harness) | ✅ Useful tool. Showed the budget-as-control did **not** beat standard methods. |
| **Part C′** `budget_trained` | Same budget, but now the **loss itself** rewards being right *and* using little shared energy. | **Budget-as-loss** → operators *learn to triage*. | ✅ **Small real win:** beats uniform by **+3** and crushes confidence-exit in the "some sub-problems are hopeless" case. Ties everything in the easy case. |
| **Part C″** `budget_dynamic` | The budget is **elastic**: an operator can *request* more energy when it needs it, but requesting hurts (convex "discomfort") and is capped. | A self-sizing budget (start small, grow on demand). | ✅ Works as designed: reaches the same frontier *adaptively from a tiny base* — but **matches**, doesn't beat, Part C′. |
| **Part C‴** `budget_jointreadout` | Stop freezing the answer-reader; train it together with the budget to try to close the last gap. | — | ❌ **No gain (kept):** the leftover gap is *irreducible noise* + an unfair label-peeking oracle, not something more training can fix. Settles the question. |

**Verdict for Stage 6:** Part A is the real new advantage. The budget line of work yields one
narrow, honest win (triage when sub-problems are hopeless) and a clear ceiling explanation.

---

## What to do next (the honest pointer)

The budget arc hit its ceiling because the toy task has **near-uniform value-of-compute** across
sub-problems — so there's little to gain by reallocating effort. The next experiment that could
produce a *new* win is a task with **genuine triage structure** (sub-problems that differ sharply
in how much an extra step helps, e.g. "confidently wrong fast" vs "slow but solvable"), plus
pushing **Part A's extrapolation** into a non-synthetic setting. See [`TODO.md`](TODO.md).

---

## Running

Each folder is self-contained with its own README and run command. Stages 0–4 and the synthetic
parts of 6 need only `torch` (+ `matplotlib` for figures) and run on CPU in seconds to minutes.
Stage 5's real-data benchmarks (6 & 7) also need `transformers` + `datasets`.

> **Hardware note:** this dev machine **hard-powers-off under heavy GPU load** (a power-delivery
> fault, not thermal — see the GPU benchmarks' notes). The real-data runs are **checkpoint/
> resumable** and were completed on **CPU**. Prefer CPU here unless the GPU power issue is fixed.

## Data / disk

Trained model blobs and datasets are **gitignored** and regenerable. They are large:
`src/tension-block/runs/` holds ~1.5 GB of completed-run checkpoints (`glue_sst2.pt`,
`babi_qa123.pt`); `src/tension-block/data/` is ~112 MB of bAbI text. Delete `runs/*.pt` to
reclaim space — the results are already recorded in the READMEs and re-training reproduces them.

## A note on honesty

Negative results are kept and labeled. They produced the key insights — *adaptive computation
only pays when the base model can already convert compute into accuracy* (Stage 5), and *a shared
budget only pays when sub-problems differ in value-of-compute* (Stage 6). Every "useful?" claim
above is measured against conventional baselines, not asserted.

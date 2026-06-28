# Tension

An operator that **holds the tension of opposites** — deliberates internally across passes,
holds (emits nothing) while unresolved, and commits *on its own* to a synthesis or a side —
instead of the usual **one forward pass = one output**. 

This repo is a **research log**: each directory in [`src/`](src/) is one experiment, and they
form a progression from a pure dynamical-systems seed to a composable operator that wins on a
real benchmark. Read them in order — each one fixes a specific limitation of the last.

## The progression

| # | directory | what it establishes | honest status |
|---|---|---|---|
| 0 | [`no-input-tension`](src/no-input-tension/) | The **seed**: an inputless model on a *tension landscape* — committing or staying balanced is cheap, being torn is costly, and a rising *thinking-cost* tilts the landscape until a **pitchfork bifurcation** forces a commit at a predicted time. Decision-under-a-deadline as pure dynamics. | works; the math predicts when it snaps |
| 1 | [`input-tension`](src/input-tension/) | Makes it **trainable on a task**: an RL agent watches a biased coin and learns *when* and *which way* to bet via a **runaway accumulator** (REINFORCE). Reaches ~Bayes-optimal. | works; but a coin has a sufficient statistic, so it only *leans* — holding isn't yet necessary |
| 2 | [`composable-multi-input-tension`](src/composable-multi-input-tension/) | Makes **holding mandatory**: two evidence streams arrive at different times, so committing early is structurally wrong. The operator emits the **zero vector** until both reconcile, then snaps. First composable operator (latent + settledness latch). | works (≈0.99); commit still nudged by a clock |
| 3 | [`true-holding-tension`](src/true-holding-tension/) | The **third thing**: answer = the intersection of two diagonals — a cell on *neither* input's axis. PonderNet halting; an ablation proves it's a **synthesis, not a stats table**. | works (≈0.99); answer knowable early, timing still scaffolded |
| 4 | [`synth`](src/synth/) | **Intrinsic timing + forced holding**: balanced evidence resolves *off-axis* to the perpendicular (a third thing interpolation can't reach); v2 removes the clock (commit = the field coming to **rest**) and staggers evidence so holding is forced by *missing information*. | clean: commit step tracks information arrival (corr 0.98) |
| 5 | [`tension-block`](src/tension-block/) | The **composable operator + head-to-head benchmarks** vs conventional models: when-to-commit (beats SPRT-class fixed budgets), adaptive compute as a Transformer layer, and the honest base-model wall — culminating in a **real-data win** (early-exit on a pretrained DistilBERT, SST-2). | the payoff; see its README for the full scorecard |

## The arc in one breath

Stage 0 shows the *phenomenon* (thinking has a price; indecision forces a choice) with no
data. Stage 1 makes it *trainable* but reveals that a task with a running tally only teaches
*leaning*. Stages 2–3 build tasks where **holding is mandatory** and the answer is a
**synthesis**, proving the operator does more than statistics. Stage 4 removes the last
scaffolds — the operator **owns its commit timing** and holds when information is genuinely
missing. Stage 5 turns it into a **drop-in block** and asks the only question that matters —
*is it genuinely useful?* — answering with measured wins, honest negatives, and the precondition
that decides when it pays off.

## Running

Each directory is self-contained with its own README and run instructions. The early stages
(0–4) need only `torch` (+ `matplotlib` for figures) and run on CPU/GPU in seconds to minutes.
Stage 5's real-data benchmark additionally uses `transformers` + `datasets`. See each README.

## A note on honesty

Negative results are kept and labeled (e.g. ListOps / from-scratch bAbI in `tension-block`):
they are what produced the key insight — *adaptive computation only pays when the base model
can already convert compute into accuracy* — which is exactly what the final real-data win
relies on. Claims here are measured against conventional baselines, not asserted.

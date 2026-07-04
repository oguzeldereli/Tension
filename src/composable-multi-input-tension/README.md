# tension-operator

> **ELI5:** two clues (one about the row, one about the column of a grid) arrive at *different
> times*; the operator must **say nothing (output all zeros) until both have arrived**, then snap
> to the one correct cell.
> **Genuinely new:** the first time **holding is mandatory** — committing early is structurally
> wrong — and the operator emits a literal **zero vector** while it waits.
> **Useful? / advantage:** Concept win (~0.99). Shows the hold-then-snap mechanism works and would
> compose inside a bigger network; no benchmark advantage yet. (Stage 2.)

A deliberation operator that breaks the assumption **one forward pass = one output**. It
holds an internal tension across many forward passes -- emitting the **zero vector**
(nothing) while it deliberates internally -- and then **snaps** to a single clean symbol
once the tension resolves. Inspired by Jung's *holding the tension of opposites*: the
resolution is a synthesis, not a tally-driven pick.

## Why this exists (and why a coin task can't show it)

A confidence accumulator (lean a little more toward the favored side each step, commit
when confident) is just sequential statistics. It never *holds* anything -- holding would
be suboptimal -- so on a task with a sufficient statistic (a coin), training drives the
"tension" out: the system just leans. To make held tension *necessary*, the task must be
one where **premature commitment is structurally wrong** and there is **no running tally
to lean on**.

## The task: reconciliation (where holding is mandatory)

A 3x3 grid; the answer is a cell `(row*, col*)`. Two streams trickle noisy hints -- one
about the **row**, one about the **column** -- and they arrive at different times (early
hints favor one stream). So for a long stretch the operator knows the row but *not* the
column. Committing then is wrong: any cell is a guess on the unknown coordinate. The only
correct behavior is to **hold (emit zero) until both streams reconcile**, then snap to the
joint cell -- a synthesis that is neither "a row" nor "a column", with no tally to lean on.

## The operator

Carries a latent state `z` across passes. Each pass it integrates the current evidence
into `z` (the deliberation -- refined internally, emitting nothing). A **settledness**
signal measures how stationary `z` is: while the competing pulls are unreconciled `z`
keeps moving (unsettled); when they reconcile into a stable point `z` stops (settled). A
**latch** (monotonic, straight-through) flips when settledness crosses a threshold (or the
clock hits the deadline), and only then is `z` decoded to a one-hot symbol. Output:

```
output = latch * one_hot(decode(z))     # ZERO vector while holding; a symbol when committed
```

The zero vector is the honest "holding" output: dropped into a larger network, zero
propagates as zero -- the operator passes no signal downstream until it has resolved. The
clock just tells it time is passing.

## Training

Pure backprop. The continuous deliberation (the `z` dynamics, unrolled across passes) and
the decoder are differentiable -- this is where the operator *learns to integrate evidence
and reconcile*. The discrete latch reads off the (continuous, differentiable) settledness,
so the non-differentiable part is reduced to a threshold crossing. Loss is cross-entropy
of the symbol decoded *at commit* against the true cell. No hold-cost is needed: holding
is free, only being right matters, so the operator learns to hold until reconciled.

## Result

```
final accuracy 0.989   (chance 0.111)   commits at the deadline, after both streams arrive
```

`figures/hold_then_snap.png` is the key picture: output magnitude sits at **0** for the
whole episode (holding the tension, deliberating in `z`, emitting nothing), then **snaps
to 1** in a single pass -- a clean one-hot, no glide, no hedge, no leaning. And it is
*correct* because it waited until row and column reconciled. On a task that needs held
tension, the operator learns to hold it.

## Honest notes / what to change later

- **Settling at the deadline.** With holding free, it learns to use all the time (commit
  at the deadline). Adding a small, well-tuned hold-cost makes it commit *as soon as*
  reconciled (earlier when evidence is clean) -- but it's a delicate speed/accuracy knob:
  too much and it commits prematurely and accuracy collapses (seen during tuning). Free
  holding is the stable first version.
- **Latch timing** is trained via straight-through on settledness, which works here
  because settledness is a smooth function of the (bounded, settling) `z` dynamics. For a
  runaway/unbounded latent this would be harder; the contractive settle dynamics are what
  make backprop-through-deliberation well-behaved.
- **Genuine "opposites in conflict".** This task is synthesis-of-orthogonal-partials
  (row + column). A richer version has the two streams genuinely *conflict* and the
  synthesis be a third thing reconciling them -- the next task to build.

## Files
```
tension_operator.py   # the operator: latent deliberation, zero-vector hold, snap-to-symbol
reconcile_demo.py     # the reconciliation task + training + plots
```

## Run
```bash
pip install torch matplotlib
python reconcile_demo.py
```

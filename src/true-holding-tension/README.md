# tension

> **ELI5:** two clues each narrow the answer to a *diagonal line* on a grid; the real answer is
> where the two lines **cross** — a cell that sits on *neither* clue's line.
> **Genuinely new:** the answer is a **third thing** (a synthesis, not a pick); an ablation proves
> it — blind one clue and accuracy collapses to chance.
> **Useful? / advantage:** Concept win (~0.99). Proves the operator computes a real synthesis, not
> a memorized lookup table; still no benchmark. (Stage 3.)

A **TensionOperator**: a small recurrent unit that deliberates *across* forward
passes without emitting anything, holds an internal tension, and emits a single
symbol only when it resolves. The point is to break the assumption that one forward
pass = one output — deliberation lives in the operator's latent state across cheap
parallel passes, not in a growing context window.

## The idea

- While deliberating it emits the **zero vector** — a literal null ("I have not
  resolved"), not a softmax sitting at 0.5. The latch is binary: hold (zero) or
  commit (a symbol).
- The answer is a **third thing**, not a selection among the inputs. The task is a
  G×G grid where stream A gives noisy evidence about `r+c` and stream B about `r−c`.
  Each stream alone pins the cell to a *diagonal* of G candidates; the cell exists
  only at their intersection, so the operator must synthesize a point on neither
  input's axis. `evaluate.py` includes an ablation that proves it: blind one stream
  and accuracy collapses to chance.
- It is **not a stats table**: holding is productive (it computes the synthesis in
  latent space), and the emitted output is a hard zero→symbol snap, not a confidence
  ramp.

## Training

Expected loss over halting time (PonderNet-style) so gradients flow through a
discrete hold/commit process without REINFORCE. The "never committed" residual mass
carries no loss (Knob B = none), so nothing *forces* a commit.

## The two knobs

- **Knob A — discomfort** (`DISCOMFORT_W`): the price of holding one step. This is
  the only pull toward committing, so it's what makes the thing resolve at all.
  At `0.0` the optimal policy is eternal silence (a real, intended corner — just not
  trainable). Default `0.01`: holds while the cell is ambiguous (committing then
  costs CE), snaps once it's synthesized (CE≈0, so the hold fee tips it over).
- **Knob B — deadline** (`DEADLINE_MODE`): `none` (default) means no deadline,
  holding forever is allowed and unpenalized, and accuracy is measured only over
  episodes that resolve. `soft` / `hard` are the next knob positions (force a commit
  at the horizon / penalize timeout like a wrong answer) — not wired yet.

## Run

```bash
python3 train.py      # writes tension.pt
python3 evaluate.py   # metrics + ablation + per-episode HOLD/COMMIT trace
```

GPU-resident; no DataLoader. Tune `GRID`, `P_SIGNAL`, `MAX_STEPS`, and the two knobs
in `config.py`.

## Files

- `config.py` — hyperparameters and the two knobs
- `task.py` — the sum/difference grid (the synthesis task)
- `model.py` — `TensionOperator` + hard-latch rollout
- `train.py` — expected-loss-over-halting-time training
- `evaluate.py` — hard-latch metrics, synthesis ablation, per-episode trace

# tension-machine

> **ELI5:** a tiny brain with *no input* that can't dither forever — the longer it stays on the
> fence, the more the floor tilts, until it's forced to fall to one side at a moment we predict
> with a formula.
> **Genuinely new:** turns "indecision forces a choice under a deadline" into exact physics (a
> pitchfork bifurcation); predicted snap time ~19.7 matches the observed ~20.9.
> **Useful? / advantage:** Concept demo only — no task, no baseline, beats nothing. It proves the
> *phenomenon* the project is built on. (Stage 0 — see the [project map](../../README.md).)

Inputless, self-driven neural dynamics. No dataset, no input — these models produce
behavior purely from their own internal weights evolving on a **tension landscape**.
The "result" isn't an accuracy number; it's the *dynamics*, which you watch in plots.

This is a small artificial-life / dynamical-systems experiment built around one idea:
a model that is rewarded for **committing to a decision** *or* for **staying perfectly
balanced**, punished for everything in between, and made to pay a rising cost for the
*act of staying undecided* — so that thinking has a price and indecision eventually
forces a choice.

---

## The core object: a tension

A `TensionModel` is inputless — its output is a pure function of its own weights. It
produces a single decision `p = P(output 1)`. The **tension loss** has three valleys:

```
  -cos(4*pi*p):   minima at p = 0 (commit to 2), 0.5 (balanced), 1 (commit to 1)
                  maxima at p = 0.25, 0.75 (torn)
```

So *being decided* and *being perfectly balanced* are both low-cost; *being torn* is
high-cost. "Inference" is just letting the weights roll downhill on this landscape.

## The thinking-cost: a landscape that tilts

Staying balanced is only free *for a while*. Every step the model spends indecisive,
a **thinking-time** `t` accumulates — in proportion to how torn it is
(`indecision(p) = 1 - (2p-1)^2`, which is 1 at p=0.5 and 0 when committed). The loss
adds a penalty `t * indecision(p)`, which **lifts the middle of the landscape**: the
balanced valley gets shallower and shallower while the committed valleys (where
indecision ≈ 0) are untouched.

The clock **decays** when the model is committed, so the balanced valley slowly
regrows — indecision is a renewable, not permanent, option.

## The bifurcation (a derived, verifiable prediction)

Expand the loss near `p = 0.5` (let `u = p - 0.5`):

```
  base      ≈ -1 + 8*pi^2 * u^2          (a valley, curvature +16*pi^2)
  penalty   ≈ t*s*(1 - 4u^2)             (lifts the middle, curvature -8*t*s)
  total curvature at center = 16*pi^2 - 8*t*s
```

The balanced point is a **minimum while** `t*s < 2*pi^2`, and flips into a **hilltop**
(a pitchfork bifurcation) once `t*s > 2*pi^2`. So the system is *forced* to commit at:

```
  t_crit = 2*pi^2 / s  ≈  19.74   (for penalty strength s = 1)
```

This is checkable, and it checks out: running `system2_decide.py` prints the predicted
`t_crit = 19.74` and the observed snap at `t ≈ 20.9` — they match. Which side it falls
to is **spontaneous symmetry breaking**: decided by the tiniest asymmetry (init / noise)
at the critical moment.

---

## The four systems

| file | what it is |
|---|---|
| `system2_decide.py` | **The atom.** Weights self-descend the tilting tension loss. The model hovers in indecision, the thinking-cost ramps, and at `t_crit` it snaps to a decision. One decision, then it rests. |
| `system3_heartbeat.py` | **The heartbeat.** A periodic self-perturbation kicks the committed model back toward balance and resets its clock, so it deliberates → commits → resets → deliberates again: an ongoing rhythm of decisions. |
| `system4_controller.py` | **One model writes another.** A separate `Controller` network reads the target's state `[p, t, indecision]` and writes a delta into the target's weights every step — a hypernetwork / fast-weights coupling, producing autonomous coupled dynamics. |

*(There is no `system1` file — "tension" is the loss/model itself, defined in `tension.py`,
and is the shared core all four systems are built on.)*

---

## Running

```bash
pip install torch matplotlib
python system2_decide.py      # the single decision + the tilting-landscape plot
python system3_heartbeat.py   # the rhythm of repeated decisions
python system4_controller.py  # the controller-writes-target coupled system
```

Figures are written to `figures/`. Everything runs on CPU in seconds (the models are tiny).
All knobs live in `config.py`.

---

## What you'll see, honestly

- **System 2** is the clean result: a flat hover at p=0.5, then a sharp snap at the
  predicted critical time, with the landscape-tilt panel showing the balanced valley
  rising into a hilltop. The math predicts *when* it decides, and it does.

- **System 3** produces a real rhythm, but note a genuine property of the dynamics:
  after the first commitment the system tends to fall to the **same side** on later
  beats unless the kick is strong enough to fully reset it past the bifurcation —
  a mild **hysteresis** (the system has a memory of its prior choice). Turn up
  `KICK_SCALE` / `KICK_NOISE` in `config.py` to make beats independent, or leave it to
  study the hysteresis. Either is interesting; it's a knob, not a bug.

- **System 4** demonstrates the *mechanism* — one network writing another's weights —
  and with a **frozen, randomly-initialized controller** the coupled system settles
  into one of three behaviors depending on `SEED`: a **fixed point** (commits and is
  held there), a **limit cycle** (oscillates), or **wandering**. Change `SEED` in
  `config.py` to explore them.

## The honest limit of System 4 (and the real extension)

The controller here is **not trained** — it's frozen at random init, which proves the
wiring and gives a self-contained autonomous system, but it has no *goal*. The deeper
version is to **train the controller on an intrinsic objective** (no data, no labels) —
for example, reward it for keeping the target at the **edge of decision** (p near a
moving target, never frozen, never chaotic — the "edge of chaos"), or for **novelty**
in the target's trajectory. That requires a differentiable inner loop (backprop through
the controller's write into the target and the resulting `p`) and is its own project.

This is the crux the whole design circles: an inputless system has no external target,
so its objective must be **intrinsic** — staying alive, staying interesting, staying at
the edge — rather than matching data. System 4's frozen controller is the mechanism;
an intrinsic objective is what would make it *purposeful*.

---

## Conceptual note

These systems implement, in order: a **decision under a thinking-deadline** (2), an
**ongoing rhythm** of such decisions (3), and **one network sculpting another's**
landscape (4). The recurring phenomena — spontaneous symmetry breaking, a pitchfork
bifurcation, hysteresis, and (in System 4) attractors that are fixed points / cycles /
chaos depending on seed — are all genuine dynamical-systems behaviors, here arising
from a deliberately tiny, fully inputless setup you can watch move.

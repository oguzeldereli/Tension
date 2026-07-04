# coin-bettor

> **ELI5:** an agent watches a biased coin flip by flip and learns *when* to bet and *which way*;
> its "decision dial" snowballs until it runs away to heads or tails.
> **Genuinely new:** makes the tension idea **trainable on a task** (REINFORCE over a runaway
> accumulator).
> **Useful? / advantage:** Reaches ~Bayes-optimal — but **no advantage**, because a plain running
> tally does just as well. A coin has a sufficient statistic, so it only learns to *lean*, never to
> *hold*. This is exactly why later stages switch to tasks where holding is mandatory. (Stage 1.)

A reinforcement-learning agent that watches a biased coin and learns *when* and *which way*
to bet — where committing is a genuine **runaway collapse**, not a discrete action. This is
the trainable version of the tension idea: a decision that stays indecisive until evidence
and a thinking-cost drive it to run away to a corner.

## The environment

A coin has a hidden bias `theta = P(heads)`, drawn fresh and **unknown** each episode. A new
flip is revealed every `X` turns, but the agent runs **every** turn, so its information grows
over time. The agent sees only the **raw tally** `(heads, tails)` — not `h/n` — so it has to
learn to weigh *which way* (the difference) against *how sure* (the total) itself.

It must commit to "heads-biased" or "tails-biased". There's a hard **deadline** (a point of no
return), but it can commit earlier. A correct bet (matching the true `theta`) earns `+1`, a
wrong one `-1`, and **each waiting turn costs a little** — so indecision is tolerated but
priced, and the agent faces a real speed/accuracy tradeoff: wait for more flips (better bet,
more cost) or commit now (cheaper, riskier).

## The mechanism: runaway collapse

The agent has one decision variable `d` that **accumulates**:

```
d_{t+1} = d_t + g_t,     p = sigmoid(d)
```

`p = 0.5` (d = 0) is indecision / "waiting". Because increments accumulate, under steady
evidence `d` keeps growing and `p` **runs away** toward 0 or 1 — a genuine collapse, an
attractor the system falls into, not a one-shot sampled bet. (An integrator is what makes the
runaway real; a contractive cell like a vanilla GRU would just settle to a fixed point.)
`p = sigmoid(d)` is exactly softmax over two logits whose difference is `d`, so this is the
two-logit / "wait = balanced" mechanism, with `d` the logit difference. Commitment is read off
when `p` crosses `0.99` / `0.01` (literal 1/0 is unreachable by softmax and untrainable, so a
high threshold is the honest stand-in for "collapsed").

## Training: REINFORCE over the trajectory

The network outputs only the **mean** increment `mu` each turn; the actual increment is sampled
`g ~ Normal(mu, sigma)`. The increments are the stochastic **actions**, so we use the
policy-gradient (REINFORCE) estimator and **never backprop through the unrolled dynamics**
(Approach B). This sidesteps the vanishing/exploding-gradient wall that backprop-through-a-
runaway-trajectory (Approach A) would hit — at the cost of the usual policy-gradient variance.

```
loss = -(R - baseline) * sum_t logprob(g_t)
R    = (+1 if committed side == (theta>0.5) else -1)  -  wait_cost * turns_waited
```

## Results (honest)

Benchmark = betting the running majority after all flips (Bayes-optimal given the data) ≈ **0.90**.

- The agent learns from chance (~0.50) to **~0.89–0.93 accuracy**, essentially matching the
  benchmark. It learns to bet the right way *and* (with a sensibly small wait-cost) to wait for
  almost all the flips before collapsing — discovering the gather-then-commit strategy on its own.
- The **runaway is real and visible** (`figures/runaway.png`): `p` starts near 0.5 and integrates
  toward 0.99 / 0.01, the corner chosen by the evidence.
- **REINFORCE variance is real too** (`figures/learning.png`): training occasionally lurches —
  the policy gets knocked into committing too early, accuracy craters, then it recovers. The two
  accuracy crashes line up exactly with dips in "flips seen before commit," which is the
  speed/accuracy coupling showing through. This wobble is the nature of policy gradients
  (the price of Approach B), not a bug.

### The speed/accuracy knob
`WAIT_COST` is the key dial. Too high (e.g. 0.01 here) and the agent commits after ~3 flips,
sacrificing accuracy to dodge the cost. Lower (0.003) and it waits for the evidence and reaches
the benchmark. The *interesting regime* — deliberate, then collapse correctly — lives in that band.

## Known limitations / what to change next

- **Timing credit-assignment is weak.** Reward shapes *direction* strongly (correct vs wrong) but
  *when* to commit only indirectly (via the wait-cost inside R). A cleaner version would make the
  collapse itself a sampled per-turn event so timing gets a direct policy-gradient signal.
- **High variance.** A learned value baseline (actor-critic) instead of the batch-mean baseline
  would steady the curve a lot.
- **Reward is vs. true `theta`.** Near-fair coins (theta≈0.5) are genuinely ~unlearnable, which
  caps accuracy at the benchmark (which has the same cap) — a fair comparison, but worth knowing.

## Files
```
config.py   # all knobs (X, deadline, wait-cost, sigma, commit threshold, ...)
policy.py   # the runaway accumulator (mean-increment network)
train.py    # vectorized REINFORCE rollout, benchmark, learning + runaway plots
```

## Run
```bash
pip install torch matplotlib
python train.py        # prints learning, writes figures/learning.png and figures/runaway.png
```
Runs on CPU in a couple of minutes. Tune `WAIT_COST` in `config.py` to move along the
speed/accuracy tradeoff; `figures/runaway.png` shows the collapse you built.

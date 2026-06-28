"""
Knobs for the coin-bettor.

The agent watches a biased coin (bias theta = P(heads), fresh & unknown each episode),
sees a new flip every X turns, and runs every turn. Its single decision variable d
accumulates -- a genuine runaway: under steady evidence d keeps growing and p=sigmoid(d)
runs toward a corner. p~=0.5 is "waiting"; p crossing COMMIT_P is the collapse/commit.
Waiting costs a little each turn (the thinking-cost), so it must eventually commit; the
deadline forces a commit if it never does. Reward: did the committed side match the
true bias. Trained by REINFORCE over the trajectory (no backprop through the dynamics).
"""
SEED = 12

# --- environment ---
X_TURNS_PER_FLIP = 3                       # a new flip is revealed every X turns
MAX_FLIPS        = 10                       # flips available before the deadline
DEADLINE         = X_TURNS_PER_FLIP * MAX_FLIPS   # hard point-of-no-return (in turns)
INPUT_SCALE      = float(MAX_FLIPS)        # scale raw counts into ~[0,1] for the net

# --- policy (the runaway accumulator) ---
HIDDEN   = 32
MAX_STEP = 1.0       # max |mean increment| to d per turn (bounds how fast it can run away)
SIGMA    = 0.3       # exploration noise on the per-turn increment (the sampled action)
COMMIT_P = 0.9      # p past this (or below 1-this) counts as a collapse/commitment

# --- reward ---
WAIT_COST = -1     # cost per waiting turn (max ~0.44 over the episode, < 1 so correctness dominates)

# --- training (REINFORCE) ---
BATCH   = 256
UPDATES = 4000
LR      = 5e-4

FIG_DIR = "figures"

STRENGTH = 4    # coefficient on the time-indecision penalty (tension loss)
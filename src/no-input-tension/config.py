"""
All knobs for the tension-machine, in one place.

The numbers here are tuned so the dynamics are *visible*: the system hovers in
indecision, the cost of thinking ramps up, and at a predictable critical time the
balanced state stops being stable and the system snaps to a decision.
"""
import math

SEED = 3263452

# --- the (inputless) tension model ---
LATENT_DIM = 8
HIDDEN     = 16

# --- the tension loss ---
PENALTY_STRENGTH = 1.0                       # s: coefficient on the time-indecision penalty
T_CRIT = 2 * math.pi**2 / PENALTY_STRENGTH   # balanced valley -> hilltop bifurcation (derived; see README)

# --- "thinking" dynamics (System 2) ---
DT     = 0.05        # how much 'thinking time' one fully-indecisive step costs
DECAY  = 0.999       # t leaks away each step (lets the indecision valley slowly regrow)
LR     = 0.05        # gradient-descent step size on the model's OWN weights
NOISE  = 0.003        # tiny weight noise -> breaks symmetry so it can fall off the hilltop
STEPS_DECIDE = 1200

# Sanity on the ramp: the steady-state t while fully indecisive is DT/(1-DECAY).
# It must exceed T_CRIT or the system can never reach the bifurcation.
#   DT/(1-DECAY) = 0.05/0.001 = 50  >  T_CRIT ~= 19.74   OK

# --- the heartbeat (System 3): kick it back into indecision periodically ---
HEART_DT     = 0.3          # faster ramp so each beat deliberates within one period
HEART_DECAY  = 0.99
PERTURB_PERIOD = 500        # every N steps, knock it back toward the balanced state
KICK_SCALE   = 0.3          # shrink output weights toward 0 (logits -> ~equal -> p -> ~0.5)
KICK_NOISE   = 0.3          # randomness in the kick -> next decision can break either way
STEPS_HEARTBEAT = 3000

# --- the controller (System 4): one model writes another's weights ---
CTRL_HIDDEN      = 16
CTRL_DELTA_SCALE = 0.05     # max weight-change the controller writes into the target per step
STEPS_CONTROLLER = 3000

FIG_DIR = "figures"

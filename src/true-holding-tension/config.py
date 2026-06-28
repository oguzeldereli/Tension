import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- task ----
GRID      = 4      # G x G grid -> N_SYM = G*G cells. The cell is the "third thing".
P_SIGNAL  = 0.6    # per-step hint accuracy. <1 so a single pass is ambiguous and
                   # deliberation (integrating over steps) actually pays off.

# ---- operator ----
HIDDEN    = 128    # latent z dimension (the held tension lives here)
INPUT_DIM = 64

# ---- unroll ----
MAX_STEPS = 20     # compute horizon for the unroll. With Knob B = "none" this is NOT
                   # a deadline: not committing by MAX_STEPS is unpenalized, it just
                   # ends the simulation. It is a compute cap, not a semantic deadline.

# ---- training ----
BATCH = 512
STEPS = 4000
LR    = 1e-3

# Decoder supervision: train the latent readout toward the answer at every step,
# INDEPENDENT of the latch. This decouples "can it synthesize the cell" from "when
# does it speak" -- without it the decoder only gets gradient where the latch fires,
# so when the latch collapses to silence the decoder never learns, committing stays
# expensive forever, and silence wins. This term keeps the synthesis pathway training
# regardless of the commit decision. The answer still never leaks into the emitted
# output (that's zero while holding); this only shapes the internal readout.
W_DECODE = 1.0

# ================= THE TWO KNOBS =================
# Knob A -- discomfort: the price of holding for one step (price of compute).
#   This is the ONLY thing that makes committing worthwhile, so it is what breaks
#   the eternal-silence degeneracy. At exactly 0.0 the optimal policy is to never
#   commit (hold forever) -- a real, intended corner, just not a trainable one.
DISCOMFORT_W = 0.05

# Knob B -- deadline mode: "none" / "soft" / "hard". Only "none" is wired here.
#   none : no deadline. Holding forever is allowed and unpenalized. Accuracy is
#          measured only over episodes that actually resolve.
#   soft : (future) commit is forced at MAX_STEPS but reaching it isn't punished.
#   hard : (future) reaching MAX_STEPS without committing is penalized like a wrong
#          answer (this is the harsh corner that needs a curriculum to train).
DEADLINE_MODE = "none"

# inference: hard-latch threshold. Hold (emit zero) until commit prob crosses TAU.
TAU = 0.5
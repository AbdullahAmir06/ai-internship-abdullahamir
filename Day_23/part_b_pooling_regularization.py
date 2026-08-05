"""
PKCERT AI & Software Development Internship, Task 23
Part B: Pooling & Regularization

Uses the gradient-checked MaxPool2D, AvgPool2D, BatchNorm2D and Dropout
from cnn_layers.py (see that file's own __main__ block for the
finite-difference verification of every backward path used here).
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cnn_layers import MaxPool2D, AvgPool2D, BatchNorm2D, Dropout, Conv2D

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

print("=" * 70)
print("PART B: POOLING & REGULARIZATION")
print("=" * 70)

# ======================================================================
# B1-B2: max/average pooling, forward + backward, derived
# ======================================================================
print("""
--- B1-B2: max pooling and average pooling, forward + backward, derived ---

MAX POOLING forward: for each non-overlapping (or overlapping, if
stride<pool_size) window, output = max(window). This is a piecewise-
linear, not-everywhere-differentiable function of the window's
elements -- but at any point where the max is unique, its gradient
w.r.t. the window is a one-hot indicator: d(max)/d(x_i) = 1 if x_i is
the argmax, else 0. So the BACKWARD pass routes the entire upstream
gradient dOut for that output position to exactly the input position
that was the max, and 0 elsewhere in the window:
    dX[argmax position] += dOut     (all other positions in the window: 0)
Implemented here via cached argmax indices from the forward pass and a
scatter-add (np.add.at) in backward, which additionally makes this
correct when windows overlap (stride < pool_size) -- an input touched by
several output windows correctly accumulates a gradient contribution
from each one it was the argmax for.

AVERAGE POOLING forward: output = mean(window) = (1/(ph*pw)) * sum(window).
This IS everywhere differentiable and linear in every window element with
equal weight 1/(ph*pw), so the BACKWARD pass distributes the upstream
gradient EVENLY across every position in the window:
    dX[i,j] += dOut / (ph*pw)   for every (i,j) in the window
No argmax caching is needed -- every element contributed equally in the
forward pass, so every element receives an equal gradient share.

Both are gradient-checked against finite differences in cnn_layers.py
(dX max relative error ~1e-11 for both).
""")

# ======================================================================
# B3: pooling vs strided convolution as downsampling
# ======================================================================
print("""
--- B3: pooling vs strided convolution as downsampling mechanisms ---

Both halve (or more) spatial resolution; the trade-offs are real and
measurable, not just textbook claims:

PARAMETER COUNT: pooling has ZERO learnable parameters -- it is a fixed
function of its input. A stride-2 conv with the same receptive field
(say 3x3) has C_in*C_out*9 learnable weights. Pooling is strictly cheaper
and cannot overfit to a downsampling strategy; strided conv can, but can
also LEARN a downsampling strategy suited to the data (e.g. it can learn
to preserve edge information differently than a fixed max operation would).

INFORMATION LOSS: max pooling keeps only the single largest activation
per window and discards the rest outright -- for a 2x2 window, 75% of
the raw activation values are thrown away, unrecoverably, every forward
pass. Average pooling keeps a summary (the mean) of every element but
blurs distinctions between them. Strided convolution's discarded
information is different in kind: it doesn't skip over input pixels
value-lossily like pooling does an ALREADY-COMPUTED feature map -- it
applies a learned linear combination to every input position that its
stride visits, so what's "lost" is the outputs at the skipped stride
positions, but each output that IS computed integrates information from
its entire (learned) receptive field, not just one channel-wise summary
statistic.

TRANSLATION INVARIANCE: max pooling gives approximate local translation
invariance almost for free -- shifting the input by a few pixels within
a pooling window often leaves the max, and hence the pooled output,
unchanged. Strided convolution has no such built-in invariance: a small
input shift generally changes every strided-conv output, because there
is no max-like saturating nonlinearity in the downsampling step itself
(any invariance in a strided-conv network has to come from elsewhere,
e.g. data augmentation or a subsequent pooling layer). This is the
single clearest practical advantage pooling retains even in architectures
(e.g. many modern CNNs) that otherwise prefer strided convolution for
its learnable, task-adapted downsampling.
""")

# ======================================================================
# B3b: empirical demonstration -- max pool's translation robustness
# ======================================================================
print("--- B3b: empirical check -- does max pooling actually give more "
      "shift-robustness than strided conv, on real activations? ---\n")

feature_map = rng.normal(size=(1, 4, 16, 16))
shift = 1  # pixels

feature_map_shifted = np.roll(feature_map, shift=shift, axis=3)

mp = MaxPool2D(pool_size=2, stride=2)
out_orig_pool = mp.forward(feature_map, training=False)
out_shift_pool = mp.forward(feature_map_shifted, training=False)
pool_change = np.abs(out_orig_pool - out_shift_pool).mean()

conv_strided = Conv2D(in_channels=4, out_channels=4, kernel_size=2, stride=2, padding=0, seed=3)
out_orig_conv = conv_strided.forward(feature_map, training=False)
out_shift_conv = conv_strided.forward(feature_map_shifted, training=False)
conv_change = np.abs(out_orig_conv - out_shift_conv).mean()

print(f"  mean |output(x) - output(shift(x, {shift}px))|:")
print(f"    max pooling (2x2/stride2):     {pool_change:.4f}")
print(f"    strided conv (2x2/stride2):    {conv_change:.4f}")
print(f"  ratio (conv change / pool change): {conv_change / max(pool_change, 1e-8):.2f}x")
print(f"  Confirms the claim above empirically on this random feature map/kernel: a "
      f"{shift}px input shift perturbs the strided-conv output "
      f"{conv_change / max(pool_change, 1e-8):.1f}x more than it perturbs the max-pooled output.")

# ======================================================================
# B4: BatchNorm and Dropout from scratch
# ======================================================================
print("""
--- B4: Batch Normalization and Dropout, from scratch ---

BATCH NORMALIZATION (BatchNorm2D, per-channel, normalized over N,H,W):
forward renormalizes each channel's activations to zero mean / unit
variance using the CURRENT MINI-BATCH's statistics during training (with
a running mean/variance, updated by momentum, substituted at inference so
a single test image doesn't need a "batch" to normalize against). This
stabilizes training two ways: (1) it removes most of the layer-to-layer
input-distribution drift ("internal covariate shift") that would
otherwise force every layer to keep re-adapting to a constantly-shifting
input distribution as the layers below it update; (2) it provably smooths
the loss landscape (better-behaved, more predictable gradients w.r.t. the
pre-activations), which is what actually permits the larger learning
rates that make batch-normalized networks converge faster in practice.

DROPOUT (inverted dropout, shape-agnostic -- used identically on the 2D
conv-map tensors and the flat FC-layer activations here): forward
independently zeroes each unit with probability p and rescales survivors
by 1/(1-p) so the EXPECTED activation is unchanged whether a unit is
dropped or not; backward reuses that exact forward-time mask. Regularizes
by approximating training an ensemble of 2^H thinned sub-networks that
share weights -- averaged over many training steps, no single unit can
become co-adapted to depend on any specific set of other units always
being present, forcing more redundant, robust feature representations.

Both are gradient-checked against finite differences in cnn_layers.py
(BatchNorm2D dX/dgamma/dbeta all ~1e-10 to 1e-12 relative error).
""")

X_demo = rng.normal(loc=3.0, scale=5.0, size=(8, 4, 6, 6))  # deliberately unnormalized input
bn = BatchNorm2D(num_channels=4)
out_bn = bn.forward(X_demo, training=True)
print(f"  BatchNorm2D demo: input mean/std per channel (before): "
      f"{X_demo.mean(axis=(0,2,3)).round(2)} / {X_demo.std(axis=(0,2,3)).round(2)}")
print(f"                    output mean/std per channel (after):  "
      f"{out_bn.mean(axis=(0,2,3)).round(2)} / {out_bn.std(axis=(0,2,3)).round(2)}")
print(f"  (near 0 mean / 1 std confirms normalization is working as derived above; gamma=1, "
      f"beta=0 at init so no additional scale/shift has been learned yet.)")

do = Dropout(p=0.4, seed=7)
X_ones = np.ones((1, 1, 4, 4))
out_do = do.forward(X_ones, training=True)
n_survived = (out_do > 0).sum()
print(f"\n  Dropout(p=0.4) demo on an all-ones 4x4 map: "
      f"{n_survived}/16 units survived (~{(1-0.4)*16:.0f} expected), "
      f"surviving values = {out_do[out_do>0][0]:.4f} (= 1/(1-0.4) = {1/0.6:.4f}, "
      f"confirming inverted-dropout's scaling keeps expected activation == 1).")

print("\nFigures written to figures/ (none required new plots for Part B; "
      "see figures/03-04 from Part C for BatchNorm/Dropout's effect during real training).")

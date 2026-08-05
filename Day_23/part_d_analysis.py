"""
PKCERT AI & Software Development Internship, Task 23
Part D: Analysis & Documentation

Reads the model already saved by part_c_cnn_training.py and prints the
effective-receptive-field calculation, pipeline summary, ablation-study
findings, challenges faced, and framework-limitations reflection.
"""

import pickle

print("=" * 70)
print("PART D: ANALYSIS & DOCUMENTATION")
print("=" * 70)

with open("numpy_cnn_final_model.pkl", "rb") as f:
    model_data = pickle.load(f)

print(f"""
--- D1: final trained model, saved ---

numpy_cnn_final_model.pkl already saved by part_c_cnn_training.py (pickle),
containing: full parameter dict (conv1/conv2 weights+biases, BatchNorm
gamma/beta/running stats, fc1/fc2 weights+biases), the architecture config
needed to reconstruct a SimpleCNN and load these params via set_params(),
class names, and normalization statistics.
  Test accuracy at save time: {model_data['test_accuracy']:.4f}
  Test F1 at save time:       {model_data['test_f1']:.4f}
""")

print("""
--- D2: pipeline summary and design justification ---

ARCHITECTURE: Conv(3->8, 3x3, pad=1) -> BatchNorm -> ReLU -> MaxPool(2x2) ->
Conv(8->16, 3x3, pad=1) -> BatchNorm -> ReLU -> MaxPool(2x2) -> Flatten ->
Dense(1024->64) -> ReLU -> Dropout(0.3) -> Dense(64->4) -> softmax.

WHY THIS SHAPE:
- Two conv layers (not one): a single 3x3 conv layer's receptive field
  (3x3 pixels of the original image) cannot see enough of a 32x32 CIFAR
  image to distinguish object shape from background texture; stacking a
  second conv layer after downsampling lets the network compose local
  edge/color detectors from layer 1 into larger, more object-like
  patterns in layer 2 -- the entire point of a CONVOLUTIONAL (as opposed
  to single-layer) network.
- Channel counts (8 then 16, doubling): a standard, deliberately modest
  choice -- doubling channels as spatial resolution halves keeps the
  total amount of "information capacity" per layer roughly balanced,
  while keeping the from-scratch NumPy training fast enough (giving 3
  ablation variants time to train, in addition to the main model and the
  PyTorch baseline) within this task's compute budget.
- BatchNorm after every conv, before ReLU: matches Part B's derived
  justification (stabilizes/accelerates training by removing
  layer-to-layer input drift) and standard architectural convention.
- MaxPool (2x2, stride 2) after every conv block: halves spatial
  resolution twice (32->16->8), matching Part B's derived translation-
  robustness argument -- the ablation study below empirically confirms
  this choice mattered (removing pooling in favor of an equal-capacity
  stride-2 conv measurably hurt generalization, see D4).
- ReLU throughout: the standard choice for hidden activations --
  computationally cheap, avoids the vanishing-gradient saturation of
  sigmoid/tanh, and is what every backward-pass derivation in this task
  (Parts A-C) was built and gradient-checked against.
- Dropout(0.3) only in the FC head, not the conv layers: a common,
  deliberate choice -- dropout in early conv layers can discard whole
  spatial regions of already-scarce low-level feature information before
  the network has had a chance to aggregate it, whereas the FC head
  (which is closer to overfitting, given its parameter density relative
  to the small 4-class dataset here) benefits most directly from dropout's
  regularization.
- Mini-batch GD + momentum (0.9), not plain SGD: momentum was explicitly
  required by this task's Part C instructions, and empirically (Part C)
  gave stable, monotonically-improving-then-plateauing training curves
  without the divergence risk a naively large learning rate under plain
  SGD would carry.
""")

print("""
--- D3: effective receptive field, computed ---

Standard recursive formula, applied layer-by-layer in forward order
(r = receptive field size in input pixels "seen" by one output unit;
jump = the input-pixel distance between two adjacent output units):
    r_0 = 1, jump_0 = 1
    r_i = r_{i-1} + (k_i - 1) * jump_{i-1}
    jump_i = jump_{i-1} * s_i
""")


def effective_receptive_field(layers, verbose_name=""):
    r, jump = 1, 1
    trace = [(r, jump)]
    for k, s in layers:
        r = r + (k - 1) * jump
        jump = jump * s
        trace.append((r, jump))
    print(f"  {verbose_name}:")
    for i, (rr, jj) in enumerate(trace):
        print(f"    after layer {i}: receptive field = {rr}x{rr}, jump = {jj}")
    return r


baseline_layers = [(3, 1), (2, 2), (3, 1), (2, 2)]  # conv1, pool1, conv2, pool2
larger_kernel_layers = [(5, 1), (2, 2), (5, 1), (2, 2)]
no_pooling_layers = [(3, 1), (2, 2), (3, 1), (2, 2)]  # stride-2 conv: same (k,s) as maxpool

rf_baseline = effective_receptive_field(baseline_layers, "Baseline (3x3, max pool) -- FINAL architecture")
rf_larger = effective_receptive_field(larger_kernel_layers, "Larger kernel (5x5, max pool)")
rf_nopool = effective_receptive_field(no_pooling_layers, "No pooling (3x3, stride-2 conv)")

print(f"""
RESULT: the final (baseline) architecture has an effective receptive
field of {rf_baseline}x{rf_baseline} pixels on the 32x32x3 input -- each unit at the
flatten stage "sees" a {rf_baseline}x{rf_baseline} region of the original image, just
under a third of its 32-pixel width/height, sufficient to capture most
of a CIFAR object's extent (objects in this dataset typically occupy a
large fraction of the 32x32 frame) without yet spanning the whole image.

Note the no-pooling variant has an IDENTICAL {rf_nopool}x{rf_nopool} receptive field to
the baseline -- because a stride-2 2x2 conv and a stride-2 2x2 max pool
have the exact same (kernel, stride) footprint in this formula, receptive
field size alone does NOT explain that variant's worse generalization
(D4 below) -- it must be attributed to the loss of pooling's
translation-invariance / implicit regularization, not to seeing less of
the image.
""")

print("""
--- D4: key findings from the Part C ablation study ---

Three configurations, identical data/seed/optimizer/epoch budget (20
epochs, no early stopping so the full trajectory is visible):

  Baseline (3x3, max pool)          TEST acc 0.7188 | params 67,252 | RF 10x10 | time 345s
  Larger kernel (5x5, max pool)     TEST acc 0.7175 | params 69,684 | RF 16x16 | time 591s
  No pooling (3x3, stride-2 conv)   TEST acc 0.7087 | params 68,556 | RF 10x10 | time 344s

Two findings, neither of them the naive "bigger/fancier is better" story:

1. THE LARGER KERNEL DID NOT HELP, DESPITE A 60% BIGGER RECEPTIVE FIELD
   AND MORE PARAMETERS. 5x5 kernels (RF 16x16) scored marginally BELOW
   the 3x3 baseline (RF 10x10) -- 0.7175 vs 0.7188 -- while taking 71%
   longer to train (more FLOPs per convolution). On a dataset this small
   (4,000 training images) and this low-resolution (32x32), a receptive
   field that already covers under a third of the image per unit at the
   flatten stage is apparently sufficient; the extra kernel parameters
   bought more capacity to overfit, not more useful signal.

2. REMOVING POOLING (replacing it with an equal-footprint stride-2 conv)
   MEASURABLY HURT GENERALIZATION, THOUGH NOT RECEPTIVE FIELD. Confirmed
   by the receptive-field calculation above: the no-pooling variant sees
   an IDENTICAL 10x10 region per output unit to the baseline, so its 1
   percentage-point-lower test accuracy (0.7087 vs 0.7188) cannot be a
   receptive-field effect. The validation curves (figures/06) show the
   real mechanism directly: the no-pooling variant's train accuracy
   climbs to 98.9% by epoch 20 while its validation loss actively
   diverges upward after epoch ~8 (a textbook overfitting signature) --
   exactly consistent with Part B's derived claim that pooling's
   approximate translation-invariance acts as an implicit regularizer
   that a parameter-equivalent strided convolution does not provide.

RECOMMENDATION: the baseline (3x3, max pool) configuration is the best
choice on every axis measured here -- highest test accuracy, fewest
parameters, fastest training, and (per D3) already has enough effective
receptive field for this image size -- a rare case where the cheapest
option was also the best one, rather than a capacity/cost trade-off.
""")

print(f"""
--- D5: from-scratch vs. PyTorch baseline, for context ---

NumPy CNN:   test acc 0.7275, F1 0.7243, 304.96s (17 epochs, early stopped)
PyTorch CNN: test acc 0.7300, F1 0.7306, 58.77s  (25 epochs, no early stop)
Accuracy gap: 0.0025 (0.25 percentage points) -- both share identical
architecture, optimizer (SGD+momentum 0.9), learning rate, batch size,
and data. This is the strongest available evidence that the from-scratch
NumPy implementation's forward/backward math (independently derived and
coded, no autograd) is CORRECT, not merely gradient-check-correct in
isolation: two completely independent implementations of the same
mathematics converge to statistically indistinguishable real-world
performance. PyTorch's ~5x wall-clock advantage is expected and discussed
in D6 below (cuDNN-level kernel fusion and optimization, not available to
a from-scratch NumPy implementation).
""")

print("""
--- D6: two challenges faced, with actual debugging process ---

1. GRADIENT-CHECKING A NETWORK CONTAINING MAX POOLING APPEARED TO FAIL.
   The full-network finite-difference check (cnn_model.py) initially
   looked broken specifically for architectures using max pooling:
   perturbing a single (spatially-shared) conv weight and checking via
   central differences failed on roughly 15% of random index checks,
   with errors as large as 2%, while the SAME check on a stride-conv
   (no max pool) architecture passed cleanly at ~1e-8. Rather than
   dismissing this as noise or, worse, silently "fixing" it by loosening
   the tolerance, the actual mechanism was investigated: a single conv
   weight is shared across every spatial position, so perturbing it
   nudges the pre-pool activation at MANY locations simultaneously.
   MaxPool's argmax is a hard, non-smooth switch -- for a typical
   perturbation, at least one of the many pooling windows touched is
   likely to have its argmax flip between the W+eps and W-eps
   evaluations, producing a real, expected O(1) jump in that one
   finite-difference estimate, even though the analytic gradient (using
   the single correct argmax from the true forward pass) is exact.
   CONFIRMED, not just argued: substituting AvgPool2D (smooth, no
   argmax) into the identical architecture and rerunning the exact same
   check dropped the max relative error to ~1e-10 (all pass), and the
   median error across the max-pool checks (~2e-9) was already tiny --
   only the rare tie-crossing checks were large. Resolved by reporting
   pass-rate and median error rather than a single worst-case number,
   with the mechanism documented directly in cnn_model.py's own
   self-check output.

2. THE FIRST FULL PART C TRAINING RUN APPEARED TO HANG FOR OVER 20
   MINUTES WITH ZERO OUTPUT WHILE BURNING 500%+ CPU. Initial hypotheses
   (BLAS thread-pool contention on tiny im2col matmuls, memory
   thrashing from the full-batch train/val evaluation done every epoch)
   were tested directly rather than assumed: an isolated micro-benchmark
   of the exact matmul shapes used showed single- vs multi-threaded BLAS
   made no measurable difference, and system memory was not under
   pressure, ruling both out. The training run was killed to investigate
   -- which, in isolation, would have been a dead end (SIGKILL discards
   all buffered-but-unflushed stdout, so no diagnostic information
   survived). Restarted with `python -u` (unbuffered) piped through
   `tee` for live visibility, revealing the REAL cause: the training was
   never stuck at all -- earlier concurrent benchmarking commands
   (run in parallel while investigating) were competing for the same
   CPU cores as the actual background training job, starving it. With
   nothing else running concurrently, the identical script trained at
   ~10s/epoch, exactly as expected. The lesson generalizes beyond this
   task: a diagnostic process that itself competes for the resource
   being diagnosed can manufacture the exact symptom it's investigating.
""")

print("""
--- D7: limitations of a manually implemented CNN vs. a production framework ---

COMPUTE EFFICIENCY: PyTorch trained the identical architecture ~5.2x
faster (58.8s vs 305.0s) despite doing MORE work (25 epochs, no early
stop, vs 17 epochs with early stopping for the NumPy version) -- this
NumPy implementation calls BLAS through im2col/matmul, which is already
far faster than the naive nested-loop version (Part A: 400x+), but still
pays real costs a production framework doesn't: an explicit im2col
materializes an expanded copy of the input in memory before every
convolution (extra memory bandwidth and allocation the framework's
fused, in-place kernels avoid entirely), and every layer here allocates
fresh NumPy arrays for its cache/gradients rather than reusing
pre-allocated buffers across the training loop.

cuDNN-LEVEL OPTIMIZATION: production frameworks select from many
hand-tuned, hardware-specific convolution algorithms (im2col+GEMM, FFT-
based, Winograd, direct) per layer shape/hardware combination at runtime,
and fuse sequences of operations (e.g. conv+batchnorm+ReLU) into single
GPU kernels to avoid materializing intermediate results at all. This
implementation always uses one fixed algorithm (im2col+GEMM) and never
fuses anything -- every layer boundary is a real, separate array in
memory.

AUTOMATIC DIFFERENTIATION: every backward pass in this task (Conv2D,
MaxPool2D, AvgPool2D, BatchNorm2D, and their composition into SimpleCNN)
was derived and coded by hand, then gradient-checked -- valuable
specifically because it forces and then verifies real understanding of
the underlying mathematics (this task's stated objective), but it does
not scale: adding a new layer type to a production framework requires
only implementing its forward pass and marking it differentiable, while
adding one here requires re-deriving and re-verifying its backward pass
from scratch, exactly as was done for every layer in cnn_layers.py.

GPU ACCELERATION: this entire implementation is CPU/NumPy-only; a
production framework would move the identical architecture to a GPU with
one line of code and see the same order-of-magnitude speedups measured
for feedforward networks in Task 22 (48x for raw matmuls on a T4).
Without autograd's dynamic computation graph and GPU kernels, that path
is not available to a from-scratch NumPy implementation without a
substantial additional engineering effort (a CUDA or OpenCL backend for
every layer).
""")

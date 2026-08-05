# Convolutional Neural Networks: Convolution, Pooling & Architecture Design

**PKCERT AI & Software Development Internship, Task 23**
Author: Abdullah Amir
**NumPy only** for every Part A/B/C core computation (no autograd) — **PyTorch 2.13.0
(CPU build), torchvision 0.28.0** used only for the Part C comparative baseline and data
loading. Random seed 42 throughout.

Every backward pass used below — 2D convolution (both naive and im2col/matmul), max and
average pooling, batch normalization, and the fully assembled CNN — is derived by hand
and verified against finite-difference gradient checks before being trusted for a real
training run.

## Dataset

CIFAR-10, restricted to 4 visually distinct classes (airplane, automobile, cat, dog),
deliberately different from every dataset used in Tasks 20-22 (all Fashion-MNIST,
greyscale). CIFAR-10 is 32×32 RGB — genuinely exercising Part A's "multi-channel input"
requirement, not just restating it. 1000/200/200 images per class for train/val/test
(4000/800/800 total), kept modest so pure-NumPy im2col training finishes in minutes.

## What's here

| File | Description |
| --- | --- |
| `cnn_layers.py` | Core from-scratch layers: `Conv2D` (naive nested-loop + im2col/matmul, both gradient-checked), `MaxPool2D`/`AvgPool2D` (forward+backward), `BatchNorm2D`, `Dropout`. Run `python cnn_layers.py` for the full gradient-check + correctness-benchmark suite. |
| `cnn_model.py` | `SimpleCNN`: a configurable 2-conv-layer CNN built from the layers above, with a full-network gradient check in its own `__main__` block. |
| `part_a_convolution_fundamentals.py` | Part A: output-shape formula, padding modes, fixed kernels, im2col vs naive benchmark. |
| `part_b_pooling_regularization.py` | Part B: pooling forward/backward derivation, pooling-vs-strided-conv comparison (with an empirical shift-robustness check), BatchNorm/Dropout from scratch. |
| `part_c_cnn_training.py` | Part C: full derivation, from-scratch CNN training (mini-batch GD + momentum), PyTorch baseline, 3-way ablation, filter/feature-map visualization. |
| `part_d_analysis.py` | Part D: effective receptive field (computed), pipeline/design summary, ablation findings, challenges faced, framework-limitations reflection. |
| `numpy_cnn_final_model.pkl` | The final trained from-scratch model's parameters (pickle). |
| `Day_23.ipynb` | All of the above as one executed notebook. |
| `figures/` | All generated plots. |
| `Report.pdf` / `Report.tex` | Full written report, Parts A-D. |

## Key results

**Part A**: im2col convolution matches the naive nested-loop implementation exactly
(max abs. difference ≤5.3×10⁻¹⁵, floating-point roundoff) across every
stride/padding/dilation combination tested, while being **400x+ faster** at 48×48 input
size — this speedup is why Part C's real training uses the im2col layer, not the naive
one.

**Part B**: a 1-pixel shift of a random feature map changes a stride-2 convolution's
output **3.17x more** than it changes a 2×2 max pool's output on the same map — an
empirical, not just textbook, confirmation of pooling's translation-robustness advantage.

**Part C**: from-scratch CNN on a 4-class CIFAR-10 subset (airplane/automobile/cat/dog)
reaches **72.75% test accuracy** — within **0.25 points** of an identically-configured
PyTorch baseline (73.00%), the strongest available evidence the hand-derived
forward/backward math is genuinely correct, not just gradient-check-correct in isolation.
Most-confused pair: true *dog* → predicted *cat* (72/200), exactly the hard pair this
class selection was chosen to exercise.

**Part D — ablation study**: two findings, neither the naive story.
- A **5×5 kernel** (60% bigger receptive field, more params) scored *marginally below*
  the 3×3 baseline (71.75% vs 71.88%) while training 47% slower — extra capacity bought
  overfitting, not signal, on this small/low-resolution dataset.
- **Removing pooling** (replaced with an equal-footprint stride-2 conv) measurably hurt
  generalization (70.87%) *despite an identical effective receptive field* to the
  baseline (both 10×10, computed via the standard recursive formula) — isolating the
  cause to pooling's translation-invariance regularization, not receptive field size. The
  no-pooling variant's validation loss visibly diverges while train accuracy hits 98.9%.

## Gradient-checking through max pooling — a genuine finding, not a bug

The full-network gradient check (`cnn_model.py`) initially looked broken for any
architecture using max pooling: perturbing a single (spatially-shared) conv weight and
checking via finite differences failed on ~15% of random checks, with errors up to 2%.
Investigated rather than dismissed: a shared conv weight nudges the pre-pool activation
at *every* spatial location simultaneously, so it's likely that at least one of the many
pooling windows touched has its argmax flip between the `W+eps` and `W-eps`
evaluations — producing a real, expected jump in that one finite-difference estimate,
even though the analytic gradient (computed from the single true forward pass) is exact.
Confirmed by substituting `AvgPool2D` (smooth, no argmax) into the identical
architecture and rerunning the same check: max relative error drops to ~10⁻¹⁰ (all
pass). The median error across the max-pool checks was ~2×10⁻⁹ — consistent with the
theory that only the rare tie-crossing checks are large, not a systematic bug.

## How to run

```bash
python3 -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib scikit-learn jupyter ipykernel nbformat pillow

python cnn_layers.py                        # gradient checks (fast, ~5s)
python cnn_model.py                         # full-network gradient check (fast, ~5s)
python part_a_convolution_fundamentals.py   # Part A (fast, ~5s)
python part_b_pooling_regularization.py     # Part B (fast, ~2s)
python part_c_cnn_training.py               # Part C, full run (CIFAR-10 downloads automatically, ~170MB, ~25-30 min CPU)
python part_d_analysis.py                   # Part D (fast, ~5s, reads the saved model from above)
```

Or open `Day_23.ipynb` directly — it ships with executed outputs.

`data/` (the downloaded CIFAR-10 files) is not included; the script and notebook
download it automatically via `torchvision.datasets.CIFAR10(..., download=True)` on
first run.

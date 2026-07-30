# Feedforward Neural Network on Fashion-MNIST

**PKCERT AI & Software Development Internship, Task 20** (Difficulty: Hard)
Author: Abdullah Amir
**PyTorch 2.13.0 (CPU build), torchvision 0.28.0**, random seed 42 throughout

Tasks 18 and 19 built the network API fluency (`nn.Module`, autograd) and training-loop
mechanics (loss functions, optimizers, epoch/batch engineering) used here. This task
assembles both into one complete, rigorously evaluated end-to-end pipeline: a proper
three-way train/validation/test split, leakage-free preprocessing, a justified
fully-connected architecture (no convolutions), honest evaluation, a controlled depth
ablation, and a verified model-persistence round-trip. Every reported number comes from
this repo's own run.

## Dataset

**Fashion-MNIST** (Xiao, Rasul & Vollgraf, 2017), loaded via
`torchvision.datasets.FashionMNIST` — the stated MNIST alternative, chosen because its
confusion matrix is far more informative than plain MNIST's: genuinely visually similar
clothing categories (shirt/pullover/coat), rather than MNIST's mostly-solved digit
pairs. 60,000 official training images + 10,000 official test images, 10 classes.

## What's here

One complete pipeline (`mnist_feedforward.py`), organized into four parts:

- **Part A — Data pipeline**: a genuine three-way split (50,000 train / 10,000 val /
  10,000 test, the official test set held out completely), normalization statistics
  computed from the training split only — **with the leakage this avoids demonstrated
  numerically**, not just described (train-only stats vs full-60k stats differ by a
  measurable, if small, amount); pipeline shapes printed at every stage; a sample-batch
  visualization; and a class-balance check across all three splits.
- **Part B — Model architecture**: a `784 → 256 → 128 → 10` fully-connected `nn.Module`
  with justified layer widths; a mathematical argument for ReLU over sigmoid/tanh; a
  numerical confirmation that logits+CrossEntropyLoss and softmax+NLLLoss produce
  identical loss values; and a hand-calculated parameter count verified against
  `sum(p.numel() for p in model.parameters())`.
- **Part C — Training, evaluation & experiments**: training with Adam, tracked
  train/val loss and val accuracy per epoch; full test-set evaluation (accuracy,
  precision, recall, macro-F1, confusion matrix) with the most-confused class pair
  identified and explained; a controlled 3-way depth ablation (0/1/2 hidden layers);
  and a save/reload round-trip verified to produce identical predictions.
- **Part D — Analysis & documentation**: a single results table, figure
  interpretations, a counter-intuitive finding from the ablation, a concrete
  implementation issue (diagnosed and fixed), and full reproducibility documentation.

## Key results

**Part A.** Normalization stats from the 50,000-image training split only: mean=0.2857,
std=0.3529. From all 60,000 (the leaky way): mean=0.2860, std=0.3530 — a small but real
difference, illustrating the principle even where today's numeric impact is modest.

**Part B.** Hand-calculated parameter count (235,146) matched
`sum(p.numel() for p in model.parameters())` exactly. `CrossEntropyLoss(logits, y)` and
`NLLLoss(log_softmax(logits), y)` produced identical loss values (2.304028 both) on a
sample batch.

**Part C — final test-set metrics:**

| Metric | Value |
| --- | --- |
| Accuracy | 0.8870 |
| Macro-Precision | 0.8880 |
| Macro-Recall | 0.8870 |
| Macro-F1 | 0.8873 |

Validation loss bottoms out at epoch 8 (mild overfitting begins after); validation
accuracy keeps drifting upward with noise until ~epoch 15 — loss and accuracy don't
necessarily peak together. Most confused pair: **Shirt ↔ T-shirt/top** (119 + 102
misclassifications) — both upper-body garments with overlapping silhouettes at 28×28
greyscale resolution.

**Depth ablation:**

| Configuration | Parameters | Test accuracy |
| --- | --- | --- |
| 0 hidden layers (linear) | 7,850 | 0.8417 |
| 1 hidden layer (256) | 203,530 | 0.8797 |
| 2 hidden layers (256, 128) | 235,146 | 0.8833 |

Adding the first hidden layer: **+3.80** points of accuracy for 195,680 more
parameters. Adding the second: only **+0.36** points for 31,616 more parameters —
sharply diminishing returns (see `Report.pdf` Part D for the full discussion of why).

**Model persistence**: reloaded `state_dict` predictions matched the original model's
exactly, element for element, on a held-out batch.

## How to run

```bash
python3 -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib scikit-learn jupyter ipykernel
jupyter notebook mnist_feedforward.ipynb   # then Run All
```

Or run the plain script (Fashion-MNIST downloads automatically on first run, ~30MB):

```bash
python mnist_feedforward.py
```

Full run (20-epoch main model + three 12-epoch ablation configurations) takes several
minutes on CPU; figures land in `figures/`.

## Files

| File | Description |
| --- | --- |
| `mnist_feedforward.ipynb` | The full notebook (Parts A–D), with outputs |
| `mnist_feedforward.py` | The same pipeline as a plain script |
| `feedforward_mnist_state_dict.pt` | The final trained model's saved weights |
| `figures/` | The five generated plots |
| `Report.pdf` / `Report.tex` | Full written report |
| `README.md` | This file |

`data/` (the downloaded Fashion-MNIST files, ~80MB) is not included; both the script
and notebook download it automatically via `torchvision.datasets.FashionMNIST(...,
download=True)` on first run.

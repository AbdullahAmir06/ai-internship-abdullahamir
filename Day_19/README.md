# Training Loops: Loss Functions, Optimizers (SGD & Adam), and Epoch/Batch Management

**PKCERT AI & Software Development Internship, Task 19** (Difficulty: Hard)
Author: Abdullah Amir
**PyTorch 2.13.0 (CPU build)**, random seed 42 throughout

Task 18 called `torch.optim` as a black box. This task opens it: SGD, SGD+momentum and
Adam are all implemented as manual parameter-update loops and verified step-for-step
against `torch.optim` on a toy convex function; loss functions are built from raw tensor
operations; and every major training-loop mechanism — batching, DataLoader shuffling,
batch-size effects, LR scheduling, gradient accumulation, clipping, early stopping — is
implemented and demonstrated on a case constructed specifically to trigger it. Every
design choice is backed by a number from this repo's own runs, not convention alone.

## Dataset

`penguins.csv` (Palmer Penguins), the same dataset and 4→8→3 architecture as Tasks
17/18, reused here so the optimizer/training-loop comparisons in Part B/C are run on a
real, previously-established network rather than a fresh toy problem for every test.

## What's here

- **Part A — Loss functions** (`part_a_losses.py`): MSE, L1, cross-entropy and hinge
  loss from raw tensor ops, each verified against PyTorch's built-ins; the
  cross-entropy-vs-MSE gradient argument demonstrated numerically (a confidently-wrong
  prediction gets a ~2239x larger gradient under cross-entropy); an L2-regularised
  custom loss with a hand-derived gradient checked against autograd; and class-imbalance
  gradient contributions before/after inverse-frequency weighting.
- **Part B — Optimizers** (`part_b_optimizers.py`): vanilla SGD, SGD+momentum and Adam
  implemented from scratch and verified step-for-step against `torch.optim` on an
  elongated quadratic bowl; a worked numerical example of Adam on 1000x-uneven
  gradients; a controlled 3-way optimizer comparison on the Penguins network; and a
  numerical demonstration that L2-in-the-gradient and AdamW's decoupled weight decay
  are equivalent for SGD but **not** for Adam.
- **Part C — Training loop engineering** (`part_c_training_loops.py`): batch GD vs
  mini-batch GD vs single-sample SGD compared directly; a proper
  `torch.utils.data.Dataset` + `DataLoader` with per-epoch shuffling; a batch-size sweep
  (8/32/128/full); a `StepLR` scheduler vs a fixed-LR baseline; gradient accumulation
  verified against a direct large-batch update (**including** a deliberately reproduced
  scaling bug); and gradient clipping / early stopping, each demonstrated on a case
  built specifically to trigger it.
- **Part D — Analysis & documentation**: a single table summarizing every experiment,
  the best evidenced optimizer/batch-size/scheduler combination, two counter-intuitive
  findings, and the concrete gradient-accumulation gotcha diagnosed and fixed live.

## Key results

**Part A.** Cross-entropy's gradient on a confidently-wrong prediction was **~2239x
larger** than MSE-on-softmax's, on the exact case that most needs correcting. A
hand-derived L2-regularised gradient matched autograd to $1.2\times10^{-7}$. Unweighted
cross-entropy's gradient split 9.0:1 between a 90:10 imbalanced pair of classes —
exactly the sample ratio; inverse-frequency weighting rebalanced it to 1.00:1.

**Part B.** All three from-scratch optimizers (SGD, SGD+momentum, Adam) matched
`torch.optim` to within $5\times10^{-7}$ on a toy quadratic. In the 3-way comparison on
the real network:

| Optimizer | Final training loss | Validation accuracy |
| --- | --- | --- |
| SGD (lr=0.1) | 0.0934 | 0.9855 |
| **SGD+momentum (lr=0.1, m=0.9)** | **0.0123** | **1.0000** |
| Adam (lr=0.01) | 0.0330 | 1.0000 |

L2-in-gradient vs decoupled weight decay: equivalent for SGD ($1.2\times10^{-7}$
difference), **not equivalent for Adam** (0.0668 difference) — confirmed numerically,
not just asserted.

**Part C — batch size sweep:**

| Batch size | Wall-clock (80 epochs) | Epochs to loss<0.3 | Train−val gap |
| --- | --- | --- | --- |
| 8 | 0.660s | 4 | 0.0000 |
| 32 | 0.414s | 11 | −0.0110 |
| 128 | 0.221s | 29 | −0.0148 |
| 273 (full) | **0.159s** | never | **+0.0535** |

Gradient accumulation: correctly scaled (loss divided by `K` before each `.backward()`)
matched a direct large-batch gradient to $1.5\times10^{-8}$; the deliberately reproduced
missing-`/K` bug produced a gradient exactly **4.00x too large** (K=4). Gradient
clipping: an unclipped lr=20.0 run exploded past $10^{35}$ by epoch 25; clipped
(max_norm=1.0), it trained stably to a final loss of 0.033. Early stopping: a
128-128-hidden network trained on a 20-row subset stopped at epoch 31 of 300 planned,
restoring the epoch-16 weights for a validation accuracy of 0.971 rather than
overfitting further.

**Recommended configuration:** SGD+momentum (lr=0.1, m=0.9), batch size 32, no
scheduler — see `Report.pdf` Part D for the full evidenced comparison table and two
counter-intuitive findings (batch size 8 converges in the fewest epochs but costs the
*most* wall-clock time; Adam did not outperform tuned SGD+momentum on this
well-conditioned network).

## How to run

```bash
python3 -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib scikit-learn jupyter ipykernel
# penguins.csv: copy from ../Day_17/
jupyter notebook training_loops.ipynb   # then Run All
```

Or run the three scripts directly (independent of each other, but in this reading
order):

```bash
python part_a_losses.py
python part_b_optimizers.py
python part_c_training_loops.py
```

Full run takes well under a minute; figures land in `figures/`.

## Files

| File | Description |
| --- | --- |
| `penguins.csv` | The dataset (same as Tasks 17/18) |
| `training_loops.ipynb` | The full notebook (Parts A–D), with outputs |
| `part_a_losses.py` | Part A: loss functions from scratch |
| `part_b_optimizers.py` | Part B: SGD/momentum/Adam from scratch, verified vs torch.optim |
| `part_c_training_loops.py` | Part C: batching, scheduling, accumulation, clipping, early stopping |
| `figures/` | The seven generated plots |
| `Report.pdf` / `Report.tex` | Full written report, including every derivation and the Part D comparison table |
| `README.md` | This file |

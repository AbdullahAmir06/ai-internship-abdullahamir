# Regularization Techniques in Deep Learning

**PKCERT AI & Software Development Internship, Task 21**
Author: Abdullah Amir
**PyTorch 2.13.0 (CPU build), torchvision 0.28.0**, random seed 42 throughout

Continues directly from Task 20's Fashion-MNIST feedforward pipeline. Task 20's
right-sized network barely overfit (validation loss rose only slightly after epoch 8),
which is the wrong setting to demonstrate what Dropout, Batch Normalization and Early
Stopping actually fix. This task's baseline is deliberately wider (784 → 512 → 256 →
10, roughly 4x Task 20's parameter count) and trained longer, so it overfits clearly —
giving each regularization technique a real, measured problem to solve rather than a
token demonstration.

## Dataset

Fashion-MNIST, the same dataset, split, and normalization approach as Task 20 (50,000
train / 10,000 validation / 10,000 test, train-only normalization statistics), reused
deliberately so this task's deliberately-overfitting baseline can be judged against
Task 20's already-established right-sized network on identical data.

## What's here

One pipeline (`regularization.py`), five trained configurations:

- **Baseline**: 784→512→256→10, no regularization, 25 epochs — deliberately overfits.
- **Dropout**: same architecture + `Dropout(p=0.5)` after each hidden activation.
- **Batch Normalization**: same architecture + `BatchNorm1d` after each linear layer.
- **Early Stopping**: the plain baseline architecture, trained with patience-5 early
  stopping on validation loss, restoring the best-validation-loss weights.
- **Combined** (bonus): Dropout + Batch Norm + Early Stopping together — the strongest
  overall configuration.

All five share the identical data split, architecture skeleton, optimizer (Adam,
lr=1e-3), batch size (256) and random seed, so the comparison isolates the
regularization technique itself.

## Key results

| Configuration | Test Accuracy | Test F1 | Train/Val Gap | Epochs | Training Time |
| --- | --- | --- | --- | --- | --- |
| Baseline (no regularization) | 0.8874 | 0.8876 | 0.0758 | 25 | 352.3s |
| Dropout (p=0.5) | 0.8879 | 0.8868 | **0.0310** | 25 | 344.0s |
| Batch Normalization | **0.8914** | **0.8916** | 0.0888 (worst) | 25 | 360.4s |
| Early Stopping | 0.8896 | 0.8895 | 0.0508 | 13 | **176.7s** |
| Combined (Dropout+BN+ES) | 0.8901 | 0.8907 | 0.0525 | 17 | 241.7s |

Three findings, in order of how much they matter:

1. **Dropout cut the overfitting gap roughly in half** (0.0758 → 0.0310) but barely
   moved test accuracy — it did exactly what dropout is supposed to do, with only a
   marginal accuracy payoff on this dataset/model size.
2. **The genuinely counter-intuitive one**: Batch Normalization gave the *best* raw
   test accuracy of any single technique (0.8914), but its overfitting gap was the
   *worst* of all five configurations (0.0888, worse than the unregularized baseline).
   Batch Norm's popular reputation as a regularizer was not borne out here — it clearly
   helped optimization (faster, more stable convergence to a better optimum), but did
   not reduce train/validation variance the way Dropout and Early Stopping did.
3. **Early Stopping improved on the baseline on every axis simultaneously**: better
   test accuracy, a much smaller gap, and **half the training time**, with zero added
   architectural complexity — it simply stopped 12 epochs earlier, at the point where
   validation loss stopped improving.

**Recommendation:** Early Stopping for the cheapest, most broadly positive single
change. The Combined configuration for the strongest overall result when the extra
tuning surface (dropout rate + patience) is worth it. Batch Normalization for
optimization speed and stability — not as a substitute for an explicit regularizer
against overfitting. Full advantages/limitations discussion in `Report.pdf` Part D.

## How to run

```bash
python3 -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib scikit-learn jupyter ipykernel
jupyter notebook regularization.ipynb   # then Run All
```

Or run the plain script (Fashion-MNIST downloads automatically on first run, ~80MB):

```bash
python regularization.py
```

Full run trains five networks (up to 25 epochs each) on 50,000 images, CPU-only —
expect roughly 20-25 minutes. Figures land in `figures/`.

## Files

| File | Description |
| --- | --- |
| `regularization.ipynb` | The full notebook (Parts A–D), with outputs |
| `regularization.py` | The same pipeline as a plain script |
| `figures/` | The five generated plots |
| `Report.pdf` / `Report.tex` | Full written report |
| `README.md` | This file |

`data/` (the downloaded Fashion-MNIST files, ~80MB) is not included; both the script
and notebook download it automatically via `torchvision.datasets.FashionMNIST(...,
download=True)` on first run.

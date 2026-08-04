# Feedforward Neural Networks on MNIST, Regularization & GPU-Accelerated Training

**PKCERT AI & Software Development Internship, Task 22**
Author: Abdullah Amir
**PyTorch 2.13.0 (CPU build), torchvision 0.28.0**, random seed 42 throughout

Continues the Fashion-MNIST pipeline from Tasks 20/21, going one level deeper: every
backward pass used below is derived by hand and verified against finite-difference
gradient checks before being trusted for a real training run — nothing here is
autograd, until the framework comparisons deliberately introduce it for contrast.

This sandbox has no CUDA-capable GPU (confirmed via `nvidia-smi`/`lspci`), so Parts D
and the GPU-dependent portion of Part E are a **separate, self-contained notebook**
(`Day_22_Colab_GPU.ipynb`), run on a free Google Colab Tesla T4 GPU — every number in
that section is a real measurement from that run, never estimated.

## Dataset

Fashion-MNIST, same split and normalization as Task 20/21 (50,000 train / 10,000
validation / 10,000 test, train-only normalization statistics: mean 0.2857, std 0.3529).

## What's here

| File | Description |
| --- | --- |
| `numpy_network.py` | Core from-scratch NumPy MLP: manual forward/backward, inverted dropout, batch normalization (forward/backward/running stats). Gradient-checked against finite differences in its own `__main__` block (run `python numpy_network.py`). |
| `part_abc_pipeline.py` | Parts A–C as a plain script: backprop re-derivation, NumPy vs PyTorch on Fashion-MNIST, 4-configuration regularization ablation. |
| `Day_22.ipynb` | Parts A–C as a notebook, executed with real outputs and figures. |
| `part_e_cpu_prep.py` | Trains and saves the final NumPy model (Dropout+BatchNorm+EarlyStopping) for Part E's integration comparison. |
| `numpy_final_model.pkl` | The saved model from the script above. |
| `Day_22_Colab_GPU.ipynb` | Part D (device check, CPU-vs-GPU benchmark, mixed precision) and Part E's GPU-dependent pieces (regularized model on GPU, NumPy-vs-framework comparison, robustness stress test). Executed on a Colab Tesla T4 GPU; ships with real outputs. |
| `figures/` | Generated plots from Parts A–C plus the GPU robustness/timing figure. |
| `Report.pdf` / `Report.tex` | Full written report, Parts A–F. |

## Key results (Parts A–C, this sandbox — CPU only)

**Part B — NumPy (from scratch) vs PyTorch**, identical 784→256→128→10 architecture,
plain SGD (lr=0.1), 20 epochs:

| Implementation | Test Accuracy | Test F1 | Training Time |
| --- | --- | --- | --- |
| NumPy (from scratch) | 0.8813 | 0.8817 | 94.5s |
| PyTorch | 0.8628 | 0.8583 | 14.3s |

Gap (0.0185) is implementation-level (float32 vs float64, mini-batch shuffle order), not
algorithmic — both use the identical architecture, optimizer, and update rule.

**Part C — regularization ablation**, same architecture/split/seed:

| Configuration | Test Accuracy | Train/Test Gap | Epochs |
| --- | --- | --- | --- |
| Baseline (no regularization) | 0.8813 | 0.0590 | 20 |
| Dropout (p=0.5) | 0.8730 | 0.0273 | 20 |
| Batch Normalization | 0.8540 (worst) | 0.0751 (worst) | 20 |
| Dropout + BatchNorm + Early Stopping | **0.8838** | **0.0442** | 31 (stopped) |

The genuinely counter-intuitive result: **Batch Normalization alone was actively
counterproductive** here — worst accuracy *and* worst overfitting gap of all four,
with visibly unstable validation loss. Investigated, not smoothed over: BatchNorm
permits larger effective gradient steps that plain SGD (no momentum/adaptive scaling)
doesn't damp the way Adam would — a real, measured interaction effect, detailed in
`Report.pdf` Part F. The combined configuration won on every axis, and is the only one
where early stopping's restore-best-weights mechanism is directly demonstrated:
training ran to epoch 31 before patience-5 expired, restoring epoch 26's weights (not
epoch 31's).

**Part C14** — a hyperparameter mistake, demonstrated empirically: leaving dropout
active at test time drops accuracy from 0.8730 to an average 0.8497 *and* makes
predictions non-deterministic (only 88.7% agreement between repeated evaluations of the
identical test set).

## Part D / Part E (GPU) — real results, run on Google Colab (Tesla T4)

`Day_22_Colab_GPU.ipynb` is self-contained and was run top-to-bottom on a free Colab T4
GPU; it ships with its real outputs, not templates. Highlights:

**Part D**: CPU-vs-GPU matmul benchmark, **48.1x speedup** (1697ms/rep CPU vs 35ms/rep
GPU). Mixed precision: **2.38x** faster per training step, but peak memory rose 3.8%
rather than falling — a genuine, small-model-scale finding (`GradScaler` bookkeeping
overhead outweighs fp16 activation savings at this size), reported honestly rather than
assuming the textbook memory reduction always applies.

**Part E (GPU side)**: PyTorch model (Dropout+BatchNorm, Adam) trained on GPU reached
**0.8928 test accuracy** in 15.5s (early stopping at epoch 21, restoring epoch 16) —
beating every CPU configuration. Prediction agreement with the from-scratch NumPy model:
**93.75%**. But the more interesting finding came from the robustness stress test
(Gaussian noise + rotation): the higher-accuracy GPU/Adam model was **less robust**
(accuracy drop 0.3663) than the lower-accuracy CPU/SGD model (drop 0.2789) — evidence
that Adam's faster convergence to sharper minima trades off against corruption
robustness, not something raw clean-set accuracy would predict. Full discussion in
`Report.pdf` Part E/F.

## How to run

```bash
python3 -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib scikit-learn jupyter ipykernel nbformat

python numpy_network.py          # gradient checks (fast, <5s)
python part_abc_pipeline.py      # Parts A-C, full run (~7 minutes on CPU)
python part_e_cpu_prep.py        # Part E's saved NumPy model (~2 minutes)
```

Or open `Day_22.ipynb` directly (Fashion-MNIST downloads automatically on first run,
~80MB) — it already ships with executed outputs. `Day_22_Colab_GPU.ipynb` likewise
ships with its real Colab-GPU outputs already executed; to reproduce, upload it to
[Google Colab](https://colab.research.google.com), select a GPU runtime, and Run All.

`data/` (the downloaded Fashion-MNIST files) is not included; both scripts and both
notebooks download it automatically via `torchvision.datasets.FashionMNIST(...,
download=True)` on first run.

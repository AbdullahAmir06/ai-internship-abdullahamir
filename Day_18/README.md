# Intro to PyTorch: Tensors, Autograd & Building a Simple Neural Network

**PKCERT AI & Software Development Internship, Task 18**
Author: Abdullah Amir
**PyTorch 2.13.0 (CPU build)**

Task 17 built a perceptron, activation functions, and a 2-layer MLP entirely from
scratch in NumPy, with backpropagation verified against finite differences. This task
rebuilds the same architecture on the same dataset in **PyTorch**, using its autograd
engine and `nn.Module` API throughout, and maps every concept back to that manual
derivation — including a direct numerical comparison between autograd's gradients and
Task 17's hand-derived formulas.

## Dataset

`penguins.csv` (Palmer Penguins), the same dataset, split methodology and 4→8→3
architecture as Task 17, copied over specifically so the two tasks' results are
meaningfully comparable, per the task's own instruction.

## What's here

- **Part A — Tensors** (`part_a_tensors.py`): PyTorch version/CUDA check, four tensor
  creation methods explained, indexing/slicing/`view` vs `reshape`/broadcasting with
  worked examples, a timed NumPy-vs-PyTorch matmul, and the `torch.Tensor`/
  `numpy.ndarray` memory-sharing relationship demonstrated with actual mutations.
- **Part B — Autograd** (`part_b_autograd.py`): computational graphs (define-by-run),
  a scalar gradient checked by hand, gradient accumulation across `.backward()` calls,
  `torch.no_grad()`/`.detach()` (including an actually-caught `RuntimeError` proving why
  `no_grad()` is required), and — the load-bearing result — **autograd's gradients
  checked against Task 17's saved weights and hand-derived NumPy backprop on identical
  data**.
- **Part C — Building & training** (`part_c_pytorch_nn.py`): a `PenguinNet(nn.Module)`,
  `nn.CrossEntropyLoss` + `SGD`, an explicit training loop, full evaluation, a **3-way
  timed comparison** (fresh NumPy `ManualMLP`, `sklearn.MLPClassifier`, and this PyTorch
  model, all trained in the same run on the same hardware), and an optimizer-choice
  hyperparameter experiment (SGD vs Adam at two learning rates).
- **Part D — Analysis & documentation**: `torch.save`/`load_state_dict` persistence,
  pipeline summary, two concrete NumPy-vs-PyTorch differences, a reflection on what
  autograd abstracts away, and the live-demonstrated in-place-leaf-tensor gotcha.

## Key results

**Part B — autograd vs Task 17's manual backprop** (the same weights, same data, two
independent computational paths): max absolute difference across all four parameter
gradients ≈ **10⁻¹⁷** — machine epsilon for float64. Task 17's finite-difference check
(~10⁻¹⁰) and this exact-vs-exact check now doubly confirm the same derivation.

**Part C — three implementations, one architecture, timed on the same machine:**

| Implementation | Accuracy | Macro-F1 | Fit time |
| --- | --- | --- | --- |
| ManualMLP (NumPy, Task 17) | 0.9855 | 0.9877 | **0.110s** |
| sklearn MLPClassifier | **1.0000** | **1.0000** | **0.109s** |
| PyTorch nn.Module (SGD) | **1.0000** | **1.0000** | 2.510s |

PyTorch matched sklearn on accuracy but took **~23x longer to train** than either the
NumPy or scikit-learn model — the expected result of per-operation framework overhead
(autograd graph construction, Python/C++ dispatch across ~5,400 mini-batch steps)
dominating when the actual matrix operations are this tiny. Frameworks earn their keep
on large models/batches/GPU execution, not necessarily wall-clock speed on a toy
network.

**Hyperparameter experiment (optimizer choice):**

| Optimizer | Final training loss | Test accuracy |
| --- | --- | --- |
| SGD (lr=0.1) | 0.0212 | **1.0000** |
| Adam (lr=0.1) | 0.0019 | 0.9855 (noisiest loss curve) |
| Adam (lr=0.01) | **0.0012** | **1.0000** |

Adam at the *same* learning rate as SGD did worse and was visibly noisier — Adam's
adaptive scaling makes its effective step size larger than SGD's at a given nominal
`lr`, and 0.1 is outside its comfortable range. An optimizer swap is not a drop-in
change at a fixed learning rate.

## How to run

```bash
python3 -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib scikit-learn jupyter ipykernel
# penguins.csv and day17_manual_mlp_params.pkl: copy from ../Day_17/
jupyter notebook intro_to_pytorch.ipynb   # then Run All
```

Or run the three scripts directly (in order — Part B needs Task 17's saved weights,
Part C is independent):

```bash
python part_a_tensors.py
python part_b_autograd.py
python part_c_pytorch_nn.py
```

Full run takes under a minute; figures land in `figures/`, and the trained PyTorch
model's weights are saved to `pytorch_mlp_state_dict.pt`.

## Files

| File | Description |
| --- | --- |
| `penguins.csv` | The dataset (same as Task 17) |
| `day17_manual_mlp_params.pkl` | Task 17's saved weights, used for the Part B autograd-vs-manual comparison |
| `intro_to_pytorch.ipynb` | The full notebook (Parts A–D), with outputs |
| `part_a_tensors.py` | Part A: tensors and basic operations |
| `part_b_autograd.py` | Part B: autograd, verified against Task 17 |
| `part_c_pytorch_nn.py` | Part C: nn.Module training, 3-way comparison, hyperparameter experiment |
| `pytorch_mlp_state_dict.pt` | The final trained PyTorch model's saved weights |
| `figures/` | The four generated plots |
| `Report.pdf` / `Report.tex` | Full written report |
| `README.md` | This file |

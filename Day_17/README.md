# Neural Network Fundamentals: Perceptron, Activation Functions & Backpropagation

**PKCERT AI & Software Development Internship, Task 17**
Author: Abdullah Amir

A from-scratch, NumPy-only implementation of a perceptron, five activation functions
with their derivatives, and a 2-layer MLP trained with hand-derived backpropagation —
verified with a numerical gradient check and benchmarked against a scikit-learn
`MLPClassifier` baseline. No autograd framework (TensorFlow/PyTorch/Keras) computes a
gradient anywhere in this submission.

## Datasets

- **Part A:** Iris (`sklearn.datasets.load_iris`), Setosa vs Versicolor, 2 features
  (petal length/width) — the textbook linearly separable pair — plus the classic 4-point
  XOR problem.
- **Part C:** `penguins.csv` (Palmer Penguins, Gorman/Williams/Fraser 2014), 342 birds
  after dropping incomplete rows, 4 numeric measurements, 3 species. Not used in any
  earlier task; a natural fit for the required 2–4 features / 2–4 classes.

## What's here

- **Part A — Perceptron** (`part_a_perceptron.py`): the perceptron learning rule from
  scratch, trained on Iris (converges in 2 epochs) and on XOR (never converges, proven
  and plotted), plus the perceptron/logistic-regression relationship written out.
- **Part B — Activation functions** (`part_b_activations.py`): Sigmoid, Tanh, ReLU,
  Leaky ReLU, Softmax and their derivatives, all from scratch; vanishing-gradient and
  dying-ReLU analysis backed by the actual saturation numbers, not just assertions.
- **Part C — Forward & backprop mini-project** (`part_c_backprop_mlp.py`): a full
  derivation (forward pass, loss, backward pass, in matrix form) implemented as a
  `ManualMLP` class, verified against finite-difference gradients, trained on Palmer
  Penguins, evaluated against `sklearn.neural_network.MLPClassifier`, and a learning-rate
  hyperparameter sweep.
- **Part D — Analysis & documentation**: model persistence (pickle), pipeline summary,
  hyperparameter findings, two implementation challenges and how they were resolved, and
  a reflection on manual-MLP limitations — all written out in the notebook/report.

## Key results

**Part A.** Perceptron converges to 100% training accuracy on Setosa vs Versicolor in 2
epochs; on XOR it never converges, plateauing at 50% accuracy (chance level) — the
textbook proof that a single linear boundary cannot separate a non-linearly-separable
problem.

**Part B.** Sigmoid's derivative is below 0.01 by `|x| ≈ 4.6` and sits below `1e-3`
across 31% of `[-10, 10]`; tanh, across 58.6%. ReLU's derivative is a clean 1 for any
`x > 0`, which is why it dominates as the default hidden-layer activation in deep
networks.

**Part C — gradient check** (the load-bearing result): analytical backprop gradients
matched independent finite-difference gradients to a max relative error of
**~$10^{-10}$–$10^{-11}$** across all four parameter matrices — essentially
floating-point noise, not approximation error.

| Model | Accuracy | Macro-F1 |
| --- | --- | --- |
| ManualMLP (from-scratch NumPy) | 0.9855 | 0.9877 |
| sklearn MLPClassifier (same architecture) | 1.0000 | 1.0000 |

The 1.45-point gap is a single misclassified penguin out of 69 test birds — ordinary
training-run variance, not a structural flaw, given the gradient check already confirmed
the math independently.

**Hyperparameter experiment (learning rate):**

| Learning rate | Final training loss | Test accuracy |
| --- | --- | --- |
| 0.001 | 0.2694 | 0.8841 (too slow, under-converged) |
| **0.01** | 0.0346 | **1.0000 (sweet spot)** |
| 0.1 | 0.0065 | 0.9855 (visible loss spikes) |
| 1.0 | **0.0021 (lowest)** | 0.9855 (spikiest; lowest loss ≠ best test accuracy) |

## How to run

```bash
python3 -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install numpy pandas matplotlib scikit-learn joblib jupyter ipykernel
curl -o penguins.csv https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv
jupyter notebook neural_network_fundamentals.ipynb   # then Run All
```

Or run the three scripts directly (Part C imports from Part B, so run in order):

```bash
python part_a_perceptron.py
python part_b_activations.py
python part_c_backprop_mlp.py
```

Full run takes a few seconds; all figures land in `figures/`, and the trained
`ManualMLP`'s weights are saved to `manual_mlp_params.pkl`.

## Files

| File | Description |
| --- | --- |
| `penguins.csv` | The Part C dataset |
| `neural_network_fundamentals.ipynb` | The full notebook (Parts A–D), with outputs and derivations |
| `part_a_perceptron.py` | Part A: perceptron from scratch |
| `part_b_activations.py` | Part B: activation functions + derivatives (imported by Part C) |
| `part_c_backprop_mlp.py` | Part C: manual forward/backprop MLP, gradient check, sklearn comparison |
| `manual_mlp_params.pkl` | The final trained model's saved weights/biases |
| `figures/` | The eight generated plots |
| `Report.pdf` / `Report.tex` | Full written report, including the matrix-form derivation |
| `README.md` | This file |

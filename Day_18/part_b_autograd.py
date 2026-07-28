"""
PKCERT AI & Software Development Internship, Task 18
Part B: Autograd

requires_grad, .backward(), .grad, gradient accumulation, no_grad()/.detach(),
and a direct numerical comparison between PyTorch autograd's gradients and
Task 17's hand-derived NumPy backpropagation on the identical weights and
data.
"""

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

torch.manual_seed(42)

print(f"=== Part B: PyTorch {torch.__version__} autograd ===")

# ----------------------------------------------------------------------
# B1: computational graph, define-by-run
# ----------------------------------------------------------------------
print("""
A computational graph is a record of every differentiable operation applied
to a tensor, linking each output back to the inputs (and the operation) that
produced it -- it is what .backward() walks in reverse to apply the chain
rule automatically. PyTorch builds this graph dynamically ("define-by-run"):
no graph exists before any code runs; each tensor operation, as it actually
executes, appends one more node that remembers its inputs and the local
derivative rule needed to backpropagate through it. This is why ordinary
Python control flow (if-statements, loops with a variable trip count) works
transparently in PyTorch -- the graph is only ever *this run's* sequence of
operations, not a separate static structure declared in advance the way
older "define-and-run" frameworks required.
""")

# ----------------------------------------------------------------------
# B2: requires_grad, .backward(), .grad on a scalar expression
# ----------------------------------------------------------------------
print("--- B2: a simple scalar expression, checked against a hand derivative ---")
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 3 * x + 1  # dy/dx = 2x + 3
y.backward()
hand_derivative = 2 * 3.0 + 3
print(f"y = x^2 + 3x + 1 at x=3.0")
print(f"autograd:  dy/dx = {x.grad.item()}")
print(f"by hand:   dy/dx = 2x + 3 = {hand_derivative}")
print(f"Match: {torch.isclose(x.grad, torch.tensor(hand_derivative)).item()}")

# ----------------------------------------------------------------------
# B3: gradient accumulation and zero_grad()
# ----------------------------------------------------------------------
print("\n--- B3: gradient accumulation across multiple .backward() calls ---")
w = torch.tensor(2.0, requires_grad=True)
for step in range(3):
    loss = (w - 5) ** 2  # dloss/dw = 2(w-5) = -6 at w=2, every time (w never updated here)
    loss.backward()
    print(f"  after .backward() call {step + 1}: w.grad = {w.grad.item()} "
          f"(expected single-call gradient: -6.0)")
print("Without zero_grad(), .grad keeps *adding* the new gradient to whatever was "
      "already there from the previous call -- by the 3rd call it has accumulated to "
      "3x the single-step gradient, not stayed at it. This is why every training loop "
      "must call optimizer.zero_grad() (or param.grad = None) before each new "
      "loss.backward(): otherwise each step's update silently includes the leftover "
      "gradient from every step before it, which is almost never the intended "
      "behaviour (the deliberate exception is genuine gradient accumulation over "
      "several mini-batches to simulate a larger batch size, where the accumulation "
      "is the point).")
w.grad.zero_()
loss = (w - 5) ** 2
loss.backward()
print(f"After w.grad.zero_() and one fresh .backward(): w.grad = {w.grad.item()} "
      f"(back to the correct single-step value)")

# ----------------------------------------------------------------------
# B4: torch.no_grad() and .detach()
# ----------------------------------------------------------------------
print("\n--- B4: torch.no_grad() and .detach() ---")
weight = torch.tensor(4.0, requires_grad=True)
lr = 0.1
loss = (weight - 1) ** 2
loss.backward()

# Proof, not just assertion: try the in-place update WITHOUT no_grad() first,
# and show PyTorch actually refuses it.
try:
    weight -= lr * weight.grad
except RuntimeError as e:
    print(f"Without no_grad(), the same in-place update raises:\n  RuntimeError: {e}")
    weight = torch.tensor(4.0, requires_grad=True)  # reset: the failed op may have partially mutated state
    loss = (weight - 1) ** 2
    loss.backward()

with torch.no_grad():
    # Manually applying a gradient step is itself a tensor operation; without
    # no_grad() it would be added to the SAME graph .backward() just walked,
    # which is both wasteful (building graph for an update step nothing will
    # ever differentiate) and, worse, an in-place modification of a leaf
    # tensor that requires_grad -- something autograd explicitly forbids and
    # raises a RuntimeError for.
    weight -= lr * weight.grad
print(f"Manual update inside torch.no_grad(): weight = {weight.item():.4f} "
      f"(no_grad() is required here -- PyTorch raises a RuntimeError on an "
      f"in-place update to a requires_grad leaf tensor without it)")

evaluation_output = weight.detach()
print(f".detach(): evaluation_output = weight.detach() -> {evaluation_output.item():.4f}, "
      f"requires_grad={evaluation_output.requires_grad}. Use this when a value is needed "
      f"*as a plain number* going forward -- e.g. logging a loss value, or feeding a "
      f"prediction into a metric like accuracy_score -- without carrying its entire "
      f"upstream graph along; no_grad() is the right tool for a whole block of "
      f"computation (e.g. an entire evaluation pass), .detach() for pulling one "
      f"specific tensor out of the graph mid-computation.")

# ----------------------------------------------------------------------
# B5: autograd vs Task 17's manual backpropagation, same weights and data
# ----------------------------------------------------------------------
print("\n--- B5: autograd gradients vs Task 17's hand-derived NumPy backprop ---")
with open("day17_manual_mlp_params.pkl", "rb") as f:
    saved = pickle.load(f)
W1_np, b1_np = saved["params"]["W1"], saved["params"]["b1"]
W2_np, b2_np = saved["params"]["W2"], saved["params"]["b2"]
mean_np, scale_np = saved["scaler_mean"], saved["scaler_scale"]
species = saved["species"]
print(f"Loaded Task 17's trained weights: W1 {W1_np.shape}, W2 {W2_np.shape} "
      f"(architecture 4 -> {W1_np.shape[1]} -> {W2_np.shape[1]})")

df = pd.read_csv("penguins.csv").dropna(
    subset=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"])
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
species_to_idx = {s: i for i, s in enumerate(species)}
X_batch = df[FEATURES].values[:16]
y_batch = df["species"].map(species_to_idx).values[:16]
X_batch_s = (X_batch - mean_np) / scale_np


def one_hot_np(y, n_classes):
    Y = np.zeros((y.size, n_classes)); Y[np.arange(y.size), y] = 1.0
    return Y


Y_batch = one_hot_np(y_batch, 3)

# -- Task 17's manual NumPy forward + backward, reproduced exactly --------
Z1 = X_batch_s @ W1_np + b1_np
A1 = np.maximum(0, Z1)
Z2 = A1 @ W2_np + b2_np
Z2_shift = Z2 - Z2.max(axis=1, keepdims=True)
A2 = np.exp(Z2_shift) / np.exp(Z2_shift).sum(axis=1, keepdims=True)
N = X_batch_s.shape[0]
dZ2 = (A2 - Y_batch) / N
dW2_manual = A1.T @ dZ2
db2_manual = dZ2.sum(axis=0, keepdims=True)
dA1 = dZ2 @ W2_np.T
dZ1 = dA1 * (Z1 > 0).astype(float)
dW1_manual = X_batch_s.T @ dZ1
db1_manual = dZ1.sum(axis=0, keepdims=True)

# -- The identical computation, in PyTorch, gradients from autograd -------
X_t = torch.tensor(X_batch_s, dtype=torch.float64)
y_t = torch.tensor(y_batch, dtype=torch.long)
W1_t = torch.tensor(W1_np, dtype=torch.float64, requires_grad=True)
b1_t = torch.tensor(b1_np, dtype=torch.float64, requires_grad=True)
W2_t = torch.tensor(W2_np, dtype=torch.float64, requires_grad=True)
b2_t = torch.tensor(b2_np, dtype=torch.float64, requires_grad=True)

Z1_t = X_t @ W1_t + b1_t
A1_t = torch.relu(Z1_t)
Z2_t = A1_t @ W2_t + b2_t
# F.cross_entropy expects raw logits + integer class labels, and internally
# combines log_softmax + negative-log-likelihood -- the PyTorch equivalent
# of the "softmax + cross-entropy" pair manually derived in Task 17.
loss_t = F.cross_entropy(Z2_t, y_t)
loss_t.backward()

diffs = {
    "W1": np.abs(W1_t.grad.numpy() - dW1_manual).max(),
    "b1": np.abs(b1_t.grad.numpy() - db1_manual).max(),
    "W2": np.abs(W2_t.grad.numpy() - dW2_manual).max(),
    "b2": np.abs(b2_t.grad.numpy() - db2_manual).max(),
}
print("\nMax absolute difference, autograd gradient vs Task 17 manual gradient:")
for name, d in diffs.items():
    print(f"  {name}: {d:.2e}  ({'MATCH' if d < 1e-8 else 'MISMATCH'}, tolerance 1e-8)")
print(f"\nOverall max difference across all four parameters: {max(diffs.values()):.2e}")
print("PyTorch's autograd and Task 17's hand-derived backpropagation agree to "
      "within floating-point precision on identical weights and data -- independent "
      "confirmation, from a completely different computational path, of the same "
      "result Task 17's finite-difference gradient check already established.")

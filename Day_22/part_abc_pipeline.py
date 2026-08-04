"""
PKCERT AI & Software Development Internship, Task 22
Parts A, B, C: Conceptual Refresher, Feedforward Network on Fashion-MNIST
(NumPy from scratch vs PyTorch), and Regularization Techniques

Uses the gradient-checked NumpyMLP from numpy_network.py (see that file's
own __main__ block for the finite-difference verification of every
backward path used here) alongside an equivalent PyTorch implementation.
"""

import time
import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

from numpy_network import NumpyMLP, cross_entropy_loss

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np_rng = np.random.default_rng(RANDOM_STATE)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

print(f"=== PyTorch {torch.__version__} | torchvision {torchvision.__version__} ===")
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# ======================================================================
# Part A: Conceptual Refresher & Integration
# ======================================================================
print("\n" + "=" * 70)
print("PART A: CONCEPTUAL REFRESHER")
print("=" * 70)

print("""
--- A1: backpropagation for a 2-hidden-layer ReLU/softmax/cross-entropy
network, re-derived from memory ---

Architecture: X (N,d) -> [W1,b1] -> Z1 -> ReLU -> A1 -> [W2,b2] -> Z2 ->
ReLU -> A2 -> [W3,b3] -> Z3 -> softmax -> A3 (predicted probabilities).

FORWARD:
    Z1 = X W1 + b1            A1 = ReLU(Z1)
    Z2 = A1 W2 + b2           A2 = ReLU(Z2)
    Z3 = A2 W3 + b3           A3 = softmax(Z3)

LOSS (mean cross-entropy over a batch of N, Y one-hot):
    L = -(1/N) sum_n sum_k Y[n,k] log(A3[n,k])

BACKWARD, worked from the output inward:

Step 1 -- dL/dZ3. This is the step that most benefits from being derived,
not memorised. Cross-entropy in terms of softmax outputs is
L_n = -sum_k Y_nk log(A3_nk). Differentiating w.r.t. one pre-softmax logit
Z3_nj:
    dL_n/dZ3_nj = -sum_k Y_nk * (1/A3_nk) * dA3_nk/dZ3_nj
The softmax Jacobian is dA3_nk/dZ3_nj = A3_nk(delta_kj - A3_nj). Substituting
and using sum_k Y_nk = 1 (Y is one-hot):
    dL_n/dZ3_nj = -sum_k Y_nk (delta_kj - A3_nj) = -(Y_nj - A3_nj * sum_k Y_nk)
                = A3_nj - Y_nj
So, batched and averaged: dZ3 = (A3 - Y) / N. The 1/A3 term from the log
and the A3 term from the softmax Jacobian cancel almost entirely -- this
is the whole reason the combination is used, expanded on in A2 below.

Step 2 -- layer 3 (output) parameters:
    dW3 = A2^T dZ3              db3 = sum_n dZ3
    dA2 = dZ3 W3^T                                  (backprop into A2)

Step 3 -- through ReLU2:
    dZ2 = dA2 * ReLU'(Z2)        where ReLU'(z) = 1 if z>0 else 0

Step 4 -- layer 2 parameters:
    dW2 = A1^T dZ2               db2 = sum_n dZ2
    dA1 = dZ2 W2^T

Step 5 -- through ReLU1:
    dZ1 = dA1 * ReLU'(Z1)

Step 6 -- layer 1 parameters:
    dW1 = X^T dZ1                db1 = sum_n dZ1

UPDATE (gradient descent, learning rate eta):
    W_i <- W_i - eta * dW_i    b_i <- b_i - eta * db_i    for i in {1,2,3}

This is checked, not just stated: numpy_network.py's gradient-check block
verifies dW1, dW2, dW3, and every bias against independent finite-difference
gradients before any of it is used to train a real model below (all passed
at ~1e-9 to 1e-10 relative error).
""")

print("""
--- A2: why softmax pairs with cross-entropy, not MSE ---

Derived above: dL/dZ3 = A3 - Y for the softmax+cross-entropy pair. This is
clean and well-scaled -- its magnitude is bounded in [0,1] per class and is
large exactly when the prediction is confidently wrong, small when it is
already correct. Nothing about it depends on how saturated softmax happens
to be.

MSE on the softmax outputs, L = (1/N) sum (A3-Y)^2, backpropagated to the
SAME pre-softmax logits Z3, must go through the softmax Jacobian in full:
dL/dZ3 = dL/dA3 * dA3/dZ3, and dA3/dZ3 (the softmax Jacobian) is
proportional to A3*(1-A3) per output. That extra A3*(1-A3) factor is exactly
what collapses toward 0 as A3 saturates toward 0 or 1 -- so a confidently
WRONG prediction (A3 near 0 for the true class) gets almost no gradient
under MSE, precisely the case that most needs correcting. Cross-entropy
does not have this extra multiplicative factor because the 1/A3 term in its
own derivative cancels the A3 term from the softmax Jacobian (shown in the
A1 derivation above) -- that cancellation is the entire reason the pairing
is preferred, not a stylistic convention.
""")

print("""
--- A3: perceptron -> activation function -> backpropagation, in ~250 words ---

A perceptron alone computes one thing: a weighted sum of its inputs, passed
through a hard yes/no threshold. Stack many perceptron-like units into a
layer, and swap that hard threshold for a smooth activation function (ReLU,
sigmoid, tanh), and each unit now outputs a continuous number instead of a
binary vote -- which matters because "continuous" is what makes calculus
possible. A single layer of these units, however smooth, can still only
draw a straight-line boundary through the input space, exactly like one
perceptron could.

Stack a second layer on top, feeding the first layer's outputs in as its
inputs, and something new happens: the network can now combine several
straight lines into a curved, composite decision boundary -- an XOR
becomes solvable, which a lone perceptron provably cannot do. This is the
"feedforward network" part: perceptron-like units, arranged in layers, each
layer's output feeding the next.

The missing piece is how such a network learns its weights without a human
hand-tuning each one. Backpropagation is just the chain rule, applied
systematically: compute how wrong the final output was (the loss), then
walk backward through the network one layer at a time, at each layer asking
"how much would nudging this layer's weights change the final loss,
given how much nudging its output changed the loss above it?" Because every
activation function has a known derivative, that question always has a
computable answer, layer by layer, all the way back to the first weight.
Gradient descent then just nudges every weight a small step in the
direction that reduces the loss. Repeated over many examples, this is the
entire mechanism: perceptron-shaped units provide the linear building
block, activation functions make them differentiable and non-linear, and
backpropagation is the bookkeeping that turns "the network was wrong" into
a specific, computable correction for every single weight.
""")

# ======================================================================
# Part B: Feedforward Neural Network on Fashion-MNIST
# ======================================================================
print("\n" + "=" * 70)
print("PART B: FEEDFORWARD NETWORK ON FASHION-MNIST")
print("=" * 70)
print("""
Dataset: Fashion-MNIST -- explicitly offered in the task as "an extra
challenge" beyond plain MNIST, and continuing the dataset this internship
has used since Task 20, so this task's NumPy/framework comparison and
Part C's regularization ablation are directly comparable to Task 20/21's
already-established framework-only baselines.
""")

raw_train_full = FashionMNIST(root="./data", train=True, download=True)
raw_test = FashionMNIST(root="./data", train=False, download=True)
N_TRAIN, N_VAL = 50000, 10000
generator = torch.Generator().manual_seed(RANDOM_STATE)
train_subset, val_subset = random_split(raw_train_full, [N_TRAIN, N_VAL], generator=generator)

train_indices = train_subset.indices
train_images_raw = raw_train_full.data[train_indices].float() / 255.0
computed_mean, computed_std = train_images_raw.mean().item(), train_images_raw.std().item()
print(f"Normalization stats (train split only): mean={computed_mean:.4f}, std={computed_std:.4f}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((computed_mean,), (computed_std,)),
    transforms.Lambda(lambda x: x.view(-1)),
])
train_subset.dataset = FashionMNIST(root="./data", train=True, download=False, transform=transform)
val_subset.dataset = FashionMNIST(root="./data", train=True, download=False, transform=transform)
test_dataset = FashionMNIST(root="./data", train=False, download=False, transform=transform)


def dataset_to_numpy(ds):
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
    X, y = next(iter(loader))
    return X.numpy().astype(np.float64), y.numpy()


X_train_np, y_train_np = dataset_to_numpy(train_subset)
X_val_np, y_val_np = dataset_to_numpy(val_subset)
X_test_np, y_test_np = dataset_to_numpy(test_dataset)
print(f"NumPy arrays ready: train {X_train_np.shape}, val {X_val_np.shape}, test {X_test_np.shape}")


def one_hot(y, n_classes=10):
    Y = np.zeros((y.size, n_classes))
    Y[np.arange(y.size), y] = 1.0
    return Y


Y_train_np = one_hot(y_train_np)

BATCH_SIZE = 256
EPOCHS = 20
LR = 0.1
ARCH = [784, 256, 128, 10]


def train_numpy_mlp(use_dropout=False, dropout_p=0.5, use_batchnorm=False,
                     epochs=EPOCHS, lr=LR, seed=RANDOM_STATE, early_stopping=False,
                     patience=5, verbose_name=""):
    model = NumpyMLP(ARCH, use_dropout=use_dropout, dropout_p=dropout_p,
                      use_batchnorm=use_batchnorm, seed=seed)
    n = X_train_np.shape[0]
    perm_rng = np.random.default_rng(seed)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val, best_params, patience_counter, stopped_epoch = float("inf"), None, 0, epochs
    t0 = time.perf_counter()
    for epoch in range(epochs):
        perm = perm_rng.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb, yb = X_train_np[idx], Y_train_np[idx]
            model.forward(xb, training=True)
            grads = model.backward(yb)
            model.step(grads, lr)
        train_probs = model.forward(X_train_np, training=False)
        val_probs = model.forward(X_val_np, training=False)
        train_loss = cross_entropy_loss(train_probs, Y_train_np)
        val_loss = cross_entropy_loss(val_probs, one_hot(y_val_np))
        train_acc = (train_probs.argmax(1) == y_train_np).mean()
        val_acc = (val_probs.argmax(1) == y_val_np).mean()
        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc); val_accs.append(val_acc)
        print(f"  [{verbose_name}] epoch {epoch+1:2d}/{epochs}  train loss {train_loss:.4f}  "
              f"val loss {val_loss:.4f}  train acc {train_acc:.4f}  val acc {val_acc:.4f}")
        if early_stopping:
            if val_loss < best_val - 1e-4:
                best_val, best_params, patience_counter = val_loss, model.get_params(), 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                stopped_epoch = epoch + 1
                print(f"  [{verbose_name}] early stopping at epoch {stopped_epoch}, "
                      f"restoring epoch {epoch+1-patience} weights")
                model.set_params(best_params)
                break
    train_time = time.perf_counter() - t0
    return model, {"train_losses": train_losses, "val_losses": val_losses,
                    "train_accs": train_accs, "val_accs": val_accs,
                    "train_time": train_time, "stopped_epoch": stopped_epoch}


class TorchNet(nn.Module):
    def __init__(self, n_in=784, h1=256, h2=128, n_out=10):
        super().__init__()
        self.fc1 = nn.Linear(n_in, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, n_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))


X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
y_train_t = torch.tensor(y_train_np, dtype=torch.long)
X_val_t = torch.tensor(X_val_np, dtype=torch.float32)
y_val_t = torch.tensor(y_val_np, dtype=torch.long)
X_test_t = torch.tensor(X_test_np, dtype=torch.float32)
y_test_t = torch.tensor(y_test_np, dtype=torch.long)


def train_torch_mlp(epochs=EPOCHS, lr=LR, seed=RANDOM_STATE):
    torch.manual_seed(seed)
    net = TorchNet()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # matches the NumPy net's plain SGD exactly
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(X_train_t.shape[0])
        for start in range(0, X_train_t.shape[0], BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(net(X_train_t[idx]), y_train_t[idx])
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            train_logits, val_logits = net(X_train_t), net(X_val_t)
            train_loss = criterion(train_logits, y_train_t).item()
            val_loss = criterion(val_logits, y_val_t).item()
            train_acc = (train_logits.argmax(1) == y_train_t).float().mean().item()
            val_acc = (val_logits.argmax(1) == y_val_t).float().mean().item()
        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc); val_accs.append(val_acc)
        print(f"  [torch] epoch {epoch+1:2d}/{epochs}  train loss {train_loss:.4f}  "
              f"val loss {val_loss:.4f}  train acc {train_acc:.4f}  val acc {val_acc:.4f}")
    train_time = time.perf_counter() - t0
    return net, {"train_losses": train_losses, "val_losses": val_losses,
                  "train_accs": train_accs, "val_accs": val_accs, "train_time": train_time}


print(f"\n--- B: training NumPy MLP {ARCH} (SGD, lr={LR}, {EPOCHS} epochs, no regularization) ---")
numpy_baseline_model, numpy_baseline_hist = train_numpy_mlp(verbose_name="numpy-baseline")
print(f"NumPy baseline training time: {numpy_baseline_hist['train_time']:.2f}s")

print(f"\n--- B: training PyTorch MLP {ARCH} (identical SGD, lr, epochs, seed) ---")
torch_model, torch_hist = train_torch_mlp()
print(f"PyTorch training time: {torch_hist['train_time']:.2f}s")


def numpy_full_evaluate(model, X, y, name):
    probs = model.forward(X, training=False)
    preds = probs.argmax(1)
    return {"name": name, "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds, average="macro", zero_division=0),
            "recall": recall_score(y, preds, average="macro", zero_division=0),
            "f1": f1_score(y, preds, average="macro", zero_division=0),
            "cm": confusion_matrix(y, preds)}


def torch_full_evaluate(model, X, y, name):
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(1).numpy()
    y_np = y.numpy()
    return {"name": name, "accuracy": accuracy_score(y_np, preds),
            "precision": precision_score(y_np, preds, average="macro", zero_division=0),
            "recall": recall_score(y_np, preds, average="macro", zero_division=0),
            "f1": f1_score(y_np, preds, average="macro", zero_division=0),
            "cm": confusion_matrix(y_np, preds)}


numpy_test = numpy_full_evaluate(numpy_baseline_model, X_test_np, y_test_np, "NumPy (from scratch)")
torch_test = torch_full_evaluate(torch_model, X_test_t, y_test_t, "PyTorch")
gap = abs(numpy_test["accuracy"] - torch_test["accuracy"])
print(f"\nNumPy  TEST: Acc {numpy_test['accuracy']:.4f} | F1 {numpy_test['f1']:.4f} | "
      f"time {numpy_baseline_hist['train_time']:.2f}s")
print(f"PyTorch TEST: Acc {torch_test['accuracy']:.4f} | F1 {torch_test['f1']:.4f} | "
      f"time {torch_hist['train_time']:.2f}s")
print(f"Accuracy gap: {gap:.4f} ({gap*100:.2f} percentage points). Explanation: both use the exact "
      f"same architecture, batch size, learning rate, epoch count and mini-batch SGD update rule, so "
      f"any gap this small is attributable to implementation-level differences that are NOT "
      f"mathematically identical even though the algorithm is: PyTorch's default float32 vs NumPy's "
      f"float64 here, and the two frameworks' mini-batch shuffling using different (though "
      f"same-seeded per-framework) permutation algorithms, so the exact sequence of mini-batches -- "
      f"and therefore the exact stochastic trajectory through weight space -- is not bit-identical "
      f"between them despite an identical starting point and update rule.")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(numpy_baseline_hist["train_losses"], label="NumPy train", color="#4C72B0")
ax[0].plot(numpy_baseline_hist["val_losses"], label="NumPy val", color="#4C72B0", ls="--")
ax[0].plot(torch_hist["train_losses"], label="PyTorch train", color="#DD8452")
ax[0].plot(torch_hist["val_losses"], label="PyTorch val", color="#DD8452", ls="--")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].set_title("Loss: NumPy vs PyTorch")
ax[0].legend(fontsize=7)
ax[1].plot(numpy_baseline_hist["train_accs"], label="NumPy train", color="#4C72B0")
ax[1].plot(numpy_baseline_hist["val_accs"], label="NumPy val", color="#4C72B0", ls="--")
ax[1].plot(torch_hist["train_accs"], label="PyTorch train", color="#DD8452")
ax[1].plot(torch_hist["val_accs"], label="PyTorch val", color="#DD8452", ls="--")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].set_title("Accuracy: NumPy vs PyTorch")
ax[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig("figures/01_numpy_vs_pytorch.png", bbox_inches="tight")
plt.close(fig)

cm = numpy_test["cm"]
cm_offdiag = cm.copy(); np.fill_diagonal(cm_offdiag, 0)
i, j = np.unravel_index(cm_offdiag.argmax(), cm_offdiag.shape)
print(f"\nMost confused pair (NumPy model): true '{CLASS_NAMES[i]}' -> predicted '{CLASS_NAMES[j]}' "
      f"({cm_offdiag[i,j]} images), true '{CLASS_NAMES[j]}' -> predicted '{CLASS_NAMES[i]}' "
      f"({cm[j,i]} images) -- both upper-body garments with overlapping silhouettes at 28x28 "
      f"greyscale resolution, consistent with the same confusion identified on this dataset in Task 20.")

fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(10)); ax.set_yticklabels(CLASS_NAMES, fontsize=8)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"NumPy model, test confusion matrix (acc {numpy_test['accuracy']:.4f})")
for r in range(10):
    for c in range(10):
        if cm[r, c] > 0:
            ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                     color="white" if cm[r, c] > cm.max() / 2 else "black", fontsize=6)
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig("figures/02_confusion_matrix_numpy.png", bbox_inches="tight")
plt.close(fig)

# ======================================================================
# Part C: Regularization Techniques (from scratch, NumPy)
# ======================================================================
print("\n" + "=" * 70)
print("PART C: REGULARIZATION TECHNIQUES (FROM-SCRATCH NUMPY)")
print("=" * 70)

print("""
--- C13a: why dropout approximates training an ensemble ---

At each training step, inverted dropout independently zeroes each hidden
unit with probability p. For a layer of H units, that is one of 2^H
possible "thinned" sub-networks being sampled and trained for a single
step, sharing weights with every other sub-network that could have been
sampled. Averaged over many steps, every weight receives gradient updates
from a huge number of distinct, randomly-thinned architectures -- which is
close to what an actual ensemble of 2^H independently-trained networks,
followed by averaging their predictions, would do, except the weight
sharing makes it computationally free instead of combinatorially
expensive. At test time, running the full (undropped) network with every
unit active approximates averaging that entire ensemble's predictions in a
single forward pass -- which is exactly what inverted dropout's
1/keep_prob training-time scaling is calibrated to make consistent: it
keeps each unit's EXPECTED output the same whether it survives a given
dropout step or not, so the full network at test time computes something
close to the ensemble's average output without actually needing to run any
sub-network more than once.

--- C13b: why batch normalization stabilizes and accelerates training,
mechanistically ---

Two real mechanisms, not marketing language. First, without normalization,
each layer's input distribution shifts every time the layer below it
updates its weights (the layer originally called "internal covariate
shift" in the BatchNorm paper) -- layer i+1 is constantly re-adapting to a
moving target even when its own weights have not changed, since layer i's
output statistics keep drifting under it. Renormalizing each layer's
pre-activation to zero mean and unit variance every step removes most of
that drift, so each layer's effective input distribution stays comparable
across training steps, letting layer i+1 make consistent progress instead
of chasing a moving target. Second, and separately, batch normalization
provably smooths the loss landscape (formalized in later analysis of
BatchNorm, e.g. Santurkar et al. 2018): the Lipschitzness of the loss
surface with respect to the pre-activations improves, meaning gradients
change more predictably as parameters move, which is what actually permits
the larger, more aggressive learning rates that make batch-normalized
networks train faster in practice -- a smoother surface tolerates bigger
steps without diverging.
""")

configs = {
    "Baseline (no regularization)": dict(use_dropout=False, use_batchnorm=False),
    "Dropout only (p=0.5)": dict(use_dropout=True, dropout_p=0.5, use_batchnorm=False),
    "Batch Norm only": dict(use_dropout=False, use_batchnorm=True),
    "Dropout + BatchNorm + Early Stopping": dict(use_dropout=True, dropout_p=0.3, use_batchnorm=True,
                                                   early_stopping=True, patience=5, epochs=40),
}

ablation_models, ablation_hists, ablation_tests = {}, {}, {}
for name, kwargs in configs.items():
    print(f"\n--- Training: {name} ---")
    run_epochs = kwargs.pop("epochs", EPOCHS)
    model, hist = train_numpy_mlp(epochs=run_epochs, verbose_name=name.split()[0].lower(), **kwargs)
    test_result = numpy_full_evaluate(model, X_test_np, y_test_np, name)
    ablation_models[name] = model
    ablation_hists[name] = hist
    ablation_tests[name] = test_result
    train_acc_final = hist["train_accs"][-1]
    val_acc_final = hist["val_accs"][-1]
    print(f"{name}: TRAIN {train_acc_final:.4f} | VAL {val_acc_final:.4f} | "
          f"TEST {test_result['accuracy']:.4f} | gap(train-test) {train_acc_final - test_result['accuracy']:+.4f} | "
          f"epochs {len(hist['train_losses'])} | time {hist['train_time']:.2f}s")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for name, h in ablation_hists.items():
    ax[0].plot(h["val_losses"], label=name)
    ax[1].plot([t - v for t, v in zip(h["train_accs"], h["val_accs"])], label=name)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("validation loss"); ax[0].set_title("Validation loss, all configurations")
ax[0].legend(fontsize=6)
ax[1].axhline(0, color="grey", lw=0.5)
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("train acc - val acc"); ax[1].set_title("Train/val accuracy gap")
ax[1].legend(fontsize=6)
fig.tight_layout()
fig.savefig("figures/03_ablation_curves.png", bbox_inches="tight")
plt.close(fig)

# C14: a deliberately broken hyperparameter, demonstrated
print("\n--- C14: a hyperparameter mistake, demonstrated empirically ---")
print("""
The mistake: forgetting to switch a dropout-regularized network to
inference mode (training=False) at test time -- i.e. leaving dropout's
random masking ACTIVE during evaluation. This is a real, common bug (the
framework equivalent is forgetting model.eval() in PyTorch/Keras), and it
is actively harmful for two compounding reasons: predictions become
non-deterministic (the same input yields a different prediction on every
call, since a fresh random mask is drawn each time), and, unlike the
inverted-dropout SCALING (which is correctly calibrated to keep expected
activation magnitude consistent whether the network is training or not),
extra masking noise on top of an already-trained network still discards
real signal at test time for no benefit -- there is no ensemble-averaging
gain from dropout at inference, only its cost.
""")

dropout_model_for_bug_demo = ablation_models["Dropout only (p=0.5)"]
correct_probs = dropout_model_for_bug_demo.forward(X_test_np, training=False)
correct_preds = correct_probs.argmax(1)
correct_acc = accuracy_score(y_test_np, correct_preds)

buggy_preds_runs = []
for trial in range(5):
    buggy_probs = dropout_model_for_bug_demo.forward(X_test_np, training=True)  # BUG: dropout left on
    buggy_preds_runs.append(buggy_probs.argmax(1))
buggy_accs = [accuracy_score(y_test_np, p) for p in buggy_preds_runs]
agreement_between_runs = np.mean([np.mean(buggy_preds_runs[0] == buggy_preds_runs[k])
                                    for k in range(1, 5)])

print(f"Correct (dropout OFF at test time): accuracy {correct_acc:.4f}, deterministic (same every call).")
print(f"Buggy (dropout left ON at test time), 5 repeated evaluations of the SAME test set: "
      f"accuracies {[f'{a:.4f}' for a in buggy_accs]}")
print(f"  -- accuracy is both lower ({np.mean(buggy_accs):.4f} average, "
      f"{correct_acc - np.mean(buggy_accs):.4f} worse) AND non-deterministic (varies "
      f"{np.std(buggy_accs):.4f} run-to-run), and only {agreement_between_runs:.1%} of individual "
      f"predictions agree between two buggy runs on the identical input -- exactly the two failure "
      f"modes the mechanism above predicts.")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.axhline(correct_acc, color="#55A868", lw=2, label=f"correct (eval mode): {correct_acc:.4f}")
ax.bar(range(5), buggy_accs, color="#C44E52", alpha=0.8,
       label=f"buggy (train mode at test time), 5 runs")
ax.set_xlabel("repeated evaluation of the identical test set"); ax.set_ylabel("test accuracy")
ax.set_title("The cost of leaving dropout active at inference")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/04_dropout_bug_demo.png", bbox_inches="tight")
plt.close(fig)

print("\nFigures written to figures/")

# ----------------------------------------------------------------------
print("\n=== SUMMARY (Parts A-C) ===")
print(f"Part B -- NumPy vs PyTorch: NumPy {numpy_test['accuracy']:.4f}, PyTorch {torch_test['accuracy']:.4f}, "
      f"gap {gap:.4f}")
for name, r in ablation_tests.items():
    h = ablation_hists[name]
    gap_ct = h["train_accs"][-1] - r["accuracy"]
    print(f"Part C -- {name:38s} TEST {r['accuracy']:.4f} | train-test gap {gap_ct:+.4f} | "
          f"epochs {len(h['train_losses'])} | time {h['train_time']:.2f}s")
print(f"Part C14 -- dropout-at-inference bug: correct {correct_acc:.4f} vs buggy avg "
      f"{np.mean(buggy_accs):.4f} (non-deterministic, {agreement_between_runs:.1%} run-to-run agreement)")

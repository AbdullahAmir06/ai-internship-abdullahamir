"""
PKCERT AI & Software Development Internship, Task 18
Part C: Building & Training a Simple Neural Network (PyTorch)

The same architecture (4 -> 8 -> 3, ReLU hidden, softmax-equivalent output)
and the same Palmer Penguins dataset/split as Task 17, rebuilt as a
torch.nn.Module and trained with PyTorch's autograd + nn.Module API. A
from-scratch NumPy MLP and a scikit-learn MLPClassifier are trained fresh
in this same script, on the same split, so all three timings are measured
on identical hardware in one run rather than compared across sessions.
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
rng = np.random.default_rng(RANDOM_STATE)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

HIDDEN_UNITS, LR, EPOCHS, BATCH_SIZE = 8, 0.1, 300, 16


# ----------------------------------------------------------------------
# The Task 17 NumPy MLP, reproduced here so its training can be timed on
# the same machine, in the same run, as the PyTorch and sklearn models.
# ----------------------------------------------------------------------
def relu_np(x): return np.maximum(0, x)
def relu_deriv_np(x): return (x > 0).astype(float)
def softmax_np(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


class ManualMLP:
    def __init__(self, n_in, n_hidden, n_out, learning_rate=0.1, seed=RANDOM_STATE):
        g = np.random.default_rng(seed)
        self.W1 = g.normal(0, np.sqrt(2.0 / n_in), size=(n_in, n_hidden))
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = g.normal(0, np.sqrt(2.0 / n_hidden), size=(n_hidden, n_out))
        self.b2 = np.zeros((1, n_out))
        self.lr = learning_rate
        self.loss_history = []

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu_np(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = softmax_np(self.Z2)
        return self.A2

    @staticmethod
    def loss(A2, Y, eps=1e-12):
        return -np.mean(np.sum(Y * np.log(np.clip(A2, eps, 1.0)), axis=1))

    def step(self, X, Y):
        self.forward(X)
        N = X.shape[0]
        dZ2 = (self.A2 - Y) / N
        dW2 = self.A1.T @ dZ2
        db2 = dZ2.sum(axis=0, keepdims=True)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_deriv_np(self.Z1)
        dW1 = X.T @ dZ1
        db1 = dZ1.sum(axis=0, keepdims=True)
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2

    def fit(self, X, Y, epochs, batch_size):
        n = X.shape[0]
        for _ in range(epochs):
            perm = rng.permutation(n)
            X_shuf, Y_shuf = X[perm], Y[perm]
            for start in range(0, n, batch_size):
                self.step(X_shuf[start:start + batch_size], Y_shuf[start:start + batch_size])
            self.loss_history.append(self.loss(self.forward(X), Y))
        return self

    def predict(self, X): return np.argmax(self.forward(X), axis=1)


def one_hot_np(y, n_classes):
    Y = np.zeros((y.size, n_classes)); Y[np.arange(y.size), y] = 1.0
    return Y


# ----------------------------------------------------------------------
# C11: the PyTorch nn.Module -- same architecture as Task 17
# ----------------------------------------------------------------------
class PenguinNet(nn.Module):
    def __init__(self, n_in=4, n_hidden=HIDDEN_UNITS, n_out=3):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        # Raw logits are returned deliberately -- nn.CrossEntropyLoss applies
        # log_softmax internally (see the loss justification below), so
        # applying softmax here too would double-apply it.
        x = self.relu(self.fc1(x))
        return self.fc2(x)


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__} (CPU)")

    # --------------------------------------------------------------------
    # Data -- identical split methodology to Task 17
    # --------------------------------------------------------------------
    df = pd.read_csv("penguins.csv").dropna(
        subset=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"])
    FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    species = sorted(df["species"].unique())
    species_to_idx = {s: i for i, s in enumerate(species)}

    X = df[FEATURES].values
    y = df["species"].map(species_to_idx).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
    Y_train_oh = one_hot_np(y_train, 3)
    print(f"Train: {X_train_s.shape[0]} | Test: {X_test_s.shape[0]}")

    results = {}

    # --------------------------------------------------------------------
    # 1) ManualMLP (Task 17), timed fresh on this machine
    # --------------------------------------------------------------------
    manual_model = ManualMLP(4, HIDDEN_UNITS, 3, learning_rate=LR)
    t0 = time.perf_counter()
    manual_model.fit(X_train_s, Y_train_oh, epochs=EPOCHS, batch_size=BATCH_SIZE)
    manual_time = time.perf_counter() - t0
    pred_manual = manual_model.predict(X_test_s)

    # --------------------------------------------------------------------
    # 2) sklearn MLPClassifier baseline, same architecture/optimizer
    # --------------------------------------------------------------------
    sk_model = MLPClassifier(hidden_layer_sizes=(HIDDEN_UNITS,), activation="relu",
                              solver="sgd", learning_rate_init=LR, max_iter=EPOCHS,
                              batch_size=BATCH_SIZE, random_state=RANDOM_STATE)
    t0 = time.perf_counter()
    sk_model.fit(X_train_s, y_train)
    sklearn_time = time.perf_counter() - t0
    pred_sklearn = sk_model.predict(X_test_s)

    # --------------------------------------------------------------------
    # C12-C15: the PyTorch model -- loss, optimizer, explicit training loop
    # --------------------------------------------------------------------
    # Loss: CrossEntropyLoss combines log_softmax + negative-log-likelihood in
    # one numerically stable call (log-sum-exp internally, avoiding the
    # separate softmax-then-log Task 17 had to guard by hand with clipping),
    # and is the standard choice for single-label multi-class classification
    # from raw logits -- the direct PyTorch analogue of the manually derived
    # softmax + cross-entropy pair from Task 17.
    # Optimizer: plain SGD at the same learning rate as Task 17/sklearn above,
    # specifically so this run is a fair, matched comparison; Adam is tried
    # separately below as the hyperparameter experiment.
    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)

    def train_pytorch(optimizer_name="sgd", lr=LR, epochs=EPOCHS, seed=RANDOM_STATE):
        torch.manual_seed(seed)
        net = PenguinNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = (torch.optim.SGD(net.parameters(), lr=lr) if optimizer_name == "sgd"
                     else torch.optim.Adam(net.parameters(), lr=lr))
        loss_history = []
        n = X_train_t.shape[0]
        for _epoch in range(epochs):
            perm = torch.randperm(n)
            for start in range(0, n, BATCH_SIZE):
                idx = perm[start:start + BATCH_SIZE]
                xb, yb = X_train_t[idx], y_train_t[idx]

                optimizer.zero_grad()          # C15: clear stale gradients first
                logits = net(xb)               # forward pass
                loss = criterion(logits, yb)   # loss computation
                loss.backward()                # backward pass (autograd)
                optimizer.step()               # parameter update
            with torch.no_grad():              # full-training-set loss for the curve;
                epoch_loss = criterion(net(X_train_t), y_train_t).item()  # no graph needed
            loss_history.append(epoch_loss)
        return net, loss_history

    t0 = time.perf_counter()
    torch_net, torch_loss_history = train_pytorch("sgd", LR, EPOCHS)
    torch_time = time.perf_counter() - t0

    with torch.no_grad():
        pred_torch = torch_net(X_test_t).argmax(dim=1).numpy()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(torch_loss_history, color="#4C72B0")
    ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy loss (train)")
    ax.set_title(f"PyTorch training loss (SGD, lr={LR}, hidden={HIDDEN_UNITS})")
    fig.tight_layout()
    fig.savefig("figures/01_pytorch_loss_curve.png", bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------------------
    # C16: evaluation, and C-required 3-way comparison (Manual / sklearn / PyTorch)
    # --------------------------------------------------------------------
    def report(name, y_true, y_pred, fit_time):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n=== {name} ===")
        print(f"Accuracy {acc:.4f} | Macro-Prec {prec:.4f} | Macro-Rec {rec:.4f} | "
              f"Macro-F1 {f1:.4f} | Fit time {fit_time:.3f}s")
        print("Confusion matrix:\n", cm)
        return {"name": name, "accuracy": acc, "precision": prec, "recall": rec,
                "f1": f1, "cm": cm, "fit_time": fit_time}

    r_manual = report("ManualMLP (NumPy, Task 17)", y_test, pred_manual, manual_time)
    r_sklearn = report("sklearn MLPClassifier", y_test, pred_sklearn, sklearn_time)
    r_torch = report("PyTorch nn.Module (SGD)", y_test, pred_torch, torch_time)

    all_results = [r_manual, r_sklearn, r_torch]

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for a, r in zip(ax, all_results):
        cm = r["cm"]
        a.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                a.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
        a.set_xticks(range(3)); a.set_xticklabels(species, rotation=20, fontsize=7)
        a.set_yticks(range(3)); a.set_yticklabels(species, fontsize=7)
        a.set_xlabel("Predicted"); a.set_ylabel("Actual")
        a.set_title(f"{r['name']}\nAcc {r['accuracy']:.3f}  F1 {r['f1']:.3f}", fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/02_confusion_matrices_3way.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    names = [r["name"] for r in all_results]
    metrics = ["accuracy", "precision", "recall", "f1"]
    xw = np.arange(len(metrics)); w = 0.25
    for i, r in enumerate(all_results):
        ax[0].bar(xw + (i - 1) * w, [r[m] for m in metrics], w, label=r["name"])
    ax[0].set_xticks(xw); ax[0].set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
    ax[0].set_ylim(0, 1.05); ax[0].legend(fontsize=7); ax[0].set_title("Metrics: all three implementations")
    ax[1].bar(names, [r["fit_time"] for r in all_results], color=["#8172B2", "#DD8452", "#4C72B0"])
    ax[1].set_ylabel("training time, seconds")
    ax[1].set_title(f"Training time ({EPOCHS} epochs each, same hardware)")
    ax[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig("figures/03_metrics_and_timing.png", bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------------------
    # C17: hyperparameter experiment -- optimizer choice
    # --------------------------------------------------------------------
    print("\n--- Hyperparameter experiment: optimizer choice (SGD vs Adam) ---")
    optimizer_configs = [("SGD (lr=0.1)", "sgd", 0.1), ("Adam (lr=0.1)", "adam", 0.1),
                          ("Adam (lr=0.01)", "adam", 0.01)]
    opt_results = {}
    for label, opt_name, lr in optimizer_configs:
        net, hist = train_pytorch(opt_name, lr, EPOCHS)
        with torch.no_grad():
            acc = (net(X_test_t).argmax(dim=1).numpy() == y_test).mean()
        opt_results[label] = {"loss_history": hist, "test_acc": acc}
        print(f"  {label:16s} final train loss {hist[-1]:.4f} | test accuracy {acc:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for label in opt_results:
        ax[0].plot(opt_results[label]["loss_history"], label=label)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training loss"); ax[0].set_yscale("log")
    ax[0].set_title("Loss curves by optimizer"); ax[0].legend(fontsize=8)
    ax[1].bar(list(opt_results.keys()), [opt_results[k]["test_acc"] for k in opt_results], color="#55A868")
    ax[1].set_ylabel("test accuracy"); ax[1].set_title("Final test accuracy by optimizer")
    ax[1].set_ylim(0, 1.05); ax[1].tick_params(axis="x", rotation=10)
    fig.tight_layout()
    fig.savefig("figures/04_optimizer_experiment.png", bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------------------
    # Part D.1: save and reload the trained model with torch.save/state_dict
    # --------------------------------------------------------------------
    torch.save(torch_net.state_dict(), "pytorch_mlp_state_dict.pt")
    reloaded_net = PenguinNet()
    reloaded_net.load_state_dict(torch.load("pytorch_mlp_state_dict.pt", weights_only=True))
    reloaded_net.eval()
    with torch.no_grad():
        pred_reloaded = reloaded_net(X_test_t).argmax(dim=1).numpy()
    print(f"\nReloaded state_dict reproduces identical test predictions: "
          f"{np.array_equal(pred_reloaded, pred_torch)}")

    print("\nFigures written to figures/")
    print("\nSUMMARY")
    for r in all_results:
        print(f"{r['name']:32s} Acc {r['accuracy']:.4f}  F1 {r['f1']:.4f}  "
              f"Fit time {r['fit_time']:.3f}s")
    best_opt = max(opt_results, key=lambda k: opt_results[k]["test_acc"])
    print(f"Best optimizer config: {best_opt} (test accuracy {opt_results[best_opt]['test_acc']:.4f})")

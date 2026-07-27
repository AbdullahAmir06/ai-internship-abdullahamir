"""
PKCERT AI & Software Development Internship, Task 17
Part C: Forward & Backpropagation Mini-Project

A 2-layer MLP (4 -> hidden -> 3) with forward propagation and
backpropagation implemented entirely by hand in NumPy -- no autograd
framework computes any gradient here. scikit-learn is used only for the
train/test split, feature scaling, evaluation metrics, and the permitted
MLPClassifier baseline; none of those are the "core computation" the task
restricts.

Dataset: Palmer Penguins (Gorman, Williams & Fraser, 2014) -- 4 numeric
bill/flipper/mass measurements, 3 species. Not used in any earlier task.

--------------------------------------------------------------------------
DERIVATION (matrix form), included here as plain text as well as in
Report.tex / the notebook.

Architecture: X (N,d) -> [W1,b1] -> Z1 -> ReLU -> A1 -> [W2,b2] -> Z2
              -> softmax -> A2 (predicted class probabilities)
    d = 4 input features, h = hidden units, c = 3 output classes.

Forward:
    Z1 = X W1 + b1                  (N,d)(d,h) + (1,h) -> (N,h)
    A1 = ReLU(Z1) = max(0, Z1)      (N,h)
    Z2 = A1 W2 + b2                 (N,h)(h,c) + (1,c) -> (N,c)
    A2 = softmax(Z2), row-wise      (N,c)

Loss (mean categorical cross-entropy over a batch of N samples, Y one-hot):
    L = -(1/N) * sum_n sum_k Y[n,k] * log(A2[n,k])

Backward (chain rule, matrix form):
    The softmax + cross-entropy combination has a well-known closed-form
    gradient: for a single example, d(loss)/d(Z2_j) = A2_j - Y_j. Batched
    and pre-divided by N (so every later gradient is already the correct
    mean gradient, no extra division needed downstream):
        dZ2 = (A2 - Y) / N                      (N,c)
        dW2 = A1^T @ dZ2                        (h,N)(N,c) -> (h,c)
        db2 = sum over N of dZ2                 (1,c)
        dA1 = dZ2 @ W2^T                        (N,c)(c,h) -> (N,h)
        dZ1 = dA1 * ReLU'(Z1)                   (N,h) elementwise
              where ReLU'(Z1) = 1 if Z1>0 else 0
        dW1 = X^T @ dZ1                         (d,N)(N,h) -> (d,h)
        db1 = sum over N of dZ1                 (1,h)

Update (gradient descent, learning rate lr):
    W2 <- W2 - lr * dW2 ;  b2 <- b2 - lr * db2
    W1 <- W1 - lr * dW1 ;  b1 <- b1 - lr * db1
--------------------------------------------------------------------------
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

from part_b_activations import relu, relu_deriv, softmax

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


class ManualMLP:
    """A 2-layer (1 hidden layer) MLP: ReLU hidden activation, softmax
    output, categorical cross-entropy loss. Forward and backward passes
    are both hand-written NumPy -- see the module docstring for the
    derivation each method below implements."""

    def __init__(self, n_in, n_hidden, n_out, learning_rate=0.1, seed=RANDOM_STATE):
        g = np.random.default_rng(seed)
        # He initialisation: matches ReLU's active-half variance, keeps
        # early activations from exploding or vanishing before training
        # even starts.
        self.W1 = g.normal(0, np.sqrt(2.0 / n_in), size=(n_in, n_hidden))
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = g.normal(0, np.sqrt(2.0 / n_hidden), size=(n_hidden, n_out))
        self.b2 = np.zeros((1, n_out))
        self.lr = learning_rate
        self.loss_history = []

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    @staticmethod
    def loss(A2, Y, eps=1e-12):
        # Clip before log(): a perfectly confident wrong prediction would
        # otherwise send log(0) to -inf and the whole batch's loss to nan.
        return -np.mean(np.sum(Y * np.log(np.clip(A2, eps, 1.0)), axis=1))

    def backward(self, X, Y):
        N = X.shape[0]
        dZ2 = (self.A2 - Y) / N
        dW2 = self.A1.T @ dZ2
        db2 = dZ2.sum(axis=0, keepdims=True)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_deriv(self.Z1)
        dW1 = X.T @ dZ1
        db1 = dZ1.sum(axis=0, keepdims=True)
        return dW1, db1, dW2, db2

    def step(self, X, Y):
        self.forward(X)
        dW1, db1, dW2, db2 = self.backward(X, Y)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def fit(self, X, Y, epochs=300, batch_size=16, verbose_every=50):
        n = X.shape[0]
        for epoch in range(epochs):
            perm = rng.permutation(n)
            X_shuf, Y_shuf = X[perm], Y[perm]
            for start in range(0, n, batch_size):
                xb = X_shuf[start:start + batch_size]
                yb = Y_shuf[start:start + batch_size]
                self.step(xb, yb)
            epoch_loss = self.loss(self.forward(X), Y)
            self.loss_history.append(epoch_loss)
            if verbose_every and (epoch + 1) % verbose_every == 0:
                print(f"  epoch {epoch + 1:4d}/{epochs}  loss {epoch_loss:.4f}")
        return self

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}


def one_hot(y, n_classes):
    Y = np.zeros((y.size, n_classes))
    Y[np.arange(y.size), y] = 1.0
    return Y


def numerical_gradient_check(model, X, Y, param_name, eps=1e-5, n_checks=6):
    """Finite-difference gradient check: perturb a handful of individual
    entries of one parameter matrix by +-eps, measure the resulting change
    in loss, and compare that numerical slope against the analytical
    gradient the backward() method computed. This is the standard way to
    catch a sign error or a transposed matrix in a hand-derived backprop
    implementation -- and the only truly convincing evidence that the
    derivation above was implemented correctly."""
    model.forward(X)
    dW1, db1, dW2, db2 = model.backward(X, Y)
    analytical = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}[param_name]
    param = getattr(model, param_name)
    idx = [tuple(rng.integers(0, s) for s in param.shape) for _ in range(n_checks)]
    max_rel_err = 0.0
    rows = []
    for ix in idx:
        orig = param[ix]
        param[ix] = orig + eps
        loss_plus = model.loss(model.forward(X), Y)
        param[ix] = orig - eps
        loss_minus = model.loss(model.forward(X), Y)
        param[ix] = orig
        numerical = (loss_plus - loss_minus) / (2 * eps)
        analytic = analytical[ix]
        rel_err = abs(numerical - analytic) / max(1e-8, abs(numerical) + abs(analytic))
        max_rel_err = max(max_rel_err, rel_err)
        rows.append((ix, numerical, analytic, rel_err))
    return max_rel_err, rows


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # C11: dataset
    # ------------------------------------------------------------------
    df = pd.read_csv("penguins.csv").dropna(
        subset=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"])
    FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    species = sorted(df["species"].unique())
    species_to_idx = {s: i for i, s in enumerate(species)}
    print("=== Part C: Palmer Penguins (4 features, 3 classes) ===")
    print(f"Shape after dropping rows missing a measurement: {df.shape}")
    print(f"Classes: {species} -> {species_to_idx}")
    print(df["species"].value_counts().to_string())

    X = df[FEATURES].values
    y = df["species"].map(species_to_idx).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    # StandardScaler is preprocessing (feature scaling), not the network's
    # core forward/backward computation, so it is fine to use here -- the
    # restriction is specifically on autograd/training frameworks.
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
    Y_train = one_hot(y_train, 3)
    Y_test = one_hot(y_test, 3)
    print(f"Train: {X_train_s.shape[0]} | Test: {X_test_s.shape[0]}")

    # ------------------------------------------------------------------
    # Gradient check -- verify the hand-derived backward() before trusting
    # any result trained with it.
    # ------------------------------------------------------------------
    check_model = ManualMLP(4, 8, 3, learning_rate=0.1)
    print("\n--- Numerical gradient check (finite differences vs analytical backprop) ---")
    for pname in ["W1", "b1", "W2", "b2"]:
        max_err, _ = numerical_gradient_check(check_model, X_train_s[:32], Y_train[:32], pname)
        print(f"  max relative error on {pname}: {max_err:.2e} "
              f"({'PASS' if max_err < 1e-5 else 'FAIL'}, threshold 1e-5)")

    # ------------------------------------------------------------------
    # C14/C15: train the manual network, track the loss curve
    # ------------------------------------------------------------------
    HIDDEN_UNITS = 8
    LR = 0.1
    EPOCHS = 300
    print(f"\n--- Training ManualMLP(4 -> {HIDDEN_UNITS} -> 3), lr={LR}, {EPOCHS} epochs ---")
    model = ManualMLP(4, HIDDEN_UNITS, 3, learning_rate=LR)
    model.fit(X_train_s, Y_train, epochs=EPOCHS, batch_size=16, verbose_every=50)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(model.loss_history, color="#4C72B0")
    ax.set_xlabel("epoch"); ax.set_ylabel("mean cross-entropy loss")
    ax.set_title(f"ManualMLP training loss (lr={LR}, hidden={HIDDEN_UNITS})")
    fig.tight_layout()
    fig.savefig("figures/04_loss_curve.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # C16: evaluate, and compare against sklearn's MLPClassifier
    # ------------------------------------------------------------------
    pred_manual = model.predict(X_test_s)

    sk_model = MLPClassifier(hidden_layer_sizes=(HIDDEN_UNITS,), activation="relu",
                              solver="sgd", learning_rate_init=LR, max_iter=EPOCHS,
                              batch_size=16, random_state=RANDOM_STATE)
    sk_model.fit(X_train_s, y_train)
    pred_sklearn = sk_model.predict(X_test_s)

    def report(name, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n=== {name} ===")
        print(f"Accuracy {acc:.4f} | Macro-Precision {prec:.4f} | Macro-Recall {rec:.4f} | "
              f"Macro-F1 {f1:.4f}")
        print("Confusion matrix (rows=actual, cols=predicted):\n", cm)
        return {"name": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cm": cm}

    r_manual = report("ManualMLP (from-scratch NumPy)", y_test, pred_manual)
    r_sklearn = report("sklearn MLPClassifier (same architecture)", y_test, pred_sklearn)

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for a, r in zip(ax, [r_manual, r_sklearn]):
        cm = r["cm"]
        a.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                a.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
        a.set_xticks(range(3)); a.set_xticklabels(species, rotation=20, fontsize=8)
        a.set_yticks(range(3)); a.set_yticklabels(species, fontsize=8)
        a.set_xlabel("Predicted"); a.set_ylabel("Actual")
        a.set_title(f"{r['name']}\nAcc {r['accuracy']:.3f}  F1 {r['f1']:.3f}", fontsize=9)
    fig.tight_layout()
    fig.savefig("figures/05_confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    metrics = ["accuracy", "precision", "recall", "f1"]
    xw = np.arange(len(metrics)); w = 0.35
    ax.bar(xw - w / 2, [r_manual[m] for m in metrics], w, label="ManualMLP", color="#4C72B0")
    ax.bar(xw + w / 2, [r_sklearn[m] for m in metrics], w, label="sklearn MLPClassifier", color="#DD8452")
    ax.set_xticks(xw); ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
    ax.set_ylim(0, 1.05)
    ax.set_title("ManualMLP vs sklearn MLPClassifier")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/06_metrics_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # C17: hyperparameter experiment -- learning rate
    # ------------------------------------------------------------------
    print("\n--- Hyperparameter experiment: learning rate ---")
    lr_values = [0.001, 0.01, 0.1, 1.0]
    lr_results = {}
    for lr in lr_values:
        m = ManualMLP(4, HIDDEN_UNITS, 3, learning_rate=lr)
        m.fit(X_train_s, Y_train, epochs=EPOCHS, batch_size=16, verbose_every=0)
        acc = (m.predict(X_test_s) == y_test).mean()
        lr_results[lr] = {"loss_history": m.loss_history, "test_acc": acc}
        print(f"  lr={lr:<6} final train loss {m.loss_history[-1]:.4f} | test accuracy {acc:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for lr in lr_values:
        ax[0].plot(lr_results[lr]["loss_history"], label=f"lr={lr}")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training loss")
    ax[0].set_title("Loss curves by learning rate")
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=8)
    ax[1].bar([str(lr) for lr in lr_values], [lr_results[lr]["test_acc"] for lr in lr_values],
              color="#55A868")
    ax[1].set_xlabel("learning rate"); ax[1].set_ylabel("test accuracy")
    ax[1].set_title("Final test accuracy by learning rate")
    ax[1].set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig("figures/07_lr_experiment.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Part D.18: save the final trained model's parameters
    # ------------------------------------------------------------------
    with open("manual_mlp_params.pkl", "wb") as f:
        pickle.dump({"params": model.params(), "scaler_mean": scaler.mean_,
                     "scaler_scale": scaler.scale_, "species": species,
                     "hidden_units": HIDDEN_UNITS, "learning_rate": LR}, f)
    print("\nSaved trained ManualMLP parameters to manual_mlp_params.pkl")

    # Verify the save/load round-trip reproduces identical predictions.
    with open("manual_mlp_params.pkl", "rb") as f:
        loaded = pickle.load(f)
    reloaded = ManualMLP(4, HIDDEN_UNITS, 3)
    reloaded.W1, reloaded.b1 = loaded["params"]["W1"], loaded["params"]["b1"]
    reloaded.W2, reloaded.b2 = loaded["params"]["W2"], loaded["params"]["b2"]
    match = np.array_equal(reloaded.predict(X_test_s), pred_manual)
    print(f"Reloaded model reproduces identical test predictions: {match}")

    print("\nFigures written to figures/")
    print("\nSUMMARY")
    print(f"ManualMLP:  Acc {r_manual['accuracy']:.4f}  F1 {r_manual['f1']:.4f}")
    print(f"sklearn:    Acc {r_sklearn['accuracy']:.4f}  F1 {r_sklearn['f1']:.4f}")
    print(f"Best learning rate tested: "
          f"{max(lr_results, key=lambda k: lr_results[k]['test_acc'])} "
          f"(test accuracy {max(r['test_acc'] for r in lr_results.values()):.4f})")

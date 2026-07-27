"""
PKCERT AI & Software Development Internship, Task 17
Part A: Perceptron Fundamentals

A single-layer perceptron implemented from scratch with only NumPy (no
scikit-learn, TensorFlow or PyTorch anywhere in this file). Trained on a
linearly separable two-class subset of Iris, then shown failing on XOR.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


class Perceptron:
    """Single-layer perceptron trained with the classic perceptron learning
    rule: w <- w + lr * (y - y_hat) * x, updated one sample at a time,
    using a hard step activation. Pure NumPy, no autodiff, no framework."""

    def __init__(self, n_features, learning_rate=0.1, n_epochs=20):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.history = []  # (w, b) snapshot after each epoch
        self.errors_per_epoch = []

    @staticmethod
    def _step(z):
        return np.where(z >= 0, 1, 0)

    def predict(self, X):
        return self._step(X @ self.w + self.b)

    def fit(self, X, y):
        self.history.append((self.w.copy(), self.b))
        for epoch in range(self.n_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                y_hat = self._step(xi @ self.w + self.b)
                update = self.lr * (yi - y_hat)
                if update != 0.0:
                    self.w += update * xi
                    self.b += update
                    errors += 1
            self.errors_per_epoch.append(errors)
            self.history.append((self.w.copy(), self.b))
            if errors == 0:
                break
        return self


# ----------------------------------------------------------------------
# A2/A3: train on a linearly separable subset of Iris
# ----------------------------------------------------------------------
# Iris is a sklearn *dataset loader* (bundled data, not a training/eval
# framework), used here only to obtain raw feature/label arrays -- all of
# the actual learning below is the from-scratch Perceptron class above.
from sklearn.datasets import load_iris  # noqa: E402

iris = load_iris()
# Setosa (0) vs Versicolor (1), the textbook example of a linearly
# separable pair in Iris; petal length & width are the most separable
# 2D projection, which also makes the decision boundary easy to plot.
mask = iris.target < 2
X_iris = iris.data[mask][:, [2, 3]]  # petal length, petal width
y_iris = iris.target[mask]
feat_names = ["petal length (cm)", "petal width (cm)"]

# Standardise by hand (mean/std), no sklearn preprocessing involved.
X_mean, X_std = X_iris.mean(axis=0), X_iris.std(axis=0)
X_iris_s = (X_iris - X_mean) / X_std

perm = rng.permutation(len(X_iris_s))
X_iris_s, y_iris = X_iris_s[perm], y_iris[perm]

p_iris = Perceptron(n_features=2, learning_rate=0.1, n_epochs=20)
p_iris.fit(X_iris_s, y_iris)
train_acc = (p_iris.predict(X_iris_s) == y_iris).mean()
print("=== Part A: Perceptron on Iris (Setosa vs Versicolor) ===")
print(f"Features used: {feat_names}")
print(f"Epochs to convergence: {len(p_iris.errors_per_epoch)} "
      f"(errors per epoch: {p_iris.errors_per_epoch})")
print(f"Final training accuracy: {train_acc:.4f}")
print(f"Final weights: {p_iris.w}, bias: {p_iris.b:.4f}")

# ----------------------------------------------------------------------
# A3: decision boundary evolution over epochs
# ----------------------------------------------------------------------
snapshot_epochs = sorted(set([0, 1, 2, len(p_iris.history) - 1]))
snapshot_epochs = [e for e in snapshot_epochs if e < len(p_iris.history)]
fig, axes = plt.subplots(1, len(snapshot_epochs), figsize=(4 * len(snapshot_epochs), 4), sharey=True)
xx, yy = np.meshgrid(np.linspace(X_iris_s[:, 0].min() - 0.5, X_iris_s[:, 0].max() + 0.5, 200),
                      np.linspace(X_iris_s[:, 1].min() - 0.5, X_iris_s[:, 1].max() + 0.5, 200))
for ax, e in zip(axes, snapshot_epochs):
    w_e, b_e = p_iris.history[e]
    Z = Perceptron._step((np.c_[xx.ravel(), yy.ravel()] @ w_e) + b_e).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, levels=[-0.5, 0.5, 1.5], colors=["#4C72B0", "#DD8452"])
    ax.scatter(X_iris_s[y_iris == 0, 0], X_iris_s[y_iris == 0, 1], s=15, color="#4C72B0", label="setosa")
    ax.scatter(X_iris_s[y_iris == 1, 0], X_iris_s[y_iris == 1, 1], s=15, color="#DD8452", label="versicolor")
    ax.set_title(f"epoch {e}" if e > 0 else "epoch 0 (init)")
    ax.set_xlabel(feat_names[0] + " (scaled)")
axes[0].set_ylabel(feat_names[1] + " (scaled)")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Perceptron decision boundary evolving over training")
fig.tight_layout()
fig.savefig("figures/01_perceptron_boundary_evolution.png", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------
# A4: the same perceptron on XOR -- demonstrate the failure
# ----------------------------------------------------------------------
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])

p_xor = Perceptron(n_features=2, learning_rate=0.1, n_epochs=50)
p_xor.fit(X_xor, y_xor)
xor_acc = (p_xor.predict(X_xor) == y_xor).mean()
print("\n=== Part A4: the same perceptron on XOR ===")
print(f"Ran the full {p_xor.n_epochs} epochs without reaching zero errors "
      f"(errors per epoch: {p_xor.errors_per_epoch})")
print(f"Final training accuracy: {xor_acc:.4f} (a single linear boundary cannot separate XOR)")

fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
w_f, b_f = p_xor.history[-1]
xx2, yy2 = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
Z2 = Perceptron._step((np.c_[xx2.ravel(), yy2.ravel()] @ w_f) + b_f).reshape(xx2.shape)
ax[0].contourf(xx2, yy2, Z2, alpha=0.25, levels=[-0.5, 0.5, 1.5], colors=["#4C72B0", "#DD8452"])
for cls, marker in [(0, "o"), (1, "x")]:
    pts = X_xor[y_xor == cls]
    ax[0].scatter(pts[:, 0], pts[:, 1], s=120, marker=marker,
                  color="black", label=f"XOR = {cls}")
ax[0].set_xlim(-0.5, 1.5); ax[0].set_ylim(-0.5, 1.5)
ax[0].set_xlabel("x1"); ax[0].set_ylabel("x2")
ax[0].set_title("Best boundary the perceptron could find")
ax[0].legend(fontsize=8)
ax[1].plot(range(1, len(p_xor.errors_per_epoch) + 1), p_xor.errors_per_epoch,
           marker="o", ms=3, color="#C44E52")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("misclassifications (of 4)")
ax[1].set_title("Errors never reach 0 -- no convergence")
fig.tight_layout()
fig.savefig("figures/02_perceptron_xor_failure.png", bbox_inches="tight")
plt.close(fig)

print("""
Why XOR defeats a single perceptron: a perceptron's decision boundary is the
single straight line (hyperplane) w.x + b = 0. XOR's four points require two
separate diagonal boundaries -- (0,0) and (1,1) on one side, (0,1) and (1,0)
on the other -- which is not a linearly separable arrangement in 2D. No
choice of w and b can put a single straight line between those two classes,
which is exactly what the flat error curve above shows: the learning rule
keeps correcting one misclassified point only to break another.
""")

print("""
Perceptron vs logistic regression: both are linear classifiers computing the
same score z = w.x + b, and both learn by adjusting w and b in response to
the labelled examples. They differ in what happens to z, and how that
choice is used to learn. The perceptron applies a hard step function --
predict 1 if z >= 0, else 0 -- and updates weights only on misclassified
points, by an amount proportional to the raw error (y - y_hat) in {-1, 0,
1}; there is no notion of a smooth loss surface, just a mistake-driven
rule. Logistic regression applies the smooth sigmoid function to z to
produce a probability, defines a smooth cross-entropy loss over that
probability, and updates every point's weights by gradient descent on that
loss (the gradient turns out to have the strikingly similar form
(y - p) * x, but p is a continuous probability, not a hard 0/1 vote). In
the limit as the sigmoid's steepness goes to infinity, logistic regression's
smooth boundary collapses to the perceptron's hard one -- the perceptron is,
informally, the "hard-threshold, mistake-driven" special case of the same
linear model that logistic regression fits with a smooth, probabilistic
loss and gradient descent.
""")

print("Figures written to figures/")

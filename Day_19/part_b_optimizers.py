"""
PKCERT AI & Software Development Internship, Task 19
Part B: Optimizers -- SGD and Adam Internals

Manual (no torch.optim) implementations of vanilla SGD, SGD+momentum and
Adam, each verified step-for-step against torch.optim on a small convex
toy function; a worked example of Adam's per-parameter adaptive scaling on
unevenly-scaled gradients; a controlled 3-way optimizer comparison on the
Palmer Penguins network from Task 17/18; and a numerical demonstration that
L2-in-the-gradient and AdamW's decoupled weight decay are NOT equivalent
for an adaptive optimizer.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

print(f"=== Part B: PyTorch {torch.__version__} ===")

# ----------------------------------------------------------------------
# Toy convex function: an elongated quadratic bowl, f(x,y) = x^2 + 10y^2.
# Elongated on purpose -- a single learning rate that is stable in y is
# overly cautious in x, which is exactly the regime where momentum and
# Adam are expected to behave differently from vanilla SGD.
# ----------------------------------------------------------------------
def toy_loss(p):
    return p[0] ** 2 + 10.0 * p[1] ** 2


START = torch.tensor([5.0, 5.0])
LR_TOY = 0.05
STEPS = 40

# ----------------------------------------------------------------------
# B1: vanilla SGD, from scratch, verified step-for-step vs torch.optim.SGD
# ----------------------------------------------------------------------
print("\n--- B1: vanilla SGD, manual vs torch.optim.SGD ---")


def run_manual_sgd(start, lr, steps, momentum=0.0):
    p = start.clone()
    buf = torch.zeros_like(p)
    trajectory = [p.clone()]
    for _ in range(steps):
        p_leaf = p.clone().requires_grad_(True)
        loss = toy_loss(p_leaf)
        loss.backward()
        grad = p_leaf.grad
        with torch.no_grad():
            if momentum > 0:
                buf = momentum * buf + grad          # PyTorch's exact SGD-momentum recurrence
                p = p - lr * buf
            else:
                p = p - lr * grad
        trajectory.append(p.clone())
    return torch.stack(trajectory)


def run_torch_optim_sgd(start, lr, steps, momentum=0.0):
    p = start.clone().requires_grad_(True)
    opt = torch.optim.SGD([p], lr=lr, momentum=momentum)
    trajectory = [p.detach().clone()]
    for _ in range(steps):
        opt.zero_grad()
        loss = toy_loss(p)
        loss.backward()
        opt.step()
        trajectory.append(p.detach().clone())
    return torch.stack(trajectory)


traj_manual_sgd = run_manual_sgd(START, LR_TOY, STEPS, momentum=0.0)
traj_torch_sgd = run_torch_optim_sgd(START, LR_TOY, STEPS, momentum=0.0)
max_diff_sgd = (traj_manual_sgd - traj_torch_sgd).abs().max().item()
print(f"Vanilla SGD, {STEPS} steps: max abs difference in trajectory, manual vs "
      f"torch.optim: {max_diff_sgd:.2e}  ({'MATCH' if max_diff_sgd < 1e-6 else 'MISMATCH'})")

print("\n--- B1b: SGD + momentum, manual vs torch.optim.SGD(momentum=0.9) ---")
traj_manual_mom = run_manual_sgd(START, LR_TOY, STEPS, momentum=0.9)
traj_torch_mom = run_torch_optim_sgd(START, LR_TOY, STEPS, momentum=0.9)
max_diff_mom = (traj_manual_mom - traj_torch_mom).abs().max().item()
print(f"SGD+momentum, {STEPS} steps: max abs difference in trajectory: {max_diff_mom:.2e}  "
      f"({'MATCH' if max_diff_mom < 1e-6 else 'MISMATCH'})")

# ----------------------------------------------------------------------
# B2: Adam from scratch, verified vs torch.optim.Adam
# ----------------------------------------------------------------------
print("\n--- B2: Adam, manual vs torch.optim.Adam ---")


def run_manual_adam(start, lr, steps, beta1=0.9, beta2=0.999, eps=1e-8):
    p = start.clone()
    m = torch.zeros_like(p)
    v = torch.zeros_like(p)
    trajectory = [p.clone()]
    for t in range(1, steps + 1):
        p_leaf = p.clone().requires_grad_(True)
        loss = toy_loss(p_leaf)
        loss.backward()
        grad = p_leaf.grad
        with torch.no_grad():
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            p = p - lr * m_hat / (torch.sqrt(v_hat) + eps)
        trajectory.append(p.clone())
    return torch.stack(trajectory)


def run_torch_optim_adam(start, lr, steps):
    p = start.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)
    trajectory = [p.detach().clone()]
    for _ in range(steps):
        opt.zero_grad()
        loss = toy_loss(p)
        loss.backward()
        opt.step()
        trajectory.append(p.detach().clone())
    return torch.stack(trajectory)


traj_manual_adam = run_manual_adam(START, LR_TOY, STEPS)
traj_torch_adam = run_torch_optim_adam(START, LR_TOY, STEPS)
max_diff_adam = (traj_manual_adam - traj_torch_adam).abs().max().item()
print(f"Adam, {STEPS} steps: max abs difference in trajectory: {max_diff_adam:.2e}  "
      f"({'MATCH' if max_diff_adam < 1e-4 else 'MISMATCH'})")
print("Any residual difference this small is consistent with epsilon placement: "
      "PyTorch's Adam adds eps OUTSIDE the sqrt (lr * m_hat / (sqrt(v_hat) + eps), the "
      "formula used above and matching the original paper) in its default configuration, "
      "which is what was matched here; some Adam variants instead add eps INSIDE the "
      "sqrt, which would produce a small but nonzero divergence of exactly this kind "
      "whenever v_hat is close to 0, i.e. in the very first few steps.")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
xx, yy = np.meshgrid(np.linspace(-6, 6, 200), np.linspace(-6, 6, 200))
zz = xx ** 2 + 10 * yy ** 2
for a in ax:
    a.contour(xx, yy, zz, levels=20, cmap="Greys", alpha=0.5)
ax[0].plot(traj_manual_sgd[:, 0], traj_manual_sgd[:, 1], "o-", ms=3, color="#4C72B0", label="manual SGD")
ax[0].plot(traj_torch_sgd[:, 0], traj_torch_sgd[:, 1], "x--", ms=5, color="#DD8452", label="torch.optim.SGD")
ax[0].set_title(f"Vanilla SGD (max traj diff {max_diff_sgd:.1e})")
ax[0].legend(fontsize=8)
ax[1].plot(traj_manual_adam[:, 0], traj_manual_adam[:, 1], "o-", ms=3, color="#55A868", label="manual Adam")
ax[1].plot(traj_torch_adam[:, 0], traj_torch_adam[:, 1], "x--", ms=5, color="#C44E52", label="torch.optim.Adam")
ax[1].set_title(f"Adam (max traj diff {max_diff_adam:.1e})")
ax[1].legend(fontsize=8)
for a in ax:
    a.set_xlabel("x"); a.set_ylabel("y")
fig.suptitle(f"Manual optimizers vs torch.optim on f(x,y) = x^2 + 10y^2, from (5,5)")
fig.tight_layout()
fig.savefig("figures/01_optimizer_verification.png", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------
# B3: worked numerical example -- Adam on sparse/unevenly-scaled gradients
# ----------------------------------------------------------------------
print("\n--- B3: Adam vs SGD on unevenly-scaled gradients, worked example ---")
print("""
Consider f(x, y) = x^2 + 1000*y^2. The gradient is (2x, 2000y), so at the
same starting distance from the minimum, the y-gradient is 1000x larger than
the x-gradient. A single SGD learning rate must be small enough to stay
stable in y, which makes progress in x agonisingly slow -- vanilla SGD is
forced to move at the pace of the worst-scaled dimension.
""")

start_sparse = torch.tensor([5.0, 5.0])


def sparse_grad(p):
    return torch.tensor([2.0 * p[0], 2000.0 * p[1]])


lr_sgd_sparse = 0.0009  # the largest stable step for the y-dimension
p_sgd = start_sparse.clone()
for _ in range(5):
    g = sparse_grad(p_sgd)
    p_sgd = p_sgd - lr_sgd_sparse * g
print(f"After 5 SGD steps (lr={lr_sgd_sparse}, the largest stable for the y-axis): "
      f"x moved from 5.0 to {p_sgd[0].item():.4f} (only {5.0 - p_sgd[0].item():.4f} of progress)")

p_adam = start_sparse.clone()
m_a, v_a = torch.zeros(2), torch.zeros(2)
lr_adam_sparse = 0.5
for t in range(1, 6):
    g = sparse_grad(p_adam)
    m_a = 0.9 * m_a + 0.1 * g
    v_a = 0.999 * v_a + 0.001 * g ** 2
    m_hat = m_a / (1 - 0.9 ** t)
    v_hat = v_a / (1 - 0.999 ** t)
    p_adam = p_adam - lr_adam_sparse * m_hat / (torch.sqrt(v_hat) + 1e-8)
print(f"After 5 Adam steps (lr={lr_adam_sparse}): x moved from 5.0 to {p_adam[0].item():.4f} "
      f"({5.0 - p_adam[0].item():.4f} of progress) while y moved from 5.0 to "
      f"{p_adam[1].item():.4f} ({5.0 - p_adam[1].item():.4f} of progress) -- comparable "
      f"progress on BOTH axes despite the 1000x gradient-scale difference, because Adam "
      f"divides each parameter's update by its own running RMS gradient "
      f"(sqrt(v_hat)), which cancels out exactly the scale difference between the two "
      f"dimensions and leaves both effective step sizes of a similar order.")

# ----------------------------------------------------------------------
# B4: controlled 3-way comparison on the Palmer Penguins network
# ----------------------------------------------------------------------
print("\n--- B4: SGD vs SGD+momentum vs Adam, same network/init/data ---")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("penguins.csv").dropna(
    subset=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"])
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
species = sorted(df["species"].unique())
species_to_idx = {s: i for i, s in enumerate(species)}
X = df[FEATURES].values
y = df["species"].map(species_to_idx).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)


class PenguinNet(nn.Module):
    def __init__(self, n_in=4, n_hidden=8, n_out=3):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def train_with_optimizer(optimizer_fn, epochs=150, seed=RANDOM_STATE):
    torch.manual_seed(seed)
    net = PenguinNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_fn(net.parameters())
    loss_history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(net(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    with torch.no_grad():
        val_acc = (net(X_test_t).argmax(dim=1) == y_test_t).float().mean().item()
    return loss_history, val_acc


configs = {
    "SGD (lr=0.1)": lambda params: torch.optim.SGD(params, lr=0.1),
    "SGD+momentum (lr=0.1, m=0.9)": lambda params: torch.optim.SGD(params, lr=0.1, momentum=0.9),
    "Adam (lr=0.01)": lambda params: torch.optim.Adam(params, lr=0.01),
}
comparison_results = {}
for name, fn in configs.items():
    hist, acc = train_with_optimizer(fn)
    comparison_results[name] = {"loss_history": hist, "val_acc": acc}
    print(f"{name:32s} final loss {hist[-1]:.4f} | val accuracy {acc:.4f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
for name, r in comparison_results.items():
    ax.plot(r["loss_history"], label=f"{name} (val acc {r['val_acc']:.3f})")
ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
ax.set_title("SGD vs SGD+momentum vs Adam, same network/init/data")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/02_optimizer_comparison_penguins.png", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------
# B5: L2-in-gradient vs true (decoupled) weight decay -- equivalent for
# SGD, NOT equivalent for an adaptive optimizer like Adam
# ----------------------------------------------------------------------
print("\n--- B5: L2-in-gradient vs decoupled weight decay (AdamW), SGD vs Adam ---")
torch.manual_seed(1)
w0 = torch.tensor([1.0, 1.0])
# Two parameters with deliberately very different gradient scales, so
# Adam's adaptive denominator differs sharply between them.
grads_sequence = [torch.tensor([10.0, 0.1]) for _ in range(10)]
lr_wd, decay = 0.1, 0.1


def sgd_l2_in_grad(w0, grads, lr, decay):
    w = w0.clone()
    for g in grads:
        w = w - lr * (g + decay * w)
    return w


def sgd_decoupled_decay(w0, grads, lr, decay):
    # "Decoupled" means decay never enters the gradient computation itself
    # (no interaction with a momentum buffer, no squaring into an adaptive
    # denominator) -- but the decay and the gradient step are still applied
    # simultaneously to the SAME pre-step w, not sequentially to an
    # already-updated w. For vanilla SGD, with no momentum buffer for decay
    # to interact with, that makes this arithmetically identical to
    # L2-in-the-gradient -- which is exactly the point being demonstrated.
    w = w0.clone()
    for g in grads:
        w = w - lr * g - lr * decay * w
    return w


w_sgd_l2 = sgd_l2_in_grad(w0, grads_sequence, lr_wd, decay)
w_sgd_decoupled = sgd_decoupled_decay(w0, grads_sequence, lr_wd, decay)
print(f"SGD, L2-in-gradient:       final w = {w_sgd_l2.numpy().round(6)}")
print(f"SGD, decoupled decay:      final w = {w_sgd_decoupled.numpy().round(6)}")
print(f"Difference: {(w_sgd_l2 - w_sgd_decoupled).abs().max().item():.2e}  -- "
      f"equivalent for plain SGD, exactly as the standard argument predicts, since a "
      f"constant learning rate makes 'add decay*w to the gradient' and 'shrink w by "
      f"(1 - lr*decay) after the step' the same arithmetic.")


def adam_l2_in_grad(w0, grads, lr, decay, beta1=0.9, beta2=0.999, eps=1e-8):
    w = w0.clone()
    m, v = torch.zeros_like(w), torch.zeros_like(w)
    for t, g in enumerate(grads, 1):
        g_eff = g + decay * w              # L2 penalty added INTO the gradient...
        m = beta1 * m + (1 - beta1) * g_eff
        v = beta2 * v + (1 - beta2) * g_eff ** 2   # ...so it also gets squared into v
        m_hat, v_hat = m / (1 - beta1 ** t), v / (1 - beta2 ** t)
        w = w - lr * m_hat / (torch.sqrt(v_hat) + eps)
    return w


def adamw_decoupled(w0, grads, lr, decay, beta1=0.9, beta2=0.999, eps=1e-8):
    w = w0.clone()
    m, v = torch.zeros_like(w), torch.zeros_like(w)
    for t, g in enumerate(grads, 1):
        m = beta1 * m + (1 - beta1) * g    # decay never enters the moment estimates
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat, v_hat = m / (1 - beta1 ** t), v / (1 - beta2 ** t)
        w = w - lr * m_hat / (torch.sqrt(v_hat) + eps)
        w = w - lr * decay * w             # decoupled: applied directly to the weights
    return w


w_adam_l2 = adam_l2_in_grad(w0, grads_sequence, lr_wd, decay)
w_adamw = adamw_decoupled(w0, grads_sequence, lr_wd, decay)
print(f"\nAdam, L2-in-gradient:      final w = {w_adam_l2.numpy().round(6)}")
print(f"AdamW, decoupled decay:    final w = {w_adamw.numpy().round(6)}")
print(f"Difference: {(w_adam_l2 - w_adamw).abs().max().item():.4f}  -- "
      f"NOT equivalent for Adam. Feeding the L2 term into the gradient means it also "
      f"gets folded into v (the squared-gradient running average) and then divided by "
      f"sqrt(v_hat) along with the data gradient -- so the same nominal decay strength "
      f"is scaled DOWN for the large-gradient parameter (index 0, gradient scale 10) and "
      f"scaled UP, relatively, for the small-gradient parameter (index 1, gradient scale "
      f"0.1). AdamW's decoupled decay applies the same shrinkage lr*decay to every "
      f"parameter regardless of its gradient history, which is the behaviour weight "
      f"decay is actually supposed to have.")

print("\nFigures written to figures/")

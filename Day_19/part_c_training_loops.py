"""
PKCERT AI & Software Development Internship, Task 19
Part C: Epochs, Batches, and Training Loop Engineering

Batch/mini-batch/stochastic gradient descent compared directly; a proper
torch.utils.data.Dataset + DataLoader with per-epoch shuffling; a batch-size
sweep; a learning-rate scheduler vs a fixed-LR baseline; gradient
accumulation verified against a direct large-batch update; and early
stopping + gradient clipping, each demonstrated on a case built specifically
to trigger it.
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
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

print(f"=== Part C: PyTorch {torch.__version__} ===")

# ----------------------------------------------------------------------
# Data and model, shared across this whole part
# ----------------------------------------------------------------------
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
N_TRAIN = X_train_t.shape[0]
print(f"Train: {N_TRAIN} | Test: {X_test_t.shape[0]}")


class PenguinNet(nn.Module):
    def __init__(self, n_in=4, n_hidden=8, n_out=3):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# ----------------------------------------------------------------------
# C11: torch.utils.data.Dataset + DataLoader, with per-epoch shuffling
# ----------------------------------------------------------------------
class PenguinDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = PenguinDataset(X_train_t, y_train_t)
print(f"\n--- C2: Dataset/DataLoader ---")
print(f"PenguinDataset: {len(train_dataset)} samples. A DataLoader(shuffle=True) draws a "
      f"fresh random permutation every epoch. Without reshuffling, the model sees "
      f"minibatch B in the same neighbourhood, alongside the same neighbours, on every "
      f"single pass -- if the data happens to be sorted or blocked by any incidental "
      f"property (species is grouped by row in the raw file here, for instance), every "
      f"batch becomes a near-constant, unrepresentative slice of the label distribution, "
      f"and the model can pick up on batch composition as if it were signal. "
      f"Reshuffling breaks that spurious ordering, decorrelates consecutive updates, and "
      f"is why held-out validation performance is systematically a little worse with "
      f"shuffling turned off, even though training loss can look identical.")


def train_with_loader(net, loader, optimizer, criterion, epochs):
    loss_history = []
    for _ in range(epochs):
        epoch_losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(net(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        loss_history.append(np.mean(epoch_losses))
    return loss_history


# ----------------------------------------------------------------------
# C10: batch GD vs mini-batch GD vs SGD, same data/model, same LR
# ----------------------------------------------------------------------
print("\n--- C10: batch GD vs mini-batch GD vs (single-sample) SGD ---")
print("""
Batch gradient descent computes the gradient over the ENTIRE training set
before taking a single step -- one update per epoch, using the true,
noise-free gradient of the full training loss. Stochastic gradient descent
(in the strict sense) computes the gradient from a single example and steps
immediately -- N updates per epoch, each a noisy, high-variance estimate of
the true gradient. Mini-batch gradient descent is the practical middle
ground: a small batch of B examples per update, trading some of SGD's noise
for some of batch GD's stability, at N/B updates per epoch.
""")

LR_C = 0.05
EPOCHS_C = 60
configs_gd = {"Batch GD (B=273, full)": N_TRAIN, "Mini-batch GD (B=32)": 32, "SGD (B=1)": 1}
gd_results = {}
for name, bsz in configs_gd.items():
    torch.manual_seed(RANDOM_STATE)
    net = PenguinNet()
    loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True,
                         generator=torch.Generator().manual_seed(RANDOM_STATE))
    optimizer = torch.optim.SGD(net.parameters(), lr=LR_C)
    hist = train_with_loader(net, loader, optimizer, nn.CrossEntropyLoss(), EPOCHS_C)
    updates_per_epoch = int(np.ceil(N_TRAIN / bsz))
    gd_results[name] = {"loss_history": hist, "updates_per_epoch": updates_per_epoch}
    print(f"{name:26s} updates/epoch: {updates_per_epoch:3d} | final epoch-mean loss: {hist[-1]:.4f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
for name, r in gd_results.items():
    ax.plot(r["loss_history"], label=f"{name} ({r['updates_per_epoch']} updates/epoch)")
ax.set_xlabel("epoch"); ax.set_ylabel("epoch-mean training loss")
ax.set_title(f"Batch vs mini-batch vs SGD, same lr={LR_C}")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/03_batch_vs_minibatch_vs_sgd.png", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------
# C12: batch size sweep -- convergence (epochs, wall-clock), noise, gap
# ----------------------------------------------------------------------
print("\n--- C12: batch size sweep (8, 32, 128, full-batch) ---")
batch_sizes = [8, 32, 128, N_TRAIN]
EPOCHS_SWEEP = 80
sweep_results = {}
for bsz in batch_sizes:
    torch.manual_seed(RANDOM_STATE)
    net = PenguinNet()
    loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True,
                         generator=torch.Generator().manual_seed(RANDOM_STATE))
    optimizer = torch.optim.SGD(net.parameters(), lr=LR_C)
    criterion = nn.CrossEntropyLoss()

    train_loss_hist, val_loss_hist = [], []
    t0 = time.perf_counter()
    for _ in range(EPOCHS_SWEEP):
        epoch_losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(net(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        train_loss_hist.append(np.mean(epoch_losses))
        with torch.no_grad():
            val_loss_hist.append(criterion(net(X_test_t), y_test_t).item())
    wall_time = time.perf_counter() - t0

    with torch.no_grad():
        final_train_acc = (net(X_train_t).argmax(1) == y_train_t).float().mean().item()
        final_val_acc = (net(X_test_t).argmax(1) == y_test_t).float().mean().item()
    # epochs to reach a fixed loss target, as a convergence-speed measure
    target = 0.3
    epochs_to_target = next((i + 1 for i, v in enumerate(train_loss_hist) if v < target), None)

    sweep_results[bsz] = {
        "train_loss": train_loss_hist, "val_loss": val_loss_hist, "wall_time": wall_time,
        "train_acc": final_train_acc, "val_acc": final_val_acc,
        "gap": final_train_acc - final_val_acc, "epochs_to_target": epochs_to_target,
    }
    print(f"B={bsz:4d}: wall-clock {wall_time:.3f}s | epochs to loss<{target}: "
          f"{epochs_to_target} | train acc {final_train_acc:.4f} | val acc {final_val_acc:.4f} | "
          f"train-val gap {final_train_acc - final_val_acc:+.4f}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for bsz, r in sweep_results.items():
    label = f"B={bsz}" + (" (full)" if bsz == N_TRAIN else "")
    ax[0].plot(r["train_loss"], label=label)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train loss"); ax[0].set_title("Loss vs epoch, by batch size")
ax[0].legend(fontsize=7)
ax[1].bar([str(b) for b in batch_sizes], [sweep_results[b]["wall_time"] for b in batch_sizes], color="#DD8452")
ax[1].set_xlabel("batch size"); ax[1].set_ylabel("wall-clock time, s")
ax[1].set_title(f"Wall-clock time for {EPOCHS_SWEEP} epochs")
ax[2].bar([str(b) for b in batch_sizes], [sweep_results[b]["gap"] for b in batch_sizes], color="#8172B2")
ax[2].set_xlabel("batch size"); ax[2].set_ylabel("train acc - val acc")
ax[2].set_title("Generalization gap by batch size")
fig.tight_layout()
fig.savefig("figures/04_batch_size_sweep.png", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------
# C13: learning rate scheduler vs fixed-LR baseline
# ----------------------------------------------------------------------
print("\n--- C13: StepLR scheduler vs fixed learning rate ---")
EPOCHS_SCHED = 100


def train_fixed_lr(lr, epochs):
    torch.manual_seed(RANDOM_STATE)
    net = PenguinNet()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    hist = []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(net(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()
        hist.append(loss.item())
    with torch.no_grad():
        val_acc = (net(X_test_t).argmax(1) == y_test_t).float().mean().item()
    return hist, val_acc


def train_with_scheduler(lr, epochs, step_size=25, gamma=0.5):
    torch.manual_seed(RANDOM_STATE)
    net = PenguinNet()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss()
    hist, lr_hist = [], []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(net(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()
        hist.append(loss.item())
        lr_hist.append(optimizer.param_groups[0]["lr"])
        scheduler.step()   # stepped once per epoch, after the epoch's optimizer step(s)
    with torch.no_grad():
        val_acc = (net(X_test_t).argmax(1) == y_test_t).float().mean().item()
    return hist, val_acc, lr_hist


LR_SCHED_BASE = 0.3  # deliberately a bit high, so a decaying schedule has something to fix
hist_fixed, val_acc_fixed = train_fixed_lr(LR_SCHED_BASE, EPOCHS_SCHED)
hist_sched, val_acc_sched, lr_hist = train_with_scheduler(LR_SCHED_BASE, EPOCHS_SCHED)
print(f"Fixed lr={LR_SCHED_BASE}:            final loss {hist_fixed[-1]:.4f} | val acc {val_acc_fixed:.4f}")
print(f"StepLR (step=25, gamma=0.5), lr0={LR_SCHED_BASE}: final loss {hist_sched[-1]:.4f} | "
      f"val acc {val_acc_sched:.4f} | lr schedule: {sorted(set(round(l,4) for l in lr_hist), reverse=True)}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(hist_fixed, label=f"fixed lr={LR_SCHED_BASE} (val acc {val_acc_fixed:.3f})")
ax[0].plot(hist_sched, label=f"StepLR from {LR_SCHED_BASE} (val acc {val_acc_sched:.3f})")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training loss"); ax[0].set_title("Fixed LR vs StepLR")
ax[0].legend(fontsize=8)
ax[1].plot(lr_hist, color="#55A868")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("learning rate"); ax[1].set_title("StepLR schedule")
fig.tight_layout()
fig.savefig("figures/05_lr_scheduler.png", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------
# C14: gradient accumulation, verified against a direct large-batch update
# ----------------------------------------------------------------------
print("\n--- C14: gradient accumulation, verified against a direct large-batch update ---")
torch.manual_seed(RANDOM_STATE)
net_direct = PenguinNet()
net_accum_correct = copy.deepcopy(net_direct)
net_accum_buggy = copy.deepcopy(net_direct)

big_batch_X, big_batch_y = X_train_t[:32], y_train_t[:32]
MICRO_BSZ, K = 8, 4  # 4 micro-batches of 8 = one effective batch of 32
criterion = nn.CrossEntropyLoss()

# Direct: one forward/backward/step on the full 32-example batch.
opt_direct = torch.optim.SGD(net_direct.parameters(), lr=0.1)
opt_direct.zero_grad()
loss_direct = criterion(net_direct(big_batch_X), big_batch_y)
loss_direct.backward()
grads_direct = [p.grad.clone() for p in net_direct.parameters()]

# Correct accumulation: divide each micro-batch's loss by K before
# backward(), so K accumulated (summed, by autograd's default behaviour)
# micro-gradients average out to the same gradient as the direct batch.
opt_correct = torch.optim.SGD(net_accum_correct.parameters(), lr=0.1)
opt_correct.zero_grad()
for i in range(K):
    xb = big_batch_X[i * MICRO_BSZ:(i + 1) * MICRO_BSZ]
    yb = big_batch_y[i * MICRO_BSZ:(i + 1) * MICRO_BSZ]
    loss = criterion(net_accum_correct(xb), yb) / K
    loss.backward()
grads_correct = [p.grad.clone() for p in net_accum_correct.parameters()]

# The gotcha: forgetting the /K. Each micro-batch loss is already a MEAN
# over its own 8 examples, so summing K of those means (via K unscaled
# backward() calls) produces a gradient K times too large, not an average.
opt_buggy = torch.optim.SGD(net_accum_buggy.parameters(), lr=0.1)
opt_buggy.zero_grad()
for i in range(K):
    xb = big_batch_X[i * MICRO_BSZ:(i + 1) * MICRO_BSZ]
    yb = big_batch_y[i * MICRO_BSZ:(i + 1) * MICRO_BSZ]
    loss = criterion(net_accum_buggy(xb), yb)  # missing "/ K"
    loss.backward()
grads_buggy = [p.grad.clone() for p in net_accum_buggy.parameters()]

max_diff_correct = max((gd - gc).abs().max().item() for gd, gc in zip(grads_direct, grads_correct))
max_diff_buggy = max((gd - gb).abs().max().item() for gd, gb in zip(grads_direct, grads_buggy))
ratio_buggy = np.mean([(gb.abs().sum() / gd.abs().sum()).item()
                        for gd, gb in zip(grads_direct, grads_buggy) if gd.abs().sum() > 0])
print(f"Correctly scaled accumulation (loss / {K} before each backward): max gradient "
      f"difference vs direct large-batch update: {max_diff_correct:.2e}  "
      f"({'MATCH' if max_diff_correct < 1e-5 else 'MISMATCH'})")
print(f"Un-scaled accumulation (the missing-'/K' gotcha): max gradient difference vs "
      f"direct: {max_diff_buggy:.4f}  ({'MISMATCH, as expected' if max_diff_buggy > 1e-3 else 'MATCH'}), "
      f"mean gradient-magnitude ratio to the correct gradient: {ratio_buggy:.2f}x "
      f"(expected exactly {K}x from summing {K} unscaled per-micro-batch mean losses)")

# ----------------------------------------------------------------------
# C15a: gradient clipping, demonstrated on a case built to explode
# ----------------------------------------------------------------------
print("\n--- C15a: gradient clipping, on a deliberately unstable learning rate ---")


UNSTABLE_LR = 20.0  # empirically found to reliably explode this network to inf within ~25 epochs


def train_watch_for_explosion(clip_norm=None, lr=UNSTABLE_LR, epochs=40):
    torch.manual_seed(RANDOM_STATE)
    net = PenguinNet()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    hist, grad_norms = [], []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(net(X_train_t), y_train_t)
        loss.backward()
        total_norm = torch.sqrt(sum((p.grad ** 2).sum() for p in net.parameters())).item()
        grad_norms.append(total_norm)
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip_norm)
        optimizer.step()
        hist.append(loss.item())
        if not np.isfinite(loss.item()):
            break
    return hist, grad_norms


hist_unclipped, norms_unclipped = train_watch_for_explosion(clip_norm=None)
hist_clipped, norms_clipped = train_watch_for_explosion(clip_norm=1.0)
print(f"Unstable lr={UNSTABLE_LR}, unclipped: loss after {len(hist_unclipped)} epochs = "
      f"{hist_unclipped[-1]:.4g} ({'exploded to non-finite' if not np.isfinite(hist_unclipped[-1]) else 'finite'}), "
      f"max finite gradient norm observed before the explosion: {np.nanmax(norms_unclipped):.3g}")
print(f"Same lr={UNSTABLE_LR}, clipped to max_norm=1.0: loss after {len(hist_clipped)} epochs = "
      f"{hist_clipped[-1]:.4g} (finite -- training completed normally), "
      f"max PRE-clip gradient norm observed: {np.nanmax(norms_clipped):.3g} (clip_grad_norm_ "
      f"rescales any gradient above 1.0 down to exactly norm 1.0 before the optimizer step "
      f"uses it, which is what keeps training stable despite pre-clip norms up to "
      f"{np.nanmax(norms_clipped):.1f})")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(hist_unclipped, label="unclipped", color="#C44E52")
ax[0].plot(hist_clipped, label="clipped (max_norm=1.0)", color="#4C72B0")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].set_yscale("log")
ax[0].set_title(f"Loss with an unstable lr={UNSTABLE_LR}, clipped vs not")
ax[0].legend(fontsize=8)
ax[1].plot(norms_unclipped, label="unclipped", color="#C44E52")
ax[1].plot(norms_clipped, label="clipped (max_norm=1.0)", color="#4C72B0")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("gradient norm (pre-clip)"); ax[1].set_yscale("log")
ax[1].set_title("Gradient norm, clipped vs not")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/06_gradient_clipping.png", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------
# C15b: early stopping, demonstrated on a case built to overfit
# ----------------------------------------------------------------------
print("\n--- C15b: early stopping, on a deliberately overfitting setup ---")


class OverfitNet(nn.Module):
    # Two hidden layers of 128 units each, aimed at a training set that is
    # about to be cut down to 20 rows -- wildly over-capacity on purpose,
    # so early stopping has a real train/val gap to catch within a
    # reasonable number of epochs.
    def __init__(self, n_in=4, n_hidden=128, n_out=3):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


ES_SUBSET = 20  # a small slice of the training set, deliberately, so a
                 # 128-128-3 network has far more capacity than the data
                 # needs and can memorise it outright
X_train_es = X_train_t[:ES_SUBSET]
y_train_es = y_train_t[:ES_SUBSET]

torch.manual_seed(RANDOM_STATE)
net_es = OverfitNet()
optimizer_es = torch.optim.Adam(net_es.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

EPOCHS_ES = 300
PATIENCE = 15
train_hist_es, val_hist_es = [], []
best_val, best_epoch, best_state, patience_counter = float("inf"), -1, None, 0
stopped_at = EPOCHS_ES

for epoch in range(EPOCHS_ES):
    optimizer_es.zero_grad()
    loss = criterion(net_es(X_train_es), y_train_es)
    loss.backward()
    optimizer_es.step()
    train_hist_es.append(loss.item())
    with torch.no_grad():
        val_loss = criterion(net_es(X_test_t), y_test_t).item()
    val_hist_es.append(val_loss)

    if val_loss < best_val - 1e-4:
        best_val, best_epoch, patience_counter = val_loss, epoch, 0
        best_state = copy.deepcopy(net_es.state_dict())
    else:
        patience_counter += 1
    if patience_counter >= PATIENCE:
        stopped_at = epoch + 1
        break

print(f"Trained up to {stopped_at} of {EPOCHS_ES} planned epochs before early stopping "
      f"triggered (patience={PATIENCE}); best validation loss {best_val:.4f} at epoch "
      f"{best_epoch + 1}. Train loss at stop: {train_hist_es[-1]:.4f}, val loss at stop: "
      f"{val_hist_es[-1]:.4f} -- {'val loss rose while train loss kept falling, the overfitting signature' if val_hist_es[-1] > best_val else ''}")
net_es.load_state_dict(best_state)
with torch.no_grad():
    restored_val_acc = (net_es(X_test_t).argmax(1) == y_test_t).float().mean().item()
print(f"Restored the epoch-{best_epoch+1} weights (not the final epoch's): val accuracy = {restored_val_acc:.4f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(train_hist_es, label="train loss", color="#4C72B0")
ax.plot(val_hist_es, label="val loss", color="#C44E52")
ax.axvline(best_epoch, color="grey", ls="--", lw=1, label=f"best epoch ({best_epoch+1})")
ax.axvline(stopped_at - 1, color="black", ls=":", lw=1, label=f"stopped ({stopped_at})")
ax.set_xlabel("epoch"); ax.set_ylabel("loss")
ax.set_title("Early stopping on a deliberately over-capacity network")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/07_early_stopping.png", bbox_inches="tight")
plt.close(fig)

print("\nFigures written to figures/")

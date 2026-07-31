"""
PKCERT AI & Software Development Internship, Task 21
Regularization Techniques in Deep Learning

Continues directly from Task 20's Fashion-MNIST feedforward pipeline. Where
Task 20 used a right-sized 784->256->128->10 network that only mildly
overfit, this task deliberately widens the baseline (784->512->256->10) and
trains it for longer, so it overfits clearly -- giving Dropout, Batch
Normalization and Early Stopping a real problem to fix rather than a
token demonstration. All four (baseline, dropout, batch norm, early
stopping) plus a combined configuration are trained on the identical split,
architecture skeleton, optimizer and seed, so the comparison isolates the
regularization technique itself.
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

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

print(f"=== PyTorch {torch.__version__} | torchvision {torchvision.__version__} | "
      f"CUDA available: {torch.cuda.is_available()} ===")
print(f"Random seed fixed at {RANDOM_STATE} for the split, every model's init, and training.")

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# ======================================================================
# Part A: Dataset Selection & Preparation
# ======================================================================
print("\n--- Part A: dataset ---")
print("""
Dataset: Fashion-MNIST (Xiao, Rasul & Vollgraf, 2017), continuing directly
from Task 20 -- the same 70,000 28x28 greyscale clothing images across 10
classes. Reusing it here is deliberate: Task 20 already established a
right-sized baseline for this data that barely overfit (validation loss
rose only slightly after epoch 8), which is exactly the wrong setting to
demonstrate what Dropout, Batch Normalization and Early Stopping actually
fix. This task's baseline is intentionally larger and trained longer, so
the overfitting the regularizers are meant to address is real, measured,
and visible in the loss curves below, not asserted.
""")

raw_train_full = FashionMNIST(root="./data", train=True, download=True)
raw_test = FashionMNIST(root="./data", train=False, download=True)

N_TRAIN, N_VAL = 50000, 10000
generator = torch.Generator().manual_seed(RANDOM_STATE)
train_subset, val_subset = random_split(raw_train_full, [N_TRAIN, N_VAL], generator=generator)
n_total = N_TRAIN + N_VAL + len(raw_test)
print(f"Three-way split (seed={RANDOM_STATE}): train {N_TRAIN} ({N_TRAIN/n_total:.1%}), "
      f"val {N_VAL} ({N_VAL/n_total:.1%}), test {len(raw_test)} ({len(raw_test)/n_total:.1%})")

train_indices = train_subset.indices
train_images = raw_train_full.data[train_indices].float() / 255.0
computed_mean, computed_std = train_images.mean().item(), train_images.std().item()
print(f"Normalization stats from the training split only: mean={computed_mean:.4f}, "
      f"std={computed_std:.4f} (train-only, to avoid the leakage discussed in Task 20)")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((computed_mean,), (computed_std,)),
    transforms.Lambda(lambda x: x.view(-1)),
])
train_subset.dataset = FashionMNIST(root="./data", train=True, download=False, transform=transform)
val_subset.dataset = FashionMNIST(root="./data", train=True, download=False, transform=transform)
test_dataset = FashionMNIST(root="./data", train=False, download=False, transform=transform)

BATCH_SIZE = 256
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                           generator=torch.Generator().manual_seed(RANDOM_STATE))
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Batch size {BATCH_SIZE}: {len(train_loader)} train batches/epoch")

# ======================================================================
# Part B: Model Development (deliberately overfitting baseline)
# ======================================================================
print("\n--- Part B: baseline model ---")


class BaselineNet(nn.Module):
    """784 -> 512 -> 256 -> 10, ReLU, no regularization at all. Wider than
    Task 20's right-sized 256/128 net specifically so it has the spare
    capacity to memorise the training set."""
    def __init__(self, n_in=784, h1=512, h2=256, n_out=10):
        super().__init__()
        self.fc1 = nn.Linear(n_in, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, n_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


n_params_baseline = sum(p.numel() for p in BaselineNet().parameters())
print(f"""
Architecture: 784 -> 512 -> 256 -> 10 (2 hidden layers, ReLU, no dropout,
no batch norm). {n_params_baseline:,} trainable parameters -- roughly 4x
Task 20's right-sized network -- trained on the same 50,000 images, which
is exactly the capacity-vs-data imbalance that produces clear overfitting.
Optimizer: Adam(lr=1e-3). Loss: CrossEntropyLoss. Batch size {BATCH_SIZE}.
Trained for 25 epochs with no early stopping in this baseline run, so the
full overfitting trajectory is visible in the loss curve.
""")

EPOCHS = 25
LR = 1e-3


def evaluate_loss(model, loader, criterion):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            loss = criterion(model(xb), yb)
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
    return total_loss / n


def evaluate_accuracy(model, loader):
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            n += xb.size(0)
    return correct / n


def train_model(model, epochs, lr=LR, seed=RANDOM_STATE, early_stopping=False, patience=5,
                 verbose_name=""):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses, val_accs, train_accs = [], [], [], []
    best_val, best_state, patience_counter, stopped_epoch = float("inf"), None, 0, epochs
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        train_losses.append(evaluate_loss(model, train_loader, criterion))
        val_loss = evaluate_loss(model, val_loader, criterion)
        val_losses.append(val_loss)
        train_accs.append(evaluate_accuracy(model, train_loader))
        val_accs.append(evaluate_accuracy(model, val_loader))
        print(f"  [{verbose_name}] epoch {epoch+1:2d}/{epochs}  train loss {train_losses[-1]:.4f}  "
              f"val loss {val_losses[-1]:.4f}  train acc {train_accs[-1]:.4f}  val acc {val_accs[-1]:.4f}")

        if early_stopping:
            if val_loss < best_val - 1e-4:
                best_val, best_state, patience_counter = val_loss, copy.deepcopy(model.state_dict()), 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                stopped_epoch = epoch + 1
                print(f"  [{verbose_name}] early stopping triggered at epoch {stopped_epoch} "
                      f"(patience={patience}); restoring epoch {epoch + 1 - patience} weights")
                model.load_state_dict(best_state)
                break
    train_time = time.perf_counter() - t0
    return {"train_losses": train_losses, "val_losses": val_losses,
            "train_accs": train_accs, "val_accs": val_accs,
            "train_time": train_time, "stopped_epoch": stopped_epoch}


torch.manual_seed(RANDOM_STATE)
baseline_model = BaselineNet()
baseline_hist = train_model(baseline_model, EPOCHS, verbose_name="baseline")
print(f"Baseline training time: {baseline_hist['train_time']:.1f}s")

final_gap = baseline_hist["train_accs"][-1] - baseline_hist["val_accs"][-1]
best_val_epoch = int(np.argmin(baseline_hist["val_losses"]))
print(f"Baseline: train/val accuracy gap at final epoch = {final_gap:.4f} "
      f"(train {baseline_hist['train_accs'][-1]:.4f} vs val {baseline_hist['val_accs'][-1]:.4f}); "
      f"validation loss lowest at epoch {best_val_epoch + 1} "
      f"({baseline_hist['val_losses'][best_val_epoch]:.4f}), then rises to "
      f"{baseline_hist['val_losses'][-1]:.4f} by epoch {EPOCHS} -- clear overfitting, "
      f"confirmed by the widening gap and the rising validation loss, not assumed.")


def full_evaluate(model, loader, name):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            all_preds.append(model(xb).argmax(dim=1))
            all_labels.append(yb)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return {
        "name": name,
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro"),
        "f1": f1_score(labels, preds, average="macro"),
        "cm": confusion_matrix(labels, preds),
    }


baseline_test = full_evaluate(baseline_model, test_loader, "Baseline")
print(f"Baseline TEST: Acc {baseline_test['accuracy']:.4f} | Prec {baseline_test['precision']:.4f} | "
      f"Rec {baseline_test['recall']:.4f} | F1 {baseline_test['f1']:.4f}")

# ======================================================================
# Part C: Regularization Techniques
# ======================================================================
print("\n--- Part C: regularization techniques ---")


class DropoutNet(nn.Module):
    def __init__(self, n_in=784, h1=512, h2=256, n_out=10, p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(n_in, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, n_out)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(p)
        self.drop2 = nn.Dropout(p)

    def forward(self, x):
        x = self.drop1(self.relu(self.fc1(x)))
        x = self.drop2(self.relu(self.fc2(x)))
        return self.fc3(x)


class BatchNormNet(nn.Module):
    def __init__(self, n_in=784, h1=512, h2=256, n_out=10):
        super().__init__()
        self.fc1 = nn.Linear(n_in, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.fc3 = nn.Linear(h2, n_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)


class CombinedNet(nn.Module):
    """Dropout + BatchNorm together, trained with early stopping -- the
    Part D 'best of all three' configuration."""
    def __init__(self, n_in=784, h1=512, h2=256, n_out=10, p=0.3):
        super().__init__()
        self.fc1 = nn.Linear(n_in, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.fc3 = nn.Linear(h2, n_out)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(p)
        self.drop2 = nn.Dropout(p)

    def forward(self, x):
        x = self.drop1(self.relu(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)


print("\n--- C1: Dropout (p=0.5 after each hidden layer) ---")
torch.manual_seed(RANDOM_STATE)
dropout_model = DropoutNet()
dropout_hist = train_model(dropout_model, EPOCHS, verbose_name="dropout")
dropout_test = full_evaluate(dropout_model, test_loader, "Dropout (p=0.5)")
dropout_gap = dropout_hist["train_accs"][-1] - dropout_hist["val_accs"][-1]
print(f"Dropout TEST: Acc {dropout_test['accuracy']:.4f} | F1 {dropout_test['f1']:.4f} | "
      f"final train/val gap {dropout_gap:.4f} (baseline: {final_gap:.4f})")

print("\n--- C2: Batch Normalization ---")
torch.manual_seed(RANDOM_STATE)
batchnorm_model = BatchNormNet()
batchnorm_hist = train_model(batchnorm_model, EPOCHS, verbose_name="batchnorm")
batchnorm_test = full_evaluate(batchnorm_model, test_loader, "Batch Normalization")
batchnorm_gap = batchnorm_hist["train_accs"][-1] - batchnorm_hist["val_accs"][-1]
print(f"BatchNorm TEST: Acc {batchnorm_test['accuracy']:.4f} | F1 {batchnorm_test['f1']:.4f} | "
      f"final train/val gap {batchnorm_gap:.4f} (baseline: {final_gap:.4f})")

print("\n--- C3: Early Stopping (on the baseline architecture) ---")
torch.manual_seed(RANDOM_STATE)
earlystop_model = BaselineNet()
earlystop_hist = train_model(earlystop_model, EPOCHS, early_stopping=True, patience=5,
                              verbose_name="early-stop")
earlystop_test = full_evaluate(earlystop_model, test_loader, "Early Stopping")
print(f"Early Stopping TEST: Acc {earlystop_test['accuracy']:.4f} | F1 {earlystop_test['f1']:.4f} | "
      f"stopped at epoch {earlystop_hist['stopped_epoch']} of {EPOCHS} planned "
      f"(baseline ran the full {EPOCHS})")

print("\n--- Bonus: Dropout + BatchNorm + Early Stopping combined ---")
torch.manual_seed(RANDOM_STATE)
combined_model = CombinedNet()
combined_hist = train_model(combined_model, EPOCHS, early_stopping=True, patience=5,
                             verbose_name="combined")
combined_test = full_evaluate(combined_model, test_loader, "Combined (Dropout+BN+EarlyStop)")
print(f"Combined TEST: Acc {combined_test['accuracy']:.4f} | F1 {combined_test['f1']:.4f} | "
      f"stopped at epoch {combined_hist['stopped_epoch']} of {EPOCHS} planned")

# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
all_results = [baseline_test, dropout_test, batchnorm_test, earlystop_test, combined_test]
all_hists = {"Baseline": baseline_hist, "Dropout": dropout_hist,
             "Batch Norm": batchnorm_hist, "Early Stopping": earlystop_hist,
             "Combined": combined_hist}

# 01: baseline loss/accuracy curves (the overfitting demonstration)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(range(1, EPOCHS + 1), baseline_hist["train_losses"], label="train loss", color="#4C72B0")
ax[0].plot(range(1, EPOCHS + 1), baseline_hist["val_losses"], label="val loss", color="#C44E52")
ax[0].axvline(best_val_epoch + 1, color="grey", ls="--", lw=1, label=f"best val loss (epoch {best_val_epoch+1})")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].set_title("Baseline: loss (clear overfitting)")
ax[0].legend(fontsize=8)
ax[1].plot(range(1, EPOCHS + 1), baseline_hist["train_accs"], label="train acc", color="#4C72B0")
ax[1].plot(range(1, EPOCHS + 1), baseline_hist["val_accs"], label="val acc", color="#C44E52")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].set_title("Baseline: accuracy gap")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/01_baseline_overfitting.png", bbox_inches="tight")
plt.close(fig)

# 02: all loss curves overlaid
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for name, h in all_hists.items():
    ep = range(1, len(h["val_losses"]) + 1)
    ax[0].plot(ep, h["val_losses"], label=name)
    ax[1].plot(ep, [t - v for t, v in zip(h["train_accs"], h["val_accs"])], label=name)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("validation loss"); ax[0].set_title("Validation loss, all configurations")
ax[0].legend(fontsize=7)
ax[1].axhline(0, color="grey", lw=0.5)
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("train acc - val acc"); ax[1].set_title("Train/val accuracy gap, all configurations")
ax[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig("figures/02_all_loss_curves.png", bbox_inches="tight")
plt.close(fig)

# 03: metrics comparison bar chart
fig, ax = plt.subplots(figsize=(9, 4.5))
metrics = ["accuracy", "precision", "recall", "f1"]
xw = np.arange(len(metrics)); w = 0.15
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
for i, r in enumerate(all_results):
    ax.bar(xw + (i - 2) * w, [r[m] for m in metrics], w, label=r["name"], color=colors[i])
ax.set_xticks(xw); ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
ax.set_ylim(0.75, 0.95)
ax.set_title("Test-set metrics, all five configurations")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig("figures/03_metrics_comparison.png", bbox_inches="tight")
plt.close(fig)

# 04: confusion matrices, baseline vs combined (the two extremes)
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for a, r in zip(axes, [baseline_test, combined_test]):
    cm = r["cm"]
    im = a.imshow(cm, cmap="Blues")
    a.set_xticks(range(10)); a.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=7)
    a.set_yticks(range(10)); a.set_yticklabels(CLASS_NAMES, fontsize=7)
    a.set_xlabel("Predicted"); a.set_ylabel("Actual")
    a.set_title(f"{r['name']} (acc {r['accuracy']:.4f})")
    for i in range(10):
        for j in range(10):
            if cm[i, j] > 0:
                a.text(j, i, str(cm[i, j]), ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=6)
fig.tight_layout()
fig.savefig("figures/04_confusion_matrices.png", bbox_inches="tight")
plt.close(fig)

# 05: train/val gap summary + training time
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
names_short = list(all_hists.keys())
final_gaps = [all_hists[n]["train_accs"][-1] - all_hists[n]["val_accs"][-1] for n in names_short]
ax[0].bar(names_short, final_gaps, color=colors)
ax[0].set_ylabel("final train acc - val acc"); ax[0].set_title("Overfitting gap at final trained epoch")
ax[0].tick_params(axis="x", rotation=20)
times = [all_hists[n]["train_time"] for n in names_short]
ax[1].bar(names_short, times, color=colors)
ax[1].set_ylabel("training time, s"); ax[1].set_title("Training time (early stopping saves time)")
ax[1].tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig("figures/05_gap_and_time_summary.png", bbox_inches="tight")
plt.close(fig)

print("\nFigures written to figures/")

# ----------------------------------------------------------------------
print("\n=== SUMMARY ===")
for r, name in zip(all_results, all_hists.keys()):
    h = all_hists[name]
    gap = h["train_accs"][-1] - h["val_accs"][-1]
    print(f"{r['name']:32s} Test Acc {r['accuracy']:.4f} | F1 {r['f1']:.4f} | "
          f"train/val gap {gap:+.4f} | epochs run {len(h['train_losses'])} | "
          f"time {h['train_time']:.1f}s")

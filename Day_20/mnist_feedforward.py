"""
PKCERT AI & Software Development Internship, Task 20
Build and Train a Simple Feedforward Neural Network on Fashion-MNIST

A single, complete, end-to-end pipeline: a leakage-free three-way
train/validation/test split, a fully-connected nn.Module (no convolutions),
training with the loss/optimizer/training-loop conventions established in
Task 19, honest evaluation (accuracy/precision/recall/F1/confusion matrix),
a controlled depth ablation, and model persistence with a verified
save/reload round-trip.

Dataset: Fashion-MNIST (Xiao, Rasul & Vollgraf, 2017), the stated MNIST
alternative -- chosen over plain MNIST because its confusion matrix is far
more informative (clothing categories that are genuinely visually similar,
e.g. shirt/pullover/coat, rather than MNIST's mostly-solved digit pairs).
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

print(f"=== PyTorch {torch.__version__} | torchvision {torchvision.__version__} | "
      f"CUDA available: {torch.cuda.is_available()} ===")
print(f"Random seed fixed at {RANDOM_STATE} for the split, model init, and training.")

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# ======================================================================
# Part A: Data Pipeline & Preprocessing
# ======================================================================
print("\n--- Part A: data pipeline ---")

raw_train_full = FashionMNIST(root="./data", train=True, download=True)
raw_test = FashionMNIST(root="./data", train=False, download=True)
print(f"Downloaded: {len(raw_train_full)} official-train images, {len(raw_test)} "
      f"official-test images, {len(raw_train_full.classes)} classes: {raw_train_full.classes}")

# A1: three-way split. The official test set (10,000 images) is kept
# completely untouched until final evaluation. The official 60,000-image
# training set is further split 50,000 / 10,000 into a real training set
# and a validation set, so hyperparameter/epoch decisions never touch the
# test set either.
N_TRAIN, N_VAL = 50000, 10000
generator = torch.Generator().manual_seed(RANDOM_STATE)
train_subset, val_subset = random_split(raw_train_full, [N_TRAIN, N_VAL], generator=generator)
n_total = N_TRAIN + N_VAL + len(raw_test)
print(f"\nThree-way split (seed={RANDOM_STATE}): train {N_TRAIN} ({N_TRAIN/n_total:.1%}), "
      f"val {N_VAL} ({N_VAL/n_total:.1%}), test {len(raw_test)} ({len(raw_test)/n_total:.1%})")

# A2: normalization statistics from the TRAINING split only.
train_indices = train_subset.indices
train_images = raw_train_full.data[train_indices].float() / 255.0  # (N_TRAIN, 28, 28), [0,1]
computed_mean = train_images.mean().item()
computed_std = train_images.std().item()
print(f"\nNormalization stats computed from the {N_TRAIN}-image TRAINING split only: "
      f"mean={computed_mean:.4f}, std={computed_std:.4f}")

# The leakage this avoids, demonstrated rather than only described: stats
# computed from the full 60,000-image official training set (i.e.
# including the 10,000 images that were just held out as validation) are
# measurably different numbers.
full_images = raw_train_full.data.float() / 255.0
leaked_mean, leaked_std = full_images.mean().item(), full_images.std().item()
print(f"For comparison, stats computed from ALL 60,000 (train+val together, the leaky "
      f"way): mean={leaked_mean:.4f}, std={leaked_std:.4f} -- a small but real "
      f"difference ({abs(computed_mean - leaked_mean):.5f} in mean). Using the leaked "
      f"version would mean every training-time normalization already encodes "
      f"information about the exact 10,000 images later used to judge the model, and "
      f"the gap would only grow on a smaller dataset or a less i.i.d. split -- the "
      f"principle matters even when today's numeric impact is small.")

# A3: flatten 28x28 -> 784, with shapes printed at every stage as a sanity check.
transform = transforms.Compose([
    transforms.ToTensor(),                                  # PIL -> (1, 28, 28), [0,1]
    transforms.Normalize((computed_mean,), (computed_std,)),  # train-only stats
    transforms.Lambda(lambda x: x.view(-1)),                 # (1, 28, 28) -> (784,)
])

raw_img, raw_label = raw_train_full[0]
print(f"\nPipeline shapes:")
print(f"  raw PIL image: {raw_img.size} (mode {raw_img.mode})")
tensor_img = transforms.ToTensor()(raw_img)
print(f"  after ToTensor(): {tuple(tensor_img.shape)}")
flattened = transform(raw_img)
print(f"  after flatten:    {tuple(flattened.shape)}")

train_subset.dataset = FashionMNIST(root="./data", train=True, download=False, transform=transform)
val_subset.dataset = FashionMNIST(root="./data", train=True, download=False, transform=transform)
test_dataset = FashionMNIST(root="./data", train=False, download=False, transform=transform)

BATCH_SIZE = 128
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                           generator=torch.Generator().manual_seed(RANDOM_STATE))
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

xb, yb = next(iter(train_loader))
print(f"  batched tensor:   {tuple(xb.shape)} (batch_size={BATCH_SIZE}, 784 features), "
      f"labels {tuple(yb.shape)}")

# A4: sample batch visualisation + class distribution across splits
fig, axes = plt.subplots(2, 8, figsize=(13, 3.5))
sample_imgs, sample_labels = xb[:16], yb[:16]
for i, ax in enumerate(axes.flat):
    img = (sample_imgs[i] * computed_std + computed_mean).view(28, 28)  # undo normalization for display
    ax.imshow(img, cmap="gray")
    ax.set_title(CLASS_NAMES[sample_labels[i]], fontsize=7)
    ax.axis("off")
fig.suptitle("A sample training batch")
fig.tight_layout()
fig.savefig("figures/01_sample_batch.png", bbox_inches="tight")
plt.close(fig)


def class_distribution(dataset_or_subset, name):
    if hasattr(dataset_or_subset, "indices"):
        labels = raw_train_full.targets[dataset_or_subset.indices].numpy()
    else:
        labels = dataset_or_subset.targets.numpy()
    counts = np.bincount(labels, minlength=10)
    return pd.Series(counts, index=CLASS_NAMES, name=name)


dist_train = class_distribution(train_subset, "train")
dist_val = class_distribution(val_subset, "val")
dist_test = class_distribution(raw_test, "test")
dist_table = pd.concat([dist_train, dist_val, dist_test], axis=1)
print(f"\nClass distribution across splits:\n{dist_table.to_string()}")
print(f"Train per-class range: {dist_train.min()}-{dist_train.max()} (of {N_TRAIN/10:.0f} "
      f"expected if perfectly balanced) -- the random (non-stratified) split leaves classes "
      f"reasonably but not perfectly balanced; val and test are similarly close.")

fig, ax = plt.subplots(figsize=(9, 4.2))
dist_table.plot(kind="bar", ax=ax)
ax.set_ylabel("count"); ax.set_xlabel("class")
ax.set_title("Class distribution: train / val / test")
plt.xticks(rotation=40, ha="right")
fig.tight_layout()
fig.savefig("figures/02_class_distribution.png", bbox_inches="tight")
plt.close(fig)

# ======================================================================
# Part B: Model Architecture & Design Justification
# ======================================================================
print("\n--- Part B: model architecture ---")


class FeedforwardNet(nn.Module):
    def __init__(self, n_in=784, hidden1=256, hidden2=128, n_out=10):
        super().__init__()
        self.fc1 = nn.Linear(n_in, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, n_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)  # raw logits


print("""
Architecture: 784 -> 256 -> 128 -> 10 (3 fully-connected layers, 2 hidden).
256 (roughly a third of the 784-dimensional input) gives the first layer
enough width to form a broad bank of pixel-pattern detectors without the
parameter count exploding (784x256 = 200,704 weights already dominates the
model); 128 narrows that down by half, compressing toward the 10-class
output through an intermediate representation rather than jumping straight
from 256 features to 10 logits. Two hidden layers, not more, was a
deliberate choice for a fully-connected network on a 50,000-image training
set: enough depth to compose non-linear features, not so much that a purely
FC stack (no weight sharing, unlike a CNN) overfits before the data can
constrain it.
""")

print("""
Activation: ReLU: for a network with two hidden layers, deep enough that a
choice starts to matter. Sigmoid and tanh both saturate: their derivatives
approach 0 as |x| grows (sigmoid'(x) = sigmoid(x)(1-sigmoid(x)) peaks at just
0.25; tanh'(x) = 1-tanh(x)^2 peaks at 1.0 but still decays fast). Because
backpropagation multiplies the upstream gradient by each layer's local
derivative, stacking two such layers multiplies together two numbers that
are each usually well below 1, shrinking the gradient reaching the first
layer's weights before training has even had a chance to move them away
from initialization. ReLU's derivative is exactly 1 for any positive input,
so it passes gradient through completely unattenuated on its active side --
the deeper the stack, the more this matters, which is why it is the default
choice here rather than sigmoid or tanh.
""")

# B3: logits+CrossEntropyLoss vs softmax+NLLLoss, confirmed numerically equal
model_check = FeedforwardNet()
xb_check, yb_check = next(iter(train_loader))
logits = model_check(xb_check)

loss_ce = F.cross_entropy(logits, yb_check)
log_probs = F.log_softmax(logits, dim=1)
loss_nll = F.nll_loss(log_probs, yb_check)
print(f"\n--- B3: logits+CrossEntropyLoss vs softmax+NLLLoss ---")
print(f"CrossEntropyLoss(logits, y):        {loss_ce.item():.6f}")
print(f"NLLLoss(log_softmax(logits), y):    {loss_nll.item():.6f}")
print(f"Difference: {abs(loss_ce.item() - loss_nll.item()):.2e}  (equal, as expected -- "
      f"CrossEntropyLoss literally computes log_softmax then NLLLoss internally)")
print("""
Both formulations compute the mathematically identical quantity. The
logits+CrossEntropyLoss form is numerically preferred because it computes
log(softmax(z)) directly via the log-sum-exp identity
(log_softmax(z)_i = z_i - max(z) - log(sum_j exp(z_j - max(z)))), in one
fused, numerically stable operation, rather than computing softmax(z) as an
intermediate tensor (risking underflow to exactly 0.0 for a confidently
wrong logit) and then taking its log separately, which the softmax+NLLLoss
form does as two distinct steps with a numerically fragile intermediate
value.
""")

# B4: parameter count, by hand and verified
n_in, h1, h2, n_out = 784, 256, 128, 10
w1, b1 = n_in * h1, h1
w2, b2 = h1 * h2, h2
w3, b3 = h2 * n_out, n_out
hand_total = (w1 + b1) + (w2 + b2) + (w3 + b3)
verified_total = sum(p.numel() for p in model_check.parameters())
print(f"--- B4: parameter count ---")
print(f"Layer 1 (784->256): {w1:,} weights + {b1} biases = {w1+b1:,}")
print(f"Layer 2 (256->128): {w2:,} weights + {b2} biases = {w2+b2:,}")
print(f"Layer 3 (128->10):  {w3:,} weights + {b3} biases = {w3+b3:,}")
print(f"Hand-calculated total: {hand_total:,}")
print(f"sum(p.numel() for p in model.parameters()): {verified_total:,}")
print(f"Match: {hand_total == verified_total}")

# ======================================================================
# Part C: Training, Evaluation & Experiments
# ======================================================================
print("\n--- Part C: training ---")

DEVICE = "cpu"
LR = 1e-3
EPOCHS = 20
print(f"""
Optimizer: Adam(lr={LR}). Task 19's own controlled comparison found tuned
SGD+momentum beating Adam on the small, 4-feature Palmer Penguins network --
but that result was explicitly attributed there to Adam's adaptive scaling
having no uneven-gradient pathology to correct on such a small, well-
conditioned problem. This network is a different regime: 784 input
dimensions of very unevenly informative pixels (edge pixels are almost
always 0, centre pixels carry most of the signal), which is closer to the
sparse/uneven-gradient case Task 19 Part B's worked example showed Adam
handling better than plain SGD. Batch size 128 and {EPOCHS} epochs were
chosen as a standard, computationally light starting point for a 50,000-
image training set on CPU; the loss curves below are used to confirm this
was enough epochs to see validation loss plateau.
""")


def evaluate_loss(model, loader, criterion):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss = criterion(logits, yb)
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


def train_model(model, epochs, lr=LR, seed=RANDOM_STATE):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses, val_accs = [], [], []
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        train_losses.append(evaluate_loss(model, train_loader, criterion))
        val_losses.append(evaluate_loss(model, val_loader, criterion))
        val_accs.append(evaluate_accuracy(model, val_loader))
        print(f"  epoch {epoch+1:2d}/{epochs}  train loss {train_losses[-1]:.4f}  "
              f"val loss {val_losses[-1]:.4f}  val acc {val_accs[-1]:.4f}")
    return train_losses, val_losses, val_accs


torch.manual_seed(RANDOM_STATE)
main_model = FeedforwardNet()
t0 = time.perf_counter()
train_losses, val_losses, val_accs = train_model(main_model, EPOCHS)
train_time = time.perf_counter() - t0
print(f"Training time: {train_time:.1f}s for {EPOCHS} epochs")

best_epoch = int(np.argmin(val_losses))
print(f"\nValidation loss is lowest at epoch {best_epoch + 1} "
      f"({val_losses[best_epoch]:.4f}); "
      f"{'it is still falling at the final epoch, no overfitting observed in this run' if best_epoch == EPOCHS - 1 else f'it rises afterward ({val_losses[-1]:.4f} at the final epoch) -- mild overfitting begins around epoch {best_epoch + 1}'}.")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(range(1, EPOCHS + 1), train_losses, label="train loss", color="#4C72B0")
ax[0].plot(range(1, EPOCHS + 1), val_losses, label="val loss", color="#C44E52")
ax[0].axvline(best_epoch + 1, color="grey", ls="--", lw=1, label=f"best val loss (epoch {best_epoch+1})")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].set_title("Training & validation loss")
ax[0].legend(fontsize=8)
ax[1].plot(range(1, EPOCHS + 1), val_accs, color="#55A868")
ax[1].axvline(best_epoch + 1, color="grey", ls="--", lw=1)
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation accuracy"); ax[1].set_title("Validation accuracy")
fig.tight_layout()
fig.savefig("figures/03_loss_accuracy_curves.png", bbox_inches="tight")
plt.close(fig)

# C: final test-set evaluation
main_model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        preds = main_model(xb).argmax(dim=1)
        all_preds.append(preds)
        all_labels.append(yb)
all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

test_acc = accuracy_score(all_labels, all_preds)
test_prec = precision_score(all_labels, all_preds, average="macro")
test_rec = recall_score(all_labels, all_preds, average="macro")
test_f1 = f1_score(all_labels, all_preds, average="macro")
cm = confusion_matrix(all_labels, all_preds)
print(f"\n--- Final test-set evaluation ({len(all_labels)} images) ---")
print(f"Accuracy {test_acc:.4f} | Macro-Precision {test_prec:.4f} | "
      f"Macro-Recall {test_rec:.4f} | Macro-F1 {test_f1:.4f}")

cm_offdiag = cm.copy()
np.fill_diagonal(cm_offdiag, 0)
i, j = np.unravel_index(cm_offdiag.argmax(), cm_offdiag.shape)
print(f"Most confused pair: true '{CLASS_NAMES[i]}' predicted as '{CLASS_NAMES[j]}' "
      f"({cm_offdiag[i, j]} of {cm[i].sum()} test images), and true '{CLASS_NAMES[j]}' "
      f"predicted as '{CLASS_NAMES[i]}' ({cm[j, i]} of {cm[j].sum()}) -- {CLASS_NAMES[i]} "
      f"and {CLASS_NAMES[j]} are both upper-body garments with a similar silhouette in a "
      f"low-resolution 28x28 greyscale image, without colour or texture to distinguish them.")

fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(10)); ax.set_yticklabels(CLASS_NAMES, fontsize=8)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Test confusion matrix (accuracy {test_acc:.4f})")
for r in range(10):
    for c in range(10):
        if cm[r, c] > 0:
            ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                     color="white" if cm[r, c] > cm.max() / 2 else "black", fontsize=6)
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig("figures/04_confusion_matrix.png", bbox_inches="tight")
plt.close(fig)

# C: controlled ablation -- depth
print("\n--- Ablation: network depth (0, 1, 2 hidden layers) ---")


class LinearNet(nn.Module):
    """No hidden layer: a direct 784 -> 10 linear map (multinomial logistic regression)."""
    def __init__(self, n_in=784, n_out=10):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)

    def forward(self, x):
        return self.fc(x)


class OneHiddenNet(nn.Module):
    def __init__(self, n_in=784, hidden=256, n_out=10):
        super().__init__()
        self.fc1 = nn.Linear(n_in, hidden)
        self.fc2 = nn.Linear(hidden, n_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


ABLATION_EPOCHS = 12
ablation_configs = {
    "0 hidden layers (linear)": lambda: LinearNet(),
    "1 hidden layer (256)": lambda: OneHiddenNet(),
    "2 hidden layers (256, 128)": lambda: FeedforwardNet(),
}
ablation_results = {}
for name, model_fn in ablation_configs.items():
    torch.manual_seed(RANDOM_STATE)
    model = model_fn()
    tr_l, va_l, va_a = train_model(model, ABLATION_EPOCHS, seed=RANDOM_STATE)
    model.eval()
    with torch.no_grad():
        preds_list = [model(xb).argmax(dim=1) for xb, _ in test_loader]
        labels_list = [yb for _, yb in test_loader]
    preds_ab = torch.cat(preds_list).numpy()
    labels_ab = torch.cat(labels_list).numpy()
    acc_ab = accuracy_score(labels_ab, preds_ab)
    n_params = sum(p.numel() for p in model.parameters())
    ablation_results[name] = {"test_acc": acc_ab, "val_loss_history": va_l,
                               "val_acc_history": va_a, "n_params": n_params}
    print(f"{name:30s} params: {n_params:,} | final val acc: {va_a[-1]:.4f} | "
          f"TEST acc: {acc_ab:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for name, r in ablation_results.items():
    ax[0].plot(range(1, ABLATION_EPOCHS + 1), r["val_loss_history"], label=name)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("validation loss")
ax[0].set_title("Depth ablation: validation loss"); ax[0].legend(fontsize=7)
names_short = list(ablation_results.keys())
ax[1].bar(names_short, [ablation_results[n]["test_acc"] for n in names_short], color="#8172B2")
ax[1].set_ylabel("test accuracy"); ax[1].set_title("Depth ablation: final test accuracy")
ax[1].tick_params(axis="x", rotation=20)
for i, n in enumerate(names_short):
    ax[1].annotate(f"{ablation_results[n]['test_acc']:.4f}", (i, ablation_results[n]["test_acc"]),
                    ha="center", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig("figures/05_depth_ablation.png", bbox_inches="tight")
plt.close(fig)

acc_0, acc_1, acc_2 = (ablation_results["0 hidden layers (linear)"]["test_acc"],
                        ablation_results["1 hidden layer (256)"]["test_acc"],
                        ablation_results["2 hidden layers (256, 128)"]["test_acc"])
print(f"\n0->1 hidden layers: {acc_1 - acc_0:+.4f} test accuracy change")
print(f"1->2 hidden layers: {acc_2 - acc_1:+.4f} test accuracy change")

# C: save/load state_dict, verify round-trip
torch.save(main_model.state_dict(), "feedforward_mnist_state_dict.pt")
reloaded_model = FeedforwardNet()
reloaded_model.load_state_dict(torch.load("feedforward_mnist_state_dict.pt", weights_only=True))
reloaded_model.eval()

held_out_xb, held_out_yb = next(iter(test_loader))
with torch.no_grad():
    original_preds = main_model(held_out_xb).argmax(dim=1)
    reloaded_preds = reloaded_model(held_out_xb).argmax(dim=1)
print(f"\nReloaded model predictions exactly match original on a held-out batch: "
      f"{torch.equal(original_preds, reloaded_preds)}")

print("\nFigures written to figures/")
print("\n=== SUMMARY ===")
print(f"Test accuracy {test_acc:.4f} | Precision {test_prec:.4f} | Recall {test_rec:.4f} | "
      f"F1 {test_f1:.4f} | Training time {train_time:.1f}s ({EPOCHS} epochs)")
for name, r in ablation_results.items():
    print(f"Ablation -- {name:30s} test acc {r['test_acc']:.4f} ({r['n_params']:,} params)")

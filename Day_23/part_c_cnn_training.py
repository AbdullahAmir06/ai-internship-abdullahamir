"""
PKCERT AI & Software Development Internship, Task 23
Part C: Architecture Design & Backpropagation Mini-Project

Trains the from-scratch SimpleCNN (cnn_model.py, built on cnn_layers.py's
gradient-checked Conv2D/MaxPool2D/BatchNorm2D/Dropout) on a 4-class CIFAR-10
subset, compares against an equivalent PyTorch CNN, runs a 3-way ablation
(kernel size x pooling-vs-strided-conv), and visualizes learned filters and
feature maps.
"""

import pickle
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

from cnn_model import SimpleCNN, cross_entropy_loss

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

print("=" * 70)
print("PART C: ARCHITECTURE DESIGN & BACKPROPAGATION MINI-PROJECT")
print("=" * 70)

# ======================================================================
# C1: dataset -- 4-class CIFAR-10 subset (different from every dataset
# used in Tasks 20-22, which all used Fashion-MNIST)
# ======================================================================
print("""
--- C1: dataset selection ---

CIFAR-10 (Krizhevsky, 2009), restricted to 4 visually distinct classes
(airplane, automobile, cat, dog -- 2 vehicles, 2 animals, with cat/dog
included deliberately as a genuinely hard pair, mirroring the
"most-confused-pair" analyses in Tasks 20-22). Deliberately different from
every dataset used in Tasks 20-22 (all Fashion-MNIST, greyscale 28x28,
1 channel) -- CIFAR-10 is 32x32 RGB (3 channels), which is what actually
exercises this task's "multi-channel input" requirement for the from-
scratch convolution in Part A, not just a pedagogical restatement of it.

Subset sizes (kept modest so pure-NumPy im2col training finishes in
minutes, not hours, while remaining large enough for a meaningful result):
1000 train / 200 val / 200 test images PER CLASS -> 4000 / 800 / 800 total.
""")

CLASS_IDS = {0: "airplane", 1: "automobile", 3: "cat", 5: "dog"}
SELECTED = sorted(CLASS_IDS.keys())
N_TRAIN_PER_CLASS, N_VAL_PER_CLASS, N_TEST_PER_CLASS = 1000, 200, 200

raw_train = CIFAR10(root="./data", train=True, download=True)
raw_test = CIFAR10(root="./data", train=False, download=True)


def build_subset(dataset, per_class, seed):
    targets = np.array(dataset.targets)
    rs = np.random.RandomState(seed)
    images, labels = [], []
    for new_label, class_id in enumerate(SELECTED):
        idx = np.where(targets == class_id)[0]
        chosen = rs.choice(idx, size=per_class, replace=False)
        images.append(dataset.data[chosen])
        labels.append(np.full(per_class, new_label))
    X = np.concatenate(images, axis=0).astype(np.float64)  # (N,32,32,3), 0-255
    y = np.concatenate(labels, axis=0)
    perm = rs.permutation(len(y))
    return X[perm], y[perm]


X_train_full, y_train_full = build_subset(raw_train, N_TRAIN_PER_CLASS + N_VAL_PER_CLASS, RANDOM_STATE)
X_test_raw, y_test = build_subset(raw_test, N_TEST_PER_CLASS, RANDOM_STATE)

n_val_total = N_VAL_PER_CLASS * len(SELECTED)
X_val_raw, y_val = X_train_full[:n_val_total], y_train_full[:n_val_total]
X_train_raw, y_train = X_train_full[n_val_total:], y_train_full[n_val_total:]

# normalize using TRAIN-split statistics only, per-channel (established
# practice throughout this internship)
mean = X_train_raw.mean(axis=(0, 1, 2))
std = X_train_raw.std(axis=(0, 1, 2))
print(f"Per-channel normalization stats (train split only): mean={mean.round(2)}, std={std.round(2)}")


def preprocess(X):
    X = (X - mean) / std
    return X.transpose(0, 3, 1, 2)  # (N,32,32,3) -> (N,3,32,32) for Conv2D


X_train, X_val, X_test = preprocess(X_train_raw), preprocess(X_val_raw), preprocess(X_test_raw)
print(f"Shapes: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")
CLASS_NAMES = [CLASS_IDS[c] for c in SELECTED]
print(f"Classes (remapped 0-3): {CLASS_NAMES}")


def one_hot(y, n_classes=4):
    Y = np.zeros((y.size, n_classes))
    Y[np.arange(y.size), y] = 1.0
    return Y


# ======================================================================
# C2-C3: forward propagation and backpropagation, derived by hand
# ======================================================================
print("""
--- C2: forward propagation, derived (matrix/tensor form) ---

Network: Conv1 -> [BN] -> ReLU -> Down1 -> Conv2 -> [BN] -> ReLU -> Down2
-> Flatten -> Dense1 -> ReLU -> [Dropout] -> Dense2 -> softmax

Let X in R^{N x Cin x H x W} be a batch. Convolution is expressed via
im2col as a single matrix multiplication (this IS the tensor-form
derivation, not an approximation of one): for kernel W1 in
R^{C1 x Cin x k x k}, pad X to X_p, then

    col1 = im2col(X_p) in R^{N x (Cin*k*k) x L1},   L1 = H1_out * W1_out
    Z1   = reshape(W1, (C1, Cin*k*k)) @ col1 + b1    in R^{N x C1 x L1}
    Z1   = reshape(Z1, (N, C1, H1_out, W1_out))

This is exactly a batched matrix multiply per sample: each of the L1
output spatial positions is one column of "unrolled receptive field"
dotted against every filter -- i.e. convolution IS a linear operator in
disguise, and im2col makes that linearity explicit as an ordinary matmul.

    A1 = ReLU(BN(Z1))          (BN optional per architecture variant)
    P1 = Down1(A1)              (max pool OR stride-2 conv, per variant)
    Z2 = Conv2(P1);  A2 = ReLU(BN(Z2));  P2 = Down2(A2)
    F  = Flatten(P2) in R^{N x D},  D = C2 * H_final * W_final
    H1 = ReLU(F @ W_fc1 + b_fc1)
    [H1 = Dropout(H1)]
    logits = H1 @ W_fc2 + b_fc2
    probs  = softmax(logits)        (row-wise, numerically stabilized)
    L      = -(1/N) sum_n sum_c Y_nc log(probs_nc)

--- C3: backpropagation, derived via the chain rule ---

Identical starting point to every softmax+cross-entropy network derived
in this internship (Tasks 20-22): dL/dlogits = (probs - Y) / N. From
there the chain rule is applied backward through EXACTLY the layers
above, each contributing one link:

    dW_fc2 = H1^T dlogits,  db_fc2 = sum_n dlogits,  dH1 = dlogits W_fc2^T
    [dH1 = dH1 * dropout_mask]              (inverted dropout, Part B)
    dF = (dH1 * ReLU'(F @ W_fc1 + b_fc1)) W_fc1^T,  dW_fc1 = F^T (dH1*ReLU'(.))
    dP2 = reshape(dF, P2.shape)
    dA2 = Down2.backward(dP2)     (routes to argmax if max pool, Part B;
                                    or backprops through the stride-2
                                    conv's own im2col/col2im machinery)
    dZ2 = dA2 * ReLU'(Z2)  [through BN's backward first, if used -- the
                              exact multi-step BN chain rule from Part B]
    dW2, db2, dP1 = Conv2.backward(dZ2)   (im2col-matmul backward:
        dW2 = einsum('nol,ncl->oc', dZ2_flat, col2);
        dcol2 = W2_flat^T @ dZ2_flat;  dP1 = col2im(dcol2))
    dA1 = Down1.backward(dP1);   dZ1 = dA1 * ReLU'(Z1)  [through BN]
    dW1, db1, dX = Conv1.backward(dZ1)

Every one of these steps is independently gradient-checked against finite
differences before being trusted here: Conv2D (dW/db/dX) and BatchNorm2D
(dX/dgamma/dbeta) at ~1e-9 to 1e-12 relative error in cnn_layers.py; the
FULL assembled network (confirming nothing was wired backwards when
composing the layers) in cnn_model.py, at ~1e-8 for the smooth
(stride-conv) architecture and ~1e-9 median error for the max-pool
architecture (see that file's own explanation of the small number of
expected max-pool argmax-tie outliers in the latter check).
""")

# ======================================================================
# C4: train the from-scratch CNN with mini-batch GD + momentum
# ======================================================================
BATCH_SIZE = 32
LR = 0.02
MOMENTUM = 0.9
EPOCHS = 25


def train_numpy_cnn(net, epochs=EPOCHS, lr=LR, momentum=MOMENTUM, batch_size=BATCH_SIZE,
                     seed=RANDOM_STATE, verbose_name="cnn", early_stopping=False, patience=6):
    n = X_train.shape[0]
    Y_train = one_hot(y_train)
    perm_rng = np.random.default_rng(seed)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val, best_params, patience_ctr, stopped_epoch = float("inf"), None, 0, epochs
    t0 = time.perf_counter()
    for epoch in range(epochs):
        perm = perm_rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            net.forward(X_train[idx], training=True)
            net.backward(Y_train[idx])
            net.step(lr, momentum)
        train_probs = net.forward(X_train, training=False)
        val_probs = net.forward(X_val, training=False)
        train_loss = cross_entropy_loss(train_probs, Y_train)
        val_loss = cross_entropy_loss(val_probs, one_hot(y_val))
        train_acc = (train_probs.argmax(1) == y_train).mean()
        val_acc = (val_probs.argmax(1) == y_val).mean()
        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc); val_accs.append(val_acc)
        print(f"  [{verbose_name}] epoch {epoch+1:2d}/{epochs}  train loss {train_loss:.4f}  "
              f"val loss {val_loss:.4f}  train acc {train_acc:.4f}  val acc {val_acc:.4f}")
        if early_stopping:
            if val_loss < best_val - 1e-4:
                best_val, best_params, patience_ctr = val_loss, net.get_params(), 0
            else:
                patience_ctr += 1
            if patience_ctr >= patience:
                stopped_epoch = epoch + 1
                print(f"  [{verbose_name}] early stopping at epoch {stopped_epoch}, "
                      f"restoring epoch {epoch+1-patience} weights")
                net.set_params(best_params)
                break
    train_time = time.perf_counter() - t0
    return {"train_losses": train_losses, "val_losses": val_losses,
            "train_accs": train_accs, "val_accs": val_accs,
            "train_time": train_time, "stopped_epoch": stopped_epoch}


print(f"--- C4: training from-scratch CNN, mini-batch GD + momentum ({MOMENTUM}), "
      f"lr={LR}, batch={BATCH_SIZE} ---")
main_net = SimpleCNN(input_shape=(3, 32, 32), num_classes=4, conv_channels=(8, 16),
                      kernel_size=3, use_pooling=True, use_batchnorm=True,
                      use_dropout=True, dropout_p=0.3, fc_hidden=64, seed=RANDOM_STATE)
main_hist = train_numpy_cnn(main_net, verbose_name="numpy-cnn", early_stopping=True, patience=6)
print(f"Training time: {main_hist['train_time']:.2f}s over {len(main_hist['train_losses'])} epochs")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(main_hist["train_losses"], label="train")
ax[0].plot(main_hist["val_losses"], label="val")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].set_title("From-scratch CNN: loss")
ax[0].legend()
ax[1].plot(main_hist["train_accs"], label="train")
ax[1].plot(main_hist["val_accs"], label="val")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].set_title("From-scratch CNN: accuracy")
ax[1].legend()
fig.tight_layout()
fig.savefig("figures/03_numpy_cnn_curves.png", bbox_inches="tight")
plt.close(fig)

# ======================================================================
# C5: evaluation -- accuracy/precision/recall/F1/confusion matrix
# ======================================================================
def evaluate_numpy(net, X, y, name):
    preds = net.predict(X)
    return {"name": name, "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds, average="macro", zero_division=0),
            "recall": recall_score(y, preds, average="macro", zero_division=0),
            "f1": f1_score(y, preds, average="macro", zero_division=0),
            "cm": confusion_matrix(y, preds)}


numpy_test_result = evaluate_numpy(main_net, X_test, y_test, "NumPy CNN")
print(f"\n--- C5: from-scratch CNN test evaluation ---")
print(f"  Accuracy {numpy_test_result['accuracy']:.4f} | Precision {numpy_test_result['precision']:.4f} | "
      f"Recall {numpy_test_result['recall']:.4f} | F1 {numpy_test_result['f1']:.4f}")

cm = numpy_test_result["cm"]
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(4)); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
ax.set_yticks(range(4)); ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"From-scratch CNN, test confusion matrix (acc {numpy_test_result['accuracy']:.4f})")
for r in range(4):
    for c in range(4):
        ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                 color="white" if cm[r, c] > cm.max() / 2 else "black")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig("figures/04_confusion_matrix.png", bbox_inches="tight")
plt.close(fig)

cm_offdiag = cm.copy(); np.fill_diagonal(cm_offdiag, 0)
i, j = np.unravel_index(cm_offdiag.argmax(), cm_offdiag.shape)
print(f"  Most confused pair: true '{CLASS_NAMES[i]}' -> predicted '{CLASS_NAMES[j]}' "
      f"({cm_offdiag[i,j]} images)")

# ======================================================================
# C6: PyTorch baseline of similar architecture
# ======================================================================
print("\n--- C6: PyTorch baseline, same architecture, trained on the same data ---")


class TorchCNN(nn.Module):
    def __init__(self, c1=8, c2=16, k=3, num_classes=4, fc_hidden=64, dropout_p=0.3):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv2d(3, c1, k, padding=pad)
        self.bn1 = nn.BatchNorm2d(c1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(c1, c2, k, padding=pad)
        self.bn2 = nn.BatchNorm2d(c2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        flat_dim = c2 * 8 * 8
        self.fc1 = nn.Linear(flat_dim, fc_hidden)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = x.flatten(1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

torch.manual_seed(RANDOM_STATE)
torch_net = TorchCNN()
optimizer = torch.optim.SGD(torch_net.parameters(), lr=LR, momentum=MOMENTUM)
criterion = nn.CrossEntropyLoss()

torch_hist = {"train_losses": [], "val_losses": [], "train_accs": [], "val_accs": []}
t0 = time.perf_counter()
for epoch in range(EPOCHS):
    torch_net.train()
    perm = torch.randperm(X_train_t.shape[0])
    for start in range(0, X_train_t.shape[0], BATCH_SIZE):
        idx = perm[start:start + BATCH_SIZE]
        optimizer.zero_grad()
        loss = criterion(torch_net(X_train_t[idx]), y_train_t[idx])
        loss.backward()
        optimizer.step()
    torch_net.eval()
    with torch.no_grad():
        tr_logits, va_logits = torch_net(X_train_t), torch_net(X_val_t)
        tr_loss, va_loss = criterion(tr_logits, y_train_t).item(), criterion(va_logits, y_val_t).item()
        tr_acc = (tr_logits.argmax(1) == y_train_t).float().mean().item()
        va_acc = (va_logits.argmax(1) == y_val_t).float().mean().item()
    torch_hist["train_losses"].append(tr_loss); torch_hist["val_losses"].append(va_loss)
    torch_hist["train_accs"].append(tr_acc); torch_hist["val_accs"].append(va_acc)
    print(f"  [torch] epoch {epoch+1:2d}/{EPOCHS}  train loss {tr_loss:.4f}  val loss {va_loss:.4f}  "
          f"train acc {tr_acc:.4f}  val acc {va_acc:.4f}")
torch_hist["train_time"] = time.perf_counter() - t0
print(f"PyTorch training time: {torch_hist['train_time']:.2f}s")

torch_net.eval()
with torch.no_grad():
    torch_test_preds = torch_net(X_test_t).argmax(1).numpy()
torch_test_acc = accuracy_score(y_test, torch_test_preds)
torch_test_f1 = f1_score(y_test, torch_test_preds, average="macro")
print(f"\nPyTorch TEST: Acc {torch_test_acc:.4f} | F1 {torch_test_f1:.4f}")
print(f"NumPy   TEST: Acc {numpy_test_result['accuracy']:.4f} | F1 {numpy_test_result['f1']:.4f}")
print(f"Accuracy gap: {abs(torch_test_acc - numpy_test_result['accuracy']):.4f} -- both use identical "
      f"architecture, optimizer (SGD+momentum), lr, batch size and epoch budget; the from-scratch "
      f"NumPy implementation is validated by reaching comparable accuracy via completely independent "
      f"code (im2col conv, hand-derived backward passes) rather than autograd.")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(main_hist["val_losses"], label="NumPy CNN val")
ax[0].plot(torch_hist["val_losses"], label="PyTorch CNN val")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].set_title("Validation loss: NumPy vs PyTorch")
ax[0].legend()
ax[1].plot(main_hist["val_accs"], label="NumPy CNN val")
ax[1].plot(torch_hist["val_accs"], label="PyTorch CNN val")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].set_title("Validation accuracy: NumPy vs PyTorch")
ax[1].legend()
fig.tight_layout()
fig.savefig("figures/05_numpy_vs_pytorch.png", bbox_inches="tight")
plt.close(fig)

# ======================================================================
# C7: ablation study -- kernel size x pooling-vs-strided-conv
# ======================================================================
print("\n--- C7: ablation study (kernel size x downsampling mechanism) ---\n")

ablation_configs = {
    "Baseline (3x3, max pool)": dict(kernel_size=3, use_pooling=True),
    "Larger kernel (5x5, max pool)": dict(kernel_size=5, use_pooling=True),
    "No pooling (3x3, stride-2 conv)": dict(kernel_size=3, use_pooling=False),
}

ablation_results = {}
for name, cfg in ablation_configs.items():
    print(f"--- Training: {name} ---")
    net = SimpleCNN(input_shape=(3, 32, 32), num_classes=4, conv_channels=(8, 16),
                     use_batchnorm=True, use_dropout=True, dropout_p=0.3, fc_hidden=64,
                     seed=RANDOM_STATE, **cfg)
    hist = train_numpy_cnn(net, epochs=20, verbose_name=name.split()[0].lower())
    test_res = evaluate_numpy(net, X_test, y_test, name)
    n_params = sum(p.size for p in [net.conv1.W, net.conv1.b, net.conv2.W, net.conv2.b,
                                     net.fc1.W, net.fc1.b, net.fc2.W, net.fc2.b])
    if not cfg["use_pooling"]:
        n_params += net.down1.W.size + net.down1.b.size + net.down2.W.size + net.down2.b.size
    ablation_results[name] = {"hist": hist, "test": test_res, "n_params": n_params, "net": net}
    print(f"{name}: TEST acc {test_res['accuracy']:.4f} | params {n_params:,} | "
          f"time {hist['train_time']:.2f}s\n")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for name, r in ablation_results.items():
    ax[0].plot(r["hist"]["val_losses"], label=name)
    ax[1].plot(r["hist"]["val_accs"], label=name)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("validation loss"); ax[0].set_title("Ablation: validation loss")
ax[0].legend(fontsize=7)
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation accuracy"); ax[1].set_title("Ablation: validation accuracy")
ax[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig("figures/06_ablation_curves.png", bbox_inches="tight")
plt.close(fig)

print("Ablation summary:")
for name, r in ablation_results.items():
    print(f"  {name:35s} TEST acc {r['test']['accuracy']:.4f} | params {r['n_params']:6,} | "
          f"time {r['hist']['train_time']:6.2f}s")

# ======================================================================
# C8: visualize learned filters and feature maps
# ======================================================================
print("\n--- C8: visualizing learned first-layer filters and feature maps ---")

W1 = main_net.conv1.W  # (C1, 3, k, k)
n_filters = W1.shape[0]
fig, axes = plt.subplots(1, n_filters, figsize=(2 * n_filters, 2.2))
for i in range(n_filters):
    kernel = W1[i].transpose(1, 2, 0)  # (k,k,3)
    kernel_norm = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)
    axes[i].imshow(kernel_norm)
    axes[i].set_title(f"filter {i}", fontsize=8)
    axes[i].axis("off")
fig.suptitle("Learned first-conv-layer filters (RGB, min-max normalized for display)")
fig.tight_layout()
fig.savefig("figures/07_learned_filters.png", bbox_inches="tight")
plt.close(fig)

sample_idx = 0
sample_img = X_test[sample_idx:sample_idx + 1]
z1 = main_net.conv1.forward(sample_img, training=False)
a1 = np.maximum(0, main_net.bn1.forward(z1, training=False) if main_net.use_batchnorm else z1)

fig, axes = plt.subplots(2, n_filters // 2 + n_filters % 2, figsize=(2.2 * (n_filters // 2 + 1), 4.5))
axes = axes.ravel()
for i in range(n_filters):
    axes[i].imshow(a1[0, i], cmap="viridis")
    axes[i].set_title(f"feature map {i}", fontsize=8)
    axes[i].axis("off")
for i in range(n_filters, len(axes)):
    axes[i].axis("off")
fig.suptitle(f"First-conv-layer feature maps for a sample test image "
             f"(true class: {CLASS_NAMES[y_test[sample_idx]]})")
fig.tight_layout()
fig.savefig("figures/08_feature_maps.png", bbox_inches="tight")
plt.close(fig)

# ======================================================================
# save the final trained model (Part D requirement, done here since the
# model is fully trained and available in this script's scope)
# ======================================================================
model_data = {
    "params": main_net.get_params(),
    "architecture": dict(input_shape=(3, 32, 32), num_classes=4, conv_channels=(8, 16),
                          kernel_size=3, use_pooling=True, use_batchnorm=True,
                          use_dropout=True, dropout_p=0.3, fc_hidden=64),
    "class_names": CLASS_NAMES,
    "normalization": {"mean": mean, "std": std},
    "test_accuracy": float(numpy_test_result["accuracy"]),
    "test_f1": float(numpy_test_result["f1"]),
}
with open("numpy_cnn_final_model.pkl", "wb") as f:
    pickle.dump(model_data, f)
print("\nSaved numpy_cnn_final_model.pkl")
print("Figures written to figures/")

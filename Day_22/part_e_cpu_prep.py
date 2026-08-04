"""
PKCERT AI & Software Development Internship, Task 22
Part E (CPU-side prep): re-trains and saves the final NumPy model with the
full regularization stack (Dropout + BatchNorm + Early Stopping) -- the
same configuration identified in Part C as the best/most balanced -- so
it's ready on disk for the NumPy-vs-framework integration comparison once
the Colab notebook's real GPU-trained model numbers are available.
"""
import pickle
import numpy as np
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch

from numpy_network import NumpyMLP, cross_entropy_loss

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)

raw_train_full = FashionMNIST(root="./data", train=True, download=True)
N_TRAIN, N_VAL = 50000, 10000
generator = torch.Generator().manual_seed(RANDOM_STATE)
train_subset, val_subset = random_split(raw_train_full, [N_TRAIN, N_VAL], generator=generator)
train_images_raw = raw_train_full.data[train_subset.indices].float() / 255.0
mean, std = train_images_raw.mean().item(), train_images_raw.std().item()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,)),
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


def one_hot(y, n_classes=10):
    Y = np.zeros((y.size, n_classes))
    Y[np.arange(y.size), y] = 1.0
    return Y


Y_train_np = one_hot(y_train_np)

BATCH_SIZE, LR, MAX_EPOCHS, PATIENCE = 256, 0.1, 40, 5
model = NumpyMLP([784, 256, 128, 10], use_dropout=True, dropout_p=0.3,
                  use_batchnorm=True, seed=RANDOM_STATE)
perm_rng = np.random.default_rng(RANDOM_STATE)
best_val, best_params, patience_ctr = float("inf"), None, 0

for epoch in range(MAX_EPOCHS):
    perm = perm_rng.permutation(X_train_np.shape[0])
    for start in range(0, X_train_np.shape[0], BATCH_SIZE):
        idx = perm[start:start + BATCH_SIZE]
        model.forward(X_train_np[idx], training=True)
        grads = model.backward(Y_train_np[idx])
        model.step(grads, LR)
    val_probs = model.forward(X_val_np, training=False)
    val_loss = cross_entropy_loss(val_probs, one_hot(y_val_np))
    val_acc = (val_probs.argmax(1) == y_val_np).mean()
    print(f"epoch {epoch+1:2d}/{MAX_EPOCHS}  val loss {val_loss:.4f}  val acc {val_acc:.4f}")
    if val_loss < best_val - 1e-4:
        best_val, best_params, patience_ctr = val_loss, model.get_params(), 0
    else:
        patience_ctr += 1
    if patience_ctr >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1}, restoring epoch {epoch+1-PATIENCE} weights")
        model.set_params(best_params)
        break

test_probs = model.forward(X_test_np, training=False)
test_preds = test_probs.argmax(1)
test_acc = (test_preds == y_test_np).mean()
print(f"\nFinal NumPy model (Dropout+BatchNorm+EarlyStopping) TEST accuracy: {test_acc:.4f}")

with open("numpy_final_model.pkl", "wb") as f:
    pickle.dump({"params": model.get_params(), "sizes": model.sizes,
                 "use_dropout": True, "dropout_p": 0.3, "use_batchnorm": True,
                 "test_accuracy": float(test_acc), "normalization": {"mean": mean, "std": std}}, f)
print("Saved numpy_final_model.pkl")

"""
Shared utilities for Task 24 (CNN & Transfer Learning): dataset subsetting,
augmentation pipelines, training/eval loops, and small helpers reused by every
part_* script. Kept in one place so Parts A-D train/evaluate on *exactly* the
same data splits -- required for the Part D custom-vs-transfer comparison to
be a fair, apples-to-apples number.

Seed 42 throughout, CPU only (no CUDA device available in this environment --
noted explicitly in Part B's inference-latency analysis and the README).
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

SEED = 42
ROOT = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
for d in (MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    d.mkdir(exist_ok=True)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Per-class image counts for the subsets used throughout this task. Full
# CIFAR-10 (50k/10k) is intractable to train >10 model configurations on a
# 12-core CPU with no GPU in reasonable time; Task 23 hit the same wall and
# subset for the same reason. All 10 classes are kept (unlike Task 23's
# 4-class subset) since Part B/C explicitly grade per-class breakdowns and a
# 3x3 confusion matrix would be too easy to read too much into.
N_TRAIN_PER_CLASS = 600   # 6,000 train images total  (>> the 5,000 min)
N_VAL_PER_CLASS = 100     # 1,000 val
N_TEST_PER_CLASS = 100    # 1,000 test

# Transfer learning (Part C/D) re-forwards full ImageNet backbones (VGG16 has
# 138M params) through every image at 128x128; a further subset keeps 6
# architecture x strategy configurations plus ablations tractable.
N_TRAIN_PER_CLASS_TL = 200  # 2,000 train
N_VAL_PER_CLASS_TL = 40     # 400 val
N_TEST_PER_CLASS_TL = 40    # 400 test

TRANSFER_RESOLUTION = 128  # see README / Report for the 224->128 rationale


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _class_balanced_indices(targets, n_per_class, seed=SEED):
    rng = np.random.RandomState(seed)
    targets = np.array(targets)
    chosen = []
    for c in range(10):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        chosen.append(idx[:n_per_class])
    idx = np.concatenate(chosen)
    rng.shuffle(idx)
    return idx.tolist()


def get_raw_cifar():
    """
    Returns the raw train/test CIFAR-10 datasets.

    The canonical torchvision download host (cs.toronto.edu) was measured at
    ~200 bytes/sec from this environment (a multi-hour download for a 170MB
    file) -- not a transient hiccup, a sustained rate-limit/throttle. The
    fast.ai S3 mirror (https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz,
    the same CIFAR-10 images, pre-extracted as train/<class>/*.png and
    test/<class>/*.png) served the same data at ~140KB/s, so that's what's
    used here via ImageFolder instead of torchvision's CIFAR10 pickle loader.
    """
    extracted = DATA_DIR / "cifar10"
    if not extracted.exists():
        archive = DATA_DIR / "cifar10_fastai.tgz"
        if not archive.exists():
            raise FileNotFoundError(
                f"{archive} not found -- download it first from "
                "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
            )
        import tarfile
        with tarfile.open(archive) as tf:
            tf.extractall(DATA_DIR)
    train_full = torchvision.datasets.ImageFolder(str(extracted / "train"))
    test_full = torchvision.datasets.ImageFolder(str(extracted / "test"))
    assert train_full.classes == CIFAR10_CLASSES, train_full.classes
    return train_full, test_full


def build_split_indices():
    """
    One fixed set of indices (train/val/test, both the small 'custom CNN'
    sizing and the smaller 'transfer learning' sizing) computed once and
    cached to disk, so every script/notebook cell that touches data uses
    identical images.
    """
    cache = RESULTS_DIR / "split_indices.json"
    if cache.exists():
        return json.loads(cache.read_text())

    train_full, test_full = get_raw_cifar()
    train_targets = train_full.targets
    test_targets = test_full.targets

    # test split for both custom + TL pipelines is the same held-out set
    # (TL just uses fewer of those same images per class for compute reasons)
    train_idx_all = _class_balanced_indices(train_targets, N_TRAIN_PER_CLASS + N_VAL_PER_CLASS, seed=SEED)
    # split into train/val: for each class, first N_TRAIN go to train, rest to val
    targets_arr = np.array(train_targets)
    train_idx, val_idx = [], []
    rng = np.random.RandomState(SEED)
    for c in range(10):
        idx = np.where(targets_arr == c)[0]
        rng.shuffle(idx)
        train_idx.extend(idx[:N_TRAIN_PER_CLASS].tolist())
        val_idx.extend(idx[N_TRAIN_PER_CLASS:N_TRAIN_PER_CLASS + N_VAL_PER_CLASS].tolist())
    test_idx = _class_balanced_indices(test_targets, N_TEST_PER_CLASS, seed=SEED)

    # TL subsets are prefixes of the above per-class lists (same images, fewer)
    train_idx_tl, val_idx_tl = [], []
    for c in range(10):
        idx = np.where(targets_arr == c)[0]
        rng2 = np.random.RandomState(SEED)
        rng2.shuffle(idx)
        train_idx_tl.extend(idx[:N_TRAIN_PER_CLASS_TL].tolist())
        val_idx_tl.extend(idx[N_TRAIN_PER_CLASS:N_TRAIN_PER_CLASS + N_VAL_PER_CLASS_TL].tolist())
    test_idx_tl = []
    rng3 = np.random.RandomState(SEED)
    targets_test_arr = np.array(test_targets)
    for c in range(10):
        idx = np.where(targets_test_arr == c)[0]
        rng3.shuffle(idx)
        test_idx_tl.extend(idx[:N_TEST_PER_CLASS_TL].tolist())

    out = dict(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
               train_idx_tl=train_idx_tl, val_idx_tl=val_idx_tl, test_idx_tl=test_idx_tl)
    cache.write_text(json.dumps(out))
    return out


# ---------------------------------------------------------------- Part A/B --

def custom_transforms(augment):
    if augment:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    return T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])


class TransformedSubset(torch.utils.data.Dataset):
    """Wraps a CIFAR dataset + index list + its own transform (so train can
    augment while val/test on the *same underlying dataset object* don't)."""
    def __init__(self, base_dataset, indices, transform):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        return self.transform(img), label


def get_custom_loaders(batch_size=128, augment_train=True, num_workers=0):
    train_full, test_full = get_raw_cifar()
    idx = build_split_indices()
    train_ds = TransformedSubset(train_full, idx["train_idx"], custom_transforms(augment_train))
    val_ds = TransformedSubset(train_full, idx["val_idx"], custom_transforms(False))
    test_ds = TransformedSubset(test_full, idx["test_idx"], custom_transforms(False))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


# -------------------------------------------------------------------- Part C

def transfer_transforms(augment, resolution=TRANSFER_RESOLUTION, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    resize_pad = [T.Resize(resolution + 8), T.CenterCrop(resolution)]
    if augment:
        return T.Compose([
            T.Resize(resolution + 8),
            T.RandomCrop(resolution),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose(resize_pad + [T.ToTensor(), T.Normalize(mean, std)])


def get_transfer_loaders(batch_size=32, augment_train=True, resolution=TRANSFER_RESOLUTION,
                          mean=IMAGENET_MEAN, std=IMAGENET_STD, num_workers=0):
    train_full, test_full = get_raw_cifar()
    idx = build_split_indices()
    train_ds = TransformedSubset(train_full, idx["train_idx_tl"], transfer_transforms(augment_train, resolution, mean, std))
    val_ds = TransformedSubset(train_full, idx["val_idx_tl"], transfer_transforms(False, resolution, mean, std))
    test_ds = TransformedSubset(test_full, idx["test_idx_tl"], transfer_transforms(False, resolution, mean, std))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


# ------------------------------------------------------------- Train / eval

def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if is_train:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def train_model(model, train_loader, val_loader, epochs, optimizer, scheduler=None,
                 device="cpu", verbose=True, early_stopping_patience=None, log_prefix=""):
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_state, patience_ctr = -1.0, None, 0
    for ep in range(epochs):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        if scheduler is not None:
            scheduler.step()
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if verbose:
            print(f"{log_prefix}epoch {ep+1}/{epochs} "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                  f"({time.time()-t0:.1f}s)")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if early_stopping_patience is not None and patience_ctr >= early_stopping_patience:
            if verbose:
                print(f"{log_prefix}early stopping at epoch {ep+1} (best val_acc={best_val_acc:.4f})")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_val_acc


@torch.no_grad()
def get_predictions(model, loader, device="cpu"):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        all_preds.append(out.argmax(1).cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def count_params(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2))


def load_json(path):
    return json.loads(Path(path).read_text())

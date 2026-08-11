"""
Shared utilities for Task 26 (Transformers & Attention -- Applied Part C):
AG News loading/subsetting and evaluation helpers.

The train/val/test subsetting logic (seed, per-class sizes, class-balanced
selection algorithm) is deliberately an exact copy of Task 25's common.py --
same AG News dataset revision, same seed=42, same algorithm, so the
resulting train/val/test *examples* are identical between the two tasks.
This is what makes the Part C LSTM-vs-Transformer comparison a genuine
apples-to-apples one rather than merely "a similar dataset."

Seed 42 throughout, CPU only (no CUDA device available in this environment).
"""
import json
import random
from pathlib import Path

import numpy as np
import torch

SEED = 42
ROOT = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    d.mkdir(exist_ok=True)

AG_NEWS_CLASSES = ["World", "Sports", "Business", "Sci/Tech"]

# Identical to Task 25's common.py -- see module docstring.
N_TRAIN = 12000
N_VAL = 2000
N_TEST = 2000


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_raw_ag_news():
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news")
    return ds["train"], ds["test"]


def _class_balanced_indices(labels, n_per_class, n_classes, seed=SEED):
    rng = np.random.RandomState(seed)
    labels = np.array(labels)
    chosen = []
    for c in range(n_classes):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        chosen.append(idx[:n_per_class])
    idx = np.concatenate(chosen)
    rng.shuffle(idx)
    return idx.tolist()


def build_split_indices():
    """Byte-for-byte the same algorithm as Task 25's build_split_indices --
    reproduces the identical train/val/test example indices given the same
    AG News dataset revision, so results are directly comparable."""
    cache = RESULTS_DIR / "split_indices.json"
    if cache.exists():
        return json.loads(cache.read_text())

    train_full, test_full = get_raw_ag_news()
    train_labels = np.array(train_full["label"])
    n_classes = len(AG_NEWS_CLASSES)

    train_idx, val_idx = [], []
    rng = np.random.RandomState(SEED)
    n_train_per_class = N_TRAIN // n_classes
    n_val_per_class = N_VAL // n_classes
    for c in range(n_classes):
        idx = np.where(train_labels == c)[0]
        rng.shuffle(idx)
        train_idx.extend(idx[:n_train_per_class].tolist())
        val_idx.extend(idx[n_train_per_class:n_train_per_class + n_val_per_class].tolist())

    test_idx = _class_balanced_indices(test_full["label"], N_TEST // n_classes, n_classes, seed=SEED)

    out = dict(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    cache.write_text(json.dumps(out))
    return out


def get_text_splits():
    """Returns (train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)."""
    train_full, test_full = get_raw_ag_news()
    idx = build_split_indices()
    train_texts = [train_full[i]["text"] for i in idx["train_idx"]]
    train_labels = [train_full[i]["label"] for i in idx["train_idx"]]
    val_texts = [train_full[i]["text"] for i in idx["val_idx"]]
    val_labels = [train_full[i]["label"] for i in idx["val_idx"]]
    test_texts = [test_full[i]["text"] for i in idx["test_idx"]]
    test_labels = [test_full[i]["label"] for i in idx["test_idx"]]
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2))


def load_json(path):
    return json.loads(Path(path).read_text())

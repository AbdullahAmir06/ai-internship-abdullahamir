"""
Shared utilities for Task 25 (Sequence Modeling / RNN-LSTM / Text Classification):
tokenization, vocabulary, padding, GloVe loading, dataset loading/subsetting,
and the PyTorch train/eval loops reused by Part C's ablation. Kept in one
place so every part_* script trains/evaluates on identical data.

Seed 42 throughout, CPU only (no CUDA device available in this environment).
"""
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
ROOT = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    d.mkdir(exist_ok=True)

AG_NEWS_CLASSES = ["World", "Sports", "Business", "Sci/Tech"]

# Subset sizes: full AG News is 120,000 train / 7,600 test. Kept modest so a
# 6+ configuration ablation matrix (Part C) trains in minutes, not hours, on
# a CPU with no GPU -- the same tractability trade-off Task 23/24 made for
# their datasets, documented rather than silently applied.
N_TRAIN = 12000   # 3,000 per class
N_VAL = 2000       # 500 per class
N_TEST = 2000       # 500 per class
MAX_LEN = 50        # covers the 95th percentile (58 words) closely; AG News
                     # headlines+snippets rarely need more than this
VOCAB_SIZE = 20000
GLOVE_DIM = 100

PAD_IDX = 0
UNK_IDX = 1


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text):
    """Lowercase + simple word/number regex tokenizer -- no external
    tokenizer library, keeping the pipeline dependency-light and fully
    inspectable (the brief asks for a from-scratch preprocessing pipeline)."""
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------- dataset

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
    """Fixed, cached train/val/test index sets (class-balanced) so every
    script/notebook cell uses identical examples."""
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


def build_vocab(texts, vocab_size=VOCAB_SIZE):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    most_common = counter.most_common(vocab_size - 2)  # reserve <pad>, <unk>
    itos = ["<pad>", "<unk>"] + [w for w, _ in most_common]
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def encode(text, stoi, max_len=MAX_LEN):
    ids = [stoi.get(tok, UNK_IDX) for tok in tokenize(text)][:max_len]
    length = len(ids)
    ids = ids + [PAD_IDX] * (max_len - length)
    return ids, max(length, 1)


class AGNewsDataset(Dataset):
    def __init__(self, texts, labels, stoi, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        ids, length = encode(self.texts[i], self.stoi, self.max_len)
        return torch.tensor(ids, dtype=torch.long), length, torch.tensor(self.labels[i], dtype=torch.long)


def get_datasets_and_vocab(vocab_size=VOCAB_SIZE, max_len=MAX_LEN):
    train_full, test_full = get_raw_ag_news()
    idx = build_split_indices()

    train_texts = [train_full[i]["text"] for i in idx["train_idx"]]
    train_labels = [train_full[i]["label"] for i in idx["train_idx"]]
    val_texts = [train_full[i]["text"] for i in idx["val_idx"]]
    val_labels = [train_full[i]["label"] for i in idx["val_idx"]]
    test_texts = [test_full[i]["text"] for i in idx["test_idx"]]
    test_labels = [test_full[i]["label"] for i in idx["test_idx"]]

    stoi, itos = build_vocab(train_texts, vocab_size)

    train_ds = AGNewsDataset(train_texts, train_labels, stoi, max_len)
    val_ds = AGNewsDataset(val_texts, val_labels, stoi, max_len)
    test_ds = AGNewsDataset(test_texts, test_labels, stoi, max_len)
    return train_ds, val_ds, test_ds, stoi, itos


def get_loaders(batch_size=64, vocab_size=VOCAB_SIZE, max_len=MAX_LEN, num_workers=0):
    train_ds, val_ds, test_ds, stoi, itos = get_datasets_and_vocab(vocab_size, max_len)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        stoi, itos,
    )


# ------------------------------------------------------------------- GloVe

def load_glove_matrix(stoi, dim=GLOVE_DIM):
    """Builds a (vocab_size, dim) embedding matrix from the downloaded
    GloVe 6B text file, aligned to `stoi`'s indices. OOV words get a small
    random vector (fixed seed) rather than zeros, so they remain
    distinguishable to the model instead of collapsing to one point."""
    glove_path = DATA_DIR / f"glove.6B.{dim}d.txt"
    if not glove_path.exists():
        raise FileNotFoundError(
            f"{glove_path} not found -- download glove.6B.{dim}d.txt.gz and gunzip it into data/"
        )
    vocab_size = len(stoi)
    rng = np.random.RandomState(SEED)
    matrix = rng.normal(scale=0.1, size=(vocab_size, dim)).astype(np.float32)
    matrix[PAD_IDX] = 0.0
    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in stoi:
                vec = np.asarray(parts[1:], dtype=np.float32)
                matrix[stoi[word]] = vec
                found += 1
    return matrix, found


# ------------------------------------------------------------- train / eval

def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x, lengths, y in loader:
            x, y = x.to(device), y.to(device)
            if is_train:
                optimizer.zero_grad()
            out = model(x, lengths)
            loss = criterion(out, y)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def train_model(model, train_loader, val_loader, epochs, optimizer, device="cpu",
                 verbose=True, early_stopping_patience=None, log_prefix=""):
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_state, patience_ctr = -1.0, None, 0
    for ep in range(epochs):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
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
    for x, lengths, y in loader:
        x = x.to(device)
        out = model(x, lengths)
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

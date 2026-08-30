"""
Shared data loading for the phishing-email capstone (v2). Mirrors model/common.py's
pattern from the movie-review version, adapted for a dataset that ships only a single
'train' split and needs its own stratified split, plus a raw-text length cap (see
PREPROCESSING_NOTES in train_baseline.py for why).
"""
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = {0: "safe", 1: "phishing"}
MAX_CHARS = 20_000  # truncation cap -- see PREPROCESSING_NOTES
RANDOM_SEED = 42


def _clean(texts, labels):
    out_texts, out_labels = [], []
    for t, lab in zip(texts, labels):
        if not t or not t.strip():
            continue
        # 533 rows (~2.9%) carry the literal placeholder text "empty" -- a
        # scraping artifact (likely an HTML-only email with no plain-text
        # part in the original source), not real content. Verified directly
        # by checking the value, not assumed from the word appearing in a
        # real email. Treated the same as a missing value.
        if t.strip().lower() == "empty":
            continue
        out_texts.append(t.strip()[:MAX_CHARS])
        out_labels.append(1 if lab == "Phishing Email" else 0)
    return out_texts, out_labels


def _split(texts, labels):
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=RANDOM_SEED
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=RANDOM_SEED
    )
    return (
        train_texts, train_labels,
        val_texts, val_labels,
        test_texts, test_labels,
    )


def get_splits():
    """Email channel."""
    ds = load_dataset("zefang-liu/phishing-email-dataset")["train"]
    texts, labels = _clean(list(ds["Email Text"]), list(ds["Email Type"]))
    return _split(texts, labels)


def get_sms_splits():
    """SMS channel -- the classic UCI SMS Spam Collection (5,574 messages,
    ham/spam, ~87/13 imbalance -- more severe than the email channel's
    ~61/39, handled the same way: class_weight="balanced" at training time,
    not resampling). Ships only a single 'train' split, same as the email
    dataset, so the same stratified 80/10/10 split is used."""
    ds = load_dataset("sms_spam")["train"]
    texts, labels = [], []
    for t, lab in zip(ds["sms"], ds["label"]):
        t = t.strip()
        if not t:
            continue
        texts.append(t)
        labels.append(int(lab))  # ClassLabel: 0=ham, 1=spam -- already 0=safe/1=phishing-shaped
    return _split(texts, labels)

"""
Shared paths and helpers for the capstone's data/model layer.
"""
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "model" / "artifacts"
RESULTS_DIR = ROOT / "model" / "results"
FIGURES_DIR = ROOT / "model" / "figures"
for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42


def get_splits():
    """Returns (train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)
    using rotten_tomatoes' own canonical split -- see Part A's justification for why this
    is preferable to an ad hoc re-split."""
    from datasets import load_dataset
    ds = load_dataset("rotten_tomatoes")
    # ds[split]["text"] returns a datasets.arrow_dataset.Column wrapper in
    # this library version, not a plain list -- list() coerces it
    # explicitly rather than relying on downstream code (e.g. the
    # tokenizer's isinstance(x, list) check) to happen to accept it, which
    # it silently doesn't for the un-sliced object.
    return (
        list(ds["train"]["text"]), list(ds["train"]["label"]),
        list(ds["validation"]["text"]), list(ds["validation"]["label"]),
        list(ds["test"]["text"]), list(ds["test"]["label"]),
    )

"""
Shared utilities for Task 27 (NLP Basics: Tokenization, Word2Vec/GloVe,
Hugging Face embeddings): paths, the small BPE demo corpus, and the AG News
loader used to train Word2Vec on a modest, realistic corpus.

Seed 42 throughout, CPU only.
"""
import json
import random
from pathlib import Path

import numpy as np

SEED = 42
ROOT = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    d.mkdir(exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------- Part A corpus
# Small, hand-written, deliberately repetitive-in-places corpus for the
# from-scratch BPE demo -- repetition is intentional (BPE's merge procedure
# is driven by pair *frequency*, so a corpus with no repeated substrings
# would make every merge step trivial/uninformative).
BPE_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox runs past the lazy dog",
    "a quick fox and a lazy dog become quick friends",
    "the dog barks at the quick fox in the low light",
    "lower and lower prices are newer and newer news",
    "the newest low price is lower than the old price",
]


# ---------------------------------------------------------------- Part B corpus
def get_word2vec_corpus(n_docs=6000):
    """AG News headlines+descriptions -- a modest, realistic corpus (not a
    toy) for training a real (if small-corpus-limited) Word2Vec model.
    Reuses the same dataset/loading approach as Tasks 25/26 for consistency
    within this internship, though the exact train/val/test split logic
    from those tasks isn't needed here (Part B just needs "a modest text
    corpus", not a held-out evaluation split)."""
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news", split=f"train[:{n_docs}]")
    return list(ds["text"])


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2))


def load_json(path):
    return json.loads(Path(path).read_text())

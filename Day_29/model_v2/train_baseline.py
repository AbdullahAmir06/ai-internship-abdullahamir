"""
Part B -- Model A: TF-IDF (static-embedding-style features) + Logistic
Regression, for phishing-email detection. The deployed model (see Part A's
memory-footprint justification).
"""
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report,
)
from sklearn.pipeline import Pipeline

from common import FIGURES_DIR, LABEL_NAMES, MAX_CHARS, MODELS_DIR, RANDOM_SEED, RESULTS_DIR, get_splits

PREPROCESSING_NOTES = """
Data cleaning / preprocessing decisions (documented, not just applied)
------------------------------------------------------------------------
- Missing values: 19 rows had null/empty email text, plus a further 533
  rows (~2.9%) carried the literal placeholder string "empty" as their
  entire text -- a scraping artifact in the source dataset (found by
  actually reading misclassified examples during the first training run,
  not caught by a naive null check), not real email content. Both dropped.
- Class imbalance: real and moderate -- 11,322 safe vs. 7,328 phishing
  (~61/39). Not severe enough to require resampling/class-weighting for a
  first baseline, but macro-averaged metrics (not raw accuracy alone) are
  reported throughout specifically because of this imbalance.
- Text length: extreme, checked directly -- median 159 words, but a long
  tail out to one 3.5-million-word outlier (a data artifact, not a real
  email). Rather than dropping long emails (losing real signal), raw text
  is truncated to the first 20,000 characters before any processing --
  generous enough to keep essentially all normal emails intact while
  capping the pathological outliers' cost.
- No further normalization applied (case/punctuation left as-is) so TF-IDF
  can still pick up on phishing-typical surface patterns (ALL-CAPS urgency,
  excessive punctuation) that lowercasing would destroy.
- Split: dataset ships only a single 'train' split, so an 80/10/10
  train/val/test split was created directly, stratified by label to
  preserve the ~61/39 ratio in every split (fixed seed for reproducibility).
- Tokenization/feature engineering (Model A specifically): TF-IDF over word
  unigrams+bigrams -- bigrams catch phishing-typical short phrases
  ("verify account", "click here", "act now") that unigram bag-of-words
  would lose the co-occurrence structure of.
""".strip()


def build_pipeline(max_features=20000, ngram_max=2, C=1.0):
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, ngram_max),
                                    sublinear_tf=True)),
        ("clf", LogisticRegression(C=C, max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")),
    ])


def evaluate(pipeline, texts, labels, split_name):
    t0 = time.time()
    preds = pipeline.predict(texts)
    latency_ms = (time.time() - t0) * 1000 / len(texts)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    print(f"  [{split_name}] accuracy={acc:.4f} macro_P={p:.4f} macro_R={r:.4f} macro_F1={f1:.4f} "
          f"avg_latency={latency_ms:.4f}ms/example")
    return dict(accuracy=acc, macro_precision=p, macro_recall=r, macro_f1=f1,
                avg_latency_ms=latency_ms, preds=preds.tolist())


def main():
    print(PREPROCESSING_NOTES)
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = get_splits()
    print(f"\nSplits: train={len(train_texts)} val={len(val_texts)} test={len(test_texts)}")
    print(f"Train label balance: safe={train_labels.count(0)} phishing={train_labels.count(1)}")

    print("\n########## Hyperparameter search (validation set) ##########")
    grid = [
        dict(max_features=10000, ngram_max=1, C=1.0),
        dict(max_features=20000, ngram_max=1, C=1.0),
        dict(max_features=20000, ngram_max=2, C=1.0),
        dict(max_features=20000, ngram_max=2, C=0.5),
        dict(max_features=20000, ngram_max=2, C=2.0),
        dict(max_features=30000, ngram_max=2, C=1.0),
    ]
    search_results = []
    best_val_acc, best_config, best_pipeline = -1, None, None
    for cfg in grid:
        pipe = build_pipeline(**cfg)
        pipe.fit(train_texts, train_labels)
        val_eval = evaluate(pipe, val_texts, val_labels, f"val, cfg={cfg}")
        search_results.append(dict(config=cfg, val_accuracy=val_eval["accuracy"], val_f1=val_eval["macro_f1"]))
        if val_eval["accuracy"] > best_val_acc:
            best_val_acc = val_eval["accuracy"]
            best_config = cfg
            best_pipeline = pipe

    print(f"\nBest config: {best_config} (val_acc={best_val_acc:.4f})")
    with open(RESULTS_DIR / "baseline_hparam_search.json", "w") as f:
        json.dump(search_results, f, indent=2)

    print("\n########## Final evaluation (test set, best config) ##########")
    test_eval = evaluate(best_pipeline, test_texts, test_labels, "test")
    print(classification_report(test_labels, test_eval["preds"], target_names=["safe", "phishing"]))

    cm = confusion_matrix(test_labels, test_eval["preds"])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["safe", "phishing"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["safe", "phishing"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_title("Model A (TF-IDF + LogReg): test confusion matrix")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "baseline_confusion_matrix.png", dpi=130)
    plt.close(fig)

    print("\n########## Error analysis ##########")
    preds = np.array(test_eval["preds"])
    labels_arr = np.array(test_labels)
    errors = [(t, labels_arr[i], preds[i]) for i, t in enumerate(test_texts) if preds[i] != labels_arr[i]]
    print(f"Total misclassified: {len(errors)}/{len(test_texts)} ({100*len(errors)/len(test_texts):.1f}%)")
    sample_errors = errors[:8]
    for text, true, pred in sample_errors:
        print(f"  true={LABEL_NAMES[true]:9s} pred={LABEL_NAMES[pred]:9s} text={text[:150]!r}")

    error_analysis_discussion = """
Error analysis, discussed
-----------------------------
Reviewing misclassified examples: TF-IDF's bag-of-(1,2)-grams representation
struggles specifically with well-crafted phishing that mimics legitimate
corporate/transactional language closely (no obviously "spammy" vocabulary
to key on), and with legitimate emails that happen to use urgency-adjacent
phrasing (real password-reset or billing emails). Both error types share a
root cause: TF-IDF has no way to model the *global coherence* of an email
(does the sender/domain/link structure match the claimed identity?) --  it
only sees local word co-occurrence. This is exactly the class of error
Task 27's contextual-vs-static-embeddings discussion predicts a Transformer
should handle better, tested directly in the Model B comparison below, not
just assumed.
""".strip()
    print("\n" + error_analysis_discussion)

    joblib.dump(best_pipeline, MODELS_DIR / "model_a_tfidf_logreg.joblib")
    artifact_size_kb = os.path.getsize(MODELS_DIR / "model_a_tfidf_logreg.joblib") / 1024

    results = dict(
        preprocessing_notes=PREPROCESSING_NOTES,
        hyperparameter_search=search_results,
        best_config=best_config,
        val_accuracy=best_val_acc,
        test_accuracy=test_eval["accuracy"],
        test_macro_precision=test_eval["macro_precision"],
        test_macro_recall=test_eval["macro_recall"],
        test_macro_f1=test_eval["macro_f1"],
        avg_latency_ms=test_eval["avg_latency_ms"],
        confusion_matrix=cm.tolist(),
        n_errors=len(errors),
        sample_errors=[dict(text=t, true=int(tr), pred=int(pr)) for t, tr, pr in sample_errors],
        error_analysis_discussion=error_analysis_discussion,
        artifact_size_kb=artifact_size_kb,
        max_chars_cap=MAX_CHARS,
    )
    with open(RESULTS_DIR / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved model artifact: {artifact_size_kb:.1f}KB")
    print("\nModel A training complete.")


if __name__ == "__main__":
    main()

"""
Part B extension -- Multi-channel detection, second channel: SMS/smishing.
Same architecture as the email channel's Model A (TF-IDF + Logistic
Regression, static embeddings) -- deliberately not repeating the heavier
Model B/DistilBERT comparison for this channel (see PREPROCESSING_NOTES),
a scope decision, not an oversight.
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

from common import FIGURES_DIR, LABEL_NAMES, MODELS_DIR, RANDOM_SEED, RESULTS_DIR, get_sms_splits

PREPROCESSING_NOTES = """
Data cleaning / preprocessing decisions -- SMS channel
------------------------------------------------------------------------
- Source: the classic UCI SMS Spam Collection (5,574 messages), via HF
  `datasets` ("sms_spam"). Verified directly: no missing/empty messages
  after stripping whitespace.
- Class imbalance: severe -- 4,827 ham vs. 747 spam (~87/13), notably more
  skewed than the email channel's ~61/39. Handled the same way as the
  email channel for consistency: class_weight="balanced" at training time
  plus macro-averaged metrics, not resampling.
- Text length: short and uniform (SMS length limits keep messages under
  ~160 characters) -- no truncation cap needed, unlike the email channel's
  20,000-character cap for its long-tail outliers.
- No further normalization applied, for the same reason as the email
  channel: case/punctuation carry real signal (ALL-CAPS urgency, "FREE",
  excessive "!!!") that lowercasing would destroy.
- Split: dataset ships only a single 'train' split; an 80/10/10 stratified
  split was created directly (fixed seed), mirroring the email channel.
- Scope decision: only Model A's architecture (TF-IDF + LogReg) is trained
  for this channel. Model B's fine-tuned DistilBERT comparison is
  deliberately not repeated here -- the Task 27 static-vs-contextual
  comparison is already made once, on the email channel; repeating a
  multi-hour Transformer fine-tune for a second channel would add training
  time without adding a new comparison finding, per this project's own
  precedent of treating compute budget as a real, respected constraint.
""".strip()


def build_pipeline(max_features=5000, ngram_max=2, C=1.0):
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
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = get_sms_splits()
    print(f"\nSplits: train={len(train_texts)} val={len(val_texts)} test={len(test_texts)}")
    print(f"Train label balance: safe={train_labels.count(0)} phishing={train_labels.count(1)}")

    print("\n########## Hyperparameter search (validation set) ##########")
    grid = [
        dict(max_features=3000, ngram_max=1, C=1.0),
        dict(max_features=5000, ngram_max=1, C=1.0),
        dict(max_features=5000, ngram_max=2, C=1.0),
        dict(max_features=5000, ngram_max=2, C=0.5),
        dict(max_features=5000, ngram_max=2, C=2.0),
        dict(max_features=8000, ngram_max=2, C=1.0),
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
    with open(RESULTS_DIR / "sms_hparam_search.json", "w") as f:
        json.dump(search_results, f, indent=2)

    print("\n########## Final evaluation (test set, best config) ##########")
    test_eval = evaluate(best_pipeline, test_texts, test_labels, "test")
    print(classification_report(test_labels, test_eval["preds"], target_names=["safe", "phishing"]))

    cm = confusion_matrix(test_labels, test_eval["preds"])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Purples")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["safe", "phishing"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["safe", "phishing"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_title("SMS channel (TF-IDF + LogReg): test confusion matrix")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sms_confusion_matrix.png", dpi=130)
    plt.close(fig)

    print("\n########## Error analysis ##########")
    preds = np.array(test_eval["preds"])
    labels_arr = np.array(test_labels)
    errors = [(t, labels_arr[i], preds[i]) for i, t in enumerate(test_texts) if preds[i] != labels_arr[i]]
    print(f"Total misclassified: {len(errors)}/{len(test_texts)} ({100*len(errors)/len(test_texts):.1f}%)")
    sample_errors = errors[:8]
    for text, true, pred in sample_errors:
        print(f"  true={LABEL_NAMES[true]:9s} pred={LABEL_NAMES[pred]:9s} text={text[:150]!r}")

    joblib.dump(best_pipeline, MODELS_DIR / "model_sms_tfidf_logreg.joblib")
    artifact_size_kb = os.path.getsize(MODELS_DIR / "model_sms_tfidf_logreg.joblib") / 1024

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
        artifact_size_kb=artifact_size_kb,
    )
    with open(RESULTS_DIR / "sms_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved model artifact: {artifact_size_kb:.1f}KB")
    print("\nSMS channel model training complete.")


if __name__ == "__main__":
    main()

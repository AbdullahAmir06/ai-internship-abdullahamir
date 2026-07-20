"""
PKCERT AI & Software Development Internship, Task 13
Advanced Model Evaluation & Handling Imbalanced Data

ROC-AUC and learning curves for a fraud classifier on the ULB Credit Card
Fraud Detection dataset (492 frauds in 284,807 transactions, 0.173%), then
SMOTE and class weighting compared against the untouched baseline. Running
this file reproduces every number quoted in Report.tex and writes all
figures into figures/.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score,
                             confusion_matrix)

RANDOM_STATE = 42
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

# ----------------------------------------------------------------------
# Part A: Dataset selection and preparation
# ----------------------------------------------------------------------
df = pd.read_csv("creditcard.csv")
print("Shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))

counts = df["Class"].value_counts()
pct = df["Class"].value_counts(normalize=True) * 100
print(f"\nClass 0 (legitimate): {counts[0]:,} ({pct[0]:.3f}%)")
print(f"Class 1 (fraud)     : {counts[1]:,} ({pct[1]:.3f}%)")
print(f"Imbalance ratio: 1 fraud per {counts[0] / counts[1]:.0f} legitimate transactions")
print(f"\nTime span: {df['Time'].max() / 3600:.1f} hours")
print("\nAmount by class:")
print(df.groupby("Class")["Amount"].describe().to_string())

FEATURES = [c for c in df.columns if c not in ("Class",)]
X_raw = df[FEATURES].copy()
y = df["Class"].values

# V1-V28 are already PCA components from the source data (anonymised for
# confidentiality) and arrive pre-scaled; only Time and Amount are on their
# raw scales and need standardising before a distance/gradient-based model
# sees them.
corr = df[[c for c in df.columns if c.startswith("V")] + ["Class"]].corr()["Class"].drop("Class")
top_corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
print("\nFeatures most correlated with fraud:")
print(top_corr.head(8).to_string())

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
print(f"\nTrain: {X_train.shape[0]:,} rows ({y_train.sum()} fraud) | "
      f"Test: {X_test.shape[0]:,} rows ({y_test.sum()} fraud)")

scaler = StandardScaler()
X_train_s = X_train.copy()
X_test_s = X_test.copy()
X_train_s[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
X_test_s[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])
X_train_s = X_train_s.values
X_test_s = X_test_s.values

# ----------------------------------------------------------------------
# Part B: Advanced model evaluation (baseline model, no imbalance handling)
# ----------------------------------------------------------------------
print("\n--- Part B: baseline Logistic Regression (no imbalance handling) ---")
base_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
base_model.fit(X_train_s, y_train)
base_proba = base_model.predict_proba(X_test_s)[:, 1]
base_pred = base_model.predict(X_test_s)

base_roc_auc = roc_auc_score(y_test, base_proba)
base_ap = average_precision_score(y_test, base_proba)
base_acc = accuracy_score(y_test, base_pred)
base_prec = precision_score(y_test, base_pred, zero_division=0)
base_rec = recall_score(y_test, base_pred, zero_division=0)
base_f1 = f1_score(y_test, base_pred, zero_division=0)
base_cm = confusion_matrix(y_test, base_pred)
print(f"Accuracy {base_acc:.4f} | Precision {base_prec:.4f} | Recall {base_rec:.4f} | "
      f"F1 {base_f1:.4f} | ROC-AUC {base_roc_auc:.4f} | PR-AUC {base_ap:.4f}")
print("Confusion matrix [ [TN FP] [FN TP] ]:\n", base_cm)

# Sanity check: what a trivial "always legitimate" classifier scores
dummy_acc = (y_test == 0).mean()
print(f"\nA classifier that always predicts 'legitimate' scores accuracy {dummy_acc:.4f} "
      f"while catching 0 of {y_test.sum()} frauds -- accuracy alone is meaningless here.")

fpr, tpr, _ = roc_curve(y_test, base_proba)
prec_curve, rec_curve, _ = precision_recall_curve(y_test, base_proba)

print("\n--- Learning curve (baseline model, scoring = ROC-AUC) ---")
train_sizes, train_scores, val_scores = learning_curve(
    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    X_train_s, y_train,
    train_sizes=np.linspace(0.1, 1.0, 6),
    cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
    scoring="roc_auc", n_jobs=-1)
train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)
for n, tm, vm in zip(train_sizes, train_mean, val_mean):
    print(f"n={n:6d}  train ROC-AUC {tm:.4f}  val ROC-AUC {vm:.4f}  gap {tm - vm:.4f}")

# ----------------------------------------------------------------------
# Part C: Handling imbalanced data
# ----------------------------------------------------------------------
print("\n--- Part C1: class imbalance ---")
print(f"Training set: {(y_train == 0).sum():,} legitimate vs {(y_train == 1).sum():,} fraud "
      f"({(y_train == 1).mean() * 100:.3f}% positive)")

print("\n--- Part C2: SMOTE ---")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train_s, y_train)
print(f"After SMOTE: {(y_train_smote == 0).sum():,} legitimate vs "
      f"{(y_train_smote == 1).sum():,} fraud (balanced)")

smote_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
smote_model.fit(X_train_smote, y_train_smote)
smote_proba = smote_model.predict_proba(X_test_s)[:, 1]
smote_pred = smote_model.predict(X_test_s)

print("\n--- Part C3: class weighting ---")
cw_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
cw_model.fit(X_train_s, y_train)
cw_proba = cw_model.predict_proba(X_test_s)[:, 1]
cw_pred = cw_model.predict(X_test_s)


def score(name, y_true, pred, proba):
    return {
        "name": name,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "cm": confusion_matrix(y_true, pred),
    }


results = [
    score("Original (no handling)", y_test, base_pred, base_proba),
    score("SMOTE", y_test, smote_pred, smote_proba),
    score("Class weighting", y_test, cw_pred, cw_proba),
]

print("\n--- Comparison table ---")
print(f"{'Method':24s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'ROC-AUC':>8s} {'PR-AUC':>8s}")
for r in results:
    print(f"{r['name']:24s} {r['accuracy']:7.4f} {r['precision']:7.4f} {r['recall']:7.4f} "
          f"{r['f1']:7.4f} {r['roc_auc']:8.4f} {r['pr_auc']:8.4f}")
    print("  confusion matrix:", r["cm"].tolist())

# ======================================================================
# Figures
# ======================================================================

# 01 class imbalance + amount by class
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
bars = ax[0].bar(["Legitimate", "Fraud"], [counts[0], counts[1]],
                  color=["#4C72B0", "#C44E52"])
ax[0].set_yscale("log")
ax[0].set_ylabel("transactions (log scale)")
ax[0].set_title(f"Class imbalance: {counts[1]} fraud in {counts[0] + counts[1]:,} "
                 f"({pct[1]:.3f}%)")
for b, c in zip(bars, [counts[0], counts[1]]):
    ax[0].annotate(f"{c:,}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)
data0 = df.loc[df["Class"] == 0, "Amount"].clip(lower=0.01)
data1 = df.loc[df["Class"] == 1, "Amount"].clip(lower=0.01)
ax[1].boxplot([data0, data1], tick_labels=["Legitimate", "Fraud"], showfliers=False)
ax[1].set_yscale("log")
ax[1].set_ylabel("transaction amount, $ (log scale)")
ax[1].set_title("Transaction amount by class")
fig.tight_layout()
fig.savefig("figures/01_class_imbalance.png", bbox_inches="tight")
plt.close(fig)

# 02 ROC and PR curve, baseline model
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(fpr, tpr, color="#4C72B0", lw=1.8, label=f"ROC-AUC = {base_roc_auc:.4f}")
ax[0].plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="chance")
ax[0].set_xlabel("False Positive Rate"); ax[0].set_ylabel("True Positive Rate")
ax[0].set_title("ROC curve (baseline)")
ax[0].legend(fontsize=8, loc="lower right")
ax[1].plot(rec_curve, prec_curve, color="#C44E52", lw=1.8, label=f"PR-AUC = {base_ap:.4f}")
ax[1].axhline(dummy_acc <= 0 and 0 or (y_test.sum() / len(y_test)), color="grey", ls="--",
              lw=1, label="a random classifier")
ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
ax[1].set_title("Precision-Recall curve (baseline)")
ax[1].legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig("figures/02_roc_pr_baseline.png", bbox_inches="tight")
plt.close(fig)

# 03 learning curve
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(train_sizes, train_mean, marker="o", ms=4, color="#4C72B0", label="training score")
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                 alpha=0.15, color="#4C72B0")
ax.plot(train_sizes, val_mean, marker="o", ms=4, color="#DD8452", label="cross-validation score")
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#DD8452")
ax.set_xlabel("training examples"); ax.set_ylabel("ROC-AUC")
ax.set_title("Learning curve (baseline Logistic Regression)")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig("figures/03_learning_curve.png", bbox_inches="tight")
plt.close(fig)

# 04 SMOTE effect in the two most fraud-correlated features
f1n, f2n = top_corr.index[0], top_corr.index[1]
i1, i2 = FEATURES.index(f1n), FEATURES.index(f2n)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True, sharey=True)
ax[0].scatter(X_train_s[y_train == 0, i1], X_train_s[y_train == 0, i2],
              s=4, alpha=0.15, color="#4C72B0", label="legitimate", linewidths=0)
ax[0].scatter(X_train_s[y_train == 1, i1], X_train_s[y_train == 1, i2],
              s=10, alpha=0.8, color="#C44E52", label="fraud", linewidths=0)
ax[0].set_title(f"Before SMOTE ({(y_train == 1).sum()} fraud points)")
synth_mask = np.arange(len(y_train_smote)) >= len(y_train)
ax[1].scatter(X_train_smote[y_train_smote == 0, i1], X_train_smote[y_train_smote == 0, i2],
              s=4, alpha=0.15, color="#4C72B0", label="legitimate", linewidths=0)
ax[1].scatter(X_train_smote[(y_train_smote == 1) & (~synth_mask), i1],
              X_train_smote[(y_train_smote == 1) & (~synth_mask), i2],
              s=10, alpha=0.8, color="#C44E52", label="real fraud", linewidths=0)
ax[1].scatter(X_train_smote[(y_train_smote == 1) & synth_mask, i1],
              X_train_smote[(y_train_smote == 1) & synth_mask, i2],
              s=6, alpha=0.35, color="#55A868", label="synthetic fraud", linewidths=0)
ax[1].set_title(f"After SMOTE ({(y_train_smote == 1).sum():,} fraud points)")
for a in ax:
    a.set_xlabel(f1n); a.legend(fontsize=7, markerscale=2)
ax[0].set_ylabel(f2n)
fig.tight_layout()
fig.savefig("figures/04_smote_effect.png", bbox_inches="tight")
plt.close(fig)

# 05 ROC comparison across the three approaches
fig, ax = plt.subplots(figsize=(6, 5))
for r, proba, color in [(results[0], base_proba, "#4C72B0"),
                         (results[1], smote_proba, "#55A868"),
                         (results[2], cw_proba, "#DD8452")]:
    fp, tp, _ = roc_curve(y_test, proba)
    ax.plot(fp, tp, lw=1.8, color=color, label=f"{r['name']} (AUC {r['roc_auc']:.4f})")
ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curve: original vs SMOTE vs class weighting")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig("figures/05_roc_comparison.png", bbox_inches="tight")
plt.close(fig)

# 06 confusion matrices, all three
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for a, r in zip(ax, results):
    cm = r["cm"]
    a.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
    a.set_xticks([0, 1]); a.set_xticklabels(["Legit", "Fraud"])
    a.set_yticks([0, 1]); a.set_yticklabels(["Legit", "Fraud"])
    a.set_xlabel("Predicted"); a.set_ylabel("Actual")
    a.set_title(f"{r['name']}\nRecall {r['recall']:.3f}  Precision {r['precision']:.3f}",
                fontsize=9)
fig.tight_layout()
fig.savefig("figures/06_confusion_matrices.png", bbox_inches="tight")
plt.close(fig)

print("\nFigures written to figures/")

# ----------------------------------------------------------------------
print("\nSUMMARY")
print(f"Dataset: {df.shape[0]:,} transactions, {counts[1]} fraud ({pct[1]:.3f}%)")
print(f"Baseline: ROC-AUC {base_roc_auc:.4f} PR-AUC {base_ap:.4f} Recall {base_rec:.4f} "
      f"Precision {base_prec:.4f} F1 {base_f1:.4f}")
print(f"Learning curve gap at max n: {train_mean[-1] - val_mean[-1]:.4f}")
for r in results:
    print(f"{r['name']:24s} Recall {r['recall']:.4f} Precision {r['precision']:.4f} "
          f"F1 {r['f1']:.4f} ROC-AUC {r['roc_auc']:.4f} PR-AUC {r['pr_auc']:.4f}")
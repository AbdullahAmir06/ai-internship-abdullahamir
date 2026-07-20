"""
PKCERT AI & Software Development Internship, Task 10
Cross-Validation & Hyperparameter Tuning

Trains a Random Forest on the HTRU2 pulsar dataset, applies K-Fold
cross-validation, then tunes it with both GridSearchCV and RandomizedSearchCV
and compares the two. Running this file reproduces every number in Report.tex
and writes all figures into figures/.
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, GridSearchCV,
                                     RandomizedSearchCV)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, roc_auc_score)

RANDOM_STATE = 42
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

# ----------------------------------------------------------------------
# Part A: Dataset preparation
# ----------------------------------------------------------------------
df = pd.read_csv("pulsar.csv")
FEATURES = [c for c in df.columns if c != "Class"]
X, y = df[FEATURES], df["Class"]

print("Shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print("Class counts:", y.value_counts().to_dict())
print(f"Positive (pulsar) rate: {y.mean():.4f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
print(f"\nTrain: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
print(f"Train positive rate {y_train.mean():.4f} | Test positive rate {y_test.mean():.4f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def evaluate(name, model):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    res = dict(name=name,
               acc=accuracy_score(y_test, pred),
               pre=precision_score(y_test, pred),
               rec=recall_score(y_test, pred),
               f1=f1_score(y_test, pred),
               auc=roc_auc_score(y_test, proba),
               cm=confusion_matrix(y_test, pred))
    print(f"\n=== {name} ===")
    print(f"Accuracy {res['acc']:.4f} | Precision {res['pre']:.4f} | "
          f"Recall {res['rec']:.4f} | F1 {res['f1']:.4f} | ROC-AUC {res['auc']:.4f}")
    print(classification_report(y_test, pred, target_names=["Noise/RFI", "Pulsar"],
                                digits=3))
    return res


# ----------------------------------------------------------------------
# Part B: Baseline model + K-Fold cross-validation
# ----------------------------------------------------------------------
baseline = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
baseline.fit(X_train, y_train)
base_res = evaluate("Baseline Random Forest (defaults)", baseline)

print("\n--- 5-Fold Cross-Validation on the training set ---")
cv_acc = cross_val_score(baseline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
cv_f1 = cross_val_score(baseline, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
print("Accuracy per fold:", np.round(cv_acc, 4))
print(f"Accuracy: mean {cv_acc.mean():.4f}, std {cv_acc.std():.4f}")
print("F1 per fold:", np.round(cv_f1, 4))
print(f"F1: mean {cv_f1.mean():.4f}, std {cv_f1.std():.4f}")

# ----------------------------------------------------------------------
# Part C: Hyperparameter tuning
# We optimise F1, not accuracy: at a 9% positive rate a model that predicts
# "noise" for everything already scores 91% accuracy, so accuracy cannot
# steer the search toward catching pulsars.
# ----------------------------------------------------------------------
param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": [None, "balanced"],
}
n_combos = int(np.prod([len(v) for v in param_grid.values()]))
print(f"\n--- GridSearchCV over {n_combos} combinations ({n_combos * 5} fits) ---")
grid = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE),
                    param_grid, scoring="f1", cv=cv, n_jobs=-1, verbose=0)
t0 = time.perf_counter()
grid.fit(X_train, y_train)
grid_time = time.perf_counter() - t0
print(f"Grid search took {grid_time:.1f} s")
print("Best params:", grid.best_params_)
print(f"Best CV F1: {grid.best_score_:.4f}")
grid_res = evaluate("GridSearchCV-tuned Random Forest", grid.best_estimator_)

param_dist = {
    "n_estimators": randint(50, 400),
    "max_depth": [None, 5, 10, 15, 20, 30],
    "min_samples_leaf": randint(1, 8),
    "min_samples_split": randint(2, 12),
    "max_features": ["sqrt", "log2", None],
    "class_weight": [None, "balanced", "balanced_subsample"],
}
N_ITER = 30
print(f"\n--- RandomizedSearchCV, {N_ITER} samples ({N_ITER * 5} fits) ---")
rand = RandomizedSearchCV(RandomForestClassifier(random_state=RANDOM_STATE),
                          param_dist, n_iter=N_ITER, scoring="f1", cv=cv,
                          n_jobs=-1, random_state=RANDOM_STATE, verbose=0)
t0 = time.perf_counter()
rand.fit(X_train, y_train)
rand_time = time.perf_counter() - t0
print(f"Randomized search took {rand_time:.1f} s")
print("Best params:", rand.best_params_)
print(f"Best CV F1: {rand.best_score_:.4f}")
rand_res = evaluate("RandomizedSearchCV-tuned Random Forest", rand.best_estimator_)

# ======================================================================
# Figures
# ======================================================================
# 01 dataset overview: imbalance + two most separating features
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
y.value_counts().plot.bar(ax=ax[0], color=["#4C72B0", "#DD8452"], rot=0)
ax[0].set_xticklabels(["Noise/RFI", "Pulsar"])
ax[0].set_title(f"Class imbalance ({y.mean()*100:.1f}% pulsar)")
ax[0].set_ylabel("count")
for a, feat in zip(ax[1:], ["KurtosisIP", "MeanIP"]):
    a.hist([df.loc[y == 0, feat], df.loc[y == 1, feat]], bins=40, stacked=False,
           label=["Noise/RFI", "Pulsar"], color=["#4C72B0", "#DD8452"], density=True)
    a.set_title(f"{feat} by class")
    a.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/01_dataset_overview.png", bbox_inches="tight")
plt.close(fig)

# 02 K-Fold CV scores
fig, ax = plt.subplots(figsize=(6.4, 3.4))
folds = np.arange(1, 6)
w = 0.38
ax.bar(folds - w/2, cv_acc, w, label="Accuracy", color="#4C72B0")
ax.bar(folds + w/2, cv_f1, w, label="F1", color="#DD8452")
ax.axhline(cv_acc.mean(), color="#4C72B0", ls="--", lw=1)
ax.axhline(cv_f1.mean(), color="#DD8452", ls="--", lw=1)
ax.set_xticks(folds, [f"Fold {i}" for i in folds])
ax.set_ylim(0.7, 1.02)
ax.set_title("5-Fold CV: accuracy looks great, F1 tells the real story")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("figures/02_kfold_cv.png", bbox_inches="tight")
plt.close(fig)

# 03 confusion matrices side by side
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
for a, r, cmap in zip(axes, [base_res, grid_res, rand_res],
                      ["Blues", "Greens", "Oranges"]):
    ConfusionMatrixDisplay(r["cm"], display_labels=["Noise", "Pulsar"]).plot(
        ax=a, cmap=cmap, colorbar=False)
    a.set_title(f"{r['name'].split()[0]}\nF1 {r['f1']:.3f}", fontsize=9)
fig.tight_layout()
fig.savefig("figures/03_confusion_matrices.png", bbox_inches="tight")
plt.close(fig)

# 04 metric comparison
metrics = ["Accuracy", "Precision", "Recall", "F1"]
results = [base_res, grid_res, rand_res]
labels = ["Baseline", "GridSearchCV", "RandomizedSearchCV"]
colors = ["#4C72B0", "#55A868", "#DD8452"]
x = np.arange(len(metrics))
w = 0.26
fig, ax = plt.subplots(figsize=(7.2, 3.8))
for i, (r, lab, c) in enumerate(zip(results, labels, colors)):
    vals = [r["acc"], r["pre"], r["rec"], r["f1"]]
    b = ax.bar(x + (i - 1) * w, vals, w, label=lab, color=c)
    ax.bar_label(b, fmt="%.3f", padding=2, fontsize=7, rotation=90)
ax.set_xticks(x, metrics)
ax.set_ylim(0, 1.18)
ax.set_title("Baseline vs tuned models on the test set")
ax.legend(fontsize=8, ncol=3, loc="upper center")
fig.tight_layout()
fig.savefig("figures/04_model_comparison.png", bbox_inches="tight")
plt.close(fig)

# 05 grid vs random: cost against payoff
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.3))
srch = ["GridSearchCV", "RandomizedSearchCV"]
times = [grid_time, rand_time]
fits = [n_combos * 5, N_ITER * 5]
scores = [grid.best_score_, rand.best_score_]
b1 = ax[0].bar(srch, times, color=["#55A868", "#DD8452"], width=0.5)
ax[0].bar_label(b1, fmt="%.0f s", padding=2, fontsize=8)
ax[0].set_ylabel("seconds")
ax[0].set_title(f"Search cost ({fits[0]} vs {fits[1]} fits)")
b2 = ax[1].bar(srch, scores, color=["#55A868", "#DD8452"], width=0.5)
ax[1].bar_label(b2, fmt="%.4f", padding=2, fontsize=8)
ax[1].set_ylim(min(scores) - 0.02, max(scores) + 0.015)
ax[1].set_title("Best cross-validated F1")
fig.tight_layout()
fig.savefig("figures/05_search_comparison.png", bbox_inches="tight")
plt.close(fig)

print("\nFigures written to figures/")

# ----------------------------------------------------------------------
# Part D summary
# ----------------------------------------------------------------------
summary = pd.DataFrame({
    "Accuracy":  [r["acc"] for r in results],
    "Precision": [r["pre"] for r in results],
    "Recall":    [r["rec"] for r in results],
    "F1":        [r["f1"] for r in results],
    "ROC-AUC":   [r["auc"] for r in results],
}, index=labels).round(4)
print("\nSUMMARY")
print(summary.to_string())
print(f"\nBaseline CV: acc {cv_acc.mean():.4f} (std {cv_acc.std():.4f}) | "
      f"f1 {cv_f1.mean():.4f} (std {cv_f1.std():.4f})")
print(f"Grid   : {grid_time:.1f}s, {n_combos*5} fits, best CV F1 {grid.best_score_:.4f}")
print(f"Random : {rand_time:.1f}s, {N_ITER*5} fits, best CV F1 {rand.best_score_:.4f}")
print("Grid best params  :", grid.best_params_)
print("Random best params:", rand.best_params_)
for r in results:
    print(f"{r['name']} cm: {r['cm'].tolist()}")
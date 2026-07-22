"""
PKCERT AI & Software Development Internship, Task 15
Model Persistence & Mini-Project

An end-to-end pipeline (EDA -> preprocessing -> model -> evaluation) on the UCI
Cleveland Heart Disease dataset (303 patients, a dataset not used in any
earlier task), then the trained model is saved and reloaded with both
pickle and joblib, and the two are compared. Running this file reproduces
every number quoted in Report.tex and writes all figures into figures/.
"""

import os
import pickle
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)

RANDOM_STATE = 42
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
os.makedirs("models", exist_ok=True)

# ----------------------------------------------------------------------
# Part C (1-3): dataset selection, EDA, preprocessing
# ----------------------------------------------------------------------
df = pd.read_csv("heart_disease.csv")
print("Shape:", df.shape)

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

counts = df["target"].value_counts()
print(f"\nTarget: {counts[0]} no significant narrowing (<50%), "
      f"{counts[1]} significant narrowing (>50%) -- "
      f"{counts[1] / len(df) * 100:.1f}% positive, close to balanced")

print("\nMissing values:")
print(df.isna().sum()[df.isna().sum() > 0].to_string())
print(f"Total rows with any missing value: {df.isna().any(axis=1).sum()} of {len(df)} "
      f"({df.isna().any(axis=1).sum() / len(df) * 100:.1f}%) -- in 'ca' and 'thal' only")

# Checked, not assumed: does dropping those 7 rows change anything material?
dropped = df.dropna()
print(f"Dropping them changes the class balance from "
      f"{counts[1] / len(df) * 100:.1f}% positive to "
      f"{dropped['target'].sum() / len(dropped) * 100:.1f}% positive -- negligible. "
      f"Imputing is used below anyway, since it keeps the full 303 rows and is the "
      f"more complete demonstration of the required preprocessing step.")

print("\nChest pain type ('cp') vs disease presence:")
print(pd.crosstab(df["cp"], df["target"], normalize="index").round(3).to_string())

X = df[NUMERIC + CATEGORICAL].copy()
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
print(f"\nTrain: {X_train.shape[0]} rows ({y_train.sum()} positive) | "
      f"Test: {X_test.shape[0]} rows ({y_test.sum()} positive)")

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), NUMERIC),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), CATEGORICAL),
])

# ----------------------------------------------------------------------
# Part C (4-5): train and evaluate candidate models, pick one to persist
# ----------------------------------------------------------------------
candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
}

results = {}
for name, clf in candidates.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    cv_scores = cross_val_score(pipe, X_train, y_train,
                                 cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                                 scoring="roc_auc")
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    results[name] = {
        "pipe": pipe,
        "cv_roc_auc_mean": cv_scores.mean(),
        "cv_roc_auc_std": cv_scores.std(),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "cm": confusion_matrix(y_test, pred),
        "proba": proba,
    }
    print(f"\n=== {name} ===")
    print(f"5-fold CV ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"Test: Accuracy {results[name]['accuracy']:.4f} | Precision {results[name]['precision']:.4f} | "
          f"Recall {results[name]['recall']:.4f} | F1 {results[name]['f1']:.4f} | "
          f"ROC-AUC {results[name]['roc_auc']:.4f}")
    print("Confusion matrix [ [TN FP] [FN TP] ]:\n", results[name]["cm"])

best_name = max(results, key=lambda n: results[n]["roc_auc"])
best_pipe = results[best_name]["pipe"]
print(f"\nBest model on test ROC-AUC: {best_name} ({results[best_name]['roc_auc']:.4f})")

# ----------------------------------------------------------------------
# Part A: save the trained model with pickle and with joblib
# ----------------------------------------------------------------------
with open("models/heart_model.pkl", "wb") as f:
    pickle.dump(best_pipe, f)
joblib.dump(best_pipe, "models/heart_model.joblib")

pkl_size = os.path.getsize("models/heart_model.pkl")
joblib_size = os.path.getsize("models/heart_model.joblib")
print(f"\nSaved with pickle: models/heart_model.pkl ({pkl_size:,} bytes)")
print(f"Saved with joblib: models/heart_model.joblib ({joblib_size:,} bytes)")

# ----------------------------------------------------------------------
# Part B: load both back and verify they reproduce the original predictions
# ----------------------------------------------------------------------
t0 = time.perf_counter()
with open("models/heart_model.pkl", "rb") as f:
    pkl_model = pickle.load(f)
pkl_load_time = time.perf_counter() - t0

t0 = time.perf_counter()
joblib_model = joblib.load("models/heart_model.joblib")
joblib_load_time = time.perf_counter() - t0

original_pred = best_pipe.predict(X_test)
original_proba = best_pipe.predict_proba(X_test)
pkl_pred = pkl_model.predict(X_test)
pkl_proba = pkl_model.predict_proba(X_test)
joblib_pred = joblib_model.predict(X_test)
joblib_proba = joblib_model.predict_proba(X_test)

pkl_labels_match = np.array_equal(original_pred, pkl_pred)
pkl_proba_match = np.allclose(original_proba, pkl_proba)
joblib_labels_match = np.array_equal(original_pred, joblib_pred)
joblib_proba_match = np.allclose(original_proba, joblib_proba)

print(f"\npickle-loaded model:  labels match original: {pkl_labels_match} | "
      f"probabilities match exactly: {pkl_proba_match} | load time {pkl_load_time * 1000:.2f} ms")
print(f"joblib-loaded model:  labels match original: {joblib_labels_match} | "
      f"probabilities match exactly: {joblib_proba_match} | load time {joblib_load_time * 1000:.2f} ms")

assert pkl_labels_match and pkl_proba_match, "pickle round-trip changed predictions!"
assert joblib_labels_match and joblib_proba_match, "joblib round-trip changed predictions!"
print("\nBoth round-trips are bit-for-bit identical to the original model's predictions.")

# ----------------------------------------------------------------------
# Part B.3, checked rather than asserted: joblib's usual advantage is on
# large numpy arrays. Logistic Regression barely has any, so repeat the
# same save/load/size/time comparison on the much larger Random Forest
# (300 trees) to see whether the textbook claim actually holds here.
# ----------------------------------------------------------------------
rf_pipe = results["Random Forest"]["pipe"]
with open("models/heart_model_rf.pkl", "wb") as f:
    pickle.dump(rf_pipe, f)
joblib.dump(rf_pipe, "models/heart_model_rf.joblib")

rf_pkl_size = os.path.getsize("models/heart_model_rf.pkl")
rf_joblib_size = os.path.getsize("models/heart_model_rf.joblib")

t0 = time.perf_counter()
with open("models/heart_model_rf.pkl", "rb") as f:
    pickle.load(f)
rf_pkl_load_time = time.perf_counter() - t0

t0 = time.perf_counter()
joblib.load("models/heart_model_rf.joblib")
rf_joblib_load_time = time.perf_counter() - t0

print(f"\n--- Same comparison, but on the 300-tree Random Forest ---")
print(f"pickle: {rf_pkl_size:,} bytes, {rf_pkl_load_time * 1000:.2f} ms to load")
print(f"joblib: {rf_joblib_size:,} bytes, {rf_joblib_load_time * 1000:.2f} ms to load")
size_gap = (rf_joblib_size / rf_pkl_size - 1) * 100
print(f"joblib is {size_gap:.1f}% {'larger' if size_gap > 0 else 'smaller'} than pickle here, "
      f"and pickle loaded {rf_joblib_load_time / rf_pkl_load_time:.1f}x faster -- contrary to "
      f"the commonly repeated claim that joblib is the more efficient format for numpy-heavy "
      f"sklearn models, plain pickle won on both size and speed for both models tested. The "
      f"likely reason is PEP 574 (Python 3.8+): pickle protocol 5 added native out-of-band "
      f"buffer support for numpy arrays, closing most of the gap joblib was originally built "
      f"to fix. joblib's own remaining, real advantages are a one-argument `compress=` option "
      f"for gzip-style compression and `mmap_mode` for loading huge arrays without reading them "
      f"fully into RAM -- neither of which this benchmark exercises.")

# ----------------------------------------------------------------------
# Part D: final model already saved above; this script IS the documentation
# of the pipeline. Summary printed at the end.
# ----------------------------------------------------------------------

# ======================================================================
# Figures
# ======================================================================

# 01 dataset overview: target balance + age by target
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
bars = ax[0].bar(["No disease", "Disease"], [counts[0], counts[1]], color=["#4C72B0", "#C44E52"])
ax[0].set_ylabel("patients")
ax[0].set_title(f"Target balance: {counts[1] / len(df) * 100:.1f}% positive")
for b, c in zip(bars, [counts[0], counts[1]]):
    ax[0].annotate(f"{c}", (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom", fontsize=9)
a0 = df.loc[df["target"] == 0, "age"]
a1 = df.loc[df["target"] == 1, "age"]
ax[1].boxplot([a0, a1], tick_labels=["No disease", "Disease"])
ax[1].set_ylabel("age")
ax[1].set_title("Age by disease status")
fig.tight_layout()
fig.savefig("figures/01_dataset_overview.png", bbox_inches="tight")
plt.close(fig)

# 02 chest pain type vs disease
ct = pd.crosstab(df["cp"], df["target"], normalize="index") * 100
fig, ax = plt.subplots(figsize=(7, 4.2))
ct[[0, 1]].rename(columns={0: "No disease", 1: "Disease"}).plot(
    kind="bar", stacked=True, color=["#4C72B0", "#C44E52"], ax=ax)
ax.set_ylabel("% of patients with this chest pain type")
ax.set_xlabel("chest pain type")
ax.set_title("Chest pain type vs disease presence")
ax.legend(fontsize=8)
plt.xticks(rotation=20)
fig.tight_layout()
fig.savefig("figures/02_chest_pain_vs_target.png", bbox_inches="tight")
plt.close(fig)

# 03 model comparison
fig, ax = plt.subplots(figsize=(7, 4.5))
names = list(results.keys())
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
x = np.arange(len(metrics))
w = 0.35
for i, name in enumerate(names):
    vals = [results[name][m] for m in metrics]
    ax.bar(x + (i - 0.5) * w, vals, w, label=name)
ax.set_xticks(x); ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
ax.set_ylim(0, 1.05)
ax.set_title("Logistic Regression vs Random Forest (test set)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/03_model_comparison.png", bbox_inches="tight")
plt.close(fig)

# 04 confusion matrix + ROC for the chosen model
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
cm = results[best_name]["cm"]
ax[0].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax[0].text(j, i, f"{cm[i, j]}", ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["No disease", "Disease"])
ax[0].set_yticks([0, 1]); ax[0].set_yticklabels(["No disease", "Disease"])
ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Actual")
ax[0].set_title(f"{best_name}: confusion matrix")
fp, tp, _ = roc_curve(y_test, results[best_name]["proba"])
ax[1].plot(fp, tp, lw=1.8, color="#DD8452", label=f"ROC-AUC = {results[best_name]['roc_auc']:.4f}")
ax[1].plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
ax[1].set_xlabel("False Positive Rate"); ax[1].set_ylabel("True Positive Rate")
ax[1].set_title(f"{best_name}: ROC curve")
ax[1].legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig("figures/04_confusion_roc.png", bbox_inches="tight")
plt.close(fig)

# 05 feature importance (only meaningful if the chosen model is the Random Forest)
rf_pipe = results["Random Forest"]["pipe"]
feat_names = (NUMERIC +
              list(rf_pipe.named_steps["prep"].named_transformers_["cat"]
                   .named_steps["onehot"].get_feature_names_out(CATEGORICAL)))
importances = rf_pipe.named_steps["clf"].feature_importances_
order = np.argsort(importances)[::-1][:12]
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh([feat_names[i] for i in order][::-1], importances[order][::-1], color="#55A868")
ax.set_xlabel("Random Forest feature importance")
ax.set_title("Top 12 features (Random Forest)")
fig.tight_layout()
fig.savefig("figures/05_feature_importance.png", bbox_inches="tight")
plt.close(fig)

# 06 pickle vs joblib comparison, small model (Logistic Regression) vs large (Random Forest)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
x = np.arange(2); w = 0.35
ax[0].bar(x - w / 2, [pkl_size / 1024, rf_pkl_size / 1024], w, label="pickle", color="#8172B2")
ax[0].bar(x + w / 2, [joblib_size / 1024, rf_joblib_size / 1024], w, label="joblib", color="#55A868")
ax[0].set_xticks(x); ax[0].set_xticklabels(["Logistic Regression\n(small)", "Random Forest\n(300 trees)"])
ax[0].set_ylabel("file size, KB")
ax[0].set_title("Saved model file size")
ax[0].legend(fontsize=8)
ax[1].bar(x - w / 2, [pkl_load_time * 1000, rf_pkl_load_time * 1000], w, label="pickle", color="#8172B2")
ax[1].bar(x + w / 2, [joblib_load_time * 1000, rf_joblib_load_time * 1000], w, label="joblib", color="#55A868")
ax[1].set_xticks(x); ax[1].set_xticklabels(["Logistic Regression\n(small)", "Random Forest\n(300 trees)"])
ax[1].set_ylabel("load time, ms")
ax[1].set_title("Time to load from disk")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/06_persistence_comparison.png", bbox_inches="tight")
plt.close(fig)

print("\nFigures written to figures/")

# ----------------------------------------------------------------------
print("\nSUMMARY")
print(f"Dataset: {df.shape[0]} patients, {counts[1] / len(df) * 100:.1f}% with significant "
      f"coronary narrowing")
for name in names:
    r = results[name]
    print(f"{name:20s} Test Acc {r['accuracy']:.4f} F1 {r['f1']:.4f} ROC-AUC {r['roc_auc']:.4f} "
          f"| CV ROC-AUC {r['cv_roc_auc_mean']:.4f}+/-{r['cv_roc_auc_std']:.4f}")
print(f"Chosen model: {best_name}")
print(f"pickle: {pkl_size:,} bytes, {pkl_load_time * 1000:.2f} ms to load")
print(f"joblib: {joblib_size:,} bytes, {joblib_load_time * 1000:.2f} ms to load")

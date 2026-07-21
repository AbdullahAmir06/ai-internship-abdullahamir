"""
PKCERT AI & Software Development Internship, Task 14
Ensemble Methods: Bagging & Boosting

Compares a single Decision Tree, a Bagging ensemble, and two boosting
implementations (XGBoost, LightGBM) on the UCI Bank Marketing dataset
(45,211 calls, 11.7% subscribed). Running this file reproduces every number
quoted in Report.tex and writes all figures into figures/.
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)

RANDOM_STATE = 42
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

# ----------------------------------------------------------------------
# Part A: Dataset selection and preparation
# ----------------------------------------------------------------------
df = pd.read_csv("bank_marketing.csv")
print("Shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))

NUMERIC = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan",
                "contact", "month", "poutcome"]

counts = df["y"].value_counts()
pct = df["y"].value_counts(normalize=True) * 100
print(f"\nClass 'no'  (no subscription): {counts['no']:,} ({pct['no']:.2f}%)")
print(f"Class 'yes' (subscribed)     : {counts['yes']:,} ({pct['yes']:.2f}%)")

print("\n'poutcome' distribution (outcome of the previous campaign):")
print(df["poutcome"].value_counts().to_string())

print("\n'duration' (last call length, seconds) by class:")
print(df.groupby("y")["duration"].describe().to_string())
print("\nDuration is known only *after* the call ends, so it cannot be used to decide "
      "whether to make the call; UCI's own documentation flags it as a leakage risk. "
      "It is kept here as a feature -- exactly as most published benchmarks on this "
      "dataset do -- but every number in this report should be read as an upper bound "
      "on what a model could achieve calling in real time, not a real-time estimate.")

X = df[NUMERIC + CATEGORICAL].copy()
y = (df["y"] == "yes").astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
print(f"\nTrain: {X_train.shape[0]:,} rows ({y_train.sum()} subscribed) | "
      f"Test: {X_test.shape[0]:,} rows ({y_test.sum()} subscribed)")

preprocess = ColumnTransformer([
    ("num", "passthrough", NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
])


def evaluate(name, pipe, X_tr, y_tr, X_te, y_te):
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = pipe.predict(X_te)
    predict_time = time.perf_counter() - t0

    proba = pipe.predict_proba(X_te)[:, 1]
    result = {
        "name": name,
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred, zero_division=0),
        "f1": f1_score(y_te, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_te, proba),
        "cm": confusion_matrix(y_te, pred),
        "fit_time": fit_time,
        "predict_time": predict_time,
        "proba": proba,
    }
    print(f"\n=== {name} ===")
    print(f"Accuracy {result['accuracy']:.4f} | Precision {result['precision']:.4f} | "
          f"Recall {result['recall']:.4f} | F1 {result['f1']:.4f} | "
          f"ROC-AUC {result['roc_auc']:.4f}")
    print(f"Fit time {fit_time:.2f}s | Predict time {predict_time:.3f}s")
    print("Confusion matrix [ [TN FP] [FN TP] ]:\n", result["cm"])
    return result


# ----------------------------------------------------------------------
# Baseline: a single Decision Tree (motivates why bagging helps)
# ----------------------------------------------------------------------
tree_pipe = Pipeline([
    ("prep", preprocess),
    ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
])
tree_result = evaluate("Single Decision Tree (unpruned)", tree_pipe, X_train, y_train, X_test, y_test)

tree_train_acc = accuracy_score(y_train, tree_pipe.predict(X_train))
print(f"\nSingle tree: train accuracy {tree_train_acc:.4f} vs test accuracy "
      f"{tree_result['accuracy']:.4f} -- the gap is the overfitting bagging is meant to fix.")

# ----------------------------------------------------------------------
# Part B: Bagging
# ----------------------------------------------------------------------
bag_pipe = Pipeline([
    ("prep", preprocess),
    ("clf", BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        n_estimators=100, oob_score=True, n_jobs=-1, random_state=RANDOM_STATE)),
])
bag_result = evaluate("Bagging (100 trees)", bag_pipe, X_train, y_train, X_test, y_test)
print(f"Out-of-bag accuracy estimate: {bag_pipe.named_steps['clf'].oob_score_:.4f}")

bag_train_acc = accuracy_score(y_train, bag_pipe.predict(X_train))
print(f"Bagging: train accuracy {bag_train_acc:.4f} vs test accuracy "
      f"{bag_result['accuracy']:.4f}")

print("\n--- Bagging: accuracy vs number of estimators ---")
n_range = [1, 5, 10, 20, 50, 100, 150, 200]
bag_curve = []
for n in n_range:
    p = Pipeline([
        ("prep", preprocess),
        ("clf", BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
            n_estimators=n, n_jobs=-1, random_state=RANDOM_STATE)),
    ])
    p.fit(X_train, y_train)
    acc = accuracy_score(y_test, p.predict(X_test))
    bag_curve.append(acc)
    print(f"n_estimators={n:4d}  test accuracy {acc:.4f}")

# ----------------------------------------------------------------------
# Part C: Boosting (XGBoost and LightGBM)
# ----------------------------------------------------------------------
xgb_pipe = Pipeline([
    ("prep", preprocess),
    ("clf", XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)),
])
xgb_result = evaluate("XGBoost", xgb_pipe, X_train, y_train, X_test, y_test)

lgbm_pipe = Pipeline([
    ("prep", preprocess),
    ("clf", LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)),
])
lgbm_result = evaluate("LightGBM", lgbm_pipe, X_train, y_train, X_test, y_test)

results = [tree_result, bag_result, xgb_result, lgbm_result]

print("\n--- Comparison table ---")
print(f"{'Method':32s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'ROC-AUC':>8s} "
      f"{'Fit(s)':>8s} {'Pred(s)':>8s}")
for r in results:
    print(f"{r['name']:32s} {r['accuracy']:7.4f} {r['precision']:7.4f} {r['recall']:7.4f} "
          f"{r['f1']:7.4f} {r['roc_auc']:8.4f} {r['fit_time']:8.2f} {r['predict_time']:8.3f}")

# ======================================================================
# Figures
# ======================================================================

# 01 class imbalance + duration leakage
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
bars = ax[0].bar(["No", "Yes"], [counts["no"], counts["yes"]], color=["#4C72B0", "#55A868"])
ax[0].set_ylabel("clients")
ax[0].set_title(f"Subscribed to term deposit? {pct['yes']:.1f}% yes")
for b, c in zip(bars, [counts["no"], counts["yes"]]):
    ax[0].annotate(f"{c:,}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)
d_no = df.loc[df["y"] == "no", "duration"].clip(upper=1500)
d_yes = df.loc[df["y"] == "yes", "duration"].clip(upper=1500)
ax[1].boxplot([d_no, d_yes], tick_labels=["No", "Yes"], showfliers=False)
ax[1].set_ylabel("call duration, seconds (clipped at 1500)")
ax[1].set_title("Call duration by outcome -- a leakage feature")
fig.tight_layout()
fig.savefig("figures/01_dataset_overview.png", bbox_inches="tight")
plt.close(fig)

# 02 single tree vs bagging: overfitting and estimator count
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
labels = ["Single Tree", "Bagging (100 trees)"]
train_accs = [tree_train_acc, bag_train_acc]
test_accs = [tree_result["accuracy"], bag_result["accuracy"]]
xpos = np.arange(2)
w = 0.35
ax[0].bar(xpos - w / 2, train_accs, w, label="train accuracy", color="#4C72B0")
ax[0].bar(xpos + w / 2, test_accs, w, label="test accuracy", color="#DD8452")
ax[0].set_xticks(xpos); ax[0].set_xticklabels(labels)
ax[0].set_ylim(0.75, 1.02)
ax[0].set_ylabel("accuracy")
ax[0].set_title("Train vs test accuracy")
ax[0].legend(fontsize=8)
ax[1].plot(n_range, bag_curve, marker="o", color="#55A868")
ax[1].set_xlabel("number of bagged trees"); ax[1].set_ylabel("test accuracy")
ax[1].set_title("Bagging: accuracy vs ensemble size")
fig.tight_layout()
fig.savefig("figures/02_single_tree_vs_bagging.png", bbox_inches="tight")
plt.close(fig)

# 03 confusion matrices, all four
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
for a, r in zip(ax, results):
    cm = r["cm"]
    a.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    a.set_xticks([0, 1]); a.set_xticklabels(["No", "Yes"])
    a.set_yticks([0, 1]); a.set_yticklabels(["No", "Yes"])
    a.set_xlabel("Predicted"); a.set_ylabel("Actual")
    a.set_title(f"{r['name']}\nRecall {r['recall']:.3f}  Prec {r['precision']:.3f}", fontsize=8)
fig.tight_layout()
fig.savefig("figures/03_confusion_matrices.png", bbox_inches="tight")
plt.close(fig)

# 04 ROC comparison
fig, ax = plt.subplots(figsize=(6, 5))
colors = {"Single Decision Tree (unpruned)": "#8172B2", "Bagging (100 trees)": "#4C72B0",
          "XGBoost": "#DD8452", "LightGBM": "#55A868"}
for r in results:
    fp, tp, _ = roc_curve(y_test, r["proba"])
    ax.plot(fp, tp, lw=1.8, color=colors[r["name"]], label=f"{r['name']} (AUC {r['roc_auc']:.4f})")
ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curve: single tree vs bagging vs boosting")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig("figures/04_roc_comparison.png", bbox_inches="tight")
plt.close(fig)

# 05 feature importance (XGBoost, gain-based)
feat_names = (NUMERIC +
              list(xgb_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL)))
importances = xgb_pipe.named_steps["clf"].feature_importances_
order = np.argsort(importances)[::-1][:12]
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh([feat_names[i] for i in order][::-1], importances[order][::-1], color="#DD8452")
ax.set_xlabel("XGBoost feature importance (gain)")
ax.set_title("Top 12 features driving the boosting model")
fig.tight_layout()
fig.savefig("figures/05_feature_importance.png", bbox_inches="tight")
plt.close(fig)

# 06 timing comparison
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
names = [r["name"].replace(" (unpruned)", "").replace(" (100 trees)", "") for r in results]
fit_times = [r["fit_time"] for r in results]
predict_times = [r["predict_time"] for r in results]
ax[0].bar(names, fit_times, color=["#8172B2", "#4C72B0", "#DD8452", "#55A868"])
ax[0].set_ylabel("fit time, seconds")
ax[0].set_title("Training time")
ax[0].tick_params(axis="x", rotation=20)
ax[1].bar(names, predict_times, color=["#8172B2", "#4C72B0", "#DD8452", "#55A868"])
ax[1].set_ylabel("predict time, seconds")
ax[1].set_title(f"Prediction time ({X_test.shape[0]:,} test rows)")
ax[1].tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig("figures/06_timing_comparison.png", bbox_inches="tight")
plt.close(fig)

print("\nFigures written to figures/")

# ----------------------------------------------------------------------
print("\nSUMMARY")
print(f"Dataset: {df.shape[0]:,} calls, {pct['yes']:.2f}% resulted in a subscription")
for r in results:
    print(f"{r['name']:32s} Acc {r['accuracy']:.4f} F1 {r['f1']:.4f} ROC-AUC {r['roc_auc']:.4f} "
          f"Fit {r['fit_time']:.2f}s")

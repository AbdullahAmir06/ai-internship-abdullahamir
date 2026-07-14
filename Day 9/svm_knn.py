"""
PKCERT AI & Software Development Internship, Task 09
Support Vector Machines (SVM) & k-Nearest Neighbors (kNN)

Builds, tunes and compares an SVM and a kNN classifier on the UCI Raisin
dataset (Kecimen vs Besni), and writes every figure used in the report into
figures/. Running this file reproduces all the numbers quoted in Report.tex.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

RANDOM_STATE = 42
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

# ----------------------------------------------------------------------
# Part A: Dataset selection and preparation
# ----------------------------------------------------------------------
df = pd.read_csv("raisin.csv")
print("Shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print(df["Class"].value_counts().to_dict())

FEATURES = ["Area", "MajorAxisLength", "MinorAxisLength", "Eccentricity",
            "ConvexArea", "Extent", "Perimeter"]
# Encode target: Besni = 1 (positive class), Kecimen = 0
y = (df["Class"] == "Besni").astype(int)
X = df[FEATURES].copy()

print("\nFeature scales (mean):")
print(X.mean().round(2).to_string())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
print(f"\nTrain: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")


def evaluate(name, model, Xtr, Xte):
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy {acc:.3f}  Precision {pre:.3f}  Recall {rec:.3f}  F1 {f1:.3f}")
    print(classification_report(y_test, pred, target_names=["Kecimen", "Besni"]))
    return dict(name=name, model=model, pred=pred, acc=acc, pre=pre,
                rec=rec, f1=f1, cm=cm)


# ----------------------------------------------------------------------
# Part B: Support Vector Machine (RBF kernel)
# ----------------------------------------------------------------------
svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)
svm_res = evaluate("SVM (RBF kernel)", svm, X_train_s, X_test_s)

# quick check: linear kernel for the report's kernel discussion
lin = SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)
lin.fit(X_train_s, y_train)
print("Linear-kernel SVM test accuracy:", round(accuracy_score(y_test, lin.predict(X_test_s)), 3))

# ----------------------------------------------------------------------
# Part C: k-Nearest Neighbors, with a sweep to choose K
# ----------------------------------------------------------------------
ks = range(1, 26)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
k_scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k),
                            X_train_s, y_train, cv=cv, scoring="accuracy").mean()
            for k in ks]
best_k = list(ks)[int(np.argmax(k_scores))]
print(f"\nBest K by 5-fold CV on train set: {best_k} (CV acc {max(k_scores):.3f})")

knn = KNeighborsClassifier(n_neighbors=best_k)
knn_res = evaluate(f"kNN (K={best_k})", knn, X_train_s, X_test_s)

# ----------------------------------------------------------------------
# Part D: Comparative analysis, 5-fold CV on the full pipeline
# ----------------------------------------------------------------------
def cv_full(model):
    Xs = StandardScaler().fit_transform(X)
    s = cross_val_score(model, Xs, y, cv=cv, scoring="accuracy")
    return s.mean(), s.std()

svm_cv = cv_full(SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE))
knn_cv = cv_full(KNeighborsClassifier(n_neighbors=best_k))
print(f"\nSVM  CV acc {svm_cv[0]:.3f} (std {svm_cv[1]:.3f})")
print(f"kNN  CV acc {knn_cv[0]:.3f} (std {knn_cv[1]:.3f})")

# ======================================================================
# Figures
# ======================================================================
# 01 class balance + feature scale spread
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
df["Class"].value_counts().plot.bar(ax=ax[0], color=["#4C72B0", "#DD8452"], rot=0)
ax[0].set_title("Class balance (450 / 450)")
ax[0].set_ylabel("count")
X.boxplot(ax=ax[1], rot=90)
ax[1].set_yscale("log")
ax[1].set_title("Raw feature scales (log axis)")
fig.tight_layout()
fig.savefig("figures/01_dataset_overview.png", bbox_inches="tight")
plt.close(fig)

# 02 SVM confusion matrix
fig, ax = plt.subplots(figsize=(3.8, 3.4))
ConfusionMatrixDisplay(svm_res["cm"], display_labels=["Kecimen", "Besni"]).plot(
    ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"SVM (RBF), acc {svm_res['acc']:.3f}")
fig.tight_layout()
fig.savefig("figures/02_confusion_svm.png", bbox_inches="tight")
plt.close(fig)

# 03 K sweep
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(list(ks), k_scores, marker="o", ms=4)
ax.axvline(best_k, color="crimson", ls="--", lw=1, label=f"best K = {best_k}")
ax.set_xlabel("K (number of neighbors)")
ax.set_ylabel("5-fold CV accuracy")
ax.set_title("Choosing K for kNN")
ax.legend()
fig.tight_layout()
fig.savefig("figures/03_knn_k_sweep.png", bbox_inches="tight")
plt.close(fig)

# 04 kNN confusion matrix
fig, ax = plt.subplots(figsize=(3.8, 3.4))
ConfusionMatrixDisplay(knn_res["cm"], display_labels=["Kecimen", "Besni"]).plot(
    ax=ax, cmap="Oranges", colorbar=False)
ax.set_title(f"kNN (K={best_k}), acc {knn_res['acc']:.3f}")
fig.tight_layout()
fig.savefig("figures/04_confusion_knn.png", bbox_inches="tight")
plt.close(fig)

# 05 metric comparison bar chart
metrics = ["Accuracy", "Precision", "Recall", "F1"]
svm_vals = [svm_res["acc"], svm_res["pre"], svm_res["rec"], svm_res["f1"]]
knn_vals = [knn_res["acc"], knn_res["pre"], knn_res["rec"], knn_res["f1"]]
x = np.arange(len(metrics))
w = 0.38
fig, ax = plt.subplots(figsize=(6.2, 3.6))
b1 = ax.bar(x - w/2, svm_vals, w, label="SVM (RBF)", color="#4C72B0")
b2 = ax.bar(x + w/2, knn_vals, w, label=f"kNN (K={best_k})", color="#DD8452")
ax.set_xticks(x, metrics)
ax.set_ylim(0, 1)
ax.set_title("SVM vs kNN on the test set")
ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=8)
ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=8)
ax.legend()
fig.tight_layout()
fig.savefig("figures/05_model_comparison.png", bbox_inches="tight")
plt.close(fig)

print("\nFigures written to figures/")

# summary line for the report
print("\nSUMMARY")
print(f"SVM : acc {svm_res['acc']:.3f} pre {svm_res['pre']:.3f} rec {svm_res['rec']:.3f} f1 {svm_res['f1']:.3f} cv {svm_cv[0]:.3f}+/-{svm_cv[1]:.3f}")
print(f"kNN : acc {knn_res['acc']:.3f} pre {knn_res['pre']:.3f} rec {knn_res['rec']:.3f} f1 {knn_res['f1']:.3f} cv {knn_cv[0]:.3f}+/-{knn_cv[1]:.3f}")
print("SVM cm:", svm_res["cm"].tolist())
print("kNN cm:", knn_res["cm"].tolist())
print("best_k:", best_k)

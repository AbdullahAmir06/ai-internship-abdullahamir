"""
PKCERT AI & Software Development Internship, Task 11
Comparative Analysis of Classification Models

Builds Logistic Regression, Random Forest and SVM on the same UCI student
dropout dataset, evaluates all three on identical data, and compares them.
Running this file reproduces every number in Report.tex and writes all the
figures into figures/.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

RANDOM_STATE = 42
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

# ----------------------------------------------------------------------
# Part A: Dataset selection and preparation
# ----------------------------------------------------------------------
df = pd.read_csv("students.csv")
print("Shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print("\nTarget distribution:")
print(df["Target"].value_counts().to_string())
print((df["Target"].value_counts(normalize=True) * 100).round(1).to_string())

# These columns are codes (identifiers of categories), not quantities. Treating
# them as numbers would tell a linear model that "Course 12 > Course 3", which is
# meaningless, so they get one-hot encoded instead.
CATEGORICAL = ["Marital status", "Application mode", "Course",
               "Previous qualification", "Nacionality",
               "Mother's qualification", "Father's qualification",
               "Mother's occupation", "Father's occupation"]
BINARY = ["Daytime/evening attendance", "Displaced",
          "Educational special needs", "Debtor", "Tuition fees up to date",
          "Gender", "Scholarship holder", "International"]

y = df["Target"]
X = df.drop(columns=["Target"])
numeric = [c for c in X.columns if c not in CATEGORICAL]
print(f"\n{len(numeric)} numeric/binary features, {len(CATEGORICAL)} categorical to encode")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# One shared preprocessor: scale the numbers, one-hot the codes. Every model is
# fed exactly the same prepared data, which is what makes the comparison fair.
preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
])
n_feat_out = preprocess.fit(X_train).transform(X_train.head(1)).shape[1]
print(f"After encoding: {X_train.shape[1]} -> {n_feat_out} model features")

LABELS = ["Dropout", "Enrolled", "Graduate"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def build_and_eval(name, clf):
    """Same preprocessing, same split, same metrics for every model."""
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    res = dict(
        name=name, pipe=pipe, pred=pred,
        acc=accuracy_score(y_test, pred),
        pre=precision_score(y_test, pred, average="macro"),
        rec=recall_score(y_test, pred, average="macro"),
        f1=f1_score(y_test, pred, average="macro"),
        f1w=f1_score(y_test, pred, average="weighted"),
        cm=confusion_matrix(y_test, pred, labels=LABELS),
        # per-class recall, to expose how each model handles "Enrolled"
        rec_per=recall_score(y_test, pred, average=None, labels=LABELS),
    )
    cvs = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    res["cv"], res["cv_std"] = cvs.mean(), cvs.std()
    print(f"\n=== {name} ===")
    print(f"Accuracy {res['acc']:.4f} | Macro Precision {res['pre']:.4f} | "
          f"Macro Recall {res['rec']:.4f} | Macro F1 {res['f1']:.4f}")
    print(f"5-fold CV macro F1: {res['cv']:.4f} (std {res['cv_std']:.4f})")
    print(classification_report(y_test, pred, digits=3))
    return res


# ----------------------------------------------------------------------
# Part B: Model development
# ----------------------------------------------------------------------
logreg = build_and_eval("Logistic Regression",
                        LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
rf = build_and_eval("Random Forest",
                    RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                           n_jobs=-1))
svm = build_and_eval("SVM (RBF kernel)",
                     SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE))

results = [logreg, rf, svm]

# ----------------------------------------------------------------------
# Part C: Evaluation & comparison
# ----------------------------------------------------------------------
summary = pd.DataFrame({
    "Accuracy": [r["acc"] for r in results],
    "Precision (macro)": [r["pre"] for r in results],
    "Recall (macro)": [r["rec"] for r in results],
    "F1 (macro)": [r["f1"] for r in results],
    "F1 (weighted)": [r["f1w"] for r in results],
    "CV F1 (std)": [f"{r['cv']:.3f} ({r['cv_std']:.3f})" for r in results],
}, index=[r["name"] for r in results])
print("\n\nCOMPARISON TABLE")
print(summary.round(4).to_string())

per_class = pd.DataFrame({r["name"]: r["rec_per"] for r in results}, index=LABELS)
print("\nPer-class RECALL (the 'Enrolled' column is the story):")
print(per_class.round(3).to_string())

# ======================================================================
# Figures
# ======================================================================
# 01 target distribution
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
counts = df["Target"].value_counts().reindex(LABELS)
ax[0].bar(LABELS, counts.values, color=["#C44E52", "#DD8452", "#55A868"])
ax[0].set_title("Target distribution (4,424 students)")
ax[0].set_ylabel("students")
for i, v in enumerate(counts.values):
    ax[0].text(i, v + 30, f"{v}\n{v/len(df)*100:.1f}%", ha="center", fontsize=8)
ax[0].set_ylim(0, 2600)
# admission grade by outcome, a feature with a real signal
data = [df.loc[df["Target"] == t, "Admission grade"] for t in LABELS]
ax[1].boxplot(data, tick_labels=LABELS)
ax[1].set_title("Admission grade by outcome")
ax[1].set_ylabel("grade")
fig.tight_layout()
fig.savefig("figures/01_dataset_overview.png", bbox_inches="tight")
plt.close(fig)

# 02 the decisive feature: 2nd semester approved units
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
col = "Curricular units 2nd sem (approved)"
for t, c in zip(LABELS, ["#C44E52", "#DD8452", "#55A868"]):
    ax[0].hist(df.loc[df["Target"] == t, col], bins=range(0, 14), alpha=0.6,
               label=t, color=c, density=True)
ax[0].set_title("2nd sem units approved")
ax[0].set_xlabel("units passed")
ax[0].legend(fontsize=8)
col2 = "Curricular units 2nd sem (grade)"
for t, c in zip(LABELS, ["#C44E52", "#DD8452", "#55A868"]):
    ax[1].hist(df.loc[df["Target"] == t, col2], bins=20, alpha=0.6, label=t,
               color=c, density=True)
ax[1].set_title("2nd sem average grade")
ax[1].set_xlabel("grade")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/02_key_features.png", bbox_inches="tight")
plt.close(fig)

# 03 confusion matrices for all three
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
for a, r, cmap in zip(axes, results, ["Blues", "Greens", "Oranges"]):
    ConfusionMatrixDisplay(r["cm"], display_labels=LABELS).plot(
        ax=a, cmap=cmap, colorbar=False, values_format="d")
    a.set_title(f"{r['name']}\nacc {r['acc']:.3f} | macro F1 {r['f1']:.3f}", fontsize=9)
    a.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig("figures/03_confusion_matrices.png", bbox_inches="tight")
plt.close(fig)

# 04 metric comparison
metrics = ["Accuracy", "Precision", "Recall", "F1"]
colors = ["#4C72B0", "#55A868", "#DD8452"]
x = np.arange(len(metrics))
w = 0.26
fig, ax = plt.subplots(figsize=(7.2, 3.8))
for i, (r, c) in enumerate(zip(results, colors)):
    vals = [r["acc"], r["pre"], r["rec"], r["f1"]]
    b = ax.bar(x + (i - 1) * w, vals, w, label=r["name"], color=c)
    ax.bar_label(b, fmt="%.3f", padding=2, fontsize=7, rotation=90)
ax.set_xticks(x, ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)"])
ax.set_ylim(0, 1.05)
ax.set_title("Three models, identical data and split")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig("figures/04_model_comparison.png", bbox_inches="tight")
plt.close(fig)

# 05 per-class recall + RF feature importances
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
xx = np.arange(len(LABELS))
for i, (r, c) in enumerate(zip(results, colors)):
    b = ax[0].bar(xx + (i - 1) * w, r["rec_per"], w, label=r["name"], color=c)
    ax[0].bar_label(b, fmt="%.2f", padding=2, fontsize=7)
ax[0].set_xticks(xx, LABELS)
ax[0].set_ylim(0, 1.15)
ax[0].set_title("Recall per class: everyone fails 'Enrolled'")
ax[0].legend(fontsize=7)

rf_model = rf["pipe"].named_steps["clf"]
feat_names = rf["pipe"].named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf_model.feature_importances_, index=feat_names).nlargest(10)
clean = [n.split("__", 1)[-1][:34] for n in imp.index][::-1]
ax[1].barh(clean, imp.values[::-1], color="#55A868")
ax[1].set_title("Random Forest: top 10 features")
ax[1].tick_params(axis="y", labelsize=7)
fig.tight_layout()
fig.savefig("figures/05_perclass_and_importance.png", bbox_inches="tight")
plt.close(fig)

print("\nFigures written to figures/")

print("\nSUMMARY")
for r in results:
    print(f"{r['name']:22s} acc {r['acc']:.4f} pre {r['pre']:.4f} rec {r['rec']:.4f} "
          f"f1 {r['f1']:.4f} f1w {r['f1w']:.4f} cv {r['cv']:.4f}+/-{r['cv_std']:.4f}")
    print(f"{'':22s} per-class recall {dict(zip(LABELS, r['rec_per'].round(3)))}")
    print(f"{'':22s} cm {r['cm'].tolist()}")
print("\nTop RF features:")
print(imp.round(4).to_string())

# Comparing Classification Models: Predicting Student Dropout

**PKCERT AI & Software Development Internship, Task 11**
Author: Abdullah Amir

A Jupyter notebook that builds **Logistic Regression**, **Random Forest** and an **SVM**
on one common dataset, trains and tests all three on identical data, and compares them
with a full set of classification metrics.

## Dataset

- **File:** `students.csv` (4,424 students, 37 columns)
- **Source:** Realinho, Vieira Martins, Machado & Baptista, UCI Machine Learning
  Repository ([dataset 697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success))
- **Task:** three-class classification, predict whether a student will **Dropout**
  (32.1%), stay **Enrolled** (17.9%) or **Graduate** (49.9%).
- **Features:** 36 covering demographics (age, gender, marital status), family
  background (parents' qualifications and occupations), the admission route (course,
  application mode, admission grade), and academic performance (units enrolled,
  evaluated, **passed** and average grade for each of the first two semesters), plus
  financial signals like scholarship, debtor and tuition status.
- **Why this one:** real administrative records rather than a tutorial set, and it is
  the rare dataset that exercises the *whole* preprocessing pipeline. Nine columns are
  category codes masquerading as integers, so they genuinely need one-hot encoding
  (36 columns become 236 features), and the numeric columns need scaling for two of the
  three models but not the third. It is also a genuine three-class problem with an
  ambiguous middle class, which turns out to be what separates the models.

## What the notebook covers

- **Part A — Dataset preparation:** the four feature groups, why some "numbers" are
  really category codes, one-hot encoding plus scaling in a shared `ColumnTransformer`,
  and a stratified 80/20 split.
- **Part B — Model development:** Logistic Regression, a 300-tree Random Forest and an
  RBF SVM, each in the same pipeline so all three see identical inputs.
- **Part C — Evaluation & comparison:** accuracy, macro precision/recall/F1, confusion
  matrices, per-class recall, 5-fold cross-validation, and the strengths and weaknesses
  of each model.
- **Part D — Recommendation & conclusion:** the best model, justified from the results.

## Key results

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | CV F1 (std) |
| --- | --- | --- | --- | --- | --- |
| **Logistic Regression** | 0.7672 | 0.7122 | 0.6847 | **0.6930** | **0.701 (0.012)** |
| Random Forest | 0.7672 | 0.7188 | 0.6572 | 0.6625 | 0.672 (0.006) |
| SVM (RBF kernel) | 0.7605 | 0.7086 | 0.6732 | 0.6839 | 0.690 (0.006) |

**Accuracy hides the whole story.** Logistic Regression and Random Forest tie at
*exactly* 0.7672. Macro F1 does not: 0.693 vs 0.663. The per-class recall shows why.

| Recall per class | Logistic Regression | Random Forest | SVM |
| --- | --- | --- | --- |
| Dropout | 0.750 | **0.768** | 0.715 |
| Enrolled | **0.390** | 0.252 | 0.377 |
| Graduate | 0.914 | **0.952** | 0.928 |

The Random Forest quietly gives up on the hard `Enrolled` minority (25.2% recall) and
banks the easy `Graduate` wins instead. Since `Graduate` is half the data, that costs it
nothing in accuracy while gutting its macro F1, exactly the failure macro averaging
exists to expose.

**Recommendation: Logistic Regression.** Best macro F1 (0.693), best cross-validated
score (0.701, ahead of the forest's 0.672 by about five times the fold spread), best on
the hard class, and by far the most interpretable, which matters when a university has
to justify flagging a student.

**The honest finding: the simplest model won.** The dominant signal here (pass your
units, you graduate) is close to monotonic, so the forest's and SVM's extra flexibility
found no real structure and the forest overfit instead. Caveats worth keeping: on
accuracy alone it is a three-way tie; the Logistic Regression vs SVM gap sits inside one
CV standard deviation; and if you only cared about catching likely dropouts early, the
Random Forest's 76.8% `Dropout` recall would win.

## How to run

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook model_comparison.ipynb   # then Run All
```

Or run the plain script, which regenerates the five figures in `figures/`:

```bash
python model_comparison.py
```

## Files

| File | Description |
| --- | --- |
| `students.csv` | The dataset |
| `model_comparison.ipynb` | The full notebook (Parts A to D), with outputs |
| `model_comparison.py` | The same analysis as a plain script |
| `figures/` | The five generated plots |
| `Report.pdf` / `Report.tex` | Short written report |
| `README.md` | This file |

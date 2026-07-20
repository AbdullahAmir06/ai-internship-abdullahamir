# Advanced Model Evaluation & Handling Imbalanced Data: Credit Card Fraud

**PKCERT AI & Software Development Internship, Task 13**
Author: Abdullah Amir

A Jupyter notebook that evaluates a fraud classifier with **ROC-AUC**, a **Precision-
Recall curve**, and a **learning curve**, then compares two ways of handling severe
class imbalance, **SMOTE** and **class weighting**, against the untouched baseline on
Accuracy, Precision, Recall, F1, ROC-AUC and the confusion matrix.

## Dataset

- **Source:** the ULB Credit Card Fraud Detection dataset (Dal Pozzolo, Caelen, Johnson
  & Bontempi, 2015), mirrored on [OpenML, dataset 1597](https://www.openml.org/d/1597).
  284,807 transactions by European cardholders over two days in September 2013.
- **Features:** `V1`-`V28` are principal components of the original transaction
  features, released already PCA-transformed for confidentiality. `Time` (seconds since
  the first transaction) and `Amount` are the only features on their original scale.
- **Target:** `Class`, 1 for fraud, 0 otherwise.
- **Why this one:** it is the textbook case for this task. Only **492 of 284,807
  transactions are fraud (0.173%)**, an imbalance ratio of about 578 to 1, which is
  exactly the regime where accuracy stops meaning anything and ROC-AUC, SMOTE and
  class weighting earn their keep.
- **Not included in this folder:** the CSV is ~150 MB, past what's sensible to commit.
  `model_evaluation.py` and the notebook both expect `creditcard.csv` in this directory;
  fetch it with:
  ```python
  from sklearn.datasets import fetch_openml
  df = fetch_openml(name="creditcard", version=1, as_frame=True).frame
  df["Class"] = df["Class"].astype(int)
  df.to_csv("creditcard.csv", index=False)
  ```

## What the notebook covers

- **Part A — Preparation:** what the features are and why only `Time`/`Amount` need
  scaling, a stratified 80/20 split so both folds keep the 0.173% fraud rate.
- **Part B — Advanced evaluation:** a baseline Logistic Regression, its ROC curve and
  ROC-AUC, why ROC-AUC alone flatters a model this imbalanced, and a learning curve
  that separates underfitting, overfitting and generalisation.
- **Part C — Handling imbalance:** the class counts, SMOTE (with a before/after scatter
  of the synthetic points), class weighting, and all three scored on the same untouched
  test set.
- **Part D — Analysis & conclusion:** what the numbers actually say and which method to
  use, and when.

## Key results

| Method | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Original (no handling) | 0.9992 | 0.8289 | 0.6429 | **0.7241** | 0.9559 | **0.7432** |
| SMOTE | 0.9742 | 0.0580 | 0.9184 | 0.1092 | 0.9699 | 0.7249 |
| Class weighting | 0.9755 | 0.0610 | **0.9184** | 0.1144 | **0.9722** | 0.7159 |

Three findings, in order of how much they matter:

1. **SMOTE and class weighting both roughly triple recall (0.643 to 0.918), but F1 gets
   substantially worse, not better** (0.724 to about 0.11). ROC-AUC alone would have
   called both a straightforward win; F1 says the opposite.
2. **PR-AUC, which does not depend on the classification threshold, is actually highest
   for the untouched baseline.** Neither technique improved the model's underlying
   ability to rank a fraud above a legitimate transaction; both mainly moved where the
   default 0.5 threshold falls on a ranking that was already about as good as it would
   get.
3. **Class weighting beats SMOTE outright**: identical recall (90 of 98 frauds caught)
   with fewer false positives (1,386 vs 1,461), no synthetic data, and no extra
   computation.

**Recommendation:** class weighting over SMOTE if a technique is going to be used at
all, but the real choice is a business one, between the untouched baseline's precision
(0.829, few false alarms) and class weighting's recall (0.918, catches far more fraud).
The better lever than resampling is tuning the baseline's own decision threshold, since
its ranking (PR-AUC 0.743) is the best of the three on offer.

**Honest caveat:** the test set has only 98 fraud cases, so every precision/recall
number here moves in ~1-percentage-point steps per transaction. The direction of every
finding is solid; the third decimal place should not be over-trusted.

## How to run

```bash
python3 -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install pandas numpy matplotlib scikit-learn imbalanced-learn jupyter
python -c "from sklearn.datasets import fetch_openml; df = fetch_openml(name='creditcard', version=1, as_frame=True).frame; df['Class'] = df['Class'].astype(int); df.to_csv('creditcard.csv', index=False)"
jupyter notebook model_evaluation.ipynb   # then Run All
```

Or run the plain script, which regenerates the six figures in `figures/`:

```bash
python model_evaluation.py
```

Heads up: the first run downloads and caches ~150 MB from OpenML; after that the full
script (train/test split, baseline, SMOTE, class weighting, learning curve, all six
figures) takes well under a minute.

## Files

| File | Description |
| --- | --- |
| `model_evaluation.ipynb` | The full notebook (Parts A to D), with outputs |
| `model_evaluation.py` | The same analysis as a plain script |
| `figures/` | The six generated plots |
| `Report.pdf` / `Report.tex` | Short written report |
| `README.md` | This file |

`creditcard.csv` is not included; see **Dataset** above for how to fetch it.
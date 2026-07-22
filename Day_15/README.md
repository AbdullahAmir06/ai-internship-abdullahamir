# Model Persistence & Mini-Project: Heart Disease Prediction

**PKCERT AI & Software Development Internship, Task 15**
Author: Abdullah Amir

A full EDA → preprocessing → model → evaluation pipeline on a fresh dataset, then the
trained model is saved and reloaded with both **pickle** and **joblib**, verified
bit-for-bit identical, and the two formats are compared for size and speed on both a
small model and a much larger one.

## Dataset

- **File:** `heart_disease.csv` (UCI Cleveland Heart Disease, 303 patients, 14 columns),
  mirrored on [OpenML, dataset 49](https://www.openml.org/d/49).
- **Features:** 6 numeric (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca`) and 7
  categorical (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `thal`) clinical
  measurements.
- **Target:** `target`, whether coronary angiography showed >50% narrowing (binarised
  from the original 5-level severity scale).
- **Why this one:** not used in any earlier task in this internship, has genuine
  missing values (`ca`, `thal`) to exercise real imputation, a close-to-balanced target,
  and a well-known, human-interpretable feature set.

## What the notebook covers

- **Part C (mini-project) — EDA, preprocessing, model, evaluation:** missing-value
  check, a chest-pain-type-vs-disease crosstab, a `ColumnTransformer` pipeline, and a
  cross-validated comparison between Logistic Regression and a 300-tree Random Forest.
- **Part A — Saving:** the chosen model is saved with `pickle` and with `joblib`.
- **Part B — Loading & verifying:** both are loaded back and checked against the
  original model's predictions, then the same save/load/size/time comparison is
  repeated on the larger Random Forest to test whether joblib's usual efficiency claim
  actually holds.
- **Part D — Documentation:** pipeline summary, key findings, and challenges faced.

## Key results

| Model | CV ROC-AUC | Test Accuracy | Test ROC-AUC |
| --- | --- | --- | --- |
| **Logistic Regression** | 0.9017 ± 0.031 | **0.8525** | **0.9275** |
| Random Forest (300 trees) | 0.9037 ± 0.029 | 0.8033 | 0.8837 |

| Format | Logistic Regression (size / load) | Random Forest (size / load) |
| --- | --- | --- |
| pickle | 3,754 B / 0.12 ms | 2,169,710 B / 4.58 ms |
| joblib | 5,930 B / 0.63 ms | 2,195,451 B / 23.21 ms |

Four findings, in order of how much they surprised me:

1. **Asymptomatic patients have the *highest* disease rate (73%)** of any chest pain
   category — the absence of the expected symptom outranks its presence as a predictor.
2. **The simpler model won.** Logistic Regression beat the Random Forest on every
   held-out metric despite the two being statistically tied in cross-validation — the
   bias-variance tradeoff, verified rather than assumed, on a 242-row training set.
3. **Both round-trips were bit-for-bit identical** to the original model's predictions
   — no precision lost to serialization, for either format.
4. **pickle beat joblib on both size and speed, for both models** — the opposite of
   the commonly repeated claim. The likely reason is PEP 574 (Python 3.8+), which gave
   pickle protocol 5 native support for numpy buffers, closing most of the gap joblib
   was built to fix in 2008. joblib's real remaining advantages are its one-line
   `compress=` option and `mmap_mode` for huge arrays, neither exercised here.

**Recommendation:** verify persistence claims against the current Python version rather
than reciting them — and never unpickle a model file from an untrusted source, since
both `pickle.load` and `joblib.load` execute arbitrary code by design.

## How to run

```bash
python3 -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install pandas numpy matplotlib scikit-learn joblib jupyter ipykernel
python -c "from sklearn.datasets import fetch_openml; d = fetch_openml(data_id=49, as_frame=True).frame; [d.__setitem__(c, d[c].astype(str)) for c in d.select_dtypes('category').columns]; d['target'] = (d['num'] == '>50_1').astype(int); d.drop(columns=['num']).to_csv('heart_disease.csv', index=False)"
jupyter notebook model_persistence.ipynb   # then Run All
```

Or run the plain script, which regenerates the six figures in `figures/` and the four
model files in `models/`:

```bash
python model_persistence.py
```

Full run takes a few seconds.

## Files

| File | Description |
| --- | --- |
| `heart_disease.csv` | The dataset |
| `model_persistence.ipynb` | The full notebook (Parts A to D), with outputs |
| `model_persistence.py` | The same analysis as a plain script |
| `figures/` | The six generated plots |
| `models/` | The four saved model files (pickle/joblib, small/large model) |
| `Report.pdf` / `Report.tex` | Short written report |
| `README.md` | This file |

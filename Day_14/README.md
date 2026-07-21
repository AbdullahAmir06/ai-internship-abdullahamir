# Ensemble Methods: Bagging & Boosting on Bank Telemarketing Calls

**PKCERT AI & Software Development Internship, Task 14**
Author: Abdullah Amir

A Jupyter notebook that compares a single Decision Tree, a **Bagging** ensemble, and two
**Boosting** implementations (**XGBoost**, **LightGBM**) on whether a bank telemarketing
call ends in a term-deposit subscription -- on performance, and on how long each one
takes to train.

## Dataset

- **File:** `bank_marketing.csv` (UCI Bank Marketing, Moro/Cortez/Rita 2014, 45,211 calls,
  17 columns), mirrored on [OpenML, dataset 1461](https://www.openml.org/d/1461).
- **Task:** binary classification. 7 numeric features (`age`, `balance`, `day`,
  `duration`, `campaign`, `pdays`, `previous`) and 9 categorical features (`job`,
  `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`).
  Target `y`: did the client subscribe to a term deposit?
- **Why this one:** it is a real business problem (which clients to call) with a genuine,
  moderate class imbalance (11.7% yes), a rich mix of categorical and numeric features
  that forces real preprocessing, and it is large enough (45K rows) that the
  computational-efficiency gap between bagging and boosting actually shows up in wall-
  clock seconds rather than being a purely theoretical claim.
- **A documented leakage caveat:** `duration` (the call length) is only known *after* the
  call ends, so a bank cannot use it to decide whether to place the call. UCI's own
  documentation flags this. It is kept in as a feature, the same choice almost every
  published benchmark on this dataset makes, so every score here is an upper bound, not
  a real-time estimate.

## What the notebook covers

- **Part A — Preparation:** what the features are, the `duration` leakage caveat, a
  stratified 80/20 split, and one-hot encoding via a `ColumnTransformer`.
- **Part B — Bagging:** a single unpruned tree as baseline, then 100 bagged copies of it,
  with an out-of-bag score and an accuracy-vs-ensemble-size curve.
- **Part C — Boosting:** XGBoost and LightGBM, both compared to Bagging on Accuracy,
  Precision, Recall, F1, ROC-AUC and confusion matrix, plus a feature-importance plot.
- **Part D — Analysis & conclusion:** performance and training/prediction time compared
  across all four models, and a recommendation.

## Key results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Fit time |
| --- | --- | --- | --- | --- | --- | --- |
| Single Decision Tree | 0.8746 | 0.4649 | 0.4754 | 0.4701 | 0.7015 | 0.29s |
| Bagging (100 trees) | 0.9055 | 0.6270 | 0.4735 | 0.5396 | 0.9253 | 6.85s |
| **XGBoost** | **0.9103** | **0.6631** | 0.4745 | **0.5532** | 0.9335 | **0.63s** |
| LightGBM | 0.9090 | 0.6520 | **0.4764** | 0.5505 | **0.9339** | **0.58s** |

Three findings, in order of how much they matter:

1. **Bagging's train accuracy is still 1.0**, identical to the single tree -- bagging
   does not stop individual trees from memorising their bootstrap sample. What it fixes
   is *test* accuracy (0.875 → 0.906) and ROC-AUC (0.70 → 0.93), by averaging out each
   tree's individual mistakes on unseen data.
2. **Boosting beats bagging on every metric while training over 10x faster** (0.6s vs
   6.85s). Bagging's 100 trees are grown to full depth; boosting's 300 trees are capped
   at depth 4 and correct sequentially, which turns out to need far less total
   computation to reach a better answer.
3. **`poutcome_success` (said yes to a previous campaign) outranks even the leakage
   feature `duration`** in feature importance — past behaviour is the strongest signal
   in the dataset, ahead of the call itself.

**Recommendation:** boosting (XGBoost or LightGBM, statistically indistinguishable here)
over bagging for this dataset — better accuracy, an order of magnitude less compute.
Bagging remains the right call when the base learner's variance is the whole problem and
training must be parallelised with no sequential dependency, which is not the situation
here.

## How to run

```bash
python3 -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install pandas numpy matplotlib scikit-learn xgboost lightgbm jupyter ipykernel
python -c "from sklearn.datasets import fetch_openml; d = fetch_openml(data_id=1461, as_frame=True).frame; d.columns = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y']; d['y'] = d['y'].astype(str).map({'1':'no','2':'yes'}); [d.__setitem__(c, d[c].astype(str)) for c in d.select_dtypes('category').columns]; d.to_csv('bank_marketing.csv', index=False)"
jupyter notebook ensemble_methods.ipynb   # then Run All
```

Or run the plain script, which regenerates the six figures in `figures/`:

```bash
python ensemble_methods.py
```

Full run (single tree, bagging with an 8-point ensemble-size sweep, XGBoost, LightGBM,
all six figures) takes under a minute.

## Files

| File | Description |
| --- | --- |
| `bank_marketing.csv` | The dataset |
| `ensemble_methods.ipynb` | The full notebook (Parts A to D), with outputs |
| `ensemble_methods.py` | The same analysis as a plain script |
| `figures/` | The six generated plots |
| `Report.pdf` / `Report.tex` | Short written report |
| `README.md` | This file |

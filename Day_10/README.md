# Cross-Validation & Hyperparameter Tuning: Finding Pulsars in Radio Noise

**PKCERT AI & Software Development Internship, Task 10**
Author: Abdullah Amir

A Jupyter notebook that trains a Random Forest on the HTRU2 pulsar dataset, applies
K-Fold cross-validation, then tunes it with both GridSearchCV and RandomizedSearchCV
and compares the two against the baseline.

## Dataset

- **File:** `pulsar.csv` (HTRU2, 17,898 samples, 9 columns)
- **Source:** Lyon et al., High Time Resolution Universe Survey, UCI Machine Learning
  Repository ([dataset 372](https://archive.ics.uci.edu/dataset/372/htru2))
- **Task:** binary classification, predict whether a radio telescope detection is a
  real **pulsar** (a rare rotating neutron star) or just interference and noise.
- **Features:** 8 continuous variables. The mean, standard deviation, excess kurtosis
  and skewness of the **integrated pulse profile** (`MeanIP`, `StdIP`, `KurtosisIP`,
  `SkewnessIP`), and the same four statistics of the **DM-SNR curve** (`MeanDMSNR`,
  `StdDMSNR`, `KurtosisDMSNR`, `SkewnessDMSNR`). Target is `Class` (1 = pulsar).
- **Why this one:** it is an uncommon dataset from a domain nobody else picks
  (astronomy rather than the usual medical or retail sets), every row was verified by
  a human annotator, and it is genuinely **imbalanced at 9.2% positive**. That
  imbalance is the point: it makes stratified folds necessary, makes accuracy
  actively misleading, and gives hyperparameter tuning something real to chew on.

## What the notebook covers

- **Part A — Dataset preparation:** the features and the imbalance, a stratified
  80/20 split, and why scaling is correctly skipped for a tree ensemble (the opposite
  of Task 09, where SVM and kNN needed it).
- **Part B — Cross-Validation:** a baseline Random Forest, 5-fold StratifiedKFold on
  both accuracy and F1, and why the gap between them is the whole story.
- **Part C — Hyperparameter tuning:** GridSearchCV (72 combinations, 360 fits) and
  RandomizedSearchCV (30 candidates, 150 fits) over a wider space, both scored on F1,
  with the optimal parameters compared and full metrics plus confusion matrices.
- **Part D — Comparative analysis:** baseline vs both tuned models, the advantages and
  limitations of each search, and a recommendation.

## Key results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Baseline (defaults) | 0.9821 | 0.9286 | 0.8720 | 0.8994 | 0.9610 |
| GridSearchCV | 0.9807 | 0.9060 | 0.8811 | 0.8934 | 0.9679 |
| RandomizedSearchCV | 0.9796 | 0.8947 | 0.8811 | 0.8879 | 0.9690 |

Baseline 5-fold CV: accuracy 0.9798 (std 0.0026), **F1 0.8843 (std 0.0162)**. Accuracy
is flattering here because always guessing "noise" scores 0.908 for free, so
everything is tuned on F1 instead.

| Search | Fits | Time | Best CV F1 |
| --- | --- | --- | --- |
| GridSearchCV | 360 | 456.5 s | 0.8928 |
| RandomizedSearchCV | 150 | 325.2 s | 0.8899 |

(Fit counts and scores are fixed by the random seeds. The times are from the
`cv_tuning.py` run that generated the figures and will vary with your machine and its
load, the notebook run of the same code took 852 s and 417 s on a busy laptop.)

**The honest headline: tuning was not worth much on this dataset.** It improved the
score it optimised (CV F1 0.8843 to 0.8928) but the baseline actually edges the test
F1. Every gap is smaller than the 0.0162 fold-to-fold noise, so the three models are
statistically tied. What tuning did buy is real if modest: both searches chose a
*balanced* class weight, trading precision for **recall** (0.8720 to 0.8811, catching
3 more pulsars) and improving **ROC-AUC** (0.9610 to 0.9690), which does not depend on
the 0.5 threshold. For pulsar hunting, where a miss is a lost discovery and a false
alarm just costs a few minutes of review, that trade is worth taking.

**Recommendation: RandomizedSearchCV.** It landed within 0.3% of grid search's score
using 42% of the fits, while searching a wider space. Best practice is randomized
first over a wide space, then a small grid around the winner only if the payoff earns
the time.

## How to run

```bash
pip install pandas numpy scipy matplotlib scikit-learn jupyter
jupyter notebook cv_tuning.ipynb   # then Run All
```

Or run the plain script, which regenerates the five figures in `figures/`:

```bash
python cv_tuning.py
```

Heads up: the two searches total roughly 13 to 21 minutes depending on your machine
and how busy it is (510 Random Forest fits in total).

## Files

| File | Description |
| --- | --- |
| `pulsar.csv` | The dataset |
| `cv_tuning.ipynb` | The full notebook (Parts A to D), with outputs |
| `cv_tuning.py` | The same analysis as a plain script |
| `figures/` | The five generated plots |
| `Report.pdf` / `Report.tex` | Short written report |
| `README.md` | This file |
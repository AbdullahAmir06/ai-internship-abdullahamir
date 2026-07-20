# Classification Models: Logistic Regression, Decision Trees & Random Forests

**PKCERT AI & Software Development Internship, Task 08**
Author: Abdullah Amir

A Jupyter notebook that builds and evaluates three classifiers on the Pima Indians
Diabetes dataset, using scikit-learn, and compares them with a full set of
classification metrics.

## Dataset

- **File:** `diabetes.csv` (Pima Indians Diabetes, 768 patients, 9 columns)
- **Source:** originally the UCI / National Institute of Diabetes repository
- **Task:** binary classification, predict `Outcome` (1 = diabetes) from 8 clinical
  measurements: glucose, blood pressure, skin thickness, insulin, BMI, diabetes
  pedigree, age, and number of pregnancies
- **Why this one:** it is a classic, genuinely imbalanced medical dataset (35%
  positive) where accuracy alone is not enough, and several columns encode missing
  values as 0, so it exercises a real cleaning step before modelling.

## What the notebook covers

- **Part A — Logistic Regression:** the sigmoid, how the weights are learned, a
  real-world application, feature scaling, and a trained model.
- **Part B — Classification metrics:** accuracy, precision, recall, and F1 with
  what each one means and why it matters, plus a confusion matrix.
- **Part C — Decision Trees & Random Forests:** how each works, a depth-4 tree
  visualisation, the forest's feature importances, and their advantages and limits.
- **Part D — Comparative analysis:** all three side by side on the test set, ROC
  curves, a metric bar chart, 5-fold cross-validation, and a recommendation.

## Key results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV acc (std) |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.708 | 0.600 | 0.500 | 0.545 | 0.813 | 0.772 (0.018) |
| Decision Tree | 0.786 | 0.698 | 0.685 | 0.692 | 0.789 | 0.732 (0.040) |
| Random Forest | 0.740 | 0.652 | 0.556 | 0.600 | 0.816 | 0.761 (0.035) |

The Decision Tree tops this single test split, but cross-validation shows it is the
least stable of the three (lowest mean, widest spread). **Random Forest is the
recommended model**: it has the best ROC-AUC (0.816), so it ranks patients by risk
best, and it is far more robust than the lone tree. Logistic Regression is a strong,
more interpretable runner-up.

## How to run

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook classification_models.ipynb   # then Run All
```

Running the notebook regenerates the six figures in `figures/`.

## Files

| File | Description |
| --- | --- |
| `diabetes.csv` | The dataset |
| `classification_models.ipynb` | The full notebook (Parts A to D), with outputs |
| `figures/` | The six generated plots |
| `Report.pdf` / `Report.tex` | Short written report |
| `README.md` | This file |

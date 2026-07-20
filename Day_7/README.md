# ML Foundations: Train/Test Split, Model Fit & Linear Regression

**PKCERT AI & Software Development Internship, Task 07**
Author: Abdullah Amir

A Jupyter notebook covering core ML theory and a Linear Regression model built
with scikit-learn on the Auto MPG dataset.

## Dataset

- **File:** `mpg.csv` (Auto MPG, 398 cars, 9 columns)
- **Source:** UCI Machine Learning Repository (via the seaborn mirror)
- **Task:** regression, predict `mpg` (miles per gallon) from the car's specs
- **Why this one:** it has genuine missing values (`horsepower`) and a
  categorical column (`origin`), so it exercises the full cleaning and encoding
  workflow the task asks for.

## What the notebook covers

- **Part A — Train/test split:** why we split, common ratios, `random_state` and
  reproducibility, validation vs test sets, and a `train_test_split` demo.
- **Part B — Overfitting, underfitting & bias-variance:** definitions, the
  tradeoff, techniques to reduce overfitting, and a training-vs-test error plot
  across polynomial degrees.
- **Part C — Linear regression:** the maths (`y = mx + b` and multiple
  regression), how coefficients are learned, a fitted model, and a regression
  line plus predicted-vs-actual plots.
- **Part D — Practical session:** load raw CSV, handle missing values, encode
  categoricals, compare three split ratios, k-fold cross-validation, a learning
  curve, and code refactored into `load_data`, `split_data`, `train_model`, and
  `evaluate_model`.
- **Bonus — Polynomial regression:** degrees 1 to 3 compared.

## Key results

| Model | Test R2 |
| --- | --- |
| Simple regression (weight only) | 0.723 |
| Multiple regression (all features) | 0.845 |
| 5-fold cross-validation | 0.814 |
| Polynomial degree 2 | 0.895 |
| Polynomial degree 3 | 0.076 (overfits) |

The plain multiple model reaches R2 = 0.845. Polynomial degree 2 generalises
best; degree 3 overfits badly (train 0.954, test 0.076).

## Screenshots

The **Environment** cell prints the installed library versions (scikit-learn,
pandas, numpy, matplotlib), and the Part D cells print the evaluation metrics.
Those cells serve as the environment and metrics screenshots the task asks for.

## How to run

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook ml_foundations.ipynb   # then Run All
```

Running the notebook regenerates the five figures in `figures/`.

## Files

| File | Description |
| --- | --- |
| `mpg.csv` | The dataset |
| `ml_foundations.ipynb` | The full notebook (Parts A to D + bonus), with outputs |
| `figures/` | The five generated plots |
| `Report.pdf` / `Report.tex` | Short written report |
| `README.md` | This file |

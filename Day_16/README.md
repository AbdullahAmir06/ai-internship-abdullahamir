# End-to-End ML Pipeline: Ames Housing Price Prediction

**PKCERT AI & Software Development Internship, Task 16 (Capstone)**
Author: Abdullah Amir

A complete, documented pipeline — EDA, data cleaning, feature engineering,
preprocessing, model development, and evaluation — built as a single project rather
than a sequence of separate exercises. This is the capstone task: it draws on
preprocessing (Days 4-5), regression (Day 7), classification metrics (Day 8),
cross-validation (Day 10), ensembles (Day 14), and persistence (Day 15) from earlier in
the internship, applied together on one new dataset.

## Dataset

- **File:** `house_prices.csv` (Ames Housing, De Cock 2011, 1,460 sales, 80 columns),
  mirrored on [OpenML, dataset 42563](https://www.openml.org/d/42563); originally
  released on Kaggle as *House Prices: Advanced Regression Techniques*.
- **Features:** 79 raw predictors describing a residential sale in Ames, Iowa — 37
  numeric (areas, room counts, `YearBuilt`, etc.) and 43 categorical (`OverallQual`,
  `KitchenQual`, `Neighborhood`, and 40 more quality/type/condition ratings).
- **Target:** `SalePrice`, in US dollars.
- **Why this one:** not used in any earlier task, and it is genuinely rich enough to
  justify every part of this task's rubric — real, meaningfully-structured missing
  values (not just noise), documented outliers, obvious feature-engineering
  opportunities, and a skewed target that rewards a log transform. Most tutorials on
  this dataset skip straight to modelling; this one spends real effort on Part A
  because that is where most of the actual decisions live.

## Methodology

1. **Target transform.** `SalePrice` is right-skewed (skew 1.88, driven by a small
   number of expensive homes); `log1p(SalePrice)` is close to symmetric (skew 0.12), so
   every model is trained on the log target and every reported metric is converted back
   to dollars.
2. **Outlier removal, checked per-point.** The dataset's author documents specific
   outliers among the largest houses by living area. Rather than filter by one
   threshold, the four largest houses were inspected individually: two are large *and*
   expensive (kept), two are large and suspiciously cheap (dropped) — only the latter
   pair are true outliers.
3. **Missing values, read against the data dictionary before filling anything.** 19 of
   80 columns have missing values, 7,829 in total. In this dataset, `NaN` in most
   quality/type columns means the feature is *absent* (`PoolQC = NaN` → no pool), not
   unknown — those are filled with an explicit `"None"` / `0`. Only `LotFrontage` (259
   rows, imputed by neighbourhood median) and `Electrical` (1 row, mode) are genuinely
   incomplete and get real imputation.
4. **Feature engineering.** 8 features built from domain knowledge: `TotalSF` (summed
   floor + basement area), `HouseAge` / `RemodAge` (relative to sale year), `TotalBath`
   (weighted bathroom count), and 4 presence flags. Each is checked against the
   correlation it was hoped to improve, not assumed to help.
5. **Preprocessing pipeline.** A single `ColumnTransformer`: numeric columns are
   median-imputed and standard-scaled; categorical columns are most-frequent-imputed
   and one-hot encoded (→ 87 total features). Wrapped in a `Pipeline` with the model so
   preprocessing and model travel together.
6. **Model comparison.** Ridge Regression and a 400-tree Random Forest, both
   cross-validated (5-fold) on the training set and then scored once on a held-out test
   set, so the model selection is justified by two independent checks rather than one.

## Implementation

- `house_price_pipeline.py` — the full pipeline as a plain, commented script; running
  it reproduces every number in this README and regenerates all six figures.
- `house_price_pipeline.ipynb` — the same pipeline as a notebook with all outputs and
  plots inline, organized into the same four parts as the task rubric (A/B/C/D).
- Code style: one `ColumnTransformer` + `Pipeline` per model (no manual
  fit/transform bookkeeping), functions and variable names that describe *what*
  rather than requiring inline comments to explain it, and print statements at every
  decision point (missing-value strategy, outlier removal, feature engineering) so the
  script's own output is a readable audit trail of what it did and why.

## Results

| Model | CV RMSE (log price) | CV R² | Test R² | Test RMSE | Test MAE | MAPE |
| --- | --- | --- | --- | --- | --- | --- |
| **Ridge Regression** | **0.1120 ± 0.010** | **0.9191** | **0.9095** | **$20,155** | **$14,611** | **9.01%** |
| Random Forest | 0.1353 ± 0.013 | 0.8829 | 0.8764 | $23,600 | $16,367 | 10.24% |

Four findings, in order of how much they mattered to the final result:

1. **The engineered `TotalSF` outperforms the features it was built from**: correlation
   0.833 with `SalePrice`, versus 0.735 for `GrLivArea` alone and 0.63-0.65 for the
   individual floor/basement areas it sums — a checked reason to keep it, not a
   convenience.
2. **Only 2 of the 4 largest houses by living area are real outliers**; the other two
   are legitimately large, expensive homes. A blanket `GrLivArea` cutoff would have
   discarded real signal.
3. **Ridge beat Random Forest on every metric**, in both cross-validation and on the
   test set. With 87 mostly-linear-in-log-price features and 1,166 training rows,
   regularised linear regression suits this problem better than an untuned tree
   ensemble — a result specific to this dataset and this (untuned) Random Forest, not a
   general claim about linear models versus ensembles.
4. **Error variance grows with price** (heteroscedasticity): residuals cluster tightly
   under $200k and widen past $300k, so the model is most trustworthy for a typical
   Ames home and least trustworthy for the priciest ones.

**Strengths:** R² = 0.910, MAE ≈ $14,600 (about 9% of a typical sale price), fully
interpretable coefficients, honest dollar-denominated error metrics.
**Limitations:** Ames-specific (no claim of transfer to another market), the Random
Forest baseline was not hyperparameter-tuned, and predictions for high-end homes carry
more uncertainty than the headline R² suggests.

## Conclusion

The thread running through this project is that every pipeline decision was checked
against the data rather than taken on faith: the log transform because the skew numbers
justified it, the two dropped rows because they were individually verified as outliers,
the `"None"`/0 fill because the data dictionary says most of those gaps aren't gaps, and
`TotalSF` because it measurably outperforms its own inputs. None of these would survive
being asserted without the number behind them. The result — Ridge Regression,
R² = 0.91, MAE ≈ 9% of price — is a genuinely useful, interpretable model for this
dataset, with its remaining weaknesses (Ames-specific, weaker at the high end) stated
rather than hidden behind a single score.

## How to run

```bash
python3 -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install pandas numpy matplotlib scikit-learn jupyter ipykernel
python -c "from sklearn.datasets import fetch_openml; d = fetch_openml(data_id=42563, as_frame=True).frame; [d.__setitem__(c, d[c].astype(object)) for c in d.columns if str(d[c].dtype)=='category']; d.to_csv('house_prices.csv', index=False)"
jupyter notebook house_price_pipeline.ipynb   # then Run All
```

Or run the plain script, which regenerates the six figures in `figures/`:

```bash
python house_price_pipeline.py
```

Full run (cleaning, feature engineering, 5-fold CV for two models, all six figures)
takes under a minute.

## Files

| File | Description |
| --- | --- |
| `house_prices.csv` | The dataset |
| `house_price_pipeline.ipynb` | The full notebook (Parts A to D), with outputs |
| `house_price_pipeline.py` | The same pipeline as a plain script |
| `figures/` | The six generated plots |
| `Report.pdf` / `Report.tex` | Short written report |
| `README.md` | This file |

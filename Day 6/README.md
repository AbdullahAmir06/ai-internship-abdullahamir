# EDA: Red Wine Quality (Jupyter Notebook)

**PKCERT AI & Software Development Internship, Task 06**
Author: Abdullah Amir

An exploratory data analysis of the UCI Red Wine Quality dataset, done in a
Jupyter Notebook with markdown notes at every step. The notebook is the main
deliverable; a written report (`Report.pdf`) pulls the findings together.

## Dataset

- **File:** `winequality-red.csv` (1,599 wines, 12 columns, semicolon separated)
- **Source:** UCI Machine Learning Repository, Wine Quality
  ([link](https://archive.ics.uci.edu/ml/datasets/wine+quality)), Cortez et al. 2009.
- **Purpose:** Each row is a Portuguese red *vinho verde* wine with 11
  physicochemical measurements and a taster `quality` score. The goal is to see
  which chemical properties relate to the score.
- **Target:** `quality`, an integer score from 3 to 8.

## What the notebook covers

- **Part A:** describes the dataset (source, purpose, records, features, target).
- **Part B:** explores structure and summary stats, checks for missing values,
  duplicates and inconsistencies, then draws six visualizations with an
  interpretation under each.
- **Part C:** written findings, what they could mean, and the dataset's
  limitations with suggested next steps.
- **Part D:** organized with markdown headings and commented code throughout.

## Data quality notes

- **No missing values** anywhere in the dataset.
- **240 duplicate rows** were found and dropped, leaving **1,359 unique rows**.
- No inconsistencies: all measurements are positive and in sensible ranges.

## Key findings

- **Alcohol** has the strongest positive link to quality (about +0.48), and
  **volatile acidity** the strongest negative link (about -0.40).
- Higher rated wines tend to be more alcoholic and less acidic.
- **Sulphates** and **citric acid** give a smaller positive push.
- The `quality` target is **imbalanced**, with most wines rated 5 or 6, which is
  the main limitation of the dataset.

## Visualizations (saved in `figures/`)

1. Distribution of the quality score
2. Histograms of all eleven features
3. Correlation heatmap
4. Alcohol content by quality score
5. Volatile acidity by quality score
6. Correlation of each feature with quality

## How to run

```bash
# 1. (optional) create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows

# 2. install dependencies
pip install pandas numpy matplotlib seaborn jupyter

# 3. open the notebook
jupyter notebook EDA_Wine_Quality.ipynb
# then Run All, or run cells top to bottom
```

Running the notebook regenerates the six figures in the `figures/` folder.

## Files

| File | Description |
| --- | --- |
| `winequality-red.csv` | The dataset |
| `EDA_Wine_Quality.ipynb` | The EDA notebook (Parts A to D), with outputs |
| `figures/` | The six generated visualizations |
| `Report.pdf` / `Report.tex` | The written report |
| `README.md` | This file |

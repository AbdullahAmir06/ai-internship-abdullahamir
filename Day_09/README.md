# SVM & k-Nearest Neighbors: Classifying Raisin Varieties

**PKCERT AI & Software Development Internship, Task 09**
Author: Abdullah Amir

A Jupyter notebook that builds, tunes and compares a Support Vector Machine and a
k-Nearest Neighbors classifier on the UCI Raisin dataset, using scikit-learn, with a
full set of classification metrics.

## Dataset

- **File:** `raisin.csv` (UCI Raisin, 900 samples, 8 columns)
- **Source:** Cinar, Koklu & Tasdemir, UCI Machine Learning Repository ([dataset 850](https://archive.ics.uci.edu/dataset/850/raisin))
- **Task:** binary classification, predict the variety (`Kecimen` vs `Besni`) from 7
  numeric shape features measured from images: area, major and minor axis lengths,
  eccentricity, convex area, extent and perimeter.
- **Why this one:** it is a less common dataset than the usual teaching sets, it is
  perfectly balanced (450 per class, so accuracy is a fair headline metric), and its
  features sit on wildly different scales (area in the tens of thousands, extent near
  0.7). That scale gap makes feature scaling essential, which is exactly what both of
  these distance-based algorithms need to show off.

## What the notebook covers

- **Part A — Dataset preparation:** the class balance, the feature scale problem, an
  80/20 stratified split, and standard scaling fit on the training set only.
- **Part B — Support Vector Machine:** the margin and support vectors, the kernel
  trick, an RBF model plus a linear-kernel sanity check, metrics and a confusion
  matrix, and the algorithm's advantages, limits and applications.
- **Part C — k-Nearest Neighbors:** how the lazy voting works, a K = 1..25 sweep with
  5-fold cross-validation to pick K, metrics and a confusion matrix, and its
  advantages, limits and applications.
- **Part D — Comparative analysis:** both models side by side on the test set, a
  metric bar chart, 5-fold cross-validation, and a recommendation.

## Key results

| Model | Accuracy | Precision | Recall | F1 | CV acc (std) |
| --- | --- | --- | --- | --- | --- |
| SVM (RBF kernel) | 0.911 | 0.963 | 0.856 | 0.906 | 0.868 (0.021) |
| kNN (K = 8) | 0.861 | 0.945 | 0.767 | 0.847 | 0.847 (0.014) |

Both models are very precise but a little cautious about the Besni class, so the gap
between them is really a gap in recall. The **SVM with an RBF kernel is the
recommended model**: it leads on every headline metric, holds that lead under
cross-validation, and suits a small low-dimensional problem far better than kNN's
slow, memory-hungry prediction. kNN is a solid, dead-simple baseline that lands only
a few points behind.

## How to run

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook svm_knn.ipynb   # then Run All
```

Or run the plain script, which regenerates the five figures in `figures/`:

```bash
python svm_knn.py
```

## Files

| File | Description |
| --- | --- |
| `raisin.csv` | The dataset |
| `svm_knn.ipynb` | The full notebook (Parts A to D), with outputs |
| `svm_knn.py` | The same analysis as a plain script |
| `figures/` | The five generated plots |
| `Report.pdf` / `Report.tex` | Short written report |
| `README.md` | This file |

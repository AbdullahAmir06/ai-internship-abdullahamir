# Clustering & Dimensionality Reduction: Finding Frog Species in Audio

**PKCERT AI & Software Development Internship, Task 12**
Author: Abdullah Amir

A Jupyter notebook that runs **K-Means** and **Hierarchical (Ward)** clustering on frog
call recordings, then compares **PCA** and **t-SNE** for 2D visualisation. This is the
first fully unsupervised task: the species labels are never shown to any model, only used
afterwards to score what the clustering found.

## Dataset

- **File:** `frogs.csv` (UCI Anuran Calls (MFCCs), 7,195 samples, 26 columns)
- **Source:** Colonna et al., UCI Machine Learning Repository ([dataset 406](https://archive.ics.uci.edu/dataset/406/anuran+calls+mfccs))
- **Task:** clustering. Each row is one **syllable** of a frog call, described by 22
  **MFCCs** (Mel-Frequency Cepstral Coefficients, the standard way to describe the timbre
  of a sound, and the same features used in speech recognition).
- **Ground truth:** a real taxonomy of 4 families, 8 genera and 10 species, held back
  from every model and used only for scoring.
- **Why this one:** it is genuinely uncommon, and the nested taxonomy gives the
  dendrogram something real to be judged against rather than being a decoration. It is
  also honestly difficult: the classes are severely imbalanced (one species is 48% of the
  data, another under 1%), which turns out to drive most of the results.

## What the notebook covers

- **Part A — Preparation:** what MFCCs are, why scaling still matters even though the
  features are already in `[-1, 1]`, and two caveats (imbalance, and non-independent
  syllables) that shape everything after.
- **Part B — Clustering:** K-Means with elbow and silhouette to choose `k`, Hierarchical
  (Ward) clustering with a truncated dendrogram, and both scored against the taxonomy
  with ARI and NMI.
- **Part C — Dimensionality reduction:** PCA with a scree plot, t-SNE at perplexity 30,
  clustering run in four different spaces, and a full comparison of the two methods.
- **Part D — Analysis & conclusion:** what the results mean and which techniques to use.

## Key results

| Method (k = 10) | ARI vs species | NMI vs species | Silhouette |
| --- | --- | --- | --- |
| K-Means | 0.409 | 0.613 | 0.258 |
| **Ward (hierarchical)** | **0.590** | **0.719** | **0.285** |

| K-Means (k = 10) run on | ARI vs species |
| --- | --- |
| All 22 scaled features | 0.409 |
| **The first 9 principal components** | **0.557** |
| The 2D PCA projection | 0.378 |
| The 2D t-SNE embedding | 0.335 |

Four findings, most of them counterintuitive:

1. **The elbow method gave no answer** (no sharp knee) and **the silhouette gave a
   confident wrong one**: it peaks at k = 4 when there are 10 species.
2. **k = 4 does *not* mean it found the 4 families.** That is the obvious guess and it is
   false: ARI vs Family at k = 4 is only 0.446, *lower* than vs Species (0.714). Worth
   checking rather than asserting.
3. **Ward beats K-Means on every measure**, because K-Means assumes round, equal-sized
   clusters and this data has one species at 48% and another at 0.9%.
4. **Fewer dimensions clustered better** (9 PCs beat all 22), while **the prettiest plot
   made the worst clusters** (t-SNE is the best picture and the lowest ARI).

**Recommendation:** Ward linkage on the first 9 principal components for clustering;
t-SNE for visualising **only**, never as a feature space to compute on, since its
distances and the gaps between its islands are not real.

**Honest caveat:** these scores are probably optimistic. The 7,195 syllables come from
only **60 recordings**, so syllables from one recording are the same individual frog in
one moment. Some of what the clusters "found" is likely *that frog on that evening*
rather than *that species*.

## How to run

```bash
pip install pandas numpy scipy matplotlib scikit-learn jupyter
jupyter notebook clustering_dimred.ipynb   # then Run All
```

Or run the plain script, which regenerates the six figures in `figures/`:

```bash
python clustering_dimred.py
```

Heads up: t-SNE on 7,195 points takes about a minute, and the full run a few minutes.

## Files

| File | Description |
| --- | --- |
| `frogs.csv` | The dataset |
| `clustering_dimred.ipynb` | The full notebook (Parts A to D), with outputs |
| `clustering_dimred.py` | The same analysis as a plain script |
| `figures/` | The six generated plots |
| `Report.pdf` / `Report.tex` | Short written report |
| `README.md` | This file |

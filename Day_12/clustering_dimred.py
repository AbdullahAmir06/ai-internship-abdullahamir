"""
PKCERT AI & Software Development Internship, Task 12
Clustering & Dimensionality Reduction

Unsupervised analysis of the UCI Anuran Calls (MFCCs) dataset: K-Means and
Hierarchical clustering, then PCA and t-SNE for 2D visualisation. The taxonomic
labels are held back from every model and used only to score the clusters
afterwards. Running this file reproduces every number in Report.tex and writes
all figures into figures/.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             normalized_mutual_info_score)

RANDOM_STATE = 42
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

# ----------------------------------------------------------------------
# Part A: Dataset selection and preparation
# ----------------------------------------------------------------------
df = pd.read_csv("frogs.csv")
MFCC = [c for c in df.columns if c.startswith("MFCC_")]

print("Shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print(f"\n{len(MFCC)} MFCC features")
for c in ["Family", "Genus", "Species"]:
    print(f"{c:8s}: {df[c].nunique()} classes")
print("\nSpecies distribution:")
print(df["Species"].value_counts().to_string())
print(f"\nRecordings: {df['RecordID'].nunique()} "
      f"(syllables per recording: min {df['RecordID'].value_counts().min()}, "
      f"max {df['RecordID'].value_counts().max()})")

X_raw = df[MFCC].values
print("\nRaw MFCC ranges: min %.3f max %.3f" % (X_raw.min(), X_raw.max()))
print("Per-feature std, min %.4f max %.4f" % (X_raw.std(axis=0).min(), X_raw.std(axis=0).max()))

# Scale: the MFCCs are already in [-1, 1] but their spreads differ by ~4x, and
# every method here (K-Means, Ward, PCA) is variance driven, so an unscaled
# high-variance coefficient would quietly dominate all of them.
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
print("After scaling: mean %.2e, std %.3f" % (X.mean(), X.std()))

y_species = df["Species"].values
y_family = df["Family"].values
y_genus = df["Genus"].values

# ----------------------------------------------------------------------
# Part B1: K-Means, choosing k
# ----------------------------------------------------------------------
print("\n--- K-Means: sweeping k ---")
ks = range(2, 16)
inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    lab = km.fit_predict(X)
    inertias.append(km.inertia_)
    s = silhouette_score(X, lab, sample_size=3000, random_state=RANDOM_STATE)
    sils.append(s)
    print(f"k={k:2d}  inertia {km.inertia_:9.0f}  silhouette {s:.4f}")

best_k_sil = list(ks)[int(np.argmax(sils))]
print(f"\nBest k by silhouette: {best_k_sil} (silhouette {max(sils):.4f})")

# Score k against each taxonomic level; labels never touched the fitting.
print("\n--- K-Means vs the taxonomy (ARI) ---")
for k in [4, 8, 10, best_k_sil]:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    lab = km.fit_predict(X)
    print(f"k={k:2d}  ARI vs Family {adjusted_rand_score(y_family, lab):.3f} | "
          f"Genus {adjusted_rand_score(y_genus, lab):.3f} | "
          f"Species {adjusted_rand_score(y_species, lab):.3f}")

# The headline K-Means model: k = 10, matching the 10 true species.
km10 = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=10)
km_labels = km10.fit_predict(X)
km_ari = adjusted_rand_score(y_species, km_labels)
km_nmi = normalized_mutual_info_score(y_species, km_labels)
km_sil = silhouette_score(X, km_labels, sample_size=3000, random_state=RANDOM_STATE)
print(f"\nK-Means k=10: ARI {km_ari:.3f} | NMI {km_nmi:.3f} | silhouette {km_sil:.3f}")

# ----------------------------------------------------------------------
# Part B2: Hierarchical clustering
# ----------------------------------------------------------------------
print("\n--- Hierarchical (Ward) ---")
Z = linkage(X, method="ward")
hc_labels = fcluster(Z, t=10, criterion="maxclust") - 1
hc_ari = adjusted_rand_score(y_species, hc_labels)
hc_nmi = normalized_mutual_info_score(y_species, hc_labels)
hc_sil = silhouette_score(X, hc_labels, sample_size=3000, random_state=RANDOM_STATE)
print(f"Ward k=10: ARI {hc_ari:.3f} | NMI {hc_nmi:.3f} | silhouette {hc_sil:.3f}")

for k in [4, 8, 10]:
    lab = fcluster(Z, t=k, criterion="maxclust") - 1
    print(f"Ward k={k:2d}  ARI vs Family {adjusted_rand_score(y_family, lab):.3f} | "
          f"Genus {adjusted_rand_score(y_genus, lab):.3f} | "
          f"Species {adjusted_rand_score(y_species, lab):.3f}")

agree = adjusted_rand_score(km_labels, hc_labels)
print(f"\nAgreement between K-Means and Ward (ARI): {agree:.3f}")

# ----------------------------------------------------------------------
# Part C1: PCA
# ----------------------------------------------------------------------
print("\n--- PCA ---")
pca_full = PCA().fit(X)
cum = np.cumsum(pca_full.explained_variance_ratio_)
n90 = int(np.searchsorted(cum, 0.90) + 1)
n95 = int(np.searchsorted(cum, 0.95) + 1)
print(f"PC1 {pca_full.explained_variance_ratio_[0]*100:.1f}% | "
      f"PC2 {pca_full.explained_variance_ratio_[1]*100:.1f}% | "
      f"PC1+PC2 {cum[1]*100:.1f}%")
print(f"Components for 90% variance: {n90} | for 95%: {n95} (of {len(MFCC)})")

pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca2.fit_transform(X)

# Clustering in PCA space (a common speedup) to see what the tradeoff costs
pca10 = PCA(n_components=n90, random_state=RANDOM_STATE)
X_pca10 = pca10.fit_transform(X)
km_pca = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=10).fit_predict(X_pca10)
print(f"K-Means on {n90} PCs: ARI vs species {adjusted_rand_score(y_species, km_pca):.3f} "
      f"(vs {km_ari:.3f} on all 22)")

# ----------------------------------------------------------------------
# Part C2: t-SNE
# ----------------------------------------------------------------------
print("\n--- t-SNE (this takes a minute) ---")
tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
            random_state=RANDOM_STATE)
X_tsne = tsne.fit_transform(X)
print("t-SNE done. KL divergence: %.3f" % tsne.kl_divergence_)

# Do the 2D embeddings preserve cluster structure? Score k-means run in each.
km_on_pca2 = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=10).fit_predict(X_pca)
km_on_tsne = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=10).fit_predict(X_tsne)
print(f"K-Means on 2D PCA  : ARI vs species {adjusted_rand_score(y_species, km_on_pca2):.3f}")
print(f"K-Means on 2D t-SNE: ARI vs species {adjusted_rand_score(y_species, km_on_tsne):.3f}")

# ======================================================================
# Figures
# ======================================================================
species_order = df["Species"].value_counts().index.tolist()
cmap10 = plt.get_cmap("tab10")
sp_color = {s: cmap10(i) for i, s in enumerate(species_order)}

# 01 dataset overview
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
counts = df["Species"].value_counts()
ax[0].barh([s[:22] for s in counts.index][::-1], counts.values[::-1],
           color=[sp_color[s] for s in counts.index][::-1])
ax[0].set_title("Syllables per species (imbalanced)")
ax[0].tick_params(axis="y", labelsize=7)
ax[0].set_xlabel("syllables")
bp = ax[1].boxplot([df[c] for c in MFCC], tick_labels=[c.replace("MFCC_", "") for c in MFCC])
ax[1].set_title("The 22 MFCC features (already in [-1, 1])")
ax[1].set_xlabel("MFCC coefficient")
ax[1].tick_params(axis="x", labelsize=6)
fig.tight_layout()
fig.savefig("figures/01_dataset_overview.png", bbox_inches="tight")
plt.close(fig)

# 02 elbow + silhouette
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
ax[0].plot(list(ks), inertias, marker="o", ms=4, color="#4C72B0")
ax[0].set_xlabel("k"); ax[0].set_ylabel("inertia")
ax[0].set_title("Elbow method (no sharp elbow)")
ax[1].plot(list(ks), sils, marker="o", ms=4, color="#DD8452")
ax[1].axvline(best_k_sil, color="crimson", ls="--", lw=1, label=f"best k = {best_k_sil}")
ax[1].axvline(10, color="grey", ls=":", lw=1, label="10 true species")
ax[1].set_xlabel("k"); ax[1].set_ylabel("silhouette score")
ax[1].set_title("Silhouette method")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/02_choosing_k.png", bbox_inches="tight")
plt.close(fig)

# 03 dendrogram (truncated: 7,195 leaves would be unreadable)
fig, ax = plt.subplots(figsize=(10, 4))
dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=8,
           show_contracted=True, ax=ax, color_threshold=Z[-9, 2])
ax.set_title("Hierarchical clustering (Ward), truncated to the last 30 merges")
ax.set_xlabel("cluster (size in brackets)")
ax.set_ylabel("Ward linkage distance")
ax.axhline(Z[-9, 2], color="crimson", ls="--", lw=1, label="cut for k = 10")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/03_dendrogram.png", bbox_inches="tight")
plt.close(fig)

# 04 PCA: scree + 2D scatter coloured by true species
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].bar(range(1, len(MFCC) + 1), pca_full.explained_variance_ratio_ * 100,
          color="#4C72B0", label="individual")
ax2 = ax[0].twinx()
ax2.plot(range(1, len(MFCC) + 1), cum * 100, marker="o", ms=3, color="#C44E52",
         label="cumulative")
ax2.axhline(90, ls="--", lw=1, color="grey")
ax2.set_ylabel("cumulative %")
ax[0].set_xlabel("principal component"); ax[0].set_ylabel("% variance")
ax[0].set_title(f"Scree plot ({n90} PCs reach 90%)")
for s in species_order:
    m = y_species == s
    ax[1].scatter(X_pca[m, 0], X_pca[m, 1], s=3, alpha=0.5, color=sp_color[s],
                  label=s[:18], linewidths=0)
ax[1].set_xlabel(f"PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}%)")
ax[1].set_ylabel(f"PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}%)")
ax[1].set_title("PCA, coloured by true species")
ax[1].legend(fontsize=5, markerscale=2, loc="best", framealpha=0.8)
fig.tight_layout()
fig.savefig("figures/04_pca.png", bbox_inches="tight")
plt.close(fig)

# 05 PCA vs t-SNE side by side, same colouring
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
for s in species_order:
    m = y_species == s
    ax[0].scatter(X_pca[m, 0], X_pca[m, 1], s=3, alpha=0.5, color=sp_color[s],
                  linewidths=0)
    ax[1].scatter(X_tsne[m, 0], X_tsne[m, 1], s=3, alpha=0.5, color=sp_color[s],
                  label=s[:18], linewidths=0)
ax[0].set_title(f"PCA (linear, {cum[1]*100:.1f}% of variance)")
ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
ax[1].set_title("t-SNE (non-linear, perplexity 30)")
ax[1].set_xlabel("t-SNE 1"); ax[1].set_ylabel("t-SNE 2")
ax[1].legend(fontsize=5, markerscale=2, loc="best", framealpha=0.8)
fig.tight_layout()
fig.savefig("figures/05_pca_vs_tsne.png", bbox_inches="tight")
plt.close(fig)

# 06 t-SNE coloured by cluster assignment vs truth
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for s in species_order:
    m = y_species == s
    ax[0].scatter(X_tsne[m, 0], X_tsne[m, 1], s=3, alpha=0.5, color=sp_color[s],
                  linewidths=0)
ax[0].set_title("t-SNE: true species")
for a, lab, name, ari in [(ax[1], km_labels, "K-Means (k=10)", km_ari),
                          (ax[2], hc_labels, "Ward (k=10)", hc_ari)]:
    a.scatter(X_tsne[:, 0], X_tsne[:, 1], s=3, alpha=0.5, c=lab, cmap="tab10",
              linewidths=0)
    a.set_title(f"t-SNE: {name}, ARI {ari:.3f}")
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.savefig("figures/06_clusters_on_tsne.png", bbox_inches="tight")
plt.close(fig)

print("\nFigures written to figures/")

# ----------------------------------------------------------------------
print("\nSUMMARY")
print(f"best k by silhouette      : {best_k_sil} (sil {max(sils):.4f})")
print(f"K-Means k=10              : ARI {km_ari:.3f} NMI {km_nmi:.3f} sil {km_sil:.3f}")
print(f"Ward     k=10             : ARI {hc_ari:.3f} NMI {hc_nmi:.3f} sil {hc_sil:.3f}")
print(f"K-Means vs Ward agreement : ARI {agree:.3f}")
print(f"PCA: PC1 {pca_full.explained_variance_ratio_[0]*100:.1f}% PC2 {pca_full.explained_variance_ratio_[1]*100:.1f}% "
      f"2D total {cum[1]*100:.1f}% | 90% needs {n90} PCs | 95% needs {n95}")
print(f"K-Means on {n90} PCs        : ARI {adjusted_rand_score(y_species, km_pca):.3f}")
print(f"K-Means on 2D PCA         : ARI {adjusted_rand_score(y_species, km_on_pca2):.3f}")
print(f"K-Means on 2D t-SNE       : ARI {adjusted_rand_score(y_species, km_on_tsne):.3f}")

# cluster vs species crosstab for the report
ct = pd.crosstab(pd.Series(km_labels, name="KMeans cluster"),
                 pd.Series(y_species, name="Species"))
print("\nK-Means k=10 vs species crosstab:")
print(ct.to_string())

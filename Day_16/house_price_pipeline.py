"""
PKCERT AI & Software Development Internship, Task 16
End-to-End Machine Learning Pipeline Project

A complete pipeline (EDA -> cleaning -> feature engineering -> preprocessing ->
model -> evaluation) on the Ames Housing dataset (1,460 houses, 79 raw
features), predicting sale price with a regularised linear model and a
Random Forest. Running this file reproduces every number quoted in
Report.tex and writes all figures into figures/.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

# ======================================================================
# Part A: Dataset selection, cleaning, feature engineering, EDA
# ======================================================================
df = pd.read_csv("house_prices.csv")
print("Raw shape:", df.shape)
print(f"Total missing values: {df.isna().sum().sum():,} across "
      f"{(df.isna().sum() > 0).sum()} of {df.shape[1]} columns")

# ----------------------------------------------------------------------
# A2/A3: target distribution and the case for a log transform
# ----------------------------------------------------------------------
print(f"\nSalePrice: mean ${df['SalePrice'].mean():,.0f}, median "
      f"${df['SalePrice'].median():,.0f}, skew {df['SalePrice'].skew():.3f}")
log_skew = np.log1p(df["SalePrice"]).skew()
print(f"log1p(SalePrice) skew: {log_skew:.3f} -- close to 0, symmetric, "
      f"which is why the model below is trained on log(price) and every "
      f"error metric is converted back to dollars afterwards.")

# ----------------------------------------------------------------------
# A3: outliers, checked individually rather than filtered by one threshold
# ----------------------------------------------------------------------
big_houses = df.nlargest(4, "GrLivArea")[["GrLivArea", "SalePrice"]]
print("\nThe 4 largest houses by living area:")
print(big_houses.to_string())
print("Two of these four are the well-documented Ames outliers (Gr Liv Area "
      "> 4,000 sq ft selling for under $200k, far below what their size and "
      "quality would predict) -- the other two are simply large, expensive "
      "houses and were kept. Only the two genuine outliers are dropped, not "
      "every large house.")
outlier_idx = df[(df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)].index
df = df.drop(index=outlier_idx)
print(f"Dropped {len(outlier_idx)} rows: {list(outlier_idx)}. New shape: {df.shape}")

# ----------------------------------------------------------------------
# A3: missing values -- most are not "missing", they mean "not present"
# ----------------------------------------------------------------------
NONE_MEANS_ABSENT_CAT = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
                          "GarageType", "GarageFinish", "GarageQual", "GarageCond",
                          "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
                          "BsmtFinType2", "MasVnrType"]
NONE_MEANS_ABSENT_NUM = ["GarageYrBlt", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
                          "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath",
                          "GarageCars", "GarageArea"]

na_before = df.isna().sum().sum()
for c in NONE_MEANS_ABSENT_CAT:
    df[c] = df[c].fillna("None")
for c in NONE_MEANS_ABSENT_NUM:
    df[c] = df[c].fillna(0)

# LotFrontage is genuinely missing (259 rows) -- imputed by neighbourhood
# median, since street frontage tracks the neighbourhood's typical lot
# shape far better than a single dataset-wide median would.
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
    lambda s: s.fillna(s.median()))
# Electrical has a single missing value -- mode-fill is fine at n=1.
df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

na_after = df.isna().sum().sum()
print(f"\nMissing values before structural fill: {na_before:,}, after: {na_after:,} "
      f"(remaining {na_after} handled by the imputers inside the pipeline below).")

# ----------------------------------------------------------------------
# A3: feature engineering
# ----------------------------------------------------------------------
df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
df["TotalBath"] = (df["FullBath"] + 0.5 * df["HalfBath"] +
                    df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
df["HasPool"] = (df["PoolArea"] > 0).astype(int)
df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
ENGINEERED = ["TotalSF", "HouseAge", "RemodAge", "TotalBath",
              "HasPool", "HasGarage", "HasFireplace", "Has2ndFloor"]
print(f"\nEngineered {len(ENGINEERED)} features: {ENGINEERED}")
print(f"TotalSF correlation with SalePrice: {df['TotalSF'].corr(df['SalePrice']):.3f} "
      f"(higher than any single floor-area feature it was built from: "
      f"1stFlrSF {df['1stFlrSF'].corr(df['SalePrice']):.3f}, "
      f"TotalBsmtSF {df['TotalBsmtSF'].corr(df['SalePrice']):.3f})")

# ----------------------------------------------------------------------
# A2/A3: correlations
# ----------------------------------------------------------------------
numeric_corr = df.select_dtypes("number").corr()["SalePrice"].drop("SalePrice")
top_corr = numeric_corr.reindex(numeric_corr.abs().sort_values(ascending=False).index)
print("\nTop 10 features correlated with SalePrice:")
print(top_corr.head(10).to_string())

# ======================================================================
# Part B: preprocessing + model pipeline
# ======================================================================
TARGET = "SalePrice"
DROP = ["Id"] if "Id" in df.columns else []
y = np.log1p(df[TARGET].values)
X = df.drop(columns=[TARGET] + DROP)

NUMERIC = X.select_dtypes("number").columns.tolist()
CATEGORICAL = X.select_dtypes(exclude="number").columns.tolist()
print(f"\nFinal feature set: {len(NUMERIC)} numeric + {len(CATEGORICAL)} categorical "
      f"= {len(NUMERIC) + len(CATEGORICAL)} features")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE)
print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), NUMERIC),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), CATEGORICAL),
])

candidates = {
    "Ridge Regression": Ridge(alpha=10.0, random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1),
}

results = {}
cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
for name, model in candidates.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    cv_rmse = -cross_val_score(pipe, X_train, y_train, cv=cv,
                                scoring="neg_root_mean_squared_error")
    cv_r2 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")
    pipe.fit(X_train, y_train)
    pred_log = pipe.predict(X_test)
    pred = np.expm1(pred_log)
    actual = np.expm1(y_test)
    rmse_log = np.sqrt(mean_squared_error(y_test, pred_log))
    rmse_dollars = np.sqrt(mean_squared_error(actual, pred))
    mae_dollars = mean_absolute_error(actual, pred)
    r2 = r2_score(y_test, pred_log)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    results[name] = {
        "pipe": pipe, "cv_rmse_log_mean": cv_rmse.mean(), "cv_rmse_log_std": cv_rmse.std(),
        "cv_r2_mean": cv_r2.mean(), "cv_r2_std": cv_r2.std(),
        "rmse_log": rmse_log, "rmse_dollars": rmse_dollars, "mae_dollars": mae_dollars,
        "r2": r2, "mape": mape, "pred": pred, "actual": actual,
        "pred_log": pred_log,
    }
    print(f"\n=== {name} ===")
    print(f"CV (train, log-price): RMSE {cv_rmse.mean():.4f} +/- {cv_rmse.std():.4f} | "
          f"R2 {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
    print(f"Test: RMSE(log) {rmse_log:.4f} | RMSE ${rmse_dollars:,.0f} | "
          f"MAE ${mae_dollars:,.0f} | R2 {r2:.4f} | MAPE {mape:.2f}%")

best_name = max(results, key=lambda n: results[n]["r2"])
print(f"\nBest model on test R2: {best_name}")

# ======================================================================
# Part C: evaluation figures
# ======================================================================

# 01 target distribution: before/after log
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
ax[0].hist(df["SalePrice"], bins=40, color="#4C72B0")
ax[0].set_xlabel("SalePrice, $"); ax[0].set_ylabel("houses")
ax[0].set_title(f"SalePrice (skew {df['SalePrice'].skew():.2f})")
ax[1].hist(np.log1p(df["SalePrice"]), bins=40, color="#55A868")
ax[1].set_xlabel("log(1 + SalePrice)"); ax[1].set_ylabel("houses")
ax[1].set_title(f"log(SalePrice) (skew {log_skew:.2f})")
fig.tight_layout()
fig.savefig("figures/01_target_distribution.png", bbox_inches="tight")
plt.close(fig)

# 02 missing values, structural vs genuine
na_counts = pd.read_csv("house_prices.csv").isna().sum()
na_counts = na_counts[na_counts > 0].sort_values(ascending=False)
colors = ["#8172B2" if c in NONE_MEANS_ABSENT_CAT + NONE_MEANS_ABSENT_NUM
          else "#C44E52" for c in na_counts.index]
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.barh(na_counts.index[::-1], na_counts.values[::-1], color=colors[::-1])
ax.set_xlabel("missing values (of 1,460 rows)")
ax.set_title("Missing values: purple = structural ('not present'), red = genuinely missing")
fig.tight_layout()
fig.savefig("figures/02_missing_values.png", bbox_inches="tight")
plt.close(fig)

# 03 correlations + outlier scatter
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
top_corr.head(10)[::-1].plot(kind="barh", ax=ax[0], color="#DD8452")
ax[0].set_xlabel("correlation with SalePrice")
ax[0].set_title("Top 10 correlated features")
full = pd.read_csv("house_prices.csv")
kept = ~full.index.isin(outlier_idx)
ax[1].scatter(full.loc[kept, "GrLivArea"], full.loc[kept, "SalePrice"],
              s=10, alpha=0.5, color="#4C72B0", label="kept")
ax[1].scatter(full.loc[outlier_idx, "GrLivArea"], full.loc[outlier_idx, "SalePrice"],
              s=60, color="#C44E52", marker="x", label="dropped (outliers)")
ax[1].set_xlabel("GrLivArea, sq ft"); ax[1].set_ylabel("SalePrice, $")
ax[1].set_title("GrLivArea vs SalePrice")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/03_correlations_outliers.png", bbox_inches="tight")
plt.close(fig)

# 04 engineered feature: TotalSF vs SalePrice
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
ax[0].scatter(df["TotalSF"], df["SalePrice"], s=10, alpha=0.4, color="#55A868")
ax[0].set_xlabel("TotalSF (engineered)"); ax[0].set_ylabel("SalePrice, $")
ax[0].set_title(f"TotalSF vs SalePrice (r={df['TotalSF'].corr(df['SalePrice']):.3f})")
ax[1].scatter(df["HouseAge"], df["SalePrice"], s=10, alpha=0.4, color="#8172B2")
ax[1].set_xlabel("HouseAge, years (engineered)"); ax[1].set_ylabel("SalePrice, $")
ax[1].set_title(f"HouseAge vs SalePrice (r={df['HouseAge'].corr(df['SalePrice']):.3f})")
fig.tight_layout()
fig.savefig("figures/04_engineered_features.png", bbox_inches="tight")
plt.close(fig)

# 05 model comparison
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
names = list(results.keys())
x = np.arange(len(names))
cv_rmse_means = [results[n]["cv_rmse_log_mean"] for n in names]
cv_rmse_stds = [results[n]["cv_rmse_log_std"] for n in names]
ax[0].bar(names, cv_rmse_means, yerr=cv_rmse_stds, capsize=4, color=["#4C72B0", "#DD8452"])
ax[0].set_ylabel("CV RMSE (log price)")
ax[0].set_title("5-fold CV error (lower is better)")
test_r2 = [results[n]["r2"] for n in names]
ax[1].bar(names, test_r2, color=["#4C72B0", "#DD8452"])
ax[1].set_ylabel("Test R2")
ax[1].set_title("Held-out R2 (higher is better)")
ax[1].set_ylim(0, 1)
fig.tight_layout()
fig.savefig("figures/05_model_comparison.png", bbox_inches="tight")
plt.close(fig)

# 06 diagnostics for the best model: actual vs predicted + residuals
r = results[best_name]
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax[0].scatter(r["actual"], r["pred"], s=10, alpha=0.4, color="#4C72B0")
lims = [min(r["actual"].min(), r["pred"].min()), max(r["actual"].max(), r["pred"].max())]
ax[0].plot(lims, lims, color="grey", ls="--", lw=1)
ax[0].set_xlabel("Actual SalePrice, $"); ax[0].set_ylabel("Predicted SalePrice, $")
ax[0].set_title(f"{best_name}: actual vs predicted (R2={r['r2']:.3f})")
residuals = r["actual"] - r["pred"]
ax[1].scatter(r["pred"], residuals, s=10, alpha=0.4, color="#C44E52")
ax[1].axhline(0, color="grey", ls="--", lw=1)
ax[1].set_xlabel("Predicted SalePrice, $"); ax[1].set_ylabel("Residual, $")
ax[1].set_title("Residuals vs predicted")
fig.tight_layout()
fig.savefig("figures/06_diagnostics.png", bbox_inches="tight")
plt.close(fig)

print("\nFigures written to figures/")

# ----------------------------------------------------------------------
print("\nSUMMARY")
print(f"Dataset: {df.shape[0]} houses (2 outliers dropped), {len(NUMERIC)} numeric + "
      f"{len(CATEGORICAL)} categorical features ({len(ENGINEERED)} engineered)")
for name in names:
    r = results[name]
    print(f"{name:20s} Test R2 {r['r2']:.4f} RMSE ${r['rmse_dollars']:,.0f} "
          f"MAE ${r['mae_dollars']:,.0f} MAPE {r['mape']:.2f}%")
print(f"Best model: {best_name}")

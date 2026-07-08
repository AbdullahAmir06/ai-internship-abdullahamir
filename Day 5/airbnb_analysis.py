"""
PKCERT AI & Software Development Internship - Task 05
Data Preprocessing, Visualization, Feature Engineering, and SQL

Dataset: New York City Airbnb Open Data (2019), from Kaggle
         (dgomonov/new-york-city-airbnb-open-data)

Running this script:
    - cleans the raw CSV (missing values, duplicates, outliers)
    - saves five figures into the figures/ folder
    - builds encoded and standardized feature tables
    - creates a small SQLite database and runs SELECT / WHERE /
      GROUP BY / JOIN queries against it from Python
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # write figures to files, no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

sns.set_theme(style="whitegrid")
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def banner(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


# ======================================================================
# PART A - DATA CLEANING & PREPROCESSING
# ======================================================================
banner("PART A: LOADING AND CLEANING THE DATA")

df = pd.read_csv("AB_NYC_2019.csv")
print("Raw shape:", df.shape)
print("\nMissing values per column:\n", df.isnull().sum())

# --- 1. Missing values ---
# name / host_name: a handful missing, fill with a placeholder
df["name"] = df["name"].fillna("Unknown")
df["host_name"] = df["host_name"].fillna("Unknown")

# reviews_per_month: missing means the listing has never been reviewed,
# so the correct value is 0, not the mean or median
df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

# last_review: a date that only exists for reviewed listings. About a
# fifth are missing and we do not use it in the numeric analysis, so we
# drop the column rather than invent dates for it
df = df.drop(columns=["last_review"])

print("\nMissing values after handling:\n", df.isnull().sum())

# --- 2. Duplicate records ---
dup_rows = df.duplicated().sum()
dup_ids = df.duplicated(subset="id").sum()
print(f"\nFully duplicated rows: {dup_rows}")
print(f"Duplicated listing ids: {dup_ids}")
df = df.drop_duplicates()
df = df.drop_duplicates(subset="id")

# --- 3. Outliers ---
# price: 0 is not a valid nightly rate, so drop those rows first
zero_price = (df["price"] == 0).sum()
print(f"\nListings with price = 0 (invalid): {zero_price}")
df = df[df["price"] > 0]

# use the IQR rule on price to trim the extreme right tail
q1, q3 = df["price"].quantile([0.25, 0.75])
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr
price_outliers = (df["price"] > upper_fence).sum()
print(f"Price IQR upper fence: {upper_fence:.0f}")
print(f"Price outliers above the fence: {price_outliers}")
df = df[df["price"] <= upper_fence]

# minimum_nights above 365 (more than a year) is unrealistic for a
# short-term rental, so treat those as outliers and remove them
long_stay = (df["minimum_nights"] > 365).sum()
print(f"Listings with minimum_nights > 365: {long_stay}")
df = df[df["minimum_nights"] <= 365]

print("\nCleaned shape:", df.shape)
df.to_csv("airbnb_cleaned.csv", index=False)
print("Saved cleaned data to airbnb_cleaned.csv")


# ======================================================================
# PART B - DATA VISUALIZATION
# ======================================================================
banner("PART B: CREATING VISUALIZATIONS")

# 1. Histogram - distribution of nightly price
plt.figure(figsize=(8, 5))
sns.histplot(df["price"], bins=40, kde=True, color="#2a6f97")
plt.title("Distribution of Nightly Price")
plt.xlabel("Price (USD)")
plt.ylabel("Number of listings")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_histogram_price.png", dpi=120)
plt.close()

# 2. Boxplot - price spread by room type
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="room_type", y="price", hue="room_type",
            palette="Set2", legend=False)
plt.title("Price Spread by Room Type")
plt.xlabel("Room type")
plt.ylabel("Price (USD)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_boxplot_price_roomtype.png", dpi=120)
plt.close()

# 3. Correlation heatmap - numeric columns
num_cols = ["price", "minimum_nights", "number_of_reviews",
            "reviews_per_month", "calculated_host_listings_count",
            "availability_365"]
plt.figure(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", center=0, square=True)
plt.title("Correlation Between Numeric Features")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_correlation_heatmap.png", dpi=120)
plt.close()

# 4. Bar chart - average price by borough
plt.figure(figsize=(8, 5))
avg_price = df.groupby("neighbourhood_group")["price"].mean().sort_values()
sns.barplot(x=avg_price.index, y=avg_price.values,
            hue=avg_price.index, palette="viridis", legend=False)
plt.title("Average Price by Borough")
plt.xlabel("Borough")
plt.ylabel("Average price (USD)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_avgprice_by_borough.png", dpi=120)
plt.close()

# 5. Scatter map - listings by location, coloured by borough
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="longitude", y="latitude",
                hue="neighbourhood_group", s=6, alpha=0.5)
plt.title("Listings Across New York City")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title="Borough", markerscale=2, fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_scatter_map.png", dpi=120)
plt.close()

print("Saved 5 figures to the figures/ folder.")


# ======================================================================
# PART C - FEATURE ENGINEERING
# ======================================================================
banner("PART C: FEATURE ENGINEERING")

# 1. Identify categorical features
cat_cols = df.select_dtypes(include="object").columns.tolist()
print("Categorical columns:", cat_cols)

fe = df.copy()

# 2a. Label Encoding - good for high-cardinality columns like neighbourhood
le = LabelEncoder()
fe["neighbourhood_encoded"] = le.fit_transform(fe["neighbourhood"])
print(f"\nLabel encoded 'neighbourhood' into "
      f"{fe['neighbourhood_encoded'].nunique()} integer codes.")

# 2b. One-Hot Encoding - good for low-cardinality columns
fe = pd.get_dummies(fe, columns=["neighbourhood_group", "room_type"],
                    prefix=["borough", "room"])
print("Columns after one-hot encoding:")
print([c for c in fe.columns if c.startswith(("borough_", "room_"))])

# 3. Standardization - put numeric features on the same scale
scaler = StandardScaler()
scaled = scaler.fit_transform(df[num_cols])
scaled_df = pd.DataFrame(scaled, columns=[c + "_scaled" for c in num_cols])
print("\nBefore scaling (means):\n", df[num_cols].mean().round(2).to_dict())
print("\nAfter scaling (means ~0, std ~1):")
print("  means:", scaled_df.mean().round(3).to_dict())
print("  stds :", scaled_df.std().round(3).to_dict())

fe.to_csv("airbnb_features.csv", index=False)
print("\nSaved engineered features to airbnb_features.csv")


# ======================================================================
# PART D - SQL FUNDAMENTALS WITH PYTHON
# ======================================================================
banner("PART D: SQL WITH PYTHON (SQLite)")

conn = sqlite3.connect("airbnb.db")

# main table: the cleaned listings
listings = df[["id", "name", "host_id", "neighbourhood_group",
               "neighbourhood", "room_type", "price",
               "minimum_nights", "number_of_reviews",
               "availability_365"]].copy()
listings.to_sql("listings", conn, if_exists="replace", index=False)

# small dimension table so we have something to JOIN on
boroughs = pd.DataFrame({
    "borough_name": ["Manhattan", "Brooklyn", "Queens",
                     "Bronx", "Staten Island"],
    "borough_code": ["MN", "BK", "QN", "BX", "SI"],
    "is_island":    [1, 0, 0, 0, 1],
})
boroughs.to_sql("boroughs", conn, if_exists="replace", index=False)

# 1. SELECT - a first look at the listings table
print("\n[SELECT] First 5 listings:")
print(pd.read_sql_query(
    "SELECT id, name, neighbourhood_group, price FROM listings LIMIT 5",
    conn).to_string(index=False))

# 2. WHERE - filter with a condition
print("\n[WHERE] Cheap entire homes in Manhattan (price < 100):")
print(pd.read_sql_query(
    """SELECT name, price, room_type
       FROM listings
       WHERE neighbourhood_group = 'Manhattan'
         AND room_type = 'Entire home/apt'
         AND price < 100
       LIMIT 5""", conn).to_string(index=False))

# 3. GROUP BY - aggregate per group
print("\n[GROUP BY] Listing count and average price per room type:")
print(pd.read_sql_query(
    """SELECT room_type,
              COUNT(*)          AS listings,
              ROUND(AVG(price),2) AS avg_price
       FROM listings
       GROUP BY room_type
       ORDER BY avg_price DESC""", conn).to_string(index=False))

# 4. JOIN - combine listings with the borough dimension table
print("\n[JOIN] Average price per borough with its code:")
print(pd.read_sql_query(
    """SELECT b.borough_code,
              l.neighbourhood_group AS borough,
              COUNT(*)            AS listings,
              ROUND(AVG(l.price),2) AS avg_price
       FROM listings l
       JOIN boroughs b
         ON l.neighbourhood_group = b.borough_name
       GROUP BY b.borough_code, l.neighbourhood_group
       ORDER BY avg_price DESC""", conn).to_string(index=False))

conn.close()
print("\nDatabase saved to airbnb.db. All parts complete.")

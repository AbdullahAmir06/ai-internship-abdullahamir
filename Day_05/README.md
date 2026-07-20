# NYC Airbnb: Preprocessing, Visualization, Feature Engineering & SQL

**PKCERT AI & Software Development Internship, Task 05**
Author: Abdullah Amir

A single Python script that takes the raw New York City Airbnb dataset through a
full early pipeline: cleaning, visualization, feature engineering, and loading
the result into a SQLite database that is queried from Python.

## Dataset

- **File:** `AB_NYC_2019.csv` (48,895 listings, 16 columns)
- **Source:** New York City Airbnb Open Data (2019) from Kaggle
  ([dgomonov/new-york-city-airbnb-open-data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)),
  downloaded via a public mirror.
- **Description:** Each row is an Airbnb listing in NYC, with its location,
  room type, price, and review activity.

| Column | Meaning |
| --- | --- |
| `id`, `name` | Listing id and title |
| `host_id`, `host_name` | Host id and name |
| `neighbourhood_group` | Borough (Manhattan, Brooklyn, Queens, Bronx, Staten Island) |
| `neighbourhood` | Specific neighbourhood (219 values) |
| `latitude`, `longitude` | Location coordinates |
| `room_type` | Entire home/apt, Private room, or Shared room |
| `price` | Nightly price in USD |
| `minimum_nights` | Minimum nights required |
| `number_of_reviews`, `reviews_per_month` | Review activity |
| `last_review` | Date of the last review |
| `calculated_host_listings_count` | How many listings the host has |
| `availability_365` | Days available in a year |

## Part A: Data Cleaning and Preprocessing

- **Missing values:** `name` and `host_name` (a few each) filled with "Unknown";
  `reviews_per_month` filled with 0 (missing means the listing was never
  reviewed); `last_review` dropped (10,052 missing and not needed for the
  numeric analysis).
- **Duplicates:** checked both full rows and repeated `id` values. None were
  found, but the drop calls stay in the pipeline for safety.
- **Outliers:** removed 11 listings priced at 0, trimmed the price right tail
  using the IQR rule (upper fence about 334, which removed 2,972 listings), and
  removed 13 listings asking for a minimum stay over a year.

The dataset went from 48,895 rows to a clean **45,899 rows**, saved to
`airbnb_cleaned.csv`.

## Part B: Data Visualization

Five figures are saved to the `figures/` folder:

1. **Histogram** of price: strongly right skewed, with hosts favouring round
   numbers like 100 and 150.
2. **Boxplot** of price by room type: entire homes are the most expensive and
   widest spread, shared rooms the cheapest.
3. **Correlation heatmap** of numeric features: the strongest pair is reviews vs
   reviews per month (0.59); price barely correlates with any numeric column.
4. **Bar chart** of average price by borough: Manhattan is highest (~$146), the
   Bronx lowest (~$77).
5. **Scatter map** of longitude and latitude coloured by borough: redraws the
   map of NYC and shows how the boroughs cluster.

## Part C: Feature Engineering

- **Label Encoding** on `neighbourhood` (219 values into integer codes), which
  is compact for a high-cardinality column.
- **One-Hot Encoding** on `neighbourhood_group` and `room_type`, giving 0/1
  columns like `borough_Manhattan` and `room_Private room` with no false
  ordering.
- **Standardization** of the numeric columns with `StandardScaler` so each has
  mean 0 and standard deviation 1, which matters for distance-based and
  gradient-based models. Saved to `airbnb_features.csv`.

## Part D: SQL Fundamentals with Python

Using the built-in `sqlite3` module, the cleaned data is loaded into
`airbnb.db` as a `listings` table, plus a small `boroughs` dimension table to
join against. Four query types are run and printed from Python:

- **SELECT** the first few listings.
- **WHERE** to filter (cheap entire homes in Manhattan under $100).
- **GROUP BY** room type for count and average price.
- **JOIN** listings to boroughs to attach a borough code to each aggregate.

## How to Run

```bash
# 1. (optional) create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows

# 2. install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# 3. run everything (cleaning, figures, features, and SQL)
python airbnb_analysis.py
```

Running the script regenerates the cleaned CSV, the feature CSV, the five
figures, and the SQLite database.

## Files

| File | Description |
| --- | --- |
| `AB_NYC_2019.csv` | The raw dataset |
| `airbnb_analysis.py` | The full pipeline (Parts A to D) |
| `airbnb_cleaned.csv` | Cleaned data from Part A |
| `airbnb_features.csv` | Encoded and scaled features from Part C |
| `airbnb.db` | SQLite database from Part D |
| `figures/` | The five generated visualizations |
| `Report.pdf` / `Report.tex` | The full written report |
| `README.md` | This file |

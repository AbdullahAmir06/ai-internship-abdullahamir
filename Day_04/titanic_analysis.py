"""
Titanic Dataset - Exploratory Data Analysis
PKCERT AI & Software Development Internship - Task 04 (Part C)
Author: Abdullah Amir

Reads the Titanic dataset, cleans missing values, generates summary
statistics, and performs basic exploratory data analysis with Pandas.
"""

import pandas as pd


def load_data(path="titanic.csv"):
    """Read the dataset and take a first look at it."""
    df = pd.read_csv(path)

    print("=" * 60)
    print("RAW DATA OVERVIEW")
    print("=" * 60)
    print(f"Shape (rows, cols): {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumn info:")
    print(df.info())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    return df


def clean_data(df):
    """Handle the missing values column by column."""
    print("\n" + "=" * 60)
    print("CLEANING MISSING VALUES")
    print("=" * 60)

    # Age: fill gaps with the median (robust to outliers)
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Embarked: only a couple missing, so use the most common port
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Cabin: too many missing to be useful, so drop the whole column
    df = df.drop(columns=["Cabin"])

    print("Missing values after cleaning:")
    print(df.isnull().sum())
    return df


def summary_statistics(df):
    """Print summary statistics for the cleaned dataset."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(df.describe())


def exploratory_analysis(df):
    """Explore the drivers of survival."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print(f"\nOverall survival rate: {df['Survived'].mean():.2%}")

    print("\nSurvival rate by passenger class:")
    print(df.groupby("Pclass")["Survived"].mean())

    print("\nSurvival rate by port of embarkation:")
    print(df.groupby("Embarked")["Survived"].mean())

    print("\nAverage fare paid per class:")
    print(df.groupby("Pclass")["Fare"].mean())

    # Age groups: survival by life stage
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teen", "Young Adult", "Adult", "Senior"],
    )
    print("\nSurvival rate by age group:")
    print(df.groupby("AgeGroup", observed=True)["Survived"].mean())

    print("\nSurvival rate by class and age group:")
    print(
        df.groupby(["Pclass", "AgeGroup"], observed=True)["Survived"]
        .mean()
        .unstack()
    )


def main():
    df = load_data()
    df = clean_data(df)
    summary_statistics(df)
    exploratory_analysis(df)
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()

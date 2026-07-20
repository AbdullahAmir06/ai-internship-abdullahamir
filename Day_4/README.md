# Titanic Dataset — Exploratory Data Analysis

**PKCERT AI & Software Development Internship — Task 04 (Part C)**
Author: Abdullah Amir

A small Pandas application that reads the Titanic passenger dataset, cleans its
missing values, generates summary statistics, and performs basic exploratory
data analysis (EDA).

## Dataset

- **File:** `titanic.csv` (891 passengers, 12 columns)
- **Source:** [datasciencedojo/datasets](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv) (public)
- **Description:** Each row is a passenger on the RMS Titanic. The target column
  `Survived` records whether they survived (1) or not (0).

| Column | Meaning |
| --- | --- |
| `PassengerId` | Unique passenger ID |
| `Survived` | 0 = did not survive, 1 = survived |
| `Pclass` | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| `Name` | Passenger name |
| `Sex` | Passenger sex |
| `Age` | Age in years |
| `SibSp` | Number of siblings / spouses aboard |
| `Parch` | Number of parents / children aboard |
| `Ticket` | Ticket number |
| `Fare` | Fare paid |
| `Cabin` | Cabin number |
| `Embarked` | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

## Data Cleaning

The raw data has missing values in three columns, each handled differently:

- **`Age`** — filled with the **median** age. The median is used instead of the
  mean because a few extreme ages would skew the mean, while the median stays
  stable.
- **`Embarked`** — only two values missing, filled with the **mode** (the most
  common port).
- **`Cabin`** — missing for the large majority of passengers, so the whole
  column is **dropped** rather than guessed at.

After cleaning there are no missing values left.

## Analysis Performed

Using `groupby` aggregations, the script explores:

- Overall survival rate
- Survival rate by passenger class (`Pclass`)
- Survival rate by port of embarkation (`Embarked`)
- Average fare paid per class
- Survival rate by age group (Child / Teen / Young Adult / Adult / Senior)
- Survival rate broken down by class **and** age group together

## Key Findings

- Only about **38%** of passengers survived overall.
- **Passenger class was a major factor.** First-class passengers survived at
  ~63%, second class at ~47%, and third class at just ~24%.
- **Fare tracks class closely** — first-class passengers paid far more on
  average (~£84) than third-class (~£14), reflecting better-located cabins and
  closer access to lifeboats.
- **Children had the best odds by age** (~58% survival), while seniors had the
  worst (~23%).
- Combining class and age shows the effects stacking: a first- or second-class
  child was very likely to survive, whereas an older third-class passenger was
  among the least likely.

## How to Run

```bash
# 1. (optional) create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows

# 2. install the one dependency
pip install pandas

# 3. run the analysis
python titanic_analysis.py
```

## Files

| File | Description |
| --- | --- |
| `titanic.csv` | The dataset |
| `titanic_analysis.py` | The EDA application |
| `README.md` | This file |

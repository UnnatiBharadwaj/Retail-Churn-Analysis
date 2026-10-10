# Retail Customer Churn Analysis

End-to-end churn analysis on a **synthetic** retail customer dataset: data cleaning, exploratory analysis, SQL, and a churn classification model.

> **Data note:** The dataset is synthetic. It was generated for practice and is not real company data. Its messy values (duplicates, missing values, invalid ages, inconsistent labels, negative prices) were added on purpose so the cleaning steps could be practiced.

## Problem
Which customers are likely to churn, and what patterns separate churned customers from active ones?

## Dataset
- `data/raw/retail_customer_churn_raw_dataset.xlsx`: 5,060 customer rows (60 duplicates), 73,913 order rows
- After cleaning: **5,000 customers, 73,873 orders**
- Snapshot date: 30 Sep 2026. Amounts in INR.

## Tools
Python (pandas, NumPy, Matplotlib, Seaborn, scikit-learn), SQL (SQLite), Jupyter-style scripts. Power BI dashboard: *not yet built*.

## Cleaning steps
| Issue | Count | Action |
|---|---|---|
| Duplicate customer rows | 60 | Removed (kept first) |
| Age outside 18–70 or missing | 190 | Set to median age |
| Missing city | 100 | Filled with "Unknown" |
| Inconsistent gender labels | — | Standardised to Male / Female / Other |
| Missing NPS score | 2,220 | Left as missing (not imputed; not used in the final model) |
| Negative unit price | 25 | Dropped (order rows) |
| Zero or invalid quantity | 15 | Dropped (order rows) |
| Missing discount / payment method | — | Filled with 0 / "Unknown" |

Run `notebooks/01_clean.py` to reproduce.

## Key findings
- **Overall churn rate: 30.4%** (1,519 of 5,000 customers)
- **Membership tier matters most:** Basic 38.1%, Silver 24.4%, Gold 13.2%
- **Channel:** Online-preferred customers churned at 36.4%, vs 23.9% for Mobile App users
- **Email opt-in:** customers who did not opt in churned at 36.0%, vs 26.9% for those who did
- **Returns:** churned customers returned 18.3% of their orders, vs 7.0% for active customers
- **Engagement (descriptive only):** churned customers averaged 9.1 app logins in the last 90 days vs 26.0 for active customers. These values are measured close to churn, so they describe the churned group but cannot be used to predict churn (see below).

## Churn model
**Features used:** age, membership tier, preferred channel, email opt-in, gender. These are known before a customer churns.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (predict "no churn" for everyone) | 0.696 | — | 0.000 | — | 0.500 |
| Logistic Regression | 0.691 | 0.463 | 0.102 | 0.167 | 0.648 |
| Random Forest | 0.694 | 0.474 | 0.059 | 0.105 | 0.634 |

Models were tested on a stratified 20% holdout (random_state = 42).

**Interpretation:** the models are only modestly better than random guessing (AUC 0.63–0.65), and they catch only a small share of churners (recall 6–10%). Demographic and plan data carry some signal, but they don't identify churners well on their own. Accuracy is not a useful measure here, because 70% of customers stay, so a model that predicts "no churn" for everyone is already 70% accurate.

### Leakage check (important)
An earlier version of the model included app logins, support tickets, and NPS. It reached 99% accuracy and AUC 1.0. That result was **leakage**, not skill: these values are recorded during or after a customer disengages, so they reveal the outcome. They were dropped from the final model. In a real project, features must be measured before the outcome, and a result that looks this good should be checked for leakage first.

## Limitations
- Synthetic data: patterns reflect how the data was generated, not real customer behaviour.
- Order-history features (order count, spend, returns) are also partly affected by how long a customer stayed, so they were excluded from the final model.
- No time-based split. A production model would train on earlier periods and test on later ones.

## How to run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
python notebooks/01_clean.py
python notebooks/02_analysis_model.py
```
SQL queries are in `sql/churn_analysis.sql` (load the cleaned CSVs into SQLite first).

## Project structure
```
data/raw/        original workbook (synthetic)
data/clean/      cleaned customers and orders CSVs
notebooks/       cleaning and analysis scripts
sql/             SQL analysis queries
images/          charts
dashboard/       Power BI file (to add)
```


# Mortgage Denial Prediction Using Rhode Island HMDA Data

**DATA 1030 Project**

## Project Overview

This project aims to predict whether a home-purchase mortgage application in Rhode Island will be denied, using Home Mortgage Disclosure Act (HMDA) data from 2018 to 2024.

We plan to compare machine learning models, identify features that contribute most to their predictions, and evaluate how well models trained on earlier years perform on later applications.

Why Rhode Island? RI is the community we live in; a state-level model reflects local lenders, prices, and neighborhoods.

Intended users: lenders for second-look review, regulators and fair-housing groups for auditing, and housing counselors for pre-application checks.

Key results: [to be completed]

## Dataset

We use publicly available HMDA data from the [Consumer Financial Protection Bureau (CFPB)](https://ffiec.cfpb.gov/data-browser/)

Scope: state = RI, years = 2018 - 2024, loan_type = 1, action_taken: 1, 2, 3

After filtering for eligible application outcomes, the dataset contains **104,779 records**.

The binary target variable, `Denied`, is defined as:

- **0 (Not denied):** Loan originated (`action_taken = 1`) or application approved but not accepted (`action_taken = 2`).
- **1 (Denied):** Application denied (`action_taken = 3`).

The dataset contains 94,870 not-denied applications (90.54%) and 9,909 denied applications (9.46%).

How to get it: run the API call in Data_Process.py

## Repository Structure

```text
├── data/
│   ├── raw/            # downloaded HMDA files
│   └── processed/      # cleaned data
├── figures/            # saved plots
├── reports/            # midterm & final reports
├── results/            # model outputs, metrics
├── src/                # python scripts (download, preprocessing, modeling)
├── .gitignore
├── LICENSE
└── README.md
```

## Exploratory Data Analysis (I don't know whether we need this part here? - XZ)

Our exploratory analysis identified three findings that guide the modeling plan.

**1. Debt-to-income ratio (DTI)**

DTI has a strong, nonlinear relationship with mortgage denial. Denial rates are approximately 5–7% for applications with DTI between 20% and 50%, compared with 15% for DTI between 50% and 60% and 79% for DTI above 60%.

**2. Missing values**

Missing values are associated with higher denial rates. Applications without a reported property value have a denial rate of 50%, compared with 8.5% when the value is reported. We plan to compare models with and without missing-value indicators, since missingness may partly reflect the lending decision itself.

**3. Changes across years**

Applicant and loan characteristics changed across years, while overall denial rates remained relatively stable. The median loan amount increased from $275,000 in the training years to $375,000 in 2024.

These changes motivate evaluating how well models generalize to later applications.

## Environment

The current development environment uses Python 3.14.6, with pandas 3.0.3, NumPy 2.4.6, Matplotlib 3.11.0, Seaborn 0.13.2, scikit-learn 1.9.0, and requests 2.34.2.

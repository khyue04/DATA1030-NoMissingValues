# DATA1030-NoMissingValues
Project Overview

This project aims to predict whether a home-purchase mortgage application in Rhode Island will be denied, using Home Mortgage Disclosure Act (HMDA) data from 2018 to 2024.

We plan to compare machine learning models, identify features that contribute most to their predictions, and evaluate how well models trained on earlier years perform on later applications.

Dataset

We use publicly available HMDA data from the Consumer Financial Protection Bureau (CFPB), covering home-purchase mortgage applications in Rhode Island from 2018 to 2024.

After filtering for eligible application outcomes, the dataset contains 104,779 records.

The binary target variable, Denied, is defined as:

0 (Not denied): Loan originated (action_taken = 1) or application approved but not accepted (action_taken = 2).

1 (Denied): Application denied (action_taken = 3).

The dataset contains 94,870 not-denied applications (90.54%) and 9,909 denied applications (9.46%).

Exploratory Data Analysis

Our exploratory analysis identified three findings that guide the modeling plan.

1. Debt-to-income ratio (DTI)

DTI has a strong, nonlinear relationship with mortgage denial. Denial rates are approximately 5–7% for applications with DTI between 20% and 50%, compared with 15% for DTI between 50% and 60% and 79% for DTI above 60%.

2. Missing values

Missing values are associated with higher denial rates. Applications without a reported property value have a denial rate of 50%, compared with 8.5% when the value is reported. We plan to compare models with and without missing-value indicators, since missingness may partly reflect the lending decision itself.

3. Changes across years

Applicant and loan characteristics changed across years, while overall denial rates remained relatively stable. The median loan amount increased from $275,000 in the training years to $375,000 in 2024.

These changes motivate evaluating how well models generalize to later applications.

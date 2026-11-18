"""Missing values: feature types, imputer and summary table.

How to use:
    train = read_split('../data/hmda_train.csv')
    LR, RF, KNN: ('cont', make_pipeline(continuous_imputer(random_state), StandardScaler()), continuous_ftrs)
    XGBoost: no imputer, XGBoost handles NaN.
"""

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # needed to import IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


continuous_ftrs = ['loan_amount', 'income', 'loan_to_value_ratio', 'property_value', 'loan_term',
                   'tract_population', 'tract_to_msa_income_percentage', 'tract_owner_occupied_units',
                   'tract_one_to_four_family_homes', 'tract_median_age_of_housing_units']

ordinal_ftrs = ['debt_to_income_ratio', 'total_units']

categorical_ftrs = ['loan_type', 'occupancy_type', 'construction_method', 'county_code', 'lien_status',
                    'conforming_loan_limit', 'open-end_line_of_credit', 'business_or_commercial_purpose',
                    'interest_only_payment', 'balloon_payment', 'submission_of_application', 'aus-1',
                    'applicant_credit_score_type', 'co-applicant_credit_score_type', 'lei']

# order of the ordinal features, low to high
dti_levels = ['<20%', '20%-<30%', '30%-<36%'] + [str(i) for i in range(36, 50)] + ['50%-60%', '>60%']
units_levels = ['1', '2', '3', '4', '5-24', '25-49', '50-99', '100-149', '>149']

# NA is a missing value; in categorical and ordinal features it is kept as its own category.
# Exempt and 1111 are kept as categories too.


def read_split(path):
    # read a split file: keep the codes as text, read continuous features as numbers (blank cells become NaN)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in continuous_ftrs:
        df[col] = pd.to_numeric(df[col].replace(['', 'NA', 'Exempt'], np.nan))
    df['denied'] = df['denied'].astype(int)
    return df


def dti_categories(missing_position):
    # DTI levels in order, with NA and Exempt put first or last
    if missing_position == 'first':
        return ['NA', 'Exempt'] + dti_levels
    return dti_levels + ['NA', 'Exempt']


def continuous_imputer(random_state):
    # fill missing continuous values using the other continuous features
    # also add a 0/1 column for each feature that had missing values
    return IterativeImputer(estimator=RandomForestRegressor(n_estimators=1, random_state=random_state),
                            random_state=random_state, add_indicator=True, skip_complete=True)


def missing_audit(df):
    # one row per feature with missing values
    # continuous features: blank cells (NA or Exempt in the raw data) count as missing; other features: only NA
    rows = []
    for col in continuous_ftrs + ordinal_ftrs + categorical_ftrs:
        if col in continuous_ftrs:
            col_type = 'continuous'
            missing = df[col].isna()
        else:
            col_type = 'ordinal' if col in ordinal_ftrs else 'categorical'
            missing = (
                df[col].isna()
                | df[col].astype("string").str.strip().isin(["", "NA"]).fillna(False)
            )
        if missing.sum() == 0:
            continue
        rows.append({'feature': col,
                     'type': col_type,
                     'missing_count': missing.sum(),
                     'missing_percent': round(missing.mean() * 100, 2),
                     'denial_rate_missing': round(df.loc[missing, 'denied'].mean(), 3),
                     'denial_rate_not_missing': round(df.loc[~missing, 'denied'].mean(), 3)})
    return pd.DataFrame(rows)

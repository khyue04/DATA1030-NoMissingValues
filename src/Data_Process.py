import numpy as np
import pandas as pd
import requests as rq
from pathlib import Path

url = 'https://ffiec.cfpb.gov/v2/data-browser-api/view/csv'

yearly_data = []
data_dir = Path(__file__).resolve().parent.parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)

for year in range(2018, 2025):
    csv_path = data_dir / f"hmda_{year}_RI.csv"

    if csv_path.exists():
        print(f"{year}: loading existing file: {csv_path.name}")

    else:
        params = {
            "years": str(year),
            "states": "RI",
            "loan_purposes": "1"
        }

        print(f"{year}: downloading...")

        response = rq.get(
            url,
            params=params,
            timeout=180
        )

        if response.status_code == 403:
            raise RuntimeError(
                f"The HMDA server refused the {year} download (HTTP 403). "
                f"Download that year's Rhode Island home-purchase CSV "
                f"through the HMDA Data Browser and save it as:\n{csv_path}"
            )

        response.raise_for_status()

        # Avoid saving an HTML error page as a CSV.
        if "text/html" in response.headers.get("Content-Type", "").lower():
            raise RuntimeError(
                f"{year}: the server returned HTML instead of CSV data."
            )

        csv_path.write_bytes(response.content)
        print("Saved to:", csv_path)

    raw_df = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False
    )

    # Verify that the file matches the requested data.
    required_columns = {"activity_year", "state_code", "loan_purpose"}
    missing = required_columns - set(raw_df.columns)

    if missing:
        raise ValueError(
            f"{csv_path.name} is missing columns: {sorted(missing)}"
        )

    if raw_df.empty:
        raise ValueError(f"{csv_path.name} contains no observations.")

    if not (
        raw_df["activity_year"].eq(str(year)).all()
        and raw_df["state_code"].eq("RI").all()
        and raw_df["loan_purpose"].eq("1").all()
    ):
        raise ValueError(
            f"{csv_path.name} must contain only {year} "
            "Rhode Island home-purchase applications."
        )

    yearly_data.append(raw_df)

# Combine the years into one DataFrame, with a unique row index
raw_df = pd.concat(yearly_data, ignore_index = True)

print('Rows and columns:', raw_df.shape)
print('Column name:', raw_df.columns.tolist())

needed_columns = [
    "activity_year",  
    "action_taken",
    "loan_amount",
    "income",
    "debt_to_income_ratio",
    "loan_to_value_ratio",
    "property_value",
    "loan_type",
    "occupancy_type",
    "construction_method",
    "total_units",
    "loan_term",
    "county_code",
    "lien_status",
    "conforming_loan_limit",
    "open-end_line_of_credit",
    "business_or_commercial_purpose",
    "interest_only_payment",
    "balloon_payment",
    "submission_of_application",
    "aus-1",
    "applicant_credit_score_type",
    "co-applicant_credit_score_type",
    "tract_population",
    "tract_to_msa_income_percentage",
    "tract_owner_occupied_units",
    "tract_one_to_four_family_homes",
    "tract_median_age_of_housing_units",
    "lei"
]

# Check for missing columns
missing_columns = set(needed_columns) - set(raw_df.columns)
print('Missing columns:', missing_columns)

if missing_columns:
    raise ValueError('The following columns are missing from the dataset: {}'.format(missing_columns))

# Check that the API returned all seven requested years.
expected_years = {"2018", "2019", "2020", "2021", "2022", "2023", "2024"}
if set(raw_df["activity_year"].unique()) != expected_years:
    raise ValueError("The downloaded CSV does not contain all requested years from 2018 to 2024.")

# Keep only the selected columns
df = raw_df[list(needed_columns)].copy()
print(df.columns)

# Keep actions 1, 2, and 3
df = df.loc[df["action_taken"].isin(["1", "2", "3"])].copy()

# Create the target: 1 = denied, 0 = not denied
df["denied"] = (df["action_taken"] == "3").astype(int) # Equal to "3" gives True -> 1; otherwise False -> 0.

print("Eligible dataset shape:", df.shape)

print("\nTarget counts:")
print(df["denied"].value_counts().sort_index())

print(f"\nDenial rate: {df['denied'].mean():.2%}") # Results in 2 decimal places and percentage format

# Containing only the features uses for prediction, excluding the target and action_taken
feature_df = df.drop(columns=['action_taken', 'denied'])

# Check for missing values, the result is true if any of the conditions is met, otherwise false
unavilable = (
    feature_df.isna()
    | feature_df.isin(["", "NA", "Exempt"])
)

missing_summary = pd.DataFrame({
    'unavailable_count': unavilable.sum(),
    'unavailable_percent': unavilable.mean() * 100 # The proportion of True values 
})

print(
    missing_summary.sort_values(
        "unavailable_percent",
        ascending = False # Highest percentage of missing values first
    )
)

numeric_features = [
    'activity_year',
    'loan_amount',
    'income',
    'loan_to_value_ratio',
    'property_value',
    'loan_term',
    'tract_population',
    'tract_to_msa_income_percentage',
    'tract_owner_occupied_units',
    'tract_one_to_four_family_homes',
    'tract_median_age_of_housing_units'
]

categorical_features = [
    "debt_to_income_ratio",
    "loan_type",
    "occupancy_type",
    "construction_method",
    "total_units",
    "county_code",
    "lien_status",
    "conforming_loan_limit",
    "open-end_line_of_credit",
    "business_or_commercial_purpose",
    "interest_only_payment",
    "balloon_payment",
    "submission_of_application",
    "aus-1",
    "applicant_credit_score_type",
    "co-applicant_credit_score_type",
    "lei"
]

for column in numeric_features:
    df[column] = pd.to_numeric(
        df[column].replace(["", "NA", "Exempt"], np.nan), # Turn empty strings, "NA", and "Exempt" into NaN
        errors="raise"
    )

# Check the numerical data types and summary statistics
print("\nNumerical data types:")
print(df[numeric_features].dtypes)

print("\nNumerical summary:")
print(df[numeric_features].describe())

# Check the categorical data types and summary statistics
print("\nCategorical data types:")
print(df[categorical_features].dtypes)

print("\nCategorical summary:")
print(df[categorical_features].describe(include="all"))

# The predictors are the numeric and categorical features, and the target is denied
X = df[numeric_features + categorical_features].copy()
y = df['denied'].copy()

# Select the rows for each period
train_rows = df['activity_year'].between(2018, 2023)
test_rows = df['activity_year'] == 2024

# Use all eligible 2018-2023 applications for training
X_train = X.loc[train_rows].copy()
y_train = y.loc[train_rows].copy()

# Use all eligible 2024 applications for testing
X_test = X.loc[test_rows].copy()
y_test = y.loc[test_rows].copy()

for name, X_part, y_part in [
    ("Training (2018-2023)", X_train, y_train),
    ("Test (2024)", X_test, y_test)
]:
    print(
        f"{name}: {len(X_part)} rows, "
        f"denial rate: {y_part.mean():.2%}"
    )  # Check the rows and denial rate for each split

for name, X_part, y_part in [
    ("train", X_train, y_train),
    ("test", X_test, y_test)
]:
    split_df = X_part.copy()
    split_df["denied"] = y_part

    output_path = data_dir / f"hmda_{name}.csv"
    split_df.to_csv(output_path, index=False)

    print("Saved:", output_path)

# Save the missing-value summary
missing_summary.to_csv(
    data_dir / "hmda_missing_summary.csv",
    index_label="feature"
)
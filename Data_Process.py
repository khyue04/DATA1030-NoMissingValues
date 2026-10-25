import numpy as np
import pandas as pd
import requests as rq
from pathlib import Path
from sklearn.model_selection import train_test_split

url = 'https://ffiec.cfpb.gov/v2/data-browser-api/view/csv'

yearly_data = []

for year in range(2018, 2025):
    params = {
        'years': str(year),
        'states': 'RI',
        'loan_purposes': '1' # Home purchase
    }

    results = rq.get(
        url,
        params = params
    )
    print(year, results)

    # Check whether sucessfully requested, and saved as csv file
    csv_path = Path(__file__).resolve().parent / f"hmda_{year}_RI.csv" # Build the path to the csv file in the same directory as this script

    if results.status_code == 200:
        with open(csv_path, 'wb') as file: # Write binary to avoid encoding issues
            file.write(results.content)

            print('Saved to:', csv_path)

    else:
        print('Response code:', results.status_code)
        print('Error message:', results.content)
        raise ValueError('The API request was unsucessful!')

    # Load csv file and check the year
    raw_df = pd.read_csv(
        csv_path,
        dtype = str,
        keep_default_na = False # Prevent empty strings from being convert to NaN
    )

    if raw_df.empty or not raw_df["activity_year"].eq(str(year)).all():
        raise ValueError(f"The downloaded CSV does not contain the expected {year} data.")

    yearly_data.append(raw_df)

# Combine the years into one DataFrame, with a unique row index
raw_df = pd.concat(yearly_data, ignore_index = True)

print('Rows and columns:', raw_df.shape)
print('Column name:', raw_df.columns.tolist())

needed_columns = [
    "activity_year", # Retain for splitting; exclude from predictors
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
    "loan_term"
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
feature_df = df.drop(columns = ['activity_year', 'action_taken', 'denied'])

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
    'loan_amount',
    'income',
    'loan_to_value_ratio',
    'property_value',
    'loan_term'
]

categorical_features = [
    "debt_to_income_ratio",
    "loan_type",
    "occupancy_type",
    "construction_method",
    "total_units"
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

from sklearn.model_selection import train_test_split

# The predictors are the numeric and categorical features, and the target is denied
X = df[numeric_features + categorical_features].copy()
y = df['denied'].copy()

# Select the rows for each period. activity_year is a string column
historical_rows = df['activity_year'].isin(['2018', '2019', '2020', '2021', '2022'])
validation_rows = df['activity_year'] == '2023'
test_2024_rows = df['activity_year'] == '2024'

# Research 8-% of 2018-2022 as training, test hold the remaining 20%
X_train, X_test, y_train, y_test = train_test_split(
    X.loc[historical_rows], y.loc[historical_rows],
    train_size = 0.80,
    random_state = 42,
    stratify = y.loc[historical_rows]
)

# Use all eligible 2023 applications for validation
X_val = X.loc[validation_rows].copy()
y_val = y.loc[validation_rows].copy()

# Use all eligible 2024 applications for the second test set
X_test_2024 = X.loc[test_2024_rows].copy()
y_test_2024 = y.loc[test_2024_rows].copy()

for name, X_part, y_part in [
    ("Training", X_train, y_train),
    ("Validation (2023)", X_val, y_val),
    ("Historical test (2018-2022)", X_test, y_test),
    ("Later-period test (2024)", X_test_2024, y_test_2024)
]:
    print(
        f"{name}: {len(X_part)} rows, "
        f"denial rate: {y_part.mean():.2%}"
    ) # Check the rows and denial rate for each split 

for name, X_part, y_part in [
    ("train", X_train, y_train),
    ("validation", X_val, y_val),
    ("test", X_test, y_test),
    ("test_2024", X_test_2024, y_test_2024)
]:
    split_df = X_part.copy()
    split_df["denied"] = y_part

    output_path = csv_path.parent / f"hmda_{name}.csv"
    split_df.to_csv(output_path, index=False)

    print("Saved:", output_path)

# Save the missing-value summary
missing_summary.to_csv(
    csv_path.parent / "hmda_missing_summary.csv",
    index_label="feature"
)
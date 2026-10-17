
import pandas as pd
from pathlib import Path

# 1. Set project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    BASE_DIR
    / "data"
    / "Instagram_Analytics_200_CLEANED_2026_SYNTHETIC.csv"
)
OUTPUT_PATH = BASE_DIR / "data" / "processed_instagram_data.csv"

# 2. Load the dataset
df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)
print("\nOriginal columns:")
print(df.columns.tolist())

# 3. Remove duplicate rows
df = df.drop_duplicates().copy()

# 4. Clean the target column
# 1 = CTA present, 0 = CTA absent
target_map = {
    "true": 1,
    "yes": 1,
    "1": 1,
    "false": 0,
    "no": 0,
    "0": 0
}

df["has_call_to_action"] = (
    df["has_call_to_action"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(target_map)
)

# 5. Convert posting date/time into useful features
if "post_datetime" in df.columns:
    post_time = pd.to_datetime(
        df["post_datetime"], errors="coerce"
    )
else:
    post_time = pd.to_datetime(
        df["post_date"], errors="coerce"
    )

df["posting_hour"] = post_time.dt.hour
df["posting_day"] = post_time.dt.day_name()

# 6. Remove rows with missing target labels
df = df.dropna(subset=["has_call_to_action"]).copy()
df["has_call_to_action"] = (
    df["has_call_to_action"].astype(int)
)

# 7. Remove identifiers and raw date columns
columns_to_drop = [
    "post_id",
    "account_id",
    "post_datetime",
    "post_date"
]

df = df.drop(
    columns=columns_to_drop,
    errors="ignore"
)

# 8. Check the cleaned dataset
print("\nCleaned dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

print("\nCTA class distribution:")
print(df["has_call_to_action"].value_counts())

print("\nCTA class percentages:")
print(
    df["has_call_to_action"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# 9. Save the prepared dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\nPrepared dataset saved to:")
print(OUTPUT_PATH)
print("\nDay 3 data preparation completed!")

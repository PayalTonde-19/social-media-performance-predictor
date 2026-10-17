
import pandas as pd

# Load the dataset
df = pd.read_csv(
    "data/Instagram_Analytics_200_CLEANED_2026_SYNTHETIC.csv"
)

# Display the first 5 rows
print("\nFIRST 5 ROWS:")
print(df.head())

# Display dataset information
print("\nDATASET INFORMATION:")
df.info()

# Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDUPLICATE ROWS:")
print(df.duplicated().sum())

# Display column names
print("\nCOLUMN NAMES:")
print(df.columns.tolist())

# Display number of rows and columns
print("\nDATASET SHAPE:")
print(df.shape)


# Check data types
print("\nDATA TYPES:")
print(df.dtypes)

# Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Check unique values in important columns
print("\nMEDIA TYPES:")
print(df["media_type"].value_counts())

print("\nACCOUNT TYPES:")
print(df["account_type"].value_counts())

print("\nCONTENT CATEGORIES:")
print(df["content_category"].value_counts())

# Summary of numerical columns
print("\nNUMERICAL SUMMARY:")
print(df.describe())


# Final dataset quality report
print("\n" + "=" * 40)
print("DATASET QUALITY REPORT")
print("=" * 40)

print("Total rows:", df.shape[0])
print("Total columns:", df.shape[1])
print("Total missing values:", df.isnull().sum().sum())
print("Total duplicate rows:", df.duplicated().sum())

print("\nColumn-wise missing values:")
print(df.isnull().sum())

print("\nUnique values in each column:")
print(df.nunique())

print("\nDataset analysis completed!")


from pathlib import Path
import pandas as pd

# Find the project folder
project_folder = Path(__file__).parent
data_folder = project_folder / "data"

# Find all CSV files, including nested folders
csv_files = list(data_folder.rglob("*.csv"))

print("CSV files found:")

for file in csv_files:
    print(file, "-", file.stat().st_size, "bytes")

# Load the dataset automatically
if not csv_files:
    print("ERROR: No CSV files found in the data folder.")

else:
    file_path = next(
        (f for f in csv_files
         if "Instagram_Analytics_200_CLEANED_2026_SYNTHETIC" in f.name),
        None
    )

    if file_path is None:
        print("ERROR: The expected dataset was not found.")
    elif file_path.stat().st_size == 0:
        print("ERROR: The dataset file is empty.")
    else:
        df = pd.read_csv(file_path)

        print("\nDataset loaded successfully!")
        print("Dataset shape:", df.shape)

        print("\nColumn names:")
        print(df.columns.tolist())

        print("\nFirst five rows:")
        print(df.head())

        print("\nMissing values:")
        print(df.isnull().sum())

        print("\nDuplicate rows:", df.duplicated().sum())

        print("\nDataset information:")
        df.info()


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Load the dataset
DATA_PATH = (
    Path(__file__).parent
    / "data"
    / "Instagram_Analytics_200_CLEANED_2026_SYNTHETIC.csv"
)

df = pd.read_csv(DATA_PATH)

print("=" * 45)
print("SOCIAL MEDIA PERFORMANCE PREDICTOR")
print("DAY 2: EXPLORATORY DATA ANALYSIS")
print("=" * 45)

# 2. Understand the dataset
print("\n1. DATASET SHAPE")
print("Rows and columns:", df.shape)

print("\n2. COLUMN NAMES")
print(df.columns.tolist())

print("\n3. DATA TYPES")
print(df.dtypes)

# 3. Check data quality
print("\n4. MISSING VALUES")
print(df.isnull().sum())

print("\n5. DUPLICATE ROWS")
print(df.duplicated().sum())

# 4. Numerical summary
print("\n6. NUMERICAL SUMMARY")
print(df.describe())

# 5. Create a folder for graphs
output_dir = Path(__file__).parent / "outputs"
output_dir.mkdir(exist_ok=True)

# 6. Graph: number of posts by media type
media_counts = df["media_type"].value_counts()

plt.figure(figsize=(8, 5))
media_counts.plot(kind="bar")
plt.title("Number of Posts by Media Type")
plt.xlabel("Media Type")
plt.ylabel("Number of Posts")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(output_dir / "media_type_distribution.png")
plt.close()

# 7. Graph: posts by content category
category_counts = df["content_category"].value_counts()

plt.figure(figsize=(9, 5))
category_counts.plot(kind="bar")
plt.title("Number of Posts by Content Category")
plt.xlabel("Content Category")
plt.ylabel("Number of Posts")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(output_dir / "content_category_distribution.png")
plt.close()

# 8. Graph: follower count distribution
plt.figure(figsize=(8, 5))
df["follower_count"].plot(kind="hist", bins=10)
plt.title("Distribution of Follower Counts")
plt.xlabel("Follower Count")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(output_dir / "follower_distribution.png")
plt.close()

# 9. Graph: posts with and without a call to action
cta_counts = (
    df["has_call_to_action"]
    .map({0: "No CTA", 1: "Has CTA"})
    .fillna("Other")
    .value_counts()
)

plt.figure(figsize=(7, 5))
cta_counts.plot(kind="bar")
plt.title("Posts With and Without a Call to Action")
plt.xlabel("Call to Action")
plt.ylabel("Number of Posts")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(output_dir / "call_to_action_distribution.png")
plt.close()

print("\n7. GRAPH FILES SAVED IN:", output_dir)
print("EDA completed successfully!")

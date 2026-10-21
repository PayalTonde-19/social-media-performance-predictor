
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed_instagram_data.csv"
OUTPUT_PATH = BASE_DIR / "feature_importance.png"

# Load the prepared dataset
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["has_call_to_action"])
y = df["has_call_to_action"].astype(int)

# Identify column types
numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(
    include=["object", "category", "bool", "str"]
).columns.tolist()

# Prepare features
preprocessor = ColumnTransformer([
    (
        "numeric",
        SimpleImputer(strategy="median"),
        numeric_cols
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_cols
    )
])

# Use the same split as the main model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Train a separate tree for interpretation
insight_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=4,
        class_weight="balanced",
        random_state=42
    ))
])

insight_model.fit(X_train, y_train)

# Extract feature importance
fitted_preprocessor = insight_model.named_steps["preprocessor"]
tree = insight_model.named_steps["classifier"]

feature_names = fitted_preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": tree.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nTop 10 Decision Tree Features:")
print(importance_df.head(10).to_string(index=False))

# Plot the top 10 features
top_features = importance_df.head(10).sort_values("Importance")

plt.figure(figsize=(10, 6))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)
plt.title("Decision Tree Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
plt.close()

print(f"\nFeature-importance chart saved to: {OUTPUT_PATH}")
print("The selected CTA model was not changed.")


import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Step 1: Load the prepared dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed_instagram_data.csv"

df = pd.read_csv(DATA_PATH)

# Step 2: Separate input features and target
target = "has_call_to_action"

X = df.drop(columns=[target])
y = df[target].astype(int)

print("\nModel input columns:")
print(X.columns.tolist())

# Step 3: Identify numerical and categorical columns
numeric_cols = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_cols = X.select_dtypes(
    include=["object", "str", "category", "bool"]
).columns.tolist()

# Step 4: Prepare numerical features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# Step 5: Prepare categorical features
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

# Step 6: Check whether both target classes have enough rows
if y.nunique() < 2 or y.value_counts().min() < 2:
    raise ValueError(
        "Both CTA classes need at least two rows each."
    )

# Step 7: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Step 8: Create and train the Decision Tree
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=4,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X_train, y_train)
# Save the trained model
MODEL_PATH = BASE_DIR / "cta_model.pkl"
joblib.dump(model, MODEL_PATH)

print("Trained model saved to:", MODEL_PATH)

# Step 9: Predict the test data
predictions = model.predict(X_test)

print("\nDecision Tree Accuracy:")
print(round(accuracy_score(y_test, predictions), 3))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Step 10: Compare with a simple baseline
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

print("\nBaseline Accuracy:")
print(round(
    accuracy_score(y_test, baseline_predictions), 3
))

print("\nDay 4 model training completed!")


# Step 11: Train Logistic Regression
logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)

print("\nLogistic Regression Accuracy:")
print(round(
    accuracy_score(y_test, logistic_predictions), 3
))

print("\nLogistic Regression Report:")
print(classification_report(
    y_test,
    logistic_predictions,
    zero_division=0
))

print("\nModel Comparison")
print("----------------")
print("Decision Tree:", round(
    accuracy_score(y_test, predictions), 3
))
print("Logistic Regression:", round(
    accuracy_score(y_test, logistic_predictions), 3
))
print("Baseline:", round(
    accuracy_score(y_test, baseline_predictions), 3
))


# Step 12: Check the distribution of CTA labels
print("\nCTA Label Counts:")
print(y.value_counts().sort_index())

print("\nCTA Label Percentages:")
print(
    y.value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

# Step 13: Check whether the classes are balanced
print("\nNumber of input features:", X.shape[1])
print("Number of training records:", len(X_train))
print("Number of testing records:", len(X_test))


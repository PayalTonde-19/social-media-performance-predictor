
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
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed_instagram_data.csv"
MODEL_PATH = BASE_DIR / "cta_model.pkl"
COMPARISON_PATH = BASE_DIR / "model_comparison.csv"
RESULTS_PATH = BASE_DIR / "model_results.txt"

# 1. Load the processed dataset
df = pd.read_csv(DATA_PATH)

target = "has_call_to_action"
X = df.drop(columns=[target])
y = df[target].astype(int)

print("Dataset shape:", df.shape)
print("Model input columns:", X.columns.tolist())
print("\nCTA class counts:")
print(y.value_counts().sort_index())

# 2. Create a fresh preprocessor for each model
numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


def make_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ])


# 3. Split into training and testing data
if y.nunique() < 2 or y.value_counts().min() < 2:
    raise ValueError("Both CTA classes need at least two records.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# 4. Define the candidate models
models = {
    "Decision Tree": Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", DecisionTreeClassifier(
            max_depth=4,
            class_weight="balanced",
            random_state=42
        ))
    ]),
    "Logistic Regression": Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ]),
    "Baseline": DummyClassifier(strategy="most_frequent")
}

results = []
predictions_by_model = {}

# 5. Train and evaluate each model
for name, candidate in models.items():
    candidate.fit(X_train, y_train)
    predictions = candidate.predict(X_test)
    predictions_by_model[name] = predictions

    results.append({
        "Model": name,
        "Accuracy": round(
            accuracy_score(y_test, predictions), 3
        ),
        "Balanced Accuracy": round(
            balanced_accuracy_score(y_test, predictions), 3
        ),
        "Macro F1": round(
            f1_score(
                y_test, predictions,
                average="macro", zero_division=0
            ), 3
        )
    })

comparison = pd.DataFrame(results).sort_values(
    "Macro F1", ascending=False
)

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))

comparison.to_csv(COMPARISON_PATH, index=False)

# 6. Select the best machine-learning model by Macro F1
# Baseline remains a benchmark, not the selected ML model.
ml_names = ["Decision Tree", "Logistic Regression"]

best_name = max(
    ml_names,
    key=lambda name: f1_score(
        y_test,
        predictions_by_model[name],
        average="macro",
        zero_division=0
    )
)

best_model = models[best_name]
best_predictions = predictions_by_model[best_name]

print("\nSELECTED MACHINE-LEARNING MODEL:", best_name)
print(
    "Accuracy:",
    round(accuracy_score(y_test, best_predictions), 3)
)
print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_test, best_predictions), 3)
)
print(
    "Macro F1:",
    round(f1_score(
        y_test, best_predictions,
        average="macro", zero_division=0
    ), 3)
)

print("\nClassification Report")
report = classification_report(
    y_test, best_predictions, zero_division=0
)
print(report)

print("Confusion Matrix")
matrix = confusion_matrix(y_test, best_predictions)
print(matrix)

# 7. Save the selected model and evaluation report
joblib.dump(best_model, MODEL_PATH)

with open(RESULTS_PATH, "w", encoding="utf-8") as file:
    file.write("SOCIAL MEDIA PERFORMANCE PREDICTOR\n")
    file.write("CTA Classification Evaluation\n\n")
    file.write("Target: has_call_to_action\n")
    file.write(f"Training rows: {len(X_train)}\n")
    file.write(f"Testing rows: {len(X_test)}\n")
    file.write(f"Selected ML model: {best_name}\n\n")
    file.write("Model comparison:\n")
    file.write(comparison.to_string(index=False))
    file.write("\n\nClassification report:\n")
    file.write(report)
    file.write("\nConfusion matrix:\n")
    file.write(str(matrix))
    file.write("\n\nImportant limitation:\n")
    file.write(
        "The dataset is synthetic and contains only 200 records. "
        "The results do not establish real-world predictive reliability.\n"
    )

# 8. Show Decision Tree feature importance
tree_model = models["Decision Tree"]
tree_preprocessor = tree_model.named_steps["preprocessor"]
tree_classifier = tree_model.named_steps["classifier"]

feature_names = tree_preprocessor.get_feature_names_out()
importance_values = tree_classifier.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
}).sort_values("Importance", ascending=False)

print("\nTOP 10 DECISION TREE FEATURES")
print(importance_df.head(10).to_string(index=False))

print("\nSaved model:", MODEL_PATH)
print("Saved comparison:", COMPARISON_PATH)
print("Saved report:", RESULTS_PATH)
print("\nModel evaluation completed!")

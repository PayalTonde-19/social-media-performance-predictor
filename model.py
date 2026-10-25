import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed_instagram_data.csv"
MODEL_PATH = BASE_DIR / "cta_model.pkl"
COMPARISON_PATH = BASE_DIR / "model_comparison.csv"
RESULTS_PATH = BASE_DIR / "model_results.txt"

# 1. Load dataset
df = pd.read_csv(DATA_PATH)

target = "has_call_to_action"

if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found.")

df = df.dropna(subset=[target]).copy()

X = df.drop(columns=[target])
y = df[target].astype(int)

if y.nunique() != 2:
    raise ValueError("CTA target must contain two classes.")

class_counts = y.value_counts()

if class_counts.min() < 5:
    raise ValueError(
        "At least five records per CTA class are required "
        "for 5-fold stratified cross-validation."
    )

print("Dataset shape:", df.shape)
print("Features:", X.columns.tolist())

print("\nCTA class counts:")
print(class_counts.sort_index())

# 2. Identify feature types
numeric_cols = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


def make_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ])


# 3. Define candidate models
def make_pipeline(classifier):
    return Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", classifier)
    ])


models = {
    "Logistic Regression": make_pipeline(
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    ),

    "Decision Tree": make_pipeline(
        DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42
        )
    ),

    "Random Forest": make_pipeline(
        RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42
        )
    ),

    "Baseline": DummyClassifier(
        strategy="most_frequent"
    )
}


# 4. Evaluate every model using the same cross-validation splits
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "macro_f1": "f1_macro"
}

results = []

print("\n5-FOLD CROSS-VALIDATION RESULTS")

for name, candidate in models.items():

    scores = cross_validate(
        candidate,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        error_score="raise"
    )

    result = {
        "Model": name,
        "Accuracy": round(
            scores["test_accuracy"].mean(), 3
        ),
        "Balanced Accuracy": round(
            scores["test_balanced_accuracy"].mean(), 3
        ),
        "Macro F1": round(
            scores["test_macro_f1"].mean(), 3
        )
    }

    results.append(result)

    print(f"\n{name}")
    print("Mean accuracy:", result["Accuracy"])
    print(
        "Mean balanced accuracy:",
        result["Balanced Accuracy"]
    )
    print("Mean Macro F1:", result["Macro F1"])


# 5. Save comparison results
comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    "Macro F1",
    ascending=False
)

comparison.to_csv(
    COMPARISON_PATH,
    index=False
)

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))


# 6. Select the best trained model by cross-validation Macro F1
# The baseline is a benchmark, not the deployed ML model.
ml_names = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest"
]

best_name = comparison[
    comparison["Model"].isin(ml_names)
].iloc[0]["Model"]

best_model = models[best_name]

print("\nSELECTED MODEL:", best_name)

# 7. Train the selected pipeline on all available records
best_model.fit(X, y)

# Save the pipeline for Streamlit
joblib.dump(best_model, MODEL_PATH)

# 8. Write evaluation report
with open(RESULTS_PATH, "w", encoding="utf-8") as file:

    file.write(
        "SOCIAL MEDIA PERFORMANCE PREDICTOR\n"
    )
    file.write(
        "CTA Classification: 5-Fold Cross-Validation\n\n"
    )

    file.write(f"Dataset records: {len(df)}\n")
    file.write(f"Selected ML model: {best_name}\n")
    file.write("Target: has_call_to_action\n\n")

    file.write("Cross-validation comparison:\n")
    file.write(comparison.to_string(index=False))

    file.write(
        "\n\nEvaluation note:\n"
        "Metrics are mean scores from five stratified folds. "
        "The selected model was then fitted on all available "
        "records for deployment. The reported cross-validation "
        "scores are not scores from an independent final test set.\n"
    )

    file.write(
        "\nDataset limitation:\n"
        "The dataset is synthetic and contains only "
        f"{len(df)} records. Results do not establish "
        "real-world predictive reliability.\n"
    )

print("\nSaved model:", MODEL_PATH)
print("Saved comparison:", COMPARISON_PATH)
print("Saved report:", RESULTS_PATH)

print("\nModel evaluation completed!")
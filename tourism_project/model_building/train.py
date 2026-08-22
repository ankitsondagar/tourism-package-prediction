
"""
Training pipeline for the Tourism Package Prediction model.
Loads the pre-split train/test data, tunes an XGBoost classifier
via GridSearchCV, evaluates it, logs everything to MLflow, and
persists the final estimator for deployment.
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ------------------------------------------------------------------
# Directory / file locations
# ------------------------------------------------------------------

BASE_DIR = Path("tourism_project")

train_csv_path = BASE_DIR / "data" / "splits" / "train.csv"
test_csv_path = BASE_DIR / "data" / "splits" / "test.csv"

deployment_dir = BASE_DIR / "deployment"
saved_model_path = deployment_dir / "tourism_model.pkl"

print("Train data source:", train_csv_path)
print("Test data source:", test_csv_path)


# ------------------------------------------------------------------
# Sanity checks on inputs
# ------------------------------------------------------------------

def ensure_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")


ensure_file_exists(train_csv_path, "Training")
ensure_file_exists(test_csv_path, "Testing")


# ------------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------------

training_data = pd.read_csv(train_csv_path)
testing_data = pd.read_csv(test_csv_path)

print("Training set shape:", training_data.shape)
print("Testing set shape:", testing_data.shape)


# ------------------------------------------------------------------
# Split features / target
# ------------------------------------------------------------------

label_column = "ProdTaken"

if label_column not in training_data.columns:
    raise ValueError(f"Target column '{label_column}' not found.")

features_train = training_data.drop(columns=[label_column])
target_train = training_data[label_column]

features_test = testing_data.drop(columns=[label_column])
target_test = testing_data[label_column]


# ------------------------------------------------------------------
# Drop identifier column (not predictive)
# ------------------------------------------------------------------

id_column = "CustomerID"

if id_column in features_train.columns:
    features_train = features_train.drop(columns=[id_column])
    features_test = features_test.drop(columns=[id_column])
    print(f"Dropped identifier column '{id_column}' from features.")


# ------------------------------------------------------------------
# Detect column types
# ------------------------------------------------------------------

cat_cols = features_train.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = features_train.select_dtypes(exclude=["object", "category"]).columns.tolist()

print("\nCategorical features:", cat_cols)
print("Numerical features:", num_cols)


# ------------------------------------------------------------------
# Build preprocessing + model pipeline
# ------------------------------------------------------------------

feature_transformer = ColumnTransformer(
    transformers=[
        ("cat_encode", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num_passthrough", "passthrough", num_cols),
    ]
)

model_pipeline = Pipeline(
    steps=[
        ("preprocess", feature_transformer),
        (
            "clf",
            XGBClassifier(
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
            ),
        ),
    ]
)


# ------------------------------------------------------------------
# Hyperparameter search space
# ------------------------------------------------------------------

hyperparam_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [3, 5, 7],
    "clf__learning_rate": [0.01, 0.1, 0.2],
    "clf__subsample": [0.8, 1.0],
    "clf__colsample_bytree": [0.8, 1.0],
}


# ------------------------------------------------------------------
# MLflow experiment setup
# ------------------------------------------------------------------

mlflow.set_experiment("Tourism Package Prediction")

with mlflow.start_run():

    print("\nRunning grid search...")

    searcher = GridSearchCV(
        estimator=model_pipeline,
        param_grid=hyperparam_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )

    searcher.fit(features_train, target_train)

    # ----------------------------------------------------------
    # Best estimator found
    # ----------------------------------------------------------

    tuned_model = searcher.best_estimator_

    print("\nBest hyperparameters found:")
    print(searcher.best_params_)

    # ----------------------------------------------------------
    # Generate predictions on the holdout set
    # ----------------------------------------------------------

    predicted_labels = tuned_model.predict(features_test)
    predicted_probs = tuned_model.predict_proba(features_test)[:, 1]

    # ----------------------------------------------------------
    # Compute evaluation metrics
    # ----------------------------------------------------------

    metrics = {
        "accuracy": accuracy_score(target_test, predicted_labels),
        "precision": precision_score(target_test, predicted_labels, zero_division=0),
        "recall": recall_score(target_test, predicted_labels, zero_division=0),
        "f1_score": f1_score(target_test, predicted_labels, zero_division=0),
        "roc_auc": roc_auc_score(target_test, predicted_probs),
    }

    print("\nEvaluation Results")
    print("-------------------")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")

    # ----------------------------------------------------------
    # Log hyperparameters and metrics to MLflow
    # ----------------------------------------------------------

    mlflow.log_params(searcher.best_params_)
    mlflow.log_metrics(metrics)

    # ----------------------------------------------------------
    # Persist the trained model for deployment
    # ----------------------------------------------------------

    deployment_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(tuned_model, saved_model_path)

    print("\nModel written to disk at:")
    print(saved_model_path)

    # ----------------------------------------------------------
    # Log the saved model artifact to MLflow
    # ----------------------------------------------------------

    mlflow.log_artifact(str(saved_model_path), artifact_path="deployment_model")


print("\n==================================================")
print("TRAINING PIPELINE FINISHED")
print("==================================================")

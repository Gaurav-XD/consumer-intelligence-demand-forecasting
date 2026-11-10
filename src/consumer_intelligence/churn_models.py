"""Train classical churn models with training-only model selection."""

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from consumer_intelligence.churn_dataset import MODEL_FEATURE_COLUMNS
from consumer_intelligence.config import MODEL_DIR, PROCESSED_DATA_DIR, REPORTS_DIR

DATASET_PATH = PROCESSED_DATA_DIR / "churn_dataset.csv"
METADATA_PATH = REPORTS_DIR / "churn_dataset_metadata.json"
RESULTS_PATH = REPORTS_DIR / "churn_model_results.csv"
SUMMARY_PATH = REPORTS_DIR / "churn_model_summary.json"
CHURN_MODEL_DIR = MODEL_DIR / "churn"
RANDOM_STATE = 42


def load_churn_data(
    dataset_path: Path, metadata_path: Path
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    """Load the predefined splits and validate the metadata feature allowlist."""
    with Path(metadata_path).open(encoding="utf-8") as file:
        metadata = json.load(file)
    feature_columns = metadata["feature_columns"]
    if feature_columns != MODEL_FEATURE_COLUMNS or metadata["target_column"] != "churn":
        raise ValueError("Churn metadata does not match the approved feature contract")

    data = pd.read_csv(dataset_path, dtype={"customer_id": "string"})
    required = {"customer_id", "snapshot_date", "split", "churn", *feature_columns}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Churn dataset is missing columns: {sorted(missing)}")
    if data["customer_id"].isna().any() or data["customer_id"].duplicated().any():
        raise ValueError("Churn dataset must contain unique, non-missing customer IDs")
    if not data["split"].isin(["train", "test"]).all():
        raise ValueError("Churn dataset contains invalid split values")
    if not data["churn"].isin([0, 1]).all():
        raise ValueError("Churn target must be binary and non-missing")

    for split in ("train", "test"):
        rows = data.loc[data["split"].eq(split)]
        if rows.empty or rows["churn"].nunique() != 2:
            raise ValueError(f"{split} split must contain both target classes")
        expected_date = metadata["periods"][split]["feature_snapshot_period"][
            "snapshot_date_inclusive"
        ]
        if not rows["snapshot_date"].eq(expected_date).all():
            raise ValueError(f"{split} split contains unexpected snapshot dates")
        if len(rows) != metadata["periods"][split]["row_count"]:
            raise ValueError(f"{split} split row count does not match metadata")

    numeric = data[feature_columns].apply(pd.to_numeric, errors="raise")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Model features cannot contain infinite values")
    train_mask = data["split"].eq("train")
    test_mask = data["split"].eq("test")
    return (
        numeric.loc[train_mask].copy(),
        data.loc[train_mask, "churn"].astype(int).copy(),
        numeric.loc[test_mask].copy(),
        data.loc[test_mask, "churn"].astype(int).copy(),
        feature_columns,
    )


def build_pipelines() -> dict[str, Pipeline]:
    """Create independent estimators with fit-time-only preprocessing."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("model", DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "model",
                    RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]
        ),
    }


def run_churn_models(
    dataset_path: Path = DATASET_PATH,
    metadata_path: Path = METADATA_PATH,
    results_path: Path = RESULTS_PATH,
    summary_path: Path = SUMMARY_PATH,
    model_dir: Path = CHURN_MODEL_DIR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select on training-only CV and evaluate each model on the fixed test set."""
    x_train, y_train, x_test, y_test, features = load_churn_data(
        Path(dataset_path), Path(metadata_path)
    )
    if y_train.value_counts().min() < 3:
        raise ValueError("Training split needs at least three examples of each class")
    pipelines = build_pipelines()
    folds = StratifiedKFold(n_splits=3, shuffle=False)
    cv_scores: dict[str, float] = {}
    for name, pipeline in pipelines.items():
        scores = cross_val_score(
            pipeline,
            x_train,
            y_train,
            scoring="average_precision",
            cv=folds,
            error_score="raise",
        )
        cv_scores[name] = float(scores.mean())

    # Stable insertion order breaks exact ties before any test metrics are computed.
    best_name = max(cv_scores, key=cv_scores.__getitem__)
    results: list[dict[str, Any]] = []
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, str] = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        predictions = pipeline.predict(x_test)
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        results.append(
            {
                "model": name,
                "train_cv_average_precision": cv_scores[name],
                "roc_auc": float(roc_auc_score(y_test, probabilities)),
                "pr_auc": float(average_precision_score(y_test, probabilities)),
                "accuracy": float(accuracy_score(y_test, predictions)),
                "precision": float(precision_score(y_test, predictions, zero_division=0)),
                "recall": float(recall_score(y_test, predictions, zero_division=0)),
                "f1": float(f1_score(y_test, predictions, zero_division=0)),
                "true_negative": int(matrix[0, 0]),
                "false_positive": int(matrix[0, 1]),
                "false_negative": int(matrix[1, 0]),
                "true_positive": int(matrix[1, 1]),
            }
        )
        artifact = model_dir / f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(pipeline, artifact)
        artifact_paths[name] = str(artifact)

    best_path = model_dir / "best_model.joblib"
    joblib.dump(pipelines[best_name], best_path)
    results_frame = pd.DataFrame(results)
    summary: dict[str, Any] = {
        "selection_criterion": (
            "Highest mean average precision across three stratified, non-shuffled "
            "cross-validation folds of the training split only; exact ties follow "
            "the model order listed in results."
        ),
        "best_model": best_name,
        "best_train_cv_average_precision": cv_scores[best_name],
        "best_model_path": str(best_path),
        "model_artifacts": artifact_paths,
        "feature_columns": features,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "random_state": RANDOM_STATE,
        "library_versions": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "test_metrics": {
            row["model"]: {
                **{key: value for key, value in row.items() if key != "model"},
                "confusion_matrix": [
                    [row["true_negative"], row["false_positive"]],
                    [row["false_negative"], row["true_positive"]],
                ],
            }
            for row in results
        },
        "metric_definitions": {
            "pr_auc": "Average precision: stepwise area under the precision-recall curve.",
            "classification_threshold": "Estimator default (binary probability threshold 0.5).",
            "confusion_matrix": "[[true_negative, false_positive], [false_negative, true_positive]]",
        },
        "class_imbalance": (
            "Balanced class weights for logistic regression, decision tree and random forest; "
            "gradient boosting uses its default unweighted objective."
        ),
        "evaluation_note": (
            "Selection uses training CV only. The test metrics compare all candidates; "
            "repeated decisions based on them would invalidate the test as an untouched holdout."
        ),
    }
    results_path = Path(results_path)
    summary_path = Path(summary_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    results_frame.to_csv(results_path, index=False)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
        file.write("\n")
    return results_frame, summary


if __name__ == "__main__":
    results, model_summary = run_churn_models()
    print(results.to_string(index=False))
    print(f"Best model by training CV average precision: {model_summary['best_model']}")
    print(f"Best model saved to: {model_summary['best_model_path']}")

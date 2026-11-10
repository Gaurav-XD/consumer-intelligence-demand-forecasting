"""Tune churn pipelines on training data before evaluating the selected model."""

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from consumer_intelligence.churn_models import (
    CHURN_MODEL_DIR,
    DATASET_PATH,
    METADATA_PATH,
    RANDOM_STATE,
    build_pipelines,
    load_churn_data,
)
from consumer_intelligence.config import REPORTS_DIR

RESULTS_PATH = REPORTS_DIR / "churn_tuning_results.csv"
SUMMARY_PATH = REPORTS_DIR / "churn_tuning_summary.json"
MODEL_PATH = CHURN_MODEL_DIR / "tuned_best_model.joblib"
SEARCH_ITERATIONS = 6
PARAMETER_SPACES = {
    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "model__class_weight": [None, "balanced"],
    },
    "Random Forest": {
        "model__n_estimators": [80, 120],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_leaf": [1, 3, 10],
        "model__max_features": ["sqrt", 0.7],
    },
    "Gradient Boosting": {
        "model__n_estimators": [80, 120],
        "model__learning_rate": [0.03, 0.1],
        "model__max_depth": [1, 2, 3],
        "model__min_samples_leaf": [5, 15],
    },
}


def run_churn_tuning(
    dataset_path: Path = DATASET_PATH,
    metadata_path: Path = METADATA_PATH,
    results_path: Path = RESULTS_PATH,
    summary_path: Path = SUMMARY_PATH,
    model_path: Path = MODEL_PATH,
    n_iter: int = SEARCH_ITERATIONS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Search training folds, select by mean average precision, evaluate once."""
    if n_iter < 1:
        raise ValueError("n_iter must be positive")
    x_train, y_train, x_test, y_test, features = load_churn_data(
        Path(dataset_path), Path(metadata_path)
    )
    if y_train.value_counts().min() < 3:
        raise ValueError("Training split needs at least three examples of each class")
    pipelines = build_pipelines()
    folds = StratifiedKFold(n_splits=3, shuffle=False)
    searches: dict[str, RandomizedSearchCV] = {}
    rows = []
    for name, space in PARAMETER_SPACES.items():
        search = RandomizedSearchCV(
            estimator=pipelines[name],
            param_distributions=space,
            n_iter=n_iter,
            scoring="average_precision",
            cv=folds,
            random_state=RANDOM_STATE,
            refit=True,
            error_score="raise",
            n_jobs=1,
        )
        search.fit(x_train, y_train)
        searches[name] = search
        rows.append(
            {
                "model": name,
                "best_parameters": json.dumps(search.best_params_, sort_keys=True),
                "mean_cv_average_precision": float(search.best_score_),
                "std_cv_average_precision": float(
                    search.cv_results_["std_test_score"][search.best_index_]
                ),
            }
        )

    # Dict insertion order is the deterministic tie-breaker.
    best_name = max(searches, key=lambda name: searches[name].best_score_)
    selected = searches[best_name].best_estimator_
    probabilities = selected.predict_proba(x_test)[:, 1]
    predictions = selected.predict(x_test)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    test_metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
    }
    results = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "selection_criterion": (
            "Highest mean training-only cross-validation average precision; "
            "ties use the candidate model order."
        ),
        "selected_model": best_name,
        "best_parameters": searches[best_name].best_params_,
        "mean_cv_average_precision": float(searches[best_name].best_score_),
        "std_cv_average_precision": float(
            searches[best_name].cv_results_["std_test_score"][
                searches[best_name].best_index_
            ]
        ),
        "models": {
            row["model"]: {
                "best_parameters": json.loads(row["best_parameters"]),
                "mean_cv_average_precision": row["mean_cv_average_precision"],
                "std_cv_average_precision": row["std_cv_average_precision"],
            }
            for row in rows
        },
        "test_metrics": test_metrics,
        "metric_definitions": {
            "pr_auc": "Average precision (stepwise area under the precision-recall curve).",
            "confusion_matrix": "[[TN, FP], [FN, TP]]",
            "classification_threshold": "Estimator default (0.5).",
        },
        "feature_columns": features,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "search": {
            "method": "RandomizedSearchCV",
            "iterations_per_model": n_iter,
            "cross_validation": "3-fold stratified, unshuffled",
            "random_state": RANDOM_STATE,
            "parameter_spaces": PARAMETER_SPACES,
        },
        "model_path": str(Path(model_path)),
        "library_versions": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "evaluation_caveat": (
            "This test split was previously evaluated for baseline models. "
            "The tuning procedure does not use it for selection, but repeated "
            "project decisions using this holdout can introduce selection bias."
        ),
    }
    results_path, summary_path, model_path = (
        Path(results_path),
        Path(summary_path),
        Path(model_path),
    )
    for path in (results_path, summary_path, model_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, model_path)
    results.to_csv(results_path, index=False)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
        file.write("\n")
    return results, summary


if __name__ == "__main__":
    comparison, tuning_summary = run_churn_tuning()
    print(comparison.to_string(index=False))
    print(f"Selected model: {tuning_summary['selected_model']}")
    print(f"Test metrics: {tuning_summary['test_metrics']}")

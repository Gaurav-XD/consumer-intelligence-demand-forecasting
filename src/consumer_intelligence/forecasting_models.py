"""Compare demand models on a chronological validation period."""

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from consumer_intelligence.config import MODEL_DIR, REPORTS_DIR
from consumer_intelligence.forecasting_baselines import SeasonalNaiveRegressor
from consumer_intelligence.forecasting_validation import (
    INPUT_PATH,
    NUMERIC_FEATURES,
    SPLIT_PERIODS,
    SUMMARY_PATH as VALIDATION_PATH,
    create_forecasting_splits,
)

RESULTS_PATH = REPORTS_DIR / "forecasting_model_results.csv"
SUMMARY_PATH = REPORTS_DIR / "forecasting_model_summary.json"
MODEL_DIR_PATH = MODEL_DIR / "forecasting"
RANDOM_STATE = 42


def build_models() -> dict[str, Any]:
    """Create independent baseline and regression pipelines."""
    return {
        "Seasonal Naive": SeasonalNaiveRegressor(),
        "Linear Regression": Pipeline(
            [("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
             ("model", LinearRegression())]
        ),
        "Random Forest": Pipeline(
            [("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
             ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1))]
        ),
        "Gradient Boosting": Pipeline(
            [("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
             ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))]
        ),
    }


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float | None]:
    """Calculate percentage metrics as percentages, excluding zeros from MAPE."""
    y = np.asarray(actual, dtype=float)
    prediction = np.asarray(predicted, dtype=float)
    if len(y) == 0 or y.shape != prediction.shape:
        raise ValueError("Actual and predicted values must have matching non-empty shapes")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Actual and predicted values must be finite")
    errors = np.abs(y - prediction)
    nonzero = y != 0
    denominator = np.abs(y).sum()
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "mape": float(np.mean(errors[nonzero] / np.abs(y[nonzero])) * 100)
        if nonzero.any() else None,
        "wape": float(errors.sum() / denominator * 100) if denominator else None,
    }


def run_forecasting_models(
    input_path: Path = INPUT_PATH,
    validation_path: Path = VALIDATION_PATH,
    results_path: Path = RESULTS_PATH,
    summary_path: Path = SUMMARY_PATH,
    model_dir: Path = MODEL_DIR_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit on train, select by validation MAE, and evaluate the winner on test."""
    with Path(validation_path).open(encoding="utf-8") as file:
        validation = json.load(file)
    if (
        validation.get("numeric_feature_columns") != NUMERIC_FEATURES
        or validation.get("target_column") != "quantity_sold"
    ):
        raise ValueError("Validation report does not match the approved feature contract")
    for split_name, (start, end) in SPLIT_PERIODS.items():
        info = validation["splits"][split_name]
        if (info["requested_start_month"], info["requested_end_month"]) != (start, end):
            raise ValueError(f"{split_name} cutoffs do not match validation contract")
    source = pd.read_csv(
        Path(input_path),
        dtype={"product_id": "string", "month": "string", "category": "string", "brand": "string"},
    )
    splits, split_summary = create_forecasting_splits(
        source, {"feature_columns": NUMERIC_FEATURES, "target_column": "quantity_sold"}
    )
    for name in SPLIT_PERIODS:
        reported = validation["splits"][name]
        actual = split_summary["splits"][name]
        for field in ("start_month", "end_month", "row_count", "product_count"):
            if reported[field] != actual[field]:
                raise ValueError(f"{name} split {field} differs from validation report")

    train, validation_rows = splits["train"], splits["validation"]
    x_train = train[NUMERIC_FEATURES]
    y_train = train["quantity_sold"]
    x_validation = validation_rows[NUMERIC_FEATURES]
    y_validation = validation_rows["quantity_sold"]
    models = build_models()
    rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        metrics = regression_metrics(y_validation, model.predict(x_validation))
        rows.append({"model": name, **metrics})
    best_name = min(rows, key=lambda row: row["mae"])["model"]

    test = splits["test"]
    best = models[best_name]
    test_metrics = regression_metrics(test["quantity_sold"], best.predict(test[NUMERIC_FEATURES]))
    result = pd.DataFrame(rows)
    model_dir, results_path, summary_path = Path(model_dir), Path(results_path), Path(summary_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "best_model.joblib"
    joblib.dump(best, model_path)
    result.to_csv(results_path, index=False)
    summary: dict[str, Any] = {
        "selected_model": best_name,
        "selection_criterion": "Lowest validation MAE; ties use the model order in results.",
        "feature_columns": NUMERIC_FEATURES,
        "target_column": "quantity_sold",
        "excluded_tracking_columns": ["product_id", "month"],
        "model_parameters": {
            name: model.get_params(deep=True) for name, model in models.items()
        },
        "validation_metrics": {row["model"]: {k: v for k, v in row.items() if k != "model"} for row in rows},
        "test_metrics": test_metrics,
        "periods": split_summary["splits"],
        "evaluation_methodology": (
            "Rolling-origin, one-step-ahead: the selected model remains fitted on the "
            "training period; each evaluation row uses actual prior-month demand, "
            "including earlier validation/test months. Not a fixed-origin multi-step forecast."
        ),
        "metric_definitions": {
            "mape": "Mean absolute percentage error as percent, excluding zero actuals; null if all actuals are zero.",
            "wape": "100 * sum(abs(actual - prediction)) / sum(abs(actual)); null if denominator is zero.",
        },
        "random_state": RANDOM_STATE,
        "model_path": str(model_path),
        "library_versions": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    # Pipeline get_params includes estimator objects; record only JSON-safe settings.
    for name, model in models.items():
        estimator = model.named_steps["model"] if isinstance(model, Pipeline) else model
        summary["model_parameters"][name] = estimator.get_params(deep=False)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
        file.write("\n")
    return result, summary


if __name__ == "__main__":
    comparison, report = run_forecasting_models()
    print(comparison.to_string(index=False))
    print(f"Selected: {report['selected_model']}; test metrics: {report['test_metrics']}")

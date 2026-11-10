"""Training-only calendar CV and walk-forward demand validation."""

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from consumer_intelligence.config import MODEL_DIR, REPORTS_DIR
from consumer_intelligence.forecasting_features import HISTORY_FEATURES, OUTPUT_COLUMNS
from consumer_intelligence.forecasting_models import build_models, regression_metrics
from consumer_intelligence.forecasting_validation import (
    INPUT_PATH, NUMERIC_FEATURES, SPLIT_PERIODS, SUMMARY_PATH as VALIDATION_PATH,
)

RESULTS_PATH = REPORTS_DIR / "forecasting_tuning_results.csv"
SUMMARY_PATH = REPORTS_DIR / "forecasting_tuning_summary.json"
WALK_PATH = REPORTS_DIR / "forecasting_walk_forward.csv"
MODEL_PATH = MODEL_DIR / "forecasting" / "tuned_best_model.joblib"
PARAMETERS = {
    "Random Forest": {
        "model__max_depth": [None, 8],
        "model__min_samples_leaf": [1, 5],
    },
    "Gradient Boosting": {
        "model__max_depth": [2, 3],
        "model__learning_rate": [0.05, 0.1],
    },
}


def calendar_cv(months: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    """Produce expanding folds whose boundaries never split a calendar month."""
    ordered = sorted(months.unique())
    if len(ordered) < 3:
        raise ValueError("Training data needs at least three eligible months for calendar CV")
    if any(pd.Period(right, freq="M") != pd.Period(left, freq="M") + 1
           for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Eligible training months must be consecutive")
    folds = []
    for month in ordered[1:]:
        train_indices = np.flatnonzero(months.lt(month).to_numpy())
        validation_indices = np.flatnonzero(months.eq(month).to_numpy())
        folds.append((train_indices, validation_indices))
    return folds


def run_forecasting_tuning(
    input_path: Path = INPUT_PATH,
    validation_path: Path = VALIDATION_PATH,
    results_path: Path = RESULTS_PATH,
    summary_path: Path = SUMMARY_PATH,
    walk_path: Path = WALK_PATH,
    model_path: Path = MODEL_PATH,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Search the training period, then evaluate and backtest validation."""
    with Path(validation_path).open(encoding="utf-8") as file:
        contract = json.load(file)
    if contract.get("numeric_feature_columns") != NUMERIC_FEATURES or contract.get("target_column") != "quantity_sold":
        raise ValueError("Validation report does not match approved feature contract")
    for name, (start, end) in SPLIT_PERIODS.items():
        info = contract["splits"][name]
        if (info["requested_start_month"], info["requested_end_month"]) != (start, end):
            raise ValueError(f"{name} cutoffs do not match validation contract")

    source = pd.read_csv(
        Path(input_path),
        dtype={"product_id": "string", "month": "string", "category": "string", "brand": "string"},
    )
    if set(OUTPUT_COLUMNS) - set(source.columns):
        raise ValueError("Forecasting source is missing required columns")
    if source["month"].isna().any() or not source["month"].str.fullmatch(r"\d{4}-(0[1-9]|1[0-2])").all():
        raise ValueError("Forecasting months must use YYYY-MM format")
    # Do not inspect final-test targets or features during model development.
    source = source.loc[source["month"].le(SPLIT_PERIODS["validation"][1]), OUTPUT_COLUMNS].copy()
    for column in ("product_id", "category", "brand"):
        if source[column].isna().any() or source[column].str.strip().eq("").any():
            raise ValueError(f"{column} must be present")
    if source.duplicated(["product_id", "month"]).any():
        raise ValueError("Product-month keys must be unique")
    for column in ["quantity_sold", *NUMERIC_FEATURES]:
        source[column] = pd.to_numeric(source[column], errors="raise")
    target = source["quantity_sold"]
    if target.isna().any() or not np.isfinite(target.to_numpy(dtype=float)).all() or target.lt(0).any() or target.mod(1).ne(0).any():
        raise ValueError("Targets must contain finite nonnegative integer values")
    if np.isinf(source[NUMERIC_FEATURES].to_numpy(dtype=float)).any():
        raise ValueError("Forecasting features must not contain infinite values")
    if source[["month_of_year", "quarter", "trend_index"]].isna().any().any():
        raise ValueError("Calendar features must be present")
    source = source.dropna(subset=HISTORY_FEATURES).sort_values(["month", "product_id"]).reset_index(drop=True)
    splits = {}
    for name in ("train", "validation"):
        start, end = SPLIT_PERIODS[name]
        frame = source.loc[source["month"].between(start, end)].copy().reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"{name} split has no eligible rows")
        expected = contract["splits"][name]
        for field, value in {
            "row_count": len(frame),
            "product_count": int(frame["product_id"].nunique()),
            "start_month": str(frame["month"].min()),
            "end_month": str(frame["month"].max()),
        }.items():
            if expected[field] != value:
                raise ValueError(f"{name} split {field} differs from validation report")
        splits[name] = frame

    train = splits["train"]
    validation = splits["validation"]
    folds = calendar_cv(train["month"])
    fold_descriptions = [
        {
            "training_start_month": str(train.iloc[fit_idx]["month"].min()),
            "training_end_month": str(train.iloc[fit_idx]["month"].max()),
            "validation_month": str(train.iloc[score_idx]["month"].iloc[0]),
            "training_rows": len(fit_idx),
            "validation_rows": len(score_idx),
        }
        for fit_idx, score_idx in folds
    ]
    base_models = build_models()
    searches: dict[str, GridSearchCV] = {}
    results = []
    for name, parameters in PARAMETERS.items():
        search = GridSearchCV(
            base_models[name], parameters, scoring="neg_mean_absolute_error",
            cv=folds, refit=True, n_jobs=1, error_score="raise",
        )
        search.fit(train[NUMERIC_FEATURES], train["quantity_sold"])
        searches[name] = search
        results.append(
            {
                "model": name,
                "best_parameters": json.dumps(search.best_params_, sort_keys=True),
                "cv_mae": float(-search.best_score_),
                "cv_mae_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            }
        )
    winner = min(searches, key=lambda name: -searches[name].best_score_)
    selected = searches[winner].best_estimator_
    validation_metrics = regression_metrics(
        validation["quantity_sold"], selected.predict(validation[NUMERIC_FEATURES])
    )

    predictions = []
    for month in sorted(validation["month"].unique()):
        past = source.loc[
            source["month"].ge(SPLIT_PERIODS["train"][0]) & source["month"].lt(month)
        ]
        current = validation.loc[validation["month"].eq(month)]
        if past.empty or pd.Period(past["month"].max(), freq="M") != pd.Period(month, freq="M") - 1:
            raise ValueError("Walk-forward training must end in the previous calendar month")
        model = clone(selected)
        model.fit(past[NUMERIC_FEATURES], past["quantity_sold"])
        forecast = model.predict(current[NUMERIC_FEATURES])
        for product_id, actual, prediction in zip(
            current["product_id"], current["quantity_sold"], forecast
        ):
            error = float(actual - prediction)
            predictions.append(
                {
                    "product_id": product_id,
                    "forecast_month": month,
                    "training_end_month": str(past["month"].max()),
                    "actual_quantity": int(actual),
                    "predicted_quantity": float(prediction),
                    "absolute_error": abs(error),
                    "squared_error": error ** 2,
                }
            )
    walk = pd.DataFrame(predictions)
    walk_metrics = regression_metrics(walk["actual_quantity"], walk["predicted_quantity"].to_numpy())
    result = pd.DataFrame(results)
    summary: dict[str, Any] = {
        "selected_model": winner,
        "selection_criterion": "Lowest mean training-only expanding-calendar CV MAE; ties use model order.",
        "best_parameters": searches[winner].best_params_,
        "cv_mae": float(-searches[winner].best_score_),
        "cv_folds": fold_descriptions,
        "models": {
            row["model"]: {
                "best_parameters": json.loads(row["best_parameters"]),
                "cv_mae": row["cv_mae"],
                "cv_mae_std": row["cv_mae_std"],
            }
            for row in results
        },
        "validation_metrics": validation_metrics,
        "walk_forward_metrics": walk_metrics,
        "walk_forward_rows": len(walk),
        "feature_columns": NUMERIC_FEATURES,
        "train_period": contract["splits"]["train"],
        "validation_period": contract["splits"]["validation"],
        "test_period": {
            "requested_start_month": SPLIT_PERIODS["test"][0],
            "requested_end_month": SPLIT_PERIODS["test"][1],
        },
        "methodology": (
            "Expanding calendar-month CV uses training rows only. The saved model is "
            "fitted only on the training period. Walk-forward clones refit using "
            "all eligible rows before each validation month; prior actual demand is "
            "assumed available. The final test period is not used for tuning or evaluation."
        ),
        "limitations": [
            "Only two training CV folds are available with the current three eligible training months.",
            "Validation estimates are reused for comparison and walk-forward assessment; future test evaluation remains necessary.",
            "Precomputed lags support one-step rolling forecasts, not fixed-origin multistep forecasts.",
        ],
        "model_path": str(Path(model_path)),
        "library_versions": {
            "python": sys.version.split()[0], "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__,
        },
    }
    results_path, summary_path, walk_path, model_path = (
        Path(results_path), Path(summary_path), Path(walk_path), Path(model_path)
    )
    for path in (results_path, summary_path, walk_path, model_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(results_path, index=False)
    walk.to_csv(walk_path, index=False)
    joblib.dump(selected, model_path)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
        file.write("\n")
    return result, summary, walk


if __name__ == "__main__":
    comparison, report, _ = run_forecasting_tuning()
    print(comparison.to_string(index=False))
    print(f"Selected: {report['selected_model']}; CV MAE: {report['cv_mae']:.4f}")
    print(f"Validation: {report['validation_metrics']}")
    print(f"Walk-forward: {report['walk_forward_metrics']}")

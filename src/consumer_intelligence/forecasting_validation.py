"""Chronological product-month validation splits for demand forecasting."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from consumer_intelligence.config import PROCESSED_DATA_DIR, REPORTS_DIR
from consumer_intelligence.forecasting_features import (
    CALENDAR_FEATURES,
    HISTORY_FEATURES,
    OUTPUT_COLUMNS,
)

INPUT_PATH = PROCESSED_DATA_DIR / "forecasting_features.csv"
METADATA_PATH = REPORTS_DIR / "forecasting_features_metadata.json"
SUMMARY_PATH = REPORTS_DIR / "forecasting_validation_summary.json"
SPLIT_PERIODS = {
    "train": ("2024-01", "2025-03"),
    "validation": ("2025-04", "2025-06"),
    "test": ("2025-07", "2025-12"),
}
NUMERIC_FEATURES = HISTORY_FEATURES + CALENDAR_FEATURES


def create_forecasting_splits(
    source: pd.DataFrame, metadata: dict[str, Any]
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Filter unavailable history and split on shared calendar cutoffs."""
    if metadata.get("feature_columns") != NUMERIC_FEATURES:
        raise ValueError("Feature metadata does not match the approved feature contract")
    if metadata.get("target_column") != "quantity_sold":
        raise ValueError("Feature metadata has an invalid target")
    missing_columns = set(OUTPUT_COLUMNS) - set(source.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if source.empty:
        raise ValueError("Forecasting feature dataset is empty")
    # Select only known fields: extra columns must never silently become features.
    frame = source[OUTPUT_COLUMNS].copy()
    for column in ("product_id", "month", "category", "brand"):
        values = frame[column].astype("string")
        if values.isna().any() or values.str.strip().eq("").any():
            raise ValueError(f"{column} must be present")
        frame[column] = values
    if not frame["month"].str.fullmatch(r"\d{4}-(0[1-9]|1[0-2])").all():
        raise ValueError("month must use YYYY-MM format")
    if frame.duplicated(["product_id", "month"]).any():
        raise ValueError("Product-month keys must be unique")
    for column in ["quantity_sold", *NUMERIC_FEATURES]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    target = frame["quantity_sold"]
    if (
        target.isna().any()
        or not np.isfinite(target.to_numpy(dtype=float)).all()
        or target.lt(0).any()
        or target.mod(1).ne(0).any()
    ):
        raise ValueError("quantity_sold must contain finite nonnegative integers")
    if np.isinf(frame[NUMERIC_FEATURES].to_numpy(dtype=float)).any():
        raise ValueError("Forecasting features must not contain infinite values")
    if frame[CALENDAR_FEATURES].isna().any().any():
        raise ValueError("Calendar features must be present")

    frame = frame.sort_values(["product_id", "month"]).reset_index(drop=True)
    eligible = frame.dropna(subset=HISTORY_FEATURES).copy()
    splits: dict[str, pd.DataFrame] = {}
    split_reports: dict[str, Any] = {}
    for name, (start, end) in SPLIT_PERIODS.items():
        split = eligible.loc[eligible["month"].between(start, end)].copy()
        if split.empty:
            raise ValueError(f"{name} split has no rows after historical feature filtering")
        splits[name] = split
        split_reports[name] = {
            "split": name,
            "requested_start_month": start,
            "requested_end_month": end,
            "start_month": str(split["month"].min()),
            "end_month": str(split["month"].max()),
            "row_count": len(split),
            "product_count": int(split["product_id"].nunique()),
            "target_mean": float(split["quantity_sold"].mean()),
            "target_sum": int(split["quantity_sold"].sum()),
            "missing_value_counts": {
                column: int(count) for column, count in split.isna().sum().items()
            },
        }

    summary: dict[str, Any] = {
        "target_column": "quantity_sold",
        "numeric_feature_columns": NUMERIC_FEATURES,
        "categorical_attributes": ["category", "brand"],
        "tracking_columns_not_model_features": ["product_id", "month"],
        "required_history_features": HISTORY_FEATURES,
        "source_row_count": len(frame),
        "rows_removed_for_unavailable_history": int(len(frame) - len(eligible)),
        "eligible_rows_outside_requested_periods": int(
            len(eligible) - sum(len(split) for split in splits.values())
        ),
        "splits": split_reports,
        "validation_methodology": (
            "Filter rows missing any required historical feature, then assign "
            "all products using the same inclusive calendar cutoffs; sort by "
            "product_id and month without shuffling. Targets are never imputed."
        ),
        "evaluation_assumptions": [
            "Rolling-origin, one-step-ahead evaluation: actual demand from prior "
            "validation or test months must be available before the next month is scored.",
            "These precomputed lags are not leakage-safe for a fixed-origin multi-month "
            "forecast unless unavailable future actuals are replaced by prior predictions.",
            "The feature builder assumes category and brand are known at prediction time "
            "and prior monthly demand has no reporting delay.",
            "No preprocessing statistics should be fit on validation or test data.",
        ],
    }
    return splits, summary


def run_forecasting_validation(
    input_path: Path = INPUT_PATH,
    metadata_path: Path = METADATA_PATH,
    summary_path: Path = SUMMARY_PATH,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load forecasting features and save the time-split summary."""
    with Path(metadata_path).open(encoding="utf-8") as file:
        metadata = json.load(file)
    source = pd.read_csv(
        Path(input_path),
        dtype={"product_id": "string", "month": "string", "category": "string", "brand": "string"},
    )
    splits, summary = create_forecasting_splits(source, metadata)
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
        file.write("\n")
    return splits, summary


if __name__ == "__main__":
    _, report = run_forecasting_validation()
    for name, info in report["splits"].items():
        print(
            f"{name}: {info['start_month']} to {info['end_month']}; "
            f"{info['row_count']} rows; {info['product_count']} products"
        )
    print(f"Summary: {SUMMARY_PATH}")

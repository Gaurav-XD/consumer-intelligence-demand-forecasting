"""Construct strictly historical monthly product demand features."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from consumer_intelligence.config import PROCESSED_DATA_DIR, REPORTS_DIR

INPUT_PATH = PROCESSED_DATA_DIR / "demand_forecasting_dataset.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "forecasting_features.csv"
METADATA_PATH = REPORTS_DIR / "forecasting_features_metadata.json"

LAGS = (1, 2, 3, 6, 12)
MEAN_WINDOWS = (3, 6, 12)
STD_WINDOWS = (3, 6)
HISTORY_FEATURES = (
    [f"lag_{lag}_quantity" for lag in LAGS]
    + [f"rolling_{window}_mean_quantity" for window in MEAN_WINDOWS]
    + [f"rolling_{window}_std_quantity" for window in STD_WINDOWS]
)
CALENDAR_FEATURES = ["month_of_year", "quarter", "trend_index"]
RETAINED_COLUMNS = ["product_id", "month", "category", "brand", "quantity_sold"]
OUTPUT_COLUMNS = RETAINED_COLUMNS + HISTORY_FEATURES + CALENDAR_FEATURES


def build_forecasting_features(source: pd.DataFrame) -> pd.DataFrame:
    """Create lag and full-window rolling features from prior product months."""
    missing_columns = set(RETAINED_COLUMNS) - set(source.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if source.empty:
        raise ValueError("Demand dataset is empty")
    result = source[RETAINED_COLUMNS].copy()
    for field in ("product_id", "month", "category", "brand"):
        values = result[field].astype("string")
        if values.isna().any() or values.str.strip().eq("").any():
            raise ValueError(f"{field} must be present")
        result[field] = values
    if not result["month"].str.fullmatch(r"\d{4}-(0[1-9]|1[0-2])").all():
        raise ValueError("month must use YYYY-MM format")
    result["month"] = pd.PeriodIndex(result["month"], freq="M")
    if result.duplicated(["product_id", "month"]).any():
        raise ValueError("Product-month keys must be unique")
    result["quantity_sold"] = pd.to_numeric(result["quantity_sold"], errors="raise")
    quantity = result["quantity_sold"]
    if (
        quantity.isna().any()
        or not np.isfinite(quantity.to_numpy(dtype=float)).all()
        or quantity.lt(0).any()
        or quantity.mod(1).ne(0).any()
    ):
        raise ValueError("quantity_sold must contain finite nonnegative integers")

    result = result.sort_values(["product_id", "month"]).reset_index(drop=True)
    month_ordinal = result["month"].astype("int64")
    gaps = month_ordinal.groupby(result["product_id"]).diff()
    if gaps.dropna().ne(1).any():
        raise ValueError("Each product must have consecutive calendar months")

    grouped = result.groupby("product_id", sort=False)["quantity_sold"]
    for lag in LAGS:
        result[f"lag_{lag}_quantity"] = grouped.shift(lag)
    for window in MEAN_WINDOWS:
        result[f"rolling_{window}_mean_quantity"] = grouped.transform(
            lambda values: values.shift(1).rolling(window, min_periods=window).mean()
        )
    for window in STD_WINDOWS:
        result[f"rolling_{window}_std_quantity"] = grouped.transform(
            lambda values: values.shift(1).rolling(window, min_periods=window).std(ddof=1)
        )
    result["month_of_year"] = result["month"].dt.month
    result["quarter"] = result["month"].dt.quarter
    result["trend_index"] = result["month"].dt.year * 12 + result["month"].dt.month - 1
    result["month"] = result["month"].astype(str)
    return result[OUTPUT_COLUMNS]


def run_forecasting_features(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    metadata_path: Path = METADATA_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Save historical demand features and their definitions."""
    source = pd.read_csv(
        Path(input_path),
        dtype={"product_id": "string", "month": "string", "category": "string", "brand": "string"},
    )
    result = build_forecasting_features(source)
    definitions = {
        **{
            f"lag_{lag}_quantity": f"Quantity sold exactly {lag} calendar months earlier."
            for lag in LAGS
        },
        **{
            f"rolling_{window}_mean_quantity": (
                f"Mean of the previous {window} months, excluding the current month; "
                "requires the full window."
            )
            for window in MEAN_WINDOWS
        },
        **{
            f"rolling_{window}_std_quantity": (
                f"Sample standard deviation (ddof=1) of the previous {window} months, "
                "excluding the current month; requires the full window."
            )
            for window in STD_WINDOWS
        },
        "month_of_year": "Calendar month number, 1 through 12.",
        "quarter": "Calendar quarter, 1 through 4.",
        "trend_index": "Absolute calendar month index: year * 12 + month_of_year - 1.",
    }
    metadata: dict[str, Any] = {
        "row_count": len(result),
        "product_count": int(result["product_id"].nunique()),
        "feature_count": len(HISTORY_FEATURES) + len(CALENDAR_FEATURES),
        "feature_columns": HISTORY_FEATURES + CALENDAR_FEATURES,
        "identifier_columns": ["product_id", "month"],
        "retained_product_attributes": ["category", "brand"],
        "target_column": "quantity_sold",
        "missing_history_counts": {
            column: int(result[column].isna().sum()) for column in HISTORY_FEATURES
        },
        "feature_definitions": definitions,
        "assumptions": [
            "Rows for each product must contain consecutive calendar months.",
            "Historical targets are available by the following month; no reporting delay is modeled.",
            "Category and brand are assumed available at prediction time; their historical versions are not tracked.",
            "Boundary months may have partial transaction coverage, as described in the demand dataset summary.",
        ],
    }
    output_path, metadata_path = Path(output_path), Path(metadata_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, allow_nan=False)
        file.write("\n")
    return result, metadata


if __name__ == "__main__":
    features, info = run_forecasting_features()
    print(f"Rows: {len(features)}; engineered features: {info['feature_count']}")
    print(f"Missing historical features: {info['missing_history_counts']}")

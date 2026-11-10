"""Interpretable customer-level revenue-based CLV estimates."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from consumer_intelligence.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR

FEATURE_PATH = PROCESSED_DATA_DIR / "customer_features.csv"
FEATURE_METADATA_PATH = REPORTS_DIR / "customer_features_metadata.json"
SEGMENT_PATH = PROCESSED_DATA_DIR / "customer_segments.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "customer_clv.csv"
SUMMARY_PATH = REPORTS_DIR / "clv_summary.json"
SEGMENT_OUTPUT_PATH = REPORTS_DIR / "clv_by_segment.csv"
ANNUAL_DAYS = 365.25
MIN_EXPOSURE_DAYS = 365


def _validate_ids(frame: pd.DataFrame, column: str, name: str) -> None:
    if frame[column].isna().any() or frame[column].str.strip().eq("").any():
        raise ValueError(f"{name} contains missing {column}")
    if frame[column].duplicated().any():
        raise ValueError(f"{name} contains duplicate {column}")


def run_clv(
    raw_data_dir: Path = RAW_DATA_DIR,
    feature_path: Path = FEATURE_PATH,
    feature_metadata_path: Path = FEATURE_METADATA_PATH,
    segment_path: Path = SEGMENT_PATH,
    output_path: Path = OUTPUT_PATH,
    summary_path: Path = SUMMARY_PATH,
    segment_output_path: Path = SEGMENT_OUTPUT_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate a one-year revenue proxy at the recorded feature snapshot."""
    raw_data_dir = Path(raw_data_dir)
    with Path(feature_metadata_path).open(encoding="utf-8") as file:
        feature_metadata = json.load(file)
    snapshot = pd.Timestamp(feature_metadata["as_of_date"])
    if pd.isna(snapshot) or snapshot.tz is not None or snapshot != snapshot.normalize():
        raise ValueError("Feature snapshot must be a timezone-naive calendar date")

    customers = pd.read_csv(
        raw_data_dir / "customers.csv", dtype={"customer_id": "string"}
    )
    transactions = pd.read_csv(
        raw_data_dir / "transactions.csv",
        dtype={"customer_id": "string", "product_id": "string", "transaction_id": "string"},
    )
    products = pd.read_csv(raw_data_dir / "products.csv", dtype={"product_id": "string"})
    features = pd.read_csv(Path(feature_path), dtype={"customer_id": "string"})
    _validate_ids(customers, "customer_id", "customers")
    _validate_ids(products, "product_id", "products")
    _validate_ids(features, "customer_id", "customer features")
    customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="raise")
    transactions["transaction_date"] = pd.to_datetime(
        transactions["transaction_date"], errors="raise"
    )
    if customers["signup_date"].isna().any() or transactions["transaction_date"].isna().any():
        raise ValueError("Customer signup and transaction dates must be present")
    if transactions.empty:
        raise ValueError("Transaction history is empty; observation start cannot be determined")

    customers = customers.loc[customers["signup_date"].le(snapshot)].copy()
    history = transactions.loc[transactions["transaction_date"].le(snapshot)].copy()
    if history.empty:
        raise ValueError("No transaction history is available at the snapshot")
    _validate_ids(history, "transaction_id", "historical transactions")
    if history[["customer_id", "product_id"]].isna().any().any():
        raise ValueError("Historical transaction customer and product IDs must be present")
    if not history["customer_id"].isin(customers["customer_id"]).all():
        raise ValueError("Historical transactions contain unknown or future customer IDs")
    if not history["product_id"].isin(products["product_id"]).all():
        raise ValueError("Historical transactions contain unknown product IDs")
    history = history.merge(
        customers[["customer_id", "signup_date"]],
        on="customer_id", how="left", validate="many_to_one",
    )
    if history["transaction_date"].lt(history["signup_date"]).any():
        raise ValueError("A historical transaction predates its customer's signup")
    for column in ("quantity", "price", "discount"):
        history[column] = pd.to_numeric(history[column], errors="raise")
    numbers = history[["quantity", "price", "discount"]]
    if not np.isfinite(numbers.to_numpy(dtype=float)).all():
        raise ValueError("Historical transaction numeric values must be finite")
    if (
        history["quantity"].le(0).any()
        or history["quantity"].mod(1).ne(0).any()
        or history["price"].le(0).any()
        or not history["discount"].between(0, 1).all()
    ):
        raise ValueError("Historical transactions contain invalid quantity, price, or discount")
    history["revenue"] = (
        history["quantity"] * history["price"] * (1 - history["discount"])
    ).round(2)
    aggregate = history.groupby("customer_id").agg(
        historical_revenue=("revenue", "sum"),
        total_orders=("transaction_id", "nunique"),
        first_purchase=("transaction_date", "min"),
        last_purchase=("transaction_date", "max"),
    )
    if set(features["customer_id"]) != set(customers["customer_id"]):
        raise ValueError("Customer feature IDs do not match customers at the snapshot")
    result = customers[["customer_id", "signup_date"]].merge(
        aggregate, left_on="customer_id", right_index=True, how="left", validate="one_to_one"
    )
    result["total_orders"] = result["total_orders"].fillna(0).astype("int64")
    result["historical_revenue"] = result["historical_revenue"].fillna(0).round(2)
    result["average_order_value"] = (
        result["historical_revenue"] / result["total_orders"].where(result["total_orders"].gt(0))
    ).fillna(0).round(2)
    result["customer_lifetime_days"] = (
        result["last_purchase"] - result["first_purchase"]
    ).dt.days.astype("Int64")

    feature_check = result.merge(
        features[["customer_id", "total_orders", "total_revenue"]],
        on="customer_id", validate="one_to_one",
    )
    if (
        not feature_check["total_orders_x"].eq(feature_check["total_orders_y"]).all()
        or not np.isclose(
            feature_check["historical_revenue"], feature_check["total_revenue"],
            rtol=0, atol=0.005,
        ).all()
    ):
        raise ValueError("Customer features do not match raw history at the recorded snapshot")

    observation_start = history["transaction_date"].min().normalize()
    effective_start = result["signup_date"].clip(lower=observation_start)
    result["observed_days"] = (snapshot - effective_start).dt.days + 1
    result["annualization_days"] = result["observed_days"].clip(lower=MIN_EXPOSURE_DAYS)
    result["purchase_frequency"] = (
        result["total_orders"] * ANNUAL_DAYS / result["annualization_days"]
    )
    result["estimated_annual_revenue"] = (
        result["historical_revenue"] * ANNUAL_DAYS / result["annualization_days"]
    ).round(2)
    result["estimated_clv"] = result["estimated_annual_revenue"]
    result = result.sort_values(
        ["estimated_clv", "customer_id"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    result.insert(1, "clv_rank", range(1, len(result) + 1))
    result = result[
        [
            "customer_id", "clv_rank", "historical_revenue", "total_orders",
            "average_order_value", "purchase_frequency", "customer_lifetime_days",
            "observed_days", "annualization_days", "estimated_annual_revenue",
            "estimated_clv",
        ]
    ]

    segment_columns = [
        "segment", "customer_count", "historical_revenue", "total_estimated_clv",
        "mean_estimated_clv", "median_estimated_clv",
    ]
    segment_path = Path(segment_path)
    segment_summary = pd.DataFrame(columns=segment_columns)
    segment_available = segment_path.is_file()
    if segment_available:
        segments = pd.read_csv(segment_path, dtype={"customer_id": "string"})
        _validate_ids(segments, "customer_id", "customer segments")
        if segments["segment"].isna().any() or segments["segment"].str.strip().eq("").any():
            raise ValueError("Customer segment assignments must be present")
        if set(segments["customer_id"]) != set(features["customer_id"]):
            raise ValueError("Customer segment IDs do not match the feature snapshot")
        # RFM input columns help detect stale segment files before reporting them.
        compare_columns = ["total_orders", "total_revenue", "days_since_last_purchase"]
        comparison = features[["customer_id", *compare_columns]].merge(
            segments[["customer_id", *compare_columns]],
            on="customer_id", validate="one_to_one", suffixes=("_feature", "_segment"),
        )
        for column in compare_columns:
            if not np.allclose(
                comparison[f"{column}_feature"].to_numpy(dtype=float),
                comparison[f"{column}_segment"].to_numpy(dtype=float),
                rtol=0, atol=1e-8, equal_nan=True,
            ):
                raise ValueError(f"Customer segments have stale {column} values")
        labeled = result.merge(
            segments[["customer_id", "segment"]], on="customer_id", validate="one_to_one"
        )
        segment_summary = (
            labeled.groupby("segment", as_index=False)
            .agg(
                customer_count=("customer_id", "size"),
                historical_revenue=("historical_revenue", "sum"),
                total_estimated_clv=("estimated_clv", "sum"),
                mean_estimated_clv=("estimated_clv", "mean"),
                median_estimated_clv=("estimated_clv", "median"),
            )
            .sort_values(["total_estimated_clv", "segment"], ascending=[False, True])
        )
        for column in segment_columns[2:]:
            segment_summary[column] = segment_summary[column].round(2)

    summary: dict[str, Any] = {
        "snapshot_date": snapshot.date().isoformat(),
        "observation_start_proxy": observation_start.date().isoformat(),
        "customer_count": len(result),
        "total_historical_revenue": round(float(result["historical_revenue"].sum()), 2),
        "mean_estimated_clv": round(float(result["estimated_clv"].mean()), 2) if len(result) else None,
        "median_estimated_clv": round(float(result["estimated_clv"].median()), 2) if len(result) else None,
        "total_estimated_clv": round(float(result["estimated_clv"].sum()), 2),
        "top_customers": result.head(10)[
            ["customer_id", "clv_rank", "historical_revenue", "estimated_clv"]
        ].to_dict(orient="records"),
        "segment_summary_available": segment_available,
        "formula": {
            "historical_revenue": "sum(round(quantity * price * (1 - discount), 2)) through snapshot",
            "observed_days": "inclusive days from max(signup date, observed transaction start) through snapshot",
            "annualization_days": "max(observed_days, 365)",
            "purchase_frequency": "total_orders * 365.25 / annualization_days",
            "estimated_annual_revenue": "round(historical_revenue * 365.25 / annualization_days, 2)",
            "estimated_clv": "estimated_annual_revenue (one-year forward revenue proxy)",
        },
        "limitations": [
            "This is a one-year revenue run-rate proxy, not discounted lifetime profit.",
            "No retention, churn, margins, seasonality or future behavior are modeled.",
            "The first observed transaction date is only a proxy for the start of complete coverage.",
            "A 365-day minimum exposure dampens short-history extrapolation; no-purchase customers receive zero.",
            "Segment input matching checks RFM values but cannot establish full segment provenance.",
        ],
    }
    output_path, summary_path, segment_output_path = (
        Path(output_path), Path(summary_path), Path(segment_output_path)
    )
    for path in (output_path, summary_path, segment_output_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    segment_summary.to_csv(segment_output_path, index=False)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
        file.write("\n")
    return result, summary


if __name__ == "__main__":
    clv, info = run_clv()
    print(f"Customers: {len(clv)}; snapshot: {info['snapshot_date']}")
    print(f"Historical revenue: {info['total_historical_revenue']:.2f}")
    print(f"Mean CLV: {info['mean_estimated_clv']:.2f}")

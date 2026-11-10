"""Build a temporally separated customer churn modeling dataset."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from consumer_intelligence.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR
from consumer_intelligence.customer_features import _read_inputs, build_customer_features

FEATURES_PATH = PROCESSED_DATA_DIR / "customer_features.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "churn_dataset.csv"
METADATA_PATH = REPORTS_DIR / "churn_dataset_metadata.json"
TRAIN_SNAPSHOT = "2025-03-31"
TEST_SNAPSHOT = "2025-09-30"
OBSERVATION_DAYS = 90

# Exclude demographic fields with no point-in-time history and redundant RFM aliases.
MODEL_FEATURE_COLUMNS = [
    "customer_tenure_days",
    "total_orders",
    "total_quantity",
    "total_revenue",
    "average_order_value",
    "average_discount",
    "unique_products_purchased",
    "unique_categories_purchased",
    "purchase_frequency",
    "average_days_between_purchases",
    "days_since_last_purchase",
    "customer_lifetime_days",
    "number_marketing_interactions",
    "number_marketing_campaigns",
]


def _customer_split(customer_id: str) -> str:
    digest = hashlib.sha256(customer_id.encode("utf-8")).digest()
    return "test" if int.from_bytes(digest[:8], "big") % 10 < 2 else "train"


def _snapshot_date(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tz is not None or timestamp != timestamp.normalize():
        raise ValueError("Snapshots must be valid, timezone-naive calendar dates")
    return timestamp


def _labeled_snapshot(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    products: pd.DataFrame,
    marketing: pd.DataFrame,
    snapshot: pd.Timestamp,
    observation_days: int,
) -> pd.DataFrame:
    features, _ = build_customer_features(
        customers, transactions, products, marketing, as_of_date=snapshot
    )
    observation_end = snapshot + pd.Timedelta(days=observation_days)
    dates = transactions["transaction_date"]
    future_customers = transactions.loc[
        dates.gt(snapshot) & dates.le(observation_end), "customer_id"
    ]
    result = features[["customer_id", *MODEL_FEATURE_COLUMNS]].copy()
    result["churn"] = (~result["customer_id"].isin(future_customers)).astype("int64")
    result["snapshot_date"] = snapshot.date().isoformat()
    return result


def run_churn_dataset(
    raw_data_dir: Path = RAW_DATA_DIR,
    customer_features_path: Path = FEATURES_PATH,
    output_path: Path = OUTPUT_PATH,
    metadata_path: Path = METADATA_PATH,
    train_snapshot: str = TRAIN_SNAPSHOT,
    test_snapshot: str = TEST_SNAPSHOT,
    observation_days: int = OBSERVATION_DAYS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create one labeled row per customer using non-overlapping temporal splits."""
    if observation_days <= 0:
        raise ValueError("observation_days must be positive")
    train_date = _snapshot_date(train_snapshot)
    test_date = _snapshot_date(test_snapshot)
    train_end = train_date + pd.Timedelta(days=observation_days)
    test_end = test_date + pd.Timedelta(days=observation_days)
    if train_end >= test_date:
        raise ValueError("The training observation window must end before the test snapshot")

    customers, transactions, products, marketing = _read_inputs(Path(raw_data_dir))
    cached_customers = pd.read_csv(
        Path(customer_features_path), usecols=["customer_id"], dtype={"customer_id": "string"}
    )
    for name, frame in (("customers", customers), ("customer_features", cached_customers)):
        if frame["customer_id"].isna().any() or frame["customer_id"].duplicated().any():
            raise ValueError(f"{name} must contain unique, non-missing customer IDs")
    if set(customers["customer_id"]) != set(cached_customers["customer_id"]):
        raise ValueError("Raw customer IDs and existing customer feature IDs do not match")
    for name, frame, column in (
        ("transactions", transactions, "transaction_date"),
        ("marketing", marketing, "campaign_date"),
    ):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
        if frame[column].isna().any() or frame["customer_id"].isna().any():
            raise ValueError(f"{name} contains missing dates or customer IDs")
        if not frame["customer_id"].isin(customers["customer_id"]).all():
            raise ValueError(f"{name} contains unknown customer IDs")
    customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="raise")
    if customers["signup_date"].isna().any():
        raise ValueError("customers contains missing signup dates")
    if transactions.empty or transactions["transaction_date"].max() < test_end:
        raise ValueError("Transaction history does not extend through the test observation window")

    # Construct outcomes for both snapshots before assigning customer-disjoint splits.
    train_candidates = _labeled_snapshot(
        customers, transactions, products, marketing, train_date, observation_days
    )
    test_candidates = _labeled_snapshot(
        customers, transactions, products, marketing, test_date, observation_days
    )
    train_rows = train_candidates.loc[
        train_candidates["customer_id"].map(_customer_split).eq("train")
    ].copy()
    test_rows = test_candidates.loc[
        test_candidates["customer_id"].map(_customer_split).eq("test")
    ].copy()
    train_rows["split"] = "train"
    test_rows["split"] = "test"
    dataset = pd.concat([train_rows, test_rows], ignore_index=True)
    dataset = dataset[["customer_id", "split", "snapshot_date", *MODEL_FEATURE_COLUMNS, "churn"]]
    if dataset["customer_id"].duplicated().any():
        raise ValueError("A customer cannot appear in both train and test splits")

    first_transaction = transactions["transaction_date"].min().date().isoformat()
    periods: dict[str, Any] = {}
    for split_name, snapshot, end in (
        ("train", train_date, train_end),
        ("test", test_date, test_end),
    ):
        split_rows = dataset.loc[dataset["split"].eq(split_name)]
        periods[split_name] = {
            "feature_snapshot_period": {
                "transaction_history_start": first_transaction,
                "snapshot_date_inclusive": snapshot.date().isoformat(),
            },
            "churn_observation_window": {
                "start_inclusive": (snapshot + pd.Timedelta(days=1)).date().isoformat(),
                "end_inclusive": end.date().isoformat(),
                "days": observation_days,
            },
            "row_count": len(split_rows),
            "target_counts": {
                "0": int(split_rows["churn"].eq(0).sum()),
                "1": int(split_rows["churn"].eq(1).sum()),
            },
        }
    metadata: dict[str, Any] = {
        "row_count": len(dataset),
        "feature_count": len(MODEL_FEATURE_COLUMNS),
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "identifier_columns_not_model_features": ["customer_id", "split", "snapshot_date"],
        "target_column": "churn",
        "churn_definition": (
            "1 if no purchase occurs in the observation window after the feature "
            "snapshot, otherwise 0. Customers with no prior purchases are included."
        ),
        "target_distribution": {
            "0": int(dataset["churn"].eq(0).sum()),
            "1": int(dataset["churn"].eq(1).sum()),
        },
        "churn_rate": float(dataset["churn"].mean()) if len(dataset) else None,
        "split_method": (
            "Define outcomes at both snapshots first; assign customer IDs "
            "deterministically via SHA-256 modulo 10, buckets 0-1 to the later "
            "test snapshot and 2-9 to the earlier train snapshot. Customers never overlap."
        ),
        "periods": periods,
        "source_coverage_assumption": (
            "The maximum observed transaction date is used as a coverage proxy; "
            "it does not independently prove source completeness."
        ),
        "customer_features_usage": (
            "Only customer IDs from the existing feature file are used to verify "
            "the customer universe; behavioral values are rebuilt at each snapshot."
        ),
        "limitations": [
            "The dataset uses static product categories because no category history is available.",
            "Age, gender and region are excluded because historical versions are unavailable.",
            "Undefined averages and recency values are left missing for downstream handling.",
        ],
    }

    output_path = Path(output_path)
    metadata_path = Path(metadata_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, allow_nan=False)
        file.write("\n")
    return dataset, metadata


if __name__ == "__main__":
    result, info = run_churn_dataset()
    print(f"Customers: {len(result)}; features: {info['feature_count']}")
    print(f"Churn rate: {info['churn_rate']:.2%}")
    print(f"Dataset: {OUTPUT_PATH}")
    print(f"Metadata: {METADATA_PATH}")

"""Build point-in-time customer-level features from raw retail CSV files."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from consumer_intelligence.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
)

FEATURE_COLUMNS = [
    "customer_id",
    "age",
    "gender",
    "region",
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
    "first_purchase_date",
    "last_purchase_date",
    "number_marketing_interactions",
    "number_marketing_campaigns",
    "recency_days",
    "frequency_orders",
    "monetary_total_revenue",
]

FEATURE_DEFINITIONS = {
    "customer_id": "Customer identifier from customers.csv.",
    "age": "Customer age from customers.csv.",
    "gender": "Customer gender from customers.csv.",
    "region": "Customer region from customers.csv.",
    "customer_tenure_days": "Days from signup date through the as-of date.",
    "total_orders": "Distinct transactions on or before the as-of date.",
    "total_quantity": "Units purchased on or before the as-of date.",
    "total_revenue": "Net item revenue: quantity * price * (1 - discount).",
    "average_order_value": "Mean net revenue across distinct transactions.",
    "average_discount": "Mean transaction-line discount rate.",
    "unique_products_purchased": "Distinct purchased product IDs.",
    "unique_categories_purchased": "Distinct purchased product categories.",
    "purchase_frequency": "Total orders per 365.25 customer-tenure days.",
    "average_days_between_purchases": (
        "Mean days between ordered transactions; undefined for fewer than two."
    ),
    "days_since_last_purchase": "Days from last purchase through the as-of date.",
    "customer_lifetime_days": (
        "Days between first and last observed purchases; undefined without purchases."
    ),
    "first_purchase_date": "Earliest observed transaction date.",
    "last_purchase_date": "Latest observed transaction date.",
    "number_marketing_interactions": (
        "Marketing records on or before the as-of date."
    ),
    "number_marketing_campaigns": (
        "Distinct campaigns with records on or before the as-of date."
    ),
    "recency_days": "Days since last purchase, equivalent to days_since_last_purchase.",
    "frequency_orders": "RFM frequency, equivalent to total_orders.",
    "monetary_total_revenue": "RFM monetary value, equivalent to total_revenue.",
}


def _read_inputs(raw_data_dir: Path) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    customers = pd.read_csv(
        raw_data_dir / "customers.csv",
        dtype={"customer_id": "string"},
        parse_dates=["signup_date"],
    )
    transactions = pd.read_csv(
        raw_data_dir / "transactions.csv",
        dtype={
            "transaction_id": "string",
            "customer_id": "string",
            "product_id": "string",
        },
        parse_dates=["transaction_date"],
    )
    products = pd.read_csv(
        raw_data_dir / "products.csv",
        dtype={"product_id": "string"},
    )
    marketing = pd.read_csv(
        raw_data_dir / "marketing.csv",
        dtype={"customer_id": "string"},
        parse_dates=["campaign_date"],
    )
    return customers, transactions, products, marketing


def build_customer_features(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    products: pd.DataFrame,
    marketing: pd.DataFrame,
    as_of_date: pd.Timestamp | str | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Build customer features using only records observed by the as-of date."""
    transactions = transactions.copy()
    customers = customers.copy()
    products = products[["product_id", "category"]].drop_duplicates("product_id")
    marketing = marketing.copy()

    transactions["transaction_date"] = pd.to_datetime(
        transactions["transaction_date"], errors="raise"
    )
    customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="raise")
    marketing["campaign_date"] = pd.to_datetime(
        marketing["campaign_date"], errors="raise"
    )

    if as_of_date is None:
        if transactions.empty:
            raise ValueError("Cannot infer as_of_date from an empty transaction dataset")
        snapshot = transactions["transaction_date"].max()
    else:
        snapshot = pd.Timestamp(as_of_date)
    if pd.isna(snapshot):
        raise ValueError("as_of_date must be a valid date")

    customers = customers.loc[customers["signup_date"] <= snapshot].copy()
    transactions = transactions.loc[
        (transactions["transaction_date"] <= snapshot)
        & transactions["customer_id"].isin(customers["customer_id"])
    ].copy()
    marketing = marketing.loc[
        (marketing["campaign_date"] <= snapshot)
        & marketing["customer_id"].isin(customers["customer_id"])
    ].copy()

    if transactions["transaction_id"].duplicated().any():
        raise ValueError("transaction_id must be unique to calculate order metrics")

    transactions["quantity"] = pd.to_numeric(
        transactions["quantity"], errors="raise"
    )
    transactions["price"] = pd.to_numeric(transactions["price"], errors="raise")
    transactions["discount"] = pd.to_numeric(
        transactions["discount"], errors="raise"
    )
    transactions["revenue"] = (
        transactions["quantity"]
        * transactions["price"]
        * (1 - transactions["discount"])
    ).round(2)
    transactions = transactions.merge(
        products, on="product_id", how="left", validate="many_to_one"
    )

    orders = (
        transactions.groupby(["customer_id", "transaction_id"], as_index=False)
        .agg(
            order_date=("transaction_date", "min"),
            order_revenue=("revenue", "sum"),
            order_quantity=("quantity", "sum"),
        )
        .sort_values(["customer_id", "order_date", "transaction_id"])
    )
    order_features = (
        orders.groupby("customer_id")
        .agg(
            total_orders=("transaction_id", "nunique"),
            total_revenue=("order_revenue", "sum"),
            total_quantity=("order_quantity", "sum"),
            average_order_value=("order_revenue", "mean"),
            first_purchase_date=("order_date", "min"),
            last_purchase_date=("order_date", "max"),
        )
    )
    order_features["customer_lifetime_days"] = (
        order_features["last_purchase_date"] - order_features["first_purchase_date"]
    ).dt.days
    order_features["days_since_last_purchase"] = (
        snapshot - order_features["last_purchase_date"]
    ).dt.days
    order_features["average_days_between_purchases"] = (
        order_features["customer_lifetime_days"]
        / (order_features["total_orders"] - 1).where(
            order_features["total_orders"] > 1
        )
    )
    transaction_features = transactions.groupby("customer_id").agg(
        average_discount=("discount", "mean"),
        unique_products_purchased=("product_id", "nunique"),
        unique_categories_purchased=("category", "nunique"),
    )
    marketing_features = (
        marketing.groupby("customer_id")
        .agg(
            number_marketing_interactions=("campaign", "size"),
            number_marketing_campaigns=("campaign", "nunique"),
        )
    )

    features = customers[["customer_id", "age", "gender", "region", "signup_date"]].copy()
    features["customer_tenure_days"] = (snapshot - features["signup_date"]).dt.days
    features = features.drop(columns="signup_date").set_index("customer_id")
    features = features.join(order_features, how="left")
    features = features.join(transaction_features, how="left")
    features = features.join(marketing_features, how="left")

    count_columns = [
        "total_orders",
        "total_quantity",
        "unique_products_purchased",
        "unique_categories_purchased",
        "number_marketing_interactions",
        "number_marketing_campaigns",
    ]
    features[count_columns] = features[count_columns].fillna(0).astype("int64")
    features["total_revenue"] = features["total_revenue"].fillna(0.0).round(2)
    features["average_order_value"] = features["average_order_value"].round(2)
    features["average_discount"] = features["average_discount"]
    features["purchase_frequency"] = (
        features["total_orders"] / (features["customer_tenure_days"] / 365.25)
    ).where(features["customer_tenure_days"] > 0)
    features["recency_days"] = features["days_since_last_purchase"]
    features["frequency_orders"] = features["total_orders"]
    features["monetary_total_revenue"] = features["total_revenue"]
    features = features.reset_index()[FEATURE_COLUMNS]
    return features, snapshot


def run_customer_feature_engineering(
    raw_data_dir: Path = RAW_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR / "customer_features.csv",
    metadata_path: Path = REPORTS_DIR / "customer_features_metadata.json",
    as_of_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Create the customer feature table and write its feature metadata."""
    raw_data_dir = Path(raw_data_dir)
    output_path = Path(output_path)
    metadata_path = Path(metadata_path)
    customers, transactions, products, marketing = _read_inputs(raw_data_dir)
    features, snapshot = build_customer_features(
        customers,
        transactions,
        products,
        marketing,
        as_of_date=as_of_date,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    metadata: dict[str, Any] = {
        "row_count": len(features),
        "feature_count": len(FEATURE_COLUMNS) - 1,
        "column_count": len(FEATURE_COLUMNS),
        "as_of_date": snapshot.date().isoformat(),
        "source_datasets": [
            "customers.csv",
            "transactions.csv",
            "products.csv",
            "marketing.csv",
        ],
        "features": {
            name: description for name, description in FEATURE_DEFINITIONS.items()
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as json_file:
        json.dump(metadata, json_file, indent=2)
        json_file.write("\n")
    return features


if __name__ == "__main__":
    customer_features = run_customer_feature_engineering()
    print(f"Customers: {len(customer_features)}")
    print(f"Features: {len(FEATURE_COLUMNS) - 1}")
    print(f"Saved to: {PROCESSED_DATA_DIR / 'customer_features.csv'}")
    print(f"Metadata: {REPORTS_DIR / 'customer_features_metadata.json'}")

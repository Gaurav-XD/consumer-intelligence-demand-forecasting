"""Build a monthly product-level dataset from observed transaction history."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from consumer_intelligence.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR

OUTPUT_PATH = PROCESSED_DATA_DIR / "demand_forecasting_dataset.csv"
SUMMARY_PATH = REPORTS_DIR / "forecasting_dataset_summary.json"


def build_forecasting_dataset(
    products: pd.DataFrame, transactions: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate observed transactions on a complete product-month grid."""
    products = products.copy()
    transactions = transactions.copy()
    for frame, fields, name in (
        (products, ("product_id", "category", "brand"), "products"),
        (
            transactions,
            ("transaction_id", "customer_id", "product_id", "transaction_date"),
            "transactions",
        ),
    ):
        for field in fields:
            if frame[field].isna().any() or frame[field].astype("string").str.strip().eq("").any():
                raise ValueError(f"{name} contains missing {field}")
    if products.empty:
        raise ValueError("Product catalog is empty")
    if transactions.empty:
        raise ValueError("Transaction history is empty; date range cannot be determined")
    if products["product_id"].duplicated().any():
        raise ValueError("Product IDs must be unique")
    if transactions["transaction_id"].duplicated().any():
        raise ValueError("Transaction IDs must be unique")
    if not transactions["product_id"].isin(products["product_id"]).all():
        raise ValueError("Transactions reference unknown product IDs")
    transactions["transaction_date"] = pd.to_datetime(
        transactions["transaction_date"], errors="raise"
    )
    if transactions["transaction_date"].isna().any():
        raise ValueError("Transaction dates must be present")
    for field in ("quantity", "price", "discount"):
        transactions[field] = pd.to_numeric(transactions[field], errors="raise")
    if not np.isfinite(transactions[["quantity", "price", "discount"]].to_numpy(dtype=float)).all():
        raise ValueError("Transaction numeric values must be finite")
    if (
        transactions["quantity"].le(0).any()
        or transactions["quantity"].mod(1).ne(0).any()
        or transactions["price"].le(0).any()
        or not transactions["discount"].between(0, 1).all()
    ):
        raise ValueError("Invalid transaction quantity, price, or discount")

    start_date = transactions["transaction_date"].min()
    end_date = transactions["transaction_date"].max()
    months = pd.period_range(start_date.to_period("M"), end_date.to_period("M"), freq="M")
    transactions["month"] = transactions["transaction_date"].dt.to_period("M")
    transactions["revenue"] = (
        transactions["quantity"] * transactions["price"] * (1 - transactions["discount"])
    )
    grouped = transactions.groupby(["product_id", "month"], as_index=False).agg(
        quantity_sold=("quantity", "sum"),
        revenue=("revenue", "sum"),
        transaction_count=("transaction_id", "nunique"),
        unique_customers=("customer_id", "nunique"),
        average_price=("price", "mean"),
        average_discount=("discount", "mean"),
    )
    grid = pd.MultiIndex.from_product(
        [products["product_id"], months], names=["product_id", "month"]
    ).to_frame(index=False)
    result = grid.merge(
        products[["product_id", "category", "brand"]],
        on="product_id", validate="many_to_one",
    ).merge(grouped, on=["product_id", "month"], how="left", validate="one_to_one")
    count_columns = ["quantity_sold", "transaction_count", "unique_customers"]
    result[count_columns] = result[count_columns].fillna(0).astype("int64")
    result["revenue"] = result["revenue"].fillna(0.0)
    result["month"] = result["month"].astype(str)
    result = result[
        ["product_id", "month", "category", "brand", "quantity_sold", "revenue",
         "transaction_count", "unique_customers", "average_price", "average_discount"]
    ].sort_values(["product_id", "month"]).reset_index(drop=True)

    summary: dict[str, Any] = {
        "product_count": len(products),
        "month_count": len(months),
        "row_count": len(result),
        "total_quantity": int(result["quantity_sold"].sum()),
        "total_revenue": float(result["revenue"].sum()),
        "transaction_count": len(transactions),
        "zero_sales_product_months": int(result["transaction_count"].eq(0).sum()),
        "transaction_date_range": {
            "minimum": start_date.date().isoformat(),
            "maximum": end_date.date().isoformat(),
        },
        "month_range": {"minimum": str(months.min()), "maximum": str(months.max())},
        "definitions": {
            "revenue": "sum(quantity * price * (1 - discount)); no per-transaction rounding",
            "average_price": "arithmetic mean of undiscounted unit prices per transaction",
            "average_discount": "arithmetic mean of transaction discount fractions",
            "empty_month": "quantity, revenue and counts are zero; price and discount averages are missing",
            "calendar": "all catalog products crossed with every month from the first to last transaction, inclusive",
            "boundary_months": "first and last months may be partial; actual transaction date bounds are retained above",
            "feature_warning": "Same-month price, discount, revenue and counts are contemporaneous outcomes, not safe predictors of that month's demand without a separate as-of policy.",
        },
    }
    return result, summary


def run_forecasting_dataset(
    raw_data_dir: Path = RAW_DATA_DIR,
    output_path: Path = OUTPUT_PATH,
    summary_path: Path = SUMMARY_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read raw CSVs and write a product-month table and summary."""
    raw_data_dir = Path(raw_data_dir)
    products = pd.read_csv(raw_data_dir / "products.csv", dtype={"product_id": "string"})
    transactions = pd.read_csv(
        raw_data_dir / "transactions.csv",
        dtype={"product_id": "string", "customer_id": "string", "transaction_id": "string"},
    )
    result, summary = build_forecasting_dataset(products, transactions)
    output_path, summary_path = Path(output_path), Path(summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
        file.write("\n")
    return result, summary


if __name__ == "__main__":
    _, report = run_forecasting_dataset()
    print(f"Products: {report['product_count']}; months: {report['month_count']}; rows: {report['row_count']}")
    print(f"Quantity: {report['total_quantity']}; revenue: {report['total_revenue']:.2f}")
    print(f"Transaction dates: {report['transaction_date_range']}")

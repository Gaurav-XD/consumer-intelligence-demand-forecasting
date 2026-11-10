"""Exploratory summaries and plots for the raw retail datasets."""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from consumer_intelligence.config import RAW_DATA_DIR, REPORTS_DIR

EDA_OUTPUT_DIR = REPORTS_DIR / "eda"
DATASET_FILES = {
    "customers": "customers.csv",
    "products": "products.csv",
    "transactions": "transactions.csv",
    "marketing": "marketing.csv",
}


def _load_datasets(raw_data_dir: Path) -> dict[str, pd.DataFrame]:
    datasets = {
        name: pd.read_csv(
            raw_data_dir / filename,
            dtype={
                "customer_id": "string",
                "product_id": "string",
                "transaction_id": "string",
            },
        )
        for name, filename in DATASET_FILES.items()
    }
    for column in ("signup_date",):
        datasets["customers"][column] = pd.to_datetime(
            datasets["customers"][column], errors="raise"
        )
    for column in ("transaction_date",):
        datasets["transactions"][column] = pd.to_datetime(
            datasets["transactions"][column], errors="raise"
        )
    datasets["marketing"]["campaign_date"] = pd.to_datetime(
        datasets["marketing"]["campaign_date"], errors="raise"
    )
    for dataset_name, columns in {
        "customers": ("age",),
        "products": ("cost",),
        "transactions": ("quantity", "price", "discount"),
    }.items():
        for column in columns:
            datasets[dataset_name][column] = pd.to_numeric(
                datasets[dataset_name][column], errors="raise"
            )
    return datasets


def _customer_demographics(customers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension in ("gender", "region"):
        counts = customers[dimension].value_counts(dropna=False).sort_index()
        for category, count in counts.items():
            rows.append(
                {
                    "dimension": dimension,
                    "category": category,
                    "customer_count": int(count),
                    "customer_share": (
                        float(count / len(customers)) if len(customers) else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _purchase_frequency(
    customers: pd.DataFrame, transactions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = transactions.groupby("customer_id").size().rename("transaction_count")
    frequency = customers[["customer_id"]].merge(
        counts, how="left", left_on="customer_id", right_index=True
    )
    frequency["transaction_count"] = (
        frequency["transaction_count"].fillna(0).astype("int64")
    )
    frequency["frequency_segment"] = pd.cut(
        frequency["transaction_count"],
        bins=[-1, 0, 1, 5, 12, float("inf")],
        labels=[
            "No purchases",
            "1 purchase",
            "2-5 purchases",
            "6-12 purchases",
            "13+ purchases",
        ],
    ).astype("string")
    distribution = (
        frequency.groupby("frequency_segment", observed=False)
        .size()
        .rename("customer_count")
        .reset_index()
    )
    return frequency, distribution


def _monthly_trends(transactions: pd.DataFrame) -> pd.DataFrame:
    monthly = transactions.copy()
    monthly["month"] = monthly["transaction_date"].dt.to_period("M").astype(str)
    return (
        monthly.groupby("month", as_index=False)
        .agg(
            revenue=("revenue", "sum"),
            transaction_count=("transaction_id", "nunique"),
            quantity=("quantity", "sum"),
            active_customers=("customer_id", "nunique"),
        )
        .sort_values("month")
    )


def _cohort_retention(transactions: pd.DataFrame) -> pd.DataFrame:
    activity = transactions[["customer_id", "transaction_date"]].copy()
    activity["activity_month"] = activity["transaction_date"].dt.to_period("M")
    activity = activity.drop_duplicates(["customer_id", "activity_month"])
    first_purchase = activity.groupby("customer_id")["activity_month"].min()
    activity["cohort_month"] = activity["customer_id"].map(first_purchase)
    activity["months_since_first_purchase"] = (
        (activity["activity_month"].dt.year - activity["cohort_month"].dt.year) * 12
        + activity["activity_month"].dt.month
        - activity["cohort_month"].dt.month
    )
    cohort_sizes = (
        first_purchase.value_counts().rename("cohort_size").rename_axis("cohort_month")
    )
    retention = (
        activity.groupby(["cohort_month", "months_since_first_purchase"])
        .agg(active_customers=("customer_id", "nunique"))
        .reset_index()
    )
    retention["cohort_size"] = retention["cohort_month"].map(cohort_sizes)
    retention["retention_rate"] = (
        retention["active_customers"] / retention["cohort_size"]
    )
    retention["cohort_month"] = retention["cohort_month"].astype(str)
    return retention


def _save_csv(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)


def _save_plots(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    demographics = tables["customer_demographics"]
    regions = demographics[demographics["dimension"] == "region"].sort_values(
        "customer_count", ascending=True
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(regions["category"], regions["customer_count"], color="#4472C4")
    axis.set(title="Customers by Region", xlabel="Customers", ylabel="Region")
    figure.tight_layout()
    figure.savefig(output_dir / "customers_by_region.png", dpi=150)
    plt.close(figure)

    monthly = tables["monthly_trends"]
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(monthly["month"], monthly["revenue"], marker="o", color="#4472C4")
    axes[0].set_ylabel("Revenue")
    axes[0].set_title("Monthly Revenue and Transactions")
    axes[1].plot(
        monthly["month"],
        monthly["transaction_count"],
        marker="o",
        color="#ED7D31",
    )
    axes[1].set_ylabel("Transactions")
    axes[1].set_xlabel("Month")
    axes[1].tick_params(axis="x", rotation=60)
    figure.tight_layout()
    figure.savefig(output_dir / "monthly_trends.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(
        monthly["month"],
        monthly["active_customers"],
        marker="o",
        color="#70AD47",
    )
    axis.set(
        title="Monthly Active Customers",
        xlabel="Month",
        ylabel="Unique Purchasing Customers",
    )
    axis.tick_params(axis="x", rotation=60)
    figure.tight_layout()
    figure.savefig(output_dir / "customer_activity.png", dpi=150)
    plt.close(figure)

    categories = tables["revenue_by_category"].sort_values("revenue")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(categories["category"], categories["revenue"], color="#70AD47")
    axis.set(title="Revenue by Product Category", xlabel="Revenue", ylabel="Category")
    figure.tight_layout()
    figure.savefig(output_dir / "revenue_by_category.png", dpi=150)
    plt.close(figure)

    brands = tables["revenue_by_brand"].sort_values("revenue")
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(brands["brand"], brands["revenue"], color="#70AD47")
    axis.set(title="Revenue by Brand", xlabel="Revenue", ylabel="Brand")
    figure.tight_layout()
    figure.savefig(output_dir / "revenue_by_brand.png", dpi=150)
    plt.close(figure)

    frequency = tables["purchase_frequency_distribution"]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        frequency["frequency_segment"].astype(str),
        frequency["customer_count"],
        color="#5B9BD5",
    )
    axis.set(
        title="Customer Purchase Frequency",
        xlabel="Transactions per Customer",
        ylabel="Customers",
    )
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(output_dir / "purchase_frequency.png", dpi=150)
    plt.close(figure)

    discount = tables["quantity_discount_summary"]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(discount["discount_band"], discount["transaction_count"], color="#A5A5A5")
    axis.set(
        title="Transaction Count by Discount",
        xlabel="Discount Band",
        ylabel="Transactions",
    )
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(output_dir / "discount_behavior.png", dpi=150)
    plt.close(figure)

    quantities = tables["quantity_distribution"]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        quantities["quantity"].astype(str),
        quantities["transaction_count"],
        color="#FFC000",
    )
    axis.set(
        title="Quantity per Transaction",
        xlabel="Quantity",
        ylabel="Transactions",
    )
    figure.tight_layout()
    figure.savefig(output_dir / "quantity_distribution.png", dpi=150)
    plt.close(figure)

    cohorts = tables["cohort_retention"]
    if not cohorts.empty:
        matrix = cohorts.pivot(
            index="cohort_month",
            columns="months_since_first_purchase",
            values="retention_rate",
        )
        figure, axis = plt.subplots(
            figsize=(max(8, matrix.shape[1] * 0.55), max(5, matrix.shape[0] * 0.35))
        )
        image = axis.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        axis.set_yticks(range(len(matrix.index)), labels=matrix.index)
        axis.set_xticks(
            range(len(matrix.columns)),
            labels=[str(value) for value in matrix.columns],
        )
        axis.set(
            title="Monthly Customer Retention by First-Purchase Cohort",
            xlabel="Months Since First Purchase",
            ylabel="Cohort Month",
        )
        figure.colorbar(image, ax=axis, label="Retention Rate")
        figure.tight_layout()
        figure.savefig(output_dir / "cohort_retention.png", dpi=150)
        plt.close(figure)

    marketing = tables["marketing_campaigns"]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(marketing["campaign"], marketing["event_count"], color="#8064A2")
    axis.set(
        title="Marketing Events by Campaign",
        xlabel="Campaign",
        ylabel="Events",
    )
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(output_dir / "marketing_campaigns.png", dpi=150)
    plt.close(figure)


def run_eda(
    raw_data_dir: Path = RAW_DATA_DIR,
    output_dir: Path = EDA_OUTPUT_DIR,
) -> dict[str, Any]:
    """Create EDA summary tables, plots, and JSON metrics from raw CSV files."""
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = _load_datasets(raw_data_dir)

    customers = datasets["customers"]
    products = datasets["products"]
    transactions = datasets["transactions"].merge(
        products[["product_id", "category", "brand"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )
    transactions["revenue"] = (
        transactions["price"] * transactions["quantity"] * (1 - transactions["discount"])
    ).round(2)

    demographics = _customer_demographics(customers)
    frequency, frequency_distribution = _purchase_frequency(customers, transactions)
    monthly = _monthly_trends(transactions)
    category_revenue = (
        transactions.groupby("category", as_index=False)
        .agg(revenue=("revenue", "sum"), transaction_count=("transaction_id", "nunique"))
        .sort_values("revenue", ascending=False)
    )
    brand_revenue = (
        transactions.groupby("brand", as_index=False)
        .agg(revenue=("revenue", "sum"), transaction_count=("transaction_id", "nunique"))
        .sort_values("revenue", ascending=False)
    )
    quantity_distribution = (
        transactions.groupby("quantity", as_index=False)
        .agg(
            transaction_count=("transaction_id", "size"),
            revenue=("revenue", "sum"),
        )
        .sort_values("quantity")
    )
    quantity_discount = (
        transactions.assign(
            discount_band=pd.cut(
                transactions["discount"],
                bins=[-0.001, 0, 0.10, 0.20, 0.30, 1],
                labels=["0%", ">0-10%", ">10-20%", ">20-30%", ">30%"],
                include_lowest=True,
            )
        )
        .groupby("discount_band", observed=False)
        .agg(
            transaction_count=("transaction_id", "size"),
            quantity=("quantity", "sum"),
            average_quantity=("quantity", "mean"),
        )
        .reset_index()
    )
    cohort_retention = _cohort_retention(transactions)
    marketing_campaigns = (
        datasets["marketing"]
        .groupby(["campaign", "channel"], as_index=False)
        .size()
        .rename(columns={"size": "event_count"})
        .sort_values("event_count", ascending=False)
    )
    tables = {
        "customer_demographics": demographics,
        "customer_purchase_frequency": frequency,
        "purchase_frequency_distribution": frequency_distribution,
        "revenue_by_category": category_revenue,
        "revenue_by_brand": brand_revenue,
        "monthly_trends": monthly,
        "quantity_discount_summary": quantity_discount,
        "quantity_distribution": quantity_distribution,
        "cohort_retention": cohort_retention,
        "marketing_campaigns": marketing_campaigns,
    }
    _save_csv(tables, output_dir)
    _save_plots(tables, output_dir)

    transaction_count = int(transactions["transaction_id"].nunique())
    total_revenue = round(float(transactions["revenue"].sum()), 2)
    order_values = transactions.groupby("transaction_id")["revenue"].sum()
    month_one_retention = cohort_retention.loc[
        cohort_retention["months_since_first_purchase"] == 1, "retention_rate"
    ]
    summary: dict[str, Any] = {
        "customer_count": int(customers["customer_id"].nunique()),
        "product_count": int(products["product_id"].nunique()),
        "transaction_count": transaction_count,
        "marketing_event_count": int(len(datasets["marketing"])),
        "total_revenue": total_revenue,
        "average_order_value": (
            round(float(order_values.mean()), 2) if not order_values.empty else 0.0
        ),
        "total_quantity": int(transactions["quantity"].sum()),
        "average_quantity_per_transaction": (
            round(float(transactions["quantity"].mean()), 2)
            if not transactions.empty
            else 0.0
        ),
        "average_discount_rate": (
            round(float(transactions["discount"].mean()), 4)
            if not transactions.empty
            else 0.0
        ),
        "transaction_date_range": {
            "minimum": (
                transactions["transaction_date"].min().date().isoformat()
                if not transactions.empty
                else None
            ),
            "maximum": (
                transactions["transaction_date"].max().date().isoformat()
                if not transactions.empty
                else None
            ),
        },
        "monthly_active_customer_average": (
            float(monthly["active_customers"].mean()) if not monthly.empty else 0.0
        ),
        "month_one_cohort_retention_average": (
            float(month_one_retention.mean()) if not month_one_retention.empty else None
        ),
        "top_revenue_category": (
            str(category_revenue.iloc[0]["category"])
            if not category_revenue.empty
            else None
        ),
        "top_revenue_brand": (
            str(brand_revenue.iloc[0]["brand"]) if not brand_revenue.empty else None
        ),
    }
    with (output_dir / "eda_summary.json").open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)
        json_file.write("\n")
    return summary


if __name__ == "__main__":
    eda_summary = run_eda()
    print(json.dumps(eda_summary, indent=2))
    print(f"EDA outputs saved to: {EDA_OUTPUT_DIR}")

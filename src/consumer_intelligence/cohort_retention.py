"""First-purchase cohorts and monthly customer and revenue retention."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from consumer_intelligence.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR

OUTPUT_PATH = PROCESSED_DATA_DIR / "cohort_retention.csv"
SUMMARY_PATH = REPORTS_DIR / "cohort_retention_summary.json"
MATRIX_PATH = REPORTS_DIR / "cohort_retention_matrix.csv"
MAX_PERIOD = 12


def _validate_id(frame: pd.DataFrame, column: str, name: str) -> None:
    if frame[column].isna().any() or frame[column].str.strip().eq("").any():
        raise ValueError(f"{name} contains missing {column}")
    if frame[column].duplicated().any():
        raise ValueError(f"{name} contains duplicate {column}")


def run_cohort_retention(
    raw_data_dir: Path = RAW_DATA_DIR,
    output_path: Path = OUTPUT_PATH,
    summary_path: Path = SUMMARY_PATH,
    matrix_path: Path = MATRIX_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Write observed M0-M12 first-purchase cohort and activity summaries."""
    raw_data_dir = Path(raw_data_dir)
    customers = pd.read_csv(
        raw_data_dir / "customers.csv",
        dtype={"customer_id": "string"},
        usecols=["customer_id", "signup_date"],
    )
    transactions = pd.read_csv(
        raw_data_dir / "transactions.csv",
        dtype={"customer_id": "string", "transaction_id": "string"},
        usecols=[
            "customer_id", "transaction_id", "transaction_date",
            "quantity", "price", "discount",
        ],
    )
    _validate_id(customers, "customer_id", "customers")
    _validate_id(transactions, "transaction_id", "transactions")
    if transactions.empty:
        raise ValueError("Transaction history is empty; observation period is unknown")
    if transactions["customer_id"].isna().any() or not transactions[
        "customer_id"
    ].isin(customers["customer_id"]).all():
        raise ValueError("Transactions contain missing or unknown customer IDs")

    customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="raise")
    transactions["transaction_date"] = pd.to_datetime(
        transactions["transaction_date"], errors="raise"
    )
    if customers["signup_date"].isna().any() or transactions["transaction_date"].isna().any():
        raise ValueError("Signup and transaction dates must be present")
    transactions = transactions.merge(
        customers, on="customer_id", how="left", validate="many_to_one"
    )
    if transactions["transaction_date"].lt(transactions["signup_date"]).any():
        raise ValueError("A transaction predates its customer's signup")
    for column in ("quantity", "price", "discount"):
        transactions[column] = pd.to_numeric(transactions[column], errors="raise")
    if not np.isfinite(transactions[["quantity", "price", "discount"]].to_numpy(dtype=float)).all():
        raise ValueError("Transaction numeric values must be finite")
    if (
        transactions["quantity"].le(0).any()
        or transactions["quantity"].mod(1).ne(0).any()
        or transactions["price"].le(0).any()
        or not transactions["discount"].between(0, 1).all()
    ):
        raise ValueError("Invalid transaction quantity, price, or discount")

    transactions["revenue"] = (
        transactions["quantity"] * transactions["price"] * (1 - transactions["discount"])
    ).round(2)
    last_observed_date = transactions["transaction_date"].max().normalize()
    last_month = last_observed_date.to_period("M")
    if last_observed_date.day != last_observed_date.days_in_month:
        last_month -= 1
    transactions["month"] = transactions["transaction_date"].dt.to_period("M")
    transactions = transactions.loc[transactions["month"].le(last_month)].copy()
    if transactions.empty:
        raise ValueError("No complete observation month is available")

    first_months = transactions.groupby("customer_id")["month"].min()
    transactions["cohort_month"] = transactions["customer_id"].map(first_months)
    transactions["period"] = (
        (transactions["month"].dt.year - transactions["cohort_month"].dt.year) * 12
        + transactions["month"].dt.month
        - transactions["cohort_month"].dt.month
    )
    cohort_sizes = first_months.value_counts().sort_index()
    cohort_sizes.index.name = "cohort_month"
    cohort_values = (
        transactions.loc[transactions["period"].le(MAX_PERIOD)]
        .groupby(["cohort_month", "month", "period"], as_index=False)
        .agg(
            retained_customers=("customer_id", "nunique"),
            cohort_revenue=("revenue", "sum"),
        )
    )
    grid = pd.DataFrame(
        [
            {"cohort_month": cohort, "month": cohort + offset, "period": offset,
             "cohort_size": int(size)}
            for cohort, size in cohort_sizes.items()
            for offset in range(MAX_PERIOD + 1)
            if cohort + offset <= last_month
        ]
    )
    retention = grid.merge(
        cohort_values, on=["cohort_month", "month", "period"],
        how="left", validate="one_to_one",
    )
    retention["retained_customers"] = retention["retained_customers"].fillna(0).astype("int64")
    retention["cohort_revenue"] = retention["cohort_revenue"].fillna(0).round(2)
    retention["retention_rate"] = retention["retained_customers"] / retention["cohort_size"]
    baseline = retention.loc[retention["period"].eq(0)].set_index("cohort_month")["cohort_revenue"]
    retention["m0_revenue"] = retention["cohort_month"].map(baseline)
    retention["revenue_retention"] = (
        retention["cohort_revenue"] / retention["m0_revenue"].where(retention["m0_revenue"].gt(0))
    )
    retention["cohort_month"] = retention["cohort_month"].astype(str)
    retention["month"] = retention["month"].astype(str)
    retention = retention[
        ["cohort_month", "month", "period", "cohort_size", "retained_customers",
         "retention_rate", "cohort_revenue", "m0_revenue", "revenue_retention"]
    ].sort_values(["cohort_month", "period"]).reset_index(drop=True)

    matrix = retention.pivot(index="cohort_month", columns="period", values="retention_rate")
    matrix = matrix.reindex(columns=range(MAX_PERIOD + 1))
    matrix.columns = [f"M{period}" for period in matrix.columns]
    matrix = matrix.reset_index()

    monthly = transactions.groupby("month")["customer_id"].nunique().reindex(
        pd.period_range(transactions["month"].min(), last_month, freq="M"),
        fill_value=0,
    ).rename("active_customers").rename_axis("month").reset_index()
    repeat_counts = (
        transactions.loc[transactions["period"].gt(0)]
        .groupby("month")["customer_id"].nunique()
    )
    monthly["repeat_customers"] = monthly["month"].map(repeat_counts).fillna(0).astype(int)
    monthly["repeat_purchase_rate"] = (
        monthly["repeat_customers"] / monthly["active_customers"].where(monthly["active_customers"].gt(0))
    )
    monthly["month"] = monthly["month"].astype(str)
    monthly_activity = monthly.astype(object).where(monthly.notna(), None).to_dict(orient="records")

    period_summary: dict[str, Any] = {}
    for period in (1, 3, 6):
        eligible = retention.loc[retention["period"].eq(period)]
        size = int(eligible["cohort_size"].sum())
        period_summary[f"M{period}"] = {
            "eligible_cohorts": len(eligible),
            "eligible_customers": size,
            "retained_customers": int(eligible["retained_customers"].sum()),
            "retention_rate": float(eligible["retained_customers"].sum() / size) if size else None,
        }
    m3 = retention.loc[retention["period"].eq(3)].sort_values(
        ["retention_rate", "cohort_month"], ascending=[False, True]
    )
    strongest = None if m3.empty else {
        "cohort_month": m3.iloc[0]["cohort_month"],
        "retention_rate": float(m3.iloc[0]["retention_rate"]),
        "cohort_size": int(m3.iloc[0]["cohort_size"]),
    }
    weakest_rows = m3.sort_values(["retention_rate", "cohort_month"])
    weakest = None if weakest_rows.empty else {
        "cohort_month": weakest_rows.iloc[0]["cohort_month"],
        "retention_rate": float(weakest_rows.iloc[0]["retention_rate"]),
        "cohort_size": int(weakest_rows.iloc[0]["cohort_size"]),
    }
    revenue_summary: dict[str, Any] = {}
    for period in (1, 3, 6):
        eligible = retention.loc[
            retention["period"].eq(period) & retention["m0_revenue"].gt(0)
        ]
        base = float(eligible["m0_revenue"].sum())
        revenue_summary[f"M{period}"] = {
            "eligible_cohorts": len(eligible),
            "cohort_revenue": round(float(eligible["cohort_revenue"].sum()), 2),
            "m0_revenue": round(base, 2),
            "revenue_retention": float(eligible["cohort_revenue"].sum() / base) if base else None,
        }
    summary: dict[str, Any] = {
        "cohort_count": len(cohort_sizes),
        "customers_with_purchases": int(len(first_months)),
        "customers_without_observed_purchases": int(len(customers) - len(first_months)),
        "observation_start_month": str(transactions["month"].min()),
        "observation_end_month": str(last_month),
        "last_transaction_date": last_observed_date.date().isoformat(),
        "total_observed_revenue": round(float(transactions["revenue"].sum()), 2),
        "monthly_active_customers": monthly_activity,
        "customer_retention": period_summary,
        "revenue_retention": revenue_summary,
        "cohort_comparison_period": "M3 (ties resolved by earliest cohort month)",
        "strongest_cohort": strongest,
        "weakest_cohort": weakest,
        "definitions": {
            "cohort": "First observed purchase month; customers without purchases have no cohort.",
            "retention_rate": "Unique purchasing customers in cohort-month divided by cohort size.",
            "cohort_revenue": "Sum of quantity * price * (1 - discount), rounded per transaction.",
            "revenue_retention": "Cohort-month revenue divided by M0 cohort revenue; may exceed 1, undefined if M0 revenue is zero.",
            "period_aggregate": "Customer retention is cohort-size weighted; revenue retention is summed period revenue divided by summed M0 revenue among matured cohorts with positive M0 revenue.",
            "repeat_purchase_rate": "Share of monthly active customers whose first purchase month was earlier than the activity month.",
            "maturity": "Include months through the last transaction month only when the last transaction date is its calendar month-end; otherwise exclude that partial month. Missing matured months are zero; future months are omitted.",
            "coverage_limit": "First and last transaction dates are coverage proxies, not evidence of complete ingestion. Historical first purchases may predate observed data.",
        },
    }

    output_path, summary_path, matrix_path = Path(output_path), Path(summary_path), Path(matrix_path)
    for path in (output_path, summary_path, matrix_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    retention.to_csv(output_path, index=False)
    matrix.to_csv(matrix_path, index=False)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
        file.write("\n")
    return retention, summary


if __name__ == "__main__":
    _, report = run_cohort_retention()
    print(f"Cohorts: {report['cohort_count']}")
    print(f"Observed months: {report['observation_start_month']} to {report['observation_end_month']}")
    print(f"Summary: {SUMMARY_PATH}")

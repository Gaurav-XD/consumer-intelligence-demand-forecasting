"""Deterministic RFM scoring and customer segmentation."""

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from consumer_intelligence.config import PROCESSED_DATA_DIR, REPORTS_DIR

INPUT_PATH = PROCESSED_DATA_DIR / "customer_features.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "customer_segments.csv"
SUMMARY_PATH = REPORTS_DIR / "rfm_segmentation_summary.json"
DISTRIBUTION_PATH = REPORTS_DIR / "rfm_segment_distribution.csv"
RFM_COLUMNS = {
    "recency": "days_since_last_purchase",
    "frequency": "frequency_orders",
    "monetary": "monetary_total_revenue",
}

SEGMENT_RULES = [
    {
        "segment": "Champions",
        "rule": "R >= 4 and F >= 4 and M >= 4",
    },
    {
        "segment": "Can't Lose Them",
        "rule": "R <= 2 and F >= 4 and M >= 4",
    },
    {
        "segment": "At Risk",
        "rule": "R <= 2 and F >= 3, excluding Can't Lose Them",
    },
    {
        "segment": "Hibernating/Lost",
        "rule": "R <= 2 and F <= 2",
    },
    {
        "segment": "Loyal Customers",
        "rule": "R >= 3 and F >= 4, excluding Champions",
    },
    {
        "segment": "New Customers",
        "rule": "R >= 4 and F == 1",
    },
    {
        "segment": "Potential Loyalists",
        "rule": "R >= 4 and F >= 2, excluding Champions and Loyal Customers",
    },
    {
        "segment": "Promising",
        "rule": "R >= 3 and F >= 2, excluding earlier matching segments",
    },
    {"segment": "Needs Attention", "rule": "All remaining RFM score combinations"},
]
SEGMENT_ORDER = [rule["segment"] for rule in SEGMENT_RULES]


def _read_features(path: Path) -> pd.DataFrame:
    features = pd.read_csv(path, dtype={"customer_id": "string"})
    required = {"customer_id", *RFM_COLUMNS.values(), "total_revenue"}
    missing_columns = required - set(features.columns)
    if missing_columns:
        raise ValueError(f"Customer feature file is missing columns: {sorted(missing_columns)}")
    if features["customer_id"].isna().any():
        raise ValueError("customer_id cannot be missing")
    if features["customer_id"].duplicated().any():
        raise ValueError("customer_id must be unique")
    for column in set(RFM_COLUMNS.values()) | {"total_revenue"}:
        features[column] = pd.to_numeric(features[column], errors="raise")
    return features


def _quantile_score(
    values: pd.Series,
    higher_is_better: bool,
    positive_values_only: bool = True,
) -> pd.Series:
    scores = pd.Series(1, index=values.index, dtype="int64")
    eligible = values.notna() & (
        values.gt(0) if positive_values_only else values.ge(0)
    )
    if not eligible.any():
        return scores
    percentiles = values.loc[eligible].rank(
        method="average",
        ascending=higher_is_better,
        pct=True,
    )
    scores.loc[eligible] = (
        (percentiles * 5).apply(math.ceil)
        .clip(1, 5)
        .astype("int64")
    )
    return scores


def _assign_segments(features: pd.DataFrame) -> pd.Series:
    recency = features["R_score"]
    frequency = features["F_score"]
    monetary = features["M_score"]
    segments = pd.Series("Needs Attention", index=features.index, dtype="string")

    segments.loc[(recency >= 3) & (frequency >= 2)] = "Promising"
    segments.loc[(recency >= 4) & (frequency >= 2)] = "Potential Loyalists"
    segments.loc[(recency >= 4) & (frequency == 1)] = "New Customers"
    segments.loc[(recency >= 3) & (frequency >= 4)] = "Loyal Customers"
    segments.loc[(recency <= 2) & (frequency <= 2)] = "Hibernating/Lost"
    segments.loc[(recency <= 2) & (frequency >= 3)] = "At Risk"
    segments.loc[
        (recency <= 2) & (frequency >= 4) & (monetary >= 4)
    ] = "Can't Lose Them"
    segments.loc[
        (recency >= 4) & (frequency >= 4) & (monetary >= 4)
    ] = "Champions"
    return segments


def segment_customers(customer_features: pd.DataFrame) -> pd.DataFrame:
    """Score and segment customer features without modifying the input frame."""
    features = customer_features.copy()
    required = {"customer_id", *RFM_COLUMNS.values(), "total_revenue"}
    missing_columns = required - set(features.columns)
    if missing_columns:
        raise ValueError(f"Customer features are missing columns: {sorted(missing_columns)}")
    if features["customer_id"].isna().any() or features["customer_id"].duplicated().any():
        raise ValueError("customer_id values must be present and unique")

    for column in set(RFM_COLUMNS.values()) | {"total_revenue"}:
        features[column] = pd.to_numeric(features[column], errors="raise")

    features["R_score"] = _quantile_score(
        features[RFM_COLUMNS["recency"]],
        higher_is_better=False,
        positive_values_only=False,
    )
    features["F_score"] = _quantile_score(
        features[RFM_COLUMNS["frequency"]], higher_is_better=True
    )
    features["M_score"] = _quantile_score(
        features[RFM_COLUMNS["monetary"]], higher_is_better=True
    )
    features["rfm_score"] = (
        features["R_score"] + features["F_score"] + features["M_score"]
    )
    features["rfm_code"] = (
        features["R_score"].astype(str)
        + features["F_score"].astype(str)
        + features["M_score"].astype(str)
    )
    features["segment"] = _assign_segments(features)
    features["segment"] = pd.Categorical(
        features["segment"], categories=SEGMENT_ORDER, ordered=True
    )
    return features


def _build_distribution(segments: pd.DataFrame) -> pd.DataFrame:
    distribution = (
        segments.groupby("segment", observed=False)
        .agg(
            customer_count=("customer_id", "size"),
            revenue=("total_revenue", "sum"),
            average_revenue_per_customer=("total_revenue", "mean"),
            average_recency_days=("days_since_last_purchase", "mean"),
            average_frequency_orders=("frequency_orders", "mean"),
            average_monetary_value=("monetary_total_revenue", "mean"),
        )
        .reindex(SEGMENT_ORDER, fill_value=0)
        .reset_index()
    )
    total_customers = len(segments)
    distribution["percentage"] = (
        distribution["customer_count"] / total_customers * 100
        if total_customers
        else 0.0
    )
    distribution["revenue"] = distribution["revenue"].round(2)
    distribution["average_revenue_per_customer"] = distribution[
        "average_revenue_per_customer"
    ].round(2)
    distribution["average_recency_days"] = distribution[
        "average_recency_days"
    ].round(2)
    distribution["average_frequency_orders"] = distribution[
        "average_frequency_orders"
    ].round(2)
    distribution["average_monetary_value"] = distribution[
        "average_monetary_value"
    ].round(2)
    distribution["percentage"] = distribution["percentage"].round(2)
    return distribution


def run_rfm_segmentation(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    summary_path: Path = SUMMARY_PATH,
    distribution_path: Path = DISTRIBUTION_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate RFM scores, segments, distribution table, and JSON summary."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    summary_path = Path(summary_path)
    distribution_path = Path(distribution_path)
    features = _read_features(input_path)
    segments = segment_customers(features)
    distribution = _build_distribution(segments)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    segments.to_csv(output_path, index=False)
    distribution_path.parent.mkdir(parents=True, exist_ok=True)
    distribution.to_csv(distribution_path, index=False)

    segment_counts = {
        str(row.segment): int(row.customer_count)
        for row in distribution.itertuples(index=False)
    }
    segment_percentages = {
        str(row.segment): float(row.percentage)
        for row in distribution.itertuples(index=False)
    }
    segment_metrics = {
        str(row.segment): {
            "customer_count": int(row.customer_count),
            "percentage": float(row.percentage),
            "revenue": float(row.revenue),
            "average_revenue_per_customer": (
                float(row.average_revenue_per_customer)
                if pd.notna(row.average_revenue_per_customer)
                else None
            ),
            "average_recency_days": (
                float(row.average_recency_days)
                if pd.notna(row.average_recency_days)
                else None
            ),
            "average_frequency_orders": (
                float(row.average_frequency_orders)
                if pd.notna(row.average_frequency_orders)
                else None
            ),
            "average_monetary_value": (
                float(row.average_monetary_value)
                if pd.notna(row.average_monetary_value)
                else None
            ),
        }
        for row in distribution.itertuples(index=False)
    }
    summary: dict[str, Any] = {
        "customer_count": len(segments),
        "segment_count": len(SEGMENT_ORDER),
        "segment_counts": segment_counts,
        "segment_percentages": segment_percentages,
        "revenue_by_segment": {
            segment: metrics["revenue"]
            for segment, metrics in segment_metrics.items()
        },
        "average_revenue_per_customer_by_segment": {
            segment: metrics["average_revenue_per_customer"]
            for segment, metrics in segment_metrics.items()
        },
        "average_recency_frequency_monetary_by_segment": {
            segment: {
                "average_recency_days": metrics["average_recency_days"],
                "average_frequency_orders": metrics["average_frequency_orders"],
                "average_monetary_value": metrics["average_monetary_value"],
            }
            for segment, metrics in segment_metrics.items()
        },
        "total_revenue": round(float(segments["total_revenue"].sum()), 2),
        "scoring_method": (
            "Five quantile-style scores from average percentile ranks. Ties share "
            "a rank and score; positive values are ranked within the positive "
            "population. Missing/zero recency and non-positive frequency or "
            "monetary values score 1."
        ),
        "segment_rules": SEGMENT_RULES,
        "segment_metrics": segment_metrics,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)
        json_file.write("\n")
    return segments, summary


if __name__ == "__main__":
    customer_segments, segmentation_summary = run_rfm_segmentation()
    print(
        f"Segmented {len(customer_segments)} customers into "
        f"{segmentation_summary['segment_count']} defined segments."
    )
    print(f"Customer segments: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Distribution: {DISTRIBUTION_PATH}")

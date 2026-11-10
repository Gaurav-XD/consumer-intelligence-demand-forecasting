"""Business-facing dashboard over existing, read-only project artifacts."""

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from consumer_intelligence.config import PROJECT_ROOT

logger = logging.getLogger(__name__)
ARTIFACTS = {
    "eda": ("reports/eda/eda_summary.json", "json", ()),
    "churn_metadata": ("reports/churn_dataset_metadata.json", "json", ()),
    "forecast_summary": ("reports/forecasting_tuning_summary.json", "json", ()),
    "churn_summary": ("reports/churn_tuning_summary.json", "json", ()),
    "impact": ("reports/business_impact_summary.json", "json", ()),
    "segments": ("reports/business_impact_segments.csv", "csv",
                 ("segment", "customer_count", "historical_revenue", "estimated_clv")),
    "cohorts": ("data/processed/cohort_retention.csv", "csv",
                ("cohort_month", "period", "retention_rate")),
    "churn_results": ("reports/churn_model_results.csv", "csv",
                      ("model", "train_cv_average_precision", "roc_auc", "pr_auc", "precision", "recall", "f1")),
    "high_risk": ("reports/business_impact_high_risk_customers.csv", "csv",
                  ("customer_id", "snapshot_date", "predicted_churn_probability")),
    "forecast": ("reports/forecasting_walk_forward.csv", "csv",
                 ("product_id", "forecast_month", "actual_quantity", "predicted_quantity")),
    "recommendations": ("reports/business_recommendations.csv", "csv",
                        ("target", "recommended_action", "expected_business_benefit", "assumptions", "limitations")),
}


def load_artifacts(root: Path = PROJECT_ROOT) -> tuple[dict[str, Any], list[str]]:
    """Load each available report independently, surfacing missing or malformed inputs."""
    data: dict[str, Any] = {}
    issues = []
    for key, (relative_path, kind, required) in ARTIFACTS.items():
        path = Path(root) / relative_path
        try:
            if kind == "json":
                with path.open(encoding="utf-8") as file:
                    value = json.load(file)
                if not isinstance(value, dict) or not value:
                    raise ValueError("JSON report is empty or not an object")
            else:
                value = pd.read_csv(path, dtype={"customer_id": "string", "product_id": "string"})
                missing = set(required) - set(value.columns)
                if missing:
                    raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
                if value.empty:
                    raise ValueError("CSV report contains no rows")
            data[key] = value
        except (FileNotFoundError, PermissionError, UnicodeError, json.JSONDecodeError,
                pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as exc:
            logger.warning("Cannot load dashboard artifact %s: %s", path, exc)
            issues.append(f"{relative_path}: {exc}")
    return data, issues


def _number(value: Any, style: str = "number") -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "Unavailable"
    if style == "percent":
        return f"{value:.2%}"
    if style == "percent_points":
        return f"{value:.2f}%"
    if style == "integer":
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def _nested(source: dict[str, Any], *keys: str) -> Any:
    value: Any = source
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def forecast_performance(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate error from individual rows, not errors of monthly totals."""
    required = {"product_id", "actual_quantity", "predicted_quantity"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=["product_id", "observations", "actual", "predicted", "mae", "wape_percent"])
    values = frame.copy()
    for field in ("actual_quantity", "predicted_quantity"):
        values[field] = pd.to_numeric(values[field], errors="raise")
    if values[["actual_quantity", "predicted_quantity"]].isna().any().any():
        raise ValueError("Forecast values contain missing data")
    if not values[["actual_quantity", "predicted_quantity"]].map(math.isfinite).all().all():
        raise ValueError("Forecast values must be finite")
    if values["actual_quantity"].lt(0).any():
        raise ValueError("Actual demand must be nonnegative")
    values["absolute_error"] = (values["actual_quantity"] - values["predicted_quantity"]).abs()
    grouped = values.groupby("product_id", as_index=False).agg(
        observations=("actual_quantity", "size"), actual=("actual_quantity", "sum"),
        predicted=("predicted_quantity", "sum"), absolute_error=("absolute_error", "sum"),
        mae=("absolute_error", "mean"),
    )
    grouped["wape_percent"] = 100 * grouped["absolute_error"] / grouped["actual"].where(grouped["actual"].gt(0))
    return grouped.sort_values(["mae", "product_id"], ascending=[False, True])


def _table(data: dict[str, Any], key: str) -> pd.DataFrame | None:
    value = data.get(key)
    if value is None:
        st.info("This section is unavailable because its source report could not be loaded.")
    return value


def _render(section: Any, data: dict[str, Any]) -> None:
    try:
        section(data)
    except (ValueError, KeyError, TypeError, pd.errors.ParserError) as exc:
        logger.exception("Dashboard section failed")
        st.error(f"This section cannot display its data: {exc}")


def executive_overview(data: dict[str, Any]) -> None:
    st.header("Executive Overview")
    eda = data.get("eda", {})
    churn = data.get("churn_metadata", {})
    forecast = data.get("forecast_summary", {})
    columns = st.columns(5)
    columns[0].metric("Customers", _number(eda.get("customer_count"), "integer"))
    columns[1].metric("Revenue", _number(eda.get("total_revenue")))
    columns[2].metric("Average order value", _number(eda.get("average_order_value")))
    columns[3].metric("Observed churn rate", _number(churn.get("churn_rate"), "percent"))
    columns[4].metric("Forecast WAPE", _number(_nested(forecast, "walk_forward_metrics", "wape"), "percent_points"))
    period = eda.get("transaction_date_range", {})
    if isinstance(period, dict) and period.get("minimum") and period.get("maximum"):
        st.caption(f"Customer and revenue history: {period['minimum']} to {period['maximum']}. Currency units are unspecified.")
    st.caption("Churn combines two historical snapshot cohorts; forecast WAPE is walk-forward validation, not a future outcome.")


def customer_analytics(data: dict[str, Any]) -> None:
    st.header("Customer Analytics")
    segments = _table(data, "segments")
    if segments is not None:
        selected = st.multiselect("RFM segments", sorted(segments["segment"].dropna().unique().tolist()),
                                  default=sorted(segments["segment"].dropna().unique().tolist()), key="segments")
        subset = segments.loc[segments["segment"].isin(selected)]
        if subset.empty:
            st.info("Select at least one RFM segment.")
        else:
            left, right = st.columns(2)
            left.plotly_chart(px.bar(subset, x="segment", y="customer_count", title="Customers by segment"),
                              use_container_width=True)
            long = subset.melt(id_vars="segment", value_vars=["historical_revenue", "estimated_clv"],
                               var_name="measure", value_name="amount")
            right.plotly_chart(px.bar(long, x="segment", y="amount", color="measure", barmode="group",
                                      title="Historical revenue and annual CLV proxy"), use_container_width=True)
            st.dataframe(subset, use_container_width=True, hide_index=True)
        st.caption("CLV is a one-year revenue run-rate proxy, not discounted lifetime profit.")
    cohorts = _table(data, "cohorts")
    if cohorts is not None:
        matrix = cohorts.pivot(index="cohort_month", columns="period", values="retention_rate").sort_index()
        st.plotly_chart(px.imshow(matrix, aspect="auto", color_continuous_scale="Blues",
                                  labels={"x": "Months since first purchase", "y": "Cohort", "color": "Retention"},
                                  title="Monthly cohort retention"), use_container_width=True)
        st.caption("Blank cells are not yet observed; M0 is the first purchase month.")


def churn_analytics(data: dict[str, Any]) -> None:
    st.header("Churn Analytics")
    summary = data.get("churn_summary", {})
    metrics = summary.get("test_metrics", {})
    if isinstance(metrics, dict) and metrics:
        cols = st.columns(4)
        for col, label, key in zip(cols, ["ROC-AUC", "PR-AUC", "Precision", "Recall"],
                                   ["roc_auc", "pr_auc", "precision", "recall"]):
            col.metric(label, _number(metrics.get(key)))
        st.caption(f"Tuned {summary.get('selected_model', 'model')} on the historical test split; the holdout was previously inspected.")
    else:
        st.info("Tuned churn model metrics are unavailable.")
    baseline = _table(data, "churn_results")
    if baseline is not None:
        st.subheader("Baseline model performance")
        st.dataframe(baseline, use_container_width=True, hide_index=True)
    risk = _table(data, "high_risk")
    if risk is not None:
        st.subheader("Predicted high-risk customers")
        st.caption("Only customers already above the original scenario threshold are available. This is a retrospective test subset, not the full customer base.")
        threshold = st.slider("Display probability at or above", 0.0, 1.0, 0.5, 0.05)
        filtered = risk.loc[risk["predicted_churn_probability"].ge(threshold)]
        if filtered.empty:
            st.info("No preselected high-risk customers meet this filter.")
        else:
            st.plotly_chart(px.histogram(filtered, x="predicted_churn_probability", nbins=20,
                                         title="Risk distribution within the preselected high-risk subset"),
                            use_container_width=True)
            st.dataframe(filtered.sort_values("predicted_churn_probability", ascending=False),
                         use_container_width=True, hide_index=True)


def demand_forecasting(data: dict[str, Any]) -> None:
    st.header("Demand Forecasting")
    forecast = _table(data, "forecast")
    if forecast is None:
        return
    products = sorted(forecast["product_id"].dropna().astype(str).unique().tolist())
    selected = st.multiselect("Products", products, default=[], help="Leave empty to include all products.")
    filtered = forecast.loc[forecast["product_id"].astype(str).isin(selected)] if selected else forecast
    if filtered.empty:
        st.info("No forecast observations match this selection.")
        return
    performance = forecast_performance(filtered)
    errors = (filtered["actual_quantity"] - filtered["predicted_quantity"]).abs()
    actual_total = filtered["actual_quantity"].sum()
    columns = st.columns(3)
    columns[0].metric("MAE (units)", _number(float(errors.mean())))
    columns[1].metric("WAPE", _number(float(errors.sum() / actual_total) if actual_total > 0 else None, "percent"))
    columns[2].metric("Product-months", _number(len(filtered), "integer"))
    monthly = filtered.groupby("forecast_month", as_index=False)[["actual_quantity", "predicted_quantity"]].sum()
    monthly = monthly.melt(id_vars="forecast_month", var_name="series", value_name="quantity")
    st.plotly_chart(px.line(monthly, x="forecast_month", y="quantity", color="series", markers=True,
                            title="Monthly actual vs walk-forward forecast"), use_container_width=True)
    st.subheader("Product-level forecast performance")
    st.dataframe(performance, use_container_width=True, hide_index=True)
    st.caption("Historical walk-forward validation; each forecast uses earlier actuals. WAPE uses summed absolute product-month errors.")


def business_impact(data: dict[str, Any]) -> None:
    st.header("Business Impact")
    impact = data.get("impact", {})
    scenario = impact.get("churn_scenario", {})
    if isinstance(scenario, dict) and scenario:
        cols = st.columns(4)
        cols[0].metric("Scenario revenue at risk", _number(scenario.get("scenario_revenue_at_risk")))
        cols[1].metric("Scenario revenue protected", _number(scenario.get("scenario_revenue_protected")))
        cols[2].metric("Scenario contact cost", _number(scenario.get("scenario_intervention_cost")))
        cols[3].metric("Scenario ROI", _number(scenario.get("scenario_roi"), "percent"))
        st.caption(f"Retrospective test subset as of {scenario.get('snapshot_date', 'unknown')}; ROI is based on assumed contribution margin and costs.")
    else:
        st.info("Business-impact scenario is unavailable.")
    st.warning("Scenario estimates based on synthetic data and assumptions—not actual savings, causal effects or validated commercial results.")
    assumptions = impact.get("assumptions")
    if isinstance(assumptions, dict) and assumptions:
        with st.expander("Scenario assumptions"):
            st.json(assumptions)
    recommendations = _table(data, "recommendations")
    if recommendations is not None:
        st.subheader("Recommendations and limitations")
        st.dataframe(recommendations, use_container_width=True, hide_index=True)


SECTIONS = {
    "Executive Overview": executive_overview,
    "Customer Analytics": customer_analytics,
    "Churn Analytics": churn_analytics,
    "Demand Forecasting": demand_forecasting,
    "Business Impact": business_impact,
}


def main() -> None:
    st.set_page_config(page_title="Consumer Intelligence", page_icon="📊", layout="wide")
    st.title("Consumer Intelligence & Demand Forecasting")
    st.caption("Business review | synthetic data | historical analysis")
    root = Path(os.environ.get("CONSUMER_INTELLIGENCE_ROOT", str(PROJECT_ROOT))).resolve()
    data, issues = load_artifacts(root)
    if issues:
        with st.expander(f"Data availability: {len(issues)} issue(s)", expanded=True):
            for issue in issues:
                st.warning(issue)
    section = st.sidebar.radio("Dashboard section", list(SECTIONS))
    _render(SECTIONS[section], data)


if __name__ == "__main__":
    main()

"""Reproducible, illustrative business scenarios from existing model outputs."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from consumer_intelligence.churn_dataset import MODEL_FEATURE_COLUMNS
from consumer_intelligence.config import MODEL_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from consumer_intelligence.forecasting_models import regression_metrics


@dataclass(frozen=True)
class ScenarioAssumptions:
    """Illustrative inputs, not empirically measured commercial effects."""

    risk_threshold: float = 0.5
    relative_churn_reduction: float = 0.10
    cost_per_contact: float = 5.0
    contribution_margin: float = 0.30
    minimum_history_days: int = 365
    assumed_net_revenue_per_unit: float = 100.0
    underforecast_lost_sale_fraction: float = 0.25
    overforecast_holding_cost_per_unit: float = 10.0

    def validate(self) -> None:
        rates = (
            self.risk_threshold, self.relative_churn_reduction,
            self.contribution_margin, self.underforecast_lost_sale_fraction,
        )
        amounts = (
            self.cost_per_contact, self.assumed_net_revenue_per_unit,
            self.overforecast_holding_cost_per_unit,
        )
        if not all(np.isfinite(value) and 0 <= value <= 1 for value in rates):
            raise ValueError("Scenario rates must be finite and between 0 and 1")
        if not all(np.isfinite(value) and value >= 0 for value in amounts):
            raise ValueError("Scenario costs and revenue per unit must be finite and nonnegative")
        if not isinstance(self.minimum_history_days, int) or self.minimum_history_days < 1:
            raise ValueError("Minimum history must be a positive whole number of days")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _customer_ids(frame: pd.DataFrame, label: str) -> None:
    ids = frame["customer_id"].astype("string")
    if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError(f"{label} must have unique, nonempty customer IDs")


def _nonnegative(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{label} contains missing, negative, or non-finite values")


def run_business_impact(
    processed_dir: Path = PROCESSED_DATA_DIR,
    reports_dir: Path = REPORTS_DIR,
    model_path: Path = MODEL_DIR / "churn" / "tuned_best_model.joblib",
    assumptions: ScenarioAssumptions = ScenarioAssumptions(),
    output_dir: Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write descriptive summaries and illustrative scenarios; never fit a model."""
    assumptions.validate()
    processed_dir, reports_dir = Path(processed_dir), Path(reports_dir)
    output_dir = Path(output_dir) if output_dir is not None else reports_dir
    segments = pd.read_csv(processed_dir / "customer_segments.csv", dtype={"customer_id": "string"})
    clv = pd.read_csv(processed_dir / "customer_clv.csv", dtype={"customer_id": "string"})
    churn = pd.read_csv(processed_dir / "churn_dataset.csv", dtype={"customer_id": "string"})
    cohort = pd.read_csv(processed_dir / "cohort_retention.csv")
    forecasts = pd.read_csv(reports_dir / "forecasting_walk_forward.csv", dtype={"product_id": "string"})
    churn_report = _read_json(reports_dir / "churn_tuning_summary.json")
    churn_metadata = _read_json(reports_dir / "churn_dataset_metadata.json")
    clv_metadata = _read_json(reports_dir / "clv_summary.json")
    feature_metadata = _read_json(reports_dir / "customer_features_metadata.json")
    forecast_report = _read_json(reports_dir / "forecasting_tuning_summary.json")

    for frame, name in ((segments, "segments"), (clv, "CLV"), (churn, "churn")):
        _customer_ids(frame, name)
    if set(segments["customer_id"]) != set(clv["customer_id"]) or set(churn["customer_id"]) != set(clv["customer_id"]):
        raise ValueError("Customer populations differ between inputs")
    if segments["segment"].isna().any() or segments["segment"].astype("string").str.strip().eq("").any():
        raise ValueError("Segment assignments must be present")
    _nonnegative(segments, ["total_revenue"], "Segments")
    _nonnegative(clv, ["historical_revenue", "estimated_clv"], "CLV")
    if clv_metadata["snapshot_date"] != feature_metadata["as_of_date"]:
        raise ValueError("CLV and segment snapshots differ")
    customers = segments[["customer_id", "segment", "total_revenue"]].merge(
        clv[["customer_id", "historical_revenue", "estimated_clv"]],
        on="customer_id", validate="one_to_one",
    )
    if not np.allclose(customers["total_revenue"], customers["historical_revenue"], rtol=0, atol=0.005):
        raise ValueError("Segment and CLV revenue do not reconcile")
    segment_table = customers.groupby("segment", as_index=False, sort=True).agg(
        customer_count=("customer_id", "size"),
        historical_revenue=("historical_revenue", "sum"),
        estimated_clv=("estimated_clv", "sum"),
    )
    for column in ("historical_revenue", "estimated_clv"):
        total = segment_table[column].sum()
        segment_table[f"{column}_share"] = segment_table[column] / total if total else 0.0
    segment_table["average_estimated_clv"] = segment_table["estimated_clv"] / segment_table["customer_count"]
    segment_table["retention_priority"] = segment_table["segment"].isin(["At Risk", "Can't Lose Them"])

    if churn_report["feature_columns"] != MODEL_FEATURE_COLUMNS or churn_metadata["feature_columns"] != MODEL_FEATURE_COLUMNS:
        raise ValueError("Churn model feature contract does not match approved features")
    if not churn["split"].isin(["train", "test"]).all():
        raise ValueError("Invalid churn split")
    test = churn.loc[churn["split"].eq("test")].copy()
    period = churn_metadata["periods"]["test"]
    snapshot = pd.Timestamp(period["feature_snapshot_period"]["snapshot_date_inclusive"])
    start = pd.Timestamp(period["feature_snapshot_period"]["transaction_history_start"])
    observation = period["churn_observation_window"]
    days = int(observation["days"])
    window_start = pd.Timestamp(observation["start_inclusive"])
    window_end = pd.Timestamp(observation["end_inclusive"])
    if (
        pd.isna(snapshot) or pd.isna(start) or pd.isna(window_start) or pd.isna(window_end)
        or start > snapshot or days < 1 or window_start != snapshot + pd.Timedelta(days=1)
        or (window_end - window_start).days + 1 != days
    ):
        raise ValueError("Invalid churn observation period")
    if test.empty or len(test) != period["row_count"] or not test["snapshot_date"].eq(snapshot.strftime("%Y-%m-%d")).all():
        raise ValueError("Churn test population does not match snapshot metadata")
    _nonnegative(test, ["total_revenue", "customer_tenure_days"], "Churn history")
    x = test[MODEL_FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    if np.isinf(x.to_numpy(dtype=float)).any():
        raise ValueError("Churn features cannot contain infinity")
    # Joblib deserialization is only safe for trusted artifacts.
    model = joblib.load(Path(model_path))
    if list(model.feature_names_in_) != MODEL_FEATURE_COLUMNS or list(model.classes_) != [0, 1]:
        raise ValueError("Churn model does not match the feature or class contract")
    probabilities = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Churn probabilities must be finite and between 0 and 1")
    test["predicted_churn_probability"] = probabilities
    risk = test.loc[test["predicted_churn_probability"].ge(assumptions.risk_threshold)].copy()
    coverage_days = (snapshot - start).days + 1
    exposure = (risk["customer_tenure_days"] + 1).clip(upper=coverage_days, lower=1)
    exposure = exposure.clip(lower=assumptions.minimum_history_days)
    risk["scenario_window_revenue_proxy"] = risk["total_revenue"] * days / exposure
    risk["scenario_revenue_at_risk"] = risk["scenario_window_revenue_proxy"] * risk["predicted_churn_probability"]
    risk["scenario_revenue_protected"] = risk["scenario_revenue_at_risk"] * assumptions.relative_churn_reduction
    risk["scenario_contact_cost"] = assumptions.cost_per_contact
    risk = risk[[
        "customer_id", "snapshot_date", "predicted_churn_probability",
        "scenario_window_revenue_proxy", "scenario_revenue_at_risk",
        "scenario_revenue_protected", "scenario_contact_cost",
    ]].sort_values(["predicted_churn_probability", "customer_id"], ascending=[False, True]).reset_index(drop=True)
    protected = float(risk["scenario_revenue_protected"].sum())
    cost = float(risk["scenario_contact_cost"].sum())
    contribution = protected * assumptions.contribution_margin
    churn_scenario = {
        "snapshot_date": snapshot.strftime("%Y-%m-%d"),
        "observation_days": days,
        "eligible_test_customers": len(test),
        "high_risk_customers": len(risk),
        "expected_churns_at_risk": float(risk["predicted_churn_probability"].sum()),
        "scenario_revenue_at_risk": float(risk["scenario_revenue_at_risk"].sum()),
        "scenario_revenue_protected": protected,
        "scenario_contribution_protected": contribution,
        "scenario_intervention_cost": cost,
        "scenario_roi": (contribution - cost) / cost if cost else None,
    }

    _nonnegative(cohort, ["cohort_size", "retained_customers", "period", "retention_rate"], "Cohort")
    if (
        cohort.duplicated(["cohort_month", "month"]).any()
        or cohort["cohort_size"].le(0).any()
        or cohort["retained_customers"].gt(cohort["cohort_size"]).any()
        or not np.allclose(cohort["retention_rate"], cohort["retained_customers"] / cohort["cohort_size"])
    ):
        raise ValueError("Invalid cohort retention records")
    retention = {}
    for period_number in (1, 3, 6):
        rows = cohort.loc[cohort["period"].eq(period_number)]
        denominator = rows["cohort_size"].sum()
        retention[f"M{period_number}"] = (
            float(rows["retained_customers"].sum() / denominator) if denominator else None
        )

    if forecasts["product_id"].isna().any() or forecasts["product_id"].str.strip().eq("").any() or forecasts.duplicated(["product_id", "forecast_month"]).any():
        raise ValueError("Forecast keys must be present and unique")
    if forecasts.empty or not forecasts["forecast_month"].astype("string").str.fullmatch(r"\d{4}-(0[1-9]|1[0-2])").all():
        raise ValueError("Forecast months must use YYYY-MM format")
    forecast_period = forecast_report["validation_period"]
    if not forecasts["forecast_month"].between(forecast_period["start_month"], forecast_period["end_month"]).all():
        raise ValueError("Walk-forward rows must fall in the recorded validation period")
    if "training_end_month" not in forecasts or forecasts["training_end_month"].isna().any():
        raise ValueError("Walk-forward training end months must be present")
    training_end = pd.PeriodIndex(forecasts["training_end_month"], freq="M")
    forecast_month = pd.PeriodIndex(forecasts["forecast_month"], freq="M")
    if not (training_end == forecast_month - 1).all():
        raise ValueError("Walk-forward training must end in the previous month")
    _nonnegative(forecasts, ["actual_quantity", "predicted_quantity", "absolute_error", "squared_error"], "Forecasts")
    difference = forecasts["actual_quantity"] - forecasts["predicted_quantity"]
    if not np.allclose(forecasts["absolute_error"], difference.abs()) or not np.allclose(forecasts["squared_error"], difference ** 2):
        raise ValueError("Forecast errors do not reconcile with actuals and predictions")
    metrics = regression_metrics(forecasts["actual_quantity"], forecasts["predicted_quantity"].to_numpy())
    recorded = forecast_report["walk_forward_metrics"]
    if len(forecasts) != forecast_report["walk_forward_rows"] or any(
        (metrics[key] is None) != (recorded[key] is None)
        or (metrics[key] is not None and not np.isclose(metrics[key], recorded[key], rtol=1e-8, atol=1e-8))
        for key in metrics
    ):
        raise ValueError("Forecast walk-forward results do not match tuning summary")
    forecasts["underforecast_units"] = difference.clip(lower=0)
    forecasts["overforecast_units"] = (-difference).clip(lower=0)
    forecasts["scenario_lost_revenue_exposure"] = (
        forecasts["underforecast_units"] * assumptions.underforecast_lost_sale_fraction
        * assumptions.assumed_net_revenue_per_unit
    )
    forecasts["scenario_holding_cost_exposure"] = forecasts["overforecast_units"] * assumptions.overforecast_holding_cost_per_unit
    forecasts = forecasts.sort_values(
        ["absolute_error", "forecast_month", "product_id"], ascending=[False, True, True]
    ).reset_index(drop=True)
    forecast_summary = {
        "selected_model": forecast_report["selected_model"],
        "walk_forward_metrics": metrics,
        "product_months": len(forecasts),
        "underforecast_units": float(forecasts["underforecast_units"].sum()),
        "overforecast_units": float(forecasts["overforecast_units"].sum()),
        "scenario_lost_revenue_exposure": float(forecasts["scenario_lost_revenue_exposure"].sum()),
        "scenario_holding_cost_exposure": float(forecasts["scenario_holding_cost_exposure"].sum()),
        "largest_errors": forecasts.head(5).to_dict(orient="records"),
    }
    recommendations = pd.DataFrame([
        {
            "target": "High-risk test-snapshot customers",
            "recommended_action": "Pilot targeted retention outreach with a randomized control group.",
            "expected_business_benefit": f"Scenario revenue protected: {protected:.2f} over {days} days.",
            "assumptions": f"Risk >= {assumptions.risk_threshold}; relative churn reduction {assumptions.relative_churn_reduction:.1%}; cost {assumptions.cost_per_contact:.2f} per contact.",
            "limitations": "Retrospective test subset; probabilities may be uncalibrated; response is assumed, not causal.",
        },
        {
            "target": "At Risk and Can't Lose Them RFM segments",
            "recommended_action": "Prioritize value-aware retention pilots and validate with holdouts.",
            "expected_business_benefit": "Protect high-value customer relationships; segment CLV shares identify concentration.",
            "assumptions": "RFM segments and annual revenue proxy are descriptive at the later CLV snapshot.",
            "limitations": "Not joined to earlier churn scores; segment labels and CLV include later behavior.",
        },
        {
            "target": "Highest-error product-months in walk-forward validation",
            "recommended_action": "Review demand drivers, stock buffers and forecast overrides for the largest errors.",
            "expected_business_benefit": "Potentially reduce stockout and holding-cost exposure; no savings are claimed.",
            "assumptions": f"Unit revenue {assumptions.assumed_net_revenue_per_unit:.2f}; lost-sale fraction {assumptions.underforecast_lost_sale_fraction:.1%}; holding cost {assumptions.overforecast_holding_cost_per_unit:.2f} per excess unit.",
            "limitations": "Forecast error is not observed inventory or lost sales; validation period is synthetic and retrospective.",
        },
    ])
    summary: dict[str, Any] = {
        "nature": "Illustrative scenario estimates on synthetic data; not actual outcomes or savings.",
        "assumptions": asdict(assumptions),
        "churn_model": {
            "selected_model": churn_report["selected_model"],
            "test_metrics": churn_report["test_metrics"],
            "caveat": churn_report["evaluation_caveat"],
        },
        "churn_scenario": churn_scenario,
        "customer_value": {
            "snapshot_date": clv_metadata["snapshot_date"],
            "customer_count": len(customers),
            "historical_revenue": float(customers["historical_revenue"].sum()),
            "estimated_clv": float(customers["estimated_clv"].sum()),
            "priority_segments": ["At Risk", "Can't Lose Them"],
            "weighted_retention_by_observed_cohort": retention,
        },
        "forecasting": forecast_summary,
        "formulas": {
            "window_revenue_proxy": "snapshot historical revenue * observation days / max(minimum history days, min(tenure days + 1, observed coverage days))",
            "revenue_at_risk": "sum(window revenue proxy * predicted churn probability) for high-risk test customers",
            "revenue_protected": "revenue at risk * assumed relative churn reduction",
            "roi": "(revenue protected * contribution margin - contact cost) / contact cost; null when cost is zero",
            "lost_revenue_exposure": "sum(max(actual - predicted, 0)) * assumed lost-sale fraction * assumed unit revenue",
            "holding_cost_exposure": "sum(max(predicted - actual, 0)) * assumed holding cost per unit",
        },
        "limitations": [
            "Synthetic data and arbitrary assumptions do not establish realized ROI or causal treatment effects.",
            "Only the held-out test customer subset is scored; no extrapolation to all customers.",
            "Churn probabilities have not been calibrated; revenue proxy is historical, not a forecast of causal revenue loss.",
            "Later CLV and RFM segments are descriptive only and never used to target the earlier churn snapshot.",
            "The segment snapshot is inferred from customer feature metadata; segment provenance is not independently verified.",
            "Forecast impacts are gross error-based exposure proxies, not observed stockouts, inventory costs, or savings.",
            "Churn test results and forecast validation have been previously inspected; use independent future evaluation.",
            "Monetary amounts use the source's unspecified currency units; no taxes or additional program costs are included.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    segment_table.to_csv(output_dir / "business_impact_segments.csv", index=False)
    forecasts.to_csv(output_dir / "business_impact_forecasting.csv", index=False)
    recommendations.to_csv(output_dir / "business_recommendations.csv", index=False)
    risk.to_csv(output_dir / "business_impact_high_risk_customers.csv", index=False)
    with (output_dir / "business_impact_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
        file.write("\n")
    return summary, segment_table, forecasts, recommendations, risk


if __name__ == "__main__":
    report, _, _, _, _ = run_business_impact()
    print(json.dumps({"churn_scenario": report["churn_scenario"], "forecasting": report["forecasting"]}, indent=2))

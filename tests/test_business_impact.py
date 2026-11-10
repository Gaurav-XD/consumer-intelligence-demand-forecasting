"""Tests for descriptive reporting and scenario-based impact estimates."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from consumer_intelligence.business_impact import ScenarioAssumptions, run_business_impact
from consumer_intelligence.churn_dataset import MODEL_FEATURE_COLUMNS


class FixtureModel:
    feature_names_in_ = np.array(MODEL_FEATURE_COLUMNS)
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        probabilities = np.where(features["total_revenue"].to_numpy() > 0, 0.8, 0.2)
        return np.column_stack((1 - probabilities, probabilities))


class BusinessImpactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.processed = self.root / "processed"
        self.reports = self.root / "reports"
        self.output = self.root / "output"
        self.processed.mkdir()
        self.reports.mkdir()
        self.model = self.root / "model.joblib"
        joblib.dump(FixtureModel(), self.model)
        pd.DataFrame([
            {"customer_id": "A", "segment": "At Risk", "total_revenue": 500},
            {"customer_id": "B", "segment": "Champions", "total_revenue": 100},
        ]).to_csv(self.processed / "customer_segments.csv", index=False)
        pd.DataFrame([
            {"customer_id": "A", "historical_revenue": 500, "estimated_clv": 250},
            {"customer_id": "B", "historical_revenue": 100, "estimated_clv": 50},
        ]).to_csv(self.processed / "customer_clv.csv", index=False)
        rows = []
        for customer, revenue in (("A", 365), ("B", 0)):
            row = {name: 0.0 for name in MODEL_FEATURE_COLUMNS}
            row.update(
                customer_id=customer, split="test", snapshot_date="2025-09-30",
                total_revenue=revenue, customer_tenure_days=1000, churn=0,
            )
            rows.append(row)
        pd.DataFrame(rows).to_csv(self.processed / "churn_dataset.csv", index=False)
        pd.DataFrame([
            {"cohort_month": "2025-01", "month": "2025-02", "period": 1,
             "cohort_size": 10, "retained_customers": 4, "retention_rate": 0.4},
            {"cohort_month": "2025-02", "month": "2025-03", "period": 1,
             "cohort_size": 30, "retained_customers": 6, "retention_rate": 0.2},
        ]).to_csv(self.processed / "cohort_retention.csv", index=False)
        pd.DataFrame([
            {"product_id": "P1", "forecast_month": "2025-04", "actual_quantity": 10,
             "predicted_quantity": 6, "absolute_error": 4, "squared_error": 16, "training_end_month": "2025-03"},
            {"product_id": "P2", "forecast_month": "2025-04", "actual_quantity": 5,
             "predicted_quantity": 7, "absolute_error": 2, "squared_error": 4, "training_end_month": "2025-03"},
        ]).to_csv(self.reports / "forecasting_walk_forward.csv", index=False)
        self.write_json("churn_tuning_summary.json", {
            "feature_columns": MODEL_FEATURE_COLUMNS, "selected_model": "Fixture",
            "test_metrics": {}, "evaluation_caveat": "Test fixture",
        })
        self.write_json("churn_dataset_metadata.json", {
            "feature_columns": MODEL_FEATURE_COLUMNS,
            "periods": {"test": {
                "row_count": 2,
                "feature_snapshot_period": {
                    "transaction_history_start": "2024-10-01",
                    "snapshot_date_inclusive": "2025-09-30",
                },
                "churn_observation_window": {
                    "days": 90, "start_inclusive": "2025-10-01", "end_inclusive": "2025-12-29",
                },
            }},
        })
        self.write_json("clv_summary.json", {"snapshot_date": "2025-12-31"})
        self.write_json("customer_features_metadata.json", {"as_of_date": "2025-12-31"})
        self.write_json("forecasting_tuning_summary.json", {
            "selected_model": "Fixture", "walk_forward_rows": 2,
            "walk_forward_metrics": {"mae": 3.0, "rmse": np.sqrt(10), "mape": 40.0, "wape": 40.0},
            "validation_period": {"start_month": "2025-04", "end_month": "2025-06"},
        })

    def write_json(self, filename, data):
        (self.reports / filename).write_text(json.dumps(data), encoding="utf-8")

    def run_report(self, assumptions=ScenarioAssumptions()):
        return run_business_impact(self.processed, self.reports, self.model, assumptions, self.output)

    def test_scenario_calculations_outputs_and_immutability(self):
        files = list(self.processed.iterdir()) + list(self.reports.iterdir()) + [self.model]
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        summary, segments, forecasts, recommendations, risk = self.run_report()
        scenario = summary["churn_scenario"]
        self.assertEqual(list(risk["customer_id"]), ["A"])
        self.assertAlmostEqual(scenario["scenario_revenue_at_risk"], 90 * 0.8)
        self.assertAlmostEqual(scenario["scenario_revenue_protected"], 7.2)
        self.assertAlmostEqual(scenario["scenario_intervention_cost"], 5)
        self.assertAlmostEqual(scenario["scenario_roi"], (7.2 * 0.3 - 5) / 5)
        self.assertEqual(segments["customer_count"].sum(), 2)
        self.assertEqual(segments["historical_revenue"].sum(), 600)
        self.assertEqual(segments["estimated_clv"].sum(), 300)
        self.assertAlmostEqual(segments["estimated_clv_share"].sum(), 1)
        self.assertAlmostEqual(summary["customer_value"]["weighted_retention_by_observed_cohort"]["M1"], 0.25)
        self.assertEqual(forecasts.iloc[0]["product_id"], "P1")
        self.assertAlmostEqual(summary["forecasting"]["scenario_lost_revenue_exposure"], 100)
        self.assertAlmostEqual(summary["forecasting"]["scenario_holding_cost_exposure"], 20)
        self.assertEqual(len(recommendations), 3)
        self.assertTrue((forecasts[["underforecast_units", "overforecast_units",
                                   "scenario_lost_revenue_exposure", "scenario_holding_cost_exposure"]] >= 0).all().all())
        self.assertEqual(set(path.name for path in self.output.iterdir()), {
            "business_impact_summary.json", "business_impact_segments.csv",
            "business_impact_forecasting.csv", "business_recommendations.csv",
            "business_impact_high_risk_customers.csv",
        })
        self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
        first = {path.name: path.read_bytes() for path in self.output.iterdir()}
        self.run_report()
        self.assertEqual(first, {path.name: path.read_bytes() for path in self.output.iterdir()})

    def test_zero_cost_and_no_high_risk(self):
        summary, _, _, _, risk = self.run_report(ScenarioAssumptions(risk_threshold=1, cost_per_contact=0))
        self.assertEqual(len(risk), 0)
        self.assertEqual(summary["churn_scenario"]["scenario_revenue_protected"], 0)
        self.assertIsNone(summary["churn_scenario"]["scenario_roi"])

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.run_report(ScenarioAssumptions(contribution_margin=1.1))
        clv_path = self.processed / "customer_clv.csv"
        clv = pd.read_csv(clv_path)
        clv.loc[0, "historical_revenue"] = 501
        clv.to_csv(clv_path, index=False)
        with self.assertRaisesRegex(ValueError, "reconcile"):
            self.run_report()

    def test_later_values_and_labels_do_not_change_targeting(self):
        _, _, _, _, initial = self.run_report()
        for name in ("customer_segments.csv", "customer_clv.csv"):
            path = self.processed / name
            data = pd.read_csv(path)
            if name == "customer_segments.csv":
                data["total_revenue"] *= 2
                data["segment"] = "At Risk"
            else:
                data["historical_revenue"] *= 2
                data["estimated_clv"] *= 2
            data.to_csv(path, index=False)
        churn_path = self.processed / "churn_dataset.csv"
        churn = pd.read_csv(churn_path)
        churn["churn"] = 1 - churn["churn"]
        churn.to_csv(churn_path, index=False)
        _, _, _, _, changed = self.run_report()
        pd.testing.assert_frame_equal(initial, changed)


if __name__ == "__main__":
    unittest.main()

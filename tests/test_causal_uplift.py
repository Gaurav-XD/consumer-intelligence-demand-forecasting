"""Verify temporal and statistical properties of the placebo workflow."""

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from consumer_intelligence.causal_uplift import (
    MODEL_FEATURE_COLUMNS, assign_hypothetical_treatment, evaluate_ranked_uplift,
    run_causal_uplift,
)


class CausalUpliftTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.churn = root / "churn.csv"
        self.features = root / "features.csv"
        self.marketing = root / "marketing.csv"
        self.metadata = root / "metadata.json"
        self.dataset = root / "output" / "dataset.csv"
        self.summary = root / "output" / "summary.json"
        self.ranked = root / "output" / "ranked.csv"
        self.model = root / "output" / "model.joblib"
        rows = []
        for i in range(240):
            row = {name: float(i % 7) for name in MODEL_FEATURE_COLUMNS}
            row.update(
                customer_id=f"C{i:03}", split="train" if i < 160 else "test",
                snapshot_date="2025-03-31" if i < 160 else "2025-09-30",
                churn=(i // 3) % 2, number_marketing_interactions=0,
                number_marketing_campaigns=0,
            )
            rows.append(row)
        pd.DataFrame(rows).to_csv(self.churn, index=False)
        pd.DataFrame({"customer_id": [row["customer_id"] for row in rows],
                      "total_revenue": [999999] * len(rows)}).to_csv(self.features, index=False)
        pd.DataFrame([{"customer_id": "C000", "campaign": "after",
                       "campaign_date": "2025-10-01"}]).to_csv(self.marketing, index=False)
        info = {
            "feature_columns": MODEL_FEATURE_COLUMNS, "target_column": "churn",
            "periods": {
                "train": {"row_count": 160,
                          "feature_snapshot_period": {"snapshot_date_inclusive": "2025-03-31"},
                          "churn_observation_window": {"start_inclusive": "2025-04-01", "end_inclusive": "2025-06-29", "days": 90}},
                "test": {"row_count": 80,
                         "feature_snapshot_period": {"snapshot_date_inclusive": "2025-09-30"},
                         "churn_observation_window": {"start_inclusive": "2025-10-01", "end_inclusive": "2025-12-29", "days": 90}},
            },
        }
        self.metadata.write_text(json.dumps(info), encoding="utf-8")

    def run_report(self):
        return run_causal_uplift(
            self.churn, self.features, self.marketing, self.metadata, self.dataset,
            self.summary, self.ranked, self.model,
        )

    def test_artifacts_integrity_and_reproducibility(self):
        inputs = [self.churn, self.features, self.marketing, self.metadata]
        before = {path: path.read_bytes() for path in inputs}
        dataset, summary, ranked = self.run_report()
        self.assertEqual(len(dataset), 240)
        self.assertTrue(dataset["customer_id"].is_unique)
        self.assertEqual(set(dataset["treatment"]), {0, 1})
        self.assertTrue(dataset["treatment_is_hypothetical"].all())
        self.assertEqual(set(dataset["outcome"]), {0, 1})
        self.assertIsNone(summary["estimated_causal_treatment_effect"])
        self.assertTrue(dataset["snapshot_date"].eq(dataset["assignment_date"]).all())
        self.assertTrue((dataset["assignment_date"] < dataset["outcome_start"]).all())
        self.assertTrue((dataset["outcome_start"] <= dataset["outcome_end"]).all())
        self.assertTrue(dataset.loc[dataset["split"].eq("train"), "predicted_uplift"].isna().all())
        test = dataset.loc[dataset["split"].eq("test")]
        self.assertTrue(test["predicted_uplift"].between(-1, 1).all())
        self.assertEqual(len(ranked), 10)
        self.assertEqual(ranked["customer_count"].sum(), 80)
        self.assertAlmostEqual(summary["holdout_observed_retention_difference"],
                               test.loc[test.treatment.eq(1), "outcome"].mean()
                               - test.loc[test.treatment.eq(0), "outcome"].mean())
        self.assertEqual(summary, json.loads(self.summary.read_text(encoding="utf-8")))
        self.assertTrue(all(path.is_file() for path in [self.dataset, self.summary, self.ranked, self.model]))
        model = joblib.load(self.model)
        np.testing.assert_allclose(model.predict(test[MODEL_FEATURE_COLUMNS]), test["predicted_uplift"])
        self.assertFalse({"treatment", "outcome", "churn", "customer_id", "snapshot_date"} & set(model.feature_names_in_))
        self.assertEqual(before, {path: path.read_bytes() for path in inputs})
        first = [path.read_bytes() for path in [self.dataset, self.summary, self.ranked, self.model]]
        self.run_report()
        self.assertEqual(first, [path.read_bytes() for path in [self.dataset, self.summary, self.ranked, self.model]])

    def test_ranked_calculations(self):
        sample = pd.DataFrame({
            "customer_id": ["A", "B", "C", "D"],
            "treatment": [1, 0, 1, 0], "outcome": [1, 0, 0, 1],
            "predicted_uplift": [0.4, 0.3, 0.2, 0.1],
        })
        groups, evaluation = evaluate_ranked_uplift(sample, groups=2)
        self.assertEqual(groups["observed_retention_difference"].tolist(), [1.0, -1.0])
        self.assertAlmostEqual(evaluation["full_population_ipw_difference"], 0)
        self.assertAlmostEqual(evaluation["auuc"], 0.375)
        self.assertAlmostEqual(evaluation["qini_style_area"], 0.375)
        sample.loc[1, "treatment"] = 1
        groups, _ = evaluate_ranked_uplift(sample, groups=2)
        self.assertTrue(pd.isna(groups.iloc[0]["observed_retention_difference"]))

    def test_future_marketing_and_outcomes_do_not_affect_scores(self):
        first, _, _ = self.run_report()
        marketing = pd.read_csv(self.marketing)
        marketing.loc[len(marketing)] = ["C001", "future", "2025-10-05"]
        marketing.to_csv(self.marketing, index=False)
        churn = pd.read_csv(self.churn)
        mask = churn["split"].eq("test")
        churn.loc[mask, "churn"] = 1 - churn.loc[mask, "churn"]
        churn.to_csv(self.churn, index=False)
        second, _, _ = self.run_report()
        pd.testing.assert_series_equal(first["treatment"], second["treatment"])
        pd.testing.assert_series_equal(first["predicted_uplift"], second["predicted_uplift"])
        self.assertEqual(assign_hypothetical_treatment("C000"), assign_hypothetical_treatment("C000"))
        marketing.loc[len(marketing)] = ["C000", "past", "2025-03-31"]
        marketing.to_csv(self.marketing, index=False)
        with self.assertRaisesRegex(ValueError, "pre-snapshot"):
            self.run_report()

    def test_bad_periods_and_untrusted_features_rejected(self):
        with self.assertRaisesRegex(ValueError, "Output paths"):
            run_causal_uplift(self.churn, self.features, self.marketing, self.metadata, self.churn,
                              self.summary, self.ranked, self.model)
        metadata = json.loads(self.metadata.read_text(encoding="utf-8"))
        metadata["periods"]["train"]["churn_observation_window"]["end_inclusive"] = "2025-10-29"
        self.metadata.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_report()


if __name__ == "__main__":
    unittest.main()

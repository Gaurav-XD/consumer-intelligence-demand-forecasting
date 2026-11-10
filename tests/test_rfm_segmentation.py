"""Tests for deterministic RFM scoring and segment reporting."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from consumer_intelligence.rfm_segmentation import (
    SEGMENT_ORDER,
    run_rfm_segmentation,
    segment_customers,
)


class RfmSegmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.input_path = self.root / "customer_features.csv"
        self.output_path = self.root / "customer_segments.csv"
        self.summary_path = self.root / "rfm_summary.json"
        self.distribution_path = self.root / "rfm_distribution.csv"
        self.features = pd.DataFrame(
            [
                ["C1", 1, 12, 1200.0, 1200.0],
                ["C2", 5, 8, 800.0, 800.0],
                ["C3", 15, 5, 400.0, 400.0],
                ["C4", 45, 2, 100.0, 100.0],
                ["C5", 120, 1, 25.0, 25.0],
                ["C6", None, 0, 0.0, 0.0],
                ["C7", 45, 2, 100.0, 100.0],
            ],
            columns=[
                "customer_id",
                "days_since_last_purchase",
                "frequency_orders",
                "monetary_total_revenue",
                "total_revenue",
            ],
        )
        self.features.to_csv(self.input_path, index=False)
        self.input_bytes = self.input_path.read_bytes()

    def test_customer_uniqueness_scores_and_segment_assignments(self) -> None:
        result = segment_customers(self.features)

        self.assertEqual(len(result), len(self.features))
        self.assertEqual(result["customer_id"].nunique(), len(self.features))
        for score_column in ("R_score", "F_score", "M_score"):
            self.assertTrue(result[score_column].between(1, 5).all())
        self.assertFalse(result["segment"].isna().any())
        self.assertGreater(
            result.loc[result["customer_id"] == "C1", "R_score"].iloc[0],
            result.loc[result["customer_id"] == "C5", "R_score"].iloc[0],
        )
        self.assertGreater(
            result.loc[result["customer_id"] == "C1", "F_score"].iloc[0],
            result.loc[result["customer_id"] == "C5", "F_score"].iloc[0],
        )
        self.assertEqual(result.loc[result["customer_id"] == "C6", "F_score"].iloc[0], 1)
        self.assertEqual(result.loc[result["customer_id"] == "C6", "M_score"].iloc[0], 1)
        self.assertEqual(
            result.loc[result["customer_id"] == "C1", "segment"].iloc[0],
            "Champions",
        )
        tied = result[result["customer_id"].isin(["C4", "C7"])]
        self.assertEqual(tied["R_score"].nunique(), 1)
        self.assertEqual(tied["F_score"].nunique(), 1)
        self.assertEqual(tied["M_score"].nunique(), 1)

    def test_outputs_reconcile_and_are_deterministic_without_input_changes(self) -> None:
        self.features.to_csv(self.input_path, index=False)
        first_segments, first_summary = run_rfm_segmentation(
            self.input_path,
            self.output_path,
            self.summary_path,
            self.distribution_path,
        )
        first_output_bytes = self.output_path.read_bytes()
        second_segments, second_summary = run_rfm_segmentation(
            self.input_path,
            self.output_path,
            self.summary_path,
            self.distribution_path,
        )

        self.assertTrue(first_segments.equals(second_segments))
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(first_output_bytes, self.output_path.read_bytes())
        self.assertEqual(self.input_bytes, self.input_path.read_bytes())
        self.assertEqual(first_summary["customer_count"], len(self.features))
        self.assertEqual(
            sum(first_summary["segment_counts"].values()), len(self.features)
        )
        self.assertAlmostEqual(
            sum(first_summary["segment_percentages"].values()), 100.0, places=2
        )
        self.assertAlmostEqual(
            sum(first_summary["revenue_by_segment"].values()),
            self.features["total_revenue"].sum(),
            places=2,
        )
        self.assertEqual(first_summary["segment_count"], len(SEGMENT_ORDER))
        saved_summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        self.assertEqual(saved_summary, first_summary)
        distribution = pd.read_csv(self.distribution_path)
        self.assertEqual(int(distribution["customer_count"].sum()), len(self.features))


if __name__ == "__main__":
    unittest.main()

"""Tests for customer-level, snapshot-safe CLV estimates."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from consumer_intelligence.clv import run_clv


class ClvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.features = self.root / "features.csv"
        self.feature_metadata = self.root / "feature_metadata.json"
        self.segments = self.root / "segments.csv"
        self.output = self.root / "clv.csv"
        self.summary = self.root / "summary.json"
        self.segment_output = self.root / "by_segment.csv"
        self.write_csv(
            self.raw / "customers.csv",
            [
                {"customer_id": "C1", "signup_date": "2023-01-01"},
                {"customer_id": "C2", "signup_date": "2024-01-05"},
                {"customer_id": "C3", "signup_date": "2024-01-06"},
            ],
        )
        self.write_csv(self.raw / "products.csv", [{"product_id": "P1"}])
        self.write_csv(
            self.raw / "transactions.csv",
            [
                {"transaction_id": "T1", "customer_id": "C1", "product_id": "P1",
                 "transaction_date": "2024-01-01", "quantity": 2, "price": 10, "discount": 0.1},
                {"transaction_id": "T2", "customer_id": "C1", "product_id": "P1",
                 "transaction_date": "2024-01-10", "quantity": 1, "price": 20, "discount": 0.25},
                {"transaction_id": "T3", "customer_id": "C2", "product_id": "P1",
                 "transaction_date": "2024-01-08", "quantity": 1, "price": 5, "discount": 0},
                {"transaction_id": "T4", "customer_id": "C1", "product_id": "P1",
                 "transaction_date": "2024-01-11", "quantity": 10, "price": 100, "discount": 0},
            ],
        )
        features = [
            {"customer_id": "C1", "total_orders": 2, "total_revenue": 33, "days_since_last_purchase": 0},
            {"customer_id": "C2", "total_orders": 1, "total_revenue": 5, "days_since_last_purchase": 2},
            {"customer_id": "C3", "total_orders": 0, "total_revenue": 0, "days_since_last_purchase": None},
        ]
        self.write_csv(self.features, features)
        self.write_csv(
            self.segments,
            [dict(row, segment="Active" if row["customer_id"] != "C3" else "No purchases") for row in features],
        )
        self.feature_metadata.write_text(
            json.dumps({"as_of_date": "2024-01-10"}), encoding="utf-8"
        )
        self.inputs = [*self.raw.glob("*.csv"), self.features, self.feature_metadata, self.segments]
        self.before = {path: path.read_bytes() for path in self.inputs}

    @staticmethod
    def write_csv(path: Path, rows: list[dict]) -> None:
        pd.DataFrame(rows).to_csv(path, index=False)

    def run_report(self) -> tuple[pd.DataFrame, dict]:
        return run_clv(
            self.raw, self.features, self.feature_metadata, self.segments,
            self.output, self.summary, self.segment_output,
        )

    def test_customer_formulas_and_snapshot(self) -> None:
        frame, summary = self.run_report()
        indexed = frame.set_index("customer_id")
        self.assertEqual(len(frame), 3)
        self.assertTrue(frame["customer_id"].is_unique)
        self.assertEqual(set(frame["customer_id"]), {"C1", "C2", "C3"})
        self.assertEqual(indexed.loc["C1", "historical_revenue"], 33)
        self.assertEqual(indexed.loc["C1", "total_orders"], 2)
        self.assertEqual(indexed.loc["C1", "average_order_value"], 16.5)
        self.assertEqual(indexed.loc["C1", "customer_lifetime_days"], 9)
        self.assertEqual(indexed.loc["C1", "observed_days"], 10)
        self.assertEqual(indexed.loc["C1", "annualization_days"], 365)
        self.assertAlmostEqual(indexed.loc["C1", "purchase_frequency"], 2 * 365.25 / 365)
        self.assertEqual(indexed.loc["C1", "estimated_clv"], round(33 * 365.25 / 365, 2))
        self.assertEqual(indexed.loc["C3", "estimated_clv"], 0)
        self.assertEqual(indexed.loc["C3", "average_order_value"], 0)
        self.assertTrue(pd.isna(indexed.loc["C3", "customer_lifetime_days"]))
        self.assertTrue((frame[["historical_revenue", "estimated_clv"]] >= 0).all().all())
        self.assertEqual(summary["total_historical_revenue"], 38)
        self.assertEqual(frame["clv_rank"].tolist(), [1, 2, 3])
        self.assertEqual(
            list(frame.columns),
            ["customer_id", "clv_rank", "historical_revenue", "total_orders",
             "average_order_value", "purchase_frequency", "customer_lifetime_days",
             "observed_days", "annualization_days", "estimated_annual_revenue", "estimated_clv"],
        )

    def test_outputs_segments_and_inputs_unchanged(self) -> None:
        frame, summary = self.run_report()
        self.assertEqual(len(pd.read_csv(self.output)), len(frame))
        self.assertEqual(json.loads(self.summary.read_text(encoding="utf-8")), summary)
        self.assertTrue(summary["segment_summary_available"])
        segments = pd.read_csv(self.segment_output).set_index("segment")
        self.assertEqual(segments.loc["Active", "customer_count"], 2)
        self.assertEqual(segments.loc["Active", "historical_revenue"], 38)
        self.assertEqual({path: path.read_bytes() for path in self.inputs}, self.before)

    def test_missing_segments_and_stale_features(self) -> None:
        self.segments.unlink()
        _, summary = self.run_report()
        self.assertFalse(summary["segment_summary_available"])
        self.assertTrue(pd.read_csv(self.segment_output).empty)
        features = pd.read_csv(self.features)
        features.loc[0, "total_revenue"] = 99
        features.to_csv(self.features, index=False)
        with self.assertRaisesRegex(ValueError, "do not match raw history"):
            self.run_report()

    def test_invalid_transactions_are_rejected(self) -> None:
        transactions = pd.read_csv(self.raw / "transactions.csv")
        transactions.loc[0, "price"] = -1
        transactions.to_csv(self.raw / "transactions.csv", index=False)
        with self.assertRaisesRegex(ValueError, "invalid quantity"):
            self.run_report()


if __name__ == "__main__":
    unittest.main()

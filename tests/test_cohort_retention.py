"""Tests for first-purchase cohort and retention reporting."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from consumer_intelligence.cohort_retention import run_cohort_retention


class CohortRetentionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.raw = root / "raw"
        self.raw.mkdir()
        self.output = root / "cohorts.csv"
        self.summary = root / "summary.json"
        self.matrix = root / "matrix.csv"
        pd.DataFrame(
            [{"customer_id": customer_id, "signup_date": "2023-01-01"}
             for customer_id in ("C1", "C2", "C3", "C4", "C5")]
        ).to_csv(self.raw / "customers.csv", index=False)
        rows = [
            ("T1", "C1", "2024-01-01", 1, 10, 0),
            ("T2", "C1", "2024-01-15", 2, 10, 0.1),
            ("T3", "C2", "2024-01-20", 1, 20, 0),
            ("T4", "C1", "2024-02-01", 1, 25, 0),
            ("T5", "C1", "2024-02-02", 1, 5, 0),
            ("T6", "C3", "2024-02-10", 1, 10, 1),
            ("T7", "C3", "2024-03-10", 1, 10, 0),
            ("T8", "C4", "2024-04-30", 1, 10, 0),
        ]
        pd.DataFrame(
            rows, columns=["transaction_id", "customer_id", "transaction_date",
                           "quantity", "price", "discount"]
        ).to_csv(self.raw / "transactions.csv", index=False)
        self.inputs = {path: path.read_bytes() for path in self.raw.glob("*.csv")}

    def run_report(self) -> tuple[pd.DataFrame, dict]:
        return run_cohort_retention(self.raw, self.output, self.summary, self.matrix)

    def test_cohorts_retention_and_revenue(self) -> None:
        frame, report = self.run_report()
        rows = frame.set_index(["cohort_month", "period"])
        self.assertEqual(report["cohort_count"], 3)
        self.assertEqual(report["customers_without_observed_purchases"], 1)
        self.assertFalse(frame.duplicated(["cohort_month", "month"]).any())
        self.assertTrue(frame["retention_rate"].between(0, 1).all())
        self.assertTrue(frame.loc[frame["period"].eq(0), "retention_rate"].eq(1).all())
        self.assertEqual(rows.loc[("2024-01", 0), "cohort_size"], 2)
        self.assertEqual(rows.loc[("2024-01", 0), "retained_customers"], 2)
        self.assertEqual(rows.loc[("2024-01", 0), "cohort_revenue"], 48)
        self.assertEqual(rows.loc[("2024-01", 1), "retained_customers"], 1)
        self.assertEqual(rows.loc[("2024-01", 1), "cohort_revenue"], 30)
        self.assertEqual(rows.loc[("2024-01", 1), "retention_rate"], 0.5)
        self.assertAlmostEqual(rows.loc[("2024-01", 1), "revenue_retention"], 30 / 48)
        self.assertEqual(rows.loc[("2024-01", 2), "retention_rate"], 0)
        self.assertTrue(pd.isna(rows.loc[("2024-02", 1), "revenue_retention"]))
        self.assertEqual(frame["cohort_revenue"].sum(), report["total_observed_revenue"])
        self.assertEqual(report["total_observed_revenue"], 98)
        february = next(row for row in report["monthly_active_customers"] if row["month"] == "2024-02")
        self.assertEqual(february["active_customers"], 2)
        self.assertEqual(february["repeat_customers"], 1)
        self.assertEqual(february["repeat_purchase_rate"], 0.5)
        self.assertEqual(report["customer_retention"]["M1"]["retention_rate"], 2 / 3)
        self.assertIsNone(report["customer_retention"]["M6"]["retention_rate"])

    def test_outputs_and_raw_data_are_unchanged(self) -> None:
        frame, report = self.run_report()
        self.assertEqual(len(pd.read_csv(self.output)), len(frame))
        self.assertEqual(json.loads(self.summary.read_text(encoding="utf-8")), report)
        matrix = pd.read_csv(self.matrix).set_index("cohort_month")
        self.assertEqual(matrix.loc["2024-01", "M0"], 1)
        self.assertEqual(matrix.loc["2024-01", "M1"], 0.5)
        self.assertEqual(matrix.loc["2024-01", "M2"], 0)
        self.assertTrue(pd.isna(matrix.loc["2024-04", "M1"]))
        self.assertIn("M12", matrix.columns)
        self.assertEqual({path: path.read_bytes() for path in self.inputs}, self.inputs)

    def test_partial_final_month_is_not_assumed_complete(self) -> None:
        transactions = pd.read_csv(self.raw / "transactions.csv")
        transactions.loc[transactions["transaction_id"].eq("T8"), "transaction_date"] = "2024-04-10"
        transactions.to_csv(self.raw / "transactions.csv", index=False)
        frame, report = self.run_report()
        self.assertEqual(report["observation_end_month"], "2024-03")
        self.assertNotIn("2024-04", set(frame["cohort_month"]))
        self.assertEqual(report["customers_without_observed_purchases"], 2)

    def test_missing_month_is_observed_zero(self) -> None:
        transactions = pd.read_csv(self.raw / "transactions.csv")
        transactions = transactions[transactions["transaction_id"] != "T7"]
        transactions.to_csv(self.raw / "transactions.csv", index=False)
        frame, report = self.run_report()
        march = next(item for item in report["monthly_active_customers"] if item["month"] == "2024-03")
        self.assertEqual(march["active_customers"], 0)
        self.assertIsNone(march["repeat_purchase_rate"])
        self.assertEqual(frame.set_index(["cohort_month", "period"]).loc[("2024-01", 2), "retained_customers"], 0)


if __name__ == "__main__":
    unittest.main()

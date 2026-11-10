"""Tests for complete monthly product demand aggregation."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from consumer_intelligence.forecasting_dataset import run_forecasting_dataset


class ForecastingDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.raw = root / "raw"
        self.raw.mkdir()
        self.output = root / "output" / "demand.csv"
        self.summary = root / "reports" / "summary.json"
        pd.DataFrame(
            [
                {"product_id": "P1", "category": "Home", "brand": "A"},
                {"product_id": "P2", "category": "Beauty", "brand": "B"},
                {"product_id": "P3", "category": "Home", "brand": "C"},
            ]
        ).to_csv(self.raw / "products.csv", index=False)
        pd.DataFrame(
            [
                {"transaction_id": "T1", "customer_id": "C1", "product_id": "P1",
                 "transaction_date": "2024-01-15", "quantity": 2, "price": 10, "discount": 0.1},
                {"transaction_id": "T2", "customer_id": "C1", "product_id": "P1",
                 "transaction_date": "2024-01-16", "quantity": 1, "price": 20, "discount": 0.25},
                {"transaction_id": "T3", "customer_id": "C2", "product_id": "P2",
                 "transaction_date": "2024-03-05", "quantity": 1, "price": 1.01, "discount": 0.05},
            ]
        ).to_csv(self.raw / "transactions.csv", index=False)
        self.original = {path: path.read_bytes() for path in self.raw.glob("*.csv")}

    def test_calendar_values_and_revenue_reconciliation(self) -> None:
        result, summary = run_forecasting_dataset(self.raw, self.output, self.summary)
        self.assertEqual(len(result), 9)
        self.assertFalse(result.duplicated(["product_id", "month"]).any())
        self.assertEqual(
            list(zip(result["product_id"], result["month"])),
            [(product, month) for product in ("P1", "P2", "P3")
             for month in ("2024-01", "2024-02", "2024-03")],
        )
        indexed = result.set_index(["product_id", "month"])
        self.assertEqual(indexed.loc[("P1", "2024-01"), "quantity_sold"], 3)
        self.assertAlmostEqual(indexed.loc[("P1", "2024-01"), "revenue"], 33)
        self.assertEqual(indexed.loc[("P1", "2024-01"), "transaction_count"], 2)
        self.assertEqual(indexed.loc[("P1", "2024-01"), "unique_customers"], 1)
        self.assertEqual(indexed.loc[("P1", "2024-01"), "average_price"], 15)
        self.assertAlmostEqual(indexed.loc[("P1", "2024-01"), "average_discount"], 0.175)
        self.assertEqual(indexed.loc[("P1", "2024-02"), "quantity_sold"], 0)
        self.assertEqual(indexed.loc[("P1", "2024-02"), "revenue"], 0)
        self.assertTrue(pd.isna(indexed.loc[("P1", "2024-02"), "average_price"]))
        self.assertEqual(indexed.loc[("P2", "2024-03"), "brand"], "B")
        self.assertEqual(indexed.loc[("P3", "2024-02"), "category"], "Home")
        self.assertTrue((result[["quantity_sold", "revenue"]] >= 0).all().all())
        self.assertAlmostEqual(result["revenue"].sum(), 33 + 1.01 * 0.95)
        self.assertEqual(summary["transaction_date_range"], {"minimum": "2024-01-15", "maximum": "2024-03-05"})
        self.assertEqual(summary["total_quantity"], 4)
        self.assertEqual(summary["zero_sales_product_months"], 7)
        self.assertEqual(json.loads(self.summary.read_text(encoding="utf-8")), summary)
        self.assertEqual(len(pd.read_csv(self.output)), 9)
        self.assertEqual({path: path.read_bytes() for path in self.original}, self.original)

    def test_invalid_product_and_empty_history_fail(self) -> None:
        transactions = pd.read_csv(self.raw / "transactions.csv")
        transactions.loc[0, "product_id"] = "UNKNOWN"
        transactions.to_csv(self.raw / "transactions.csv", index=False)
        with self.assertRaisesRegex(ValueError, "unknown product"):
            run_forecasting_dataset(self.raw, self.output, self.summary)
        transactions.iloc[:0].to_csv(self.raw / "transactions.csv", index=False)
        with self.assertRaisesRegex(ValueError, "empty"):
            run_forecasting_dataset(self.raw, self.output, self.summary)


if __name__ == "__main__":
    unittest.main()

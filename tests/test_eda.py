"""Tests for EDA calculations and output generation."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from consumer_intelligence.eda import run_eda


class EdaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.raw_dir = self.root / "raw"
        self.output_dir = self.root / "reports" / "eda"
        self.raw_dir.mkdir()
        self._write_csv(
            "customers.csv",
            ["customer_id", "age", "gender", "region", "signup_date"],
            [
                ["C1", 30, "Female", "West", "2024-01-01"],
                ["C2", 40, "Male", "East", "2024-01-01"],
            ],
        )
        self._write_csv(
            "products.csv",
            ["product_id", "category", "brand", "cost"],
            [["P1", "Apparel", "Brand A", 5], ["P2", "Beauty", "Brand B", 10]],
        )
        self._write_csv(
            "transactions.csv",
            [
                "transaction_id",
                "customer_id",
                "transaction_date",
                "product_id",
                "quantity",
                "price",
                "discount",
            ],
            [
                ["T1", "C1", "2024-01-10", "P1", 2, 10, 0.1],
                ["T2", "C1", "2024-02-10", "P2", 1, 20, 0.25],
                ["T3", "C2", "2024-02-12", "P1", 1, 10, 0],
            ],
        )
        self._write_csv(
            "marketing.csv",
            ["customer_id", "campaign", "campaign_date", "channel"],
            [
                ["C1", "Welcome Series", "2024-01-02", "Email"],
                ["C2", "Spring Sale", "2024-02-01", "SMS"],
            ],
        )
        self.raw_contents = {
            path.name: path.read_bytes() for path in self.raw_dir.iterdir()
        }

    def _write_csv(self, filename: str, fields: list[str], rows: list[list[object]]) -> None:
        with (self.raw_dir / filename).open(
            "w", newline="", encoding="utf-8"
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(fields)
            writer.writerows(rows)

    def test_metrics_and_outputs(self) -> None:
        summary = run_eda(self.raw_dir, self.output_dir)

        self.assertEqual(summary["customer_count"], 2)
        self.assertEqual(summary["product_count"], 2)
        self.assertEqual(summary["transaction_count"], 3)
        self.assertAlmostEqual(summary["total_revenue"], 43.0)
        self.assertEqual(summary["average_order_value"], 14.33)
        self.assertEqual(summary["total_quantity"], 4)
        self.assertEqual(
            summary["transaction_date_range"],
            {"minimum": "2024-01-10", "maximum": "2024-02-12"},
        )
        self.assertEqual(summary["top_revenue_category"], "Apparel")
        self.assertAlmostEqual(summary["month_one_cohort_retention_average"], 1.0)

        expected_outputs = {
            "eda_summary.json",
            "customer_demographics.csv",
            "customer_purchase_frequency.csv",
            "purchase_frequency_distribution.csv",
            "revenue_by_category.csv",
            "revenue_by_brand.csv",
            "monthly_trends.csv",
            "quantity_discount_summary.csv",
            "quantity_distribution.csv",
            "cohort_retention.csv",
            "marketing_campaigns.csv",
            "customers_by_region.png",
            "monthly_trends.png",
            "customer_activity.png",
            "revenue_by_category.png",
            "revenue_by_brand.png",
            "purchase_frequency.png",
            "discount_behavior.png",
            "quantity_distribution.png",
            "cohort_retention.png",
            "marketing_campaigns.png",
        }
        self.assertEqual({path.name for path in self.output_dir.iterdir()}, expected_outputs)
        saved_summary = json.loads(
            (self.output_dir / "eda_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(saved_summary["transaction_count"], 3)
        with (self.output_dir / "monthly_trends.csv").open(
            newline="", encoding="utf-8"
        ) as monthly_file:
            monthly = list(csv.DictReader(monthly_file))
        self.assertEqual([row["transaction_count"] for row in monthly], ["1", "2"])
        self.assertEqual(
            {path.name: path.read_bytes() for path in self.raw_dir.iterdir()},
            self.raw_contents,
        )


if __name__ == "__main__":
    unittest.main()

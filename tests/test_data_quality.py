"""Tests for dataset quality reporting."""

import csv
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from consumer_intelligence.data_generator import generate_datasets, write_datasets
from consumer_intelligence.data_quality import run_data_quality_report


class DataQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.data_dir = self.root / "raw"
        datasets = generate_datasets(
            seed=81,
            customer_count=120,
            product_count=18,
            transaction_count=500,
            start_date=date(2024, 1, 1),
            end_date=date(2025, 12, 31),
        )
        write_datasets(datasets, self.data_dir)
        self.report_path = self.root / "reports" / "data_quality_report.json"

    def test_valid_generated_data_passes_and_report_is_saved(self) -> None:
        report = run_data_quality_report(self.data_dir, self.report_path)

        self.assertEqual(report["overall_status"], "PASS")
        self.assertEqual(report["issue_count"], 0)
        self.assertEqual(report["datasets"]["transactions"]["row_count"], 500)
        self.assertTrue(self.report_path.is_file())
        self.assertEqual(json.loads(self.report_path.read_text(encoding="utf-8")), report)

    def test_invalid_transactions_and_customer_age_are_reported(self) -> None:
        transaction_path = self.data_dir / "transactions.csv"
        with transaction_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            fields = reader.fieldnames
            rows = list(reader)
        assert fields is not None
        rows[0].update(
            {
                "quantity": "0",
                "price": "-1",
                "discount": "1.5",
                "customer_id": "C999999",
                "product_id": "P9999",
                "transaction_date": "2023-12-31",
            }
        )
        with transaction_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        customers_path = self.data_dir / "customers.csv"
        with customers_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            customer_fields = reader.fieldnames
            customers = list(reader)
        assert customer_fields is not None
        customers[0]["age"] = "5"
        with customers_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=customer_fields)
            writer.writeheader()
            writer.writerows(customers)

        report = run_data_quality_report(self.data_dir, self.report_path)
        transactions = report["datasets"]["transactions"]
        customers_report = report["datasets"]["customers"]

        self.assertEqual(report["overall_status"], "FAIL")
        self.assertEqual(transactions["invalid_numeric_values"]["quantity"], 1)
        self.assertEqual(transactions["invalid_numeric_values"]["price"], 1)
        self.assertEqual(transactions["invalid_numeric_values"]["discount"], 1)
        self.assertEqual(transactions["referential_integrity"]["customer_id"], 1)
        self.assertEqual(transactions["referential_integrity"]["product_id"], 1)
        self.assertEqual(
            dict((issue["check"], issue["count"]) for issue in transactions["issues"])[
                "transaction_date_outside_expected_period"
            ],
            1,
        )
        self.assertEqual(customers_report["invalid_numeric_values"]["age"], 1)


if __name__ == "__main__":
    unittest.main()

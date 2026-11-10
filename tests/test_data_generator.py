"""Tests for synthetic dataset generation and data integrity."""

import csv
import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

from consumer_intelligence.data_generator import (
    DATASET_FIELDS,
    generate_datasets,
    write_datasets,
)


class DataGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.datasets = generate_datasets(
            seed=123,
            customer_count=300,
            product_count=30,
            transaction_count=2_000,
            start_date=date(2024, 1, 1),
            end_date=date(2025, 12, 31),
        )

    def test_expected_columns(self) -> None:
        for dataset_name, fields in DATASET_FIELDS.items():
            with self.subTest(dataset=dataset_name):
                self.assertEqual(
                    tuple(self.datasets[dataset_name][0].keys()), fields
                )

    def test_csv_outputs_have_expected_columns_and_row_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_paths = write_datasets(self.datasets, Path(temporary_directory))
            for name, path in output_paths.items():
                with self.subTest(dataset=name), path.open(
                    newline="", encoding="utf-8"
                ) as csv_file:
                    reader = csv.reader(csv_file)
                    self.assertEqual(tuple(next(reader)), DATASET_FIELDS[name])
                    self.assertEqual(sum(1 for _ in reader), len(self.datasets[name]))

    def test_positive_quantities_and_prices(self) -> None:
        for transaction in self.datasets["transactions"]:
            self.assertGreater(transaction["quantity"], 0)
            self.assertGreater(transaction["price"], Decimal("0"))

    def test_discounts_are_in_valid_range(self) -> None:
        for transaction in self.datasets["transactions"]:
            self.assertGreaterEqual(transaction["discount"], Decimal("0"))
            self.assertLessEqual(transaction["discount"], Decimal("1"))

    def test_foreign_keys_are_valid(self) -> None:
        customer_ids = {
            customer["customer_id"] for customer in self.datasets["customers"]
        }
        product_ids = {
            product["product_id"] for product in self.datasets["products"]
        }
        for transaction in self.datasets["transactions"]:
            self.assertIn(transaction["customer_id"], customer_ids)
            self.assertIn(transaction["product_id"], product_ids)
        for event in self.datasets["marketing"]:
            self.assertIn(event["customer_id"], customer_ids)

    def test_same_seed_is_reproducible(self) -> None:
        repeated = generate_datasets(
            seed=123,
            customer_count=300,
            product_count=30,
            transaction_count=2_000,
            start_date=date(2024, 1, 1),
            end_date=date(2025, 12, 31),
        )
        self.assertEqual(self.datasets, repeated)


if __name__ == "__main__":
    unittest.main()

"""Tests for point-in-time customer feature engineering."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from consumer_intelligence.customer_features import (
    FEATURE_COLUMNS,
    run_customer_feature_engineering,
)


class CustomerFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.raw_dir = self.root / "raw"
        self.raw_dir.mkdir()
        self.output_path = self.root / "processed" / "customer_features.csv"
        self.metadata_path = self.root / "reports" / "metadata.json"
        self._write_csv(
            "customers.csv",
            [
                {
                    "customer_id": "C1",
                    "age": 31,
                    "gender": "Female",
                    "region": "West",
                    "signup_date": "2024-01-01",
                },
                {
                    "customer_id": "C2",
                    "age": 42,
                    "gender": "Male",
                    "region": "South",
                    "signup_date": "2024-01-01",
                },
                {
                    "customer_id": "C3",
                    "age": 27,
                    "gender": "Female",
                    "region": "Central",
                    "signup_date": "2024-01-01",
                },
                {
                    "customer_id": "C4",
                    "age": 35,
                    "gender": "Male",
                    "region": "West",
                    "signup_date": "2024-03-01",
                },
            ],
        )
        self._write_csv(
            "products.csv",
            [
                {"product_id": "P1", "category": "Apparel", "brand": "A", "cost": 5},
                {"product_id": "P2", "category": "Beauty", "brand": "B", "cost": 7},
            ],
        )
        self._write_csv(
            "transactions.csv",
            [
                {
                    "transaction_id": "T1",
                    "customer_id": "C1",
                    "transaction_date": "2024-01-10",
                    "product_id": "P1",
                    "quantity": 2,
                    "price": 10,
                    "discount": 0.1,
                },
                {
                    "transaction_id": "T2",
                    "customer_id": "C1",
                    "transaction_date": "2024-02-10",
                    "product_id": "P2",
                    "quantity": 1,
                    "price": 20,
                    "discount": 0.25,
                },
                {
                    "transaction_id": "T3",
                    "customer_id": "C2",
                    "transaction_date": "2024-02-01",
                    "product_id": "P1",
                    "quantity": 3,
                    "price": 5,
                    "discount": 0,
                },
                {
                    "transaction_id": "T4",
                    "customer_id": "C1",
                    "transaction_date": "2024-03-01",
                    "product_id": "P1",
                    "quantity": 100,
                    "price": 100,
                    "discount": 0,
                },
                {
                    "transaction_id": "T5",
                    "customer_id": "C4",
                    "transaction_date": "2024-04-01",
                    "product_id": "P1",
                    "quantity": 1,
                    "price": 100,
                    "discount": 0,
                },
            ],
        )
        self._write_csv(
            "marketing.csv",
            [
                {
                    "customer_id": "C1",
                    "campaign": "Welcome",
                    "campaign_date": "2024-01-05",
                    "channel": "Email",
                },
                {
                    "customer_id": "C1",
                    "campaign": "Sale",
                    "campaign_date": "2024-02-05",
                    "channel": "SMS",
                },
                {
                    "customer_id": "C1",
                    "campaign": "Future",
                    "campaign_date": "2024-03-01",
                    "channel": "Email",
                },
            ],
        )
        self.raw_contents = {
            path.name: path.read_bytes() for path in self.raw_dir.iterdir()
        }

    def _write_csv(self, filename: str, rows: list[dict[str, object]]) -> None:
        pd.DataFrame(rows).to_csv(self.raw_dir / filename, index=False)

    def test_one_row_per_customer_and_expected_calculations(self) -> None:
        features = run_customer_feature_engineering(
            self.raw_dir,
            self.output_path,
            self.metadata_path,
            as_of_date="2024-02-10",
        )
        result = features.set_index("customer_id")

        self.assertEqual(list(features.columns), FEATURE_COLUMNS)
        self.assertEqual(len(features), 3)
        self.assertEqual(result.index.nunique(), len(result))
        self.assertEqual(set(result.index), {"C1", "C2", "C3"})
        self.assertEqual(result.loc["C1", "total_orders"], 2)
        self.assertEqual(result.loc["C1", "total_quantity"], 3)
        self.assertEqual(result.loc["C1", "total_revenue"], 33.0)
        self.assertEqual(result.loc["C1", "average_order_value"], 16.5)
        self.assertEqual(result.loc["C1", "average_discount"], 0.175)
        self.assertEqual(result.loc["C1", "unique_products_purchased"], 2)
        self.assertEqual(result.loc["C1", "unique_categories_purchased"], 2)
        self.assertEqual(result.loc["C1", "average_days_between_purchases"], 31.0)
        self.assertEqual(result.loc["C1", "days_since_last_purchase"], 0)
        self.assertEqual(result.loc["C1", "number_marketing_interactions"], 2)
        self.assertEqual(result.loc["C1", "number_marketing_campaigns"], 2)
        self.assertEqual(result.loc["C1", "first_purchase_date"], pd.Timestamp("2024-01-10"))
        self.assertEqual(result.loc["C1", "last_purchase_date"], pd.Timestamp("2024-02-10"))
        self.assertEqual(result.loc["C3", "total_orders"], 0)

    def test_nonnegative_features_valid_dates_and_raw_immutability(self) -> None:
        features = run_customer_feature_engineering(
            self.raw_dir,
            self.output_path,
            self.metadata_path,
            as_of_date="2024-02-10",
        )
        numeric = features.select_dtypes(include="number")
        self.assertTrue((numeric.fillna(0) >= 0).all().all())
        self.assertTrue(pd.to_datetime(features["first_purchase_date"], errors="coerce").notna().sum() == 2)
        self.assertTrue(
            {
                path.name: path.read_bytes() for path in self.raw_dir.iterdir()
            }
            == self.raw_contents
        )

    def test_output_and_feature_metadata_are_saved(self) -> None:
        run_customer_feature_engineering(
            self.raw_dir,
            self.output_path,
            self.metadata_path,
            as_of_date="2024-02-10",
        )
        saved = pd.read_csv(self.output_path)
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(len(saved), 3)
        self.assertEqual(metadata["feature_count"], len(FEATURE_COLUMNS) - 1)
        self.assertEqual(metadata["as_of_date"], "2024-02-10")
        self.assertIn("recency_days", metadata["features"])


if __name__ == "__main__":
    unittest.main()

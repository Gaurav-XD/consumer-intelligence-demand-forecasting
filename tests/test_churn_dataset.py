"""Tests for time-based, customer-disjoint churn dataset construction."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from consumer_intelligence.churn_dataset import (
    MODEL_FEATURE_COLUMNS,
    _customer_split,
    run_churn_dataset,
)


class ChurnDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.features_path = self.root / "customer_features.csv"
        self.output = self.root / "churn.csv"
        self.metadata_path = self.root / "metadata.json"
        ids = [f"C{i:03}" for i in range(100)]
        train_ids = [customer_id for customer_id in ids if _customer_split(customer_id) == "train"]
        test_ids = [customer_id for customer_id in ids if _customer_split(customer_id) == "test"]
        self.train_active, self.train_churn = train_ids[:2]
        self.test_active, self.test_churn = test_ids[:2]
        self.ids = [self.train_active, self.train_churn, self.test_active, self.test_churn]
        self._write(
            "customers",
            [
                {
                    "customer_id": customer_id,
                    "age": 30,
                    "gender": "Female",
                    "region": "West",
                    "signup_date": "2023-12-01",
                }
                for customer_id in self.ids
            ],
        )
        self._write("products", [{"product_id": "P1", "category": "Home", "brand": "A", "cost": 5}])
        transactions = []

        def purchase(customer_id: str, day: str, price: int) -> None:
            transactions.append(
                {
                    "transaction_id": f"T{len(transactions)}",
                    "customer_id": customer_id,
                    "transaction_date": day,
                    "product_id": "P1",
                    "quantity": 1,
                    "price": price,
                    "discount": 0,
                }
            )

        for customer_id in self.ids:
            purchase(customer_id, "2024-01-01", 10)
        purchase(self.train_active, "2024-01-10", 20)
        purchase(self.train_active, "2024-01-11", 100)
        purchase(self.train_churn, "2024-01-21", 200)
        purchase(self.test_active, "2024-02-10", 30)
        purchase(self.test_active, "2024-02-20", 500)
        purchase(self.test_churn, "2024-02-21", 600)
        self._write("transactions", transactions)
        self._write(
            "marketing",
            [
                {"customer_id": self.train_active, "campaign": "a", "campaign_date": "2024-01-10", "channel": "Email"},
                {"customer_id": self.train_active, "campaign": "future", "campaign_date": "2024-01-11", "channel": "Email"},
            ],
        )
        pd.DataFrame(
            {"customer_id": self.ids, "total_revenue": [999999] * len(self.ids)}
        ).to_csv(self.features_path, index=False)
        self.input_bytes = {
            path: path.read_bytes()
            for path in [*self.raw.glob("*.csv"), self.features_path]
        }

    def _write(self, name: str, rows: list[dict[str, object]]) -> None:
        pd.DataFrame(rows).to_csv(self.raw / f"{name}.csv", index=False)

    def _run(self) -> tuple[pd.DataFrame, dict]:
        return run_churn_dataset(
            self.raw,
            self.features_path,
            self.output,
            self.metadata_path,
            train_snapshot="2024-01-10",
            test_snapshot="2024-02-10",
            observation_days=10,
        )

    def test_features_and_labels_respect_snapshot_boundaries(self) -> None:
        dataset, metadata = self._run()
        rows = dataset.set_index("customer_id")
        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset["customer_id"].nunique(), 4)
        self.assertEqual(set(dataset["churn"]), {0, 1})
        self.assertEqual(set(dataset.loc[dataset["split"] == "train", "customer_id"]), {self.train_active, self.train_churn})
        self.assertEqual(set(dataset.loc[dataset["split"] == "test", "customer_id"]), {self.test_active, self.test_churn})
        self.assertEqual(rows.loc[self.train_active, "total_orders"], 2)
        self.assertEqual(rows.loc[self.train_active, "total_revenue"], 30)
        self.assertEqual(rows.loc[self.train_active, "number_marketing_interactions"], 1)
        self.assertEqual(rows.loc[self.train_active, "churn"], 0)
        self.assertEqual(rows.loc[self.train_churn, "churn"], 1)
        self.assertEqual(rows.loc[self.test_active, "total_revenue"], 40)
        self.assertEqual(rows.loc[self.test_active, "churn"], 0)
        self.assertEqual(rows.loc[self.test_churn, "churn"], 1)
        self.assertEqual(metadata["feature_columns"], MODEL_FEATURE_COLUMNS)
        self.assertNotIn("customer_id", metadata["feature_columns"])
        self.assertNotIn("churn", metadata["feature_columns"])
        self.assertNotIn("snapshot_date", metadata["feature_columns"])

    def test_metadata_outputs_and_inputs_are_unchanged(self) -> None:
        dataset, metadata = self._run()
        self.assertEqual(json.loads(self.metadata_path.read_text(encoding="utf-8")), metadata)
        self.assertEqual(len(pd.read_csv(self.output)), len(dataset))
        self.assertEqual(metadata["target_distribution"], {"0": 2, "1": 2})
        train = metadata["periods"]["train"]
        test = metadata["periods"]["test"]
        self.assertLess(train["feature_snapshot_period"]["snapshot_date_inclusive"], train["churn_observation_window"]["start_inclusive"])
        self.assertLess(train["churn_observation_window"]["end_inclusive"], test["feature_snapshot_period"]["snapshot_date_inclusive"])
        self.assertLess(test["feature_snapshot_period"]["snapshot_date_inclusive"], test["churn_observation_window"]["start_inclusive"])
        self.assertEqual({path: path.read_bytes() for path in self.input_bytes}, self.input_bytes)

    def test_rejects_overlapping_or_incomplete_windows(self) -> None:
        with self.assertRaisesRegex(ValueError, "must end before"):
            run_churn_dataset(
                self.raw, self.features_path, self.output, self.metadata_path,
                train_snapshot="2024-01-10", test_snapshot="2024-01-20", observation_days=10,
            )
        with self.assertRaisesRegex(ValueError, "does not extend"):
            run_churn_dataset(
                self.raw, self.features_path, self.output, self.metadata_path,
                train_snapshot="2024-01-10", test_snapshot="2024-02-15", observation_days=10,
            )


if __name__ == "__main__":
    unittest.main()

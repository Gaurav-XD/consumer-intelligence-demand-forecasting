"""Spark customer feature parity and point-in-time tests."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pyspark.sql import functions as F

from consumer_intelligence.customer_features import FEATURE_COLUMNS, run_customer_feature_engineering
from consumer_intelligence.pyspark_customer_features import (
    AOV_TOLERANCE, FLOAT_TOLERANCE, REFERENCE_SCHEMA, _read_csv, create_spark_session,
    reconcile_features, run_pyspark_customer_features,
)


class SparkCustomerFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = create_spark_session("local[2]")
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.raw = root / "raw"
        self.raw.mkdir()
        self.reference = root / "pandas.csv"
        self.metadata = root / "pandas.json"
        self.output = root / "spark.csv"
        self.summary = root / "spark.json"
        self._write("customers", [
            {"customer_id": "C1", "age": 31, "gender": "F", "region": "West", "signup_date": "2024-01-01"},
            {"customer_id": "C2", "age": 20, "gender": "M", "region": "East", "signup_date": "2024-01-01"},
            {"customer_id": "C3", "age": 40, "gender": "F", "region": "West", "signup_date": "2024-02-11"},
            {"customer_id": "C4", "age": 33, "gender": "F", "region": "East", "signup_date": "2024-02-10"},
        ])
        self._write("products", [
            {"product_id": "P1", "category": "A", "brand": "B", "cost": 3},
            {"product_id": "P2", "category": "C", "brand": "D", "cost": 4},
        ])
        self._write("transactions", [
            {"transaction_id": "T1", "customer_id": "C1", "transaction_date": "2024-01-10", "product_id": "P1", "quantity": 2, "price": 10, "discount": 0.1},
            {"transaction_id": "T2", "customer_id": "C1", "transaction_date": "2024-02-10", "product_id": "P2", "quantity": 1, "price": 20, "discount": 0.25},
            {"transaction_id": "T3", "customer_id": "C1", "transaction_date": "2024-02-11", "product_id": "P1", "quantity": 100, "price": 100, "discount": 0},
            {"transaction_id": "T4", "customer_id": "C3", "transaction_date": "2024-02-11", "product_id": "P1", "quantity": 1, "price": 10, "discount": 0},
            {"transaction_id": "T5", "customer_id": "C2", "transaction_date": "2024-02-01", "product_id": "P1", "quantity": 3, "price": 5, "discount": 0},
        ])
        self._write("marketing", [
            {"customer_id": "C1", "campaign": "Welcome", "campaign_date": "2024-02-10", "channel": "Email"},
            {"customer_id": "C1", "campaign": "Future", "campaign_date": "2024-02-11", "channel": "SMS"},
        ])
        run_customer_feature_engineering(self.raw, self.reference, self.metadata, as_of_date="2024-02-10")

    def _write(self, name, rows):
        pd.DataFrame(rows).to_csv(self.raw / f"{name}.csv", index=False)

    def _run(self):
        return run_pyspark_customer_features(
            self.raw, self.reference, self.metadata, self.output, self.summary,
            as_of_date="2024-02-10", spark=self.spark,
        )

    def test_execution_parity_cutoff_and_determinism(self):
        inputs = list(self.raw.iterdir()) + [self.reference, self.metadata]
        original = {path: path.read_bytes() for path in inputs}
        report = self._run()
        result = pd.read_csv(self.output).set_index("customer_id")
        self.assertEqual(list(pd.read_csv(self.output).columns), FEATURE_COLUMNS)
        self.assertEqual(report["row_count"], 3)
        self.assertEqual(report["column_count"], len(FEATURE_COLUMNS))
        self.assertEqual(report["reconciliation"]["absolute_tolerance"], FLOAT_TOLERANCE)
        self.assertTrue(report["reconciliation"]["passed"])
        self.assertEqual(report["reconciliation"]["average_order_value_absolute_tolerance"], AOV_TOLERANCE)
        self.assertEqual(set(result.index), {"C1", "C2", "C4"})
        self.assertTrue(result.index.is_unique)
        self.assertEqual(result.loc["C1", "total_orders"], 2)
        self.assertEqual(result.loc["C1", "total_revenue"], 33)
        self.assertEqual(result.loc["C1", "number_marketing_interactions"], 1)
        self.assertEqual(result.loc["C2", "total_orders"], 1)
        self.assertTrue(pd.isna(result.loc["C2", "average_days_between_purchases"]))
        self.assertEqual(result.loc["C4", "total_orders"], 0)
        self.assertTrue(pd.isna(result.loc["C4", "days_since_last_purchase"]))
        self.assertTrue(pd.isna(result.loc["C4", "purchase_frequency"]))
        self.assertTrue((result.select_dtypes("number").fillna(0) >= 0).all().all())
        self.assertEqual(json.loads(self.summary.read_text(encoding="utf-8")), report)
        output = self.output.read_bytes()
        self.assertEqual(report, self._run())
        self.assertEqual(output, self.output.read_bytes())
        self.assertEqual(original, {path: path.read_bytes() for path in inputs})

    def test_reference_mismatch_and_snapshot_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "snapshot"):
            run_pyspark_customer_features(
                self.raw, self.reference, self.metadata, self.output, self.summary,
                as_of_date="2024-02-09", spark=self.spark,
            )
        reference = pd.read_csv(self.reference)
        reference.loc[0, "total_revenue"] += 1
        reference.to_csv(self.reference, index=False)
        with self.assertRaisesRegex(ValueError, "do not reconcile"):
            self._run()

    def test_rounding_tolerance_is_strict(self):
        baseline = _read_csv(self.spark, self.reference, REFERENCE_SCHEMA)
        for offset, expected in ((0.01, True), (0.02, False)):
            adjusted = baseline.withColumn(
                "average_order_value",
                F.when(F.col("customer_id") == "C1", F.col("average_order_value") + offset)
                .otherwise(F.col("average_order_value")),
            )
            result = reconcile_features(adjusted, baseline)
            self.assertEqual(result["passed"], expected)


if __name__ == "__main__":
    unittest.main()

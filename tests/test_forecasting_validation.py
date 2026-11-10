"""Tests for chronological forecasting splits and their saved metadata."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from consumer_intelligence.forecasting_features import (
    CALENDAR_FEATURES,
    HISTORY_FEATURES,
    build_forecasting_features,
)
from consumer_intelligence.forecasting_validation import (
    SPLIT_PERIODS,
    create_forecasting_splits,
    run_forecasting_validation,
)


class ForecastingValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.source = root / "features.csv"
        self.metadata_path = root / "metadata.json"
        self.summary_path = root / "reports" / "summary.json"
        raw = pd.DataFrame(
            [
                {
                    "product_id": product,
                    "month": str(month),
                    "category": "Home",
                    "brand": "A",
                    "quantity_sold": index + offset,
                }
                for product, offset in (("P1", 0), ("P2", 100))
                for index, month in enumerate(pd.period_range("2024-01", "2025-12", freq="M"))
            ]
        )
        self.features = build_forecasting_features(raw)
        self.features.iloc[::-1].to_csv(self.source, index=False)
        self.source_bytes = self.source.read_bytes()
        self.metadata = {
            "feature_columns": HISTORY_FEATURES + CALENDAR_FEATURES,
            "target_column": "quantity_sold",
        }
        self.metadata_path.write_text(json.dumps(self.metadata), encoding="utf-8")

    def test_strict_dates_products_history_and_target(self) -> None:
        splits, summary = run_forecasting_validation(
            self.source, self.metadata_path, self.summary_path
        )
        self.assertEqual(set(splits), {"train", "validation", "test"})
        self.assertLess(splits["train"]["month"].max(), splits["validation"]["month"].min())
        self.assertLess(splits["validation"]["month"].max(), splits["test"]["month"].min())
        self.assertEqual(
            [(splits[name]["month"].min(), splits[name]["month"].max())
             for name in ("train", "validation", "test")],
            [("2025-01", "2025-03"), ("2025-04", "2025-06"), ("2025-07", "2025-12")],
        )
        all_keys = set()
        expected = self.features.set_index(["product_id", "month"])["quantity_sold"]
        for name, split in splits.items():
            start, end = SPLIT_PERIODS[name]
            self.assertTrue(split["month"].between(start, end).all())
            self.assertEqual(set(split["product_id"]), {"P1", "P2"})
            self.assertFalse(split[HISTORY_FEATURES].isna().any().any())
            self.assertEqual(
                list(zip(split["product_id"], split["month"])),
                sorted(zip(split["product_id"], split["month"])),
            )
            for row in split.itertuples(index=False):
                key = (row.product_id, row.month)
                self.assertNotIn(key, all_keys)
                all_keys.add(key)
                self.assertEqual(row.quantity_sold, expected.loc[key])
        self.assertEqual([len(splits[name]) for name in ("train", "validation", "test")], [6, 6, 12])
        self.assertEqual(summary["rows_removed_for_unavailable_history"], 24)
        self.assertNotIn("product_id", summary["numeric_feature_columns"])
        self.assertNotIn("quantity_sold", summary["numeric_feature_columns"])
        self.assertEqual(json.loads(self.summary_path.read_text(encoding="utf-8")), summary)
        self.assertEqual(self.source.read_bytes(), self.source_bytes)

    def test_rejects_leakage_contract_missing_target_and_duplicate_keys(self) -> None:
        metadata = dict(self.metadata, feature_columns=self.metadata["feature_columns"] + ["quantity_sold"])
        with self.assertRaisesRegex(ValueError, "feature contract"):
            create_forecasting_splits(self.features, metadata)
        bad = self.features.copy()
        bad.loc[0, "quantity_sold"] = float("nan")
        with self.assertRaisesRegex(ValueError, "quantity_sold"):
            create_forecasting_splits(bad, self.metadata)
        with self.assertRaisesRegex(ValueError, "unique"):
            create_forecasting_splits(pd.concat([self.features, self.features.iloc[[0]]]), self.metadata)


if __name__ == "__main__":
    unittest.main()

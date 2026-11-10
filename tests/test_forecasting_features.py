"""Tests for strictly historical forecasting features."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from consumer_intelligence.forecasting_features import (
    HISTORY_FEATURES,
    OUTPUT_COLUMNS,
    build_forecasting_features,
    run_forecasting_features,
)


class ForecastingFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.source = root / "demand.csv"
        self.output = root / "features.csv"
        self.metadata = root / "metadata.json"
        rows = [
            {
                "product_id": product,
                "month": str(month),
                "category": "Home",
                "brand": "A",
                "quantity_sold": index + offset,
                "revenue": 999999,
            }
            for product, offset in (("P1", 0), ("P2", 100))
            for index, month in enumerate(pd.period_range("2024-01", periods=14, freq="M"))
        ]
        pd.DataFrame(rows[::-1]).to_csv(self.source, index=False)
        self.source_bytes = self.source.read_bytes()

    def test_exact_lags_rolling_and_persisted_outputs(self) -> None:
        result, metadata = run_forecasting_features(self.source, self.output, self.metadata)
        self.assertEqual(list(result.columns), OUTPUT_COLUMNS)
        self.assertEqual(len(result), 28)
        self.assertFalse(result.duplicated(["product_id", "month"]).any())
        self.assertEqual(
            list(zip(result["product_id"], result["month"])),
            sorted(zip(result["product_id"], result["month"])),
        )
        p1 = result[result["product_id"].eq("P1")].reset_index(drop=True)
        self.assertTrue(p1.loc[0, HISTORY_FEATURES].isna().all())
        self.assertTrue(pd.isna(p1.loc[2, "rolling_3_mean_quantity"]))
        self.assertEqual(p1.loc[3, "lag_1_quantity"], 2)
        self.assertEqual(p1.loc[3, "rolling_3_mean_quantity"], 1)
        self.assertEqual(p1.loc[3, "rolling_3_std_quantity"], 1)
        self.assertEqual(p1.loc[12, "lag_12_quantity"], 0)
        self.assertEqual(p1.loc[12, "rolling_12_mean_quantity"], 5.5)
        p2 = result[result["product_id"].eq("P2")].reset_index(drop=True)
        self.assertEqual(p2.loc[3, "lag_1_quantity"], 102)
        for product_frame in (p1, p2):
            targets = product_frame["quantity_sold"]
            for lag in (1, 2, 3, 6, 12):
                pd.testing.assert_series_equal(
                    product_frame[f"lag_{lag}_quantity"],
                    targets.shift(lag),
                    check_names=False,
                )
            for window in (3, 6, 12):
                pd.testing.assert_series_equal(
                    product_frame[f"rolling_{window}_mean_quantity"],
                    targets.shift(1).rolling(window, min_periods=window).mean(),
                    check_names=False,
                )
            for window in (3, 6):
                pd.testing.assert_series_equal(
                    product_frame[f"rolling_{window}_std_quantity"],
                    targets.shift(1).rolling(window, min_periods=window).std(ddof=1),
                    check_names=False,
                )
        source = pd.read_csv(self.source).sort_values(["product_id", "month"]).reset_index(drop=True)
        self.assertEqual(result["quantity_sold"].tolist(), source["quantity_sold"].tolist())
        self.assertEqual(result.loc[0, "month_of_year"], 1)
        self.assertEqual(result.loc[0, "quarter"], 1)
        self.assertEqual(result.loc[1, "trend_index"] - result.loc[0, "trend_index"], 1)
        self.assertEqual(metadata["feature_count"], 13)
        self.assertEqual(metadata["missing_history_counts"]["lag_12_quantity"], 24)
        self.assertEqual(json.loads(self.metadata.read_text(encoding="utf-8")), metadata)
        self.assertEqual(len(pd.read_csv(self.output)), 28)
        self.assertEqual(self.source.read_bytes(), self.source_bytes)

    def test_current_and_future_targets_cannot_change_past_features(self) -> None:
        source = pd.read_csv(self.source)
        before = build_forecasting_features(source)
        changed = source.copy()
        changed.loc[changed["month"].ge("2024-08"), "quantity_sold"] += 5000
        after = build_forecasting_features(changed)
        columns = ["product_id", "month", *HISTORY_FEATURES]
        assert_frame_equal(
            before.loc[before["month"].le("2024-08"), columns].reset_index(drop=True),
            after.loc[after["month"].le("2024-08"), columns].reset_index(drop=True),
        )
        self.assertNotEqual(
            before.loc[before["month"].eq("2024-09"), "lag_1_quantity"].iloc[0],
            after.loc[after["month"].eq("2024-09"), "lag_1_quantity"].iloc[0],
        )

    def test_invalid_calendar_and_duplicate_keys_are_rejected(self) -> None:
        source = pd.read_csv(self.source)
        with self.assertRaisesRegex(ValueError, "consecutive"):
            build_forecasting_features(source.drop(source.index[3]))
        with self.assertRaisesRegex(ValueError, "unique"):
            build_forecasting_features(pd.concat([source, source.iloc[[0]]]))


if __name__ == "__main__":
    unittest.main()

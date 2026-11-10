"""Tests for validation-selected chronological demand forecasting models."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from consumer_intelligence.forecasting_features import build_forecasting_features
from consumer_intelligence.forecasting_models import (
    NUMERIC_FEATURES,
    build_models,
    regression_metrics,
    run_forecasting_models,
)
from consumer_intelligence.forecasting_validation import create_forecasting_splits


class ForecastingModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.source = root / "features.csv"
        self.validation = root / "validation.json"
        self.results = root / "results.csv"
        self.summary = root / "summary.json"
        self.models = root / "models"
        rows = [
            {
                "product_id": f"P{product}",
                "month": str(month),
                "category": "Home",
                "brand": "A",
                "quantity_sold": 5 + product * 3 + (index % 12),
            }
            for product in range(4)
            for index, month in enumerate(pd.period_range("2024-01", "2025-12", freq="M"))
        ]
        features = build_forecasting_features(pd.DataFrame(rows))
        features.to_csv(self.source, index=False)
        _, report = create_forecasting_splits(
            features, {"feature_columns": NUMERIC_FEATURES, "target_column": "quantity_sold"}
        )
        self.validation.write_text(json.dumps(report), encoding="utf-8")

    def run_models(self):
        return run_forecasting_models(
            self.source, self.validation, self.results, self.summary, self.models
        )

    def test_validation_selection_artifact_and_metrics(self) -> None:
        results, report = self.run_models()
        self.assertEqual(set(results["model"]), set(build_models()))
        self.assertEqual(report["selected_model"], results.loc[results["mae"].idxmin(), "model"])
        self.assertEqual(report["feature_columns"], NUMERIC_FEATURES)
        self.assertFalse({"customer_id", "product_id", "month", "quantity_sold"} & set(report["feature_columns"]))
        for column in ("mae", "rmse", "mape", "wape"):
            self.assertTrue((results[column] >= 0).all())
            self.assertGreaterEqual(report["test_metrics"][column], 0)
        self.assertTrue(self.results.is_file())
        self.assertTrue(self.summary.is_file())
        self.assertTrue((self.models / "best_model.joblib").is_file())
        self.assertEqual(json.loads(self.summary.read_text(encoding="utf-8")), report)
        predictor = joblib.load(self.models / "best_model.joblib")
        features = pd.read_csv(self.source)
        sample = features.loc[features["month"].eq("2025-07"), NUMERIC_FEATURES]
        self.assertEqual(len(predictor.predict(sample)), len(sample))
        second, second_report = self.run_models()
        pd.testing.assert_frame_equal(results, second)
        self.assertEqual(report, second_report)

    def test_test_period_cannot_affect_fitting_or_selection(self) -> None:
        first, summary = self.run_models()
        saved_model = joblib.load(self.models / "best_model.joblib")
        frame = pd.read_csv(self.source)
        train = frame.loc[frame["month"].between("2025-01", "2025-03"), NUMERIC_FEATURES]
        baseline = saved_model.predict(train)
        test = frame["month"].ge("2025-07")
        frame.loc[test, "quantity_sold"] += 500
        frame.loc[test, NUMERIC_FEATURES] = 10000
        frame.to_csv(self.source, index=False)
        second, second_summary = self.run_models()
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(summary["selected_model"], second_summary["selected_model"])
        np.testing.assert_array_equal(
            baseline, joblib.load(self.models / "best_model.joblib").predict(train)
        )
        self.assertNotEqual(summary["test_metrics"]["mae"], second_summary["test_metrics"]["mae"])

    def test_validation_mae_decides_winner(self) -> None:
        values = iter([8.0, 3.0, 6.0, 5.0, 10.0])

        def fake_metrics(actual, predicted):
            return {"mae": next(values), "rmse": 1.0, "mape": 1.0, "wape": 1.0}

        with patch("consumer_intelligence.forecasting_models.regression_metrics", fake_metrics):
            results, summary = self.run_models()
        self.assertEqual(summary["selected_model"], "Linear Regression")
        self.assertEqual(results.loc[results["model"].eq("Linear Regression"), "mae"].iloc[0], 3.0)
        self.assertEqual(summary["test_metrics"]["mae"], 10.0)

    def test_zero_actual_metrics_and_leakage_contract(self) -> None:
        metrics = regression_metrics(pd.Series([0, 2]), np.array([4.0, 1.0]))
        self.assertEqual(metrics["mape"], 50)
        self.assertEqual(metrics["wape"], 250)
        metrics = regression_metrics(pd.Series([0, 0]), np.array([0.0, 1.0]))
        self.assertIsNone(metrics["mape"])
        self.assertIsNone(metrics["wape"])
        validation = json.loads(self.validation.read_text(encoding="utf-8"))
        validation["numeric_feature_columns"].append("quantity_sold")
        self.validation.write_text(json.dumps(validation), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "feature contract"):
            self.run_models()


if __name__ == "__main__":
    unittest.main()

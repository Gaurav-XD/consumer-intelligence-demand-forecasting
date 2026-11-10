"""Verify chronological demand tuning and validation backtesting."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from consumer_intelligence.forecasting_features import build_forecasting_features
from consumer_intelligence.forecasting_tuning import (
    NUMERIC_FEATURES, calendar_cv, run_forecasting_tuning,
)
from consumer_intelligence.forecasting_validation import create_forecasting_splits


class ForecastingTuningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.source = root / "features.csv"
        self.validation = root / "validation.json"
        self.results = root / "results.csv"
        self.summary = root / "summary.json"
        self.walk = root / "walk.csv"
        self.model = root / "best.joblib"
        raw = pd.DataFrame([
            {
                "product_id": f"P{product}",
                "month": str(month),
                "category": "Home",
                "brand": "A",
                "quantity_sold": 5 + product * 3 + index % 12,
            }
            for product in range(4)
            for index, month in enumerate(pd.period_range("2024-01", "2025-12", freq="M"))
        ])
        self.features = build_forecasting_features(raw)
        self.features.to_csv(self.source, index=False)
        _, report = create_forecasting_splits(
            self.features, {"feature_columns": NUMERIC_FEATURES, "target_column": "quantity_sold"}
        )
        self.validation.write_text(json.dumps(report), encoding="utf-8")

    def run_report(self):
        return run_forecasting_tuning(
            self.source, self.validation, self.results, self.summary, self.walk, self.model
        )

    def test_calendar_folds_keep_months_disjoint(self) -> None:
        months = pd.Series(["2025-01", "2025-02", "2025-01", "2025-03", "2025-02"])
        folds = calendar_cv(months)
        self.assertEqual(len(folds), 2)
        for train_indices, validation_indices in folds:
            self.assertLess(months.iloc[train_indices].max(), months.iloc[validation_indices].min())
            self.assertEqual(months.iloc[validation_indices].nunique(), 1)
        with self.assertRaisesRegex(ValueError, "consecutive"):
            calendar_cv(pd.Series(["2025-01", "2025-03", "2025-04"]))

    def test_tuning_is_train_only_and_results_reproducible(self) -> None:
        original_fit = GridSearchCV.fit
        pipeline_fit = Pipeline.fit
        observed = []
        fitted = []
        source_bytes = self.source.read_bytes()

        def capture(search, x, y, **kwargs):
            observed.append((x.copy(), y.copy(), search.cv))
            return original_fit(search, x, y, **kwargs)

        def capture_pipeline(model, x, y=None, **kwargs):
            fitted.append(x.copy())
            return pipeline_fit(model, x, y, **kwargs)

        with patch.object(GridSearchCV, "fit", capture), patch.object(Pipeline, "fit", capture_pipeline):
            results, summary, walk = self.run_report()
        self.assertEqual(source_bytes, self.source.read_bytes())
        self.assertEqual(len(observed), 2)
        for x, y, folds in observed:
            self.assertEqual(len(x), 12)
            self.assertEqual(list(x.columns), NUMERIC_FEATURES)
            self.assertFalse({"product_id", "month", "quantity_sold"} & set(x.columns))
            self.assertEqual(len(folds), 2)
            self.assertEqual(len(y), 12)
        self.assertEqual(set(results["model"]), {"Random Forest", "Gradient Boosting"})
        self.assertTrue(all(json.loads(value) for value in results["best_parameters"]))
        self.assertEqual(summary["selected_model"], results.loc[results["cv_mae"].idxmin(), "model"])
        self.assertEqual(list(walk["forecast_month"].unique()), ["2025-04", "2025-05", "2025-06"])
        self.assertTrue((walk["training_end_month"] < walk["forecast_month"]).all())
        self.assertEqual([len(x) for x in fitted[-3:]], [12, 16, 20])
        self.assertEqual([int(x["month_of_year"].max()) for x in fitted[-3:]], [3, 4, 5])
        self.assertTrue(all(x["month_of_year"].max() <= 3 for x in fitted[:-3]))
        self.assertEqual(len(walk), 12)
        for metrics in (summary["validation_metrics"], summary["walk_forward_metrics"]):
            for key in ("mae", "rmse", "mape", "wape"):
                self.assertGreaterEqual(metrics[key], 0)
        self.assertTrue(all(path.is_file() for path in (self.results, self.summary, self.walk, self.model)))
        self.assertEqual(json.loads(self.summary.read_text(encoding="utf-8")), summary)
        self.assertAlmostEqual(summary["walk_forward_metrics"]["mae"], walk["absolute_error"].mean())
        baseline = joblib.load(self.model).predict(observed[0][0])
        frame = pd.read_csv(self.source)
        test = frame["month"].ge("2025-07")
        frame.loc[test, "quantity_sold"] += 5000
        frame.loc[test, NUMERIC_FEATURES] = 10000
        frame.to_csv(self.source, index=False)
        second, second_summary, second_walk = self.run_report()
        pd.testing.assert_frame_equal(results, second)
        pd.testing.assert_frame_equal(walk, second_walk)
        self.assertEqual(summary, second_summary)
        np.testing.assert_array_equal(baseline, joblib.load(self.model).predict(observed[0][0]))
        validation = frame["month"].between("2025-04", "2025-06")
        frame.loc[validation, "quantity_sold"] += 500
        frame.to_csv(self.source, index=False)
        third, third_summary, _ = self.run_report()
        pd.testing.assert_frame_equal(results, third)
        self.assertEqual(summary["selected_model"], third_summary["selected_model"])
        self.assertEqual(summary["best_parameters"], third_summary["best_parameters"])
        np.testing.assert_array_equal(baseline, joblib.load(self.model).predict(observed[0][0]))

    def test_validation_contract_rejects_leakage(self) -> None:
        metadata = json.loads(self.validation.read_text(encoding="utf-8"))
        metadata["numeric_feature_columns"].append("quantity_sold")
        self.validation.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "feature contract"):
            self.run_report()


if __name__ == "__main__":
    unittest.main()

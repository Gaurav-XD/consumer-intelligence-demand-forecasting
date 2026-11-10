"""Tests for training-only hyperparameter search and saved artifacts."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from consumer_intelligence.churn_dataset import MODEL_FEATURE_COLUMNS
from consumer_intelligence.churn_tuning import PARAMETER_SPACES, run_churn_tuning


class ChurnTuningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.dataset = root / "churn.csv"
        self.metadata = root / "metadata.json"
        self.results = root / "results.csv"
        self.summary = root / "summary.json"
        self.model = root / "model.joblib"
        rows = []
        for index in range(60):
            row = {
                name: (index + offset) % 17
                for offset, name in enumerate(MODEL_FEATURE_COLUMNS)
            }
            row["average_discount"] = None if index % 4 == 0 else 0.1
            row.update(
                customer_id=f"C{index:03d}",
                split="train" if index < 48 else "test",
                snapshot_date="2025-03-31" if index < 48 else "2025-09-30",
                churn=index % 2,
            )
            rows.append(row)
        pd.DataFrame(rows).to_csv(self.dataset, index=False)
        self.metadata.write_text(
            json.dumps(
                {
                    "feature_columns": MODEL_FEATURE_COLUMNS,
                    "target_column": "churn",
                    "periods": {
                        "train": {
                            "feature_snapshot_period": {
                                "snapshot_date_inclusive": "2025-03-31"
                            },
                            "row_count": 48,
                        },
                        "test": {
                            "feature_snapshot_period": {
                                "snapshot_date_inclusive": "2025-09-30"
                            },
                            "row_count": 12,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

    def run_tuning(self) -> tuple[pd.DataFrame, dict]:
        return run_churn_tuning(
            self.dataset, self.metadata, self.results, self.summary, self.model, n_iter=2
        )

    def test_training_is_isolated_reproducible_and_outputs_valid(self) -> None:
        original_fit = RandomizedSearchCV.fit
        fit_inputs = []

        def capture_fit(search, x, y, **kwargs):
            fit_inputs.append((x.copy(), y.copy()))
            return original_fit(search, x, y, **kwargs)

        with patch.object(RandomizedSearchCV, "fit", capture_fit):
            first, summary = self.run_tuning()
        self.assertEqual(len(fit_inputs), 3)
        for x, y in fit_inputs:
            self.assertEqual(len(x), 48)
            self.assertEqual(len(y), 48)
            self.assertTrue(all(index < 48 for index in x.index))
            self.assertEqual(list(x.columns), MODEL_FEATURE_COLUMNS)
            self.assertFalse({"customer_id", "snapshot_date", "split", "churn"} & set(x.columns))
        self.assertEqual(set(first["model"]), set(PARAMETER_SPACES))
        self.assertTrue(all(json.loads(value) for value in first["best_parameters"]))
        self.assertEqual(set(summary["models"]), set(PARAMETER_SPACES))
        self.assertEqual(summary["selected_model"], first.loc[
            first["mean_cv_average_precision"].idxmax(), "model"
        ])
        for name in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"):
            self.assertGreaterEqual(summary["test_metrics"][name], 0)
            self.assertLessEqual(summary["test_metrics"][name], 1)
        self.assertEqual(sum(map(sum, summary["test_metrics"]["confusion_matrix"])), 12)
        self.assertTrue(self.model.is_file())
        self.assertTrue(self.results.is_file())
        self.assertEqual(json.loads(self.summary.read_text(encoding="utf-8")), summary)
        before_predictions = joblib.load(self.model).predict_proba(fit_inputs[0][0])

        data = pd.read_csv(self.dataset)
        is_test = data["split"].eq("test")
        data.loc[is_test, "churn"] = 1 - data.loc[is_test, "churn"]
        data.loc[is_test, MODEL_FEATURE_COLUMNS] = 10000
        data.to_csv(self.dataset, index=False)
        second, second_summary = self.run_tuning()
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(summary["selected_model"], second_summary["selected_model"])
        self.assertEqual(summary["best_parameters"], second_summary["best_parameters"])
        pd.testing.assert_frame_equal(
            pd.DataFrame(before_predictions),
            pd.DataFrame(joblib.load(self.model).predict_proba(fit_inputs[0][0])),
        )

    def test_rejects_leakage_columns(self) -> None:
        metadata = json.loads(self.metadata.read_text(encoding="utf-8"))
        metadata["feature_columns"].append("churn")
        self.metadata.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "feature contract"):
            self.run_tuning()


if __name__ == "__main__":
    unittest.main()

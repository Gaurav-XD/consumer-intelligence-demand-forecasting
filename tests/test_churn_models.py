"""Tests for churn training, split isolation, and model artifacts."""

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from consumer_intelligence.churn_dataset import MODEL_FEATURE_COLUMNS
from consumer_intelligence.churn_models import (
    build_pipelines,
    load_churn_data,
    run_churn_models,
)


class ChurnModelsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.dataset = root / "churn.csv"
        self.metadata = root / "metadata.json"
        self.results = root / "results.csv"
        self.summary = root / "summary.json"
        self.models = root / "models"
        rows = []
        for index in range(60):
            train = index < 48
            row = {
                name: (index + offset) % 17
                for offset, name in enumerate(MODEL_FEATURE_COLUMNS)
            }
            row["average_discount"] = None if index % 4 == 0 else 0.1
            row.update(
                customer_id=f"C{index:03d}",
                split="train" if train else "test",
                snapshot_date="2025-03-31" if train else "2025-09-30",
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

    def run_models(self) -> tuple[pd.DataFrame, dict]:
        return run_churn_models(
            self.dataset, self.metadata, self.results, self.summary, self.models
        )

    def test_feature_target_and_split_separation(self) -> None:
        x_train, y_train, x_test, y_test, features = load_churn_data(
            self.dataset, self.metadata
        )
        self.assertEqual(len(x_train), 48)
        self.assertEqual(len(x_test), 12)
        self.assertEqual(len(y_train), 48)
        self.assertEqual(len(y_test), 12)
        self.assertEqual(list(x_train.columns), MODEL_FEATURE_COLUMNS)
        self.assertEqual(features, MODEL_FEATURE_COLUMNS)
        self.assertTrue(set(x_train.index).isdisjoint(x_test.index))
        self.assertFalse({"customer_id", "snapshot_date", "split", "churn"} & set(features))
        pipelines = build_pipelines()
        self.assertEqual(pipelines["Logistic Regression"].steps[1][0], "scaler")
        for pipeline in pipelines.values():
            self.assertEqual(pipeline.steps[0][0], "imputer")

    def test_results_artifacts_and_reproducibility(self) -> None:
        first, summary = self.run_models()
        saved = json.loads(self.summary.read_text(encoding="utf-8"))
        self.assertEqual(saved, summary)
        self.assertEqual(set(first["model"]), set(build_pipelines()))
        for column in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"):
            self.assertTrue(first[column].between(0, 1).all(), column)
        self.assertTrue(
            (first[["true_negative", "false_positive", "false_negative", "true_positive"]]
             .sum(axis=1) == 12).all()
        )
        self.assertEqual(
            summary["best_model"],
            first.loc[first["train_cv_average_precision"].idxmax(), "model"],
        )
        self.assertTrue(Path(summary["best_model_path"]).is_file())
        self.assertTrue(all(Path(path).is_file() for path in summary["model_artifacts"].values()))
        x_train, _, x_test, _, _ = load_churn_data(self.dataset, self.metadata)
        best = joblib.load(summary["best_model_path"])
        self.assertEqual(len(best.predict(x_test)), len(x_test))
        self.assertEqual(best.named_steps["imputer"].n_features_in_, x_train.shape[1])
        second, second_summary = self.run_models()
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(summary, second_summary)

    def test_rejects_leakage_and_bad_split(self) -> None:
        meta = json.loads(self.metadata.read_text(encoding="utf-8"))
        meta["feature_columns"].append("churn")
        self.metadata.write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "feature contract"):
            load_churn_data(self.dataset, self.metadata)
        meta["feature_columns"].remove("churn")
        self.metadata.write_text(json.dumps(meta), encoding="utf-8")
        frame = pd.read_csv(self.dataset)
        frame.loc[0, "split"] = "test"
        frame.to_csv(self.dataset, index=False)
        with self.assertRaisesRegex(ValueError, "snapshot dates|row count"):
            load_churn_data(self.dataset, self.metadata)

    def test_test_data_does_not_influence_selection_or_fitting(self) -> None:
        first, summary = self.run_models()
        x_train, _, x_test, _, _ = load_churn_data(self.dataset, self.metadata)
        first_best = joblib.load(summary["best_model_path"])
        baseline_predictions = first_best.predict_proba(x_train)
        frame = pd.read_csv(self.dataset)
        test_rows = frame["split"].eq("test")
        frame.loc[test_rows, "churn"] = 1 - frame.loc[test_rows, "churn"]
        frame.loc[test_rows, MODEL_FEATURE_COLUMNS] = 10000
        frame.to_csv(self.dataset, index=False)
        second, second_summary = self.run_models()
        pd.testing.assert_series_equal(
            first["train_cv_average_precision"],
            second["train_cv_average_precision"],
        )
        self.assertEqual(summary["best_model"], second_summary["best_model"])
        second_best = joblib.load(second_summary["best_model_path"])
        pd.testing.assert_frame_equal(
            pd.DataFrame(baseline_predictions),
            pd.DataFrame(second_best.predict_proba(x_train)),
        )
        self.assertEqual(len(x_test), 12)


if __name__ == "__main__":
    unittest.main()

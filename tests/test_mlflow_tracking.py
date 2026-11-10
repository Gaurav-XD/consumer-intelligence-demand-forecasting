"""Verify local tracking, artifact provenance and deterministic definitions."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import joblib
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from consumer_intelligence.mlflow_tracking import build_experiment_definitions, run_mlflow_tracking


class MlflowTrackingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for directory in ("reports", "data/processed", "models/churn", "models/forecasting"):
            (self.root / directory).mkdir(parents=True)
        (self.root / "data/processed/churn_dataset.csv").write_text("customer_id,churn\nC1,0\n", encoding="utf-8")
        (self.root / "data/processed/forecasting_features.csv").write_text("product_id,month\nP1,2025-01\n", encoding="utf-8")
        for filename, model in (
            ("logistic_regression.joblib", LogisticRegression()),
            ("random_forest.joblib", RandomForestClassifier(random_state=42)),
            ("gradient_boosting.joblib", GradientBoostingClassifier()),
            ("tuned_best_model.joblib", RandomForestClassifier(n_estimators=7, random_state=42)),
        ):
            joblib.dump(Pipeline([("imputer", SimpleImputer()), ("model", model)]), self.root / "models/churn" / filename)
        joblib.dump(Pipeline([("model", RandomForestRegressor(random_state=42))]), self.root / "models/forecasting/best_model.joblib")
        joblib.dump(Pipeline([("model", RandomForestRegressor(min_samples_leaf=5, random_state=42))]), self.root / "models/forecasting/tuned_best_model.joblib")
        churn_metrics = {
            "train_cv_average_precision": 0.6, "roc_auc": 0.7, "pr_auc": 0.6,
            "accuracy": 0.7, "precision": 0.6, "recall": 0.5, "f1": 0.55,
            "true_negative": 4, "false_positive": 1, "false_negative": 2, "true_positive": 3,
        }
        self.write("churn_model_summary.json", {
            "feature_columns": ["a"], "selection_criterion": "training CV",
            "test_metrics": {name: churn_metrics for name in ("Logistic Regression", "Random Forest", "Gradient Boosting")},
        })
        self.write("churn_tuning_summary.json", {
            "selected_model": "Random Forest", "feature_columns": ["a"],
            "best_parameters": {"model__n_estimators": 7},
            "mean_cv_average_precision": 0.7, "std_cv_average_precision": 0.1,
            "test_metrics": {"roc_auc": 0.8, "pr_auc": 0.6, "precision": 0.7,
                             "recall": 0.5, "f1": 0.58, "confusion_matrix": [[4, 1], [2, 3]]},
            "search": {"cross_validation": "training folds"}, "selection_criterion": "training CV",
        })
        self.write("churn_dataset_metadata.json", {"feature_columns": ["a"], "periods": {"train": {"row_count": 8}, "test": {"row_count": 2}}})
        models = ("Seasonal Naive", "Linear Regression", "Random Forest", "Gradient Boosting")
        self.write("forecasting_model_summary.json", {
            "selected_model": "Random Forest", "feature_columns": ["lag_1"],
            "selection_criterion": "validation MAE", "evaluation_methodology": "rolling origin",
            "model_parameters": {name: {} for name in models},
            "validation_metrics": {name: {"mae": 1.0, "rmse": 2.0, "mape": 3.0, "wape": 4.0} for name in models},
            "test_metrics": {"mae": 2.0, "rmse": 3.0, "mape": 4.0, "wape": 5.0},
        })
        self.write("forecasting_tuning_summary.json", {
            "selected_model": "Random Forest", "feature_columns": ["lag_1"],
            "best_parameters": {"model__min_samples_leaf": 5}, "cv_mae": 0.9,
            "models": {"Random Forest": {"cv_mae_std": 0.1}},
            "validation_metrics": {"mae": 1.0, "rmse": 2.0, "mape": 3.0, "wape": 4.0},
            "walk_forward_metrics": {"mae": 0.9, "rmse": 1.9, "mape": 2.9, "wape": 3.9},
            "methodology": "training time CV", "selection_criterion": "training CV MAE",
        })
        self.write("forecasting_validation_summary.json", {"numeric_feature_columns": ["lag_1"], "splits": {"train": {"row_count": 3}, "validation": {"row_count": 2}, "test": {"row_count": 1}}})

    def write(self, name, data):
        (self.root / "reports" / name).write_text(json.dumps(data), encoding="utf-8")

    def test_runs_metadata_artifacts_and_idempotency(self):
        inputs = list((self.root / "data/processed").iterdir()) + list((self.root / "reports").iterdir()) + list((self.root / "models/churn").iterdir()) + list((self.root / "models/forecasting").iterdir())
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
        definitions = build_experiment_definitions(self.root)
        self.assertEqual(definitions, build_experiment_definitions(self.root))
        summary = run_mlflow_tracking(self.root)
        self.assertEqual(summary["tracking_uri"], (self.root / "mlruns").resolve().as_uri())
        self.assertEqual(summary["run_count"], 9)
        self.assertEqual(len(summary["experiments"]), 2)
        self.assertEqual(sum(run["model_artifact_status"] == "available" for run in summary["runs"]), 6)
        client = MlflowClient(tracking_uri=summary["tracking_uri"])
        for recorded, definition in zip(summary["runs"], definitions):
            run = client.get_run(recorded["run_id"])
            self.assertEqual(run.info.status, "FINISHED")
            self.assertEqual(run.data.tags["validation_methodology"], definition["validation_methodology"])
            self.assertEqual(run.data.tags["selection_criterion"], definition["selection_criterion"])
            self.assertEqual(json.loads(run.data.tags["split_info"]), definition["split_info"])
            self.assertEqual(run.data.params["dataset"], definition["dataset"])
            self.assertEqual(run.data.params["model_name"], definition["model_name"])
            if definition["domain"] == "churn":
                self.assertEqual(json.loads(run.data.params["imputer__missing_values"]), "nan")
            self.assertEqual(run.data.metrics, definition["metrics"])
            self.assertTrue(client.list_artifacts(run.info.run_id, "source_reports"))
            if definition["model_path"]:
                files = client.list_artifacts(run.info.run_id, "model")
                self.assertEqual(len(files), 1)
                downloaded = client.download_artifacts(run.info.run_id, files[0].path)
                self.assertEqual(Path(downloaded).read_bytes(), (self.root / definition["model_path"]).read_bytes())
            else:
                self.assertEqual(client.list_artifacts(run.info.run_id, "model"), [])
                self.assertTrue(run.data.tags["model_artifact_reason"])
        self.assertEqual(summary, run_mlflow_tracking(self.root))
        self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs})
        self.assertEqual(summary, json.loads((self.root / "reports/mlflow_tracking_summary.json").read_text(encoding="utf-8")))
        tuned = next(r for r in summary["runs"] if r["experiment"].endswith("Forecasting") and r["variant"] == "tuned")
        self.assertNotIn("test_mae", client.get_run(tuned["run_id"]).data.metrics)

    def test_rejects_bad_contract_and_unavailable_required_model(self):
        info = json.loads((self.root / "reports/forecasting_tuning_summary.json").read_text(encoding="utf-8"))
        info["selected_model"] = "Gradient Boosting"
        self.write("forecasting_tuning_summary.json", info)
        with self.assertRaisesRegex(ValueError, "Random Forest"):
            build_experiment_definitions(self.root)
        with self.assertRaisesRegex(ValueError, "inside the project"):
            run_mlflow_tracking(self.root, self.root.parent)

    def test_existing_artifact_corruption_is_detected(self):
        summary = run_mlflow_tracking(self.root)
        client = MlflowClient(tracking_uri=summary["tracking_uri"])
        run = summary["runs"][0]
        artifact = client.list_artifacts(run["run_id"], "model")[0]
        path = Path(client.download_artifacts(run["run_id"], artifact.path))
        path.write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "modified"):
            run_mlflow_tracking(self.root)


if __name__ == "__main__":
    unittest.main()

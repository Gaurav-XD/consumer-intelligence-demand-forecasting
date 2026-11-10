"""Inference API contracts and lifecycle tests."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
from fastapi.testclient import TestClient

from consumer_intelligence.api import MODEL_SPECS, app, load_models
from consumer_intelligence.config import PROJECT_ROOT

CHURN = {
    "customer_tenure_days": 600, "total_orders": 5, "total_quantity": 8,
    "total_revenue": 500.0, "average_order_value": 100.0, "average_discount": 0.1,
    "unique_products_purchased": 3, "unique_categories_purchased": 2,
    "purchase_frequency": 3.0, "average_days_between_purchases": 40.0,
    "days_since_last_purchase": 90, "customer_lifetime_days": 160,
    "number_marketing_interactions": 2, "number_marketing_campaigns": 2,
}
FORECAST = {
    "lag_1_quantity": 30, "lag_2_quantity": 31, "lag_3_quantity": 32,
    "lag_6_quantity": 30, "lag_12_quantity": 28,
    "rolling_3_mean_quantity": 31.0, "rolling_6_mean_quantity": 30.0,
    "rolling_12_mean_quantity": 29.0, "rolling_3_std_quantity": 1.0,
    "rolling_6_std_quantity": 2.0, "month_of_year": 4, "quarter": 2,
    "trend_index": 2025 * 12 + 4 - 1,
}


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = [PROJECT_ROOT / spec[0] for spec in MODEL_SPECS.values()]
        cls.before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in cls.files]
        cls.patcher = patch("consumer_intelligence.api.joblib.load", wraps=joblib.load)
        cls.loader = cls.patcher.start()
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)
        cls.patcher.stop()
        after = [hashlib.sha256(path.read_bytes()).hexdigest() for path in cls.files]
        if cls.before != after:
            raise AssertionError("Model artifacts changed during API tests")

    def test_health_models_and_load_once(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        response = self.client.get("/models")
        self.assertEqual(response.status_code, 200)
        models = response.json()["models"]
        self.assertEqual(set(models), {"churn", "forecast"})
        for model in models.values():
            self.assertEqual(len(model["artifact_sha256"]), 64)
            self.assertEqual(model["version"], "sha256:" + model["artifact_sha256"])
            self.assertEqual(model["trained_scikit_learn_version"], "1.4.2")
            self.assertTrue(model["feature_columns"])
        self.assertEqual(self.loader.call_count, 2)

    def test_predictions_deterministic_and_metadata(self):
        for route, data, field in (
            ("/predict/churn", CHURN, "churn_probability"),
            ("/predict/forecast", FORECAST, "predicted_quantity"),
        ):
            first = self.client.post(route, json=data)
            self.assertEqual(first.status_code, 200, first.text)
            body = first.json()
            self.assertIn(field, body)
            self.assertEqual(body["model"], self.client.get("/models").json()["models"][route.rsplit("/", 1)[1]])
            self.assertEqual(body, self.client.post(route, json=data).json())
        body = self.client.post("/predict/churn", json=CHURN).json()
        self.assertTrue(0 <= body["churn_probability"] <= 1)
        self.assertIn(body["predicted_class"], [0, 1])
        self.assertEqual(self.loader.call_count, 2)

    def test_missing_and_invalid_rejected(self):
        for route, payload in (
            ("/predict/churn", {key: value for key, value in CHURN.items() if key != "average_discount"}),
            ("/predict/churn", {**CHURN, "total_orders": -1}),
            ("/predict/churn", {**CHURN, "average_discount": 1.1}),
            ("/predict/churn", {**CHURN, "total_revenue": "500"}),
            ("/predict/churn", {**CHURN, "unexpected": 3}),
            ("/predict/forecast", {key: value for key, value in FORECAST.items() if key != "lag_1_quantity"}),
            ("/predict/forecast", {**FORECAST, "month_of_year": 13}),
            ("/predict/forecast", {**FORECAST, "quarter": 1}),
            ("/predict/forecast", {**FORECAST, "trend_index": FORECAST["trend_index"] + 1}),
            ("/predict/forecast", {**FORECAST, "lag_1_quantity": None}),
        ):
            response = self.client.post(route, json=payload)
            self.assertEqual(response.status_code, 422, response.text)
            self.assertIn("detail", response.json())
        nullable = {**CHURN, "average_order_value": None, "average_discount": None,
                    "average_days_between_purchases": None, "days_since_last_purchase": None,
                    "customer_lifetime_days": None, "purchase_frequency": None}
        self.assertEqual(self.client.post("/predict/churn", json=nullable).status_code, 200)

    def test_inference_error_is_structured(self):
        model = app.state.models["forecast"].pipeline
        with patch.object(model, "predict", side_effect=ValueError("broken model")), self.assertLogs("consumer_intelligence.api", level="ERROR"):
            response = self.client.post("/predict/forecast", json=FORECAST)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"error": "inference_error", "message": "Model inference failed."})

    def test_missing_artifact_fails_explicitly(self):
        with self.assertRaises(FileNotFoundError):
            load_models(Path("nonexistent_model_root"))

    def test_incompatible_report_fails_startup(self):
        source = PROJECT_ROOT / MODEL_SPECS["churn"][1]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / MODEL_SPECS["churn"][1]
            path.parent.mkdir(parents=True)
            report = json.loads(source.read_text(encoding="utf-8"))
            report["library_versions"]["scikit_learn"] = "0.0.0"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "requires scikit-learn"):
                load_models(root)

    def test_direct_estimator_results_match_api(self):
        import pandas as pd

        for kind, payload in (("churn", CHURN), ("forecast", FORECAST)):
            loaded = app.state.models[kind]
            frame = pd.DataFrame([payload], columns=loaded.metadata.feature_columns, dtype=float)
            response = self.client.post(f"/predict/{kind}", json=payload).json()
            if kind == "churn":
                self.assertAlmostEqual(response["churn_probability"], float(loaded.pipeline.predict_proba(frame)[0, 1]))
                self.assertEqual(response["predicted_class"], int(loaded.pipeline.predict(frame)[0]))
            else:
                self.assertAlmostEqual(response["predicted_quantity"], float(loaded.pipeline.predict(frame)[0]))


if __name__ == "__main__":
    unittest.main()

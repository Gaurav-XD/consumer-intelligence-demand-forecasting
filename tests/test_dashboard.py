"""Tests for dashboard loading, section rendering and metrics."""

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from consumer_intelligence.config import PROJECT_ROOT
from consumer_intelligence.dashboard import ARTIFACTS, SECTIONS, forecast_performance, load_artifacts


class DashboardTests(unittest.TestCase):
    def test_artifacts_load_and_sections_are_registered(self):
        data, issues = load_artifacts()
        self.assertFalse(issues, issues)
        self.assertEqual(set(data), set(ARTIFACTS))
        self.assertEqual(list(SECTIONS), [
            "Executive Overview", "Customer Analytics", "Churn Analytics",
            "Demand Forecasting", "Business Impact",
        ])
        self.assertTrue(all(callable(value) for value in SECTIONS.values()))

    def test_missing_empty_and_malformed_inputs_are_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "reports" / "eda").mkdir(parents=True)
            (root / "reports" / "eda" / "eda_summary.json").write_text("{invalid", encoding="utf-8")
            (root / "reports" / "churn_model_results.csv").write_text("", encoding="utf-8")
            with self.assertLogs("consumer_intelligence.dashboard", level="WARNING"):
                data, issues = load_artifacts(root)
            self.assertEqual(data, {})
            self.assertEqual(len(issues), len(ARTIFACTS))
            self.assertTrue(any("eda_summary.json" in issue for issue in issues))
            self.assertTrue(any("churn_model_results.csv" in issue for issue in issues))

    def test_product_metrics_use_row_level_errors(self):
        frame = pd.DataFrame([
            {"product_id": "P1", "actual_quantity": 10, "predicted_quantity": 0},
            {"product_id": "P1", "actual_quantity": 0, "predicted_quantity": 10},
            {"product_id": "P2", "actual_quantity": 0, "predicted_quantity": 2},
        ])
        result = forecast_performance(frame).set_index("product_id")
        self.assertEqual(result.loc["P1", "mae"], 10)
        self.assertEqual(result.loc["P1", "wape_percent"], 200)
        self.assertTrue(pd.isna(result.loc["P2", "wape_percent"]))
        self.assertTrue(forecast_performance(pd.DataFrame()).empty)

    def test_all_sections_render_without_modifying_inputs(self):
        files = [PROJECT_ROOT / spec[0] for spec in ARTIFACTS.values()]
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        script = str(PROJECT_ROOT / "src" / "consumer_intelligence" / "dashboard.py")
        with patch.dict(os.environ, {"CONSUMER_INTELLIGENCE_ROOT": str(PROJECT_ROOT)}):
            app = AppTest.from_file(script, default_timeout=30).run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
            self.assertEqual(app.header[0].value, "Executive Overview")
            self.assertEqual(len(app.metric), 5)
            for section in list(SECTIONS)[1:]:
                app.sidebar.radio[0].set_value(section).run()
                self.assertFalse(app.exception, [e.message for e in app.exception])
                self.assertFalse(app.error, [e.value for e in app.error])
                self.assertEqual(app.header[0].value, section)
        self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files})

    def test_empty_root_renders_availability_without_crashing(self):
        script = str(PROJECT_ROOT / "src" / "consumer_intelligence" / "dashboard.py")
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"CONSUMER_INTELLIGENCE_ROOT": temp}):
            app = AppTest.from_file(script, default_timeout=30).run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
            self.assertTrue(app.warning)
            for section in SECTIONS:
                app.sidebar.radio[0].set_value(section).run()
                self.assertFalse(app.exception, [e.message for e in app.exception])


if __name__ == "__main__":
    unittest.main()

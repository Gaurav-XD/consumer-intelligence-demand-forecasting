"""Static checks for the Databricks customer analytics evidence."""

import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "databricks" / "customer_analytics.py"
README = ROOT / "databricks" / "README.md"
SUMMARY = ROOT / "reports" / "databricks_capability_summary.json"


class DatabricksArtifactTests(unittest.TestCase):
    def test_notebook_structure_and_syntax(self):
        source = NOTEBOOK.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(NOTEBOOK))
        self.assertTrue(tree.body)
        self.assertTrue(source.startswith("# Databricks notebook source"))
        self.assertGreaterEqual(source.count("# COMMAND ----------"), 3)
        for section in ("Configuration:", "Input data:", "Transformations and feature engineering:", "Output:"):
            self.assertIn(section, source)
        self.assertNotRegex(source, r"(?i)\b(?:todo|fixme|placeholder)\b")

    def test_shared_spark_pipeline_and_explicit_input_schemas(self):
        source = NOTEBOOK.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "consumer_intelligence.pyspark_customer_features"
        ]
        self.assertEqual(len(imports), 1)
        self.assertEqual(
            {alias.name for alias in imports[0].names},
            {"SCHEMAS", "build_spark_customer_features"},
        )
        for name in ("customers", "transactions", "products", "marketing"):
            self.assertIn(f'read_input("{name}")', source)
        for expression in (
            "spark.read", '.option("mode", "FAILFAST")', ".schema(SCHEMAS[name])",
            "build_spark_customer_features(", "as_of_date=as_of_date",
            '.write.mode("errorifexists").parquet(output_path)',
        ):
            self.assertIn(expression, source)
        self.assertNotIn(".toPandas(", source)
        self.assertNotIn("import pandas", source)

    def test_parameters_and_input_output_safety(self):
        source = NOTEBOOK.read_text(encoding="utf-8")
        for name in ("project_src_path", "raw_data_path", "output_path", "as_of_date"):
            self.assertIn(f'dbutils.widgets.text("{name}", "")', source)
            self.assertIn(f'dbutils.widgets.get("{name}")', source)
        self.assertIn('output_path.startswith(raw_data_path + "/")', source)
        self.assertIn('raw_data_path.startswith("/Volumes/")', source)
        self.assertIn('output_path.startswith("/Volumes/")', source)

    def test_readme_and_summary_describe_actual_evidence(self):
        text = README.read_text(encoding="utf-8")
        self.assertIn("has not been run on Databricks", text)
        self.assertIn("run and reconciled locally", text)
        self.assertIn("as_of_date", text)
        self.assertIn("fails if that path already exists", text)
        summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
        self.assertEqual(summary["artifact"], "databricks/customer_analytics.py")
        self.assertIn("Not performed", summary["databricks_execution"])
        self.assertIn("not this Databricks entry point", summary["local_pyspark_execution"]["status"])
        self.assertEqual(
            summary["inputs"],
            ["customers.csv", "transactions.csv", "products.csv", "marketing.csv"],
        )
        self.assertTrue(summary["requirements"])
        self.assertTrue(summary["limitations"])
        self.assertTrue((ROOT / summary["local_pyspark_execution"]["evidence"]).is_file())


if __name__ == "__main__":
    unittest.main()

"""Verify the documented PostgreSQL portfolio artifacts."""

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SQL_PATH = ROOT / "sql" / "consumer_analytics.sql"
SUMMARY_PATH = ROOT / "reports" / "sql_capability_summary.json"
SECTIONS = (
    "CUSTOMER REVENUE AND ORDER METRICS",
    "RFM-STYLE CUSTOMER METRICS",
    "CUSTOMER SEGMENTATION INPUTS",
    "MONTHLY REVENUE AND DEMAND TRENDS",
    "COHORT AND RETENTION ANALYSIS",
    "CHURN FEATURE PREPARATION",
    "PRODUCT AND CATEGORY PERFORMANCE",
)


class SqlArtifactTests(unittest.TestCase):
    def test_sql_exists_is_readable_and_has_all_sections(self):
        self.assertTrue(SQL_PATH.is_file())
        text = SQL_PATH.read_text(encoding="utf-8")
        self.assertTrue(text.strip())
        headings = re.findall(r"^-- [1-7]\. (.+)$", text, re.MULTILINE)
        self.assertEqual(headings, list(SECTIONS))
        self.assertNotRegex(text, r"(?i)\b(?:todo|fixme|placeholder)\b")

    def test_independent_read_only_parameterized_statements(self):
        text = SQL_PATH.read_text(encoding="utf-8")
        sql = re.sub(r"--[^\n]*", "", text)
        statements = [part.strip() for part in sql.split(";") if part.strip()]
        self.assertEqual(len(statements), len(SECTIONS))
        for statement in statements:
            with self.subTest(statement=statement[:50]):
                self.assertRegex(statement, r"(?i)^(?:WITH|SELECT)\b")
                self.assertIn("$1::date", statement)
                self.assertNotRegex(
                    statement, r"(?i)\b(?:insert|update|delete|drop|alter|truncate|create)\b"
                )
        for construct in ("JOIN", "WITH", "CASE", "GROUP BY", "OVER"):
            self.assertRegex(sql, rf"(?i)\b{construct}\b")

    def test_point_in_time_and_cohort_guards_are_documented(self):
        text = SQL_PATH.read_text(encoding="utf-8")
        churn = text.split("-- 6. CHURN FEATURE PREPARATION", 1)[1].split(
            "-- 7. PRODUCT AND CATEGORY PERFORMANCE", 1
        )[0]
        self.assertIn("t.transaction_date <= cutoff.snapshot_date", churn)
        self.assertIn("m.campaign_date <= p.snapshot_date", churn)
        self.assertIn("No future-derived", churn)
        cohort = text.split("-- 5. COHORT AND RETENTION ANALYSIS", 1)[1].split(
            "-- 6. CHURN FEATURE PREPARATION", 1
        )[0]
        self.assertIn("generate_series(0, 12)", cohort)
        self.assertIn("last_complete_month", cohort)
        self.assertIn("COUNT(DISTINCT h.customer_id)", cohort)

    def test_summary_is_honest_and_covers_all_sections(self):
        self.assertTrue(SUMMARY_PATH.is_file())
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(summary["artifact"], "sql/consumer_analytics.sql")
        self.assertEqual(summary["dialect"], "PostgreSQL")
        self.assertEqual([name.upper() for name in summary["sections"]], list(SECTIONS))
        self.assertIn("Not executed", summary["execution_status"])
        self.assertIn("$1", summary["parameter"])
        self.assertTrue(summary["assumptions"])
        self.assertTrue(summary["limitations"])


if __name__ == "__main__":
    unittest.main()

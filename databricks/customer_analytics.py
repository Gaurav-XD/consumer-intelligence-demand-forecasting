# Databricks notebook source
"""Databricks entry point for the shared point-in-time customer features."""

# COMMAND ----------

# Configuration: Spark and dbutils are supplied by the Databricks notebook.
import sys
from pathlib import Path

dbutils.widgets.text("project_src_path", "")
dbutils.widgets.text("raw_data_path", "")
dbutils.widgets.text("output_path", "")
dbutils.widgets.text("as_of_date", "")

project_src_path = dbutils.widgets.get("project_src_path").strip()
raw_data_path = dbutils.widgets.get("raw_data_path").strip().rstrip("/")
output_path = dbutils.widgets.get("output_path").strip().rstrip("/")
as_of_date = dbutils.widgets.get("as_of_date").strip() or None

if not project_src_path or not raw_data_path or not output_path:
    raise ValueError("project_src_path, raw_data_path and output_path are required")
if not Path(project_src_path).is_absolute() or not Path(project_src_path).is_dir():
    raise ValueError("project_src_path must be an accessible absolute workspace path")
if not raw_data_path.startswith("/Volumes/") or not output_path.startswith("/Volumes/"):
    raise ValueError("Use absolute Unity Catalog volume paths beginning with /Volumes/")
if any(part in (".", "..") for path in (raw_data_path, output_path) for part in path.split("/")):
    raise ValueError("Volume paths must not contain . or .. segments")
if output_path == raw_data_path or output_path.startswith(raw_data_path + "/"):
    raise ValueError("output_path must be outside raw_data_path")

sys.path.insert(0, project_src_path)

from consumer_intelligence.pyspark_customer_features import SCHEMAS, build_spark_customer_features

# COMMAND ----------

# Input data: read the four contract CSVs with explicit schemas.
def read_input(name):
    return (
        spark.read.option("header", True)
        .option("mode", "FAILFAST")
        .option("enforceSchema", False)
        .schema(SCHEMAS[name])
        .csv(f"{raw_data_path}/{name}.csv")
    )


customers = read_input("customers")
transactions = read_input("transactions")
products = read_input("products")
marketing = read_input("marketing")

# COMMAND ----------

# Transformations and feature engineering: the shared Spark DataFrame function
# applies the inclusive snapshot cutoff, validates keys, and builds the features.
customer_features, snapshot = build_spark_customer_features(
    customers, transactions, products, marketing, as_of_date=as_of_date
)

# COMMAND ----------

# Output: write a new distributed Parquet directory; never overwrite existing data.
customer_features.write.mode("errorifexists").parquet(output_path)
print(f"Customer feature output written for snapshot {snapshot.isoformat()}.")

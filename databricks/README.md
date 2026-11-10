# Databricks customer analytics compatibility

This notebook-source Python file is an entry point for the existing
`build_spark_customer_features` PySpark DataFrame pipeline. The existing
pipeline has been run and reconciled locally with PySpark, as recorded in
[`pyspark_customer_features_summary.json`](../reports/pyspark_customer_features_summary.json).
**This Databricks entry point has not been run on Databricks**; it is
compatibility evidence, not proof of Databricks execution or production use.

## Preparation

- Import [`customer_analytics.py`](./customer_analytics.py) as a Databricks
  Python source notebook, or open it as a notebook in a workspace Git folder.
- Make this project's `src` directory available as an absolute workspace
  filesystem path (for example `/Workspace/Users/<user>/<project>/src`).
  The notebook imports the shared project package from that directory; it is
  not a standalone copy. The runtime needs Python 3.10+, PySpark 3.5 and pandas
  because the shared package imports the existing pandas feature contract.
  Compatibility with a particular Databricks Runtime has not been tested.
- Stage the four input CSVs in a Unity Catalog volume directory accessible to
  Spark. Their filenames and column order must match the existing
  [`data_schema.py`](../src/consumer_intelligence/data_schema.py) contract:
  `customers.csv`, `transactions.csv`, `products.csv`, and `marketing.csv`.
  The reader uses the shared explicit schemas and checks CSV headers.
- Choose a **new**, separate output directory in a writable Unity Catalog
  volume; the notebook writes Parquet and fails if that path already exists.
  Input and output privileges and volume access must already be configured.

## Parameters and execution

Set the notebook widgets, then run all cells on an appropriate Spark compute:

| Widget | Meaning |
| --- | --- |
| `project_src_path` | Absolute workspace filesystem path to this project's `src` directory. |
| `raw_data_path` | Absolute `/Volumes/...` directory containing the four input CSVs. |
| `output_path` | New `/Volumes/...` Parquet output directory, outside the input directory. |
| `as_of_date` | Optional inclusive ISO date, such as `2025-12-31`; blank uses the maximum transaction date. |

No credentials are embedded or required for local static tests. Actual
Databricks execution requires a configured workspace, compute, access to the
workspace files and authorized volume access. Run in an isolated notebook
session to avoid Python import caching of a different project version.

## Semantics and limitations

The shared function retains customers signed up by the snapshot, filters
transactions and marketing interactions through that same inclusive date,
and produces one row per eligible customer. It uses Spark joins and
aggregations to calculate order, revenue, purchase-frequency, marketing and
RFM features. Its existing validation and null/zero handling are unchanged;
see [`pyspark_customer_features.py`](../src/consumer_intelligence/pyspark_customer_features.py)
and [`customer_features.py`](../src/consumer_intelligence/customer_features.py).
An explicit snapshot is recommended when historical reproducibility matters.
Parquet output is a distributed directory, not the existing local single CSV.
Ordering of output files and rows is not guaranteed. The notebook does not
reconcile its output against the local pandas artifact or update local reports.

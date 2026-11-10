# Consumer Intelligence & Demand Forecasting Platform

An independent, production-minded data science portfolio project for consumer analytics and demand forecasting in a retail or e-commerce context. The project uses synthetic data and is intended for reproducible local development; it is not a production retail deployment.

## Business objective

Demonstrate an end-to-end analytics workflow for understanding customer behavior, modeling churn, forecasting product demand, and translating analyses into explicitly assumption-based business scenarios. All source datasets are synthetic. Metrics and scenario estimates are portfolio results, not commercial outcomes or evidence of real-world business impact.

## Implemented project capabilities

- Python data contracts, synthetic data generation, data-quality checks, and exploratory analysis
- Point-in-time customer features, RFM segmentation, CLV revenue proxy, and cohort retention
- Churn classification and product-month demand forecasting, with feature pipelines, chronological validation, tuning, and walk-forward evaluation
- A retrospective uplift placebo diagnostic; it does not estimate a causal treatment effect
- Local PySpark customer-feature processing and reconciliation against the pandas pipeline
- Local file-based MLflow tracking of existing experiments and results
- Local FastAPI inference and a Streamlit business-facing dashboard
- Assumption-based business impact scenarios and PostgreSQL-style SQL examples
- A Databricks notebook-source compatibility example; it has not been executed on Databricks

This is a tested local portfolio implementation, not a production or cloud deployment. The SQL examples have not been run against PostgreSQL. Local PySpark execution is verified, but Databricks execution is not. Business-impact figures are scenario estimates based on configurable assumptions, not realized savings. Previously inspected holdout results are not fresh independent evaluations.

## High-level architecture

```text
Synthetic CSV inputs (data/raw/, generated locally and not committed)
    ↓
src/consumer_intelligence/ (reusable analytics, features, modeling and inference)
    ├── tests/ (automated checks)
    ├── databricks/ and sql/ (compatibility and query examples)
    ├── models/ (two tuned inference artifacts are committed)
    ├── reports/ (two model summaries are committed; other outputs are generated)
    └── deployment/ (local API Dockerfile)
```

The `src` package contains reusable Python code. Tests, configuration, model artifacts, reports, SQL, Databricks examples, and deployment assets are kept in separate locations. The API requires the committed tuned model artifacts and their matching tuning summaries; other generated data, reports, models, and the local MLflow store are excluded from version control.

## Project layout

```text
.
├── configs/
├── data/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── databricks/
├── deployment/
├── models/
├── notebooks/
├── reports/
├── sql/
├── src/
│   └── consumer_intelligence/
└── tests/
```

## Getting started

From the project root, create a local environment, install the requirements, and run the test suite:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

Python 3.10 or later is required by the project code. Keep raw and processed datasets, local MLflow runtime data, and generated artifacts out of version control; the tuned churn and forecasting models and their required summaries are intentional exceptions.

## Generate synthetic data

Generate the reproducible development datasets in `data/raw/` with the standard library:

```powershell
$env:PYTHONPATH = "src"
python -m consumer_intelligence.data_generator
```

The default seed is `42`; adjust the seed, row counts, date range, or output directory with CLI options such as `--seed 7`, `--customers 5000`, `--products 100`, and `--transactions 60000`. The generator writes customers, products, transactions, and marketing CSVs; these synthetic files are ignored by Git.

## Dataset contract

The initial input contract defines four record schemas in `src/consumer_intelligence/data_schema.py`:

| Dataset | Fields and Python types |
|---|---|
| `customers` | `customer_id: str`, `age: int`, `gender: str`, `region: str`, `signup_date: date` |
| `transactions` | `transaction_id: str`, `customer_id: str`, `transaction_date: date`, `product_id: str`, `quantity: int`, `price: Decimal`, `discount: Decimal` |
| `products` | `product_id: str`, `category: str`, `brand: str`, `cost: Decimal` |
| `marketing` | `customer_id: str`, `campaign: str`, `campaign_date: date`, `channel: str` |

The generator writes these fields as CSV columns; date fields use `datetime.date` in Python, and monetary and discount values use `decimal.Decimal`.

## Exploratory analysis

After generating the raw datasets, create summary CSVs, plots, and a JSON metrics summary under `reports/eda/`:

```powershell
$env:PYTHONPATH = "src"
python -m consumer_intelligence.eda
```

## Customer features

Build the point-in-time customer feature table from the raw CSVs. The snapshot is the latest transaction date, and records after it are excluded.

```powershell
$env:PYTHONPATH = "src"
python -m consumer_intelligence.customer_features
```

The table is written to `data/processed/customer_features.csv`, with feature definitions and snapshot details in `reports/customer_features_metadata.json`.

## RFM customer segmentation

Score and segment the customer feature table into deterministic RFM groups:

```powershell
$env:PYTHONPATH = "src"
python -m consumer_intelligence.rfm_segmentation
```

The customer-level scores and segments are written to `data/processed/customer_segments.csv`; segment metrics and distribution are written to `reports/rfm_segmentation_summary.json` and `reports/rfm_segment_distribution.csv`.

## Churn modeling dataset

Run `python -m consumer_intelligence.churn_dataset` with `PYTHONPATH=src` to
create `data/processed/churn_dataset.csv` and
`reports/churn_dataset_metadata.json`. Features are rebuilt from raw history as
of 2025-03-31 for training and 2025-09-30 for testing. Churn means no purchase
in the following 90 calendar days (days 1–90, inclusive); customers without
prior purchases remain eligible. The training label window ends before the
test snapshot. After labels are defined, a deterministic customer-ID hash
assigns disjoint customers to the earlier training or later test snapshot.

The existing customer feature table is used only to check customer IDs, not
as a source of later-snapshot feature values. The metadata lists the model
feature columns: `customer_id`, `split`, and `snapshot_date` are for tracking
and must not be used as predictors. Undefined historical values remain blank.
The maximum transaction date is only a proxy for data coverage, not proof
that all events were received.

## Classical churn models

After creating the churn dataset, run:

```powershell
$env:PYTHONPATH = "src"
python -m consumer_intelligence.churn_models
```

Each model uses median imputation fitted within its pipeline; logistic regression
also scales inputs. Logistic regression, decision tree and random forest use
balanced class weights; gradient boosting uses its default unweighted objective.
The best model is selected by mean average precision (PR-AUC) from three
stratified, unshuffled cross-validation folds within the predefined training
split. The predefined test split is not used for selection. All candidates
are then evaluated on it at their default prediction threshold, with results in
`reports/churn_model_results.csv` and `reports/churn_model_summary.json`.
Repeated model decisions based on the test results would compromise the holdout.

The fitted pipelines, including preprocessing, are saved in `models/churn/`;
`best_model.joblib` is the training-CV-selected model. Load joblib files only
from trusted sources, using compatible Python and scikit-learn versions.

## Churn hyperparameter tuning

With the churn dataset prepared, run:

```powershell
$env:PYTHONPATH = "src"
python -m consumer_intelligence.churn_tuning
```

Six randomized configurations per model are evaluated using three stratified
training-only folds and average precision. The best mean training-CV score
selects one pipeline, which is refitted on all training data before test
evaluation. Preprocessing remains inside each pipeline. Outputs are
`reports/churn_tuning_results.csv`, `reports/churn_tuning_summary.json`, and
`models/churn/tuned_best_model.joblib`.

The test split was already evaluated in the baseline comparison; although this
search does not use it for selection, repeated project decisions based on that
holdout can bias evaluation. An independent future holdout is needed for an
unbiased assessment after further development.

## Customer lifetime value proxy

Run `python -m consumer_intelligence.clv` with `PYTHONPATH=src` after generating
the customer feature table. It writes `data/processed/customer_clv.csv`,
`reports/clv_summary.json` and `reports/clv_by_segment.csv`. Segment summaries
are included when a matching `customer_segments.csv` is available; otherwise
the segment CSV has headers only and the JSON reports that it is unavailable.

The feature metadata snapshot is the inclusive cutoff. Historical revenue is
the sum of net transaction revenue rounded to cents per transaction. Observed
days run from the later of signup or the first transaction in the dataset
through the snapshot, inclusive. Annual revenue is historical revenue times
365.25 divided by the greater of observed days and 365. Estimated CLV is that
one-year revenue proxy, not discounted lifetime profit; it assumes a steady
run rate and does not model retention, margin or seasonality. The first
transaction date is only a proxy for the start of complete data coverage.
Customers without purchases receive zero; undefined purchase lifetimes are blank.

## Cohort and retention analysis

Run `python -m consumer_intelligence.cohort_retention` with `PYTHONPATH=src`.
It writes cohort-month customer and revenue retention to
`data/processed/cohort_retention.csv`, a customer-retention matrix to
`reports/cohort_retention_matrix.csv`, and monthly activity and weighted
retention summaries to `reports/cohort_retention_summary.json`.

Cohorts are defined by the first **observed** purchase month. M0–M12 customer
retention is unique purchasers divided by the original cohort size; revenue
retention is each month's net revenue divided by the cohort's M0 revenue and
can exceed 100%. A zero M0 revenue gives an undefined revenue-retention ratio.
The monthly repeat-purchase rate is the share of active customers whose first
purchase month was earlier than the current month.

Matured months with no purchases are zero; future periods are blank in the
matrix. If the last transaction is not on its month's final day, that month is
excluded as potentially partial. Transaction coverage is inferred from observed
dates and cannot prove that ingestion is complete or that earlier purchases
do not exist. Strongest and weakest cohorts are compared at M3.

## Monthly demand dataset

Run `python -m consumer_intelligence.forecasting_dataset` with `PYTHONPATH=src`
to write `data/processed/demand_forecasting_dataset.csv` and
`reports/forecasting_dataset_summary.json`. Every catalog product is crossed
with every month between the earliest and latest transactions. First and last
months may be partial; exact transaction dates are recorded in the summary.
No-sales months have zero quantity, revenue, and counts; average price and
discount are blank. Averages are unweighted transaction means. Revenue is the
unrounded sum of `quantity * price * (1 - discount)`.

The table contains observed same-month outcomes, not forecasting features.
Revenue, transaction counts, average price, and discounts must not be used to
predict that same month's demand without an appropriate as-of availability policy.

## Demand forecasting features

Run `python -m consumer_intelligence.forecasting_features` with `PYTHONPATH=src`
to write `data/processed/forecasting_features.csv` and
`reports/forecasting_features_metadata.json`. Per-product lags use exact prior
calendar months; rolling means and sample standard deviations use only preceding
months and require the complete window. Insufficient history remains missing.
Calendar month, quarter and an absolute month index are known in advance.
The source must have consecutive months for each product. The output retains
category and brand, assuming these attributes are available at prediction time;
historical changes to them are not tracked.

## Forecasting validation splits

Run `python -m consumer_intelligence.forecasting_validation` with
`PYTHONPATH=src` to write `reports/forecasting_validation_summary.json`.
The reusable function returns train, validation, and test frames using the
same inclusive calendar cutoffs for every product: 2024-01–2025-03,
2025-04–2025-06, and 2025-07–2025-12, respectively. Rows missing any
required historical feature are removed before splitting. The report shows
the effective date ranges after filtering.

This framework supports rolling-origin, one-step-ahead evaluation: prior
validation or test months' actual demand must be available before scoring the
next month. Precomputed lags must not be used unchanged for fixed-origin
multi-month forecasts, where those actuals would be unavailable. Product IDs
are for tracking, not numeric predictors; preprocessing must be fit only on
the training split.

## Demand forecasting model comparison

Run `python -m consumer_intelligence.forecasting_models` with `PYTHONPATH=src`
after preparing the validation summary. Seasonal naive uses the 12-month lag;
linear regression, random forest, and gradient boosting use the documented
numeric historical and calendar features with median imputation inside each
pipeline. Models fit only the training period. Lowest validation MAE selects
the winner, with ties resolved in the listed model order; only that fitted
model is evaluated on the test period.

This is rolling one-step-ahead evaluation, not a fixed-origin multistep forecast:
the features assume actual demand from prior validation/test months is available.
MAPE excludes zero actuals; WAPE is undefined when total actual demand is zero.
Percentage metrics are reported on a 0–100 scale and may exceed 100. Outputs
are `reports/forecasting_model_results.csv`,
`reports/forecasting_model_summary.json`, and
`models/forecasting/best_model.joblib`. Load model files only from trusted sources.

## Demand tuning and walk-forward validation

Run `python -m consumer_intelligence.forecasting_tuning` with `PYTHONPATH=src`
to tune random forest and gradient boosting using expanding calendar-month
folds from the training period only. With the current three eligible training
months, this produces two folds. The lowest mean fold MAE selects the model;
the saved pipeline remains trained on the training period. Validation metrics
use that fixed fit, while the separate walk-forward report refits before each
validation month using all eligible earlier months. Actual prior-month demand
must be available; this is not a fixed-origin multistep forecast. The final
test period is not used by this tuning or walk-forward process.
Step 16 already reported test results, so that period is no longer fully
untouched for subsequent development decisions.

Outputs are `reports/forecasting_tuning_results.csv`,
`reports/forecasting_tuning_summary.json`,
`reports/forecasting_walk_forward.csv`, and
`models/forecasting/tuned_best_model.joblib`. Comparing or choosing further
models using the validation period makes it part of development; a later test
evaluation should be reserved for a final frozen approach.

## Business impact scenarios

After producing the churn, CLV, segment, cohort, and forecasting outputs, run
`python -m consumer_intelligence.business_impact` with `PYTHONPATH=src`. This
also requires the churn dataset and CLV/feature metadata reports and the trusted
`models/churn/tuned_best_model.joblib` artifact. Only load model files from
trusted sources. Results are hypothetical estimates in the source data's
unspecified currency units, not measured outcomes or savings.

All scenario inputs are in `ScenarioAssumptions` in the module; override them
by passing a configured instance to `run_business_impact`. The defaults use a
0.5 risk threshold, 10% relative churn reduction, 5 units per contact, and a
30% contribution margin. Only the held-out customers at the 2025-09-30 snapshot
are scored; predicted probabilities are not calibrated and no extrapolation to
all customers is made. For each targeted customer, the 90-day revenue proxy is
historical revenue multiplied by 90 and divided by observed exposure days,
using a minimum denominator of 365 days. Risk-weighted revenue is this proxy
times predicted churn probability; revenue protected is 10% of that total.
Scenario ROI is `(revenue protected * contribution margin - contact cost) /
contact cost`, or null for zero cost. Revenue is not profit; the assumed margin
converts it to contribution before calculating ROI.

The later 2025-12-31 CLV and RFM summaries are descriptive, not features or
segment assignments for the earlier churn scenario. CLV is a one-year revenue
proxy, not lifetime profit. Weighted cohort retention uses the sizes of cohorts
observed at each period. Forecast exposure assumes 100 revenue units per unit,
25% of underforecast units become lost sales, and 10 units holding cost per
overforecast unit. These are error-based proxies, not observed stockouts,
inventory or savings; no inventory constraints, future replenishment, uplift
or seasonality adjustments are modeled. Validate decisions with prospective
holdouts and an independent future evaluation.

Outputs are `reports/business_impact_summary.json`,
`reports/business_impact_segments.csv`, `reports/business_impact_forecasting.csv`,
`reports/business_recommendations.csv`, and
`reports/business_impact_high_risk_customers.csv`.

## Retention uplift placebo diagnostic

Run `python -m consumer_intelligence.causal_uplift` with `PYTHONPATH=src`
after preparing the churn dataset and its metadata. This uses the later
customer feature table only to check IDs, and checks historical marketing
counts against events through each churn snapshot. The model uses only
point-in-time churn predictors. The outcome is observed purchase in the 90
days after each snapshot; the earlier customer-disjoint split trains two
arm-specific random-forest pipelines, and the later split evaluates rankings.

Treatment is a **hypothetical** 50/50 assignment, deterministically hashed
from the customer ID and seed 42. It is not inferred from marketing, and no
randomized intervention was actually delivered. Observed outcomes are left
unchanged. Consequently, scores, group differences, and the Qini-style area
are placebo diagnostics—not estimates of actual intervention effects or a
basis for selecting customers. Marketing records are observational, and other
campaigns may affect outcomes. A prospectively randomized intervention with
verified assignment, delivery and outcomes is needed for causal conclusions.

The held-out customers are ranked by the treated-minus-control predicted
retention probabilities and split into ten groups. The Qini-style area
integrates cumulative inverse-propensity-weighted outcome differences divided
by holdout size over the ranked population fraction, subtracting the
straight-line random-ranking baseline; it is not normalized. Groups missing an
arm have undefined differences. The existing holdout was used in earlier
steps and should not be treated as an independent final experiment.

Outputs: `data/processed/causal_uplift_dataset.csv`,
`reports/causal_uplift_summary.json`,
`reports/causal_uplift_ranked_segments.csv`, and
`models/causal_uplift/uplift_model.joblib`. Only held-out rows receive scores;
training rows remain unscored. Load model artifacts only from trusted sources.

## PySpark customer features

Install the project's Python requirements and use a Spark-compatible Java 17
runtime configured through `JAVA_HOME`. Run the following after producing the
pandas customer features and metadata:

```powershell
$env:PYTHONPATH = "src"
python -m consumer_intelligence.pyspark_customer_features
```

By default, the snapshot is the maximum transaction date, as in the pandas
pipeline. Pass `--as-of-date YYYY-MM-DD` for another snapshot; the pandas
reference and its metadata must already represent that same snapshot. Spark
filters transactions and marketing events through the snapshot and retains
customers signed up by then. Counts are zero for no history; undefined
recency, averages and frequency remain missing.

The pipeline compares all 23 fields and customer coverage with
`data/processed/customer_features.csv` in Spark; numeric comparisons use an
absolute tolerance of `1e-6`, except average order value allows `0.010001`
(one cent plus floating-point tolerance). Different distributed sum orders
can place a half-cent mean on opposite sides of pandas' rounding boundary.
Nulls must match exactly. The report includes maximum absolute differences. It fails
on mismatches instead of writing a successful report. The output is sorted by
customer ID and streamed to a single local CSV through the driver, without
converting the main pipeline to pandas. This final single-file export is for
local interoperability rather than distributed-scale output; large production
datasets should use a partitioned format. Local mode uses two workers and four
shuffle partitions; callers can supply their own Spark session.

Outputs are `data/processed/pyspark_customer_features.csv` and
`reports/pyspark_customer_features_summary.json`. PySpark requires a working
Java installation and platform-specific Hadoop/Spark prerequisites; on Windows,
configuration varies by local setup.

On Windows, the Spark launcher may fail when its installation or Java path
contains `&`. If both are inside this project's `.venv`, an unused temporary
drive mapping avoids that launcher issue. The following example uses the local
JDK installed for this project; install a Java 17 runtime separately if absent:

```powershell
if (Test-Path 'X:\') { throw 'X: is already in use' }
subst.exe X: (Get-Location).Path
if ($LASTEXITCODE -ne 0) { throw 'Could not map project directory' }
$oldJava = $env:JAVA_HOME
$oldSpark = $env:SPARK_HOME
$oldPython = $env:PYSPARK_PYTHON
$oldPath = $env:PYTHONPATH
try {
    $env:JAVA_HOME = 'X:\.venv\Lib\site-packages\jdk4py\java-runtime'
    $env:SPARK_HOME = 'X:\.venv\Lib\site-packages\pyspark'
    $env:PYSPARK_PYTHON = 'X:\.venv\Scripts\python.exe'
    $env:PYTHONPATH = 'X:\src'
    & 'X:\.venv\Scripts\python.exe' -m consumer_intelligence.pyspark_customer_features
} finally {
    subst.exe X: /D
    $env:JAVA_HOME = $oldJava
    $env:SPARK_HOME = $oldSpark
    $env:PYSPARK_PYTHON = $oldPython
    $env:PYTHONPATH = $oldPath
}
```

`jdk4py` is not a project dependency; the example path only applies if its JDK
17 package has been installed locally. Use an existing Java 17 installation
instead when available.

## Local MLflow experiment tracking

After generating the churn and forecasting model reports and artifacts, run
`python -m consumer_intelligence.mlflow_tracking` with `PYTHONPATH=src`. Install
the requirements first. MLflow uses the project-local `mlruns/` file store; no
tracking server or cloud service is needed. The command imports existing results
without retraining or selecting models. Run definitions contain source-file
SHA-256 hashes, dataset and split metadata, feature names, model parameters,
validation methodology and selection criterion. Unchanged definitions reuse
existing finished runs, with stored artifact hashes checked on reuse.

Baseline churn runs record training CV and previously reported test metrics;
baseline forecasting runs record validation metrics and the selected model's
previously reported test metrics. Tuned churn records training CV and its
previously reported test metrics; tuned forecasting records training CV,
validation and walk-forward metrics, but no final-test metrics. Metric names
include their split or evaluation prefix. Model selection remains exactly as
documented in the source reports; test metrics are never used by this importer.
Previously inspected holdouts are not fresh independent evaluations.

Original trusted joblib files are copied unchanged as MLflow artifacts, not
retrained or repackaged as MLflow model flavors. The original forecasting step
saved only its selected baseline model: Seasonal Naive, Linear Regression and
Gradient Boosting runs therefore explicitly record their fitted artifacts as
unavailable, while retaining their reported metrics and parameters. The
tracking summary is `reports/mlflow_tracking_summary.json`. Load joblib files
only from trusted sources with compatible dependencies.

## Local inference API

With the trained churn and forecasting artifacts and their tuning summaries
present, install the requirements and run:

```powershell
$env:PYTHONPATH = "src"
uvicorn consumer_intelligence.api:app --host 127.0.0.1 --port 8000
```

`GET /health` reports readiness after both models load, and `GET /models`
lists their feature contracts and SHA-256 versions. `POST /predict/churn`
accepts all 14 churn feature fields and returns `churn_probability`,
`predicted_class` and `model` metadata. Undefined churn averages or recency
values must be supplied as JSON `null`; their fields may not be omitted.
`POST /predict/forecast` accepts all 13 forecasting feature fields and returns
`predicted_quantity` and model metadata. Forecast history must be complete;
the model is evaluated for one-step-ahead forecasts using prior observed
demand, not fixed-origin multistep forecasts. Numeric features must be
nonnegative and finite, discounts must lie in `[0, 1]`, and forecast calendar
fields must agree. Forecast output is the raw model prediction, not rounded
or clipped. Complete schemas and examples can be explored at `/docs`.

Models load once at startup and the service fails startup if files are
missing, feature contracts disagree, or the Python/scikit-learn training
versions do not match. Model versions are SHA-256 artifact hashes. Joblib
artifacts must come from trusted sources; loading untrusted pickle files can
execute code. This local service has no authentication, rate limiting, TLS or
deployment monitoring; add those controls before exposing it to untrusted
networks. Synthetic-data models and predictions are not validated for real
business decisions.

Build from the project root (the artifacts and reports are required in the
build context):

```powershell
docker build -f deployment/Dockerfile -t consumer-intelligence-api .
docker run --rm -p 8000:8000 consumer-intelligence-api
```

The image uses Python 3.11 and scikit-learn 1.4.2, matching the saved models.
The Docker ignore file excludes datasets, other models and tracking artifacts.

## Business dashboard

After generating the EDA, churn, forecasting, customer and business-impact
reports, launch the read-only dashboard:

```powershell
$env:PYTHONPATH = "src"
streamlit run src/consumer_intelligence/dashboard.py --server.address 127.0.0.1
```

The five sections cover executive metrics, RFM/CLV and cohort retention, churn
model evaluation and the historical high-risk test subset, walk-forward
forecast accuracy, and illustrative business scenarios. Filters affect only
the charts or tables in their section. The risk distribution contains only
customers preselected by the original scenario threshold; it does not
represent all customers. Forecast MAE and WAPE are recalculated from
product-month errors for the selected products. CLV is a one-year revenue
proxy, and ROI and revenue at risk are assumptions-based estimates, not
realized savings. Monetary units are unspecified.

The dashboard reads only these existing artifacts: `reports/eda/eda_summary.json`,
`reports/churn_dataset_metadata.json`, `reports/churn_tuning_summary.json`,
`reports/churn_model_results.csv`, `reports/forecasting_tuning_summary.json`,
`reports/forecasting_walk_forward.csv`, `reports/business_impact_summary.json`,
`reports/business_impact_segments.csv`, `reports/business_impact_high_risk_customers.csv`,
`reports/business_recommendations.csv`, and `data/processed/cohort_retention.csv`.
Missing or malformed reports are surfaced without preventing other sections
from loading. No models are retrained or invoked. The dashboard has no
authentication or authorization; keep it local until appropriate access
controls are in place, especially because the risk table contains customer IDs.

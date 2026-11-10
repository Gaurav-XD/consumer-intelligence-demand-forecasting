"""Record existing experiment results and unchanged model files in local MLflow."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

import joblib
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from consumer_intelligence.config import PROJECT_ROOT

EXPERIMENTS = {
    "churn": "Consumer Intelligence - Churn",
    "forecasting": "Consumer Intelligence - Forecasting",
}
BASELINE_CHURN = {
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest.joblib",
    "Gradient Boosting": "gradient_boosting.joblib",
}
BASELINE_FORECASTING = ("Seasonal Naive", "Linear Regression", "Random Forest", "Gradient Boosting")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parameters(model_path: Path, expected_type: type) -> dict[str, Any]:
    """Read parameters from an existing, trusted fitted joblib pipeline."""
    model = joblib.load(model_path)
    if not hasattr(model, "named_steps") or not isinstance(model.named_steps.get("model"), expected_type):
        raise ValueError(f"Unexpected estimator in {model_path}")
    return {
        key: str(value) if isinstance(value, float) and not math.isfinite(value) else value
        for key, value in model.get_params(deep=True).items()
        if value is None or isinstance(value, (str, bool, int, float))
    }


def _metrics(prefix: str, values: dict[str, Any]) -> dict[str, float]:
    result = {}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {prefix}{key} metric")
        result[prefix + key] = float(value)
    return result


def build_experiment_definitions(root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    """Assemble deterministic run definitions from existing, trusted artifacts."""
    root = Path(root)
    reports = root / "reports"
    churn_baseline = _read_json(reports / "churn_model_summary.json")
    churn_tuned = _read_json(reports / "churn_tuning_summary.json")
    churn_splits = _read_json(reports / "churn_dataset_metadata.json")
    forecast_baseline = _read_json(reports / "forecasting_model_summary.json")
    forecast_tuned = _read_json(reports / "forecasting_tuning_summary.json")
    forecast_splits = _read_json(reports / "forecasting_validation_summary.json")
    if churn_tuned["selected_model"] != "Random Forest" or forecast_tuned["selected_model"] != "Random Forest":
        raise ValueError("Tuned reports do not identify the requested Random Forest models")
    if forecast_baseline["selected_model"] != "Random Forest":
        raise ValueError("The saved forecasting baseline is not the requested Random Forest")
    if churn_baseline["feature_columns"] != churn_splits["feature_columns"] or churn_tuned["feature_columns"] != churn_splits["feature_columns"]:
        raise ValueError("Churn model and dataset feature contracts differ")
    if forecast_baseline["feature_columns"] != forecast_splits["numeric_feature_columns"] or forecast_tuned["feature_columns"] != forecast_splits["numeric_feature_columns"]:
        raise ValueError("Forecast model and dataset feature contracts differ")

    definitions = []

    def add(
        domain: str, name: str, variant: str, artifact: Path | None,
        parameters: dict[str, Any], metrics: dict[str, float],
        report: str, split_report: str, dataset: str,
        splits: dict[str, Any], methodology: str, selection: str,
        features: list[str], artifact_reason: str = "",
    ) -> None:
        report_path, split_path, dataset_path = root / report, root / split_report, root / dataset
        paths = [report_path, split_path, dataset_path]
        if artifact is not None:
            paths.append(artifact)
        hashes = {str(path.relative_to(root)): _sha256(path) for path in paths}
        definition = {
            "experiment": EXPERIMENTS[domain], "domain": domain, "model_name": name,
            "variant": variant, "dataset": dataset, "feature_columns": features,
            "split_info": splits, "validation_methodology": methodology,
            "selection_criterion": selection, "parameters": parameters, "metrics": metrics,
            "model_path": str(artifact.relative_to(root)) if artifact else None,
            "model_artifact_status": "available" if artifact else "not_saved_by_original_run",
            "model_artifact_reason": artifact_reason, "source_hashes": hashes,
            "source_reports": [report, split_report],
        }
        definitions.append(definition)

    churn_periods = churn_splits["periods"]
    for name, filename in BASELINE_CHURN.items():
        model = root / "models" / "churn" / filename
        model_type = {
            "Logistic Regression": LogisticRegression, "Random Forest": RandomForestClassifier,
            "Gradient Boosting": GradientBoostingClassifier,
        }[name]
        baseline_metrics = churn_baseline["test_metrics"][name]
        metrics = _metrics("test_", {k: v for k, v in baseline_metrics.items() if k != "train_cv_average_precision"})
        metrics["train_cv_average_precision"] = float(baseline_metrics["train_cv_average_precision"])
        add(
            "churn", name, "baseline", model, _parameters(model, model_type), metrics,
            "reports/churn_model_summary.json", "reports/churn_dataset_metadata.json",
            "data/processed/churn_dataset.csv", churn_periods,
            "Three stratified, unshuffled training-only CV folds; test evaluation is descriptive.",
            churn_baseline["selection_criterion"], churn_baseline["feature_columns"],
        )
    churn_model = root / "models" / "churn" / "tuned_best_model.joblib"
    parameters = _parameters(churn_model, RandomForestClassifier)
    for key, value in churn_tuned["best_parameters"].items():
        if parameters.get(key) != value:
            raise ValueError(f"Churn tuned artifact parameter {key} disagrees with its report")
    tuned_metrics = _metrics("test_", churn_tuned["test_metrics"])
    for name, value in zip(("true_negative", "false_positive", "false_negative", "true_positive"), sum(churn_tuned["test_metrics"]["confusion_matrix"], [])):
        tuned_metrics[f"test_{name}"] = float(value)
    tuned_metrics.update({
        "train_cv_average_precision": float(churn_tuned["mean_cv_average_precision"]),
        "train_cv_average_precision_std": float(churn_tuned["std_cv_average_precision"]),
    })
    add(
        "churn", "Random Forest", "tuned", churn_model, parameters, tuned_metrics,
        "reports/churn_tuning_summary.json", "reports/churn_dataset_metadata.json",
        "data/processed/churn_dataset.csv", churn_periods,
        churn_tuned["search"]["cross_validation"], churn_tuned["selection_criterion"],
        churn_tuned["feature_columns"],
    )

    for name in BASELINE_FORECASTING:
        artifact = root / "models" / "forecasting" / "best_model.joblib" if name == "Random Forest" else None
        if artifact is not None:
            artifact_params = _parameters(artifact, RandomForestRegressor)
            for key, value in forecast_baseline["model_parameters"][name].items():
                if artifact_params.get(f"model__{key}") != value:
                    raise ValueError(f"Forecast baseline artifact parameter {key} disagrees with its report")
        metrics = _metrics("validation_", forecast_baseline["validation_metrics"][name])
        if name == forecast_baseline["selected_model"]:
            metrics.update(_metrics("test_", forecast_baseline["test_metrics"]))
        add(
            "forecasting", name, "baseline", artifact,
            forecast_baseline["model_parameters"][name], metrics,
            "reports/forecasting_model_summary.json", "reports/forecasting_validation_summary.json",
            "data/processed/forecasting_features.csv", forecast_splits["splits"],
            forecast_baseline["evaluation_methodology"], forecast_baseline["selection_criterion"],
            forecast_baseline["feature_columns"],
            "" if artifact else "Original forecasting step saved only the selected model; no fitted artifact is available.",
        )
    forecast_model = root / "models" / "forecasting" / "tuned_best_model.joblib"
    parameters = _parameters(forecast_model, RandomForestRegressor)
    for key, value in forecast_tuned["best_parameters"].items():
        if parameters.get(key) != value:
            raise ValueError(f"Forecast tuned artifact parameter {key} disagrees with its report")
    metrics = {
        "train_cv_mae": float(forecast_tuned["cv_mae"]),
        "train_cv_mae_std": float(forecast_tuned["models"]["Random Forest"]["cv_mae_std"]),
        **_metrics("validation_", forecast_tuned["validation_metrics"]),
        **_metrics("walk_forward_", forecast_tuned["walk_forward_metrics"]),
    }
    add(
        "forecasting", "Random Forest", "tuned", forecast_model, parameters, metrics,
        "reports/forecasting_tuning_summary.json", "reports/forecasting_validation_summary.json",
        "data/processed/forecasting_features.csv", forecast_splits["splits"],
        forecast_tuned["methodology"], forecast_tuned["selection_criterion"],
        forecast_tuned["feature_columns"],
    )
    return definitions


def _text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _require_local_artifacts(uri: str, tracking_dir: Path) -> None:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise ValueError(f"MLflow artifact location is not a local file URI: {uri}")
    path = Path(url2pathname(parsed.path)).resolve()
    if path != tracking_dir and tracking_dir not in path.parents:
        raise ValueError(f"MLflow artifact location is outside the tracking directory: {uri}")


def run_mlflow_tracking(
    root: Path = PROJECT_ROOT,
    tracking_dir: Path | None = None,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    """Import existing experiment records into a local file store, idempotently."""
    root = Path(root).resolve()
    tracking_dir = Path(tracking_dir).resolve() if tracking_dir is not None else root / "mlruns"
    summary_path = Path(summary_path).resolve() if summary_path is not None else root / "reports" / "mlflow_tracking_summary.json"
    if tracking_dir == root or root not in tracking_dir.parents:
        raise ValueError("MLflow tracking directory must be inside the project")
    if root not in summary_path.parents or tracking_dir == summary_path or tracking_dir in summary_path.parents:
        raise ValueError("Summary must be in the project outside the MLflow store")
    definitions = build_experiment_definitions(root)
    inputs = {root / path for definition in definitions for path in definition["source_hashes"]}
    if summary_path in inputs or any(tracking_dir == path or tracking_dir in path.parents for path in inputs):
        raise ValueError("Tracking outputs must not overwrite source inputs")
    tracking_dir.mkdir(parents=True, exist_ok=True)
    uri = tracking_dir.as_uri()
    client = MlflowClient(tracking_uri=uri)
    runs = []
    for definition in definitions:
        experiment = client.get_experiment_by_name(definition["experiment"])
        experiment_id = experiment.experiment_id if experiment else client.create_experiment(definition["experiment"])
        _require_local_artifacts(client.get_experiment(experiment_id).artifact_location, tracking_dir)
        fingerprint = hashlib.sha256(_text(definition).encode("utf-8")).hexdigest()
        existing = client.search_runs(
            [experiment_id], filter_string=f"tags.definition_sha256 = '{fingerprint}' and attributes.status = 'FINISHED'"
        )
        params = {key: _text(value) for key, value in definition["parameters"].items()}
        params.update({
            "model_name": definition["model_name"], "variant": definition["variant"],
            "dataset": definition["dataset"], "model_artifact_status": definition["model_artifact_status"],
        })
        if existing:
            run = existing[0]
            _require_local_artifacts(run.info.artifact_uri, tracking_dir)
            if run.data.params != params or run.data.metrics != definition["metrics"]:
                raise ValueError(f"Existing MLflow run {run.info.run_id} does not match its definition")
            sources = [(report, "source_reports") for report in definition["source_reports"]]
            if definition["model_path"]:
                sources.append((definition["model_path"], "model"))
            for source, folder in sources:
                artifact_path = f"{folder}/{Path(source).name}"
                artifact_names = {item.path for item in client.list_artifacts(run.info.run_id, folder)}
                if artifact_path not in artifact_names:
                    raise ValueError(f"Existing MLflow run {run.info.run_id} is missing {artifact_path}")
                downloaded = Path(client.download_artifacts(run.info.run_id, artifact_path))
                if _sha256(downloaded) != definition["source_hashes"][str(Path(source))]:
                    raise ValueError(f"Existing MLflow run {run.info.run_id} has a modified {artifact_path}")
        else:
            run = client.create_run(experiment_id, tags={
                "mlflow.runName": f"{definition['model_name']} ({definition['variant']})",
                "definition_sha256": fingerprint,
                "model_name": definition["model_name"],
                "variant": definition["variant"],
                "validation_methodology": definition["validation_methodology"],
                "selection_criterion": definition["selection_criterion"],
                "dataset": definition["dataset"],
                "split_info": _text(definition["split_info"]),
                "feature_columns": _text(definition["feature_columns"]),
                "model_artifact_status": definition["model_artifact_status"],
                "model_artifact_reason": definition["model_artifact_reason"],
                "source_hashes": _text(definition["source_hashes"]),
                "test_metrics_use": "Previously reported results only; never used for selection by this tracking process.",
            })
            try:
                _require_local_artifacts(run.info.artifact_uri, tracking_dir)
                for key, value in params.items():
                    client.log_param(run.info.run_id, key, value)
                for key, value in definition["metrics"].items():
                    if not math.isfinite(value):
                        raise ValueError(f"Metric {key} must be finite")
                    client.log_metric(run.info.run_id, key, value)
                for report in definition["source_reports"]:
                    client.log_artifact(run.info.run_id, str(root / report), "source_reports")
                if definition["model_path"]:
                    client.log_artifact(run.info.run_id, str(root / definition["model_path"]), "model")
                client.set_terminated(run.info.run_id, "FINISHED")
            except Exception:
                client.set_terminated(run.info.run_id, "FAILED")
                raise
        runs.append({
            "experiment": definition["experiment"], "run_id": run.info.run_id,
            "model_name": definition["model_name"], "variant": definition["variant"],
            "model_artifact_status": definition["model_artifact_status"],
            "model_artifact_reason": definition["model_artifact_reason"],
            "metric_names": sorted(definition["metrics"]), "definition_sha256": fingerprint,
        })
    summary: dict[str, Any] = {
        "mlflow_version": mlflow.__version__, "tracking_uri": uri,
        "artifact_location": str(tracking_dir),
        "experiments": list(EXPERIMENTS.values()), "run_count": len(runs), "runs": runs,
        "methodology": (
            "Imports existing reports and original joblib files without training or selection. "
            "Existing test metrics are descriptive and were not used for model selection here."
        ),
        "limitations": [
            "Three non-selected forecasting baseline artifacts were not saved by the original step.",
            "Stored joblib files are original artifacts, not MLflow model flavors; load only from trusted sources.",
            "Previously examined holdout metrics do not constitute a fresh independent test.",
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
        file.write("\n")
    return summary


if __name__ == "__main__":
    report = run_mlflow_tracking()
    print(f"MLflow {report['mlflow_version']}: {report['run_count']} runs in {report['tracking_uri']}")

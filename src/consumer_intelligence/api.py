"""FastAPI inference service for the previously trained model artifacts."""

import hashlib
import json
import logging
import math
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

from consumer_intelligence.api_schemas import (
    ChurnFeatures, ChurnResponse, ErrorResponse, ForecastFeatures, ForecastResponse,
    HealthResponse, ModelListResponse, ModelMetadata,
)
from consumer_intelligence.config import PROJECT_ROOT

logger = logging.getLogger(__name__)
MODEL_SPECS = {
    "churn": ("models/churn/tuned_best_model.joblib", "reports/churn_tuning_summary.json", ChurnFeatures, RandomForestClassifier),
    "forecast": ("models/forecasting/tuned_best_model.joblib", "reports/forecasting_tuning_summary.json", ForecastFeatures, RandomForestRegressor),
}


@dataclass(frozen=True)
class LoadedModel:
    pipeline: Pipeline
    metadata: ModelMetadata


class InferenceError(Exception):
    """A model could not produce a valid prediction."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_models(root: Path = PROJECT_ROOT) -> dict[str, LoadedModel]:
    """Load trusted local model files and verify their saved feature contracts."""
    loaded: dict[str, LoadedModel] = {}
    for kind, (model_name, report_name, schema, estimator_type) in MODEL_SPECS.items():
        artifact = root / model_name
        with (root / report_name).open(encoding="utf-8") as file:
            report: dict[str, Any] = json.load(file)
        trained_version = report["library_versions"]["scikit_learn"]
        if trained_version != sklearn.__version__:
            raise RuntimeError(
                f"{kind} model requires scikit-learn {trained_version}; installed version is {sklearn.__version__}"
            )
        if report["selected_model"] != "Random Forest":
            raise RuntimeError(f"{kind} report does not describe the expected Random Forest")
        trained_python = report["library_versions"]["python"]
        if trained_python.split(".")[:2] != [str(value) for value in sys.version_info[:2]]:
            raise RuntimeError(f"{kind} model was trained using Python {trained_python}")
        # Joblib may execute code while loading; these files must be trusted.
        model = joblib.load(artifact)
        if not isinstance(model, Pipeline) or not isinstance(model.named_steps.get("model"), estimator_type):
            raise RuntimeError(f"{kind} artifact has an unexpected estimator")
        features = report["feature_columns"]
        if features != list(schema.model_fields) or features != list(model.feature_names_in_):
            raise RuntimeError(f"{kind} model, report and API feature contracts differ")
        if kind == "churn" and list(model.classes_) != [0, 1]:
            raise RuntimeError("Churn model does not expose binary classes 0 and 1")
        parameters = model.get_params(deep=True)
        for name, value in report["best_parameters"].items():
            if name not in parameters or parameters[name] != value:
                raise RuntimeError(f"{kind} model parameter {name} differs from its report")
        digest = _sha256(artifact)
        metadata = ModelMetadata(
            name=f"{report['selected_model']} ({kind}, tuned)",
            version=f"sha256:{digest}",
            artifact_sha256=digest,
            trained_scikit_learn_version=trained_version,
            feature_columns=features,
        )
        loaded[kind] = LoadedModel(model, metadata)
    return loaded


def _frame(features: ChurnFeatures | ForecastFeatures, metadata: ModelMetadata) -> pd.DataFrame:
    return pd.DataFrame([features.model_dump()], columns=metadata.feature_columns, dtype=float)


@asynccontextmanager
async def lifespan(app: FastAPI):
    root = Path(os.environ.get("CONSUMER_INTELLIGENCE_ROOT", str(PROJECT_ROOT))).resolve()
    app.state.models = load_models(root)
    try:
        yield
    finally:
        app.state.models = {}


app = FastAPI(title="Consumer Intelligence Inference", version="1.0.0", lifespan=lifespan)


@app.exception_handler(InferenceError)
async def inference_error_handler(request: Request, exc: InferenceError) -> JSONResponse:
    logger.error("Inference failed for %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(status_code=500, content=ErrorResponse(
        error="inference_error", message="Model inference failed."
    ).model_dump())


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/models", response_model=ModelListResponse)
def models(request: Request) -> ModelListResponse:
    return ModelListResponse(models={name: model.metadata for name, model in request.app.state.models.items()})


@app.post("/predict/churn", response_model=ChurnResponse, responses={500: {"model": ErrorResponse}})
def predict_churn(features: ChurnFeatures, request: Request) -> ChurnResponse:
    loaded: LoadedModel = request.app.state.models["churn"]
    try:
        frame = _frame(features, loaded.metadata)
        probabilities = np.asarray(loaded.pipeline.predict_proba(frame), dtype=float)
        prediction = np.asarray(loaded.pipeline.predict(frame))
        if probabilities.shape != (1, 2) or prediction.shape != (1,):
            raise InferenceError("Unexpected churn model output shape")
        probability = float(probabilities[0, 1])
        predicted = int(prediction[0])
        if not math.isfinite(probability) or not 0 <= probability <= 1 or predicted not in (0, 1):
            raise InferenceError("Invalid churn model output")
    except (ValueError, TypeError, ArithmeticError, IndexError) as exc:
        raise InferenceError("Churn model prediction failed") from exc
    return ChurnResponse(churn_probability=probability, predicted_class=predicted, model=loaded.metadata)


@app.post("/predict/forecast", response_model=ForecastResponse, responses={500: {"model": ErrorResponse}})
def predict_forecast(features: ForecastFeatures, request: Request) -> ForecastResponse:
    loaded: LoadedModel = request.app.state.models["forecast"]
    try:
        prediction = np.asarray(loaded.pipeline.predict(_frame(features, loaded.metadata)), dtype=float)
        if prediction.shape != (1,) or not math.isfinite(float(prediction[0])):
            raise InferenceError("Invalid forecast model output")
    except (ValueError, TypeError, ArithmeticError, IndexError) as exc:
        raise InferenceError("Forecast model prediction failed") from exc
    return ForecastResponse(predicted_quantity=float(prediction[0]), model=loaded.metadata)

"""Placebo uplift workflow; hypothetical assignments do not cause the outcomes."""

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from consumer_intelligence.churn_dataset import MODEL_FEATURE_COLUMNS
from consumer_intelligence.config import MODEL_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR

SEED = 42
ASSIGNMENT_PROBABILITY = 0.5


def assign_hypothetical_treatment(customer_id: str, seed: int = SEED) -> int:
    """Assign a stable, uniformly hashed placebo arm, independent of outcomes."""
    digest = hashlib.sha256(f"retention-placebo:{seed}:{customer_id}".encode("utf-8")).digest()
    return int(int.from_bytes(digest[:8], "big") < 2**63)


class TwoModelUplift(BaseEstimator):
    """Difference in retention probabilities from two arm-specific pipelines."""

    def __init__(self, random_state: int = SEED, n_estimators: int = 100, min_samples_leaf: int = 20):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf

    def fit(self, x: pd.DataFrame, treatment: pd.Series, outcome: pd.Series) -> "TwoModelUplift":
        if len(x) != len(treatment) or len(x) != len(outcome):
            raise ValueError("Features, treatment and outcome must have equal lengths")
        if not treatment.isin([0, 1]).all() or not outcome.isin([0, 1]).all():
            raise ValueError("Treatment and outcome must be binary")
        self.models_: dict[int, Pipeline] = {}
        self.feature_names_in_ = np.asarray(x.columns, dtype=object)
        for arm in (0, 1):
            mask = treatment.to_numpy() == arm
            if not mask.any() or outcome.iloc[np.flatnonzero(mask)].nunique() != 2:
                raise ValueError(f"Training arm {arm} must contain both outcome classes")
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("model", RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state,
                    n_jobs=1,
                )),
            ])
            model.fit(x.iloc[np.flatnonzero(mask)], outcome.iloc[np.flatnonzero(mask)])
            self.models_[arm] = model
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if list(x.columns) != list(self.feature_names_in_):
            raise ValueError("Uplift predictors do not match the fitted feature contract")
        return self.models_[1].predict_proba(x)[:, 1] - self.models_[0].predict_proba(x)[:, 1]


def evaluate_ranked_uplift(
    evaluation: pd.DataFrame, assignment_probability: float = ASSIGNMENT_PROBABILITY, groups: int = 10
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rank holdout rows and compute group differences and propensity-weighted gain."""
    if not 0 < assignment_probability < 1 or groups < 1 or len(evaluation) < groups:
        raise ValueError("Evaluation needs a valid propensity and enough rows for ranked groups")
    frame = evaluation.copy()
    if frame["customer_id"].isna().any() or frame["customer_id"].duplicated().any():
        raise ValueError("Evaluation customers must be unique and present")
    if not frame["treatment"].isin([0, 1]).all() or not frame["outcome"].isin([0, 1]).all():
        raise ValueError("Treatment and outcome must be binary")
    if not np.isfinite(frame["predicted_uplift"].to_numpy(dtype=float)).all():
        raise ValueError("Predicted uplift must be finite")
    if frame["treatment"].nunique() != 2:
        raise ValueError("Evaluation requires both hypothetical arms")
    frame = frame.sort_values(["predicted_uplift", "customer_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    frame["ranked_group"] = np.arange(len(frame)) * groups // len(frame) + 1
    weighted = (
        frame["outcome"] * frame["treatment"] / assignment_probability
        - frame["outcome"] * (1 - frame["treatment"]) / (1 - assignment_probability)
    ).to_numpy(dtype=float)
    cumulative = np.cumsum(weighted) / len(frame)
    x_axis = np.arange(1, len(frame) + 1) / len(frame)
    # Trapezoidal area over population fraction, including the origin.
    area = float(np.trapz(np.insert(cumulative, 0, 0.0), np.insert(x_axis, 0, 0.0)))
    baseline_area = float(cumulative[-1] / 2)
    rows = []
    for group, members in frame.groupby("ranked_group", sort=True):
        treated = members.loc[members["treatment"].eq(1), "outcome"]
        control = members.loc[members["treatment"].eq(0), "outcome"]
        t_rate = float(treated.mean()) if len(treated) else None
        c_rate = float(control.mean()) if len(control) else None
        rows.append({
            "ranked_group": int(group),
            "customer_count": len(members),
            "treated_count": len(treated),
            "control_count": len(control),
            "mean_predicted_uplift": float(members["predicted_uplift"].mean()),
            "treated_retention_rate": t_rate,
            "control_retention_rate": c_rate,
            "observed_retention_difference": t_rate - c_rate if t_rate is not None and c_rate is not None else None,
        })
    return pd.DataFrame(rows), {
        "auuc": area,
        "random_baseline_area": baseline_area,
        "qini_style_area": area - baseline_area,
        "full_population_ipw_difference": float(cumulative[-1]),
        "definition": (
            "Each ordered customer's transformed outcome is y*(t/p - (1-t)/(1-p)); "
            "cumulative sums divided by holdout size are integrated over population fraction. "
            "Qini-style area subtracts the straight-line random-ranking baseline; it is not normalized."
        ),
    }


def _read_ids(frame: pd.DataFrame, name: str) -> None:
    ids = frame["customer_id"]
    if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError(f"{name} must contain unique nonempty customer IDs")


def run_causal_uplift(
    churn_path: Path = PROCESSED_DATA_DIR / "churn_dataset.csv",
    features_path: Path = PROCESSED_DATA_DIR / "customer_features.csv",
    marketing_path: Path = RAW_DATA_DIR / "marketing.csv",
    metadata_path: Path = REPORTS_DIR / "churn_dataset_metadata.json",
    dataset_path: Path = PROCESSED_DATA_DIR / "causal_uplift_dataset.csv",
    summary_path: Path = REPORTS_DIR / "causal_uplift_summary.json",
    ranked_path: Path = REPORTS_DIR / "causal_uplift_ranked_segments.csv",
    model_path: Path = MODEL_DIR / "causal_uplift" / "uplift_model.joblib",
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Run a retrospective placebo diagnostic, not a real treatment experiment."""
    inputs = [Path(path).resolve() for path in (churn_path, features_path, marketing_path, metadata_path)]
    outputs = [Path(path).resolve() for path in (dataset_path, summary_path, ranked_path, model_path)]
    if len(set(outputs)) != len(outputs) or set(inputs) & set(outputs):
        raise ValueError("Output paths must be distinct from inputs and each other")
    churn = pd.read_csv(churn_path, dtype={"customer_id": "string"})
    customers = pd.read_csv(features_path, usecols=["customer_id"], dtype={"customer_id": "string"})
    marketing = pd.read_csv(marketing_path, dtype={"customer_id": "string", "campaign": "string"})
    with Path(metadata_path).open(encoding="utf-8") as file:
        metadata = json.load(file)
    _read_ids(churn, "Churn dataset")
    _read_ids(customers, "Customer features")
    if set(churn["customer_id"]) != set(customers["customer_id"]):
        raise ValueError("Churn and customer feature populations differ")
    if metadata["feature_columns"] != MODEL_FEATURE_COLUMNS or metadata["target_column"] != "churn":
        raise ValueError("Churn feature contract does not match approved point-in-time features")
    if not churn["split"].isin(["train", "test"]).all() or not churn["churn"].isin([0, 1]).all():
        raise ValueError("Churn split or target is invalid")
    if marketing["customer_id"].isna().any() or not marketing["customer_id"].isin(customers["customer_id"]).all():
        raise ValueError("Marketing has missing or unknown customer IDs")
    if marketing["campaign"].isna().any() or marketing["campaign"].str.strip().eq("").any():
        raise ValueError("Marketing campaigns must be present")
    marketing["campaign_date"] = pd.to_datetime(marketing["campaign_date"], errors="raise")
    if marketing["campaign_date"].isna().any():
        raise ValueError("Marketing dates must be present")
    churn[MODEL_FEATURE_COLUMNS] = churn[MODEL_FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    if np.isinf(churn[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)).any():
        raise ValueError("Predictors must not contain infinite values")

    periods = {}
    for split in ("train", "test"):
        rows = churn.loc[churn["split"].eq(split)]
        info = metadata["periods"][split]
        snapshot = pd.Timestamp(info["feature_snapshot_period"]["snapshot_date_inclusive"])
        window = info["churn_observation_window"]
        start = pd.Timestamp(window["start_inclusive"])
        end = pd.Timestamp(window["end_inclusive"])
        if (
            pd.isna(snapshot) or pd.isna(start) or pd.isna(end)
            or start != snapshot + pd.Timedelta(days=1)
            or (end - start).days + 1 != window["days"]
            or len(rows) != info["row_count"] or rows.empty
            or not rows["snapshot_date"].eq(snapshot.strftime("%Y-%m-%d")).all()
        ):
            raise ValueError(f"Invalid {split} snapshot or outcome window")
        periods[split] = {"snapshot": snapshot, "start": start, "end": end}
        # Independently validate the pre-snapshot marketing feature provenance.
        history = marketing.loc[marketing["campaign_date"].le(snapshot)]
        counts = history.groupby("customer_id")["campaign"].agg(["size", "nunique"])
        expected = rows["customer_id"].map(counts["size"]).fillna(0)
        campaigns = rows["customer_id"].map(counts["nunique"]).fillna(0)
        if not rows["number_marketing_interactions"].eq(expected).all() or not rows["number_marketing_campaigns"].eq(campaigns).all():
            raise ValueError(f"{split} marketing features do not match pre-snapshot history")
    if periods["train"]["end"] >= periods["test"]["snapshot"]:
        raise ValueError("Training outcomes must precede the test snapshot")

    dataset = churn[["customer_id", "split", "snapshot_date", *MODEL_FEATURE_COLUMNS]].copy()
    dataset["assignment_date"] = dataset["snapshot_date"]
    dataset["outcome_start"] = dataset["split"].map({key: value["start"].date().isoformat() for key, value in periods.items()})
    dataset["outcome_end"] = dataset["split"].map({key: value["end"].date().isoformat() for key, value in periods.items()})
    dataset["treatment"] = dataset["customer_id"].map(lambda identifier: assign_hypothetical_treatment(identifier, seed)).astype(int)
    dataset["treatment_is_hypothetical"] = True
    dataset["outcome"] = 1 - churn["churn"].astype(int)
    train = dataset.loc[dataset["split"].eq("train")]
    test = dataset.loc[dataset["split"].eq("test")].copy()
    model = TwoModelUplift(random_state=seed)
    model.fit(train[MODEL_FEATURE_COLUMNS], train["treatment"], train["outcome"])
    dataset["predicted_uplift"] = np.nan
    dataset.loc[test.index, "predicted_uplift"] = model.predict(test[MODEL_FEATURE_COLUMNS])
    dataset["prediction_scope"] = np.where(dataset["split"].eq("test"), "held_out", "not_scored")
    test = dataset.loc[dataset["split"].eq("test")].copy()
    ranked, evaluation = evaluate_ranked_uplift(test)
    arms = {}
    for arm, name in ((1, "hypothetical_treatment"), (0, "hypothetical_control")):
        outcomes = test.loc[test["treatment"].eq(arm), "outcome"]
        arms[name] = {"count": len(outcomes), "retention_rate": float(outcomes.mean())}
    difference = arms["hypothetical_treatment"]["retention_rate"] - arms["hypothetical_control"]["retention_rate"]
    summary: dict[str, Any] = {
        "analysis_type": "retrospective placebo diagnostic; no randomized intervention occurred",
        "seed": seed,
        "assignment_probability": ASSIGNMENT_PROBABILITY,
        "assignment": "SHA-256 of retention-placebo:seed:customer_id; first 64 bits below 2^63 assign hypothetical treatment.",
        "treatment_definition": "Hypothetical assignment to receive a retention intervention; no receipt or delivery is observed.",
        "outcome_definition": "Observed retention: 1 minus existing churn label, i.e. at least one purchase in the subsequent observation window.",
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "periods": {
            key: {name: date.date().isoformat() for name, date in values.items()}
            for key, values in periods.items()
        },
        "training_arm_counts": {str(arm): int(train["treatment"].eq(arm).sum()) for arm in (0, 1)},
        "holdout_arms": arms,
        "estimated_causal_treatment_effect": None,
        "holdout_observed_retention_difference": difference,
        "ranked_evaluation": evaluation,
        "top_ranked_group": {
            key: None if pd.isna(value) else value
            for key, value in ranked.iloc[0].to_dict().items()
        },
        "limitations": [
            "The treatment assignment is hypothetical and made after outcomes were observed; no intervention was delivered.",
            "Observed arm differences and individual uplift scores are placebo diagnostics, not causal effects or evidence of who would benefit.",
            "Marketing is observational and may include other actual campaigns; it is not used as treatment.",
            "Customer features from the later feature table are used only for ID validation; predictors come from earlier churn snapshots.",
            "The churn labels and snapshot features rely on upstream data provenance and transaction coverage.",
            "Ranked group differences are noisy, not significance tests; the holdout was used in earlier project steps.",
            "A prospectively randomized intervention with verified delivery and outcomes is required for causal interpretation.",
        ],
    }
    dataset_path, summary_path, ranked_path, model_path = map(Path, (dataset_path, summary_path, ranked_path, model_path))
    for path in (dataset_path, summary_path, ranked_path, model_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(dataset_path, index=False)
    ranked.to_csv(ranked_path, index=False)
    joblib.dump(model, model_path)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
        file.write("\n")
    return dataset, summary, ranked


if __name__ == "__main__":
    from consumer_intelligence.causal_uplift import run_causal_uplift as run_module

    _, report, _ = run_module()
    print(json.dumps({key: report[key] for key in (
        "analysis_type", "training_arm_counts", "holdout_arms",
        "holdout_observed_retention_difference", "ranked_evaluation", "top_ranked_group"
    )}, indent=2))

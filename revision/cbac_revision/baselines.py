"""Leakage-resistant analytic baselines for held-out perturbation profiles."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import canonical_sha256
from .errors import RevisionProtocolError

BASELINE_METHODS = (
    "zero_control_delta",
    "training_condition_mean",
    "deterministic_ridge_linear",
)


def analytic_baseline_predictions(
    training_profiles: np.ndarray,
    training_target_indicators: np.ndarray,
    held_out_target_indicator: np.ndarray,
    *,
    ridge_alpha: float = 1.0,
) -> dict[str, np.ndarray]:
    """Predict a held-out profile without accepting its outcome as an input."""

    profiles = np.asarray(training_profiles, dtype=np.float64)
    indicators = np.asarray(training_target_indicators, dtype=np.float64)
    held_out = np.asarray(held_out_target_indicator, dtype=np.float64)
    if profiles.ndim != 2 or indicators.ndim != 2 or held_out.ndim != 1:
        raise RevisionProtocolError("Baseline arrays have invalid dimensions")
    if profiles.shape != indicators.shape or profiles.shape[1] != len(held_out):
        raise RevisionProtocolError("Baseline profiles and target indicators must share shape")
    if len(profiles) < 1 or not np.isfinite(profiles).all():
        raise RevisionProtocolError("At least one finite training profile is required")
    if not np.isin(indicators, (0.0, 1.0)).all() or not np.isin(held_out, (0.0, 1.0)).all():
        raise RevisionProtocolError("Baseline target indicators must be binary")
    if not held_out.any():
        raise RevisionProtocolError("Held-out target indicator cannot be empty")
    if isinstance(ridge_alpha, bool) or not isinstance(ridge_alpha, (int, float)):
        raise RevisionProtocolError("ridge_alpha must be numeric")
    alpha = float(ridge_alpha)
    if not np.isfinite(alpha) or alpha <= 0:
        raise RevisionProtocolError("ridge_alpha must be finite and positive")

    profile_mean = profiles.mean(axis=0)
    indicator_mean = indicators.mean(axis=0)
    centered_indicators = indicators - indicator_mean
    centered_profiles = profiles - profile_mean
    dual_system = centered_indicators @ centered_indicators.T + alpha * np.eye(len(profiles))
    dual_weights = np.linalg.solve(dual_system, centered_profiles)
    ridge_prediction = (
        held_out - indicator_mean
    ) @ centered_indicators.T @ dual_weights + profile_mean
    return {
        "zero_control_delta": np.zeros(profiles.shape[1], dtype=np.float64),
        "training_condition_mean": profile_mean,
        "deterministic_ridge_linear": ridge_prediction,
    }


def build_baseline_artifact(
    *,
    dataset: str,
    hvg: int,
    panel: str,
    condition: str,
    gene_names: Sequence[str],
    training_condition_ids: Sequence[str],
    training_profiles: np.ndarray,
    training_target_indicators: np.ndarray,
    held_out_target_indicator: np.ndarray,
    y_true: np.ndarray,
    ridge_alpha: float,
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Build a self-contained artifact that can independently reproduce all baselines."""

    genes = tuple(str(gene) for gene in gene_names)
    conditions = tuple(str(value) for value in training_condition_ids)
    profiles = np.asarray(training_profiles, dtype=np.float64)
    indicators = np.asarray(training_target_indicators, dtype=np.float64)
    held_out = np.asarray(held_out_target_indicator, dtype=np.float64)
    truth = np.asarray(y_true, dtype=np.float64)
    if len(conditions) != len(profiles) or len(set(conditions)) != len(conditions):
        raise RevisionProtocolError("Baseline training condition IDs are invalid")
    if len(genes) != hvg or truth.shape != (hvg,) or not np.isfinite(truth).all():
        raise RevisionProtocolError("Baseline truth vector is invalid")
    predictions = analytic_baseline_predictions(
        profiles,
        indicators,
        held_out,
        ridge_alpha=ridge_alpha,
    )
    metrics = {
        method: _absolute_skill(truth, prediction) for method, prediction in predictions.items()
    }
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "analytic_baseline_lopo_fold",
        "identity": {
            "dataset": dataset,
            "hvg": int(hvg),
            "panel": panel,
            "condition": condition,
        },
        "methods": list(BASELINE_METHODS),
        "ridge_alpha": float(ridge_alpha),
        "fit_scope": "training_conditions_only_leave_one_perturbation_out",
        "held_out_outcome_used_for_prediction": False,
        "gene_names": list(genes),
        "gene_names_sha256": canonical_sha256(genes),
        "training_condition_ids": list(conditions),
        "training_profiles": profiles.tolist(),
        "training_profiles_sha256": canonical_sha256(profiles.tolist()),
        "training_target_indicators": indicators.tolist(),
        "training_target_indicators_sha256": canonical_sha256(indicators.tolist()),
        "held_out_target_indicator": held_out.tolist(),
        "held_out_target_indicator_sha256": canonical_sha256(held_out.tolist()),
        "y_true": truth.tolist(),
        "y_true_sha256": canonical_sha256(truth.tolist()),
        "predictions": {method: values.tolist() for method, values in predictions.items()},
        "prediction_sha256": {
            method: canonical_sha256(values.tolist()) for method, values in predictions.items()
        },
        "absolute_skill": metrics,
        "input_hashes": dict(input_hashes),
        "accounting": {
            "analytic_baseline_evaluations": len(BASELINE_METHODS),
            "ridge_refits": 1,
            "included_in_neural_fit_count": False,
        },
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    return payload


def validate_baseline_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Independently reproduce predictions, metrics, and every retained-vector hash."""

    if not isinstance(payload, dict):
        raise RevisionProtocolError("Baseline artifact must be a mapping")
    unsigned = dict(payload)
    declared_hash = unsigned.pop("artifact_sha256", None)
    if declared_hash != canonical_sha256(unsigned):
        raise RevisionProtocolError("Baseline artifact self hash is invalid")
    if payload.get("schema_version") != "1.0" or payload.get("methods") != list(BASELINE_METHODS):
        raise RevisionProtocolError("Baseline artifact schema or methods are invalid")
    if payload.get("held_out_outcome_used_for_prediction") is not False:
        raise RevisionProtocolError("Held-out outcome leakage declaration is invalid")
    identity = payload.get("identity")
    if not isinstance(identity, dict) or set(identity) != {
        "dataset",
        "hvg",
        "panel",
        "condition",
    }:
        raise RevisionProtocolError("Baseline identity must be seed-independent")
    if (
        not isinstance(identity["dataset"], str)
        or not identity["dataset"]
        or isinstance(identity["hvg"], bool)
        or not isinstance(identity["hvg"], int)
        or identity["hvg"] <= 0
        or not isinstance(identity["panel"], str)
        or not identity["panel"]
        or not isinstance(identity["condition"], str)
        or not identity["condition"]
    ):
        raise RevisionProtocolError("Baseline identity values are invalid")
    if payload.get("accounting") != {
        "analytic_baseline_evaluations": 3,
        "ridge_refits": 1,
        "included_in_neural_fit_count": False,
    }:
        raise RevisionProtocolError("Baseline accounting is invalid")
    profiles = np.asarray(payload.get("training_profiles"), dtype=np.float64)
    indicators = np.asarray(payload.get("training_target_indicators"), dtype=np.float64)
    held_out = np.asarray(payload.get("held_out_target_indicator"), dtype=np.float64)
    truth = np.asarray(payload.get("y_true"), dtype=np.float64)
    genes = tuple(str(value) for value in payload.get("gene_names", ()))
    hashes = {
        "gene_names_sha256": canonical_sha256(genes),
        "training_profiles_sha256": canonical_sha256(profiles.tolist()),
        "training_target_indicators_sha256": canonical_sha256(indicators.tolist()),
        "held_out_target_indicator_sha256": canonical_sha256(held_out.tolist()),
        "y_true_sha256": canonical_sha256(truth.tolist()),
    }
    if any(payload.get(field) != observed for field, observed in hashes.items()):
        raise RevisionProtocolError("Baseline retained-vector hash mismatch")
    reproduced = analytic_baseline_predictions(
        profiles,
        indicators,
        held_out,
        ridge_alpha=payload.get("ridge_alpha"),
    )
    for method in BASELINE_METHODS:
        observed = np.asarray(payload.get("predictions", {}).get(method), dtype=np.float64)
        if observed.shape != truth.shape or not np.array_equal(observed, reproduced[method]):
            raise RevisionProtocolError(f"Baseline prediction mismatch for {method}")
        if payload.get("prediction_sha256", {}).get(method) != canonical_sha256(observed.tolist()):
            raise RevisionProtocolError(f"Baseline prediction hash mismatch for {method}")
        if payload.get("absolute_skill", {}).get(method) != _absolute_skill(truth, observed):
            raise RevisionProtocolError(f"Baseline absolute skill mismatch for {method}")
    return dict(payload)


def write_baseline_artifact(path: Path, payload: Mapping[str, Any]) -> Path:
    """Atomically write one validated gzipped baseline artifact."""

    validated = validate_baseline_artifact(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as handle:
        json.dump(validated, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)
    return path


def read_baseline_artifact(path: Path) -> dict[str, Any]:
    """Read and independently validate one gzipped baseline artifact."""

    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle, parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RevisionProtocolError(f"Invalid baseline artifact: {error}") from error
    return validate_baseline_artifact(payload)


def baseline_metric_frame(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Return long absolute-skill rows from validated baseline artifacts."""

    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = read_baseline_artifact(path)
        for method in BASELINE_METHODS:
            rows.append(
                {
                    **payload["identity"],
                    "baseline": method,
                    **payload["absolute_skill"][method],
                    "artifact_sha256": payload["artifact_sha256"],
                }
            )
    return rows


def _absolute_skill(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = truth - prediction
    truth_sd = float(np.std(truth))
    prediction_sd = float(np.std(prediction))
    if truth_sd > 0 and prediction_sd > 0:
        pearson = float(np.corrcoef(truth, prediction)[0, 1])
        pearson_status = "MEASURED"
    else:
        pearson = None
        pearson_status = "UNDEFINED_CONSTANT_VECTOR"
    return {
        "pearson_r": pearson,
        "pearson_r_status": pearson_status,
        "mse": float(np.mean(np.square(error))),
        "mae": float(np.mean(np.abs(error))),
    }


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant {value!r} is prohibited")

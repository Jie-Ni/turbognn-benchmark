"""Lossless fold-result records for diagnostics and metric regeneration."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .hashing import nfc_text, sha256_json


def _finite_vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if not vector:
        raise ValueError(f"{name} must not be empty")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{name} contains a non-finite value")
    return vector


@dataclass(frozen=True)
class FoldResult:
    """Complete outputs needed to recompute metrics and diagnose collapsed models."""

    schema_version: str
    run_key: str
    condition: str
    fold_index: int
    gene_order: tuple[str, ...]
    gene_order_hash: str
    y_true: tuple[float, ...]
    y_pred: tuple[float, ...]
    y_true_delta: tuple[float, ...]
    delta_pred: tuple[float, ...]
    training_mean: tuple[float, ...]
    training_mean_delta: tuple[float, ...]
    control_profile: tuple[float, ...]
    training_loss_by_epoch: tuple[float, ...]
    metrics: Mapping[str, float | None]
    metric_states: Mapping[str, str]
    vector_states: Mapping[str, str]
    metadata: Mapping[str, Any]
    vector_hashes: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record."""
        value = asdict(self)
        for field in (
            "gene_order",
            "y_true",
            "y_pred",
            "y_true_delta",
            "delta_pred",
            "training_mean",
            "training_mean_delta",
            "control_profile",
            "training_loss_by_epoch",
        ):
            value[field] = list(value[field])
        return value


def build_fold_result(
    *,
    run_identity: Mapping[str, Any],
    condition: str,
    fold_index: int,
    gene_order: Sequence[str],
    y_true: Sequence[float],
    y_pred: Sequence[float],
    training_mean: Sequence[float],
    control_profile: Sequence[float],
    training_loss_by_epoch: Sequence[float],
    metrics: Mapping[str, float | None],
    metric_states: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any],
) -> FoldResult:
    """Build a validated, lossless fold record with exact vector hashes."""
    canonical_genes = tuple(nfc_text(str(gene)) for gene in gene_order)
    if any(not gene for gene in canonical_genes) or len(set(canonical_genes)) != len(
        canonical_genes
    ):
        raise ValueError("gene_order must contain non-empty canonically unique identifiers")
    vectors = {
        "y_true": _finite_vector(y_true, "y_true"),
        "y_pred": _finite_vector(y_pred, "y_pred"),
        "training_mean": _finite_vector(training_mean, "training_mean"),
        "control_profile": _finite_vector(control_profile, "control_profile"),
    }
    lengths = {len(vector) for vector in vectors.values()}
    if lengths != {len(canonical_genes)}:
        raise ValueError(
            f"Expression vectors and gene order must share one length; got {lengths} "
            f"versus {len(canonical_genes)} genes"
        )
    losses = _finite_vector(training_loss_by_epoch, "training_loss_by_epoch")
    derived_vectors = {
        "y_true_delta": tuple(
            value - control
            for value, control in zip(vectors["y_true"], vectors["control_profile"], strict=True)
        ),
        "delta_pred": tuple(
            value - control
            for value, control in zip(vectors["y_pred"], vectors["control_profile"], strict=True)
        ),
        "training_mean_delta": tuple(
            value - control
            for value, control in zip(
                vectors["training_mean"], vectors["control_profile"], strict=True
            )
        ),
    }
    numeric_metrics = {
        str(name): (None if value is None else float(value)) for name, value in metrics.items()
    }
    if not all(value is None or math.isfinite(value) for value in numeric_metrics.values()):
        raise ValueError("metrics contains a non-finite value")
    states = {str(name): str(value) for name, value in (metric_states or {}).items()}
    unknown_states = set(states) - set(numeric_metrics)
    if unknown_states:
        raise ValueError(f"metric_states contains unknown metrics: {sorted(unknown_states)}")
    for name, value in numeric_metrics.items():
        state = states.setdefault(name, "valid" if value is not None else "")
        if value is None and (not state or state == "valid"):
            raise ValueError(f"Undefined metric {name!r} requires an explicit invalid state")
        if value is not None and state != "valid":
            raise ValueError(f"Invalid metric {name!r} must be stored as JSON null")
    if any(not name or not value for name, value in states.items()):
        raise ValueError("metric_states contains a blank name or state")

    def population_sd(values: tuple[float, ...]) -> float:
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))

    vector_states = {
        "y_true_delta": (
            "valid" if population_sd(derived_vectors["y_true_delta"]) > 1e-8 else "constant_truth"
        ),
        "delta_pred": (
            "valid"
            if population_sd(derived_vectors["delta_pred"]) > 1e-8
            else "constant_prediction"
        ),
    }
    run_key = sha256_json(dict(run_identity))
    vector_hashes = {
        name: sha256_json(vector) for name, vector in {**vectors, **derived_vectors}.items()
    }
    return FoldResult(
        schema_version="1.0.0",
        run_key=run_key,
        condition=condition,
        fold_index=fold_index,
        gene_order=canonical_genes,
        gene_order_hash=sha256_json(list(canonical_genes)),
        y_true=vectors["y_true"],
        y_pred=vectors["y_pred"],
        y_true_delta=derived_vectors["y_true_delta"],
        delta_pred=derived_vectors["delta_pred"],
        training_mean=vectors["training_mean"],
        training_mean_delta=derived_vectors["training_mean_delta"],
        control_profile=vectors["control_profile"],
        training_loss_by_epoch=losses,
        metrics=numeric_metrics,
        metric_states=states,
        vector_states=vector_states,
        metadata=dict(metadata),
        vector_hashes=vector_hashes,
    )

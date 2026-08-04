#!/usr/bin/env python
"""Strict condition-level and canonical primary analysis for the TurboGNN revision.

Revision records are accepted only when their immutable input, preprocessing, graph,
model, split, environment, and code identities are complete. Invalid correlations are
kept in the support ledger but never interpreted as numerical zero. The singular primary
estimand is the STRING-GO-union GAT minus the self-loop GAT on exact seed-matched,
three-scale complete conditions with equal weighting at every prespecified level.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from merge_all_seeds import parse_split_file
from turbognn_audit.hashing import (
    canonical_json,
    nfc_text,
    sha256_file,
    sha256_json,
    utf8_sort,
)
from turbognn_audit.inference import (
    MonteCarloTest,
    centered_residual_q_test,
    dataset_sign_flip_holm_family,
)
from turbognn_audit.io import write_json_atomic
from turbognn_audit.lineage import LineageValidationError, validate_chunk_lineage

PRIMARY_TREATMENT = "string_go_union"
PRIMARY_COMPARATOR = "self_loop_gat"
LEGACY_COMPARATOR = "legacy_no_graph_transformer"
REQUIRED_SEEDS = (42, 43, 44)
FROZEN_SCALES = (200, 500, 1000)
CANONICAL_DATASETS = ("norman", "adamson", "replogle_k562", "replogle_rpe1")
FISHER_CLIP_EPSILON = 1e-7
CANONICAL_BOOTSTRAP_REPLICATES = 10_000
CANONICAL_BOOTSTRAP_SEED = 20_260_804
G3_BOOTSTRAP_SEED = 20_260_811
MINIMUM_COMPLETE_CONDITIONS = 45

GRAPH_LABEL_ALIASES = {
    "self_loop": PRIMARY_COMPARATOR,
    "self_loop_gat": PRIMARY_COMPARATOR,
    "no_graph": LEGACY_COMPARATOR,
    "combined": "legacy_combined_with_coexpression",
    LEGACY_COMPARATOR: LEGACY_COMPARATOR,
}

PAIR_SCOPE = (
    "dataset",
    "hvg",
    "input_hash",
    "environment_lock_hash",
    "gene_panel_hash",
    "condition_panel_hash",
    "preprocessing_hash",
    "planned_condition_list_hash",
    "planned_conditions_json",
    "code_commit",
    "split_hash",
    "condition",
    "metric",
)
ARM_FIELDS = ("graph_instance", "graph_hash", "model_name", "model_config_hash")
RAW_KEY = (
    *PAIR_SCOPE,
    "graph_type",
    *ARM_FIELDS,
    "seed",
    "fold_seed",
)
SUMMARY_SCOPE = (
    "dataset",
    "hvg",
    "input_hash",
    "environment_lock_hash",
    "gene_panel_hash",
    "condition_panel_hash",
    "preprocessing_hash",
    "planned_condition_list_hash",
    "planned_conditions_json",
    "code_commit",
    "metric",
    "treatment_graph_instance",
    "treatment_graph_hash",
    "treatment_model_name",
    "treatment_model_config_hash",
    "comparator_graph_instance",
    "comparator_graph_hash",
    "comparator_model_name",
    "comparator_model_config_hash",
)


class StatisticalValidationError(ValueError):
    """Raised when inputs cannot support an unambiguous prespecified analysis."""


@dataclass(frozen=True)
class ComparisonSpec:
    """One frozen treatment/comparator contrast and its evidentiary class."""

    treatment: str
    comparator: str
    comparison_class: str


@dataclass(frozen=True)
class PrimaryAnalysis:
    """Machine-readable source tables for the canonical primary and G3 estimands."""

    seed_pairs: pd.DataFrame
    support_ledger: pd.DataFrame
    condition_scale: pd.DataFrame
    condition_estimates: pd.DataFrame
    dataset_estimates: pd.DataFrame
    primary_summary: pd.DataFrame
    resolution_summary: pd.DataFrame
    g3_summary: pd.DataFrame
    bootstrap_distribution: pd.DataFrame


def canonical_graph_label(label: str) -> str:
    """Normalize labels without treating the legacy Transformer as a no-prior GAT."""
    normalized = nfc_text(label)
    return GRAPH_LABEL_ALIASES.get(normalized, normalized)


def comparison_specs(graph_labels: Iterable[str]) -> tuple[ComparisonSpec, ...]:
    """Return frozen contrast families without promoting sensitivities to primary."""
    available = {canonical_graph_label(label) for label in graph_labels}
    specs: list[ComparisonSpec] = []
    if PRIMARY_COMPARATOR in available:
        if PRIMARY_TREATMENT in available:
            specs.append(
                ComparisonSpec(
                    PRIMARY_TREATMENT,
                    PRIMARY_COMPARATOR,
                    "singular_primary",
                )
            )
        for treatment in ("string_ppi", "gene_ontology"):
            if treatment in available:
                specs.append(
                    ComparisonSpec(treatment, PRIMARY_COMPARATOR, "holm_secondary_graph_family")
                )
        for treatment in ("coexpression", "barabasi_albert", "degree_preserving_rewired"):
            if treatment in available:
                specs.append(ComparisonSpec(treatment, PRIMARY_COMPARATOR, "sensitivity_only"))
    if LEGACY_COMPARATOR in available:
        for treatment in (
            PRIMARY_TREATMENT,
            "string_ppi",
            "gene_ontology",
            "coexpression",
        ):
            if treatment in available:
                specs.append(
                    ComparisonSpec(treatment, LEGACY_COMPARATOR, "legacy_architecture_secondary")
                )
    return tuple(specs)


def _metric_value(fold: Mapping[str, Any], metric: str) -> float | None:
    nested = fold.get("metrics")
    if not isinstance(nested, Mapping) or metric not in nested:
        raise StatisticalValidationError(
            f"Fold {fold.get('condition')!r} lacks revision-schema metric {metric!r}"
        )
    raw_value = nested[metric]
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        raise StatisticalValidationError(
            f"Fold {fold.get('condition')!r} has a Boolean metric {metric!r}"
        )
    value = float(raw_value)
    if not math.isfinite(value):
        raise StatisticalValidationError(
            f"Fold {fold.get('condition')!r} has non-finite metric {metric!r}"
        )
    return value


def _metric_state(fold: Mapping[str, Any], metric: str) -> str:
    states = fold.get("metric_states")
    if not isinstance(states, Mapping) or not isinstance(states.get(metric), str):
        raise StatisticalValidationError(
            f"Fold {fold.get('condition')!r} lacks explicit metric state for {metric!r}"
        )
    return str(states[metric])


def _validated_pearson_from_vectors(
    fold: Mapping[str, Any],
) -> tuple[float | None, dict[str, str]]:
    condition = str(fold.get("condition"))
    genes = fold.get("gene_order")
    if not isinstance(genes, list) or not genes:
        raise StatisticalValidationError(f"Fold {condition!r} lacks the ordered gene identifiers")
    canonical_genes = [nfc_text(str(gene)) for gene in genes]
    if any(not gene for gene in canonical_genes) or len(set(canonical_genes)) != len(
        canonical_genes
    ):
        raise StatisticalValidationError(f"Fold {condition!r} has an invalid canonical gene order")
    gene_order_hash = fold.get("gene_order_hash")
    if gene_order_hash != sha256_json(canonical_genes):
        raise StatisticalValidationError(f"Fold {condition!r} gene_order_hash mismatch")
    vector_hashes = fold.get("vector_hashes")
    if not isinstance(vector_hashes, Mapping):
        raise StatisticalValidationError(f"Fold {condition!r} lacks vector_hashes")
    vectors: dict[str, np.ndarray] = {}
    for name in ("y_true", "y_pred", "training_mean", "control_profile"):
        value = fold.get(name)
        if not isinstance(value, list) or len(value) != len(canonical_genes):
            raise StatisticalValidationError(
                f"Fold {condition!r} vector {name!r} does not match gene order"
            )
        vector = np.asarray(value, dtype=float)
        if not np.isfinite(vector).all() or vector_hashes.get(name) != sha256_json(value):
            raise StatisticalValidationError(
                f"Fold {condition!r} vector {name!r} is non-finite or hash-mismatched"
            )
        vectors[name] = vector
    delta_true = vectors["y_true"] - vectors["control_profile"]
    delta_pred = vectors["y_pred"] - vectors["control_profile"]
    derived = {
        "y_true_delta": delta_true,
        "delta_pred": delta_pred,
        "training_mean_delta": vectors["training_mean"] - vectors["control_profile"],
    }
    for name, expected in derived.items():
        raw = fold.get(name)
        if not isinstance(raw, list) or len(raw) != len(canonical_genes):
            raise StatisticalValidationError(f"Fold {condition!r} lacks explicit vector {name!r}")
        observed = np.asarray(raw, dtype=float)
        if (
            not np.isfinite(observed).all()
            or not np.array_equal(observed, expected)
            or vector_hashes.get(name) != sha256_json(raw)
        ):
            raise StatisticalValidationError(
                f"Fold {condition!r} explicit vector {name!r} is inconsistent or hash-mismatched"
            )
    vector_states = fold.get("vector_states")
    expected_vector_states = {
        "y_true_delta": ("valid" if np.std(delta_true, ddof=0) > 1e-8 else "constant_truth"),
        "delta_pred": ("valid" if np.std(delta_pred, ddof=0) > 1e-8 else "constant_prediction"),
    }
    if vector_states != expected_vector_states:
        raise StatisticalValidationError(f"Fold {condition!r} vector states are inconsistent")
    truth_valid = bool(np.std(delta_true, ddof=0) > 1e-8)
    prediction_valid = bool(np.std(delta_pred, ddof=0) > 1e-8)
    if truth_valid and prediction_valid:
        state = "valid"
        value = float(stats.pearsonr(delta_true, delta_pred).statistic)
    elif not truth_valid and not prediction_valid:
        state = "constant_truth_and_prediction"
        value = None
    elif not truth_valid:
        state = "constant_truth"
        value = None
    else:
        state = "constant_prediction"
        value = None
    if value is not None and not math.isfinite(value):
        raise StatisticalValidationError(f"Fold {condition!r} recomputed Pearson r is non-finite")
    stored_state = _metric_state(fold, "pearson_r")
    stored_value = _metric_value(fold, "pearson_r")
    values_agree = (
        stored_value is None
        if value is None
        else stored_value is not None
        and math.isclose(stored_value, value, rel_tol=0.0, abs_tol=1e-10)
    )
    if stored_state != state or not values_agree:
        raise StatisticalValidationError(
            f"Fold {condition!r} stored Pearson metric/state disagrees with persisted vectors"
        )
    hashes = {
        "gene_order_hash": str(gene_order_hash),
        "gene_order_json": canonical_json(canonical_genes),
        "y_true_hash": str(vector_hashes["y_true"]),
        "training_mean_hash": str(vector_hashes["training_mean"]),
        "control_profile_hash": str(vector_hashes["control_profile"]),
    }
    return value, hashes


def _metadata(fold: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = fold.get("metadata")
    if not isinstance(metadata, Mapping):
        raise StatisticalValidationError(f"Fold {fold.get('condition')!r} lacks revision metadata")
    return metadata


def _required_text(metadata: Mapping[str, Any], name: str, condition: str) -> str:
    value = metadata.get(name)
    if not isinstance(value, str) or not value:
        raise StatisticalValidationError(
            f"Fold {condition!r} lacks required metadata field {name!r}"
        )
    return value


def _planned_conditions(metadata: Mapping[str, Any], condition: str) -> tuple[str, ...]:
    values = metadata.get("evaluated_conditions")
    if not isinstance(values, list) or not values:
        raise StatisticalValidationError(
            f"Fold {condition!r} lacks the frozen evaluated_conditions panel"
        )
    planned = tuple(nfc_text(str(value)) for value in values)
    if any(not value for value in planned) or len(set(planned)) != len(planned):
        raise StatisticalValidationError(
            f"Fold {condition!r} has a blank or duplicate planned condition"
        )
    if condition not in planned:
        raise StatisticalValidationError(
            f"Observed condition {condition!r} is outside its frozen condition panel"
        )
    return planned


def _hvg_from_directory(path: Path) -> int:
    match = re.fullmatch(r"hvg(?P<hvg>\d+)", path.parent.name)
    if match is None:
        raise StatisticalValidationError(
            f"Aggregate must be stored under an hvg<N> directory: {path}"
        )
    return int(match.group("hvg"))


def aggregate_files(results_root: Path) -> list[Path]:
    """Find strict aggregate JSONs while excluding every seed/fold task file."""
    files: list[Path] = []
    for path in sorted(results_root.glob("hvg*/*.json")):
        if parse_split_file(path) is not None or path.name.startswith("results_"):
            continue
        if path.name.endswith(".manifest.json"):
            continue
        with path.open(encoding="utf-8") as handle:
            header = json.load(handle)
        if isinstance(header, Mapping) and {"dataset", "graph_type", "seeds"} <= set(header):
            files.append(path)
    if not files:
        raise StatisticalValidationError(
            f"No strict aggregate result files found below {results_root}"
        )
    return files


def _validate_aggregate_lineage(path: Path, data: Mapping[str, Any]) -> None:
    provenance = data.get("merge_provenance")
    if not isinstance(provenance, Mapping) or not isinstance(
        provenance.get("source_manifests"), list
    ):
        raise StatisticalValidationError(f"Aggregate {path} lacks source manifest lineage")
    declared = provenance["source_manifests"]
    if not declared:
        raise StatisticalValidationError(f"Aggregate {path} has empty source manifest lineage")
    expected_keys: set[str] = set()
    seen_sources: set[str] = set()
    for entry in declared:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("result_file"), str):
            raise StatisticalValidationError(f"Aggregate {path} has malformed lineage entry")
        source_name = str(entry["result_file"])
        if source_name in seen_sources:
            raise StatisticalValidationError(f"Aggregate {path} repeats source {source_name!r}")
        seen_sources.add(source_name)
        source_path = path.parent / source_name
        if not source_path.is_file():
            raise StatisticalValidationError(f"Aggregate source result is missing: {source_path}")
        with source_path.open(encoding="utf-8") as handle:
            source_data = json.load(handle)
        if not isinstance(source_data, Mapping):
            raise StatisticalValidationError(f"Aggregate source is not an object: {source_path}")
        try:
            observed = validate_chunk_lineage(source_path, source_data).as_dict()
        except (LineageValidationError, ValueError) as error:
            raise StatisticalValidationError(
                f"Aggregate source lineage failed for {source_path}: {error}"
            ) from error
        if canonical_json(dict(entry)) != canonical_json(observed):
            raise StatisticalValidationError(
                f"Aggregate lineage entry drifted from source manifest: {source_path}"
            )
        expected_keys.update(str(value) for value in observed["succeeded_run_keys"])
    seeds = data.get("seeds")
    if not isinstance(seeds, Mapping):
        raise StatisticalValidationError(f"Aggregate {path} lacks a seeds object")
    actual_keys: list[str] = []
    for seed_data in seeds.values():
        if not isinstance(seed_data, Mapping) or not isinstance(seed_data.get("folds"), list):
            raise StatisticalValidationError(f"Aggregate {path} has malformed fold records")
        actual_keys.extend(str(fold.get("run_key")) for fold in seed_data["folds"])
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys:
        raise StatisticalValidationError(
            f"Aggregate {path} does not conserve terminal succeeded manifest records"
        )


def load_fold_data(results_root: Path, metric: str = "pearson_r") -> pd.DataFrame:
    """Load revision folds; missing identities or metric states fail immediately."""
    rows: list[dict[str, Any]] = []
    validated_feature_artifacts: set[tuple[Path, str, str]] = set()
    for path in aggregate_files(results_root):
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, Mapping):
            raise StatisticalValidationError(f"Aggregate is not a JSON object: {path}")
        _validate_aggregate_lineage(path, data)
        dataset = str(data["dataset"])
        raw_graph_type = str(data["graph_type"])
        graph_type = canonical_graph_label(raw_graph_type)
        hvg = _hvg_from_directory(path)
        seeds = data.get("seeds")
        if not isinstance(seeds, Mapping):
            raise StatisticalValidationError(f"Missing seeds object in {path}")
        for seed_key, seed_data in seeds.items():
            if not isinstance(seed_data, Mapping):
                raise StatisticalValidationError(f"Invalid seed record {seed_key!r} in {path}")
            try:
                seed = int(str(seed_key).removeprefix("seed_"))
            except ValueError as error:
                raise StatisticalValidationError(
                    f"Invalid seed key {seed_key!r} in {path}"
                ) from error
            folds = seed_data.get("folds")
            if not isinstance(folds, list):
                raise StatisticalValidationError(f"Missing folds array for {seed_key!r} in {path}")
            for fold in folds:
                if not isinstance(fold, Mapping):
                    raise StatisticalValidationError(f"Non-object fold in {path}")
                raw_condition = fold.get("condition")
                if not isinstance(raw_condition, str) or not raw_condition:
                    raise StatisticalValidationError(f"Fold without condition in {path}")
                condition = nfc_text(raw_condition)
                metadata = _metadata(fold)
                delta_true = np.asarray(fold["y_true_delta"], dtype=float)
                delta_pred = np.asarray(fold["delta_pred"], dtype=float)
                expected_diagnostics = {
                    "delta_true_sd_ddof0": float(np.std(delta_true, ddof=0)),
                    "delta_pred_sd_ddof0": float(np.std(delta_pred, ddof=0)),
                    "delta_true_l2_norm": float(np.linalg.norm(delta_true)),
                    "delta_pred_l2_norm": float(np.linalg.norm(delta_pred)),
                }
                for diagnostic, expected in expected_diagnostics.items():
                    observed = metadata.get(diagnostic)
                    if (
                        isinstance(observed, bool)
                        or not isinstance(observed, (int, float))
                        or not math.isfinite(float(observed))
                        or not math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-10)
                    ):
                        raise StatisticalValidationError(
                            f"Fold {condition!r} has invalid {diagnostic!r}"
                        )
                metadata_seed = metadata.get("seed")
                if metadata_seed != seed:
                    raise StatisticalValidationError(
                        f"Seed key {seed} disagrees with fold metadata {metadata_seed!r} in {path}"
                    )
                graph = metadata.get("graph")
                if not isinstance(graph, Mapping):
                    raise StatisticalValidationError(f"Fold {condition!r} lacks graph provenance")
                graph_instance = graph.get("graph_instance")
                graph_hash = graph.get("edge_hash")
                if not isinstance(graph_instance, str) or not graph_instance:
                    raise StatisticalValidationError(
                        f"Fold {condition!r} lacks graph_instance provenance"
                    )
                if not isinstance(graph_hash, str) or not graph_hash:
                    raise StatisticalValidationError(f"Fold {condition!r} lacks graph edge hash")
                metadata_graph_type = graph.get("graph_type")
                if isinstance(metadata_graph_type, str):
                    if canonical_graph_label(metadata_graph_type) != graph_type:
                        raise StatisticalValidationError(
                            f"Graph label mismatch in {path}: {raw_graph_type!r} versus "
                            f"{metadata_graph_type!r}"
                        )
                planned = _planned_conditions(metadata, condition)
                fold_seed = metadata.get("fold_seed")
                fold_index = fold.get("fold_index")
                run_key = fold.get("run_key")
                if not isinstance(fold_seed, int) or not 0 <= fold_seed <= 2**32 - 1:
                    raise StatisticalValidationError(f"Fold {condition!r} lacks a valid fold_seed")
                if not isinstance(fold_index, int) or fold_index < 0:
                    raise StatisticalValidationError(f"Fold {condition!r} lacks a valid fold_index")
                if not isinstance(run_key, str) or re.fullmatch(r"[0-9a-f]{64}", run_key) is None:
                    raise StatisticalValidationError(f"Fold {condition!r} lacks a valid run_key")
                vector_identity: dict[str, str] = {}
                if metric == "pearson_r":
                    metric_value, vector_identity = _validated_pearson_from_vectors(fold)
                else:
                    metric_value = _metric_value(fold, metric)
                target_mask = metadata.get("target_mask")
                if (
                    not isinstance(target_mask, list)
                    or len(target_mask) != len(json.loads(vector_identity["gene_order_json"]))
                    or any(not isinstance(value, bool) for value in target_mask)
                    or metadata.get("target_mask_hash") != sha256_json(target_mask)
                ):
                    raise StatisticalValidationError(
                        f"Fold {condition!r} has an invalid target-mask lineage"
                    )
                artifact_relative = _required_text(
                    metadata, "initial_raw_feature_artifact", condition
                )
                artifact_sha256 = _required_text(
                    metadata, "initial_raw_feature_artifact_sha256", condition
                )
                raw_feature_hash = _required_text(metadata, "initial_raw_feature_hash", condition)
                relative_path = Path(artifact_relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise StatisticalValidationError(
                        f"Fold {condition!r} has an unsafe feature-artifact path"
                    )
                artifact_path = (path.parent / relative_path).resolve()
                cache_key = (artifact_path, artifact_sha256, raw_feature_hash)
                if cache_key not in validated_feature_artifacts:
                    if not artifact_path.is_file() or sha256_file(artifact_path) != artifact_sha256:
                        raise StatisticalValidationError(
                            f"Fold {condition!r} feature artifact is missing or hash-mismatched"
                        )
                    feature_array = np.load(artifact_path, allow_pickle=False)
                    logical_hash = sha256_json(
                        {
                            "dtype": "float32",
                            "shape": list(feature_array.shape),
                            "values": feature_array.tolist(),
                        }
                    )
                    if (
                        feature_array.dtype != np.float32
                        or feature_array.ndim != 2
                        or feature_array.shape[1] != 68
                        or not np.isfinite(feature_array).all()
                        or logical_hash != raw_feature_hash
                    ):
                        raise StatisticalValidationError(
                            f"Fold {condition!r} feature artifact violates TDS-41"
                        )
                    validated_feature_artifacts.add(cache_key)
                rows.append(
                    {
                        "dataset": dataset,
                        "execution_stage": _required_text(metadata, "execution_stage", condition),
                        "configuration_universe_hash": _required_text(
                            metadata, "configuration_universe_hash", condition
                        ),
                        "hvg": hvg,
                        "input_hash": _required_text(metadata, "input_hash", condition),
                        "dataset_passport_hash": _required_text(
                            metadata, "dataset_passport_hash", condition
                        ),
                        "dataset_passport_registry_hash": _required_text(
                            metadata, "dataset_passport_registry_hash", condition
                        ),
                        "matrix_contract_hash": _required_text(
                            metadata, "matrix_contract_hash", condition
                        ),
                        "environment_lock_hash": _required_text(
                            metadata, "environment_lock_hash", condition
                        ),
                        "gene_panel_hash": _required_text(metadata, "gene_panel_hash", condition),
                        "condition_panel_hash": _required_text(
                            metadata, "condition_panel_hash", condition
                        ),
                        "preprocessing_hash": _required_text(
                            metadata, "preprocessing_hash", condition
                        ),
                        "control_selection_hash": _required_text(
                            metadata, "control_selection_hash", condition
                        ),
                        "condition_eligibility_ledger_hash": _required_text(
                            metadata, "condition_eligibility_ledger_hash", condition
                        ),
                        "cell_qc_policy_hash": _required_text(
                            metadata, "cell_qc_policy_hash", condition
                        ),
                        "response_definition_hash": _required_text(
                            metadata, "response_definition_hash", condition
                        ),
                        "target_mapping_hash": _required_text(
                            metadata, "target_mapping_hash", condition
                        ),
                        "planned_condition_list_hash": sha256_json(list(planned)),
                        "planned_conditions_json": canonical_json(list(planned)),
                        "graph_type": graph_type,
                        "graph_type_raw": raw_graph_type,
                        "graph_instance": graph_instance,
                        "graph_hash": graph_hash,
                        "graph_parent_edge_hash": graph.get("parent_edge_hash"),
                        "graph_instance_seed": graph.get("seed"),
                        "graph_ensemble_config_hash": graph.get("graph_ensemble_config_hash"),
                        "model_name": _required_text(metadata, "model_name", condition),
                        "model_config_hash": _required_text(
                            metadata, "model_config_hash", condition
                        ),
                        "initialization_hash": _required_text(
                            metadata, "initialization_hash", condition
                        ),
                        "target_mask_hash": _required_text(metadata, "target_mask_hash", condition),
                        "initial_perturbation_context_hash": _required_text(
                            metadata, "initial_perturbation_context_hash", condition
                        ),
                        "initial_raw_feature_hash": _required_text(
                            metadata, "initial_raw_feature_hash", condition
                        ),
                        "initial_raw_feature_artifact": artifact_relative,
                        "initial_raw_feature_artifact_sha256": artifact_sha256,
                        "baseline_feature_hash": _required_text(
                            metadata, "baseline_feature_hash", condition
                        ),
                        "input_definition_hash": _required_text(
                            metadata, "input_definition_hash", condition
                        ),
                        "input_contract_policy_hash": _required_text(
                            metadata, "input_contract_policy_hash", condition
                        ),
                        "input_instance_hash": _required_text(
                            metadata, "input_instance_hash", condition
                        ),
                        "input_contract_hash": _required_text(
                            metadata, "input_contract_hash", condition
                        ),
                        "code_commit": _required_text(metadata, "code_commit", condition),
                        "split_hash": _required_text(metadata, "split_hash", condition),
                        "training_condition_list_hash": _required_text(
                            metadata, "training_condition_list_hash", condition
                        ),
                        "fold_order_seed_policy_hash": _required_text(
                            metadata, "fold_order_seed_policy_hash", condition
                        ),
                        "seed": seed,
                        "fold_seed": fold_seed,
                        "condition": condition,
                        "fold_index": fold_index,
                        "run_key": run_key,
                        **vector_identity,
                        "value": metric_value,
                        "metric_state": _metric_state(fold, metric),
                        "metric": metric,
                        "source_file": path.name,
                    }
                )
    frame = pd.DataFrame(rows)
    validate_raw_fold_data(frame)
    return frame


def validate_raw_fold_data(frame: pd.DataFrame) -> None:
    """Reject duplicate run identities, malformed panels, and non-finite outcomes."""
    required = {*RAW_KEY, "run_key", "fold_index", "value", "metric_state"}
    missing = required - set(frame.columns)
    if missing:
        raise StatisticalValidationError(f"Fold table lacks columns: {sorted(missing)}")
    if frame.empty:
        raise StatisticalValidationError("Fold table is empty")
    duplicate_run_key = frame.duplicated("run_key", keep=False)
    if duplicate_run_key.any():
        examples = frame.loc[duplicate_run_key, "run_key"].head(5).tolist()
        raise StatisticalValidationError(f"Duplicate run_key values: {examples}")
    duplicate_mask = frame.duplicated(list(RAW_KEY), keep=False)
    if duplicate_mask.any():
        examples = frame.loc[duplicate_mask, list(RAW_KEY)].head(5).to_dict("records")
        raise StatisticalValidationError(f"Duplicate raw fold keys: {examples}")
    valid_mask = frame["metric_state"] == "valid"
    valid_values = pd.to_numeric(frame.loc[valid_mask, "value"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(valid_values).all():
        raise StatisticalValidationError("Valid fold rows must contain finite metric values")
    invalid_values = frame.loc[~valid_mask, "value"]
    if not invalid_values.isna().all():
        raise StatisticalValidationError("Invalid fold rows must store the metric as JSON null")
    for row in frame.itertuples():
        planned = json.loads(row.planned_conditions_json)
        if not isinstance(planned, list) or row.condition not in planned:
            raise StatisticalValidationError(
                f"Condition {row.condition!r} is absent from its declared panel"
            )
        if sha256_json(planned) != row.planned_condition_list_hash:
            raise StatisticalValidationError("Planned-condition list hash mismatch")


def canonicalize_analysis_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize statistical identifiers and return a canonical row ordering."""
    canonical = frame.copy()
    for column in ("dataset", "condition", "graph_type", "graph_instance", "model_name"):
        if column in canonical:
            canonical[column] = canonical[column].map(lambda value: nfc_text(str(value)))
    if "planned_conditions_json" in canonical:
        normalized_panels = [
            [nfc_text(str(condition)) for condition in json.loads(value)]
            for value in canonical["planned_conditions_json"]
        ]
        canonical["planned_conditions_json"] = [
            canonical_json(panel) for panel in normalized_panels
        ]
        canonical["planned_condition_list_hash"] = [
            sha256_json(panel) for panel in normalized_panels
        ]
    sort_columns = [
        column
        for column in (
            "dataset",
            "hvg",
            "condition",
            "graph_type",
            "graph_instance",
            "model_name",
            "seed",
        )
        if column in canonical
    ]
    return canonical.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def _fisher_z(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -1.0 + FISHER_CLIP_EPSILON, 1.0 - FISHER_CLIP_EPSILON)
    return np.arctanh(clipped)


def aggregate_seeds_within_condition(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate valid seeds within one immutable graph instance and condition."""
    frame = canonicalize_analysis_frame(frame)
    validate_raw_fold_data(frame)
    valid = frame.loc[frame["metric_state"] == "valid"].copy()
    if valid.empty:
        return pd.DataFrame()
    valid["fisher_z"] = np.where(
        valid["metric"] == "pearson_r",
        _fisher_z(valid["value"].to_numpy(dtype=float)),
        valid["value"].to_numpy(dtype=float),
    )
    group_columns = [*PAIR_SCOPE, "graph_type", *ARM_FIELDS]
    ordered = valid.sort_values([*group_columns, "seed"], kind="mergesort")
    ordered["seed_value_pair"] = [
        (int(seed), int(fold_seed), float(value), float(fisher_z))
        for seed, fold_seed, value, fisher_z in zip(
            ordered["seed"],
            ordered["fold_seed"],
            ordered["value"],
            ordered["fisher_z"],
            strict=True,
        )
    ]
    condition_level = (
        ordered.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            value=("value", "mean"),
            fisher_z_value=("fisher_z", "mean"),
            seed_run_n=("seed", "nunique"),
            seed_set=(
                "seed",
                lambda values: canonical_json(sorted(int(value) for value in set(values))),
            ),
            seed_values=(
                "seed_value_pair",
                lambda values: canonical_json(
                    [
                        {
                            "seed": seed,
                            "fold_seed": fold_seed,
                            "value": value,
                            "fisher_z": fisher_z,
                        }
                        for seed, fold_seed, value, fisher_z in values
                    ]
                ),
            ),
            seed_fold_seeds=(
                "seed_value_pair",
                lambda values: canonical_json(
                    [
                        {"seed": seed, "fold_seed": fold_seed}
                        for seed, fold_seed, _value, _fisher_z in values
                    ]
                ),
            ),
        )
        .sort_values(group_columns, kind="mergesort")
        .reset_index(drop=True)
    )
    return condition_level


def pair_condition_values(
    condition_level: pd.DataFrame,
    spec: ComparisonSpec,
) -> pd.DataFrame:
    """Pair graph arms on exact condition identity and require identical seed sets."""
    if condition_level.empty:
        return pd.DataFrame()
    keys = list(PAIR_SCOPE)
    arm_columns = [
        *ARM_FIELDS,
        "value",
        "fisher_z_value",
        "seed_run_n",
        "seed_set",
        "seed_fold_seeds",
    ]
    treatment = condition_level.loc[
        condition_level["graph_type"] == spec.treatment,
        [*keys, *arm_columns],
    ].rename(columns={name: f"treatment_{name}" for name in arm_columns})
    comparator = condition_level.loc[
        condition_level["graph_type"] == spec.comparator,
        [*keys, *arm_columns],
    ].rename(columns={name: f"comparator_{name}" for name in arm_columns})
    treatment_identity = [
        *keys,
        "treatment_graph_instance",
        "treatment_graph_hash",
        "treatment_model_name",
        "treatment_model_config_hash",
    ]
    if treatment.duplicated(treatment_identity).any():
        raise StatisticalValidationError(
            f"Treatment rows are not unique for graph instances in {spec.treatment}"
        )
    if comparator.duplicated(keys).any():
        raise StatisticalValidationError(
            f"Comparator {spec.comparator!r} has multiple instances/configurations per condition"
        )
    paired = treatment.merge(comparator, on=keys, how="inner", validate="many_to_one")
    if paired.empty:
        return paired
    seed_mismatch = paired["treatment_seed_set"] != paired["comparator_seed_set"]
    if seed_mismatch.any():
        example = paired.loc[
            seed_mismatch,
            ["dataset", "hvg", "condition", "treatment_seed_set", "comparator_seed_set"],
        ].iloc[0]
        raise StatisticalValidationError(
            f"Treatment/comparator seed sets differ: {example.to_dict()}"
        )
    fold_seed_mismatch = paired["treatment_seed_fold_seeds"] != paired["comparator_seed_fold_seeds"]
    if fold_seed_mismatch.any():
        raise StatisticalValidationError("Treatment/comparator fold-seed identities differ")
    if spec.comparison_class != "legacy_architecture_secondary":
        architecture_mismatch = (
            paired["treatment_model_name"] != paired["comparator_model_name"]
        ) | (paired["treatment_model_config_hash"] != paired["comparator_model_config_hash"])
        if architecture_mismatch.any():
            raise StatisticalValidationError(
                f"Same-GAT contrast {spec.treatment!r} is not architecture/config matched"
            )
    paired["treatment"] = spec.treatment
    paired["comparator"] = spec.comparator
    paired["comparison_class"] = spec.comparison_class
    paired["difference"] = paired["treatment_value"] - paired["comparator_value"]
    paired["difference_fisher_z"] = (
        paired["treatment_fisher_z_value"] - paired["comparator_fisher_z_value"]
    )
    return paired.sort_values(treatment_identity, kind="mergesort").reset_index(drop=True)


def _bootstrap_mean_difference(
    differences: np.ndarray,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[float, float]:
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(differences), size=(bootstrap_replicates, len(differences)))
    means = differences[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975], method="linear")
    return float(low), float(high)


def _coverage_counts(
    condition_level: pd.DataFrame,
    scope_pairs: pd.DataFrame,
    spec: ComparisonSpec,
) -> dict[str, float | int | str]:
    first = scope_pairs.iloc[0]
    planned = json.loads(first["planned_conditions_json"])
    planned_set = set(str(condition) for condition in planned)
    scoped = condition_level
    for name in (
        "dataset",
        "hvg",
        "input_hash",
        "environment_lock_hash",
        "gene_panel_hash",
        "condition_panel_hash",
        "preprocessing_hash",
        "planned_condition_list_hash",
        "code_commit",
        "metric",
    ):
        scoped = scoped.loc[scoped[name] == first[name]]
    treatment = scoped.loc[
        (scoped["graph_type"] == spec.treatment)
        & (scoped["graph_instance"] == first["treatment_graph_instance"])
        & (scoped["graph_hash"] == first["treatment_graph_hash"])
    ]
    comparator = scoped.loc[
        (scoped["graph_type"] == spec.comparator)
        & (scoped["graph_instance"] == first["comparator_graph_instance"])
        & (scoped["graph_hash"] == first["comparator_graph_hash"])
    ]
    treatment_conditions = set(treatment["condition"].astype(str))
    comparator_conditions = set(comparator["condition"].astype(str))
    if not treatment_conditions <= planned_set or not comparator_conditions <= planned_set:
        raise StatisticalValidationError("Observed valid conditions fall outside the frozen panel")
    paired_conditions = set(scope_pairs["condition"].astype(str))
    observed_union = treatment_conditions | comparator_conditions
    planned_n = len(planned_set)
    return {
        "planned_condition_n": planned_n,
        "treatment_condition_n": len(treatment_conditions),
        "comparator_condition_n": len(comparator_conditions),
        "unique_condition_n": len(paired_conditions),
        "union_condition_n": len(observed_union),
        "coverage": len(paired_conditions) / planned_n,
        "treatment_coverage": len(treatment_conditions) / planned_n,
        "comparator_coverage": len(comparator_conditions) / planned_n,
        "coverage_basis": "frozen_evaluated_conditions_panel",
    }


def summarize_paired_comparison(
    condition_level: pd.DataFrame,
    paired: pd.DataFrame,
    spec: ComparisonSpec,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 42,
) -> pd.DataFrame:
    """Produce audit-only dataset-scale summaries; never the canonical headline estimate."""
    rows: list[dict[str, Any]] = []
    if paired.empty:
        return pd.DataFrame()
    for scope_values, scope_pairs in paired.groupby(list(SUMMARY_SCOPE), sort=True, dropna=False):
        scope = dict(zip(SUMMARY_SCOPE, scope_values, strict=True))
        analysis_column = "difference_fisher_z" if scope["metric"] == "pearson_r" else "difference"
        differences = scope_pairs[analysis_column].to_numpy(dtype=float)
        treatment_values = scope_pairs["treatment_value"].to_numpy(dtype=float)
        comparator_values = scope_pairs["comparator_value"].to_numpy(dtype=float)
        raw_differences = treatment_values - comparator_values
        arithmetic_residual = float(
            raw_differences.mean() - (treatment_values.mean() - comparator_values.mean())
        )
        if not np.isclose(arithmetic_residual, 0.0, atol=1e-12, rtol=1e-12):
            raise StatisticalValidationError(f"Paired arithmetic invariant failed for {scope}")
        ci_low, ci_high = _bootstrap_mean_difference(
            differences,
            bootstrap_replicates,
            bootstrap_seed,
        )
        if len(differences) >= 2:
            t_statistic, t_p = stats.ttest_rel(
                scope_pairs["treatment_fisher_z_value"],
                scope_pairs["comparator_fisher_z_value"],
            )
            try:
                wilcoxon_statistic, wilcoxon_p = stats.wilcoxon(differences)
            except ValueError:
                wilcoxon_statistic, wilcoxon_p = np.nan, np.nan
        else:
            t_statistic = t_p = wilcoxon_statistic = wilcoxon_p = np.nan
        coverage = _coverage_counts(condition_level, scope_pairs, spec)
        condition_keys = sorted(
            f"{row.dataset}|{row.hvg}|{row.condition}|{row.split_hash}"
            for row in scope_pairs.itertuples()
        )
        rows.append(
            {
                **scope,
                "treatment": spec.treatment,
                "comparator": spec.comparator,
                "comparison_class": spec.comparison_class,
                "analysis_role": "dataset_scale_audit_only",
                **coverage,
                "seed_run_n": int(
                    scope_pairs["treatment_seed_run_n"].sum()
                    + scope_pairs["comparator_seed_run_n"].sum()
                ),
                "mean_treatment_raw_r": float(treatment_values.mean()),
                "mean_comparator_raw_r": float(comparator_values.mean()),
                "mean_paired_raw_difference_descriptive": float(raw_differences.mean()),
                "mean_difference_analysis_scale": float(differences.mean()),
                "bootstrap_ci_low_analysis_scale": ci_low,
                "bootstrap_ci_high_analysis_scale": ci_high,
                "paired_t_statistic": float(t_statistic),
                "paired_t_p": float(t_p),
                "wilcoxon_statistic": float(wilcoxon_statistic),
                "wilcoxon_p": float(wilcoxon_p),
                "paired_arithmetic_residual": arithmetic_residual,
                "paired_arithmetic_invariant": True,
                "paired_condition_hash": sha256_json(condition_keys),
            }
        )
    return pd.DataFrame(rows)


def analyze_fold_data(
    frame: pd.DataFrame,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 42,
    require_primary: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return valid condition aggregates, exact pairs, and audit-only cell summaries."""
    condition_level = aggregate_seeds_within_condition(frame)
    if condition_level.empty:
        raise StatisticalValidationError("No folds have a valid metric state")
    labels = set(condition_level["graph_type"])
    if require_primary and not {PRIMARY_TREATMENT, PRIMARY_COMPARATOR} <= labels:
        raise StatisticalValidationError(
            f"Canonical analysis requires both {PRIMARY_TREATMENT!r} and {PRIMARY_COMPARATOR!r}"
        )
    paired_tables: list[pd.DataFrame] = []
    summary_tables: list[pd.DataFrame] = []
    for spec in comparison_specs(labels):
        paired = pair_condition_values(condition_level, spec)
        if paired.empty:
            continue
        paired_tables.append(paired)
        summary_tables.append(
            summarize_paired_comparison(
                condition_level,
                paired,
                spec,
                bootstrap_replicates,
                bootstrap_seed,
            )
        )
    paired_rows = pd.concat(paired_tables, ignore_index=True) if paired_tables else pd.DataFrame()
    summaries = pd.concat(summary_tables, ignore_index=True) if summary_tables else pd.DataFrame()
    if require_primary:
        primary_pairs = paired_rows.loc[
            (paired_rows["treatment"] == PRIMARY_TREATMENT)
            & (paired_rows["comparator"] == PRIMARY_COMPARATOR)
        ]
        if primary_pairs.empty:
            raise StatisticalValidationError(
                "Primary graph arms exist but have no exact paired support"
            )
    return condition_level, paired_rows, summaries


def _primary_seed_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.loc[
        (frame["metric"] == "pearson_r")
        & (frame["metric_state"] == "valid")
        & frame["graph_type"].isin([PRIMARY_TREATMENT, PRIMARY_COMPARATOR])
        & frame["hvg"].isin(FROZEN_SCALES)
    ].copy()
    simple_keys = ["dataset", "hvg", "condition", "seed"]
    treatment = valid.loc[valid["graph_type"] == PRIMARY_TREATMENT].copy()
    comparator = valid.loc[valid["graph_type"] == PRIMARY_COMPARATOR].copy()
    if treatment.empty or comparator.empty:
        raise StatisticalValidationError("Primary union/self-loop folds are missing")
    for name, arm in (("treatment", treatment), ("comparator", comparator)):
        if arm.duplicated(simple_keys).any():
            examples = arm.loc[arm.duplicated(simple_keys, keep=False), simple_keys].head(5)
            raise StatisticalValidationError(
                f"Primary {name} has multiple graph/config instances per seed-condition: "
                f"{examples.to_dict('records')}"
            )
    keep = [
        *simple_keys,
        "value",
        "input_hash",
        "environment_lock_hash",
        "gene_panel_hash",
        "condition_panel_hash",
        "preprocessing_hash",
        "control_selection_hash",
        "gene_order_hash",
        "gene_order_json",
        "y_true_hash",
        "training_mean_hash",
        "control_profile_hash",
        "planned_condition_list_hash",
        "planned_conditions_json",
        "graph_instance",
        "graph_hash",
        "model_name",
        "model_config_hash",
        "initialization_hash",
        "target_mask_hash",
        "initial_perturbation_context_hash",
        "initial_raw_feature_hash",
        "initial_raw_feature_artifact_sha256",
        "baseline_feature_hash",
        "input_contract_policy_hash",
        "input_definition_hash",
        "input_instance_hash",
        "input_contract_hash",
        "code_commit",
        "split_hash",
        "fold_seed",
        "run_key",
    ]
    paired = treatment[keep].merge(
        comparator[keep],
        on=simple_keys,
        how="inner",
        suffixes=("_treatment", "_comparator"),
        validate="one_to_one",
    )
    matching_fields = (
        "input_hash",
        "environment_lock_hash",
        "gene_panel_hash",
        "condition_panel_hash",
        "preprocessing_hash",
        "control_selection_hash",
        "gene_order_hash",
        "gene_order_json",
        "y_true_hash",
        "training_mean_hash",
        "control_profile_hash",
        "planned_condition_list_hash",
        "planned_conditions_json",
        "model_name",
        "model_config_hash",
        "initialization_hash",
        "target_mask_hash",
        "initial_perturbation_context_hash",
        "initial_raw_feature_hash",
        "initial_raw_feature_artifact_sha256",
        "baseline_feature_hash",
        "input_contract_policy_hash",
        "input_definition_hash",
        "input_instance_hash",
        "input_contract_hash",
        "code_commit",
        "split_hash",
        "fold_seed",
    )
    for field in matching_fields:
        mismatch = paired[f"{field}_treatment"] != paired[f"{field}_comparator"]
        if mismatch.any():
            example = paired.loc[mismatch, simple_keys].iloc[0].to_dict()
            raise StatisticalValidationError(
                f"Primary arms differ in immutable field {field!r} for {example}"
            )
    paired["treatment_r"] = paired.pop("value_treatment")
    paired["comparator_r"] = paired.pop("value_comparator")
    paired["treatment_z"] = _fisher_z(paired["treatment_r"].to_numpy(dtype=float))
    paired["comparator_z"] = _fisher_z(paired["comparator_r"].to_numpy(dtype=float))
    paired["delta_z"] = paired["treatment_z"] - paired["comparator_z"]
    paired["paired_raw_delta_r_descriptive"] = paired["treatment_r"] - paired["comparator_r"]
    return paired.sort_values(simple_keys, kind="mergesort").reset_index(drop=True)


def _validate_primary_build_coherence(frame: pd.DataFrame) -> None:
    """Reject cross-scale vectors assembled from different datasets, builds, or policies."""
    primary = frame.loc[
        frame["graph_type"].isin([PRIMARY_TREATMENT, PRIMARY_COMPARATOR])
        & frame["hvg"].isin(FROZEN_SCALES)
    ].copy()
    required = {
        "code_commit",
        "environment_lock_hash",
        "input_hash",
        "dataset_passport_hash",
        "dataset_passport_registry_hash",
        "condition_panel_hash",
        "planned_condition_list_hash",
        "planned_conditions_json",
        "control_selection_hash",
        "condition_eligibility_ledger_hash",
        "cell_qc_policy_hash",
        "response_definition_hash",
        "target_mapping_hash",
        "matrix_contract_hash",
        "model_name",
        "model_config_hash",
        "initialization_hash",
        "target_mask_hash",
        "initial_perturbation_context_hash",
        "initial_raw_feature_hash",
        "initial_raw_feature_artifact_sha256",
        "baseline_feature_hash",
        "input_contract_policy_hash",
        "input_definition_hash",
        "input_instance_hash",
        "gene_order_hash",
        "gene_order_json",
        "y_true_hash",
        "training_mean_hash",
        "control_profile_hash",
    }
    missing = required - set(primary.columns)
    if missing:
        raise StatisticalValidationError(
            f"Primary build-coherence audit lacks columns: {sorted(missing)}"
        )
    for field in (
        "code_commit",
        "environment_lock_hash",
        "model_name",
        "model_config_hash",
        "dataset_passport_registry_hash",
        "input_contract_policy_hash",
    ):
        values = primary[field].drop_duplicates()
        if len(values) != 1:
            raise StatisticalValidationError(
                f"Canonical primary mixes {field} values: {values.astype(str).tolist()}"
            )
    dataset_fields = (
        "input_hash",
        "dataset_passport_hash",
        "condition_panel_hash",
        "planned_condition_list_hash",
        "planned_conditions_json",
        "control_selection_hash",
        "condition_eligibility_ledger_hash",
        "cell_qc_policy_hash",
        "response_definition_hash",
        "target_mapping_hash",
    )
    for dataset, rows in primary.groupby("dataset", sort=False):
        for field in dataset_fields:
            values = rows[field].drop_duplicates()
            if len(values) != 1:
                raise StatisticalValidationError(
                    f"Dataset {dataset!r} mixes cross-scale {field} values: "
                    f"{values.astype(str).tolist()}"
                )
    for (dataset, hvg), rows in primary.groupby(["dataset", "hvg"], sort=False):
        for field in (
            "gene_order_hash",
            "gene_order_json",
            "control_profile_hash",
            "preprocessing_hash",
            "matrix_contract_hash",
            "baseline_feature_hash",
            "input_definition_hash",
        ):
            if rows[field].nunique(dropna=False) != 1:
                raise StatisticalValidationError(
                    f"Dataset-scale {(dataset, hvg)!r} mixes {field} across primary arms/seeds"
                )
    for (dataset, hvg, condition), rows in primary.groupby(
        ["dataset", "hvg", "condition"], sort=False
    ):
        for field in (
            "y_true_hash",
            "training_mean_hash",
            "control_profile_hash",
            "target_mask_hash",
        ):
            if rows[field].nunique(dropna=False) != 1:
                raise StatisticalValidationError(
                    f"Fold vector identity drift for {(dataset, hvg, condition)!r}: {field}"
                )
    for key, rows in primary.groupby(["dataset", "hvg", "seed"], sort=False):
        if rows["initialization_hash"].nunique(dropna=False) != 1:
            raise StatisticalValidationError(
                f"Primary GAT arms do not share an exact initialization for {key!r}"
            )
    for key, rows in primary.groupby(["dataset", "hvg", "condition", "seed"], sort=False):
        for field in (
            "initial_perturbation_context_hash",
            "initial_raw_feature_hash",
            "initial_raw_feature_artifact_sha256",
            "input_instance_hash",
        ):
            if rows[field].nunique(dropna=False) != 1:
                raise StatisticalValidationError(
                    f"Initial dynamic input drift for {key!r}: {field}"
                )


def _primary_complete_support(
    frame: pd.DataFrame,
    seed_pairs: pd.DataFrame,
    expected_datasets: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_datasets = set(
        frame.loc[frame["graph_type"].isin([PRIMARY_TREATMENT, PRIMARY_COMPARATOR]), "dataset"]
    )
    if observed_datasets != set(expected_datasets):
        raise StatisticalValidationError(
            f"Canonical dataset strata mismatch: expected {sorted(expected_datasets)}, "
            f"observed {sorted(observed_datasets)}"
        )
    required_seed_set = set(REQUIRED_SEEDS)
    valid_primary = frame.loc[
        frame["graph_type"].isin([PRIMARY_TREATMENT, PRIMARY_COMPARATOR])
        & frame["hvg"].isin(FROZEN_SCALES)
    ]
    ledger: list[dict[str, Any]] = []
    complete_keys: list[tuple[str, str]] = []
    for dataset in expected_datasets:
        dataset_rows = valid_primary.loc[valid_primary["dataset"] == dataset]
        panels = dataset_rows["planned_conditions_json"].unique()
        if len(panels) != 1:
            raise StatisticalValidationError(
                f"Dataset {dataset!r} does not have one invariant frozen condition panel"
            )
        planned = [str(value) for value in json.loads(panels[0])]
        for condition in planned:
            reasons: list[str] = []
            for hvg in FROZEN_SCALES:
                for graph_type, arm_name in (
                    (PRIMARY_TREATMENT, "treatment"),
                    (PRIMARY_COMPARATOR, "comparator"),
                ):
                    arm_rows = dataset_rows.loc[
                        (dataset_rows["hvg"] == hvg)
                        & (dataset_rows["condition"] == condition)
                        & (dataset_rows["graph_type"] == graph_type)
                    ]
                    valid_seeds = set(
                        int(seed)
                        for seed in arm_rows.loc[arm_rows["metric_state"] == "valid", "seed"]
                    )
                    if valid_seeds != required_seed_set:
                        reasons.append(
                            f"hvg{hvg}:{arm_name}_valid_seeds="
                            f"{','.join(str(seed) for seed in sorted(valid_seeds)) or 'none'}"
                        )
                paired_seeds = set(
                    int(seed)
                    for seed in seed_pairs.loc[
                        (seed_pairs["dataset"] == dataset)
                        & (seed_pairs["hvg"] == hvg)
                        & (seed_pairs["condition"] == condition),
                        "seed",
                    ]
                )
                if paired_seeds != required_seed_set:
                    reasons.append(
                        f"hvg{hvg}:paired_seeds="
                        f"{','.join(str(seed) for seed in sorted(paired_seeds)) or 'none'}"
                    )
            included = not reasons
            if included:
                complete_keys.append((dataset, condition))
            ledger.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "planned": True,
                    "primary_complete_case": included,
                    "exclusion_reason": "" if included else ";".join(sorted(set(reasons))),
                }
            )
    complete_index = pd.MultiIndex.from_tuples(complete_keys, names=["dataset", "condition"])
    pair_index = pd.MultiIndex.from_frame(seed_pairs[["dataset", "condition"]])
    complete = seed_pairs.loc[pair_index.isin(complete_index)].copy()
    for dataset in expected_datasets:
        count = complete.loc[complete["dataset"] == dataset, "condition"].nunique()
        if count == 0:
            raise StatisticalValidationError(
                f"Dataset {dataset!r} has no three-scale, three-seed complete conditions"
            )
    return complete.reset_index(drop=True), pd.DataFrame(ledger)


def _condition_tables(complete: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_scale = (
        complete.groupby(["dataset", "condition", "hvg"], as_index=False)
        .agg(
            treatment_mu_z=("treatment_z", "mean"),
            comparator_mu_z=("comparator_z", "mean"),
            paired_seed_raw_delta_r_descriptive=(
                "paired_raw_delta_r_descriptive",
                "mean",
            ),
            seed_n=("seed", "nunique"),
        )
        .sort_values(["dataset", "condition", "hvg"], kind="mergesort")
    )
    if set(condition_scale["seed_n"]) != {len(REQUIRED_SEEDS)}:
        raise StatisticalValidationError("Primary condition-scale rows are not three-seed complete")
    condition_estimates = (
        condition_scale.groupby(["dataset", "condition"], as_index=False)
        .agg(
            treatment_mu_z=("treatment_mu_z", "mean"),
            comparator_mu_z=("comparator_mu_z", "mean"),
            paired_raw_delta_r_descriptive=(
                "paired_seed_raw_delta_r_descriptive",
                "mean",
            ),
            scale_n=("hvg", "nunique"),
        )
        .sort_values(["dataset", "condition"], kind="mergesort")
    )
    if set(condition_estimates["scale_n"]) != {len(FROZEN_SCALES)}:
        raise StatisticalValidationError("Primary conditions are not three-scale complete")
    condition_estimates["delta_z"] = (
        condition_estimates["treatment_mu_z"] - condition_estimates["comparator_mu_z"]
    )
    condition_estimates["raw_delta_r_estimand"] = np.tanh(
        condition_estimates["treatment_mu_z"]
    ) - np.tanh(condition_estimates["comparator_mu_z"])
    return condition_scale, condition_estimates


def _resolution_statistic(
    complete: pd.DataFrame,
    sampled_conditions: Mapping[str, Sequence[str]],
    datasets: Sequence[str],
) -> tuple[float, dict[str, float]]:
    seed_condition = (
        complete.groupby(["dataset", "seed", "condition"], as_index=False)
        .agg(
            treatment_mu_z=("treatment_z", "mean"),
            comparator_mu_z=("comparator_z", "mean"),
            scale_n=("hvg", "nunique"),
        )
        .sort_values(["dataset", "seed", "condition"], kind="mergesort")
    )
    if set(seed_condition["scale_n"]) != {len(FROZEN_SCALES)}:
        raise StatisticalValidationError("Resolution estimator requires complete scale vectors")
    dataset_seed_effects: dict[str, dict[int, float]] = {}
    for dataset in datasets:
        dataset_seed_effects[dataset] = {}
        for seed in REQUIRED_SEEDS:
            lookup = seed_condition.loc[
                (seed_condition["dataset"] == dataset) & (seed_condition["seed"] == seed)
            ].set_index("condition")
            selected = lookup.loc[list(sampled_conditions[dataset])]
            treatment_mu_z = float(selected["treatment_mu_z"].mean())
            comparator_mu_z = float(selected["comparator_mu_z"].mean())
            dataset_seed_effects[dataset][seed] = float(
                np.tanh(treatment_mu_z) - np.tanh(comparator_mu_z)
            )
    dataset_values = {
        dataset: float(
            np.quantile(
                [
                    abs(
                        dataset_seed_effects[dataset][seed_a]
                        - dataset_seed_effects[dataset][seed_b]
                    )
                    / math.sqrt(2.0)
                    for seed_a, seed_b in combinations(REQUIRED_SEEDS, 2)
                ],
                0.95,
                method="linear",
            )
        )
        for dataset in datasets
    }
    global_seed_effects: dict[int, float] = {}
    for seed in REQUIRED_SEEDS:
        dataset_treatment_z: list[float] = []
        dataset_comparator_z: list[float] = []
        for dataset in datasets:
            lookup = seed_condition.loc[
                (seed_condition["dataset"] == dataset) & (seed_condition["seed"] == seed)
            ].set_index("condition")
            selected = lookup.loc[list(sampled_conditions[dataset])]
            dataset_treatment_z.append(float(selected["treatment_mu_z"].mean()))
            dataset_comparator_z.append(float(selected["comparator_mu_z"].mean()))
        global_seed_effects[seed] = float(
            np.tanh(np.mean(dataset_treatment_z)) - np.tanh(np.mean(dataset_comparator_z))
        )
    global_values = [
        abs(global_seed_effects[seed_a] - global_seed_effects[seed_b]) / math.sqrt(2.0)
        for seed_a, seed_b in combinations(REQUIRED_SEEDS, 2)
    ]
    global_resolution = float(np.quantile(global_values, 0.95, method="linear"))
    return global_resolution, dataset_values


def analyze_primary_estimand(
    frame: pd.DataFrame,
    *,
    bootstrap_replicates: int = CANONICAL_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = CANONICAL_BOOTSTRAP_SEED,
    expected_datasets: Sequence[str] = CANONICAL_DATASETS,
) -> PrimaryAnalysis:
    """Execute the frozen G1, TDS-32, and G3 complete-case estimators."""
    frame = canonicalize_analysis_frame(frame)
    validate_raw_fold_data(frame)
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    _validate_primary_build_coherence(frame)
    seed_pairs = _primary_seed_pairs(frame)
    complete, support_ledger = _primary_complete_support(frame, seed_pairs, expected_datasets)
    condition_scale, condition_estimates = _condition_tables(complete)
    dataset_estimates = (
        condition_estimates.groupby("dataset", as_index=False)
        .agg(
            treatment_mu_z=("treatment_mu_z", "mean"),
            comparator_mu_z=("comparator_mu_z", "mean"),
            paired_raw_delta_r_descriptive=("paired_raw_delta_r_descriptive", "mean"),
            complete_condition_n=("condition", "nunique"),
        )
        .sort_values("dataset", kind="mergesort")
    )
    dataset_estimates["delta_z"] = (
        dataset_estimates["treatment_mu_z"] - dataset_estimates["comparator_mu_z"]
    )
    dataset_estimates["raw_delta_r_estimand"] = np.tanh(
        dataset_estimates["treatment_mu_z"]
    ) - np.tanh(dataset_estimates["comparator_mu_z"])
    global_treatment_mu_z = float(dataset_estimates["treatment_mu_z"].mean())
    global_comparator_mu_z = float(dataset_estimates["comparator_mu_z"].mean())
    global_delta_z = global_treatment_mu_z - global_comparator_mu_z
    global_raw_delta_r = float(np.tanh(global_treatment_mu_z) - np.tanh(global_comparator_mu_z))
    descriptive_raw_delta = float(dataset_estimates["paired_raw_delta_r_descriptive"].mean())

    condition_names = {
        dataset: utf8_sort(
            condition_estimates.loc[condition_estimates["dataset"] == dataset, "condition"].tolist()
        )
        for dataset in expected_datasets
    }
    point_samples = {dataset: tuple(values) for dataset, values in condition_names.items()}
    resolution_point, resolution_dataset_points = _resolution_statistic(
        complete, point_samples, expected_datasets
    )

    condition_lookup = {
        dataset: condition_estimates.loc[condition_estimates["dataset"] == dataset].set_index(
            "condition"
        )
        for dataset in expected_datasets
    }
    scale_lookup = {
        dataset: condition_scale.loc[condition_scale["dataset"] == dataset]
        .set_index(["condition", "hvg"])
        .sort_index()
        for dataset in expected_datasets
    }
    rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    bootstrap_rows: list[dict[str, Any]] = []
    dataset_bootstrap_effects: dict[str, list[float]] = {
        dataset: [] for dataset in expected_datasets
    }
    dataset_bootstrap_resolution: dict[str, list[float]] = {
        dataset: [] for dataset in expected_datasets
    }
    for replicate in range(bootstrap_replicates):
        sampled: dict[str, list[str]] = {}
        dataset_treatment_z: dict[str, float] = {}
        dataset_comparator_z: dict[str, float] = {}
        for dataset in expected_datasets:
            conditions = condition_names[dataset]
            indices = rng.integers(0, len(conditions), size=len(conditions))
            sampled_conditions = [conditions[index] for index in indices]
            sampled[dataset] = sampled_conditions
            lookup = condition_lookup[dataset]
            dataset_treatment_z[dataset] = float(
                lookup.loc[sampled_conditions, "treatment_mu_z"].mean()
            )
            dataset_comparator_z[dataset] = float(
                lookup.loc[sampled_conditions, "comparator_mu_z"].mean()
            )
            dataset_effect = float(
                np.tanh(dataset_treatment_z[dataset]) - np.tanh(dataset_comparator_z[dataset])
            )
            dataset_bootstrap_effects[dataset].append(dataset_effect)
        treatment_mu_z = float(np.mean(list(dataset_treatment_z.values())))
        comparator_mu_z = float(np.mean(list(dataset_comparator_z.values())))
        bootstrap_delta_z = treatment_mu_z - comparator_mu_z
        bootstrap_delta_r = float(np.tanh(treatment_mu_z) - np.tanh(comparator_mu_z))
        bootstrap_resolution, dataset_resolution = _resolution_statistic(
            complete, sampled, expected_datasets
        )
        for dataset in expected_datasets:
            dataset_bootstrap_resolution[dataset].append(dataset_resolution[dataset])
        bootstrap_rows.append(
            {
                "replicate": replicate,
                "treatment_mu_z": treatment_mu_z,
                "comparator_mu_z": comparator_mu_z,
                "delta_z": bootstrap_delta_z,
                "raw_delta_r_estimand": bootstrap_delta_r,
                "paired_raw_delta_r_descriptive": float(
                    np.mean(
                        [
                            condition_lookup[dataset]
                            .loc[sampled[dataset], "paired_raw_delta_r_descriptive"]
                            .mean()
                            for dataset in expected_datasets
                        ]
                    )
                ),
                "delta_res_statistic": bootstrap_resolution,
                **{
                    f"delta_res_dataset__{dataset}": dataset_resolution[dataset]
                    for dataset in expected_datasets
                },
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)

    g3_rng = np.random.Generator(np.random.PCG64(G3_BOOTSTRAP_SEED))
    dataset_bootstrap_g3: dict[str, list[float]] = {dataset: [] for dataset in expected_datasets}
    global_bootstrap_g3: list[float] = []
    for _ in range(bootstrap_replicates):
        dataset_g3: dict[str, float] = {}
        arm_z_by_scale: dict[int, dict[str, list[float]]] = {
            hvg: {"treatment": [], "comparator": []} for hvg in (200, 1000)
        }
        for dataset in expected_datasets:
            conditions = condition_names[dataset]
            indices = g3_rng.integers(0, len(conditions), size=len(conditions))
            sampled_conditions = [conditions[index] for index in indices]
            scale_table = scale_lookup[dataset]
            scale_effects: dict[int, float] = {}
            for hvg in (200, 1000):
                index = pd.MultiIndex.from_tuples(
                    [(condition, hvg) for condition in sampled_conditions],
                    names=["condition", "hvg"],
                )
                selected = scale_table.loc[index]
                treatment_z = float(selected["treatment_mu_z"].mean())
                comparator_z = float(selected["comparator_mu_z"].mean())
                arm_z_by_scale[hvg]["treatment"].append(treatment_z)
                arm_z_by_scale[hvg]["comparator"].append(comparator_z)
                scale_effects[hvg] = float(np.tanh(treatment_z) - np.tanh(comparator_z))
            dataset_g3[dataset] = scale_effects[1000] - scale_effects[200]
            dataset_bootstrap_g3[dataset].append(dataset_g3[dataset])
        global_scale_effects = {
            hvg: float(
                np.tanh(np.mean(arm_z_by_scale[hvg]["treatment"]))
                - np.tanh(np.mean(arm_z_by_scale[hvg]["comparator"]))
            )
            for hvg in (200, 1000)
        }
        global_bootstrap_g3.append(global_scale_effects[1000] - global_scale_effects[200])
    bootstrap["g3_hvg1000_minus_hvg200"] = global_bootstrap_g3
    if not np.isfinite(bootstrap.select_dtypes(include=[np.number]).to_numpy()).all():
        raise StatisticalValidationError("A primary, resolution, or G3 bootstrap replicate failed")
    delta_r_ci = np.quantile(bootstrap["raw_delta_r_estimand"], [0.025, 0.975], method="linear")
    delta_z_ci = np.quantile(bootstrap["delta_z"], [0.025, 0.975], method="linear")
    resolution_upper = float(np.quantile(bootstrap["delta_res_statistic"], 0.95, method="linear"))
    support_gate = bool(
        (dataset_estimates["complete_condition_n"] >= MINIMUM_COMPLETE_CONDITIONS).all()
    )
    primary_summary = pd.DataFrame(
        [
            {
                "treatment": PRIMARY_TREATMENT,
                "comparator": PRIMARY_COMPARATOR,
                "metric": "pearson_r",
                "estimand": "equal-dataset Fisher-z arm means; separately back-transformed",
                "treatment_mu_z": global_treatment_mu_z,
                "comparator_mu_z": global_comparator_mu_z,
                "delta_z": global_delta_z,
                "raw_delta_r_estimand": global_raw_delta_r,
                "raw_delta_r_ci_low": float(delta_r_ci[0]),
                "raw_delta_r_ci_high": float(delta_r_ci[1]),
                "delta_z_ci_low": float(delta_z_ci[0]),
                "delta_z_ci_high": float(delta_z_ci[1]),
                "paired_raw_delta_r_descriptive": descriptive_raw_delta,
                "delta_res_point": resolution_point,
                "delta_res_upper_95": resolution_upper,
                "practical_margin_primary": 0.02,
                "complete_condition_n": int(dataset_estimates["complete_condition_n"].sum()),
                "dataset_n": len(expected_datasets),
                "minimum_45_per_dataset_gate": support_gate,
                "bootstrap_replicates": bootstrap_replicates,
                "bootstrap_seed": bootstrap_seed,
            }
        ]
    )

    dataset_rows: list[dict[str, Any]] = []
    resolution_rows: list[dict[str, Any]] = [
        {
            "scope": "global_equal_dataset",
            "dataset": "ALL",
            "delta_res_point": resolution_point,
            "delta_res_upper_95": resolution_upper,
        }
    ]
    g3_rows: list[dict[str, Any]] = []
    point_scale_effects: dict[str, dict[int, float]] = {}
    point_scale_arm_z: dict[int, dict[str, list[float]]] = {
        hvg: {"treatment": [], "comparator": []} for hvg in (200, 1000)
    }
    for dataset in expected_datasets:
        row = dataset_estimates.loc[dataset_estimates["dataset"] == dataset].iloc[0]
        dataset_effect_ci = np.quantile(
            dataset_bootstrap_effects[dataset], [0.025, 0.975], method="linear"
        )
        dataset_rows.append(
            {
                **row.to_dict(),
                "raw_delta_r_ci_low": float(dataset_effect_ci[0]),
                "raw_delta_r_ci_high": float(dataset_effect_ci[1]),
                "raw_delta_r_ci_half_width": float(
                    max(
                        row["raw_delta_r_estimand"] - dataset_effect_ci[0],
                        dataset_effect_ci[1] - row["raw_delta_r_estimand"],
                    )
                ),
                "delta_res_upper_95": float(
                    np.quantile(
                        dataset_bootstrap_resolution[dataset],
                        0.95,
                        method="linear",
                    )
                ),
                "minimum_45_condition_gate": bool(
                    row["complete_condition_n"] >= MINIMUM_COMPLETE_CONDITIONS
                ),
            }
        )
        resolution_rows.append(
            {
                "scope": "dataset_equal_scale",
                "dataset": dataset,
                "delta_res_point": resolution_dataset_points[dataset],
                "delta_res_upper_95": float(
                    np.quantile(
                        dataset_bootstrap_resolution[dataset],
                        0.95,
                        method="linear",
                    )
                ),
            }
        )
        point_scale_effects[dataset] = {}
        dataset_scales = condition_scale.loc[condition_scale["dataset"] == dataset]
        for hvg in FROZEN_SCALES:
            scale_rows = dataset_scales.loc[dataset_scales["hvg"] == hvg]
            treatment_z = float(scale_rows["treatment_mu_z"].mean())
            comparator_z = float(scale_rows["comparator_mu_z"].mean())
            point_scale_effects[dataset][hvg] = float(np.tanh(treatment_z) - np.tanh(comparator_z))
            if hvg in point_scale_arm_z:
                point_scale_arm_z[hvg]["treatment"].append(treatment_z)
                point_scale_arm_z[hvg]["comparator"].append(comparator_z)
        interaction = point_scale_effects[dataset][1000] - point_scale_effects[dataset][200]
        interaction_ci = np.quantile(dataset_bootstrap_g3[dataset], [0.025, 0.975], method="linear")
        g3_rows.append(
            {
                "scope": "dataset",
                "dataset": dataset,
                "effect_hvg200": point_scale_effects[dataset][200],
                "effect_hvg1000": point_scale_effects[dataset][1000],
                "interaction_1000_minus_200": interaction,
                "ci_low": float(interaction_ci[0]),
                "ci_high": float(interaction_ci[1]),
                "bootstrap_replicates": bootstrap_replicates,
                "bootstrap_seed": G3_BOOTSTRAP_SEED,
            }
        )
    global_scale_effects = {
        hvg: float(
            np.tanh(np.mean(point_scale_arm_z[hvg]["treatment"]))
            - np.tanh(np.mean(point_scale_arm_z[hvg]["comparator"]))
        )
        for hvg in (200, 1000)
    }
    global_interaction = global_scale_effects[1000] - global_scale_effects[200]
    global_g3_ci = np.quantile(
        bootstrap["g3_hvg1000_minus_hvg200"], [0.025, 0.975], method="linear"
    )
    g3_rows.insert(
        0,
        {
            "scope": "global_equal_dataset",
            "dataset": "ALL",
            "effect_hvg200": global_scale_effects[200],
            "effect_hvg1000": global_scale_effects[1000],
            "interaction_1000_minus_200": global_interaction,
            "ci_low": float(global_g3_ci[0]),
            "ci_high": float(global_g3_ci[1]),
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": G3_BOOTSTRAP_SEED,
        },
    )
    dataset_result_frame = pd.DataFrame(dataset_rows)
    dataset_result_frame["operative_margin"] = np.maximum(
        dataset_result_frame["delta_res_upper_95"], 0.02
    )
    dataset_result_frame["precision_gate"] = (
        dataset_result_frame["raw_delta_r_ci_half_width"] <= 0.05
    )
    dataset_result_frame["ci_inside_practical_band"] = (
        dataset_result_frame["raw_delta_r_ci_low"] >= -dataset_result_frame["operative_margin"]
    ) & (dataset_result_frame["raw_delta_r_ci_high"] <= dataset_result_frame["operative_margin"])
    global_half_width = float(
        max(global_raw_delta_r - delta_r_ci[0], delta_r_ci[1] - global_raw_delta_r)
    )
    global_margin = max(resolution_upper, 0.02)
    global_precision = global_half_width <= 0.02
    all_dataset_precision = bool(dataset_result_frame["precision_gate"].all())
    resolved_positive = bool(
        support_gate
        and global_precision
        and all_dataset_precision
        and delta_r_ci[0] > 0
        and global_raw_delta_r > global_margin
        and not (dataset_result_frame["raw_delta_r_ci_high"] < 0).any()
    )
    resolved_negative = bool(
        support_gate
        and global_precision
        and all_dataset_precision
        and delta_r_ci[1] < 0
        and global_raw_delta_r < -global_margin
        and not (dataset_result_frame["raw_delta_r_ci_low"] > 0).any()
    )
    bounded_small = bool(
        support_gate
        and global_precision
        and all_dataset_precision
        and delta_r_ci[0] >= -global_margin
        and delta_r_ci[1] <= global_margin
        and dataset_result_frame["ci_inside_practical_band"].all()
        and not resolved_positive
        and not resolved_negative
    )
    branch = (
        "TDS07_resolved_positive"
        if resolved_positive
        else (
            "TDS35_resolved_negative"
            if resolved_negative
            else (
                "TDS08_bounded_small"
                if bounded_small
                else "TDS09_or_unresolved_requires_frozen_heterogeneity_family"
            )
        )
    )
    primary_summary["raw_delta_r_ci_half_width"] = global_half_width
    primary_summary["operative_margin"] = global_margin
    primary_summary["global_precision_gate"] = global_precision
    primary_summary["all_dataset_precision_gate"] = all_dataset_precision
    primary_summary["tds07_resolved_positive"] = resolved_positive
    primary_summary["tds35_resolved_negative"] = resolved_negative
    primary_summary["tds08_bounded_small"] = bounded_small
    primary_summary["preliminary_branch"] = branch
    return PrimaryAnalysis(
        seed_pairs=complete,
        support_ledger=support_ledger,
        condition_scale=condition_scale,
        condition_estimates=condition_estimates,
        dataset_estimates=dataset_result_frame,
        primary_summary=primary_summary,
        resolution_summary=pd.DataFrame(resolution_rows),
        g3_summary=pd.DataFrame(g3_rows),
        bootstrap_distribution=bootstrap,
    )


def finalize_tds09_decision(
    primary_summary: pd.DataFrame,
    dataset_estimates: pd.DataFrame,
    dataset_sign_flip: pd.DataFrame,
    heterogeneity: MonteCarloTest,
    *,
    expected_datasets: Sequence[str] = CANONICAL_DATASETS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply TDS-09 only after the higher-precedence TDS-07/35/08 branches fail."""
    if len(primary_summary) != 1:
        raise StatisticalValidationError("TDS-09 requires exactly one primary summary row")
    required_estimate = {
        "dataset",
        "raw_delta_r_estimand",
        "raw_delta_r_ci_low",
        "raw_delta_r_ci_high",
        "operative_margin",
        "minimum_45_condition_gate",
        "precision_gate",
    }
    missing_estimate = required_estimate - set(dataset_estimates.columns)
    if missing_estimate:
        raise StatisticalValidationError(
            f"TDS-09 dataset estimates lack columns: {sorted(missing_estimate)}"
        )
    required_test = {"dataset", "raw_p", "holm_p"}
    missing_test = required_test - set(dataset_sign_flip.columns)
    if missing_test:
        raise StatisticalValidationError(
            f"TDS-09 sign-flip table lacks columns: {sorted(missing_test)}"
        )
    expected = tuple(nfc_text(dataset) for dataset in expected_datasets)
    if len(set(expected)) != len(expected):
        raise StatisticalValidationError("TDS-09 dataset order duplicates after NFC normalization")
    estimates = dataset_estimates.copy()
    tests = dataset_sign_flip.copy()
    estimates["dataset"] = estimates["dataset"].map(lambda value: nfc_text(str(value)))
    tests["dataset"] = tests["dataset"].map(lambda value: nfc_text(str(value)))
    for label, table in (("estimate", estimates), ("sign-flip", tests)):
        if table["dataset"].duplicated().any() or set(table["dataset"]) != set(expected):
            raise StatisticalValidationError(
                f"TDS-09 {label} datasets must equal the frozen family {list(expected)}"
            )
    tests = tests[["dataset", "raw_p", "holm_p"]]
    estimates = estimates.merge(tests, on="dataset", how="left", validate="one_to_one")
    numeric = estimates[
        [
            "raw_delta_r_estimand",
            "raw_delta_r_ci_low",
            "raw_delta_r_ci_high",
            "operative_margin",
            "raw_p",
            "holm_p",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not 0.0 <= heterogeneity.p_value <= 1.0:
        raise StatisticalValidationError("TDS-09 received a non-finite estimate or invalid p value")
    estimates["ci_excludes_zero"] = (estimates["raw_delta_r_ci_low"] > 0.0) | (
        estimates["raw_delta_r_ci_high"] < 0.0
    )
    estimates["magnitude_gate"] = (
        estimates["raw_delta_r_estimand"].abs() > estimates["operative_margin"]
    )
    estimates["holm_gate"] = estimates["holm_p"] < 0.05
    estimates["tds09_local_gate"] = (
        estimates["minimum_45_condition_gate"].astype(bool)
        & estimates["precision_gate"].astype(bool)
        & estimates["ci_excludes_zero"]
        & estimates["magnitude_gate"]
        & estimates["holm_gate"]
    )
    estimates["_dataset_order"] = estimates["dataset"].map(
        {dataset: index for index, dataset in enumerate(expected)}
    )
    estimates = estimates.sort_values("_dataset_order", kind="mergesort").drop(
        columns="_dataset_order"
    )

    summary = primary_summary.copy()
    preliminary = str(summary.iloc[0]["preliminary_branch"])
    integrity_gate = bool(
        summary.iloc[0]["minimum_45_per_dataset_gate"]
        and summary.iloc[0]["global_precision_gate"]
        and summary.iloc[0]["all_dataset_precision_gate"]
    )
    heterogeneity_gate = heterogeneity.p_value < 0.05
    claimed = estimates.loc[estimates["tds09_local_gate"], "dataset"].tolist()
    tds09_pass = bool(
        preliminary == "TDS09_or_unresolved_requires_frozen_heterogeneity_family"
        and integrity_gate
        and heterogeneity_gate
        and claimed
    )
    if preliminary != "TDS09_or_unresolved_requires_frozen_heterogeneity_family":
        final_branch = preliminary
        claimed = []
    elif tds09_pass:
        final_branch = "TDS09_dataset_specific"
    else:
        final_branch = "unresolved_or_imprecise"

    nonpassing: dict[str, list[str]] = {}
    for row in estimates.itertuples(index=False):
        reasons: list[str] = []
        if not bool(row.minimum_45_condition_gate):
            reasons.append("support_below_45")
        if not bool(row.precision_gate):
            reasons.append("ci_half_width_above_0.05")
        if not bool(row.ci_excludes_zero):
            reasons.append("ci_includes_zero")
        if not bool(row.magnitude_gate):
            reasons.append("effect_not_above_operative_margin")
        if not bool(row.holm_gate):
            reasons.append("holm_p_not_below_0.05")
        if reasons or str(row.dataset) not in claimed:
            nonpassing[str(row.dataset)] = reasons or ["higher_precedence_branch_selected"]
    branch_failures: list[str] = []
    if preliminary == "TDS09_or_unresolved_requires_frozen_heterogeneity_family":
        if not integrity_gate:
            branch_failures.append("support_or_precision_integrity_gate_failed")
        if not heterogeneity_gate:
            branch_failures.append("heterogeneity_p_not_below_0.05")
        if not claimed:
            branch_failures.append("no_dataset_passed_all_local_gates")
    summary["tds09_heterogeneity_p"] = heterogeneity.p_value
    summary["tds09_heterogeneity_gate"] = heterogeneity_gate
    summary["tds09_integrity_gate"] = integrity_gate
    summary["tds09_pass"] = tds09_pass
    summary["final_branch"] = final_branch
    summary["claimed_datasets_json"] = canonical_json(claimed)
    summary["unclaimed_datasets_json"] = canonical_json(nonpassing)
    summary["final_branch_failures_json"] = canonical_json(branch_failures)
    decision = {
        "schema_version": "1.0.0",
        "precedence": ["TDS07", "TDS35", "TDS08", "TDS09", "unresolved"],
        "preliminary_branch": preliminary,
        "final_branch": final_branch,
        "claimed_datasets": claimed,
        "unclaimed_datasets": nonpassing,
        "heterogeneity_p": heterogeneity.p_value,
        "heterogeneity_gate": heterogeneity_gate,
        "integrity_gate": integrity_gate,
        "branch_failures": branch_failures,
    }
    return summary, estimates, decision


def validate_topology_ensemble_instances(frame: pd.DataFrame) -> pd.DataFrame:
    """Block T1 unless every actual dataset-scale ensemble has five distinct frozen hashes."""
    expected = {
        "degree_preserving_rewired": {101, 102, 103, 104, 105},
        "barabasi_albert": {201, 202, 203, 204, 205},
    }
    topology = frame.loc[frame["graph_type"].isin(expected)].copy()
    if topology.empty:
        return pd.DataFrame()
    required = {
        "graph_parent_edge_hash",
        "graph_instance_seed",
        "graph_ensemble_config_hash",
    }
    missing = required - set(topology.columns)
    if missing:
        raise StatisticalValidationError(
            f"Topology ensemble audit lacks provenance columns: {sorted(missing)}"
        )
    report: list[dict[str, object]] = []
    for (dataset, hvg, graph_type), rows in topology.groupby(
        ["dataset", "hvg", "graph_type"], sort=True
    ):
        if rows[list(required)].isna().any().any():
            raise StatisticalValidationError(
                f"Topology ensemble {(dataset, hvg, graph_type)!r} has missing provenance"
            )
        seeds = {int(value) for value in rows["graph_instance_seed"]}
        if seeds != expected[str(graph_type)]:
            raise StatisticalValidationError(
                f"Topology ensemble {(dataset, hvg, graph_type)!r} seeds {sorted(seeds)} "
                f"!= {sorted(expected[str(graph_type)])}"
            )
        per_seed = rows[
            [
                "graph_instance_seed",
                "graph_hash",
                "graph_parent_edge_hash",
                "graph_ensemble_config_hash",
            ]
        ].drop_duplicates()
        if len(per_seed) != 5 or per_seed["graph_instance_seed"].nunique() != 5:
            raise StatisticalValidationError(
                f"Topology ensemble {(dataset, hvg, graph_type)!r} has seed/hash drift"
            )
        if per_seed["graph_hash"].nunique() != 5:
            raise StatisticalValidationError(
                f"Topology ensemble {(dataset, hvg, graph_type)!r} instances are not distinct"
            )
        if per_seed["graph_parent_edge_hash"].nunique() != 1:
            raise StatisticalValidationError(
                f"Topology ensemble {(dataset, hvg, graph_type)!r} mixes union parents"
            )
        parent_hash = str(per_seed["graph_parent_edge_hash"].iloc[0])
        if parent_hash in set(per_seed["graph_hash"].astype(str)):
            raise StatisticalValidationError(
                f"Topology ensemble {(dataset, hvg, graph_type)!r} reuses its union edge set"
            )
        if per_seed["graph_ensemble_config_hash"].nunique() != 1:
            raise StatisticalValidationError(
                f"Topology ensemble {(dataset, hvg, graph_type)!r} mixes config hashes"
            )
        report.append(
            {
                "dataset": dataset,
                "hvg": int(hvg),
                "graph_type": graph_type,
                "instance_count": 5,
                "distinct_edge_hash_count": 5,
                "parent_edge_hash": parent_hash,
                "graph_ensemble_config_hash": str(per_seed["graph_ensemble_config_hash"].iloc[0]),
                "status": "valid",
            }
        )
    return pd.DataFrame(report)


def write_analysis(results_root: Path, output_dir: Path) -> None:
    """Run strict audit tables and the frozen canonical Pearson-r estimators."""
    frame = load_fold_data(results_root, metric="pearson_r")
    if set(frame["execution_stage"]) != {"full"}:
        raise StatisticalValidationError(
            "Canonical claim analysis accepts only --stage full artifacts; E0/smoke are nonclaim"
        )
    topology_report = validate_topology_ensemble_instances(frame)
    condition_level, paired_rows, summaries = analyze_fold_data(frame)
    primary = analyze_primary_estimand(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "raw_fold_ledger.csv", index=False)
    topology_report.to_csv(output_dir / "topology_ensemble_integrity.csv", index=False)
    condition_level.to_csv(output_dir / "condition_level_seed_aggregates.csv", index=False)
    paired_rows.to_csv(output_dir / "paired_condition_values.csv", index=False)
    summaries.to_csv(output_dir / "dataset_scale_audit_summaries.csv", index=False)
    primary.seed_pairs.to_csv(output_dir / "primary_seed_pairs.csv", index=False)
    primary.support_ledger.to_csv(output_dir / "primary_support_ledger.csv", index=False)
    primary.condition_scale.to_csv(output_dir / "primary_condition_scale.csv", index=False)
    primary.condition_estimates.to_csv(output_dir / "primary_condition_estimates.csv", index=False)
    primary.resolution_summary.to_csv(output_dir / "empirical_resolution.csv", index=False)
    primary.g3_summary.to_csv(output_dir / "g3_scale_interaction.csv", index=False)
    primary.bootstrap_distribution.to_csv(
        output_dir / "primary_bootstrap_distribution.csv", index=False
    )
    resolution_columns = [
        column
        for column in primary.bootstrap_distribution.columns
        if column == "replicate" or column.startswith("delta_res_")
    ]
    primary.bootstrap_distribution[resolution_columns].to_csv(
        output_dir / "empirical_resolution_bootstrap_distribution.csv", index=False
    )
    dataset_sign_flip = dataset_sign_flip_holm_family(
        primary.condition_estimates,
        datasets=CANONICAL_DATASETS,
        draws=100_000,
        seed=20_260_804,
    )
    heterogeneity = centered_residual_q_test(
        primary.condition_estimates,
        datasets=CANONICAL_DATASETS,
        draws=10_000,
        seed=20_260_805,
    )
    final_summary, final_dataset_estimates, branch_decision = finalize_tds09_decision(
        primary.primary_summary,
        primary.dataset_estimates,
        dataset_sign_flip,
        heterogeneity,
    )
    final_dataset_estimates.to_csv(output_dir / "primary_dataset_estimates.csv", index=False)
    final_summary.to_csv(output_dir / "primary_effect.csv", index=False)
    dataset_sign_flip.to_csv(output_dir / "dataset_sign_flip_holm.csv", index=False)
    write_json_atomic(output_dir / "heterogeneity_centered_residual_q.json", asdict(heterogeneity))
    write_json_atomic(output_dir / "primary_branch_decision.json", branch_decision)
    estimator_config = json.loads(
        (Path(__file__).resolve().parent / "config" / "estimands.json").read_text(encoding="utf-8")
    )
    manifest = {
        "schema_version": "1.0.0",
        "results_root": str(results_root.resolve()),
        "metric": "pearson_r",
        "primary_treatment": PRIMARY_TREATMENT,
        "primary_comparator": PRIMARY_COMPARATOR,
        "invalid_metric_policy": "retain in support ledger; exclude from numeric estimators",
        "legacy_policy": "revision metadata required; no silent legacy fallback",
        "pairing_policy": f"exact join on {list(PAIR_SCOPE)} with exact seed sets",
        "estimator_config": estimator_config,
        "estimator_config_hash": sha256_json(estimator_config),
        "raw_seed_run_n": len(frame),
        "valid_condition_graph_n": len(condition_level),
        "paired_row_n": len(paired_rows),
        "comparison_scope_n": len(summaries),
        "comparison_specs": [
            asdict(spec) for spec in comparison_specs(condition_level["graph_type"].unique())
        ],
        "tds09": {
            "dataset_family": (
                "four frozen-order dataset condition-level delta-z sign flips "
                "with Holm adjustment"
            ),
            "heterogeneity_q": asdict(heterogeneity),
            "branch_decision": branch_decision,
        },
    }
    write_json_atomic(output_dir / "analysis_manifest.json", manifest)


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point for the frozen canonical analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    write_analysis(args.results_root, args.output_dir)


if __name__ == "__main__":
    main()

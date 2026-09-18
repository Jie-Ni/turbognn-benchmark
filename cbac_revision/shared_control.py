"""Hash-bound shared-control cell evidence and cell-resampling sensitivity."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import canonical_sha256, file_sha256, read_fold_artifact
from .errors import RevisionProtocolError

SHA256_RE = re.compile(r"[0-9a-f]{64}")


def build_shared_control_evidence(
    *,
    dataset: str,
    hvg: int,
    gene_names: Sequence[str],
    control_row_ids: Sequence[str],
    standardized_control_profiles: np.ndarray,
    dataset_sha256: str,
    preprocessing_state_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    """Create a deterministic manifest plus NPY bytes for standardized control cells."""

    genes = tuple(str(value) for value in gene_names)
    row_ids = tuple(str(value) for value in control_row_ids)
    matrix = np.asarray(standardized_control_profiles, dtype=np.float64)
    if (
        not dataset
        or isinstance(hvg, bool)
        or not isinstance(hvg, int)
        or hvg <= 0
        or len(genes) != hvg
        or len(set(genes)) != hvg
        or matrix.shape != (len(row_ids), hvg)
        or len(row_ids) < 2
        or len(set(row_ids)) != len(row_ids)
        or not np.isfinite(matrix).all()
        or not SHA256_RE.fullmatch(dataset_sha256)
        or not SHA256_RE.fullmatch(preprocessing_state_sha256)
    ):
        raise RevisionProtocolError("[SHARED_CONTROL_EVIDENCE_INPUT_INVALID]")
    buffer = io.BytesIO()
    np.save(buffer, matrix, allow_pickle=False)
    matrix_bytes = buffer.getvalue()
    matrix_file_sha256 = hashlib.sha256(matrix_bytes).hexdigest()
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "evidence_id": "SHARED-CONTROL-CELL-PROFILES",
        "dataset": dataset,
        "hvg": hvg,
        "vector_space": "control_fitted_standardized_expression",
        "gene_names": list(genes),
        "gene_names_sha256": canonical_sha256(genes),
        "control_row_ids": list(row_ids),
        "control_row_ids_sha256": canonical_sha256(row_ids),
        "n_control_cells": len(row_ids),
        "profile_matrix_shape": list(matrix.shape),
        "profile_matrix_content_sha256": canonical_sha256(matrix.tolist()),
        "profile_matrix_source_id": "shared_control_cell_profiles.npy",
        "profile_matrix_file_sha256": matrix_file_sha256,
        "original_control_profile": matrix.mean(axis=0).tolist(),
        "original_control_profile_sha256": canonical_sha256(matrix.mean(axis=0).tolist()),
        "dataset_sha256": dataset_sha256,
        "preprocessing_state_sha256": preprocessing_state_sha256,
        "selection_rule": "all_and_only_hash_bound_resolved_control_cells",
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload, matrix_bytes


def write_shared_control_evidence(
    directory: Path, manifest: Mapping[str, Any], matrix_bytes: bytes
) -> tuple[Path, Path]:
    """Atomically write and revalidate one evidence pair."""

    directory.mkdir(parents=True, exist_ok=True)
    matrix_path = directory / str(manifest["profile_matrix_source_id"])
    matrix_temporary = matrix_path.with_name(f".{matrix_path.name}.tmp")
    matrix_temporary.write_bytes(matrix_bytes)
    os.replace(matrix_temporary, matrix_path)
    manifest_path = directory / "shared_control_cell_evidence.json"
    manifest_temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    manifest_temporary.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(manifest_temporary, manifest_path)
    read_shared_control_evidence(manifest_path)
    return manifest_path, matrix_path


def read_shared_control_evidence(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    """Validate manifest, path containment, retained bytes, shape, hashes, and mean."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RevisionProtocolError("[SHARED_CONTROL_MANIFEST_INVALID]") from error
    required = {
        "schema_version",
        "evidence_id",
        "dataset",
        "hvg",
        "vector_space",
        "gene_names",
        "gene_names_sha256",
        "control_row_ids",
        "control_row_ids_sha256",
        "n_control_cells",
        "profile_matrix_shape",
        "profile_matrix_content_sha256",
        "profile_matrix_source_id",
        "profile_matrix_file_sha256",
        "original_control_profile",
        "original_control_profile_sha256",
        "dataset_sha256",
        "preprocessing_state_sha256",
        "selection_rule",
        "manifest_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise RevisionProtocolError("[SHARED_CONTROL_MANIFEST_SCHEMA_INVALID]")
    unsigned = dict(payload)
    declared_hash = unsigned.pop("manifest_sha256")
    source = Path(str(payload["profile_matrix_source_id"]))
    resolved = (path.resolve().parent / source).resolve()
    if (
        declared_hash != canonical_sha256(unsigned)
        or payload["schema_version"] != "1.0"
        or payload["evidence_id"] != "SHARED-CONTROL-CELL-PROFILES"
        or payload["vector_space"] != "control_fitted_standardized_expression"
        or source.is_absolute()
        or ".." in source.parts
        or not resolved.is_relative_to(path.resolve().parent)
        or not resolved.is_file()
        or file_sha256(resolved) != payload["profile_matrix_file_sha256"]
    ):
        raise RevisionProtocolError("[SHARED_CONTROL_MANIFEST_BINDING_INVALID]")
    try:
        matrix = np.load(resolved, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise RevisionProtocolError("[SHARED_CONTROL_MATRIX_INVALID]") from error
    genes = tuple(payload["gene_names"])
    rows = tuple(payload["control_row_ids"])
    mean = np.asarray(payload["original_control_profile"], dtype=np.float64)
    hvg = payload["hvg"]
    if (
        isinstance(hvg, bool)
        or not isinstance(hvg, int)
        or hvg <= 0
        or len(genes) != hvg
        or len(set(genes)) != hvg
        or len(rows) < 2
        or len(set(rows)) != len(rows)
        or payload["n_control_cells"] != len(rows)
        or payload["profile_matrix_shape"] != [len(rows), hvg]
        or matrix.shape != (len(rows), hvg)
        or not np.isfinite(matrix).all()
        or payload["gene_names_sha256"] != canonical_sha256(genes)
        or payload["control_row_ids_sha256"] != canonical_sha256(rows)
        or payload["profile_matrix_content_sha256"] != canonical_sha256(matrix.tolist())
        or mean.shape != (hvg,)
        or not np.array_equal(mean, matrix.mean(axis=0))
        or payload["original_control_profile_sha256"] != canonical_sha256(mean.tolist())
        or payload["selection_rule"] != "all_and_only_hash_bound_resolved_control_cells"
        or not SHA256_RE.fullmatch(str(payload["dataset_sha256"]))
        or not SHA256_RE.fullmatch(str(payload["preprocessing_state_sha256"]))
    ):
        raise RevisionProtocolError("[SHARED_CONTROL_EVIDENCE_SEMANTICS_INVALID]")
    return payload, np.asarray(matrix, dtype=np.float64)


def attach_shared_control_bootstrap(
    primary: Any,
    *,
    artifact_paths: Sequence[Path],
    evidence_manifest_paths: Sequence[Path],
    n_bootstrap: int,
    random_seed: int,
) -> Any:
    """Recompute the primary contrast while resampling shared controls and conditions."""

    if not primary.released:
        return primary
    evidence = [read_shared_control_evidence(path) for path in evidence_manifest_paths]
    evidence_keys = [(str(manifest["dataset"]), int(manifest["hvg"])) for manifest, _ in evidence]
    if len(evidence_keys) != len(set(evidence_keys)):
        raise RevisionProtocolError("[SHARED_CONTROL_DATASET_HVG_DUPLICATE]")
    primary_evidence = [
        (manifest, matrix) for manifest, matrix in evidence if manifest["hvg"] == 200
    ]
    by_dataset = {
        str(manifest["dataset"]): (manifest, matrix) for manifest, matrix in primary_evidence
    }
    datasets = tuple(
        sorted(
            str(value) for value in primary.detail_tables["condition_contrasts"]["dataset"].unique()
        )
    )
    if set(by_dataset) != set(datasets):
        raise RevisionProtocolError("[SHARED_CONTROL_DATASET_SET_MISMATCH]")
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap < 100:
        raise RevisionProtocolError("[SHARED_CONTROL_BOOTSTRAP_COUNT_INVALID]")
    rows = []
    for path in sorted({Path(value).resolve() for value in artifact_paths}):
        payload = read_fold_artifact(path)
        identity = payload["identity"]
        if (
            identity["hvg"] != 200
            or payload["config"]["panel"] != "primary"
            or identity["arm"] not in {"string_go", "dense"}
        ):
            continue
        manifest, _ = by_dataset.get(str(identity["dataset"]), ({}, np.empty((0, 0))))
        if (
            tuple(payload["gene_names"]) != tuple(manifest.get("gene_names", ()))
            or payload["preprocessing_state_hash"] != manifest.get("preprocessing_state_sha256")
            or payload["input_hashes"].get("dataset") != manifest.get("dataset_sha256")
            or payload["input_hashes"].get("shared_control_cell_evidence")
            != manifest.get("manifest_sha256")
        ):
            raise RevisionProtocolError("[SHARED_CONTROL_FOLD_EVIDENCE_BINDING_INVALID]")
        rows.append(
            {
                **identity,
                "gene_names": tuple(payload["gene_names"]),
                "y_true": np.asarray(payload["y_true"], dtype=np.float64),
                "y_pred": np.asarray(payload["y_pred"], dtype=np.float64),
            }
        )
    expected = len(datasets) * 50 * 3 * 2
    if len(rows) != expected:
        raise RevisionProtocolError("[SHARED_CONTROL_PRIMARY_FOLD_SUPPORT_INCOMPLETE]")
    seed_sets = {
        tuple(
            sorted(
                int(row["seed"])
                for row in rows
                if row["dataset"] == dataset and row["condition"] == condition and row["arm"] == arm
            )
        )
        for dataset in datasets
        for condition in sorted({row["condition"] for row in rows if row["dataset"] == dataset})
        for arm in ("string_go", "dense")
    }
    if len(seed_sets) != 1 or len(next(iter(seed_sets), ())) != 3:
        raise RevisionProtocolError("[SHARED_CONTROL_SEED_SUPPORT_INVALID]")
    seeds = next(iter(seed_sets))
    rng = np.random.default_rng(random_seed)
    draws = np.empty(n_bootstrap, dtype=np.float64)
    digest = hashlib.sha256()
    dataset_rows = {
        dataset: [row for row in rows if row["dataset"] == dataset] for dataset in datasets
    }
    for replicate in range(n_bootstrap):
        dataset_effects = []
        for dataset in datasets:
            manifest, controls = by_dataset[dataset]
            control_indices = rng.integers(0, len(controls), len(controls))
            shift = np.asarray(manifest["original_control_profile"]) - controls[
                control_indices
            ].mean(axis=0)
            digest.update(dataset.encode("utf-8"))
            digest.update(control_indices.astype(np.int64).tobytes())
            fold_metrics: dict[tuple[str, str, int], float] = {}
            for row in dataset_rows[dataset]:
                fold_metrics[(row["condition"], row["arm"], row["seed"])] = _pearson(
                    row["y_true"] + shift,
                    row["y_pred"] + shift,
                )
            conditions = sorted({row["condition"] for row in dataset_rows[dataset]})
            condition_indices = rng.integers(0, len(conditions), len(conditions))
            digest.update(condition_indices.astype(np.int64).tobytes())
            condition_effects = []
            for index in condition_indices:
                condition = conditions[int(index)]
                string_go = np.mean(
                    [fold_metrics[(condition, "string_go", seed)] for seed in seeds]
                )
                dense = np.mean([fold_metrics[(condition, "dense", seed)] for seed in seeds])
                condition_effects.append(float(string_go - dense))
            dataset_effects.append(float(np.mean(condition_effects)))
        draws[replicate] = float(np.mean(dataset_effects))
    low, high = np.quantile(draws, [0.025, 0.975])
    registry = deepcopy(primary.registry)
    registry["shared_control_cell_bootstrap"] = {
        "status": "MEASURED",
        "estimate": float(np.mean(draws)),
        "uncertainty_interval_95_low": float(low),
        "uncertainty_interval_95_high": float(high),
        "interval_label": "95% conditional bootstrap uncertainty interval",
        "resampling_scheme": "shared_control_cells_and_guide_conditions_within_dataset",
        "bootstrap_replicates": n_bootstrap,
        "bootstrap_random_seed": random_seed,
        "neural_seed_support": list(seeds),
        "bootstrap_index_hash": digest.hexdigest(),
        "evidence_manifest_hashes": {
            dataset: by_dataset[dataset][0]["manifest_sha256"] for dataset in datasets
        },
    }
    return type(primary)(
        registry=registry, detail_tables=primary.detail_tables, failures=primary.failures
    )


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if float(np.std(left)) <= 0 or float(np.std(right)) <= 0:
        raise RevisionProtocolError("[SHARED_CONTROL_BOOTSTRAP_DEGENERATE_VECTOR]")
    value = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(value):
        raise RevisionProtocolError("[SHARED_CONTROL_BOOTSTRAP_NONFINITE]")
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant {value!r} is prohibited")

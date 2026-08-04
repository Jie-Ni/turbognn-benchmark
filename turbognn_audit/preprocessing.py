"""Fail-closed preprocessing fitted only on permitted training-side cells."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import jsonschema
import numpy as np

from .hashing import sha256_json


@dataclass(frozen=True)
class PreprocessingSpec:
    """Explicit fit scopes for every outcome-dependent preprocessing operation."""

    input_expression_state: str
    input_expression_evidence: str
    normalization_method: str
    normalize_total_target_sum: float
    log1p_pseudocount: float
    log1p_base: float | None
    hvg_fit_scope: str
    scaler_fit_scope: str
    coexpression_fit_scope: str
    hvg_method: str
    hvg_flavor: str
    scaler_ddof: int
    scaler_epsilon: float
    scaler_constant_scale: float
    coexpression_method: str
    coexpression_threshold: float
    coexpression_threshold_rule: str
    coexpression_constant_nan_policy: str
    coexpression_symmetry_policy: str
    coexpression_self_loop_policy: str
    cell_qc_policy: str
    perturbation_type_policy: str
    batch_policy: str
    multi_target_policy: str
    minimum_control_cells: int
    minimum_perturbed_cells: int

    def validate(self) -> None:
        """Reject scopes that can include held-out outcome cells in the current runner."""
        if self.input_expression_state not in {"raw_counts", "log1p_normalized"}:
            raise ValueError("input_expression_state must be 'raw_counts' or 'log1p_normalized'")
        if (
            not self.input_expression_evidence.strip()
            or "TODO" in self.input_expression_evidence.upper()
        ):
            raise ValueError("input_expression_evidence must be source-verified")
        expected_normalization = {
            "raw_counts": "scanpy_normalize_total_then_log1p",
            "log1p_normalized": "identity_preverified_log1p",
        }[self.input_expression_state]
        if self.normalization_method != expected_normalization:
            raise ValueError(
                f"normalization_method must be {expected_normalization!r} for "
                f"{self.input_expression_state!r} input"
            )
        if self.normalize_total_target_sum != 10_000.0:
            raise ValueError("normalize_total_target_sum must be exactly 10000")
        if self.log1p_pseudocount != 1.0 or self.log1p_base is not None:
            raise ValueError("log1p must use pseudocount 1 and natural-log base")
        if self.hvg_fit_scope != "control_only":
            raise ValueError("hvg_fit_scope must be 'control_only'")
        if self.scaler_fit_scope != "control_only":
            raise ValueError("scaler_fit_scope must be 'control_only'")
        if self.coexpression_fit_scope not in {"control_only", "training_only"}:
            raise ValueError("coexpression_fit_scope must be 'control_only' or 'training_only'")
        if self.hvg_method != "scanpy_highly_variable_genes" or self.hvg_flavor != "seurat":
            raise ValueError(
                "HVG selection must freeze Scanpy highly_variable_genes flavor='seurat'"
            )
        if self.scaler_ddof != 0:
            raise ValueError("scaler_ddof must be exactly 0")
        if self.scaler_epsilon != 1e-8 or self.scaler_constant_scale != 1.0:
            raise ValueError("Scaler epsilon/constant scale must be exactly 1e-8/1.0")
        if self.coexpression_method != "pearson":
            raise ValueError("coexpression_method must be 'pearson'")
        if not 0.0 < self.coexpression_threshold < 1.0:
            raise ValueError("coexpression_threshold must lie strictly between 0 and 1")
        if self.coexpression_threshold_rule != "strict_abs_gt":
            raise ValueError("coexpression_threshold_rule must be 'strict_abs_gt'")
        if self.coexpression_constant_nan_policy != "exclude_edge":
            raise ValueError("coexpression_constant_nan_policy must be 'exclude_edge'")
        if self.coexpression_symmetry_policy != "simple_undirected_then_bidirectional":
            raise ValueError(
                "coexpression_symmetry_policy must be 'simple_undirected_then_bidirectional'"
            )
        if self.coexpression_self_loop_policy != "one_per_gene":
            raise ValueError("coexpression_self_loop_policy must be 'one_per_gene'")
        if self.cell_qc_policy != "source_filtered_matrix_no_additional_cell_filter":
            raise ValueError(
                "cell_qc_policy must be 'source_filtered_matrix_no_additional_cell_filter'"
            )
        if self.perturbation_type_policy != "explicit_mapped_single_or_multi_target":
            raise ValueError(
                "perturbation_type_policy must be 'explicit_mapped_single_or_multi_target'"
            )
        if self.batch_policy != "no_batch_correction":
            raise ValueError("batch_policy must be 'no_batch_correction'")
        if self.multi_target_policy != "retain_all_explicit_mapped_targets":
            raise ValueError("multi_target_policy must be 'retain_all_explicit_mapped_targets'")
        if self.minimum_control_cells < 2:
            raise ValueError("minimum_control_cells must be at least 2")
        if self.minimum_perturbed_cells != 20:
            raise ValueError("minimum_perturbed_cells must be exactly the frozen value 20")


@dataclass(frozen=True)
class ControlScaler:
    """Per-gene mean and standard deviation fitted on control cells only."""

    mean: np.ndarray
    scale: np.ndarray
    fit_cell_count: int
    scaler_hash: str

    def transform(self, expression: np.ndarray) -> np.ndarray:
        """Apply the frozen control-only scaling parameters."""
        matrix = np.asarray(expression, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.mean):
            raise ValueError("Expression matrix is incompatible with the fitted scaler")
        return (matrix - self.mean) / self.scale


def load_preprocessing_spec(path: Path) -> PreprocessingSpec:
    """Load and validate an explicit split-safe preprocessing configuration."""
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("Preprocessing config must be a JSON object")
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "preprocessing.schema.json"
    with schema_path.open(encoding="utf-8") as handle:
        schema = json.load(handle)
    try:
        jsonschema.validate(raw, schema)
    except jsonschema.ValidationError as error:
        raise ValueError(f"Preprocessing config fails its JSON schema: {error.message}") from error
    if raw.get("schema_version") != "1.0.0":
        raise ValueError("Unsupported preprocessing schema_version")
    required = (
        "input_expression_state",
        "input_expression_evidence",
        "normalization_method",
        "normalize_total_target_sum",
        "log1p_pseudocount",
        "log1p_base",
        "hvg_fit_scope",
        "scaler_fit_scope",
        "coexpression_fit_scope",
        "hvg_method",
        "hvg_flavor",
        "scaler_ddof",
        "scaler_epsilon",
        "scaler_constant_scale",
        "coexpression_method",
        "coexpression_threshold",
        "coexpression_threshold_rule",
        "coexpression_constant_nan_policy",
        "coexpression_symmetry_policy",
        "coexpression_self_loop_policy",
        "cell_qc_policy",
        "perturbation_type_policy",
        "batch_policy",
        "multi_target_policy",
        "minimum_control_cells",
        "minimum_perturbed_cells",
    )
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"Preprocessing config lacks fields: {missing}")
    spec = PreprocessingSpec(
        input_expression_state=str(raw["input_expression_state"]),
        input_expression_evidence=str(raw["input_expression_evidence"]),
        normalization_method=str(raw["normalization_method"]),
        normalize_total_target_sum=float(raw["normalize_total_target_sum"]),
        log1p_pseudocount=float(raw["log1p_pseudocount"]),
        log1p_base=(float(raw["log1p_base"]) if raw["log1p_base"] is not None else None),
        hvg_fit_scope=str(raw["hvg_fit_scope"]),
        scaler_fit_scope=str(raw["scaler_fit_scope"]),
        coexpression_fit_scope=str(raw["coexpression_fit_scope"]),
        hvg_method=str(raw["hvg_method"]),
        hvg_flavor=str(raw["hvg_flavor"]),
        scaler_ddof=int(raw["scaler_ddof"]),
        scaler_epsilon=float(raw["scaler_epsilon"]),
        scaler_constant_scale=float(raw["scaler_constant_scale"]),
        coexpression_method=str(raw["coexpression_method"]),
        coexpression_threshold=float(raw["coexpression_threshold"]),
        coexpression_threshold_rule=str(raw["coexpression_threshold_rule"]),
        coexpression_constant_nan_policy=str(raw["coexpression_constant_nan_policy"]),
        coexpression_symmetry_policy=str(raw["coexpression_symmetry_policy"]),
        coexpression_self_loop_policy=str(raw["coexpression_self_loop_policy"]),
        cell_qc_policy=str(raw["cell_qc_policy"]),
        perturbation_type_policy=str(raw["perturbation_type_policy"]),
        batch_policy=str(raw["batch_policy"]),
        multi_target_policy=str(raw["multi_target_policy"]),
        minimum_control_cells=int(raw["minimum_control_cells"]),
        minimum_perturbed_cells=int(raw["minimum_perturbed_cells"]),
    )
    spec.validate()
    return spec


def _boolean_mask(mask: Sequence[bool], row_count: int, name: str) -> np.ndarray:
    array = np.asarray(mask, dtype=bool)
    if array.shape != (row_count,):
        raise ValueError(f"{name} must have shape ({row_count},), got {array.shape}")
    return array


def fit_control_scaler(
    expression: np.ndarray,
    control_mask: Sequence[bool],
    *,
    minimum_control_cells: int,
    ddof: int,
    epsilon: float,
    constant_scale: float,
) -> ControlScaler:
    """Fit scaling on control cells only; non-control outcomes cannot affect the fit."""
    matrix = np.asarray(expression, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError("expression must have shape [cells, genes]")
    controls = _boolean_mask(control_mask, matrix.shape[0], "control_mask")
    fit_cell_count = int(controls.sum())
    if fit_cell_count < minimum_control_cells:
        raise ValueError(f"Only {fit_cell_count} control cells; minimum is {minimum_control_cells}")
    fit_matrix = matrix[controls]
    mean = fit_matrix.mean(axis=0)
    if ddof != 0 or epsilon != 1e-8 or constant_scale != 1.0:
        raise ValueError("Canonical scaler requires ddof=0, epsilon=1e-8, constant_scale=1.0")
    scale = fit_matrix.std(axis=0, ddof=ddof)
    scale = np.where(scale > epsilon, scale, constant_scale)
    scaler_hash = sha256_json(
        {
            "fit_scope": "control_only",
            "fit_cell_count": fit_cell_count,
            "ddof": ddof,
            "epsilon": epsilon,
            "constant_scale": constant_scale,
            "mean": mean.tolist(),
            "scale": scale.tolist(),
        }
    )
    return ControlScaler(
        mean=mean,
        scale=scale,
        fit_cell_count=fit_cell_count,
        scaler_hash=scaler_hash,
    )


def rank_genes_by_control_variance(
    expression: np.ndarray,
    gene_names: Sequence[str],
    control_mask: Sequence[bool],
) -> tuple[str, ...]:
    """Toy/control selector for mutation tests; ranks variance using controls only."""
    matrix = np.asarray(expression, dtype=np.float64)
    genes = tuple(str(gene) for gene in gene_names)
    if matrix.ndim != 2 or matrix.shape[1] != len(genes):
        raise ValueError("Expression columns and gene_names must have the same length")
    if len(set(genes)) != len(genes):
        raise ValueError("gene_names must be unique")
    controls = _boolean_mask(control_mask, matrix.shape[0], "control_mask")
    if int(controls.sum()) < 2:
        raise ValueError("Control-only ranking requires at least two control cells")
    variances = matrix[controls].var(axis=0)
    ranked_indices = sorted(range(len(genes)), key=lambda index: (-variances[index], genes[index]))
    return tuple(genes[index] for index in ranked_indices)


def select_coexpression_fit_rows(
    expression: np.ndarray,
    *,
    scope: str,
    control_mask: Sequence[bool],
    training_mask: Sequence[bool] | None = None,
    held_out_mask: Sequence[bool] | None = None,
) -> tuple[np.ndarray, str]:
    """Select an explicit control/training-only matrix and prove held-out exclusion."""
    matrix = np.asarray(expression, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("expression must have shape [cells, genes]")
    controls = _boolean_mask(control_mask, matrix.shape[0], "control_mask")
    held_out = (
        _boolean_mask(held_out_mask, matrix.shape[0], "held_out_mask")
        if held_out_mask is not None
        else np.zeros(matrix.shape[0], dtype=bool)
    )
    if scope == "control_only":
        selected = controls
    elif scope == "training_only":
        if training_mask is None or held_out_mask is None:
            raise ValueError("training_only coexpression requires training_mask and held_out_mask")
        selected = _boolean_mask(training_mask, matrix.shape[0], "training_mask")
    else:
        raise ValueError("Coexpression fit scope must be 'control_only' or 'training_only'")
    if np.any(selected & held_out):
        raise ValueError("Coexpression fit rows overlap held-out outcome cells")
    if int(selected.sum()) < 2:
        raise ValueError("Coexpression fitting requires at least two permitted cells")
    fit_matrix = matrix[selected]
    fit_hash = sha256_json(
        {
            "scope": scope,
            "row_count": int(selected.sum()),
            "matrix": fit_matrix.tolist(),
        }
    )
    return fit_matrix, fit_hash


def preprocessing_fit_hash(
    *,
    gene_panel: Sequence[str],
    scaler: ControlScaler,
    coexpression_fit_hash: str,
    target_mapping_hash: str,
    control_selection_hash: str,
    condition_eligibility_ledger_hash: str,
    spec: PreprocessingSpec,
) -> str:
    """Hash only fitted preprocessing state, excluding all transformed outcomes."""
    return sha256_json(
        {
            "gene_panel": list(gene_panel),
            "scaler_hash": scaler.scaler_hash,
            "coexpression_fit_hash": coexpression_fit_hash,
            "target_mapping_hash": target_mapping_hash,
            "control_selection_hash": control_selection_hash,
            "condition_eligibility_ledger_hash": condition_eligibility_ledger_hash,
            "spec": {
                "input_expression_state": spec.input_expression_state,
                "input_expression_evidence": spec.input_expression_evidence,
                "normalization_method": spec.normalization_method,
                "normalize_total_target_sum": spec.normalize_total_target_sum,
                "log1p_pseudocount": spec.log1p_pseudocount,
                "log1p_base": spec.log1p_base,
                "hvg_fit_scope": spec.hvg_fit_scope,
                "scaler_fit_scope": spec.scaler_fit_scope,
                "coexpression_fit_scope": spec.coexpression_fit_scope,
                "hvg_method": spec.hvg_method,
                "hvg_flavor": spec.hvg_flavor,
                "scaler_ddof": spec.scaler_ddof,
                "scaler_epsilon": spec.scaler_epsilon,
                "scaler_constant_scale": spec.scaler_constant_scale,
                "coexpression_method": spec.coexpression_method,
                "coexpression_threshold": spec.coexpression_threshold,
                "coexpression_threshold_rule": spec.coexpression_threshold_rule,
                "coexpression_constant_nan_policy": spec.coexpression_constant_nan_policy,
                "coexpression_symmetry_policy": spec.coexpression_symmetry_policy,
                "coexpression_self_loop_policy": spec.coexpression_self_loop_policy,
                "cell_qc_policy": spec.cell_qc_policy,
                "perturbation_type_policy": spec.perturbation_type_policy,
                "batch_policy": spec.batch_policy,
                "multi_target_policy": spec.multi_target_policy,
                "minimum_control_cells": spec.minimum_control_cells,
                "minimum_perturbed_cells": spec.minimum_perturbed_cells,
                "target_retention_policy": (
                    "held_out_identity_may_retain_explicit_targets; "
                    "held_out_expression_and_outcome_rankings_forbidden"
                ),
            },
        }
    )


def assert_held_out_mutation_invariant(before_hash: str, after_hash: str) -> None:
    """Fail if mutating held-out outcomes changes any fitted preprocessing state."""
    if before_hash != after_hash:
        raise ValueError(
            "Held-out mutation changed fitted preprocessing state; outcome leakage is present"
        )

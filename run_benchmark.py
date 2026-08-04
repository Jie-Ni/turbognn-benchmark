"""Strict one-row LOPO runner for a frozen revision schedule.

Every invocation requires explicit dataset, target, preprocessing, graph, model,
environment, code-commit, schedule, seed, and fold identities. No implicit quick mode,
placeholder configuration, overwrite, or unscheduled graph arm is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

try:
    import scanpy as sc
except ModuleNotFoundError:
    sc = None

# ---------------------------------------------------------------------------
# Import models from turbognn_v2_models.py (same directory)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from slurm.download_data import (
    DatasetPassport,
    load_dataset_passports,
    validate_dataset_file,
)
from turbognn_audit.controls import ControlSpec, load_control_map, resolve_control
from turbognn_audit.graph_sources import (
    FrozenGraphSource,
    build_frozen_edge_graph,
    load_frozen_graph_sources,
)
from turbognn_audit.graphs import (
    GraphEnsembleSpec,
    build_barabasi_albert_instance,
    build_degree_preserving_rewired_instance,
    edge_hash,
    graph_diagnostics,
    graph_provenance,
    load_ensemble_specs,
    self_loop_edge_index,
)
from turbognn_audit.hashing import canonical_json, sha256_file, sha256_json
from turbognn_audit.io import write_json_atomic
from turbognn_audit.manifest import RunManifest, RunRecord
from turbognn_audit.model_config import ModelSpec, load_model_spec
from turbognn_audit.panels import (
    ConditionPanel,
    build_canonical_condition_eligibility_panel,
    canonical_condition_label,
    limit_condition_panel,
)
from turbognn_audit.preprocessing import (
    PreprocessingSpec,
    fit_control_scaler,
    load_preprocessing_spec,
    preprocessing_fit_hash,
    select_coexpression_fit_rows,
)
from turbognn_audit.results import build_fold_result
from turbognn_audit.targets import (
    TargetMapSpec,
    load_target_maps,
    resolve_condition_targets,
    target_preserving_gene_panel,
    validate_condition_targets_in_panel,
)
from turbognn_v2_models import SimpleTransformer, TurboGNN

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOME = Path(os.path.expanduser("~"))
_DATA_BASE = Path(os.environ.get("TURBOGNN_DATA_DIR", str(HOME / "TurboGNN" / "data")))
DATA_DIR_PROCESSED = _DATA_BASE / "processed"
DATA_DIR_SCPERTURB = _DATA_BASE / "scperturb"
RESULTS_DIR = Path(__file__).resolve().parent / "results_benchmark"

NUM_HVG_DEFAULT = 200
FOLD_ORDER_SEED_POLICY = "sha256_training_seed_condition_split_v1"
CONDITION_ORDER_POLICY = "sha256_fold_seed_epoch_nfc_condition_v1"
FOLD_ORDER_SEED_POLICY_HASH = sha256_json(
    {
        "policy": FOLD_ORDER_SEED_POLICY,
        "canonical_json_keys": [
            "policy",
            "training_seed",
            "held_out_condition",
            "split_hash",
        ],
        "digest_projection": "first_8_hex_as_base16_integer",
    }
)
CONDITION_ORDER_POLICY_HASH = sha256_json(
    {
        "policy": CONDITION_ORDER_POLICY,
        "canonical_json_keys": [
            "policy",
            "fold_order_seed",
            "epoch",
            "condition",
        ],
        "ordering": "ascending_sha256_hex",
    }
)
CANONICAL_DATASETS = ("adamson", "norman", "replogle_k562", "replogle_rpe1")
DATASET_PATHS: dict[str, Path] = {}
DATASET_PASSPORTS: dict[str, DatasetPassport] = {}

GRAPH_TYPES = [
    "string_ppi",
    "gene_ontology",
    "coexpression",
    "string_go_union",
    "degree_preserving_rewired",
    "barabasi_albert",
    "self_loop_gat",
    "transformer",
]
DEFAULT_GRAPH_TYPES = [
    "string_ppi",
    "gene_ontology",
    "coexpression",
    "string_go_union",
    "self_loop_gat",
    "transformer",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("benchmark")

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_partial_results: dict[str, dict] = {}
_results_dir: Path = RESULTS_DIR
_active_manifest_path: Path | None = None
_active_manifest_records: tuple[RunRecord, ...] = ()


def _save_partial(tag: str = "partial") -> None:
    """Persist whatever results we have so far."""
    if not _partial_results:
        return
    _results_dir.mkdir(parents=True, exist_ok=True)
    out = _results_dir / f"results_{tag}_{os.getpid()}.json"
    write_json_atomic(out, _partial_results, overwrite=out.exists())
    logger.info("Saved %d result entries to %s", len(_partial_results), out)


def _sigterm_handler(signum, frame):
    del frame
    logger.error("Signal %d received — persisting terminal failed manifest.", signum)
    if _active_manifest_path is not None and _active_manifest_records:
        failed = tuple(
            replace(
                record,
                status="failed",
                result_path=None,
                failure_reason=f"terminated_by_signal_{signum}",
            )
            for record in _active_manifest_records
        )
        RunManifest.build(failed).write(_active_manifest_path, overwrite=True)
    _save_partial("sigterm")
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, _sigterm_handler)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_dataset(name: str, preprocessing_spec: PreprocessingSpec) -> sc.AnnData:
    """Load and normalize one dataset without fitting across cells."""
    if sc is None:
        raise RuntimeError(
            "scanpy is required to load datasets; install the benchmark environment first"
        )
    path = DATASET_PATHS[name]
    logger.info("Loading dataset '%s' from %s", name, path)
    adata = sc.read_h5ad(str(path))
    adata = select_passport_matrix(adata, DATASET_PASSPORTS[name])

    if preprocessing_spec.input_expression_state == "raw_counts":
        sc.pp.normalize_total(adata, target_sum=preprocessing_spec.normalize_total_target_sum)
        sc.pp.log1p(adata, base=preprocessing_spec.log1p_base)
    elif preprocessing_spec.input_expression_state != "log1p_normalized":
        raise ValueError(
            f"Unsupported input_expression_state: {preprocessing_spec.input_expression_state!r}"
        )

    return adata


def select_passport_matrix(adata: sc.AnnData, passport: DatasetPassport) -> sc.AnnData:
    """Bind the exact passport layer to canonical ``X`` before any transformation."""
    if passport.matrix_layer == "X":
        matrix = adata.X
    else:
        layer_name = passport.matrix_layer.removeprefix("layers/")
        if layer_name not in adata.layers:
            raise ValueError(f"Frozen matrix layer {layer_name!r} is absent")
        matrix = adata.layers[layer_name]
        adata.X = matrix.copy()
        matrix = adata.X
    observed_dtype = np.dtype(matrix.dtype).name
    if observed_dtype != passport.matrix_dtype:
        raise ValueError(
            f"Frozen matrix dtype drift: {observed_dtype!r} != {passport.matrix_dtype!r}"
        )
    if matrix.shape != (passport.n_obs, passport.n_vars):
        raise ValueError("Frozen matrix shape differs from its dataset passport")
    return adata


def extract_perturbations(
    adata: sc.AnnData,
    dataset: str,
    control_map: Mapping[str, ControlSpec],
    target_maps: Mapping[str, TargetMapSpec],
    preprocessing_spec: PreprocessingSpec,
) -> tuple[str, list[str], str, ConditionPanel]:
    """Resolve an explicit control and build a deterministic condition panel."""
    configured = control_map.get(dataset)
    if configured is None:
        raise ValueError(
            f"Dataset {dataset!r} has no explicit control mapping; "
            "modal-label fallback is forbidden"
        )
    if configured.perturbation_column not in adata.obs.columns:
        raise ValueError(
            f"Configured perturbation column {configured.perturbation_column!r} is absent from "
            f"dataset {dataset!r}"
        )
    observed = adata.obs[configured.perturbation_column].astype(str)
    noncanonical = sorted(
        {label for label in observed.unique() if canonical_condition_label(label) != label}
    )
    if noncanonical:
        raise ValueError(
            f"Dataset {dataset!r} contains non-canonical condition labels; "
            f"curate them explicitly before benchmarking: {noncanonical[:10]}"
        )
    spec = resolve_control(dataset, adata.obs.columns, observed.unique(), control_map)
    target_spec = target_maps.get(dataset)
    if target_spec is None:
        raise ValueError(f"Dataset {dataset!r} lacks an explicit target-mapping passport")
    counts = observed.value_counts(sort=False).to_dict()
    panel = build_canonical_condition_eligibility_panel(
        dataset=dataset,
        pre_qc_label_counts=counts,
        post_qc_label_counts=counts,
        control_label=spec.control_label,
        minimum_cells=preprocessing_spec.minimum_perturbed_cells,
        raw_to_canonical_id=target_spec.raw_to_canonical_id,
        condition_targets=target_spec.condition_targets,
        canonical_objects=target_spec.canonical_objects,
        raw_normalization_evidence=target_spec.raw_normalization_evidence,
        raw_record_hashes=target_spec.raw_record_hashes,
        allowed_perturbation_types=target_spec.allowed_perturbation_types,
        available_genes=[str(gene) for gene in adata.var_names],
        cell_qc_policy=preprocessing_spec.cell_qc_policy,
        perturbation_type_policy=preprocessing_spec.perturbation_type_policy,
    )
    expected_ledger_hash = DATASET_PASSPORTS[dataset].condition_eligibility_ledger_sha256
    if panel.eligibility_ledger_hash != expected_ledger_hash:
        raise ValueError(
            f"Dataset {dataset!r} eligibility-ledger hash mismatch: "
            f"{panel.eligibility_ledger_hash} != {expected_ledger_hash}"
        )
    return (
        spec.perturbation_column,
        list(panel.conditions),
        spec.control_label,
        panel,
    )


def rank_control_only_hvgs(
    adata: sc.AnnData,
    control_mask: np.ndarray,
    num_hvg: int,
    method: str,
) -> list[str]:
    """Rank HVGs using control cells only and a declared Scanpy method."""
    if method != "scanpy_highly_variable_genes":
        raise ValueError(f"Unsupported hvg_method {method!r}; no method fallback is permitted")
    if num_hvg < 1 or num_hvg > adata.n_vars:
        raise ValueError(f"num_hvg must be within [1, {adata.n_vars}], got {num_hvg}")
    control_adata = adata[control_mask].copy()
    sc.pp.highly_variable_genes(
        control_adata,
        n_top_genes=num_hvg,
        flavor="seurat",
        inplace=True,
    )
    required = {"highly_variable", "dispersions_norm"}
    missing = required - set(control_adata.var.columns)
    if missing:
        raise RuntimeError(f"Scanpy HVG output lacks fields: {sorted(missing)}")
    ranking = pd.DataFrame(
        {
            "gene": [str(gene) for gene in control_adata.var_names],
            "selected": control_adata.var["highly_variable"].astype(bool).to_numpy(),
            "score": control_adata.var["dispersions_norm"].astype(float).to_numpy(),
        }
    )
    ranking = ranking.loc[ranking["selected"]].sort_values(
        ["score", "gene"], ascending=[False, True], kind="mergesort"
    )
    genes = ranking["gene"].tolist()
    if len(genes) < num_hvg:
        raise RuntimeError(
            f"Control-only HVG selection returned {len(genes)} genes, expected {num_hvg}"
        )
    return genes


def prepare_dataset(
    name: str,
    control_map: Mapping[str, ControlSpec],
    target_maps: Mapping[str, TargetMapSpec],
    preprocessing_spec: PreprocessingSpec,
    max_folds: int,
    num_hvg: int = NUM_HVG_DEFAULT,
) -> dict:
    """Fit a target-preserving gene panel, scaler, and graph inputs without held-out outcomes."""
    preprocessing_spec.validate()
    if preprocessing_spec.coexpression_fit_scope != "control_only":
        raise ValueError(
            "The shared-graph runner supports only control_only coexpression. "
            "training_only requires a fold-specific graph runner."
        )
    input_hash = sha256_file(DATASET_PATHS[name])
    adata = load_dataset(name, preprocessing_spec)
    pert_col, _conditions, ctrl_label, eligible_condition_panel = extract_perturbations(
        adata,
        name,
        control_map,
        target_maps,
        preprocessing_spec,
    )
    perturbation_labels = adata.obs[pert_col].astype(str)
    target_spec = target_maps[name]
    canonical_condition_labels = perturbation_labels.map(target_spec.raw_to_canonical_id)
    obs_ids = tuple(str(value) for value in adata.obs_names)
    if len(set(obs_ids)) != len(obs_ids):
        raise ValueError(f"Dataset {name!r} has non-unique stable observation identifiers")
    if any(
        not value or canonical_condition_label(value) != value or value != value.strip()
        for value in obs_ids
    ):
        raise ValueError(
            f"Dataset {name!r} observation identifiers must be non-empty, NFC, and unpadded"
        )
    obs_id_array = np.asarray(obs_ids, dtype=object)
    ctrl_mask = (perturbation_labels == ctrl_label).to_numpy()
    if int(ctrl_mask.sum()) < preprocessing_spec.minimum_control_cells:
        raise ValueError(
            f"Dataset {name!r} has only {int(ctrl_mask.sum())} configured control cells"
        )
    evaluation_panel = limit_condition_panel(eligible_condition_panel, max_folds)
    evaluation_conditions = list(evaluation_panel.conditions)
    if len(evaluation_conditions) != max_folds:
        raise ValueError(
            f"Dataset {name!r} has only {len(evaluation_conditions)} eligible conditions; "
            f"the frozen panel size is exactly {max_folds}"
        )
    condition_targets, target_mapping_hash = resolve_condition_targets(
        name,
        evaluation_conditions,
        [str(gene) for gene in adata.var_names],
        target_maps,
    )
    ranked_hvgs = rank_control_only_hvgs(
        adata,
        ctrl_mask,
        num_hvg,
        preprocessing_spec.hvg_method,
    )
    gene_panel = target_preserving_gene_panel(
        all_genes=[str(gene) for gene in adata.var_names],
        ranked_control_hvgs=ranked_hvgs,
        condition_targets=condition_targets,
        panel_size=num_hvg,
    )
    validate_condition_targets_in_panel(condition_targets, gene_panel)
    adata = adata[:, list(gene_panel)].copy()
    gene_list = np.asarray(gene_panel)
    num_genes = len(gene_list)
    X_np = (
        adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X, dtype=np.float64)
    )
    control_feature_mean = np.asarray(X_np[ctrl_mask].mean(axis=0), dtype=np.float32)
    control_feature_log_variance = np.asarray(
        np.log1p(X_np[ctrl_mask].var(axis=0, ddof=0)),
        dtype=np.float32,
    )
    if (
        not np.isfinite(control_feature_mean).all()
        or not np.isfinite(control_feature_log_variance).all()
    ):
        raise ValueError("Pre-centering control mean/log1p-variance features must be finite")
    control_feature_hash = sha256_json(
        {
            "expression_state": preprocessing_spec.input_expression_state,
            "mean": control_feature_mean.tolist(),
            "log1p_variance_ddof0": control_feature_log_variance.tolist(),
        }
    )
    scaler = fit_control_scaler(
        X_np,
        ctrl_mask,
        minimum_control_cells=preprocessing_spec.minimum_control_cells,
        ddof=preprocessing_spec.scaler_ddof,
        epsilon=preprocessing_spec.scaler_epsilon,
        constant_scale=preprocessing_spec.scaler_constant_scale,
    )
    X_norm = scaler.transform(X_np)
    X_coexpression_fit, coexpression_fit_hash = select_coexpression_fit_rows(
        X_norm,
        scope=preprocessing_spec.coexpression_fit_scope,
        control_mask=ctrl_mask,
    )
    control_selection_hash = sha256_json(
        {
            "dataset": name,
            "perturbation_column": pert_col,
            "control_label": ctrl_label,
            "control_evidence": control_map[name].evidence,
            "control_row_indices": np.flatnonzero(ctrl_mask).tolist(),
        }
    )
    preprocessing_hash = preprocessing_fit_hash(
        gene_panel=gene_panel,
        scaler=scaler,
        coexpression_fit_hash=coexpression_fit_hash,
        target_mapping_hash=target_mapping_hash,
        control_selection_hash=control_selection_hash,
        condition_eligibility_ledger_hash=str(eligible_condition_panel.eligibility_ledger_hash),
        spec=preprocessing_spec,
    )
    ctrl_np = X_norm[ctrl_mask].mean(axis=0)
    control_feature_mean_tensor = torch.tensor(
        control_feature_mean, dtype=torch.float32, device=DEVICE
    )
    control_feature_log_variance_tensor = torch.tensor(
        control_feature_log_variance, dtype=torch.float32, device=DEVICE
    )

    cond_profiles: dict[str, np.ndarray] = {}
    condition_alias_members: dict[str, tuple[str, ...]] = {}
    condition_cell_ids: dict[str, tuple[str, ...]] = {}
    for condition in evaluation_conditions:
        members = tuple(
            sorted(
                (
                    raw_label
                    for raw_label, canonical_id in target_spec.raw_to_canonical_id.items()
                    if canonical_id == condition
                ),
                key=lambda value: value.encode("utf-8"),
            )
        )
        if not members:
            raise RuntimeError(f"Canonical condition {condition!r} has no raw alias members")
        condition_alias_members[condition] = members
        mask = (canonical_condition_labels == condition).fillna(False).to_numpy(dtype=bool)
        condition_cell_ids[condition] = tuple(
            sorted(obs_id_array[mask].tolist(), key=lambda value: value.encode("utf-8"))
        )
        if int(mask.sum()) < eligible_condition_panel.minimum_cells:
            raise RuntimeError(
                f"Evaluated condition {condition!r} fell below the frozen cell-count threshold"
            )
        cond_profiles[condition] = X_norm[mask].mean(axis=0)
        if (
            cond_profiles[condition].shape != (num_genes,)
            or not np.isfinite(cond_profiles[condition]).all()
        ):
            raise ValueError(
                f"Condition {condition!r} has a malformed or non-finite response vector"
            )
    condition_response_set_hash = sha256_json(
        {
            "ordered_conditions": evaluation_conditions,
            "gene_order": [str(gene) for gene in gene_list],
            "condition_profiles": {
                condition: cond_profiles[condition].tolist() for condition in evaluation_conditions
            },
            "condition_alias_members": {
                condition: list(condition_alias_members[condition])
                for condition in evaluation_conditions
            },
            "condition_cell_set_hashes": {
                condition: sha256_json(list(condition_cell_ids[condition]))
                for condition in evaluation_conditions
            },
        }
    )
    response_definition_hash = sha256_json(
        {
            "response": "post_qc_condition_mean_scaled_expression",
            "control_reference": "post_qc_control_mean_scaled_expression",
            "delta": "condition_mean_minus_control_mean",
            "gene_order_hash": sha256_json([str(gene) for gene in gene_list]),
            "scaler_hash": scaler.scaler_hash,
        }
    )
    logger.info(
        "Dataset '%s': %d genes, %d deterministic evaluated conditions",
        name,
        num_genes,
        len(evaluation_conditions),
    )

    return {
        "adata": adata,
        "input_hash": input_hash,
        "dataset_passport_hash": DATASET_PASSPORTS[name].passport_hash,
        "matrix_contract_hash": sha256_json(
            {
                "matrix_layer": DATASET_PASSPORTS[name].matrix_layer,
                "matrix_dtype": DATASET_PASSPORTS[name].matrix_dtype,
                "input_expression_state": DATASET_PASSPORTS[name].input_expression_state,
                "dataset_sha256": input_hash,
            }
        ),
        "gene_list": gene_list,
        "num_genes": num_genes,
        "X_coexpression_fit": X_coexpression_fit,
        "ctrl_np": ctrl_np,
        "control_feature_mean_tensor": control_feature_mean_tensor,
        "control_feature_log_variance_tensor": control_feature_log_variance_tensor,
        "control_feature_hash": control_feature_hash,
        "cond_profiles": cond_profiles,
        "valid_conds": evaluation_conditions,
        "eligible_condition_panel": eligible_condition_panel,
        "condition_eligibility_ledger_hash": (eligible_condition_panel.eligibility_ledger_hash),
        "cell_qc_policy_hash": eligible_condition_panel.cell_qc_policy_hash,
        "condition_response_set_hash": condition_response_set_hash,
        "response_definition_hash": response_definition_hash,
        "evaluation_panel": evaluation_panel,
        "panel_size": max_folds,
        "condition_targets": condition_targets,
        "condition_alias_members": condition_alias_members,
        "condition_cell_ids": condition_cell_ids,
        "canonical_objects": {
            condition: target_spec.canonical_objects[condition]
            for condition in evaluation_conditions
        },
        "raw_to_canonical_id": dict(target_spec.raw_to_canonical_id),
        "raw_normalization_evidence": dict(target_spec.raw_normalization_evidence),
        "raw_record_hashes": dict(target_spec.raw_record_hashes),
        "target_mapping_hash": target_mapping_hash,
        "preprocessing_hash": preprocessing_hash,
        "scaler_hash": scaler.scaler_hash,
        "coexpression_fit_hash": coexpression_fit_hash,
        "control_selection_hash": control_selection_hash,
        "preprocessing_spec": preprocessing_spec,
        "gene_panel_hash": sha256_json([str(gene) for gene in gene_list]),
        "control_label": ctrl_label,
        "control_evidence": control_map[name].evidence,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_coexpression_graph(
    X_norm: np.ndarray,
    gene_list: np.ndarray,
    *,
    threshold: float,
) -> torch.Tensor:
    """Connect genes with |Pearson correlation| > threshold across cells."""
    num_genes = len(gene_list)
    logger.info("Computing co-expression graph (threshold=%.2f) ...", threshold)

    corr = np.corrcoef(X_norm.T)
    upper_row, upper_col = np.triu_indices(num_genes, k=1)
    upper_values = corr[upper_row, upper_col]
    retained = np.isfinite(upper_values) & (np.abs(upper_values) > threshold)
    source = upper_row[retained]
    target = upper_col[retained]
    loops = np.arange(num_genes)
    row = np.concatenate([source, target, loops])
    column = np.concatenate([target, source, loops])
    edge_index = torch.tensor(np.stack([row, column]), dtype=torch.long).to(DEVICE)
    edge_index = torch.unique(edge_index, dim=1)
    logger.info("Co-expression graph: %d edges (including self-loops)", edge_index.shape[1])
    return edge_index


def build_string_go_union_graph(
    ppi_ei: torch.Tensor | None,
    go_ei: torch.Tensor | None,
    num_genes: int,
) -> torch.Tensor | None:
    """Fixed union of STRING and GO edges; co-expression is a separate arm."""
    del num_genes
    if ppi_ei is None or go_ei is None:
        logger.error("STRING-GO union requires both component graphs")
        return None
    union = torch.unique(torch.cat([ppi_ei, go_ei], dim=1), dim=1)
    logger.info("STRING-GO union graph: %d directed edges", union.shape[1])
    return union


def ba_attachment_for_density(reference_edge_index: torch.Tensor, node_count: int) -> int:
    """Choose the BA attachment count closest to a reference undirected edge count."""
    edges = reference_edge_index.detach().cpu().numpy()
    undirected = {
        tuple(sorted((int(source), int(target)))) for source, target in edges.T if source != target
    }
    if not undirected:
        raise ValueError("BA density matching requires a non-empty reference graph")
    target_edges = len(undirected)
    candidates = range(1, node_count)
    return min(candidates, key=lambda m: (abs(m * (node_count - m) - target_edges), m))


def build_all_graphs(
    gene_list: np.ndarray,
    X_coexpression_fit: np.ndarray,
    num_genes: int,
    requested_graph_types: Sequence[str],
    frozen_sources: Mapping[str, FrozenGraphSource],
    graph_instance_seed: int | None,
    ensemble_specs: Mapping[str, GraphEnsembleSpec],
    preprocessing_spec: PreprocessingSpec,
) -> tuple[dict[str, torch.Tensor | None], dict[str, dict[str, object]]]:
    """Construct only requested graphs from frozen sources or permitted training data."""
    graphs: dict[str, torch.Tensor | None] = {}
    build_audits: dict[str, dict[str, object]] = {}
    requested = set(requested_graph_types)
    topology_types = {"degree_preserving_rewired", "barabasi_albert"}
    need_union = bool(requested & ({"string_go_union"} | topology_types))
    need_string = bool(requested & {"string_ppi"}) or need_union
    need_go = bool(requested & {"gene_ontology"}) or need_union

    if need_string:
        logger.info("--- Loading frozen STRING PPI edge list ---")
        edges = build_frozen_edge_graph(frozen_sources["string_ppi"], gene_list)
        graphs["string_ppi"] = torch.tensor(edges, dtype=torch.long, device=DEVICE)

    if need_go:
        logger.info("--- Loading frozen Gene Ontology projected edge list ---")
        edges = build_frozen_edge_graph(frozen_sources["gene_ontology"], gene_list)
        graphs["gene_ontology"] = torch.tensor(edges, dtype=torch.long, device=DEVICE)

    if "coexpression" in requested:
        logger.info("--- Building control-only Co-expression graph ---")
        graphs["coexpression"] = build_coexpression_graph(
            X_coexpression_fit,
            gene_list,
            threshold=preprocessing_spec.coexpression_threshold,
        )

    if need_union:
        logger.info("--- Building frozen STRING-GO union graph ---")
        graphs["string_go_union"] = build_string_go_union_graph(
            graphs["string_ppi"], graphs["gene_ontology"], num_genes
        )

    if requested & topology_types:
        if graph_instance_seed is None:
            raise ValueError("Topology ensemble graphs require graph_instance_seed")
        union = graphs["string_go_union"]
        if union is None:
            raise RuntimeError("Topology ensembles require the frozen STRING-GO union")
        if "degree_preserving_rewired" in requested:
            spec = ensemble_specs["degree_preserving_rewired"]
            rewired, rewiring_audit = build_degree_preserving_rewired_instance(
                union.detach().cpu().numpy(),
                num_genes,
                seed=graph_instance_seed,
                swaps_per_edge=int(spec.swaps_per_edge),
                max_attempts_per_rewirable_edge=int(spec.max_attempts_per_rewirable_edge),
                minimum_rewirable_edge_fraction=float(spec.minimum_rewirable_edge_fraction),
                maximum_rewirable_original_overlap=float(spec.maximum_rewirable_original_overlap),
                return_audit=True,
            )
            graphs["degree_preserving_rewired"] = torch.tensor(
                rewired, dtype=torch.long, device=DEVICE
            )
            build_audits["degree_preserving_rewired"] = rewiring_audit.as_dict()
        if "barabasi_albert" in requested:
            attachment = ba_attachment_for_density(union, num_genes)
            ba_graph = build_barabasi_albert_instance(
                num_genes,
                attachment,
                seed=graph_instance_seed,
            )
            graphs["barabasi_albert"] = torch.tensor(ba_graph, dtype=torch.long, device=DEVICE)

    if "self_loop_gat" in requested:
        graphs["self_loop_gat"] = torch.tensor(
            self_loop_edge_index(num_genes), dtype=torch.long, device=DEVICE
        )

    if "transformer" in requested:
        graphs["transformer"] = None

    return graphs, build_audits


def describe_graph(
    graph_type: str,
    edge_index: torch.Tensor | None,
    gene_panel_hash: str,
    num_genes: int,
    coexpression_fit_hash: str,
    frozen_sources: Mapping[str, FrozenGraphSource],
    graph_instance_seed: int | None,
    ensemble_specs: Mapping[str, GraphEnsembleSpec],
    all_graphs: Mapping[str, torch.Tensor | None],
    graph_build_audits: Mapping[str, Mapping[str, object]],
    preprocessing_spec: PreprocessingSpec,
    graph_ensemble_config_hash: str | None,
) -> dict:
    """Create exact graph provenance for every GAT condition."""
    if edge_index is None:
        return {
            "graph_type": graph_type,
            "graph_instance": "not_applicable",
            "edge_hash": sha256_json(
                {"graph_type": graph_type, "graph_instance": "not_applicable"}
            ),
            "gene_panel_hash": gene_panel_hash,
            "source": "separate architecture baseline",
        }
    sources = {
        "string_ppi": (
            frozen_sources["string_ppi"].source_url,
            f"{frozen_sources['string_ppi'].release};"
            f"sha256={frozen_sources['string_ppi'].edge_list_sha256}",
        ),
        "gene_ontology": (
            frozen_sources["gene_ontology"].source_url,
            f"{frozen_sources['gene_ontology'].release};"
            f"sha256={frozen_sources['gene_ontology'].edge_list_sha256}",
        ),
        "coexpression": ("control-only expression matrix", coexpression_fit_hash),
        "string_go_union": (
            "fixed union of frozen edge lists",
            sha256_json(
                {
                    graph_type: source.edge_list_sha256
                    for graph_type, source in sorted(frozen_sources.items())
                }
            ),
        ),
        "self_loop_gat": ("deterministic control", "self-loops only"),
        "degree_preserving_rewired": (
            "frozen STRING-GO union",
            "component_preserving_canonical_double_edge_swap_v2",
        ),
        "barabasi_albert": (
            "density-matched sensitivity control",
            "networkx barabasi_albert_graph pinned by environment lock",
        ),
    }
    source, source_version = sources[graph_type]
    edge_array = edge_index.detach().cpu().numpy()
    topology_graph = graph_type in {"degree_preserving_rewired", "barabasi_albert"}
    if topology_graph and graph_instance_seed is None:
        raise ValueError(f"Graph {graph_type!r} lacks graph-instance seed")
    if topology_graph and graph_ensemble_config_hash is None:
        raise ValueError(f"Graph {graph_type!r} lacks graph-ensemble config hash")
    graph_instance = (
        f"{graph_type}__seed_{graph_instance_seed}"
        if topology_graph
        else f"{graph_type}__instance_0"
    )
    parameters: dict[str, object]
    parent_edge_hash: str | None = None
    if graph_type == "coexpression":
        parameters = {
            "coexpression_method": preprocessing_spec.coexpression_method,
            "coexpression_threshold": preprocessing_spec.coexpression_threshold,
            "coexpression_threshold_rule": preprocessing_spec.coexpression_threshold_rule,
            "constant_nan_policy": preprocessing_spec.coexpression_constant_nan_policy,
            "symmetry_policy": preprocessing_spec.coexpression_symmetry_policy,
            "self_loop_policy": preprocessing_spec.coexpression_self_loop_policy,
            "coexpression_fit_hash": coexpression_fit_hash,
        }
    elif graph_type == "degree_preserving_rewired":
        union = all_graphs["string_go_union"]
        if union is None:
            raise RuntimeError("Rewired graph lacks its union parent")
        parent_edge_hash = edge_hash(union.detach().cpu().numpy(), num_genes)
        parameters = {
            "algorithm": "component_preserving_canonical_double_edge_swap_v2",
            "swaps_per_edge": ensemble_specs[graph_type].swaps_per_edge,
            "max_attempts_per_rewirable_edge": ensemble_specs[
                graph_type
            ].max_attempts_per_rewirable_edge,
            "connectivity_policy": ensemble_specs[graph_type].connectivity_policy,
            "minimum_rewirable_edge_fraction": ensemble_specs[
                graph_type
            ].minimum_rewirable_edge_fraction,
            "maximum_rewirable_original_overlap": ensemble_specs[
                graph_type
            ].maximum_rewirable_original_overlap,
            "self_loops_excluded_during_swaps": True,
            "duplicates_forbidden": True,
            "swap_audit": dict(graph_build_audits[graph_type]),
            "graph_ensemble_config_hash": graph_ensemble_config_hash,
        }
    elif graph_type == "barabasi_albert":
        union = all_graphs["string_go_union"]
        if union is None:
            raise RuntimeError("BA graph lacks its union density reference")
        parent_edge_hash = edge_hash(union.detach().cpu().numpy(), num_genes)
        union_array = union.detach().cpu().numpy()
        reference_nonself_edges = {
            tuple(sorted((int(source), int(target))))
            for source, target in union_array.T
            if source != target
        }
        ba_spec = ensemble_specs[graph_type]
        parameters = {
            "algorithm": ba_spec.algorithm,
            "networkx_version": nx.__version__,
            "implementation_version_policy": ba_spec.implementation_version_policy,
            "density_reference": ba_spec.density_reference,
            "reference_node_count": num_genes,
            "reference_nonself_edge_count": len(reference_nonself_edges),
            "m": ba_attachment_for_density(union, num_genes),
            "m_selection_policy": ba_spec.m_selection_policy,
            "node_mapping_policy": ba_spec.node_mapping_policy,
            "initial_graph_policy": ba_spec.initial_graph_policy,
            "self_loop_policy": ba_spec.self_loop_policy,
            "graph_ensemble_config_hash": graph_ensemble_config_hash,
        }
    else:
        parameters = {
            "string_source_sha256": (
                frozen_sources["string_ppi"].edge_list_sha256
                if "string_ppi" in frozen_sources
                else None
            ),
            "go_source_sha256": (
                frozen_sources["gene_ontology"].edge_list_sha256
                if "gene_ontology" in frozen_sources
                else None
            ),
            "coexpression_excluded": graph_type == "string_go_union",
        }
    provenance = graph_provenance(
        edge_array,
        graph_type=graph_type,
        graph_instance=graph_instance,
        gene_panel_hash=gene_panel_hash,
        node_count=num_genes,
        source=source,
        source_version=source_version,
        parameters=parameters,
        parent_edge_hash=parent_edge_hash,
        seed=graph_instance_seed if topology_graph else None,
    ).as_dict()
    if topology_graph:
        provenance["graph_ensemble_config_hash"] = graph_ensemble_config_hash
    parent = None
    if topology_graph:
        parent_graph = all_graphs["string_go_union"]
        if parent_graph is None:
            raise RuntimeError("Topology graph lacks its curated parent for diagnostics")
        parent = parent_graph.detach().cpu().numpy()
    provenance["diagnostics"] = graph_diagnostics(
        edge_array,
        num_genes,
        parent_edge_index=parent,
    )
    return provenance


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def make_mask(gene_list: np.ndarray, target_genes: tuple[str, ...]) -> torch.Tensor:
    """Create a complete perturbation mask from a verified explicit target mapping."""
    if not target_genes:
        raise ValueError("Every evaluated condition must map to at least one target gene")
    mask = torch.zeros(len(gene_list), dtype=torch.bool)
    missing: list[str] = []
    for gene in target_genes:
        indices = np.where(gene_list == gene)[0]
        if len(indices) != 1:
            missing.append(gene)
            continue
        mask[indices[0]] = True
    if missing:
        raise ValueError(f"Mapped target genes are missing or non-unique in the panel: {missing}")
    if int(mask.sum()) != len(set(target_genes)):
        raise RuntimeError("Perturbation mask does not preserve every mapped target")
    return mask


def gene_top20_absolute_delta_jaccard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Jaccard of exactly 20 genes ordered by (-abs(delta), frozen gene index)."""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if truth.ndim != 1 or prediction.shape != truth.shape or len(truth) < 20:
        raise ValueError("Gene top-20 Jaccard requires aligned vectors with at least 20 genes")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Gene top-20 Jaccard requires finite delta vectors")
    gene_index = np.arange(len(truth))
    top_true = set(np.lexsort((gene_index, -np.abs(truth)))[:20].tolist())
    top_pred = set(np.lexsort((gene_index, -np.abs(prediction)))[:20].tolist())
    if len(top_true) != 20 or len(top_pred) != 20:
        raise RuntimeError("Gene top-20 ranking did not produce exactly 20 unique genes")
    union = top_true | top_pred
    return len(top_true & top_pred) / len(union)


def compute_fold_metrics(
    delta_true: np.ndarray,
    delta_pred: np.ndarray,
    condition: str,
) -> tuple[dict[str, float | str | None], dict[str, str]]:
    """Compute fold metrics while making undefined correlations explicit."""
    pr: float | None = None
    sr: float | None = None
    truth_valid = bool(np.std(delta_true) > 1e-8)
    prediction_valid = bool(np.std(delta_pred) > 1e-8)
    if truth_valid and prediction_valid:
        pr, _ = pearsonr(delta_true, delta_pred)
        sr, _ = spearmanr(delta_true, delta_pred)
        correlation_state = "valid"
    elif not truth_valid and not prediction_valid:
        correlation_state = "constant_truth_and_prediction"
    elif not truth_valid:
        correlation_state = "constant_truth"
    else:
        correlation_state = "constant_prediction"
    mse = float(np.mean((delta_true - delta_pred) ** 2))
    if (
        delta_true.ndim == 1
        and delta_pred.shape == delta_true.shape
        and len(delta_true) >= 20
        and np.isfinite(delta_true).all()
        and np.isfinite(delta_pred).all()
    ):
        jac = gene_top20_absolute_delta_jaccard(delta_true, delta_pred)
        jaccard_state = "valid"
    else:
        jac = None
        jaccard_state = "fewer_than_20_or_malformed_gene_vector"
    metrics: dict[str, float | str | None] = {
        "condition": condition,
        "pearson_r": pr,
        "spearman_rho": sr,
        "mse": mse,
        "gene_top20_absolute_delta_jaccard": jac,
    }
    states = {
        "pearson_r": correlation_state,
        "spearman_rho": correlation_state,
        "mse": "valid",
        "gene_top20_absolute_delta_jaccard": jaccard_state,
    }
    return metrics, states


# ---------------------------------------------------------------------------
# LOPO evaluation
# ---------------------------------------------------------------------------


def model_parameter_hash(model: nn.Module) -> str:
    """Hash the complete initial non-adjacency state, including normalization buffers."""
    digest = hashlib.sha256()
    state = model.state_dict()
    included = [name for name in state if name != "edge_index" and not name.endswith(".edge_index")]
    if not included:
        raise ValueError("Model has no non-adjacency state to hash")
    for name in sorted(included):
        tensor = state[name].detach().cpu().contiguous()
        header = canonical_json(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = tensor.numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def persist_initial_feature_artifact(features: torch.Tensor) -> tuple[str, str, str]:
    """Persist one deterministic 68-column FP32 tensor under its logical content hash."""
    array = features.detach().cpu().to(dtype=torch.float32).contiguous().numpy()
    if array.ndim != 2 or array.shape[1] != 68 or not np.isfinite(array).all():
        raise ValueError("TDS-41 feature artifact must be a finite [genes, 68] FP32 tensor")
    logical_hash = sha256_json(
        {"dtype": "float32", "shape": list(array.shape), "values": array.tolist()}
    )
    artifact_dir = _results_dir / "feature_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    destination = artifact_dir / f"{logical_hash}.npy"
    if not destination.exists():
        partial = artifact_dir / f".{logical_hash}.{os.getpid()}.{time.time_ns()}.part"
        try:
            with partial.open("xb") as handle:
                np.save(handle, array, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, destination)
        finally:
            if partial.exists():
                partial.unlink()
    observed = np.load(destination, allow_pickle=False)
    if (
        observed.dtype != np.float32
        or observed.shape != array.shape
        or not np.array_equal(observed, array)
    ):
        raise RuntimeError("Existing TDS-41 feature artifact fails content validation")
    return (
        logical_hash,
        destination.relative_to(_results_dir).as_posix(),
        sha256_file(destination),
    )


def build_input_contract_policy_hash(model_spec: ModelSpec, code_commit: str) -> str:
    """Hash the global TDS-41 feature policy and exact code/configuration lineage."""
    return sha256_json(
        {
            "policy": "tds41_input_contract_policy_v1",
            "model_contract_canonical_hash": model_spec.canonical_hash,
            "code_commit": code_commit,
            "feature_schema": list(model_spec.feature_encoder["features"]),
            "feature_dimension": int(model_spec.feature_encoder["input_dim"]),
            "transformations": {
                "control_mean": "verified_log_normalized_precentering_mean",
                "control_variance": "log1p_population_variance_ddof0",
                "target_mask": "explicit_mapped_target_indicator",
                "context": "mean_initialized_target_embeddings_plus_log1p_target_count",
            },
        }
    )


def build_input_definition_hash(
    *,
    policy_hash: str,
    dataset: str,
    control_feature_hash: str,
    matrix_contract_hash: str,
    preprocessing_hash: str,
    gene_order: Sequence[str],
) -> str:
    """Hash dataset-scale static inputs separately from seed/unit-dependent features."""
    genes = [str(gene) for gene in gene_order]
    return sha256_json(
        {
            "input_contract_policy_hash": policy_hash,
            "dataset": dataset,
            "control_feature_hash": control_feature_hash,
            "matrix_contract_hash": matrix_contract_hash,
            "preprocessing_hash": preprocessing_hash,
            "gene_order": genes,
            "gene_order_hash": sha256_json(genes),
        }
    )


def fold_resource_usage(
    *,
    training_seconds: float,
    inference_seconds: float,
    total_seconds: float,
) -> dict[str, object]:
    """Persist per-fold timing, hardware, scheduler identity, and accelerator memory."""
    if DEVICE.type == "cuda":
        properties = torch.cuda.get_device_properties(DEVICE)
        peak_memory: int | None = int(torch.cuda.max_memory_allocated(DEVICE))
        peak_memory_state = "measured_torch_cuda_max_memory_allocated"
        device_name = str(properties.name)
        device_uuid = str(getattr(properties, "uuid", "unavailable"))
    else:
        peak_memory = None
        peak_memory_state = "not_applicable_cpu"
        device_name = "cpu"
        device_uuid = "not_applicable"
    return {
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "total_seconds": float(total_seconds),
        "peak_accelerator_memory_bytes": peak_memory,
        "peak_accelerator_memory_state": peak_memory_state,
        "device_type": DEVICE.type,
        "device_name": device_name,
        "device_uuid": device_uuid,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"),
        "terminal_status": "succeeded",
    }


def run_lopo(
    model_cls,
    model_kwargs: dict,
    control_feature_mean_tensor: torch.Tensor,
    control_feature_log_variance_tensor: torch.Tensor,
    ctrl_np: np.ndarray,
    gene_list: np.ndarray,
    cond_profiles: dict[str, np.ndarray],
    condition_targets: Mapping[str, tuple[str, ...]],
    valid_conds: list[str],
    is_gnn: bool,
    model_spec: ModelSpec,
    seed: int,
    max_folds: int,
    condition_alias_members: Mapping[str, tuple[str, ...]],
    condition_cell_ids: Mapping[str, tuple[str, ...]],
    fold_start: int = 0,
    fold_end: int | None = None,
    run_identity_base: Mapping[str, object] | None = None,
    split_hashes: Mapping[str, str] | None = None,
    fold_metadata: Mapping[str, object] | None = None,
) -> list[dict]:
    """Run LOPO cross-validation for one (model, graph) configuration.

    fold_start/fold_end: run only folds[fold_start:fold_end] for parallel splitting.
    """
    del is_gnn
    training_spec = model_spec.training
    criterion = nn.MSELoss(reduction="mean")
    n_folds = min(max_folds, len(valid_conds))
    if fold_end is None:
        fold_end = n_folds
    fold_end = min(fold_end, n_folds)
    fold_start = min(fold_start, fold_end)
    results: list[dict] = []
    identity_base = dict(run_identity_base or {})
    condition_split_hashes = dict(split_hashes or {})
    shared_metadata = dict(fold_metadata or {})
    alias_members = {
        condition: tuple(members) for condition, members in condition_alias_members.items()
    }
    cells_by_condition = {
        condition: tuple(cell_ids) for condition, cell_ids in condition_cell_ids.items()
    }
    if set(alias_members) != set(valid_conds) or set(cells_by_condition) != set(valid_conds):
        raise ValueError("Canonical LOPO alias/cell maps must exactly match the panel universe")
    for condition in valid_conds:
        members = alias_members[condition]
        cells = cells_by_condition[condition]
        if (
            not members
            or len(set(members)) != len(members)
            or list(members) != sorted(members, key=lambda value: value.encode("utf-8"))
            or not cells
            or len(set(cells)) != len(cells)
            or list(cells) != sorted(cells, key=lambda value: value.encode("utf-8"))
        ):
            raise ValueError(f"Canonical LOPO group {condition!r} has invalid alias/cell lineage")

    for fi, held_out in enumerate(valid_conds[fold_start:fold_end], start=fold_start):
        fold_wall_start = time.perf_counter()
        train_conds = [c for c in valid_conds if c != held_out]
        held_out_aliases = alias_members.get(held_out, (held_out,))
        training_aliases = [
            alias
            for condition in train_conds
            for alias in alias_members.get(condition, (condition,))
        ]
        if set(held_out_aliases) & set(training_aliases):
            raise ValueError("Held-out canonical alias group overlaps the training universe")
        held_out_cells = cells_by_condition.get(held_out, ())
        training_cells = [
            cell_id
            for condition in train_conds
            for cell_id in cells_by_condition.get(condition, ())
        ]
        if set(held_out_cells) & set(training_cells):
            raise ValueError("Held-out perturbation cells overlap the training cell universe")
        current_split_hash = condition_split_hashes[held_out]
        current_fold_seed = derive_fold_seed(seed, held_out, current_split_hash)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
        ho_target = cond_profiles[held_out]
        ho_mask = make_mask(gene_list, condition_targets[held_out])

        model = model_cls(**model_kwargs).to(DEVICE)
        initialization_hash = model_parameter_hash(model)
        with torch.no_grad():
            initial_raw_features = model.feature_encoder.raw_features(
                control_feature_mean_tensor,
                control_feature_log_variance_tensor,
                ho_mask.to(DEVICE),
            )
        gene_embedding_dim = int(model_spec.feature_encoder["gene_embedding_dim"])
        context_start = gene_embedding_dim + 3
        initial_context = initial_raw_features[0, context_start:]
        target_mask_values = ho_mask.to(dtype=torch.bool).cpu().tolist()
        target_mask_hash = sha256_json(target_mask_values)
        initial_context_hash = sha256_json(initial_context.detach().cpu().tolist())
        raw_feature_logical_hash, raw_feature_artifact, raw_feature_artifact_sha256 = (
            persist_initial_feature_artifact(initial_raw_features)
        )
        current_input_contract_policy_hash = build_input_contract_policy_hash(
            model_spec, str(shared_metadata.get("code_commit"))
        )
        current_input_definition_hash = build_input_definition_hash(
            policy_hash=current_input_contract_policy_hash,
            dataset=str(shared_metadata.get("dataset")),
            control_feature_hash=str(shared_metadata.get("control_feature_hash")),
            matrix_contract_hash=str(shared_metadata.get("matrix_contract_hash")),
            preprocessing_hash=str(shared_metadata.get("preprocessing_hash")),
            gene_order=[str(gene) for gene in gene_list],
        )
        if shared_metadata.get("input_contract_policy_hash") != current_input_contract_policy_hash:
            raise ValueError("Preplanned input-contract policy hash disagrees with execution")
        if shared_metadata.get("input_definition_hash") != current_input_definition_hash:
            raise ValueError("Preplanned input-definition hash disagrees with execution")
        input_instance_hash = sha256_json(
            {
                "input_definition_hash": current_input_definition_hash,
                "training_seed": seed,
                "initialization_hash": initialization_hash,
                "held_out_condition": held_out,
                "target_mapping_hash": shared_metadata.get("target_mapping_hash"),
                "mapped_targets": list(condition_targets[held_out]),
                "target_mask_hash": target_mask_hash,
                "initial_context_hash": initial_context_hash,
                "raw_feature_logical_hash": raw_feature_logical_hash,
            }
        )
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_spec["learning_rate"]),
            weight_decay=float(training_spec["weight_decay"]),
            betas=tuple(float(value) for value in training_spec["betas"]),
            eps=float(training_spec["epsilon"]),
            amsgrad=bool(training_spec["adamw_amsgrad"]),
            maximize=bool(training_spec["adamw_maximize"]),
            foreach=bool(training_spec["adamw_foreach"]),
            capturable=bool(training_spec["adamw_capturable"]),
            differentiable=bool(training_spec["adamw_differentiable"]),
            fused=bool(training_spec["adamw_fused"]),
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=int(training_spec["scheduler_t_max"]),
            eta_min=float(training_spec["scheduler_eta_min"]),
        )
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # --- Training ---
        training_loss_by_epoch: list[float] = []
        training_start = time.perf_counter()
        for ep in range(int(training_spec["epochs"])):
            model.train()
            epoch_losses: list[float] = []
            for tc in training_condition_order(train_conds, current_fold_seed, ep):
                tm = make_mask(gene_list, condition_targets[tc])
                target_delta = cond_profiles[tc] - ctrl_np
                tgt = torch.tensor(target_delta, dtype=torch.float32, device=DEVICE)
                opt.zero_grad()
                predicted_delta = model(
                    control_feature_mean_tensor,
                    control_feature_log_variance_tensor,
                    perturbation_mask=tm.to(DEVICE),
                )
                loss = criterion(predicted_delta, tgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(training_spec["gradient_l2_clip"]),
                    norm_type=float(training_spec["gradient_clip_norm_type"]),
                    error_if_nonfinite=bool(training_spec["gradient_clip_error_if_nonfinite"]),
                    foreach=bool(training_spec["gradient_clip_foreach"]),
                )
                opt.step()
                epoch_losses.append(float(loss.detach().cpu()))
            sched.step()
            training_loss_by_epoch.append(float(np.mean(epoch_losses)))
        training_seconds = time.perf_counter() - training_start

        # --- Inference ---
        model.eval()
        inference_start = time.perf_counter()
        with torch.no_grad():
            delta_pred = (
                model(
                    control_feature_mean_tensor,
                    control_feature_log_variance_tensor,
                    perturbation_mask=ho_mask.to(DEVICE),
                )
                .cpu()
                .numpy()
            )
        inference_seconds = time.perf_counter() - inference_start
        pred = ctrl_np + delta_pred

        delta_true = ho_target - ctrl_np
        metrics, metric_states = compute_fold_metrics(delta_true, delta_pred, held_out)
        training_mean = np.mean([cond_profiles[condition] for condition in train_conds], axis=0)
        run_identity = {
            **identity_base,
            "split_hash": current_split_hash,
            "seed": seed,
            "fold_seed": current_fold_seed,
            "condition": held_out,
        }
        fold_result = build_fold_result(
            run_identity=run_identity,
            condition=held_out,
            fold_index=fi,
            gene_order=[str(gene) for gene in gene_list],
            y_true=ho_target,
            y_pred=pred,
            training_mean=training_mean,
            control_profile=ctrl_np,
            training_loss_by_epoch=training_loss_by_epoch,
            metrics={key: value for key, value in metrics.items() if key != "condition"},
            metric_states=metric_states,
            metadata={
                **shared_metadata,
                "split_hash": current_split_hash,
                "seed": seed,
                "fold_seed": current_fold_seed,
                "fold_order_seed": current_fold_seed,
                "fold_order_seed_policy": FOLD_ORDER_SEED_POLICY,
                "fold_order_seed_policy_hash": FOLD_ORDER_SEED_POLICY_HASH,
                "condition_order_policy": CONDITION_ORDER_POLICY,
                "condition_order_policy_hash": CONDITION_ORDER_POLICY_HASH,
                "model_stochastic_seed": seed,
                "model_stochastic_seed_policy": "top_level_training_seed_reset_per_fold",
                "held_out_condition": held_out,
                "held_out_alias_members": list(held_out_aliases),
                "held_out_alias_member_hash": sha256_json(list(held_out_aliases)),
                "training_alias_members": training_aliases,
                "training_alias_member_hash": sha256_json(training_aliases),
                "held_out_cell_count": len(held_out_cells),
                "held_out_cell_ids": list(held_out_cells),
                "held_out_cell_set_hash": sha256_json(list(held_out_cells)),
                "training_cell_count": len(training_cells),
                "training_cell_set_hash": sha256_json(training_cells),
                "train_test_cell_overlap_count": 0,
                "training_condition_count": len(train_conds),
                "training_conditions": train_conds,
                "training_condition_list_hash": sha256_json(train_conds),
                "training_condition_universe_hash": sha256_json(train_conds),
                "initialization_hash": initialization_hash,
                "target_mask": target_mask_values,
                "target_mask_hash": target_mask_hash,
                "initial_perturbation_context_hash": initial_context_hash,
                "baseline_feature_hash": shared_metadata.get("control_feature_hash"),
                "input_contract_policy_hash": current_input_contract_policy_hash,
                "input_definition_hash": current_input_definition_hash,
                "input_instance_hash": input_instance_hash,
                "initial_raw_feature_hash": raw_feature_logical_hash,
                "initial_raw_feature_artifact": raw_feature_artifact,
                "initial_raw_feature_artifact_sha256": raw_feature_artifact_sha256,
                "input_contract_hash": input_instance_hash,
                "epochs": int(training_spec["epochs"]),
                "learning_rate": float(training_spec["learning_rate"]),
                "weight_decay": float(training_spec["weight_decay"]),
                "vector_space": "scaled_expression",
                "metric_vector_space": "delta_from_control_profile",
                "gene_top20_absolute_delta_jaccard_policy": (
                    "exactly_20_unique_genes_sorted_by_negative_absolute_delta_then_"
                    "frozen_gene_index"
                ),
                "gene_top20_absolute_delta_jaccard_policy_hash": sha256_json(
                    {
                        "metric": "gene_top20_absolute_delta_jaccard",
                        "gene_count": 20,
                        "ranking": ["negative_absolute_delta", "frozen_gene_index"],
                        "undersized_policy": "invalid_no_k_reduction",
                    }
                ),
                "delta_true_sd_ddof0": float(np.std(delta_true, ddof=0)),
                "delta_pred_sd_ddof0": float(np.std(delta_pred, ddof=0)),
                "delta_true_l2_norm": float(np.linalg.norm(delta_true)),
                "delta_pred_l2_norm": float(np.linalg.norm(delta_pred)),
                "resource_usage": fold_resource_usage(
                    training_seconds=training_seconds,
                    inference_seconds=inference_seconds,
                    total_seconds=time.perf_counter() - fold_wall_start,
                ),
            },
        )
        results.append(fold_result.as_dict())

        if (fi + 1) % 5 == 0:
            recent = results[-5:]
            valid_recent = [
                float(result["metrics"]["pearson_r"])
                for result in recent
                if result["metric_states"]["pearson_r"] == "valid"
            ]
            avg_p = float(np.mean(valid_recent)) if valid_recent else None
            logger.info(
                "  Fold %d/%d done (recent-5 Pearson=%s; valid n=%d)",
                fi + 1,
                n_folds,
                f"{avg_p:.4f}" if avg_p is not None else "invalid",
                len(valid_recent),
            )

    return results


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------


def fold_slice_conditions(
    conditions: list[str],
    max_folds: int,
    fold_start: int,
    fold_end: int | None,
) -> list[str]:
    """Return the exact deterministic condition slice encoded by a task."""
    n_folds = min(max_folds, len(conditions))
    end = n_folds if fold_end is None else min(fold_end, n_folds)
    start = min(fold_start, end)
    selected = conditions[start:end]
    if not selected:
        raise ValueError(f"Fold slice [{start}, {end}) selects no conditions")
    return selected


def split_hash(
    condition_panel_hash: str,
    all_conditions: list[str],
    held_out_condition: str,
    condition_alias_members: Mapping[str, Sequence[str]],
    condition_cell_ids: Mapping[str, Sequence[str]],
) -> str:
    """Hash held-out and training condition identities for one LOPO execution unit."""
    if held_out_condition not in all_conditions:
        raise ValueError(f"Held-out condition {held_out_condition!r} is outside the panel")
    training_conditions = [
        condition for condition in all_conditions if condition != held_out_condition
    ]
    aliases = {
        condition: tuple(str(value) for value in members)
        for condition, members in condition_alias_members.items()
    }
    if set(aliases) != set(all_conditions):
        raise ValueError("Split alias mapping must exactly match the canonical panel")
    for condition, members in aliases.items():
        if (
            not members
            or len(set(members)) != len(members)
            or list(members) != sorted(members, key=lambda value: value.encode("utf-8"))
        ):
            raise ValueError(
                f"Split alias group {condition!r} must be nonempty, unique, and sorted"
            )
    held_out_aliases = aliases[held_out_condition]
    training_alias_groups = {
        condition: list(aliases[condition]) for condition in training_conditions
    }
    training_aliases = [
        alias for condition in training_conditions for alias in training_alias_groups[condition]
    ]
    if len(set(held_out_aliases)) != len(held_out_aliases):
        raise ValueError("Held-out canonical group contains duplicate aliases")
    if set(held_out_aliases) & set(training_aliases):
        raise ValueError("Held-out aliases overlap training canonical groups")
    cells = {
        condition: tuple(str(value) for value in cell_ids)
        for condition, cell_ids in condition_cell_ids.items()
    }
    if set(cells) != set(all_conditions):
        raise ValueError("Split cell mapping must exactly match the canonical panel")
    for condition, cell_ids in cells.items():
        if (
            not cell_ids
            or len(set(cell_ids)) != len(cell_ids)
            or list(cell_ids) != sorted(cell_ids, key=lambda value: value.encode("utf-8"))
        ):
            raise ValueError(f"Split cell group {condition!r} must be nonempty, unique, and sorted")
    held_out_cells = cells[held_out_condition]
    training_cells = [cell_id for condition in training_conditions for cell_id in cells[condition]]
    if len(set(held_out_cells)) != len(held_out_cells) or len(set(training_cells)) != len(
        training_cells
    ):
        raise ValueError("Canonical split cell identifiers must be unique")
    overlap = set(held_out_cells) & set(training_cells)
    if overlap:
        raise ValueError(f"Held-out cells overlap training cells: {sorted(overlap)[:10]}")
    return sha256_json(
        {
            "condition_panel_hash": condition_panel_hash,
            "held_out_condition": held_out_condition,
            "training_conditions": training_conditions,
            "training_condition_list_hash": sha256_json(training_conditions),
            "training_condition_count": len(training_conditions),
            "held_out_alias_members": list(held_out_aliases),
            "held_out_alias_member_hash": sha256_json(list(held_out_aliases)),
            "training_alias_groups": training_alias_groups,
            "training_alias_member_hash": sha256_json(training_aliases),
            "held_out_cell_count": len(held_out_cells),
            "held_out_cell_ids": list(held_out_cells),
            "held_out_cell_set_hash": sha256_json(list(held_out_cells)),
            "training_cell_count": len(training_cells),
            "training_cell_set_hash": sha256_json(training_cells),
            "train_test_cell_overlap_count": 0,
        }
    )


def derive_fold_seed(seed: int, held_out_condition: str, condition_split_hash: str) -> int:
    """Derive the chunk-invariant fold-order seed without changing model initialization."""
    condition = canonical_condition_label(held_out_condition)
    if not condition:
        raise ValueError("held_out_condition must be canonically non-empty")
    digest = sha256_json(
        {
            "policy": FOLD_ORDER_SEED_POLICY,
            "training_seed": seed,
            "held_out_condition": condition,
            "split_hash": condition_split_hash,
        }
    )
    return int(digest[:8], 16)


def training_condition_order(
    training_conditions: list[str],
    fold_order_seed: int,
    epoch: int,
) -> list[str]:
    """Return an epoch order independent of earlier folds or chunk boundaries."""
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    canonical = [canonical_condition_label(condition) for condition in training_conditions]
    if any(not condition for condition in canonical) or len(set(canonical)) != len(canonical):
        raise ValueError("Training conditions must be canonically non-empty and unique")
    return sorted(
        canonical,
        key=lambda condition: sha256_json(
            {
                "policy": CONDITION_ORDER_POLICY,
                "fold_order_seed": fold_order_seed,
                "epoch": epoch,
                "condition": condition,
            }
        ),
    )


def verify_code_commit(expected_commit: str, repository: Path) -> None:
    """Bind provenance to the exact clean Git checkout that executes the benchmark."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Cannot verify the benchmark Git checkout") from error
    if head != expected_commit:
        raise RuntimeError(
            f"--code-commit {expected_commit!r} does not match checked-out HEAD {head!r}"
        )
    if dirty:
        raise RuntimeError(
            "Canonical runs require a clean Git worktree; uncommitted files detected"
        )


def _fold_metric(fold: Mapping[str, object], name: str) -> float | None:
    """Read a valid revision metric without treating invalid placeholders as data."""
    nested = fold.get("metrics")
    states = fold.get("metric_states")
    if not isinstance(nested, Mapping) or name not in nested:
        raise ValueError(f"Fold lacks revision metric {name!r}")
    if not isinstance(states, Mapping) or not isinstance(states.get(name), str):
        raise ValueError(f"Fold lacks explicit state for metric {name!r}")
    if states[name] != "valid":
        return None
    return float(nested[name])


def _valid_metric_values(folds: list[dict], name: str) -> list[float]:
    """Return only metrics whose explicit state is valid."""
    return [value for fold in folds if (value := _fold_metric(fold, name)) is not None]


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _std_or_none(values: list[float]) -> float | None:
    return float(np.std(values)) if values else None


def save_combination_result(
    dataset: str,
    graph_type: str,
    graph_instance: str,
    output_label: str,
    seed_results: dict[int, list[dict]],
    results_dir: Path,
    fold_tag: str = "",
) -> Path:
    """Save fold-level results for one (dataset, graph_type) combination."""
    results_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{dataset}__{output_label}{fold_tag}.json"
    out_path = results_dir / fname

    # Aggregate across seeds
    all_pearson, all_spearman, all_gene_jaccard, all_mse = [], [], [], []
    seed_summaries = {}
    for seed, folds in seed_results.items():
        ps = _valid_metric_values(folds, "pearson_r")
        ss = _valid_metric_values(folds, "spearman_rho")
        js = _valid_metric_values(folds, "gene_top20_absolute_delta_jaccard")
        ms = _valid_metric_values(folds, "mse")
        all_pearson.extend(ps)
        all_spearman.extend(ss)
        all_gene_jaccard.extend(js)
        all_mse.extend(ms)
        seed_summaries[f"seed_{seed}"] = {
            "pearson_mean": _mean_or_none(ps),
            "pearson_std": _std_or_none(ps),
            "pearson_valid_n": len(ps),
            "spearman_mean": _mean_or_none(ss),
            "spearman_valid_n": len(ss),
            "gene_top20_absolute_delta_jaccard_mean": _mean_or_none(js),
            "gene_top20_absolute_delta_jaccard_valid_n": len(js),
            "mse_mean": _mean_or_none(ms),
            "mse_valid_n": len(ms),
            "n_folds": len(folds),
            "folds": folds,
        }

    payload = {
        "dataset": dataset,
        "graph_type": graph_type,
        "graph_instance": graph_instance,
        "overall": {
            "pearson_mean": _mean_or_none(all_pearson),
            "pearson_std": _std_or_none(all_pearson),
            "pearson_valid_n": len(all_pearson),
            "spearman_mean": _mean_or_none(all_spearman),
            "spearman_std": _std_or_none(all_spearman),
            "spearman_valid_n": len(all_spearman),
            "gene_top20_absolute_delta_jaccard_mean": _mean_or_none(all_gene_jaccard),
            "gene_top20_absolute_delta_jaccard_std": _std_or_none(all_gene_jaccard),
            "gene_top20_absolute_delta_jaccard_valid_n": len(all_gene_jaccard),
            "mse_mean": _mean_or_none(all_mse),
            "mse_std": _std_or_none(all_mse),
            "mse_valid_n": len(all_mse),
        },
        "seeds": seed_summaries,
    }

    write_json_atomic(out_path, payload)
    logger.info("Saved results: %s", out_path)

    # Update global tracker for SIGTERM handler
    key = f"{dataset}__{graph_instance}"
    _partial_results[key] = payload["overall"]
    return out_path


def build_summary_csv(results_dir: Path) -> pd.DataFrame:
    """Aggregate all per-combination JSONs into a summary CSV."""

    def rounded(value: object, digits: int) -> float | None:
        return round(float(value), digits) if value is not None else None

    rows = []
    for jf in sorted(results_dir.glob("*.json")):
        if jf.name.startswith("results_") or jf.name.endswith(".manifest.json"):
            continue
        with open(str(jf)) as fh:
            data = json.load(fh)
        ov = data["overall"]
        rows.append(
            {
                "dataset": data["dataset"],
                "graph_type": data["graph_type"],
                "pearson_mean": rounded(ov["pearson_mean"], 4),
                "pearson_std": rounded(ov["pearson_std"], 4),
                "spearman_mean": rounded(ov["spearman_mean"], 4),
                "spearman_std": rounded(ov["spearman_std"], 4),
                "gene_top20_absolute_delta_jaccard_mean": rounded(
                    ov["gene_top20_absolute_delta_jaccard_mean"], 4
                ),
                "gene_top20_absolute_delta_jaccard_std": rounded(
                    ov["gene_top20_absolute_delta_jaccard_std"], 4
                ),
                "mse_mean": rounded(ov["mse_mean"], 6),
                "mse_std": rounded(ov["mse_std"], 6),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = results_dir / "summary.csv"
    df.to_csv(str(csv_path), index=False)
    logger.info("Summary table saved to %s", csv_path)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def validate_schedule_binding(
    *,
    manifest_path: Path,
    configuration_hash: str,
    code_commit: str,
    expected_configuration: Mapping[str, object],
    frozen_input_paths: Mapping[str, Path],
) -> tuple[str, str]:
    """Bind one runner invocation to one checksummed generated schedule record."""
    if re.fullmatch(r"[0-9a-f]{64}", configuration_hash) is None:
        raise ValueError("--schedule-config-hash must be a lowercase SHA-256 digest")
    with manifest_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("Schedule manifest must be a JSON object")
    declared_content_hash = raw.get("manifest_content_hash")
    content = dict(raw)
    content.pop("manifest_content_hash", None)
    if declared_content_hash != sha256_json(content):
        raise ValueError("Schedule manifest content hash mismatch")
    if raw.get("schema_version") != "1.0.0" or raw.get("code_commit") != code_commit:
        raise ValueError("Schedule schema or code-commit binding mismatch")
    configurations = raw.get("configurations")
    if not isinstance(configurations, list):
        raise ValueError("Schedule manifest lacks configurations")
    matches = [
        value
        for value in configurations
        if isinstance(value, Mapping) and value.get("configuration_hash") == configuration_hash
    ]
    if len(matches) != 1:
        raise ValueError("Schedule configuration hash must resolve to exactly one record")
    record = dict(matches[0])
    declared_configuration_hash = record.pop("configuration_hash", None)
    if declared_configuration_hash != sha256_json(record):
        raise ValueError("Schedule configuration record hash mismatch")
    if record != dict(expected_configuration):
        raise ValueError(
            "Runner arguments differ from the frozen schedule configuration: "
            f"expected {record}, observed {dict(expected_configuration)}"
        )
    frozen_inputs = raw.get("frozen_inputs")
    if not isinstance(frozen_inputs, Mapping) or set(frozen_inputs) != set(frozen_input_paths):
        raise ValueError("Schedule frozen-input set differs from the canonical runner contract")
    for name, path in frozen_input_paths.items():
        entry = frozen_inputs[name]
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"sha256", "size_bytes"}
            or entry.get("sha256") != sha256_file(path)
            or entry.get("size_bytes") != path.stat().st_size
        ):
            raise ValueError(f"Schedule frozen input {name!r} has drifted")
    return sha256_file(manifest_path), str(declared_content_hash)


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboGNN Graph Topology Benchmark")
    parser.add_argument("--stage", choices=("e0", "full", "smoke"), required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        choices=list(CANONICAL_DATASETS),
        help="Explicit dataset task membership",
    )
    parser.add_argument(
        "--graph-types",
        nargs="+",
        required=True,
        choices=GRAPH_TYPES,
        help="Explicit graph-arm task membership",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Fresh output directory outside the Git repository",
    )
    parser.add_argument(
        "--dataset-passport",
        type=Path,
        required=True,
        help="Frozen executable dataset registry with byte and AnnData schema passports",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root resolved against each passport relative_path",
    )
    parser.add_argument(
        "--control-map",
        type=Path,
        required=True,
        help="Verified JSON control mapping; no implicit control inference is allowed",
    )
    parser.add_argument(
        "--target-map",
        type=Path,
        required=True,
        help="Verified per-condition target mapping; condition-string inference is forbidden",
    )
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        required=True,
        help="Split-safe preprocessing scopes and verified input-expression state",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Exact checksummed TDS-42 model/optimizer contract",
    )
    parser.add_argument(
        "--code-commit",
        required=True,
        help="Exact lowercase 40-character Git commit SHA for this executable snapshot",
    )
    parser.add_argument(
        "--environment-lock",
        type=Path,
        required=True,
        help="Exact conda/pip lock artifact whose SHA-256 is bound into every run key",
    )
    parser.add_argument(
        "--schedule-manifest",
        type=Path,
        required=True,
        help="Generated revision matrix manifest that freezes this configuration",
    )
    parser.add_argument(
        "--schedule-config-hash",
        required=True,
        help="Exact configuration_hash from one schedule task row",
    )
    parser.add_argument(
        "--graph-source-config",
        type=Path,
        required=True,
        help=(
            "Frozen STRING/GO projected edge lists, releases, policies, and SHA-256 values; "
            "required for every curated graph arm"
        ),
    )
    parser.add_argument(
        "--graph-ensemble-config",
        type=Path,
        required=True,
        help="Frozen 5+5 topology-instance seeds and rewiring policy",
    )
    parser.add_argument(
        "--graph-instance-seed",
        type=int,
        help="One frozen graph-instance seed; required for BA or degree-preserving rewiring",
    )
    parser.add_argument(
        "--panel-size",
        type=int,
        choices=(10, 20, 50),
        required=True,
        help="Frozen deterministic panel size: 10 (E0), 20 (smoke), or 50 (full)",
    )
    parser.add_argument(
        "--num-hvg",
        type=int,
        required=True,
        choices=(200, 500, 1000),
        help="Explicit frozen gene-panel scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        choices=(42, 43, 44),
        help="One explicit top-level training seed",
    )
    parser.add_argument(
        "--fold-start",
        type=int,
        required=True,
        help="Start fold index (for parallel fold splitting)",
    )
    parser.add_argument(
        "--fold-end",
        type=int,
        required=True,
        help="End fold index exclusive (for parallel fold splitting)",
    )
    args = parser.parse_args()
    expected_panel_size = {"e0": 10, "full": 50, "smoke": 20}[args.stage]
    if args.panel_size != expected_panel_size:
        parser.error(f"--stage {args.stage!r} requires --panel-size {expected_panel_size}")
    if args.stage == "e0" and args.seed != 42:
        parser.error("--stage e0 requires the frozen training seed 42")
    if not 0 <= args.fold_start < args.fold_end <= args.panel_size:
        parser.error("Fold bounds must satisfy 0 <= start < end <= panel-size")
    if args.stage == "e0" and (args.fold_start, args.fold_end) != (0, 10):
        parser.error("--stage e0 requires the complete fold interval [0, 10)")
    frozen_input_paths = {
        "dataset-passport": args.dataset_passport,
        "control-map": args.control_map,
        "target-map": args.target_map,
        "preprocessing-config": args.preprocessing_config,
        "model-config": args.model_config,
        "environment-lock": args.environment_lock,
        "graph-source-config": args.graph_source_config,
        "graph-ensemble-config": args.graph_ensemble_config,
    }
    missing_frozen_inputs = [
        f"{name}={path}" for name, path in frozen_input_paths.items() if not path.is_file()
    ]
    if missing_frozen_inputs:
        parser.error(f"Frozen schedule inputs are missing: {missing_frozen_inputs}")
    if not args.schedule_manifest.is_file():
        parser.error(f"Schedule manifest does not exist: {args.schedule_manifest}")
    schedule_record = {
        "record_type": "strict_lopo_runner",
        "executor": "run_benchmark.py",
        "stage": args.stage,
        "dataset": args.datasets[0] if len(args.datasets) == 1 else None,
        "hvg": args.num_hvg,
        "graph_type": args.graph_types[0] if len(args.graph_types) == 1 else None,
        "training_seed": args.seed,
        "panel_size": args.panel_size,
        "fold_start": args.fold_start,
        "fold_end": args.fold_end,
        "planned_folds": args.panel_size,
        "graph_instance_seed": args.graph_instance_seed,
    }
    if len(args.datasets) != 1 or len(args.graph_types) != 1:
        parser.error("Scheduled canonical invocations require exactly one dataset and graph type")
    try:
        schedule_manifest_hash, schedule_content_hash = validate_schedule_binding(
            manifest_path=args.schedule_manifest,
            configuration_hash=args.schedule_config_hash,
            code_commit=args.code_commit,
            expected_configuration=schedule_record,
            frozen_input_paths=frozen_input_paths,
        )
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    try:
        dataset_passport_registry_hash = sha256_file(args.dataset_passport)
        passports = load_dataset_passports(args.dataset_passport)
        passport_map = {passport.dataset: passport for passport in passports}
        missing_passports = sorted(set(args.datasets) - set(passport_map))
        if missing_passports:
            raise ValueError(f"Dataset registry lacks requested datasets: {missing_passports}")
        data_root = args.data_root.resolve()
        resolved_paths: dict[str, Path] = {}
        for dataset in args.datasets:
            destination = (data_root / passport_map[dataset].relative_path).resolve()
            try:
                destination.relative_to(data_root)
            except ValueError as error:
                raise ValueError(f"Dataset {dataset!r} resolves outside --data-root") from error
            validate_dataset_file(destination, passport_map[dataset])
            resolved_paths[dataset] = destination
        DATASET_PATHS.clear()
        DATASET_PATHS.update(resolved_paths)
        DATASET_PASSPORTS.clear()
        DATASET_PASSPORTS.update({dataset: passport_map[dataset] for dataset in args.datasets})
        control_map = load_control_map(args.control_map)
        target_maps = load_target_maps(args.target_map)
        preprocessing_spec = load_preprocessing_spec(args.preprocessing_config)
        for dataset in args.datasets:
            passport = passport_map[dataset]
            control = control_map.get(dataset)
            if control is None:
                raise ValueError(f"Control map lacks requested dataset {dataset!r}")
            if (
                passport.perturbation_column != control.perturbation_column
                or passport.control_field != control.perturbation_column
                or passport.control_value != control.control_label
            ):
                raise ValueError(f"Dataset passport/control-map mismatch for {dataset!r}")
            if passport.input_expression_state != preprocessing_spec.input_expression_state:
                raise ValueError(
                    f"Dataset passport/preprocessing expression-state mismatch for {dataset!r}"
                )
            if (
                passport.minimum_perturbed_cells != preprocessing_spec.minimum_perturbed_cells
                or passport.cell_qc_policy != preprocessing_spec.cell_qc_policy
            ):
                raise ValueError(f"Dataset passport/QC-policy mismatch for {dataset!r}")
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    try:
        model_spec = load_model_spec(args.model_config)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if cublas_workspace not in {None, ":4096:8"}:
        parser.error("CUBLAS_WORKSPACE_CONFIG must be ':4096:8' for the frozen model contract")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(bool(model_spec.values["deterministic_algorithms"]))
    torch.set_float32_matmul_precision(str(model_spec.runtime["torch_float32_matmul_precision"]))
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = bool(model_spec.runtime["cuda_matmul_allow_tf32"])
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = bool(model_spec.runtime["cudnn_allow_tf32"])
    if re.fullmatch(r"[0-9a-f]{40}", args.code_commit) is None:
        parser.error("--code-commit must be a lowercase 40-character Git SHA")
    try:
        verify_code_commit(args.code_commit, Path(__file__).resolve().parent)
    except RuntimeError as error:
        parser.error(str(error))
    if not args.environment_lock.is_file():
        parser.error(f"--environment-lock is not a readable file: {args.environment_lock}")
    environment_lock_hash = sha256_file(args.environment_lock)
    graph_source_config_hash = sha256_file(args.graph_source_config)
    graph_ensemble_config_hash = sha256_file(args.graph_ensemble_config)
    global_input_contract_policy_hash = build_input_contract_policy_hash(
        model_spec, args.code_commit
    )
    selected_graph_types = set(args.graph_types)
    topology_selected = selected_graph_types & {
        "degree_preserving_rewired",
        "barabasi_albert",
    }
    if len(topology_selected) > 1:
        parser.error("Run at most one topology ensemble type per invocation")
    ensemble_specs: dict[str, GraphEnsembleSpec] = {}
    if topology_selected:
        if args.graph_instance_seed is None:
            parser.error("--graph-instance-seed is required for topology ensemble graphs")
        if args.graph_ensemble_config is None or not args.graph_ensemble_config.is_file():
            parser.error(f"Graph ensemble config does not exist: {args.graph_ensemble_config}")
        with args.graph_ensemble_config.open(encoding="utf-8") as handle:
            ensemble_values = json.load(handle)
        bundled_ensemble_path = Path(__file__).resolve().parent / "config" / "graph_ensembles.json"
        with bundled_ensemble_path.open(encoding="utf-8") as handle:
            bundled_ensemble_values = json.load(handle)
        if canonical_json(ensemble_values) != canonical_json(bundled_ensemble_values):
            parser.error(
                "--graph-ensemble-config differs from the bundled frozen ensemble contract"
            )
        if not isinstance(ensemble_values, Mapping) or not isinstance(
            ensemble_values.get("ensembles"), list
        ):
            parser.error("Graph ensemble config lacks an ensembles array")
        try:
            ensemble_specs = {
                spec.graph_type: spec for spec in load_ensemble_specs(ensemble_values["ensembles"])
            }
        except (KeyError, TypeError, ValueError) as error:
            parser.error(f"Invalid graph ensemble config: {error}")
        topology_type = next(iter(topology_selected))
        spec = ensemble_specs.get(topology_type)
        if spec is None or args.graph_instance_seed not in spec.instance_seeds:
            parser.error(
                f"Graph seed {args.graph_instance_seed} is not frozen for {topology_type!r}"
            )
    elif args.graph_instance_seed is not None:
        parser.error("--graph-instance-seed is only valid for topology ensemble graph types")

    curated_required: list[str] = []
    if selected_graph_types & {
        "string_ppi",
        "string_go_union",
        "degree_preserving_rewired",
        "barabasi_albert",
    }:
        curated_required.append("string_ppi")
    if selected_graph_types & {
        "gene_ontology",
        "string_go_union",
        "degree_preserving_rewired",
        "barabasi_albert",
    }:
        curated_required.append("gene_ontology")
    frozen_sources: dict[str, FrozenGraphSource] = {}
    if curated_required:
        if args.graph_source_config is None or not args.graph_source_config.is_file():
            parser.error(
                "--graph-source-config with frozen checksummed edge lists is required for "
                f"curated graph types: {sorted(set(curated_required))}"
            )
        with args.graph_source_config.open(encoding="utf-8") as handle:
            graph_source_values = json.load(handle)
        if not isinstance(graph_source_values, Mapping):
            parser.error("--graph-source-config must contain a JSON object")
        try:
            frozen_sources = load_frozen_graph_sources(
                graph_source_values,
                config_path=args.graph_source_config,
                required=sorted(set(curated_required)),
            )
        except ValueError as error:
            parser.error(str(error))
    missing_control_specs = sorted(set(args.datasets) - set(control_map))
    if missing_control_specs:
        parser.error(f"Control map lacks requested datasets: {missing_control_specs}")
    missing_target_specs = sorted(set(args.datasets) - set(target_maps))
    if missing_target_specs:
        parser.error(f"Target map lacks requested datasets: {missing_target_specs}")

    global _active_manifest_path, _active_manifest_records, _results_dir
    num_hvg = args.num_hvg
    repository = Path(__file__).resolve().parent
    _results_dir = args.results_dir.resolve()
    try:
        _results_dir.relative_to(repository)
    except ValueError:
        pass
    else:
        parser.error("--results-dir must be outside the Git repository")
    _results_dir.mkdir(parents=True, exist_ok=True)

    max_folds = args.panel_size
    if args.seed not in model_spec.training["training_seeds"]:
        parser.error(f"--seed must be one of frozen seeds {model_spec.training['training_seeds']}")
    seeds = [args.seed]
    mode_str = (
        f"{args.stage.upper()} (seed={args.seed}, panel={max_folds}, "
        f"folds={args.fold_start}:{args.fold_end}, hvg={num_hvg})"
    )
    configuration_universe_hash = sha256_json(
        {
            "stage": args.stage,
            "datasets": list(args.datasets),
            "graph_types": list(args.graph_types),
            "num_hvg": num_hvg,
            "panel_size": max_folds,
            "training_seed": args.seed,
            "fold_start": args.fold_start,
            "fold_end": args.fold_end,
            "graph_instance_seed": args.graph_instance_seed,
            "dataset_passport_registry_hash": dataset_passport_registry_hash,
            "control_map_hash": sha256_file(args.control_map),
            "target_map_hash": sha256_file(args.target_map),
            "preprocessing_config_hash": sha256_file(args.preprocessing_config),
            "model_config_hash": model_spec.file_hash,
            "environment_lock_hash": environment_lock_hash,
            "graph_ensemble_config_hash": graph_ensemble_config_hash,
            "graph_source_config_hash": graph_source_config_hash,
            "schedule_manifest_hash": schedule_manifest_hash,
            "schedule_content_hash": schedule_content_hash,
            "schedule_configuration_hash": args.schedule_config_hash,
        }
    )

    logger.info("=" * 70)
    logger.info("TurboGNN Graph Topology Benchmark — %s", mode_str)
    logger.info("Device: %s", DEVICE)
    logger.info("Datasets: %s", args.datasets)
    logger.info("Graph types: %s", args.graph_types)
    logger.info("Seeds: %s", seeds)
    logger.info("HVG: %d", num_hvg)
    logger.info("Results dir: %s", _results_dir)
    logger.info("=" * 70)

    t0 = time.time()

    for ds_name in args.datasets:
        logger.info("=" * 60)
        logger.info("DATASET: %s", ds_name)
        logger.info("=" * 60)

        # Check dataset exists
        if not DATASET_PATHS[ds_name].exists():
            raise FileNotFoundError(f"Dataset file not found: {DATASET_PATHS[ds_name]}")

        # Prepare data
        ds = prepare_dataset(
            ds_name,
            control_map=control_map,
            target_maps=target_maps,
            preprocessing_spec=preprocessing_spec,
            max_folds=max_folds,
            num_hvg=num_hvg,
        )
        evaluation_panel = ds["evaluation_panel"]
        evaluation_conditions = list(evaluation_panel.conditions)
        current_input_definition_hash = build_input_definition_hash(
            policy_hash=global_input_contract_policy_hash,
            dataset=ds_name,
            control_feature_hash=ds["control_feature_hash"],
            matrix_contract_hash=ds["matrix_contract_hash"],
            preprocessing_hash=ds["preprocessing_hash"],
            gene_order=ds["gene_list"],
        )

        # Build all graphs for this dataset
        graphs, graph_build_audits = build_all_graphs(
            ds["gene_list"],
            ds["X_coexpression_fit"],
            ds["num_genes"],
            args.graph_types,
            frozen_sources,
            args.graph_instance_seed,
            ensemble_specs,
            preprocessing_spec,
        )

        for gt in args.graph_types:
            logger.info("-" * 50)
            logger.info("Graph type: %s", gt)
            logger.info("-" * 50)

            edge_index = graphs.get(gt)
            is_gnn = gt != "transformer"

            # Skip if graph construction failed (but not for Transformer)
            if is_gnn and edge_index is None:
                raise RuntimeError(f"Requested graph {gt!r} is unavailable for dataset {ds_name!r}")

            # Configure model
            if is_gnn:
                gat_spec = model_spec.gat
                model_cls = TurboGNN
                model_kwargs = {
                    "num_genes": ds["num_genes"],
                    "edge_index": edge_index,
                    "hidden_dim": int(gat_spec["hidden_dim"]),
                    "num_heads": int(gat_spec["heads"]),
                    "dropout": float(gat_spec["dropout"]),
                    "gene_embedding_dim": int(model_spec.feature_encoder["gene_embedding_dim"]),
                    "feature_encoder_options": dict(model_spec.feature_encoder),
                    "gat_options": dict(gat_spec),
                }
            else:
                transformer_spec = model_spec.transformer
                model_cls = SimpleTransformer
                model_kwargs = {
                    "num_genes": ds["num_genes"],
                    "d_model": int(transformer_spec["d_model"]),
                    "nhead": int(transformer_spec["heads"]),
                    "num_layers": int(transformer_spec["layers"]),
                    "dim_feedforward": int(transformer_spec["feedforward_dim"]),
                    "dropout": float(transformer_spec["dropout"]),
                    "gene_embedding_dim": int(model_spec.feature_encoder["gene_embedding_dim"]),
                    "feature_encoder_options": dict(model_spec.feature_encoder),
                    "transformer_options": dict(transformer_spec),
                }

            graph_metadata = describe_graph(
                gt,
                edge_index,
                ds["gene_panel_hash"],
                ds["num_genes"],
                ds["coexpression_fit_hash"],
                frozen_sources,
                args.graph_instance_seed,
                ensemble_specs,
                graphs,
                graph_build_audits,
                preprocessing_spec,
                graph_ensemble_config_hash,
            )
            graph_instance = str(graph_metadata["graph_instance"])
            graph_contract_hash = sha256_json(graph_metadata)
            output_label = (
                gt if graph_instance in {f"{gt}__instance_0", "not_applicable"} else graph_instance
            )

            # Freeze the exact chunk and write a planned manifest before training.
            chunk_conditions = fold_slice_conditions(
                evaluation_conditions,
                max_folds,
                args.fold_start,
                args.fold_end,
            )
            slice_end = min(
                args.fold_end if args.fold_end is not None else max_folds,
                len(evaluation_conditions),
            )
            slice_start = min(args.fold_start, slice_end)
            if args.seed is not None:
                fold_tag = f"__s{args.seed}f{slice_start}-{slice_end}"
            elif args.fold_start > 0 or args.fold_end is not None:
                fold_tag = f"__f{slice_start}-{slice_end}"
            else:
                fold_tag = ""
            result_name = f"{ds_name}__{output_label}{fold_tag}.json"
            manifest_path = _results_dir / f"{ds_name}__{output_label}{fold_tag}.manifest.json"
            result_path_target = _results_dir / result_name
            if result_path_target.exists() or manifest_path.exists():
                raise FileExistsError(
                    "Refusing a silent rerun/overwrite; move or explicitly reconcile existing "
                    f"artifacts: {result_path_target}, {manifest_path}"
                )
            current_model_config_hash = sha256_json(
                {
                    "model_contract_file_hash": model_spec.file_hash,
                    "model_contract_canonical_hash": model_spec.canonical_hash,
                    "model_name": model_cls.__name__,
                }
            )
            condition_split_hashes = {
                condition: split_hash(
                    evaluation_panel.panel_hash,
                    evaluation_conditions,
                    condition,
                    ds["condition_alias_members"],
                    ds["condition_cell_ids"],
                )
                for condition in chunk_conditions
            }
            planned_records = tuple(
                RunRecord(
                    dataset=ds_name,
                    input_hash=ds["input_hash"],
                    environment_lock_hash=environment_lock_hash,
                    gene_panel_hash=ds["gene_panel_hash"],
                    condition_panel_hash=evaluation_panel.panel_hash,
                    preprocessing_hash=ds["preprocessing_hash"],
                    dataset_passport_registry_hash=dataset_passport_registry_hash,
                    dataset_passport_hash=ds["dataset_passport_hash"],
                    matrix_contract_hash=ds["matrix_contract_hash"],
                    control_selection_hash=ds["control_selection_hash"],
                    target_mapping_hash=ds["target_mapping_hash"],
                    condition_eligibility_ledger_hash=ds["condition_eligibility_ledger_hash"],
                    cell_qc_policy_hash=ds["cell_qc_policy_hash"],
                    response_definition_hash=ds["response_definition_hash"],
                    graph_type=gt,
                    graph_instance=str(graph_metadata["graph_instance"]),
                    graph_hash=str(graph_metadata["edge_hash"]),
                    graph_source_config_hash=graph_source_config_hash,
                    graph_ensemble_config_hash=graph_ensemble_config_hash,
                    graph_contract_hash=graph_contract_hash,
                    model_name=model_cls.__name__,
                    model_config_hash=current_model_config_hash,
                    input_contract_policy_hash=global_input_contract_policy_hash,
                    input_definition_hash=current_input_definition_hash,
                    configuration_universe_hash=configuration_universe_hash,
                    schedule_manifest_hash=schedule_manifest_hash,
                    schedule_content_hash=schedule_content_hash,
                    schedule_configuration_hash=args.schedule_config_hash,
                    split_hash=condition_split_hashes[condition],
                    code_commit=args.code_commit,
                    seed=seed,
                    fold_seed=derive_fold_seed(
                        seed,
                        condition,
                        condition_split_hashes[condition],
                    ),
                    condition=condition,
                )
                for seed in seeds
                for condition in chunk_conditions
            )
            RunManifest.build(planned_records).write(manifest_path)
            _active_manifest_path = manifest_path
            _active_manifest_records = planned_records

            seed_results: dict[int, list[dict]] = {}
            try:
                for seed_idx, seed in enumerate(seeds):
                    logger.info("  Seed %d/%d (seed=%d)", seed_idx + 1, len(seeds), seed)
                    fold_results = run_lopo(
                        model_cls=model_cls,
                        model_kwargs=model_kwargs,
                        control_feature_mean_tensor=ds["control_feature_mean_tensor"],
                        control_feature_log_variance_tensor=ds[
                            "control_feature_log_variance_tensor"
                        ],
                        ctrl_np=ds["ctrl_np"],
                        gene_list=ds["gene_list"],
                        cond_profiles=ds["cond_profiles"],
                        condition_targets=ds["condition_targets"],
                        valid_conds=evaluation_conditions,
                        is_gnn=is_gnn,
                        model_spec=model_spec,
                        seed=seed,
                        max_folds=max_folds,
                        fold_start=args.fold_start,
                        fold_end=args.fold_end,
                        run_identity_base={
                            "dataset": ds_name,
                            "input_hash": ds["input_hash"],
                            "environment_lock_hash": environment_lock_hash,
                            "gene_panel_hash": ds["gene_panel_hash"],
                            "condition_panel_hash": evaluation_panel.panel_hash,
                            "preprocessing_hash": ds["preprocessing_hash"],
                            "dataset_passport_registry_hash": dataset_passport_registry_hash,
                            "dataset_passport_hash": ds["dataset_passport_hash"],
                            "matrix_contract_hash": ds["matrix_contract_hash"],
                            "control_selection_hash": ds["control_selection_hash"],
                            "target_mapping_hash": ds["target_mapping_hash"],
                            "condition_eligibility_ledger_hash": ds[
                                "condition_eligibility_ledger_hash"
                            ],
                            "cell_qc_policy_hash": ds["cell_qc_policy_hash"],
                            "response_definition_hash": ds["response_definition_hash"],
                            "graph_type": gt,
                            "graph_instance": graph_metadata["graph_instance"],
                            "graph_hash": graph_metadata["edge_hash"],
                            "graph_source_config_hash": graph_source_config_hash,
                            "graph_ensemble_config_hash": graph_ensemble_config_hash,
                            "graph_contract_hash": graph_contract_hash,
                            "model_name": model_cls.__name__,
                            "model_config_hash": current_model_config_hash,
                            "input_contract_policy_hash": global_input_contract_policy_hash,
                            "input_definition_hash": current_input_definition_hash,
                            "configuration_universe_hash": configuration_universe_hash,
                            "schedule_manifest_hash": schedule_manifest_hash,
                            "schedule_content_hash": schedule_content_hash,
                            "schedule_configuration_hash": args.schedule_config_hash,
                            "code_commit": args.code_commit,
                        },
                        split_hashes=condition_split_hashes,
                        fold_metadata={
                            "dataset": ds_name,
                            "execution_stage": args.stage,
                            "configuration_universe_hash": configuration_universe_hash,
                            "schedule_manifest_hash": schedule_manifest_hash,
                            "schedule_content_hash": schedule_content_hash,
                            "schedule_configuration_hash": args.schedule_config_hash,
                            "input_hash": ds["input_hash"],
                            "dataset_passport_hash": ds["dataset_passport_hash"],
                            "dataset_passport_registry_hash": dataset_passport_registry_hash,
                            "matrix_contract_hash": ds["matrix_contract_hash"],
                            "matrix_layer": DATASET_PASSPORTS[ds_name].matrix_layer,
                            "matrix_dtype": DATASET_PASSPORTS[ds_name].matrix_dtype,
                            "environment_lock_hash": environment_lock_hash,
                            "condition_panel_hash": evaluation_panel.panel_hash,
                            "eligible_condition_panel_hash": ds[
                                "eligible_condition_panel"
                            ].panel_hash,
                            "condition_eligibility_ledger": [
                                row.as_dict()
                                for row in ds["eligible_condition_panel"].eligibility_ledger
                            ],
                            "condition_eligibility_ledger_hash": ds[
                                "condition_eligibility_ledger_hash"
                            ],
                            "cell_qc_policy_hash": ds["cell_qc_policy_hash"],
                            "evaluated_conditions": evaluation_conditions,
                            "panel_size": ds["panel_size"],
                            "gene_panel_hash": ds["gene_panel_hash"],
                            "target_mapping_hash": ds["target_mapping_hash"],
                            "canonical_objects": ds["canonical_objects"],
                            "condition_alias_members": {
                                condition: list(members)
                                for condition, members in ds["condition_alias_members"].items()
                            },
                            "raw_to_canonical_id": ds["raw_to_canonical_id"],
                            "raw_normalization_evidence": ds["raw_normalization_evidence"],
                            "raw_record_hashes": ds["raw_record_hashes"],
                            "preprocessing_hash": ds["preprocessing_hash"],
                            "scaler_hash": ds["scaler_hash"],
                            "coexpression_fit_hash": ds["coexpression_fit_hash"],
                            "control_selection_hash": ds["control_selection_hash"],
                            "condition_response_set_hash": ds["condition_response_set_hash"],
                            "response_definition_hash": ds["response_definition_hash"],
                            "preprocessing": {
                                **asdict(preprocessing_spec),
                                "target_retention_policy": (
                                    "explicit held-out identity targets retained "
                                    "outcome-independently; "
                                    "held-out expression and outcome-derived ranking forbidden"
                                ),
                            },
                            "control_label": ds["control_label"],
                            "control_evidence": ds["control_evidence"],
                            "graph": graph_metadata,
                            "graph_source_config_hash": graph_source_config_hash,
                            "graph_ensemble_config_hash": graph_ensemble_config_hash,
                            "graph_contract_hash": graph_contract_hash,
                            "model_name": model_cls.__name__,
                            "model_config_hash": current_model_config_hash,
                            "model_contract": dict(model_spec.values),
                            "model_contract_file_hash": model_spec.file_hash,
                            "model_contract_canonical_hash": model_spec.canonical_hash,
                            "control_feature_hash": ds["control_feature_hash"],
                            "input_contract_policy_hash": global_input_contract_policy_hash,
                            "input_definition_hash": current_input_definition_hash,
                            "code_commit": args.code_commit,
                        },
                        condition_alias_members=ds["condition_alias_members"],
                        condition_cell_ids=ds["condition_cell_ids"],
                    )
                    if len(fold_results) != len(chunk_conditions):
                        raise RuntimeError(
                            f"Seed {seed} returned {len(fold_results)} folds; "
                            f"manifest planned {len(chunk_conditions)}"
                        )
                    seed_results[seed] = fold_results
                    pearson_values = _valid_metric_values(fold_results, "pearson_r")
                    jaccard_values = _valid_metric_values(
                        fold_results, "gene_top20_absolute_delta_jaccard"
                    )
                    avg_p = _mean_or_none(pearson_values)
                    avg_j = _mean_or_none(jaccard_values)
                    logger.info(
                        "  Seed %d done: Pearson=%s (valid n=%d), Jaccard=%s (valid n=%d)",
                        seed,
                        f"{avg_p:.4f}" if avg_p is not None else "invalid",
                        len(pearson_values),
                        f"{avg_j:.4f}" if avg_j is not None else "invalid",
                        len(jaccard_values),
                    )

                result_path = save_combination_result(
                    ds_name,
                    gt,
                    graph_instance,
                    output_label,
                    seed_results,
                    _results_dir,
                    fold_tag,
                )
                if result_path.name != result_name:
                    raise RuntimeError(
                        f"Result filename {result_path.name!r} disagrees with manifest "
                        f"{result_name!r}"
                    )
                succeeded_records = tuple(
                    replace(
                        record,
                        status="succeeded",
                        result_path=result_name,
                        result_hash=sha256_file(result_path),
                    )
                    for record in planned_records
                )
                RunManifest.build(succeeded_records).write(manifest_path, overwrite=True)
            except BaseException as error:
                failed_records = tuple(
                    replace(
                        record,
                        status="failed",
                        failure_reason=f"{type(error).__name__}: {error}",
                    )
                    for record in planned_records
                )
                RunManifest.build(failed_records).write(manifest_path, overwrite=True)
                raise
            finally:
                _active_manifest_path = None
                _active_manifest_records = ()

    # Final summary
    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info("All experiments complete. Total time: %.1f min", elapsed / 60)
    logger.info("=" * 70)

    logger.info("Done. Results in %s", _results_dir)


if __name__ == "__main__":
    main()

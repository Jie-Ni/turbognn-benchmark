"""Dry-run-first runner for the matched-GAT major-revision benchmark."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .artifacts import (
    ArtifactValidationError,
    FoldArtifact,
    FoldIdentity,
    RuntimeRecord,
    canonical_sha256,
    file_sha256,
    replay_early_stopping,
    read_fold_artifact,
    write_fold_artifact,
)
from .baselines import (
    build_baseline_artifact,
    read_baseline_artifact,
    write_baseline_artifact,
)
from .controls import ControlDefinition, canonicalize_conditions, resolve_control_labels
from .data import (
    BenchmarkDataset,
    DatasetPassport,
    ExpressionSource,
    expression_scale_audit,
    hash_named_edges,
    load_benchmark_dataset,
    load_dataset_passport,
    load_named_edges,
)
from .design import hyperparameter_provenance_manifest, hyperparameter_provenance_table
from .errors import GraphSupportError, RevisionProtocolError
from .graph_supports import GraphMode, GraphSupport, build_graph_support
from .global_trigger import read_global_trigger_manifest
from .panels import ConditionPanels, build_condition_panels
from .preprocessing import (
    ControlOnlyPreprocessor,
    PreprocessingConfig,
    fit_nested_preprocessors,
    nested_gene_panel_manifest,
)
from .precision import read_precision_registry
from .protocol import load_protocol
from .shared_control import build_shared_control_evidence, write_shared_control_evidence
from .targets import TargetEncoder, TargetEncoding

DEFAULT_REWIRE_COUNT = 10
ENVIRONMENT_LOCK_FORMAT = "cbac-complete-pip-freeze-v2"
PYG_EXTENSION_PACKAGES = (
    "pyg-lib",
    "torch-cluster",
    "torch-scatter",
    "torch-sparse",
    "torch-spline-conv",
)
ENVIRONMENT_LOCK_HEADER_ORDER = (
    "lock-format",
    "python-version",
    "cuda-runtime-version",
    "cudnn-version",
    "nvidia-driver-version",
    "pytorch-version",
    "pytorch-geometric-version",
    "complete-distribution-set-sha256",
    "direct-runtime-dependencies-sha256",
    "pyg-extension-versions-sha256",
)


class PreflightBlockedError(RevisionProtocolError):
    """Raised when execution is requested despite reason-coded preflight blockers."""


@dataclass(frozen=True)
class RunnerSettings:
    """One dataset/HVG/panel execution request."""

    protocol_path: Path
    dataset_name: str
    data_path: Path
    output_dir: Path
    hvg: int
    analysis_block: str = "topology_primary"
    panel_name: str = "primary"
    panel_size: int = 50
    seeds: tuple[int, ...] = (42, 43, 44)
    arms: tuple[str, ...] = ()
    graph_paths: Mapping[str, Path] = field(default_factory=dict)
    dataset_passport_path: Path | None = None
    environment_lock_path: Path | None = None
    global_trigger_manifest_path: Path | None = None
    precision_registry_path: Path | None = None
    precision_condition_table_path: Path | None = None
    precision_condition_table_sha256: str | None = None
    precision_target_map_path: Path | None = None
    precision_target_map_sha256: str | None = None
    baseline_root_path: Path | None = None
    epochs_override: int | None = None
    device: str = "cpu"
    fixture_mode: bool = False

    def validate(self) -> None:
        """Validate CLI-level choices before loading any data."""

        if self.hvg <= 0 or self.panel_size <= 0:
            raise RevisionProtocolError("hvg and panel_size must be positive")
        if self.panel_name not in {"primary", "sensitivity"}:
            raise RevisionProtocolError("panel_name must be primary or sensitivity")
        if self.analysis_block not in {
            "topology_primary",
            "mixed_support_sensitivity",
            "scale_extension",
            "conditional_nonoverlap",
        }:
            raise RevisionProtocolError("Unknown analysis_block")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise RevisionProtocolError("seeds must be a non-empty unique sequence")
        if self.arms and len(set(self.arms)) != len(self.arms):
            raise RevisionProtocolError("arms must not contain duplicates")
        if self.epochs_override is not None and self.epochs_override <= 0:
            raise RevisionProtocolError("epochs_override must be positive")
        if self.fixture_mode and self.data_path.suffix.casefold() != ".npz":
            raise RevisionProtocolError("fixture_mode is permitted only for NPZ inputs")


@dataclass
class PreflightBundle:
    """Computed preflight state reused verbatim by explicit execution."""

    settings: RunnerSettings
    protocol: dict[str, Any]
    dataset: BenchmarkDataset
    expression_source: ExpressionSource
    expression_scale_audit: dict[str, Any]
    control_definition: ControlDefinition
    canonical_labels: np.ndarray
    preprocessor: ControlOnlyPreprocessor
    nested_gene_panels: dict[str, Any]
    panels: ConditionPanels
    selected_conditions: tuple[str, ...]
    target_encodings: dict[str, TargetEncoding]
    condition_profiles: dict[str, np.ndarray]
    model_control_expression: np.ndarray
    control_profile: np.ndarray
    shared_control_evidence_manifest: dict[str, Any]
    shared_control_matrix_bytes: bytes
    supports: dict[str, GraphSupport]
    input_hashes: dict[str, str]
    graph_manifest: pd.DataFrame
    graph_overlap: pd.DataFrame
    graph_provenance: pd.DataFrame
    hvg_manifest: pd.DataFrame
    target_audit: pd.DataFrame
    related_target_exclusion_audit: pd.DataFrame
    fit_matrix: pd.DataFrame
    condition_panel_manifest: dict[str, Any]
    topology_null_ensemble_audit: dict[str, Any]
    environment_manifest: dict[str, Any]
    git_provenance: dict[str, Any]
    global_trigger_manifest: dict[str, Any] | None
    precision_registry: dict[str, Any]
    hyperparameter_manifest: dict[str, Any]
    blockers: list[dict[str, str]]
    run_id: str
    code_hash: str
    preflight_summary_hash: str = ""

    @property
    def can_execute(self) -> bool:
        """Return whether every requested fit passed preflight."""

        return not self.blockers

    def _summary_payload(self) -> dict[str, Any]:
        """Return canonical summary content before adding its self hash."""

        state = self.preprocessor.state
        summary_input_hashes = {
            key: value for key, value in self.input_hashes.items() if key != "preflight_summary"
        }
        return {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "mode": "dry_run_preflight",
            "dataset": self.settings.dataset_name,
            "hvg": self.settings.hvg,
            "panel": self.settings.panel_name,
            "analysis_block": self.settings.analysis_block,
            "requested_panel_size": self.settings.panel_size,
            "selected_conditions": list(self.selected_conditions),
            "n_selected_conditions": len(self.selected_conditions),
            "seeds": list(self.settings.seeds),
            "requested_arms": list(self.fit_matrix["arm"]),
            "available_supports": sorted(self.supports),
            "planned_model_fits": int(self.fit_matrix["model_fits"].sum()),
            "preprocessing_fits": 1,
            "matched_control_labels": list(state.matched_control_labels if state else ()),
            "n_control_cells": int(state.n_control_cells if state else 0),
            "selected_genes": list(state.selected_genes if state else ()),
            "preprocessing_state_hash": state.sha256() if state else None,
            "nested_gene_panel_manifest": self.nested_gene_panels,
            "nested_gene_panel_manifest_hash": self.nested_gene_panels["manifest_sha256"],
            "input_hashes": summary_input_hashes,
            "expression_source": asdict(self.expression_source),
            "expression_scale_audit": dict(self.expression_scale_audit),
            "environment_manifest": self.environment_manifest,
            "git_provenance": self.git_provenance,
            "code_hash": self.code_hash,
            "can_execute": self.can_execute,
            "blockers": self.blockers,
            "sensitivity_trigger": dict(self.panels.sensitivity_trigger),
            "panel_representativeness_table_hash": self.panels.sensitivity_trigger[
                "representativeness_table_hash"
            ],
            "condition_panel_manifest": self.condition_panel_manifest,
            "condition_panel_manifest_hash": self.condition_panel_manifest["manifest_hash"],
            "shared_control_cell_evidence": self.shared_control_evidence_manifest,
            "shared_control_cell_evidence_hash": self.shared_control_evidence_manifest[
                "manifest_sha256"
            ],
            "conditional_sensitivity_feasible": len(self.panels.sensitivity)
            == self.settings.panel_size,
            "topology_null_ensemble_audit": dict(self.topology_null_ensemble_audit),
            "global_trigger_manifest": self.global_trigger_manifest,
            "precision_design_registry": self.precision_registry,
            "hyperparameter_provenance_manifest": self.hyperparameter_manifest,
        }

    def summary(self) -> dict[str, Any]:
        """Return the canonical summary with a verifiable content hash."""

        payload = self._summary_payload()
        observed_hash = canonical_sha256(payload)
        if self.preflight_summary_hash and self.preflight_summary_hash != observed_hash:
            raise RevisionProtocolError("[PREFLIGHT_SUMMARY_HASH_STATE_MISMATCH]")
        payload["summary_hash"] = observed_hash
        return payload


def preflight_matched_benchmark(settings: RunnerSettings) -> PreflightBundle:
    """Run data, label, panel, graph, hash, and fit-count checks without training."""

    settings.validate()
    protocol = load_protocol(settings.protocol_path)
    protocol_file_hash = file_sha256(settings.protocol_path)
    hyperparameter_manifest = hyperparameter_provenance_manifest(protocol)
    if settings.precision_registry_path is None:
        if settings.fixture_mode:
            precision_registry = {
                "status": "NOT_APPLICABLE_SYNTHETIC_FIXTURE",
                "registry_sha256": canonical_sha256({"status": "NOT_APPLICABLE_SYNTHETIC_FIXTURE"}),
            }
        else:
            raise PreflightBlockedError(
                "[PRECISION_REGISTRY_MISSING] Production execution requires the frozen "
                "pre-outcome precision registry"
            )
    else:
        if any(
            value is None
            for value in (
                settings.precision_condition_table_path,
                settings.precision_condition_table_sha256,
                settings.precision_target_map_path,
                settings.precision_target_map_sha256,
            )
        ):
            raise PreflightBlockedError(
                "[PRECISION_TRUSTED_SOURCES_MISSING] Production precision validation requires "
                "caller-pinned archive rows and target mapping"
            )
        precision_registry = read_precision_registry(
            settings.precision_registry_path,
            condition_table_path=settings.precision_condition_table_path,
            target_map_path=settings.precision_target_map_path,
            expected_condition_table_sha256=settings.precision_condition_table_sha256,
            expected_target_map_sha256=settings.precision_target_map_sha256,
        )
    global_trigger_manifest = None
    if settings.analysis_block == "conditional_nonoverlap":
        if settings.global_trigger_manifest_path is None:
            raise PreflightBlockedError(
                "[GLOBAL_TRIGGER_MANIFEST_MISSING] Conditional execution requires the "
                "four-dataset aggregate manifest"
            )
        global_trigger_manifest = read_global_trigger_manifest(
            settings.global_trigger_manifest_path, protocol, protocol_file_hash
        )
        if not global_trigger_manifest.get("global_triggered"):
            raise PreflightBlockedError(
                "[GLOBAL_CONDITIONAL_TRIGGER_NOT_MET] Conditional 1,200-fit block is not required"
            )
    if settings.dataset_name not in protocol["datasets"]:
        raise RevisionProtocolError(f"Dataset {settings.dataset_name!r} is absent from protocol")
    dataset_protocol = protocol["datasets"][settings.dataset_name]
    panel_protocol = protocol["condition_panel"]
    descriptive_metadata = tuple(panel_protocol["descriptive_metadata_columns"])
    dataset_passport = None
    if settings.data_path.suffix.casefold() == ".h5ad":
        if settings.dataset_passport_path is None:
            raise PreflightBlockedError(
                "[DATASET_PASSPORT_MISSING] Production H5AD input requires --dataset-passport"
            )
        dataset_passport = load_dataset_passport(
            settings.dataset_passport_path,
            dataset_name=settings.dataset_name,
            data_path=settings.data_path,
            expected_scale=protocol["preprocessing"]["input_scale"],
            expected_condition_column=(
                None
                if dataset_protocol["condition_column"] == "DATASET_PASSPORT_REQUIRED"
                else str(dataset_protocol["condition_column"])
            ),
        )
        expression_source = dataset_passport.expression_source
        condition_column = dataset_passport.control_evidence.condition_column
        if settings.environment_lock_path is None:
            raise PreflightBlockedError(
                "[ENVIRONMENT_LOCK_MISSING] Production H5AD requires --environment-lock"
            )
    else:
        expression_source = ExpressionSource(
            container="X",
            declared_scale=protocol["preprocessing"]["input_scale"],
        )
        condition_column = str(dataset_protocol["condition_column"])
    dataset = load_benchmark_dataset(
        settings.data_path,
        settings.dataset_name,
        condition_column,
        ("perturbation_order", *descriptive_metadata),
        expression_source=expression_source,
    )
    scale_audit = expression_scale_audit(dataset.expression, expression_source)
    control_protocol = dataset_protocol["control"]
    if dataset_passport is not None:
        passport_labels = dataset_passport.control_evidence.raw_labels
        control_like = {
            str(control_protocol["canonical_label"]).casefold(),
            *(str(value).casefold() for value in control_protocol["aliases"]),
            *(value.casefold() for value in passport_labels),
        }
        unverified_control_like = sorted(
            {
                str(value)
                for value in dataset.condition_labels
                if str(value).casefold() in control_like and str(value) not in set(passport_labels)
            }
        )
        if unverified_control_like:
            raise PreflightBlockedError(
                "[UNVERIFIED_CONTROL_LABEL_INJECTION] " + "|".join(unverified_control_like)
            )
        observed_control_counts = {
            label: int(np.sum(np.asarray(dataset.condition_labels, dtype=object) == label))
            for label in passport_labels
        }
        if observed_control_counts != dict(dataset_passport.control_evidence.raw_label_counts):
            raise PreflightBlockedError(
                "[CONTROL_EVIDENCE_COUNT_MISMATCH] "
                f"expected={dict(dataset_passport.control_evidence.raw_label_counts)};"
                f"observed={observed_control_counts}"
            )
        declared_control_values = (
            str(control_protocol["canonical_label"]),
            *(label for label in passport_labels if label != control_protocol["canonical_label"]),
        )
    else:
        declared_control_values = (
            str(control_protocol["canonical_label"]),
            *(str(value) for value in control_protocol["aliases"]),
        )
    seen_control_values: set[str] = set()
    deduplicated_controls: list[str] = []
    for value in declared_control_values:
        key = value if bool(control_protocol.get("case_sensitive", False)) else value.casefold()
        if key not in seen_control_values:
            seen_control_values.add(key)
            deduplicated_controls.append(value)
    control_definition = ControlDefinition(
        condition_column=condition_column,
        canonical_label=control_protocol["canonical_label"],
        aliases=tuple(deduplicated_controls[1:]),
        case_sensitive=(
            True
            if dataset_passport is not None
            else bool(control_protocol.get("case_sensitive", False))
        ),
    )
    resolution = resolve_control_labels(dataset.condition_labels, control_definition)
    canonical_labels = canonicalize_conditions(
        dataset.condition_labels,
        control_definition,
        aliases=dataset_protocol.get("condition_aliases", {}),
    )

    minimum_cells = int(protocol["evaluation"]["minimum_cells_per_condition"])
    target_separator_pattern = dataset_protocol.get(
        "target_separator_pattern",
        protocol["target_encoding"]["multi_target_separator_pattern"],
    )
    full_target_encoder = TargetEncoder(
        dataset.gene_names,
        gene_aliases=dataset_protocol.get("gene_aliases", {}),
        condition_targets=dataset_protocol.get("condition_targets", {}),
        case_sensitive=bool(protocol["target_encoding"]["case_sensitive"]),
        separators_pattern=target_separator_pattern,
        gene_space_name="dataset_gene_universe",
        condition_regex_pattern=dataset_protocol.get("condition_target_regex_pattern"),
        condition_regex_group=dataset_protocol.get("condition_target_regex_group", "target"),
        ignored_target_tokens=dataset_protocol.get("ignored_target_tokens", ()),
    )
    canonical_labels, full_target_encodings, raw_condition_members = (
        _canonicalize_unordered_target_conditions(
            canonical_labels,
            control_definition.canonical_label,
            full_target_encoder,
            pool_target_equivalent_labels=(
                dataset_protocol.get("target_equivalent_label_policy")
                == "pool_cells_before_panel_selection"
            ),
            excluded_endpoint_conditions=dataset_protocol.get("excluded_endpoint_conditions", {}),
        )
    )
    if dataset_passport is not None:
        validate_dataset_passport_binding(
            dataset,
            dataset_passport,
            canonical_labels=canonical_labels,
            target_encodings=full_target_encodings,
            control_label=control_definition.canonical_label,
        )
    cell_counts: dict[str, int] = {}
    eligibility_labels = canonical_labels.copy()
    all_conditions = sorted(set(str(value) for value in canonical_labels))
    for condition in all_conditions:
        if condition == control_definition.canonical_label:
            continue
        mask = canonical_labels == condition
        cell_counts[condition] = int(mask.sum())
        encoding = full_target_encodings[condition]
        if not encoding.success or cell_counts[condition] < minimum_cells:
            eligibility_labels[mask] = control_definition.canonical_label

    target_mapping_records = []
    for condition in all_conditions:
        if condition == control_definition.canonical_label:
            continue
        encoding = full_target_encodings[condition]
        cell_count_eligible = cell_counts[condition] >= minimum_cells
        if not cell_count_eligible:
            exclusion_reason = "INSUFFICIENT_CONDITION_CELLS"
        elif not encoding.success:
            exclusion_reason = encoding.reason_code or "TARGET_MAPPING_FAILED"
        else:
            exclusion_reason = None
        target_mapping_records.append(
            {
                "condition": condition,
                "raw_non_control_label_count": len(
                    raw_condition_members.get(condition, (condition,))
                ),
                "mapped": bool(encoding.success),
                "cell_count_eligible": bool(cell_count_eligible),
                "exclusion_reason_code": exclusion_reason,
            }
        )

    panels = build_condition_panels(
        eligibility_labels,
        control_definition.canonical_label,
        dataset.cell_metadata,
        panel_size=settings.panel_size,
        descriptive_metadata_columns=descriptive_metadata,
        selection_seed=int(panel_protocol["selection_seed"]),
        trigger_total_variation_threshold=float(
            panel_protocol["conditional_sensitivity_trigger"][
                "combined_selection_stratum_total_variation_distance_gt"
            ]
        ),
        trigger_cell_count_threshold=float(
            panel_protocol["conditional_sensitivity_trigger"][
                "absolute_standardized_log1p_condition_cell_count_difference_gt"
            ]
        ),
        dataset_name=settings.dataset_name,
        target_mapping_records=target_mapping_records,
        legacy_first50_conditions=(
            dataset_passport.legacy_first50_panel.ordered_condition_ids
            if dataset_passport is not None
            else ()
        ),
    )
    selected_conditions = panels.primary if settings.panel_name == "primary" else panels.sensitivity
    blockers: list[dict[str, str]] = []
    if len(panels.primary) != settings.panel_size:
        _add_blocker(
            blockers,
            "INSUFFICIENT_PRIMARY_PANEL_CONDITIONS",
            f"Requested {settings.panel_size}; selected {len(panels.primary)}",
        )
    if len(panels.sensitivity) != settings.panel_size and (
        settings.panel_name == "sensitivity" or settings.analysis_block == "conditional_nonoverlap"
    ):
        _add_blocker(
            blockers,
            "INSUFFICIENT_NONOVERLAP_SENSITIVITY_CONDITIONS",
            f"Requested {settings.panel_size}; selected {len(panels.sensitivity)}",
        )
    if len(selected_conditions) != settings.panel_size:
        _add_blocker(
            blockers,
            "INSUFFICIENT_PANEL_CONDITIONS",
            f"Requested {settings.panel_size}; selected {len(selected_conditions)}",
        )
    if settings.analysis_block == "conditional_nonoverlap":
        if settings.panel_name != "sensitivity":
            _add_blocker(
                blockers,
                "CONDITIONAL_NONOVERLAP_REQUIRES_SENSITIVITY_PANEL",
                "Use --panel sensitivity for the conditional non-overlap block",
            )

    forced_sources: dict[str, set[str]] = {}
    for condition in selected_conditions:
        for gene in full_target_encodings[condition].canonical_targets:
            forced_sources.setdefault(gene, set()).add(condition)
    smallest_required_panel = settings.hvg if settings.fixture_mode else 200
    if len(forced_sources) > smallest_required_panel:
        early_output = settings.output_dir / "preflight"
        early_output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "gene": gene,
                    "forced_by_conditions": "|".join(sorted(conditions)),
                    "forced": True,
                }
                for gene, conditions in sorted(forced_sources.items())
            ]
        ).to_csv(early_output / "hvg_forced_retention_overflow.csv", index=False)
        raise PreflightBlockedError(
            "[HVG_FORCED_RETENTION_OVERFLOW] "
            f"{len(forced_sources)} unique panel targets exceed the smallest required "
            f"panel={smallest_required_panel}"
        )

    preprocessing_protocol = protocol["preprocessing"]
    if not settings.fixture_mode and settings.hvg in {200, 500, 1000}:
        nested_preprocessors = fit_nested_preprocessors(
            dataset.expression,
            dataset.gene_names,
            dataset.condition_labels,
            control_definition,
            scales=(200, 500, 1000),
            input_scale=str(preprocessing_protocol["input_scale"]),
            target_library_size=float(preprocessing_protocol["target_library_size"]),
            hvg_method=str(preprocessing_protocol["hvg_method"]),
            forced_gene_names=tuple(sorted(forced_sources)),
        )
        preprocessor = nested_preprocessors[settings.hvg]
        nested_panels = nested_gene_panel_manifest(nested_preprocessors)
    else:
        preprocessor = ControlOnlyPreprocessor(
            PreprocessingConfig(
                n_hvg=settings.hvg,
                input_scale=preprocessing_protocol["input_scale"],
                target_library_size=float(preprocessing_protocol["target_library_size"]),
                hvg_method=preprocessing_protocol["hvg_method"],
            )
        )
        preprocessor.fit(
            dataset.expression,
            dataset.gene_names,
            dataset.condition_labels,
            control_definition,
            forced_gene_names=tuple(sorted(forced_sources)),
        )
        nested_panels = nested_gene_panel_manifest({settings.hvg: preprocessor})
    state = preprocessor.state
    if state is None:
        raise RevisionProtocolError("Preprocessor did not produce a fitted state")
    all_mean_profiles = preprocessor.mean_profiles_by_label(dataset.expression, canonical_labels)
    target_encoder = TargetEncoder(
        state.selected_genes,
        gene_aliases=dataset_protocol.get("gene_aliases", {}),
        condition_targets=dataset_protocol.get("condition_targets", {}),
        case_sensitive=bool(protocol["target_encoding"]["case_sensitive"]),
        separators_pattern=target_separator_pattern,
        gene_space_name="selected_gene_space",
        condition_regex_pattern=dataset_protocol.get("condition_target_regex_pattern"),
        condition_regex_group=dataset_protocol.get("condition_target_regex_group", "target"),
        ignored_target_tokens=dataset_protocol.get("ignored_target_tokens", ()),
    )
    target_encodings: dict[str, TargetEncoding] = {}
    condition_profiles: dict[str, np.ndarray] = {}
    target_rows: list[dict[str, Any]] = []
    for condition in all_conditions:
        if condition == control_definition.canonical_label:
            continue
        full_encoding = full_target_encodings[condition]
        n_cells = cell_counts[condition]
        eligible = full_encoding.success and n_cells >= minimum_cells
        post_selection_encoding = target_encoder.encode(condition) if eligible else full_encoding
        status = "ELIGIBLE" if eligible else "INELIGIBLE"
        reason_code = full_encoding.reason_code
        detail = full_encoding.detail
        if n_cells < minimum_cells:
            reason_code = "INSUFFICIENT_CONDITION_CELLS"
            detail = f"Observed {n_cells}; required at least {minimum_cells}"
        if condition in selected_conditions:
            target_encodings[condition] = post_selection_encoding
            if not post_selection_encoding.success:
                _add_blocker(
                    blockers,
                    post_selection_encoding.reason_code or "FORCED_TARGET_RETENTION_FAILED",
                    f"condition={condition!r}: {post_selection_encoding.detail}",
                )
            else:
                status = "SELECTED_PANEL"
                reason_code = None
                detail = None
        if condition in selected_conditions and post_selection_encoding.success:
            condition_profiles[condition] = all_mean_profiles[condition]
        target_rows.append(
            {
                "condition": condition,
                "raw_condition_labels": "|".join(
                    raw_condition_members.get(condition, (condition,))
                ),
                "n_cells": n_cells,
                "status": status,
                "reason_code": reason_code,
                "detail": detail,
                "canonical_targets": "|".join(full_encoding.canonical_targets),
                "target_indices": "|".join(
                    str(index) for index in post_selection_encoding.target_indices
                ),
                "eligible_before_hvg_selection": eligible,
                "target_forced_in_current_run": condition in selected_conditions,
                "in_primary_panel": condition in panels.primary,
                "in_sensitivity_panel": condition in panels.sensitivity,
            }
        )
    target_audit = pd.DataFrame(target_rows)
    condition_panel_manifest = _build_condition_panel_manifest(
        dataset_name=settings.dataset_name,
        protocol=protocol,
        protocol_file_hash=protocol_file_hash,
        dataset_source_hash=dataset.source_hash,
        dataset_passport_hash=(dataset_passport.source_hash if dataset_passport else None),
        requested_panel_size=settings.panel_size,
        control_definition=control_definition,
        matched_control_labels=resolution.matched_raw_labels,
        panels=panels,
        full_target_encodings=full_target_encodings,
        raw_condition_members=raw_condition_members,
    )
    if len(condition_profiles) < 3:
        _add_blocker(
            blockers,
            "INSUFFICIENT_ENCODABLE_CONDITIONS",
            "At least three encodable conditions are required for train/validation/held-out splits",
        )
    related_target_exclusion_audit = _build_related_target_exclusion_audit(
        selected_conditions, target_encodings
    )
    insufficient_unrelated = related_target_exclusion_audit[
        related_target_exclusion_audit["remaining_unrelated_conditions"] < 2
    ]
    if len(insufficient_unrelated):
        _add_blocker(
            blockers,
            "INSUFFICIENT_UNRELATED_TRAIN_VALIDATION_CONDITIONS",
            "Held-out conditions with fewer than two unrelated candidates: "
            + "|".join(insufficient_unrelated["held_out_condition"].astype(str)),
        )

    hvg_manifest = pd.DataFrame(
        [
            {
                "selected_position": position,
                "dataset_gene_index": state.selected_indices[position],
                "gene": gene,
                "control_variance_score": state.hvg_scores[position],
                "forced_selected_panel_target": gene in forced_sources,
                "forced_by_conditions": "|".join(sorted(forced_sources.get(gene, set()))),
                "selection_policy": state.selection_policy,
            }
            for position, gene in enumerate(state.selected_genes)
        ]
    )

    control_profile = all_mean_profiles[control_definition.canonical_label]
    control_mask = canonical_labels == control_definition.canonical_label
    standardized_control_profiles = preprocessor.transform(dataset.expression[control_mask])
    control_row_indices = np.flatnonzero(control_mask)
    shared_control_manifest, shared_control_matrix_bytes = build_shared_control_evidence(
        dataset=settings.dataset_name,
        hvg=settings.hvg,
        gene_names=state.selected_genes,
        control_row_ids=[
            f"{settings.dataset_name}::source_row_{int(index)}" for index in control_row_indices
        ],
        standardized_control_profiles=standardized_control_profiles,
        dataset_sha256=dataset.source_hash,
        preprocessing_state_sha256=state.sha256(),
    )
    unscaled_controls = preprocessor.transform_selected_unscaled(
        dataset.expression[resolution.mask]
    )
    model_control_expression = np.asarray(unscaled_controls.mean(axis=0)).reshape(-1)
    requested_arms = settings.arms or tuple(_default_arms(protocol, settings))
    (
        supports,
        graph_manifest,
        graph_overlap,
        graph_provenance,
        graph_hashes,
        graph_blockers,
    ) = _build_supports(
        dataset,
        state.selected_genes,
        unscaled_controls,
        protocol,
        requested_arms,
        settings.graph_paths,
        settings.fixture_mode,
    )
    blockers.extend(graph_blockers)
    topology_null_ensemble_audit, topology_null_blockers = _audit_topology_null_ensemble(
        supports,
        protocol,
        required=settings.analysis_block == "topology_primary",
        enforce=not settings.fixture_mode,
    )
    blockers.extend(topology_null_blockers)

    input_hashes = {
        "dataset": dataset.source_hash,
        "protocol": protocol_file_hash,
        "condition_panel_manifest": condition_panel_manifest["manifest_hash"],
        "panel_representativeness": panels.sensitivity_trigger["representativeness_table_hash"],
        "target_encoding_audit": canonical_sha256(
            target_audit.fillna("NOT_APPLICABLE").to_dict(orient="records")
        ),
        "nested_gene_panel_manifest": nested_panels["manifest_sha256"],
        "precision_design_registry": precision_registry["registry_sha256"],
        "precision_archive_condition_table": precision_registry.get(
            "trusted_source_bindings", {}
        ).get(
            "condition_table_file_sha256",
            canonical_sha256({"status": "NOT_APPLICABLE_SYNTHETIC_FIXTURE"}),
        ),
        "precision_target_map": precision_registry.get("trusted_source_bindings", {}).get(
            "target_map_file_sha256",
            canonical_sha256({"status": "NOT_APPLICABLE_SYNTHETIC_FIXTURE"}),
        ),
        "hyperparameter_provenance": hyperparameter_manifest["manifest_sha256"],
        "shared_control_cell_evidence": shared_control_manifest["manifest_sha256"],
        **graph_hashes,
    }
    if dataset_passport is not None:
        input_hashes["dataset_passport"] = dataset_passport.source_hash
    else:
        input_hashes["dataset_passport"] = canonical_sha256(
            {
                "status": "NOT_APPLICABLE_SYNTHETIC_NPZ_FIXTURE",
                "dataset": settings.dataset_name,
                "dataset_sha256": dataset.source_hash,
            }
        )
    if settings.global_trigger_manifest_path is not None:
        input_hashes["global_trigger_manifest"] = file_sha256(settings.global_trigger_manifest_path)
    code_hash = _code_sha256(Path(__file__).resolve().parent)
    environment_manifest = _build_environment_manifest(
        settings.environment_lock_path,
        fixture_mode=settings.fixture_mode,
        require_h5ad_runtime=settings.data_path.suffix.casefold() == ".h5ad",
    )
    input_hashes["environment_manifest"] = environment_manifest["manifest_hash"]
    input_hashes["environment_lock"] = environment_manifest["environment_lock_sha256"]
    git_provenance = _build_git_provenance(fixture_mode=settings.fixture_mode)
    input_hashes["git_provenance"] = git_provenance["provenance_hash"]
    fit_matrix = pd.DataFrame(
        [
            {
                "dataset": settings.dataset_name,
                "hvg": settings.hvg,
                "panel": settings.panel_name,
                "analysis_block": settings.analysis_block,
                "arm": arm,
                "panel_conditions": len(selected_conditions),
                "seeds": len(settings.seeds),
                "model_fits": len(selected_conditions) * len(settings.seeds),
                "support_preflight": "PASS" if arm in supports else "BLOCKED",
            }
            for arm in requested_arms
        ]
    )
    run_id = canonical_sha256(
        {
            "dataset": settings.dataset_name,
            "hvg": settings.hvg,
            "panel": settings.panel_name,
            "analysis_block": settings.analysis_block,
            "conditions": selected_conditions,
            "seeds": settings.seeds,
            "arms": requested_arms,
            "input_hashes": input_hashes,
            "preprocessing_state_hash": state.sha256(),
            "code_hash": code_hash,
        }
    )[:20]
    bundle = PreflightBundle(
        settings=settings,
        protocol=protocol,
        dataset=dataset,
        expression_source=expression_source,
        expression_scale_audit=scale_audit,
        control_definition=control_definition,
        canonical_labels=canonical_labels,
        preprocessor=preprocessor,
        nested_gene_panels=nested_panels,
        panels=panels,
        selected_conditions=tuple(selected_conditions),
        target_encodings=target_encodings,
        condition_profiles=condition_profiles,
        model_control_expression=model_control_expression,
        control_profile=control_profile,
        shared_control_evidence_manifest=shared_control_manifest,
        shared_control_matrix_bytes=shared_control_matrix_bytes,
        supports=supports,
        input_hashes=input_hashes,
        graph_manifest=graph_manifest,
        graph_overlap=graph_overlap,
        graph_provenance=graph_provenance,
        hvg_manifest=hvg_manifest,
        target_audit=target_audit,
        related_target_exclusion_audit=related_target_exclusion_audit,
        fit_matrix=fit_matrix,
        condition_panel_manifest=condition_panel_manifest,
        topology_null_ensemble_audit=topology_null_ensemble_audit,
        environment_manifest=environment_manifest,
        git_provenance=git_provenance,
        global_trigger_manifest=global_trigger_manifest,
        precision_registry=precision_registry,
        hyperparameter_manifest=hyperparameter_manifest,
        blockers=blockers,
        run_id=run_id,
        code_hash=code_hash,
    )
    bundle.preflight_summary_hash = canonical_sha256(bundle._summary_payload())
    bundle.input_hashes["preflight_summary"] = bundle.preflight_summary_hash
    write_preflight_outputs(bundle)
    return bundle


def write_preflight_outputs(bundle: PreflightBundle) -> None:
    """Write the complete dry-run evidence package."""

    output = bundle.settings.output_dir / "preflight"
    output.mkdir(parents=True, exist_ok=True)
    (output / "preflight_summary.json").write_text(
        json.dumps(bundle.summary(), indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )
    bundle.panels.condition_table.to_csv(output / "condition_panels.csv", index=False)
    bundle.panels.representativeness.to_csv(output / "panel_representativeness.csv", index=False)
    bundle.target_audit.to_csv(output / "target_encoding_audit.csv", index=False)
    bundle.related_target_exclusion_audit.to_csv(
        output / "related_target_exclusion_audit.csv", index=False
    )
    bundle.graph_manifest.to_csv(output / "graph_manifest.csv", index=False)
    bundle.graph_overlap.to_csv(output / "graph_edge_overlap.csv", index=False)
    bundle.graph_provenance.to_csv(output / "graph_provenance.csv", index=False)
    bundle.hvg_manifest.to_csv(output / "hvg_selection_manifest.csv", index=False)
    bundle.fit_matrix.to_csv(output / "fit_matrix.csv", index=False)
    write_shared_control_evidence(
        output,
        bundle.shared_control_evidence_manifest,
        bundle.shared_control_matrix_bytes,
    )
    (output / "condition_panel_manifest.json").write_text(
        json.dumps(bundle.condition_panel_manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )
    (output / "nested_gene_panel_manifest.json").write_text(
        json.dumps(bundle.nested_gene_panels, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )
    (output / "precision_design_registry.json").write_text(
        json.dumps(bundle.precision_registry, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )
    (output / "expression_scale_audit.json").write_text(
        json.dumps(bundle.expression_scale_audit, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    (output / "topology_null_ensemble_audit.json").write_text(
        json.dumps(bundle.topology_null_ensemble_audit, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    (output / "environment_manifest.json").write_text(
        json.dumps(bundle.environment_manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )
    (output / "git_provenance.json").write_text(
        json.dumps(bundle.git_provenance, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )
    hyperparameter_provenance_table(bundle.protocol).to_csv(
        output / "hyperparameter_provenance.csv", index=False
    )
    (output / "hyperparameter_provenance_manifest.json").write_text(
        json.dumps(bundle.hyperparameter_manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )


def execute_matched_benchmark(bundle: PreflightBundle) -> list[Path]:
    """Train only after an explicit caller invokes the execute path."""

    if not bundle.can_execute:
        codes = ", ".join(blocker["reason_code"] for blocker in bundle.blockers)
        raise PreflightBlockedError(f"Execution blocked by preflight: {codes}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise RevisionProtocolError("Execution requires torch and torch-geometric") from error
    from .model import (
        MatchedGAT,
        MatchedGATConfig,
        initialize_matched_gat,
        state_dict_sha256,
    )

    deterministic_backend = _configure_deterministic_backend(torch)

    training = bundle.protocol["training"]
    model_protocol = bundle.protocol["model"]
    identity_protocol = model_protocol["gene_identity_embedding"]
    model_config = MatchedGATConfig(
        n_genes=len(bundle.model_control_expression),
        hidden_dim=int(model_protocol["hidden_dim"]),
        num_heads=int(model_protocol["attention_heads"]),
        num_layers=int(model_protocol["layers"]),
        dropout=float(model_protocol["dropout"]),
        gene_identity_dim=int(identity_protocol["dimension"]),
    )
    epochs = bundle.settings.epochs_override or int(training["maximum_epochs"])
    device = torch.device(bundle.settings.device)
    artifact_root = bundle.settings.output_dir / "artifacts"
    artifacts: list[Path] = sorted(artifact_root.glob("*.json.gz"))
    failure_rows: list[dict[str, Any]] = []
    attempt_rows = _load_existing_compute_attempts(bundle)
    unresolved_attempts = [
        row for row in attempt_rows if row.get("execution_status") == "STARTED_UNFINALIZED"
    ]
    if unresolved_attempts:
        raise RevisionProtocolError(
            "[COMPUTE_UNRECONCILED_INTERRUPTED_ATTEMPT] "
            f"count={len(unresolved_attempts)}; reconcile scheduler elapsed time before resume"
        )
    hardware = _runtime_hardware(device, torch)
    baseline_paths: list[Path] = []

    for held_out in bundle.selected_conditions:
        encoding = bundle.target_encodings[held_out]
        if not encoding.success:
            raise PreflightBlockedError(
                f"[{encoding.reason_code}] selected condition {held_out!r} is unencodable"
            )
        train_conditions, validation_conditions = _condition_split(
            sorted(bundle.condition_profiles),
            held_out,
            training,
            bundle.target_encodings,
        )
        baseline_root = bundle.settings.baseline_root_path or (
            bundle.settings.output_dir.resolve().parent / "shared_analytic_baselines"
        )
        baseline_path = (
            baseline_root
            / bundle.settings.dataset_name
            / f"hvg{bundle.settings.hvg}"
            / bundle.settings.panel_name
            / f"{_safe_name(held_out)}.json.gz"
        )
        baseline_paths.append(baseline_path)
        if baseline_path.exists():
            baseline_payload = read_baseline_artifact(baseline_path)
            expected_identity = {
                "dataset": bundle.settings.dataset_name,
                "hvg": bundle.settings.hvg,
                "panel": bundle.settings.panel_name,
                "condition": held_out,
            }
            if baseline_payload.get("identity") != expected_identity:
                raise RevisionProtocolError(
                    "[BASELINE_ARTIFACT_IDENTITY_MISMATCH] Existing artifact cannot be reused"
                )
        else:
            training_profiles = np.stack(
                [
                    bundle.condition_profiles[condition] - bundle.control_profile
                    for condition in train_conditions
                ]
            )
            training_indicators = np.stack(
                [
                    bundle.target_encodings[condition].mask(len(bundle.control_profile))
                    for condition in train_conditions
                ]
            )
            baseline_payload = build_baseline_artifact(
                dataset=bundle.settings.dataset_name,
                hvg=bundle.settings.hvg,
                panel=bundle.settings.panel_name,
                condition=held_out,
                gene_names=bundle.preprocessor.state.selected_genes,
                training_condition_ids=train_conditions,
                training_profiles=training_profiles,
                training_target_indicators=training_indicators,
                held_out_target_indicator=encoding.mask(len(bundle.control_profile)),
                y_true=bundle.condition_profiles[held_out] - bundle.control_profile,
                ridge_alpha=float(bundle.protocol["analytic_baselines"]["ridge_alpha"]),
                input_hashes=bundle.input_hashes,
            )
            write_baseline_artifact(baseline_path, baseline_payload)
        for seed in bundle.settings.seeds:
            _set_random_seeds(seed)
            base_model = initialize_matched_gat(model_config, seed)
            initial_state = copy.deepcopy(base_model.state_dict())
            initialization_hash = state_dict_sha256(base_model)
            del base_model
            for arm in bundle.fit_matrix["arm"]:
                if _attempt_has_success(attempt_rows, bundle, str(arm), held_out, seed):
                    continue
                support = bundle.supports[str(arm)]
                source_support = (
                    bundle.supports.get("string_go")
                    if str(arm).startswith("string_go_rewire_")
                    else None
                )
                started = datetime.now(timezone.utc)
                wall_start = time.perf_counter()
                attempt_index = _next_attempt_index(attempt_rows, bundle, str(arm), held_out, seed)
                _start_compute_attempt(
                    bundle,
                    attempt_rows,
                    str(arm),
                    held_out,
                    seed,
                    hardware,
                    attempt_index,
                    started,
                )
                try:
                    _set_random_seeds(seed)
                    model = MatchedGAT(model_config).to(device)
                    model.load_state_dict(initial_state)
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=float(training["optimizer"]["learning_rate"]),
                        weight_decay=float(training["optimizer"]["weight_decay"]),
                    )
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=epochs,
                        eta_min=float(training["scheduler"]["eta_min"]),
                    )
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    edge_index = torch.as_tensor(
                        support.edge_index, dtype=torch.long, device=device
                    )
                    control = torch.as_tensor(
                        bundle.model_control_expression, dtype=torch.float32, device=device
                    )
                except Exception as error:
                    wall_seconds = float(time.perf_counter() - wall_start)
                    peak_memory, peak_status = _attempt_peak_memory(device, torch)
                    _record_compute_attempt(
                        bundle,
                        attempt_rows,
                        _compute_attempt_row(
                            bundle,
                            str(arm),
                            held_out,
                            seed,
                            hardware,
                            status="FAILED_SETUP_EXCEPTION",
                            wall_seconds=wall_seconds,
                            training_seconds=0.0,
                            inference_seconds=0.0,
                            checkpoint_seconds=0.0,
                            failure_reason=type(error).__name__,
                            attempt_index=attempt_index,
                            peak_device_memory_bytes=peak_memory,
                            peak_memory_status=peak_status,
                        ),
                    )
                    raise
                training_start = time.perf_counter()
                try:
                    (
                        train_loss,
                        validation_loss,
                        learning_rate,
                        best_state,
                        best_epoch,
                        early_stopping_observation,
                    ) = _train_one_fold(
                        model,
                        optimizer,
                        scheduler,
                        edge_index,
                        control,
                        train_conditions,
                        validation_conditions,
                        bundle.condition_profiles,
                        bundle.target_encodings,
                        epochs,
                        training,
                        device,
                        functional.mse_loss,
                    )
                except Exception as error:
                    training_seconds = float(time.perf_counter() - training_start)
                    wall_seconds = float(time.perf_counter() - wall_start)
                    peak_memory, peak_status = _attempt_peak_memory(device, torch)
                    _record_compute_attempt(
                        bundle,
                        attempt_rows,
                        _compute_attempt_row(
                            bundle,
                            str(arm),
                            held_out,
                            seed,
                            hardware,
                            status="FAILED_TRAINING_EXCEPTION",
                            wall_seconds=wall_seconds,
                            training_seconds=training_seconds,
                            inference_seconds=0.0,
                            checkpoint_seconds=0.0,
                            failure_reason=type(error).__name__,
                            attempt_index=attempt_index,
                            peak_device_memory_bytes=peak_memory,
                            peak_memory_status=peak_status,
                        ),
                    )
                    raise
                training_seconds = float(time.perf_counter() - training_start)
                inference_start = time.perf_counter()
                try:
                    model.load_state_dict(best_state)
                    model.eval()
                    held_out_indicator = torch.as_tensor(
                        encoding.mask(len(bundle.control_profile)),
                        dtype=torch.bool,
                        device=device,
                    )
                    with torch.no_grad():
                        predicted_profile = (
                            model(control, edge_index, held_out_indicator).cpu().numpy()
                        )
                except Exception as error:
                    inference_seconds = float(time.perf_counter() - inference_start)
                    wall_seconds = float(time.perf_counter() - wall_start)
                    peak_memory, peak_status = _attempt_peak_memory(device, torch)
                    _record_compute_attempt(
                        bundle,
                        attempt_rows,
                        _compute_attempt_row(
                            bundle,
                            str(arm),
                            held_out,
                            seed,
                            hardware,
                            status="FAILED_INFERENCE_EXCEPTION",
                            wall_seconds=wall_seconds,
                            training_seconds=training_seconds,
                            inference_seconds=inference_seconds,
                            checkpoint_seconds=0.0,
                            failure_reason=type(error).__name__,
                            attempt_index=attempt_index,
                            peak_device_memory_bytes=peak_memory,
                            peak_memory_status=peak_status,
                        ),
                    )
                    raise
                inference_seconds = float(time.perf_counter() - inference_start)
                true_delta = bundle.condition_profiles[held_out] - bundle.control_profile
                predicted_delta = predicted_profile - bundle.control_profile
                if np.std(true_delta) == 0 or np.std(predicted_delta) == 0:
                    failure_rows.append(
                        {
                            "dataset": bundle.settings.dataset_name,
                            "hvg": bundle.settings.hvg,
                            "panel": bundle.settings.panel_name,
                            "arm": arm,
                            "condition": held_out,
                            "seed": seed,
                            "reason_code": "DEGENERATE_CORRELATION_ZERO_VARIANCE",
                            "detail": "Stored prediction vectors would have undefined Pearson r",
                        }
                    )
                    wall_seconds = float(time.perf_counter() - wall_start)
                    peak_memory, peak_status = _attempt_peak_memory(device, torch)
                    _record_compute_attempt(
                        bundle,
                        attempt_rows,
                        _compute_attempt_row(
                            bundle,
                            str(arm),
                            held_out,
                            seed,
                            hardware,
                            status="FAILED_DEGENERATE_VECTOR",
                            wall_seconds=wall_seconds,
                            training_seconds=training_seconds,
                            inference_seconds=inference_seconds,
                            checkpoint_seconds=0.0,
                            failure_reason="DEGENERATE_CORRELATION_ZERO_VARIANCE",
                            attempt_index=attempt_index,
                            peak_device_memory_bytes=peak_memory,
                            peak_memory_status=peak_status,
                        ),
                    )
                    continue
                base_name = f"{_safe_name(held_out)}__{_safe_name(str(arm))}__seed{seed}"
                checkpoint_path = artifact_root / "checkpoints" / f"{base_name}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
                artifact_config = {
                    "protocol_id": bundle.protocol["protocol_id"],
                    "panel": bundle.settings.panel_name,
                    "analysis_block": bundle.settings.analysis_block,
                    "train_conditions": train_conditions,
                    "validation_conditions": validation_conditions,
                    "held_out_condition": held_out,
                    "held_out_canonical_targets": list(encoding.canonical_targets),
                    "canonical_target_set": "+".join(
                        sorted(encoding.canonical_targets, key=str.casefold)
                    ),
                    "gene_panel_contract": {
                        "nested_manifest_sha256": bundle.nested_gene_panels["manifest_sha256"],
                        "native_gene_order_sha256": canonical_sha256(
                            tuple(bundle.preprocessor.state.selected_genes)
                        ),
                        "common_evaluation_hvg": bundle.nested_gene_panels["common_evaluation_hvg"],
                        "common_gene_order": bundle.nested_gene_panels["common_gene_order"],
                        "common_gene_order_sha256": bundle.nested_gene_panels[
                            "common_gene_order_sha256"
                        ],
                        "nesting_rule": bundle.nested_gene_panels["nesting_rule"],
                    },
                    "model": asdict(model_config),
                    "training": training,
                    "epochs_requested": epochs,
                    "epochs_completed": len(train_loss),
                    "early_stopping_observation": early_stopping_observation,
                    "deterministic_backend": deterministic_backend,
                    "git_commit": bundle.git_provenance["git_commit"],
                    "git_worktree_status": bundle.git_provenance["git_worktree_status"],
                    "git_status_hash": bundle.git_provenance["git_status_hash"],
                    "graph_mode": support.mode.value,
                    "graph_diagnostics": _graph_diagnostics_payload(
                        str(arm), support, source_support
                    ),
                    "swapped_edge_fraction": support.swapped_edge_fraction,
                    "topology_null_ensemble_gate_status": (
                        bundle.topology_null_ensemble_audit["status"]
                    ),
                    "topology_null_ensemble_audit_hash": canonical_sha256(
                        bundle.topology_null_ensemble_audit
                    ),
                    "graph_manifest_hash": canonical_sha256(
                        bundle.graph_manifest.fillna("NOT_APPLICABLE").to_dict(orient="records")
                    ),
                    "global_trigger_manifest_hash": (
                        bundle.global_trigger_manifest.get("manifest_hash")
                        if bundle.global_trigger_manifest is not None
                        else None
                    ),
                    "spaces": {
                        "model_control_input": "control_only_log_normalized_mean_expression",
                        "gene_identity": (
                            "learned_embedding_bound_to_selected_gene_order_and_"
                            "preprocessing_state_hash"
                        ),
                        "training_target": ("control_fitted_standardized_condition_expression"),
                        "y_true_and_y_pred": ("control_fitted_standardized_delta_expression"),
                    },
                }
                checkpoint_start = time.perf_counter()
                try:
                    torch.save(
                        {
                            "schema_version": "1.0",
                            "identity": {
                                "run_id": bundle.run_id,
                                "dataset": bundle.settings.dataset_name,
                                "hvg": bundle.settings.hvg,
                                "arm": str(arm),
                                "condition": held_out,
                                "seed": seed,
                            },
                            "model_state_dict": best_state,
                            "optimizer_state_dict_at_stop": optimizer.state_dict(),
                            "scheduler_state_dict_at_stop": scheduler.state_dict(),
                            "best_epoch_zero_based": best_epoch,
                            "early_stopping_observation": early_stopping_observation,
                            "deterministic_backend": deterministic_backend,
                            "architecture": asdict(model_config),
                            "graph_support_hash": support.sha256(),
                            "preprocessing_state_hash": bundle.preprocessor.state.sha256(),
                            "git_commit": bundle.git_provenance["git_commit"],
                            "git_worktree_status": bundle.git_provenance["git_worktree_status"],
                            "git_status_hash": bundle.git_provenance["git_status_hash"],
                            "vector_space": "control_fitted_standardized_delta_expression",
                            "artifact_config_hash": canonical_sha256(artifact_config),
                            "input_hashes_hash": canonical_sha256(bundle.input_hashes),
                            "code_hash": bundle.code_hash,
                            "gene_names_hash": canonical_sha256(
                                tuple(bundle.preprocessor.state.selected_genes)
                            ),
                            "initialization_hash": initialization_hash,
                        },
                        checkpoint_temporary,
                    )
                    os.replace(checkpoint_temporary, checkpoint_path)
                except Exception as error:
                    checkpoint_seconds = float(time.perf_counter() - checkpoint_start)
                    wall_seconds = float(time.perf_counter() - wall_start)
                    peak_memory, peak_status = _attempt_peak_memory(device, torch)
                    checkpoint_temporary.unlink(missing_ok=True)
                    _record_compute_attempt(
                        bundle,
                        attempt_rows,
                        _compute_attempt_row(
                            bundle,
                            str(arm),
                            held_out,
                            seed,
                            hardware,
                            status="FAILED_CHECKPOINT_WRITE_EXCEPTION",
                            wall_seconds=wall_seconds,
                            training_seconds=training_seconds,
                            inference_seconds=inference_seconds,
                            checkpoint_seconds=checkpoint_seconds,
                            failure_reason=type(error).__name__,
                            attempt_index=attempt_index,
                            peak_device_memory_bytes=peak_memory,
                            peak_memory_status=peak_status,
                        ),
                    )
                    raise
                checkpoint_seconds = float(time.perf_counter() - checkpoint_start)
                finished = datetime.now(timezone.utc)
                wall_seconds = float(time.perf_counter() - wall_start)
                peak_device_memory, peak_memory_status = _attempt_peak_memory(device, torch)
                other_overhead = max(
                    0.0,
                    wall_seconds - training_seconds - inference_seconds - checkpoint_seconds,
                )
                runtime = RuntimeRecord(
                    started_at_utc=started.isoformat().replace("+00:00", "Z"),
                    finished_at_utc=finished.isoformat().replace("+00:00", "Z"),
                    wall_seconds=wall_seconds,
                    training_wall_seconds=training_seconds,
                    inference_wall_seconds=inference_seconds,
                    checkpoint_io_wall_seconds=checkpoint_seconds,
                    other_overhead_wall_seconds=other_overhead,
                    useful_compute_seconds=training_seconds + inference_seconds,
                    failed_attempt_wall_seconds=0.0,
                    retry_overhead_wall_seconds=(other_overhead if attempt_index > 1 else 0.0),
                    attempt_index=attempt_index,
                    retry_count=attempt_index - 1,
                    execution_status="SUCCESS",
                    device=str(device),
                    host=socket.gethostname(),
                    accelerator_model=hardware["accelerator_model"],
                    accelerator_count=hardware["accelerator_count"],
                    accelerator_hours=(wall_seconds * hardware["accelerator_count"] / 3600.0),
                    cpu_model=hardware["cpu_model"],
                    logical_cpu_count=hardware["logical_cpu_count"],
                    host_ram_bytes=hardware["host_ram_bytes"],
                    peak_device_memory_bytes=peak_device_memory,
                    peak_memory_status=peak_memory_status,
                    device_hours=wall_seconds / 3600.0,
                    timing_scope=(
                        "setup_plus_training_plus_inference_plus_checkpoint_io;"
                        "artifact_json_serialization_excluded"
                    ),
                )
                checkpoint_hash = file_sha256(checkpoint_path)
                artifact_path = artifact_root / f"{base_name}.json.gz"
                relative_checkpoint = os.path.relpath(checkpoint_path, artifact_path.parent)
                artifact = FoldArtifact(
                    identity=FoldIdentity(
                        run_id=bundle.run_id,
                        dataset=bundle.settings.dataset_name,
                        hvg=bundle.settings.hvg,
                        arm=str(arm),
                        condition=held_out,
                        seed=seed,
                    ),
                    gene_names=tuple(bundle.preprocessor.state.selected_genes),
                    y_true=tuple(float(value) for value in true_delta),
                    y_pred=tuple(float(value) for value in predicted_delta),
                    train_loss=tuple(train_loss),
                    validation_loss=tuple(validation_loss),
                    learning_rate=tuple(learning_rate),
                    vector_space="control_fitted_standardized_delta_expression",
                    config=artifact_config,
                    input_hashes=bundle.input_hashes,
                    graph_support_hash=support.sha256(),
                    preprocessing_state_hash=bundle.preprocessor.state.sha256(),
                    architecture_hash=model_config.sha256(),
                    initialization_hash=initialization_hash,
                    code_hash=bundle.code_hash,
                    checkpoint_path=relative_checkpoint,
                    checkpoint_hash=checkpoint_hash,
                    runtime=runtime,
                )
                artifact_hash = artifact.to_payload()["artifact_hash"]
                try:
                    artifacts.append(write_fold_artifact(artifact_path, artifact))
                except Exception as error:
                    wall_seconds = float(time.perf_counter() - wall_start)
                    peak_memory, peak_status = _attempt_peak_memory(device, torch)
                    _record_compute_attempt(
                        bundle,
                        attempt_rows,
                        _compute_attempt_row(
                            bundle,
                            str(arm),
                            held_out,
                            seed,
                            hardware,
                            status="FAILED_ARTIFACT_WRITE_EXCEPTION",
                            wall_seconds=wall_seconds,
                            training_seconds=training_seconds,
                            inference_seconds=inference_seconds,
                            checkpoint_seconds=checkpoint_seconds,
                            failure_reason=type(error).__name__,
                            attempt_index=attempt_index,
                            peak_device_memory_bytes=peak_memory,
                            peak_memory_status=peak_status,
                            checkpoint_hash=checkpoint_hash,
                        ),
                    )
                    raise
                _record_compute_attempt(
                    bundle,
                    attempt_rows,
                    _compute_attempt_row(
                        bundle,
                        str(arm),
                        held_out,
                        seed,
                        hardware,
                        status="SUCCESS",
                        wall_seconds=wall_seconds,
                        training_seconds=training_seconds,
                        inference_seconds=inference_seconds,
                        checkpoint_seconds=checkpoint_seconds,
                        failure_reason="NOT_APPLICABLE",
                        attempt_index=attempt_index,
                        peak_device_memory_bytes=peak_device_memory,
                        peak_memory_status=peak_memory_status,
                        artifact_hash=artifact_hash,
                        checkpoint_hash=checkpoint_hash,
                    ),
                )

    _write_measured_compute_outputs(bundle, attempt_rows)
    if failure_rows:
        failure_frame = pd.DataFrame(failure_rows)
        failure_frame.to_csv(bundle.settings.output_dir / "reason_coded_failures.csv", index=False)
        raise RevisionProtocolError(
            f"{len(failure_rows)} folds failed with reason-coded vector degeneracy; "
            "see reason_coded_failures.csv"
        )
    artifact_rows = []
    for path in sorted(set(artifacts)):
        payload = read_fold_artifact(path)
        artifact_rows.append(
            {
                "source_id": os.path.relpath(path, bundle.settings.output_dir).replace("\\", "/"),
                "file_sha256": file_sha256(path),
                "artifact_sha256": payload["artifact_hash"],
            }
        )
    pd.DataFrame(artifact_rows).to_csv(
        bundle.settings.output_dir / "artifact_manifest.csv", index=False
    )
    baseline_rows = []
    for path in sorted(set(baseline_paths)):
        payload = read_baseline_artifact(path)
        baseline_rows.append(
            {
                "source_id": os.path.relpath(path, bundle.settings.output_dir).replace("\\", "/"),
                "file_sha256": file_sha256(path),
                "artifact_sha256": payload["artifact_sha256"],
                "dataset": payload["identity"]["dataset"],
                "hvg": payload["identity"]["hvg"],
                "panel": payload["identity"]["panel"],
                "condition": payload["identity"]["condition"],
                "ridge_refits": payload["accounting"]["ridge_refits"],
                "analytic_baseline_evaluations": payload["accounting"][
                    "analytic_baseline_evaluations"
                ],
            }
        )
    pd.DataFrame(baseline_rows).to_csv(
        bundle.settings.output_dir / "analytic_baseline_manifest.csv", index=False
    )
    return artifacts


def _train_one_fold(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    edge_index: Any,
    control: Any,
    train_conditions: Sequence[str],
    validation_conditions: Sequence[str],
    condition_profiles: Mapping[str, np.ndarray],
    encodings: Mapping[str, TargetEncoding],
    epochs: int,
    training_protocol: Mapping[str, Any],
    device: Any,
    loss_function: Any,
) -> tuple[
    list[float],
    list[float],
    list[float],
    Mapping[str, Any],
    int,
    dict[str, Any],
]:
    train_history: list[float] = []
    validation_history: list[float] = []
    learning_rate_history: list[float] = []
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1
    epochs_without_improvement = 0
    patience = int(training_protocol["early_stopping"]["patience"])
    minimum_delta = float(training_protocol["early_stopping"]["minimum_delta"])
    stopped_by_patience = False

    for epoch in range(epochs):
        learning_rate_history.append(float(optimizer.param_groups[0]["lr"]))
        model.train()
        epoch_losses: list[float] = []
        for condition in train_conditions:
            indicator = _indicator_tensor(encodings[condition], len(control), device)
            target = _profile_tensor(condition_profiles[condition], device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(control, edge_index, indicator)
            loss = loss_function(prediction, target)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        train_history.append(float(np.mean(epoch_losses)))

        model.eval()
        losses: list[float] = []
        import torch

        with torch.no_grad():
            for condition in validation_conditions:
                indicator = _indicator_tensor(encodings[condition], len(control), device)
                target = _profile_tensor(condition_profiles[condition], device)
                losses.append(
                    float(loss_function(model(control, edge_index, indicator), target).cpu())
                )
        current_validation = float(np.mean(losses))
        if not np.isfinite(current_validation):
            raise RevisionProtocolError(
                "Non-finite validation loss prevents auditable early-stopping selection"
            )
        validation_history.append(current_validation)
        if current_validation < best_loss - minimum_delta:
            best_loss = current_validation
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        scheduler.step()
        if epochs_without_improvement >= patience:
            stopped_by_patience = True
            break
    early_stopping_observation = replay_early_stopping(
        validation_history,
        minimum_delta=minimum_delta,
        patience=patience,
        epochs_requested=epochs,
    )
    expected_stop_reason = "PATIENCE_EXHAUSTED" if stopped_by_patience else "MAXIMUM_EPOCHS_REACHED"
    if (
        best_epoch != early_stopping_observation["best_epoch_zero_based"]
        or early_stopping_observation["stopped_epoch_zero_based"] != len(validation_history) - 1
        or early_stopping_observation["stop_reason"] != expected_stop_reason
    ):
        raise RevisionProtocolError(
            "Observed early-stopping state differs from deterministic replay"
        )
    return (
        train_history,
        validation_history,
        learning_rate_history,
        best_state,
        best_epoch,
        early_stopping_observation,
    )


def _build_related_target_exclusion_audit(
    conditions: Sequence[str], encodings: Mapping[str, TargetEncoding]
) -> pd.DataFrame:
    """Record target-equivalent and shared-target exclusions before any model fit."""

    columns = [
        "held_out_condition",
        "held_out_targets",
        "excluded_related_conditions",
        "excluded_related_count",
        "remaining_unrelated_conditions",
    ]
    rows: list[dict[str, Any]] = []
    for held_out in conditions:
        held_targets = set(encodings[held_out].canonical_targets)
        excluded = sorted(
            condition
            for condition in conditions
            if condition != held_out
            and not held_targets.isdisjoint(encodings[condition].canonical_targets)
        )
        rows.append(
            {
                "held_out_condition": held_out,
                "held_out_targets": "|".join(sorted(held_targets)),
                "excluded_related_conditions": "|".join(excluded),
                "excluded_related_count": len(excluded),
                "remaining_unrelated_conditions": len(conditions) - 1 - len(excluded),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _runtime_hardware(device: Any, torch_module: Any) -> dict[str, Any]:
    """Capture stable hardware fields once per runner invocation."""

    try:
        import psutil
    except ImportError as error:
        raise RevisionProtocolError(
            "Measured compute accounting requires the declared psutil dependency"
        ) from error
    cpu_model = platform.processor().strip() or platform.machine().strip()
    if not cpu_model:
        raise RevisionProtocolError("[CPU_MODEL_UNAVAILABLE]")
    logical_cpu_count = os.cpu_count()
    if logical_cpu_count is None or logical_cpu_count <= 0:
        raise RevisionProtocolError("[CPU_COUNT_UNAVAILABLE]")
    if device.type == "cuda":
        accelerator_model = str(torch_module.cuda.get_device_name(device))
        accelerator_count = 1
    else:
        accelerator_model = "NOT_APPLICABLE_CPU_EXECUTION"
        accelerator_count = 0
    return {
        "accelerator_model": accelerator_model,
        "accelerator_count": accelerator_count,
        "cpu_model": cpu_model,
        "logical_cpu_count": int(logical_cpu_count),
        "host_ram_bytes": int(psutil.virtual_memory().total),
        "host": socket.gethostname(),
        "device": str(device),
    }


def _attempt_peak_memory(device: Any, torch_module: Any) -> tuple[int | None, str]:
    """Return measured CUDA peak memory or an explicit reason-coded CPU state."""

    if device.type == "cuda":
        try:
            return (
                int(torch_module.cuda.max_memory_allocated(device)),
                "MEASURED_CUDA_MAX_MEMORY_ALLOCATED",
            )
        except Exception:
            return None, "NOT_MEASURED_CUDA_QUERY_FAILED"
    return None, "NOT_APPLICABLE_CPU"


def _compute_attempt_row(
    bundle: PreflightBundle,
    arm: str,
    condition: str,
    seed: int,
    hardware: Mapping[str, Any],
    *,
    status: str,
    wall_seconds: float,
    training_seconds: float,
    inference_seconds: float,
    checkpoint_seconds: float,
    failure_reason: str,
    attempt_index: int,
    peak_device_memory_bytes: int | None = None,
    peak_memory_status: str = "NOT_MEASURED_ATTEMPT_FAILED_BEFORE_FINAL_SAMPLING",
    artifact_hash: str | None = None,
    checkpoint_hash: str | None = None,
) -> dict[str, Any]:
    """Create one success-or-failure row with an explicit monotonic attempt index."""

    overhead = max(
        0.0,
        wall_seconds - training_seconds - inference_seconds - checkpoint_seconds,
    )
    failed_seconds = wall_seconds if status != "SUCCESS" else 0.0
    fit_scope_seconds = training_seconds + inference_seconds
    return {
        "run_id": bundle.run_id,
        "dataset": bundle.settings.dataset_name,
        "hvg": bundle.settings.hvg,
        "panel": bundle.settings.panel_name,
        "analysis_block": bundle.settings.analysis_block,
        "arm": arm,
        "condition": condition,
        "seed": int(seed),
        "attempt_index": int(attempt_index),
        "retry_count": int(attempt_index - 1),
        "execution_status": status,
        "failure_reason": failure_reason,
        "wall_seconds": float(wall_seconds),
        "training_wall_seconds": float(training_seconds),
        "inference_wall_seconds": float(inference_seconds),
        "checkpoint_io_wall_seconds": float(checkpoint_seconds),
        "other_overhead_wall_seconds": float(overhead),
        "useful_compute_seconds": float(fit_scope_seconds),
        "fit_scope_compute_seconds": float(fit_scope_seconds),
        "scheduler_allocation_seconds": float(wall_seconds),
        "failed_attempt_wall_seconds": float(failed_seconds),
        "failed_preempted_allocation_seconds": float(failed_seconds),
        "retry_overhead_wall_seconds": float(overhead if attempt_index > 1 else 0.0),
        "retry_allocation_overhead_seconds": float(
            wall_seconds if attempt_index > 1 and status != "SUCCESS" else 0.0
        ),
        "serialization_registry_other_nonfit_seconds": float(checkpoint_seconds + overhead),
        "unattributed_interrupted_allocation_seconds": 0.0,
        "fit_scope_timing_status": (
            "MEASURED_IN_PROCESS" if status != "STARTED_UNFINALIZED" else "NOT_FINALIZED"
        ),
        "device": hardware["device"],
        "host": hardware["host"],
        "accelerator_model": hardware["accelerator_model"],
        "accelerator_count": int(hardware["accelerator_count"]),
        "accelerator_hours": float(wall_seconds * int(hardware["accelerator_count"]) / 3600.0),
        "device_hours": float(wall_seconds / 3600.0),
        "peak_device_memory_bytes": peak_device_memory_bytes,
        "peak_memory_status": peak_memory_status,
        "artifact_hash": artifact_hash,
        "checkpoint_hash": checkpoint_hash,
        "cpu_model": hardware["cpu_model"],
        "logical_cpu_count": int(hardware["logical_cpu_count"]),
        "host_ram_bytes": int(hardware["host_ram_bytes"]),
        "condition_panel_manifest_hash": bundle.condition_panel_manifest["manifest_hash"],
        "preflight_summary_hash": bundle.preflight_summary_hash,
        "code_hash": bundle.code_hash,
    }


def _record_compute_attempt(
    bundle: PreflightBundle,
    attempt_rows: list[dict[str, Any]],
    row: Mapping[str, Any],
) -> None:
    """Append and atomically publish every finalized attempt before continuing."""

    finalized = dict(row)
    key = _row_attempt_key(finalized)
    attempt_index = int(finalized["attempt_index"])
    matching_indices = [
        index
        for index, candidate in enumerate(attempt_rows)
        if _row_attempt_key(candidate) == key
        and int(candidate["attempt_index"]) == attempt_index
        and candidate.get("execution_status") == "STARTED_UNFINALIZED"
    ]
    if len(matching_indices) > 1:
        raise RevisionProtocolError("[COMPUTE_DUPLICATE_STARTED_ATTEMPT]")
    if matching_indices:
        started = attempt_rows[matching_indices[0]]
        finalized["attempt_started_at_utc"] = started["attempt_started_at_utc"]
        finalized["scheduler_job_id"] = started["scheduler_job_id"]
        finalized["attempt_finished_at_utc"] = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        finalized["timing_reconciliation_status"] = "FINALIZED_MEASURED"
        attempt_rows[matching_indices[0]] = finalized
    else:
        finalized.setdefault(
            "attempt_started_at_utc",
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        finalized.setdefault(
            "attempt_finished_at_utc",
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        finalized.setdefault("scheduler_job_id", _scheduler_job_id())
        finalized.setdefault("timing_reconciliation_status", "FINALIZED_MEASURED")
        attempt_rows.append(finalized)
    _write_measured_compute_outputs(bundle, attempt_rows)


def _start_compute_attempt(
    bundle: PreflightBundle,
    attempt_rows: list[dict[str, Any]],
    arm: str,
    condition: str,
    seed: int,
    hardware: Mapping[str, Any],
    attempt_index: int,
    started: datetime,
) -> None:
    """Durably publish a STARTED row so process loss cannot become invisible work."""

    row = _compute_attempt_row(
        bundle,
        arm,
        condition,
        seed,
        hardware,
        status="STARTED_UNFINALIZED",
        wall_seconds=0.0,
        training_seconds=0.0,
        inference_seconds=0.0,
        checkpoint_seconds=0.0,
        failure_reason="UNRECONCILED_PROCESS_INTERRUPTION_IF_NOT_FINALIZED",
        attempt_index=attempt_index,
        peak_memory_status="NOT_MEASURED_ATTEMPT_NOT_FINALIZED",
    )
    row.update(
        {
            "attempt_started_at_utc": started.isoformat().replace("+00:00", "Z"),
            "attempt_finished_at_utc": None,
            "scheduler_job_id": _scheduler_job_id(),
            "timing_reconciliation_status": "UNRECONCILED_STARTED",
        }
    )
    attempt_rows.append(row)
    _write_measured_compute_outputs(bundle, attempt_rows)


def _scheduler_job_id() -> str:
    for name in ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID", "JOB_ID"):
        value = os.environ.get(name, "").strip()
        if value:
            return f"{name}:{value}"
    return f"LOCAL_PROCESS:{os.getpid()}"


def _attempt_key(
    bundle: PreflightBundle, arm: str, condition: str, seed: int
) -> tuple[str, int, str, str, str, int]:
    return (
        bundle.settings.dataset_name,
        bundle.settings.hvg,
        bundle.settings.panel_name,
        arm,
        condition,
        int(seed),
    )


def _row_attempt_key(row: Mapping[str, Any]) -> tuple[str, int, str, str, str, int]:
    return (
        str(row["dataset"]),
        int(row["hvg"]),
        str(row["panel"]),
        str(row["arm"]),
        str(row["condition"]),
        int(row["seed"]),
    )


def _next_attempt_index(
    rows: Sequence[Mapping[str, Any]],
    bundle: PreflightBundle,
    arm: str,
    condition: str,
    seed: int,
) -> int:
    key = _attempt_key(bundle, arm, condition, seed)
    indices = [int(row["attempt_index"]) for row in rows if _row_attempt_key(row) == key]
    if indices and sorted(indices) != list(range(1, max(indices) + 1)):
        raise RevisionProtocolError("[COMPUTE_ATTEMPT_INDEX_SEQUENCE_INVALID]")
    return max(indices, default=0) + 1


def _attempt_has_success(
    rows: list[dict[str, Any]],
    bundle: PreflightBundle,
    arm: str,
    condition: str,
    seed: int,
) -> bool:
    key = _attempt_key(bundle, arm, condition, seed)
    matches = [
        index
        for index, row in enumerate(rows)
        if _row_attempt_key(row) == key and row["execution_status"] == "SUCCESS"
    ]
    if len(matches) > 1:
        raise RevisionProtocolError("[COMPUTE_MULTIPLE_SUCCESS_ATTEMPTS]")
    if not matches:
        return False
    index = matches[0]
    row = rows[index]
    base_name = f"{_safe_name(condition)}__{_safe_name(arm)}__seed{seed}"
    artifact_path = bundle.settings.output_dir / "artifacts" / f"{base_name}.json.gz"
    try:
        payload = read_fold_artifact(artifact_path)
        identity = payload["identity"]
        valid = (
            identity["run_id"] == bundle.run_id
            and identity["dataset"] == bundle.settings.dataset_name
            and int(identity["hvg"]) == bundle.settings.hvg
            and identity["arm"] == arm
            and identity["condition"] == condition
            and int(identity["seed"]) == int(seed)
            and payload["artifact_hash"] == row.get("artifact_hash")
            and payload["checkpoint_hash"] == row.get("checkpoint_hash")
        )
        if not valid:
            raise ArtifactValidationError("retained artifact identity or ledger hash mismatch")
    except (OSError, json.JSONDecodeError, ArtifactValidationError, KeyError, TypeError) as error:
        invalidated = dict(row)
        invalidated.update(
            {
                "execution_status": "FAILED_RETAINED_ARTIFACT_INVALID",
                "failure_reason": f"RETAINED_ARTIFACT_OR_CHECKPOINT_INVALID:{type(error).__name__}",
                "failed_attempt_wall_seconds": float(row["wall_seconds"]),
                "failed_preempted_allocation_seconds": float(row["wall_seconds"]),
                "artifact_validation_status": "INVALIDATED_BEFORE_RESUME_SKIP",
            }
        )
        rows[index] = invalidated
        _write_measured_compute_outputs(bundle, rows)
        return False
    return True


def _load_existing_compute_attempts(bundle: PreflightBundle) -> list[dict[str, Any]]:
    output = bundle.settings.output_dir
    registry_path = output / "measured_compute_registry.json"
    if not registry_path.exists():
        return []
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RevisionProtocolError(f"[COMPUTE_REGISTRY_INVALID] {error}") from error
    if not isinstance(registry, dict):
        raise RevisionProtocolError("[COMPUTE_REGISTRY_INVALID]")
    registry_payload = dict(registry)
    registry_hash = registry_payload.pop("registry_hash", None)
    if registry_hash != canonical_sha256(registry_payload):
        raise RevisionProtocolError("[COMPUTE_REGISTRY_HASH_MISMATCH]")
    source_id = str(registry.get("attempt_ledger_source_id", ""))
    source = Path(source_id)
    if not source_id or source.is_absolute() or ".." in source.parts:
        raise RevisionProtocolError("[COMPUTE_LEDGER_SOURCE_INVALID]")
    ledger_path = (output / source).resolve()
    if not ledger_path.is_relative_to(output.resolve()) or not ledger_path.is_file():
        raise RevisionProtocolError("[COMPUTE_LEDGER_MISSING]")
    try:
        rows = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RevisionProtocolError(f"[COMPUTE_LEDGER_INVALID] {error}") from error
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RevisionProtocolError("[COMPUTE_LEDGER_INVALID] root must be a list of objects")
    if registry.get("attempt_ledger_file_sha256") != file_sha256(ledger_path) or registry.get(
        "attempt_ledger_hash"
    ) != canonical_sha256(rows):
        raise RevisionProtocolError("[COMPUTE_LEDGER_HASH_MISMATCH]")
    if any(
        row.get("run_id") != bundle.run_id
        or row.get("preflight_summary_hash") != bundle.preflight_summary_hash
        for row in rows
    ):
        raise RevisionProtocolError("[COMPUTE_LEDGER_RUN_BINDING_MISMATCH]")
    for candidate in output.glob("compute_attempt_ledger.*.json"):
        if candidate.resolve() == ledger_path:
            continue
        try:
            candidate_rows = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(candidate_rows, list)
            and len(candidate_rows) > len(rows)
            and candidate_rows[: len(rows)] == rows
        ):
            raise RevisionProtocolError(
                "[COMPUTE_ORPHAN_LEDGER_DETECTED] newer durable ledger exists without "
                "an atomically published registry pointer"
            )
    return [dict(row) for row in rows]


def _write_measured_compute_outputs(
    bundle: PreflightBundle, attempt_rows: Sequence[Mapping[str, Any]]
) -> None:
    """Persist a fold-attempt ledger and self-hashed measured-compute registry."""

    output = bundle.settings.output_dir
    output.mkdir(parents=True, exist_ok=True)
    attempts = pd.DataFrame(attempt_rows)
    csv_path = output / "compute_attempt_ledger.csv"
    csv_temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    attempts.to_csv(csv_temporary, index=False)
    os.replace(csv_temporary, csv_path)
    ledger_payload = list(attempt_rows)
    ledger_content_hash = canonical_sha256(ledger_payload)
    ledger_path = output / f"compute_attempt_ledger.{ledger_content_hash}.json"
    ledger_temporary = ledger_path.with_name(f".{ledger_path.name}.tmp")
    ledger_temporary.write_text(
        json.dumps(ledger_payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(ledger_temporary, ledger_path)
    expected_keys = {
        _attempt_key(bundle, str(arm), condition, seed)
        for arm in bundle.fit_matrix["arm"]
        for condition in bundle.selected_conditions
        for seed in bundle.settings.seeds
    }
    observed_keys = {_row_attempt_key(row) for row in attempt_rows}
    success_counts = {
        key: sum(
            _row_attempt_key(row) == key and row["execution_status"] == "SUCCESS"
            for row in attempt_rows
        )
        for key in observed_keys
    }
    sequence_valid = all(
        sorted(int(row["attempt_index"]) for row in attempt_rows if _row_attempt_key(row) == key)
        == list(
            range(
                1,
                1 + sum(_row_attempt_key(row) == key for row in attempt_rows),
            )
        )
        for key in observed_keys
    )
    expected = len(expected_keys)
    successes = int(sum(row["execution_status"] == "SUCCESS" for row in attempt_rows))
    failures = len(attempt_rows) - successes
    complete = (
        observed_keys == expected_keys
        and all(success_counts.get(key) == 1 for key in expected_keys)
        and sequence_valid
        and all(
            max(
                (row for row in attempt_rows if _row_attempt_key(row) == key),
                key=lambda row: int(row["attempt_index"]),
            )["execution_status"]
            == "SUCCESS"
            for key in expected_keys
        )
    )

    def total(column: str) -> float:
        return float(sum(float(row.get(column, 0.0)) for row in attempt_rows))

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "registry_id": "MEASURED-COMPUTE",
        "status": ("RELEASED" if complete else "WITHHELD"),
        "run_id": bundle.run_id,
        "dataset": bundle.settings.dataset_name,
        "hvg": bundle.settings.hvg,
        "panel": bundle.settings.panel_name,
        "analysis_block": bundle.settings.analysis_block,
        "expected_attempts": expected,
        "observed_attempts": len(attempt_rows),
        "successful_attempts": successes,
        "failed_attempts": failures,
        "retried_attempts": int(sum(int(row["attempt_index"]) > 1 for row in attempt_rows)),
        "wall_seconds": total("wall_seconds"),
        "training_wall_seconds": total("training_wall_seconds"),
        "inference_wall_seconds": total("inference_wall_seconds"),
        "checkpoint_io_wall_seconds": total("checkpoint_io_wall_seconds"),
        "other_overhead_wall_seconds": total("other_overhead_wall_seconds"),
        "useful_compute_seconds": total("useful_compute_seconds"),
        "fit_scope_compute_seconds": total("fit_scope_compute_seconds"),
        "scheduler_allocation_seconds": total("scheduler_allocation_seconds"),
        "failed_attempt_wall_seconds": total("failed_attempt_wall_seconds"),
        "failed_preempted_allocation_seconds": total("failed_preempted_allocation_seconds"),
        "retry_overhead_wall_seconds": total("retry_overhead_wall_seconds"),
        "retry_allocation_overhead_seconds": float(
            sum(
                float(row.get("scheduler_allocation_seconds", row["wall_seconds"]))
                for row in attempt_rows
                if row["execution_status"] != "SUCCESS"
                and any(
                    _row_attempt_key(other) == _row_attempt_key(row)
                    and int(other["attempt_index"]) > int(row["attempt_index"])
                    for other in attempt_rows
                )
            )
        ),
        "serialization_registry_other_nonfit_seconds": total(
            "serialization_registry_other_nonfit_seconds"
        ),
        "unattributed_interrupted_allocation_seconds": total(
            "unattributed_interrupted_allocation_seconds"
        ),
        "accelerator_hours": total("accelerator_hours"),
        "device_hours": total("device_hours"),
        "attempt_ledger_hash": ledger_content_hash,
        "attempt_ledger_source_id": ledger_path.name,
        "attempt_ledger_file_sha256": file_sha256(ledger_path),
        "expected_fit_key_coverage": "PASS" if complete else "FAIL",
        "attempt_index_sequence_status": "PASS" if sequence_valid else "FAIL",
        "preflight_summary_hash": bundle.preflight_summary_hash,
        "condition_panel_manifest_hash": bundle.condition_panel_manifest["manifest_hash"],
        "accounting_scope": (
            "one retained row per fold attempt across invocations; scheduler allocation, "
            "in-process fit scope, failed/preempted allocation, retry allocation, and "
            "serialization/checkpoint/registry/other non-fit scope are separate; completed "
            "fits are skipped only after artifact and checkpoint revalidation"
        ),
    }
    payload["registry_hash"] = canonical_sha256(payload)
    registry_path = output / "measured_compute_registry.json"
    registry_temporary = registry_path.with_name(f".{registry_path.name}.tmp")
    registry_temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(registry_temporary, registry_path)


def _build_condition_panel_manifest(
    *,
    dataset_name: str,
    protocol: Mapping[str, Any],
    protocol_file_hash: str,
    dataset_source_hash: str,
    dataset_passport_hash: str | None,
    requested_panel_size: int,
    control_definition: ControlDefinition,
    matched_control_labels: Sequence[str],
    panels: ConditionPanels,
    full_target_encodings: Mapping[str, TargetEncoding],
    raw_condition_members: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Freeze both deterministic panels independently of HVG scale and graph arm."""

    by_condition = panels.condition_table.set_index("condition", drop=False)

    def entry(condition: str) -> dict[str, Any]:
        if condition not in by_condition.index or condition not in full_target_encodings:
            raise RevisionProtocolError(
                f"[PANEL_MANIFEST_CONDITION_MISSING] {dataset_name}:{condition}"
            )
        row = by_condition.loc[condition]
        encoding = full_target_encodings[condition]
        if not encoding.success:
            raise RevisionProtocolError(
                f"[PANEL_MANIFEST_UNENCODABLE_CONDITION] {dataset_name}:{condition}"
            )
        return {
            "condition": condition,
            "raw_members": list(raw_condition_members.get(condition, (condition,))),
            "canonical_targets": list(encoding.canonical_targets),
            "n_cells": int(row["n_cells"]),
            "condition_cell_count_quartile": str(row["condition_cell_count_quartile"]),
            "perturbation_order": str(row["perturbation_order"]),
            "selection_stratum": str(row["panel_stratum"]),
        }

    primary_entries = [entry(condition) for condition in panels.primary]
    sensitivity_entries = [entry(condition) for condition in panels.sensitivity]
    universe_entries = [entry(str(condition)) for condition in by_condition.index]

    def multiplicity(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for panel_entry in entries:
            target_key = "+".join(sorted(panel_entry["canonical_targets"], key=str.casefold))
            grouped.setdefault(target_key, []).append(panel_entry)
        return [
            {
                "canonical_target_set": target_key,
                "guide_condition_count": len(grouped_entries),
                "canonical_condition_ids": sorted(
                    str(value["condition"]) for value in grouped_entries
                ),
                "raw_member_count": sum(len(value["raw_members"]) for value in grouped_entries),
            }
            for target_key, grouped_entries in sorted(grouped.items())
        ]

    overlap = sorted(set(panels.primary) & set(panels.sensitivity))
    legacy_primary_overlap = sorted(set(panels.legacy_first50) & set(panels.primary))
    legacy_sensitivity_overlap = sorted(set(panels.legacy_first50) & set(panels.sensitivity))
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "manifest_type": "deterministic_condition_panel_contract",
        "protocol_id": str(protocol["protocol_id"]),
        "protocol_file_sha256": protocol_file_hash,
        "dataset": dataset_name,
        "dataset_file_sha256": dataset_source_hash,
        "dataset_passport_sha256": dataset_passport_hash or "NOT_APPLICABLE_FIXTURE",
        "control_binding": {
            "condition_column": control_definition.condition_column,
            "canonical_label": control_definition.canonical_label,
            "matched_raw_labels": list(matched_control_labels),
            "binding_source": (
                "hash_verified_dataset_passport"
                if dataset_passport_hash is not None
                else "synthetic_fixture_protocol"
            ),
        },
        "selection_seed": int(protocol["condition_panel"]["selection_seed"]),
        "selection_strata": list(protocol["condition_panel"]["required_metadata_strata"]),
        "requested_panel_size": int(requested_panel_size),
        "legacy_first50": {
            "ordered_condition_ids": list(panels.legacy_first50),
            "n_conditions": len(panels.legacy_first50),
            "binding_source": (
                "hash_verified_dataset_passport"
                if dataset_passport_hash is not None
                else "NOT_APPLICABLE_SYNTHETIC_FIXTURE"
            ),
            "primary_overlap": legacy_primary_overlap,
            "sensitivity_overlap": legacy_sensitivity_overlap,
        },
        "primary": {
            "ordered_canonical_condition_ids": list(panels.primary),
            "n_conditions": len(panels.primary),
            "entries": primary_entries,
            "canonical_target_multiplicity": multiplicity(primary_entries),
        },
        "sensitivity": {
            "ordered_canonical_condition_ids": list(panels.sensitivity),
            "n_conditions": len(panels.sensitivity),
            "entries": sensitivity_entries,
            "canonical_target_multiplicity": multiplicity(sensitivity_entries),
        },
        "primary_sensitivity_overlap": overlap,
        "non_overlap_status": (
            "PASS"
            if not overlap and not legacy_primary_overlap and not legacy_sensitivity_overlap
            else "FAIL"
        ),
        "primary_panel_contract_status": (
            "PASS" if len(panels.primary) == requested_panel_size else "PREFLIGHT_BLOCK"
        ),
        "conditional_panel_contract_status": (
            "PASS" if len(panels.sensitivity) == requested_panel_size else "WITHHELD_IF_TRIGGERED"
        ),
        "sensitivity_trigger": dict(panels.sensitivity_trigger),
        "representativeness_table_hash": panels.sensitivity_trigger[
            "representativeness_table_hash"
        ],
        "eligible_universe_target_mapping_hash": canonical_sha256(universe_entries),
    }
    payload["manifest_hash"] = canonical_sha256(payload)
    return payload


def _condition_split(
    conditions: Sequence[str],
    held_out: str,
    training_protocol: Mapping[str, Any],
    encodings: Mapping[str, TargetEncoding],
) -> tuple[list[str], list[str]]:
    if (
        training_protocol.get("related_target_exclusion")
        != "exclude_any_shared_target_with_held_out"
    ):
        raise RevisionProtocolError("Unsupported related-target exclusion policy")
    held_out_targets = set(encodings[held_out].canonical_targets)
    candidates = [
        condition
        for condition in conditions
        if condition != held_out
        and held_out_targets.isdisjoint(encodings[condition].canonical_targets)
    ]
    if len(candidates) < 2:
        raise PreflightBlockedError("Need at least two non-held-out encodable conditions")
    validation_seed = int(training_protocol["validation_selection_seed"])
    ordered = sorted(
        candidates,
        key=lambda value: hashlib.sha256(f"{validation_seed}:{value}".encode()).hexdigest(),
    )
    fraction = float(training_protocol["validation_fraction"])
    validation_size = min(len(ordered) - 1, max(1, int(round(len(ordered) * fraction))))
    validation = sorted(ordered[:validation_size])
    train = sorted(ordered[validation_size:])
    return train, validation


def _build_supports(
    dataset: BenchmarkDataset,
    selected_genes: Sequence[str],
    transformed_controls: np.ndarray | sparse.spmatrix,
    protocol: Mapping[str, Any],
    requested_arms: Sequence[str],
    graph_paths: Mapping[str, Path],
    fixture_mode: bool,
) -> tuple[
    dict[str, GraphSupport],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, str],
    list[dict[str, str]],
]:
    n_genes = len(selected_genes)
    supports: dict[str, GraphSupport] = {}
    blockers: list[dict[str, str]] = []
    graph_hashes: dict[str, str] = {}
    provenance, provenance_hashes, provenance_blockers = _graph_provenance_preflight(
        dataset, graph_paths, requested_arms, protocol, fixture_mode
    )
    graph_hashes.update(provenance_hashes)
    blockers.extend(provenance_blockers)
    named_sources: dict[str, tuple[tuple[str, str], ...]] = {}
    for source in ("string_ppi", "gene_ontology"):
        if source in graph_paths:
            named_sources[source] = load_named_edges(graph_paths[source])
            graph_hashes[f"graph_source_{source}"] = file_sha256(graph_paths[source])
        elif source in dataset.embedded_graphs:
            named_sources[source] = dataset.embedded_graphs[source]
            graph_hashes[f"graph_source_{source}"] = hash_named_edges(named_sources[source])
        elif any(arm in requested_arms for arm in (source, "combined", "string_go")) or any(
            arm.startswith("string_go_rewire_") for arm in requested_arms
        ):
            _add_blocker(
                blockers,
                "MISSING_GRAPH_SOURCE",
                f"Required named edge source {source!r} was not supplied",
            )
    if all(source in named_sources for source in ("string_ppi", "gene_ontology")):
        named_sources["string_go"] = tuple(
            sorted(
                {
                    tuple(sorted((left, right)))
                    for source in ("string_ppi", "gene_ontology")
                    for left, right in named_sources[source]
                    if left != right
                }
            )
        )

    indexed_sources: dict[str, np.ndarray] = {}
    mapping_audits: dict[str, dict[str, Any]] = {}
    for source, edges in named_sources.items():
        indexed_sources[source], mapping_audits[source] = _map_named_edges(edges, selected_genes)
    coexpression_protocol = protocol["graph_composition"]["coexpression"]
    indexed_sources["coexpression"] = _control_coexpression_edges(
        transformed_controls,
        threshold=float(coexpression_protocol["absolute_pearson_threshold"]),
    )
    graph_hashes["derived_control_coexpression"] = canonical_sha256(
        indexed_sources["coexpression"].tolist()
    )
    mapping_audits["coexpression"] = {
        "source_edges_total": len(indexed_sources["coexpression"]),
        "mapped_edges": len(indexed_sources["coexpression"]),
        "mapping_edge_coverage": 1.0,
        "source_genes_total": n_genes,
        "mapped_source_genes": n_genes,
        "mapping_node_coverage": 1.0,
        "mapping_status": "DERIVED_IN_SELECTED_GENE_SPACE",
    }

    if "dense" in requested_arms:
        supports["dense"] = build_graph_support(GraphMode.DENSE, n_genes)
    if "self_loop" in requested_arms:
        supports["self_loop"] = build_graph_support(GraphMode.SELF_LOOP, n_genes)
    for source in ("string_ppi", "gene_ontology", "coexpression"):
        if source not in requested_arms:
            continue
        edges = indexed_sources.get(source)
        if edges is None or not len(edges):
            _add_blocker(
                blockers,
                "EMPTY_GRAPH_AFTER_HVG_MAPPING",
                f"Graph arm {source!r} has no selected-gene edge",
            )
            continue
        supports[source] = build_graph_support(GraphMode.CURATED, n_genes, edges)

    string_go_needed = "string_go" in requested_arms or any(
        arm.startswith("string_go_rewire_") for arm in requested_arms
    )
    string_go_edges: np.ndarray | None = None
    if string_go_needed and "string_go" in indexed_sources:
        string_go_edges = indexed_sources["string_go"]
        if len(string_go_edges):
            if "string_go" in requested_arms:
                supports["string_go"] = build_graph_support(
                    GraphMode.CURATED, n_genes, string_go_edges
                )
        else:
            _add_blocker(
                blockers,
                "EMPTY_GRAPH_AFTER_HVG_MAPPING",
                "STRING-GO union has no selected-gene edge",
            )

    combined_needed = "combined" in requested_arms
    combined_edges: np.ndarray | None = None
    if combined_needed and all(
        source in indexed_sources for source in ("string_ppi", "gene_ontology")
    ):
        combined_set = {
            tuple(sorted((int(left), int(right))))
            for source in ("string_ppi", "gene_ontology", "coexpression")
            for left, right in indexed_sources[source]
            if left != right
        }
        combined_edges = np.asarray(sorted(combined_set), dtype=np.int64).reshape(-1, 2)
        if not len(combined_edges):
            _add_blocker(
                blockers,
                "EMPTY_GRAPH_AFTER_HVG_MAPPING",
                "Combined graph has no selected-gene edge",
            )
        elif "combined" in requested_arms:
            supports["combined"] = build_graph_support(GraphMode.CURATED, n_genes, combined_edges)

    rewire_seeds = protocol["graph_supports"]["topology_null"]["replicate_seeds"]
    if len(rewire_seeds) != DEFAULT_REWIRE_COUNT:
        _add_blocker(
            blockers,
            "REWIRE_REPLICATE_COUNT_MISMATCH",
            f"Protocol declares {len(rewire_seeds)} rewires; required {DEFAULT_REWIRE_COUNT}",
        )
    if string_go_edges is not None and len(string_go_edges):
        for index, rewire_seed in enumerate(rewire_seeds, start=1):
            arm = f"string_go_rewire_{index:02d}"
            if arm not in requested_arms:
                continue
            try:
                supports[arm] = build_graph_support(
                    GraphMode.DEGREE_PRESERVING_REWIRE,
                    n_genes,
                    string_go_edges,
                    rewire_seed=int(rewire_seed),
                    rewire_multiplier=float(
                        protocol["graph_supports"]["topology_null"]["rewire_multiplier"]
                    ),
                )
            except GraphSupportError as error:
                _add_blocker(blockers, "REWIRE_CONSTRUCTION_FAILED", f"{arm}: {error}")

    for arm in requested_arms:
        if arm not in supports and not any(
            blocker["detail"].startswith(f"Graph arm {arm!r}") for blocker in blockers
        ):
            _add_blocker(blockers, "SUPPORT_NOT_AVAILABLE", f"Requested arm {arm!r}")
    mapping_by_arm: dict[str, dict[str, Any]] = {
        **mapping_audits,
        "combined": _aggregate_mapping_audits(
            mapping_audits,
            ("string_go", "coexpression"),
            "EXTERNAL_PLUS_CONTROL_DERIVED_UNION",
        ),
    }
    manifest = pd.DataFrame(
        [
            {
                "arm": arm,
                "mode": support.mode.value,
                "n_nodes": support.n_nodes,
                "n_undirected_nonself_edges": support.n_undirected_nonself_edges,
                "degree_sequence_sha256": canonical_sha256(support.undirected_degree_sequence),
                "n_connected_components": support.n_connected_components,
                "giant_component_size": support.giant_component_size,
                "giant_component_fraction": support.giant_component_fraction,
                "n_isolates": support.n_isolates,
                "component_partition_hash": support.component_partition_hash,
                "mean_clustering_coefficient": support.mean_clustering_coefficient,
                "degree_assortativity": support.degree_assortativity,
                "degree_assortativity_status": support.degree_assortativity_status,
                "modularity": support.modularity,
                "modularity_status": support.modularity_status,
                "swapped_edge_fraction": support.swapped_edge_fraction,
                "rewire_seed": support.rewire_seed,
                "source_hash": support.source_hash,
                "source_arm": ("string_go" if arm.startswith("string_go_rewire_") else arm),
                "source_support_sha256": (
                    supports["string_go"].sha256()
                    if arm.startswith("string_go_rewire_") and "string_go" in supports
                    else support.sha256()
                ),
                "support_hash": support.sha256(),
                **mapping_by_arm.get(
                    "string_go" if arm.startswith("string_go_rewire_") else arm,
                    {
                        "source_edges_total": np.nan,
                        "mapped_edges": np.nan,
                        "mapping_edge_coverage": np.nan,
                        "source_genes_total": np.nan,
                        "mapped_source_genes": np.nan,
                        "mapping_node_coverage": np.nan,
                        "mapping_status": "NOT_APPLICABLE_SYNTHETIC_SUPPORT",
                    },
                ),
            }
            for arm, support in sorted(supports.items())
        ]
    )
    overlap = _graph_edge_overlap(supports)
    return supports, manifest, overlap, provenance, graph_hashes, blockers


def _map_named_edges(
    edges: Sequence[tuple[str, str]], genes: Sequence[str]
) -> tuple[np.ndarray, dict[str, Any]]:
    index = {str(gene).casefold(): position for position, gene in enumerate(genes)}
    canonical_source_edges = {
        tuple(sorted((str(left).casefold(), str(right).casefold())))
        for left, right in edges
        if str(left).casefold() != str(right).casefold()
    }
    mapped = {
        tuple(sorted((index[left.casefold()], index[right.casefold()])))
        for left, right in edges
        if left.casefold() in index
        and right.casefold() in index
        and left.casefold() != right.casefold()
    }
    source_genes = {gene for edge in canonical_source_edges for gene in edge}
    mapped_source_genes = source_genes & set(index)
    audit = {
        "source_edges_total": len(canonical_source_edges),
        "mapped_edges": len(mapped),
        "mapping_edge_coverage": (
            float(len(mapped) / len(canonical_source_edges)) if canonical_source_edges else 0.0
        ),
        "source_genes_total": len(source_genes),
        "mapped_source_genes": len(mapped_source_genes),
        "mapping_node_coverage": (
            float(len(mapped_source_genes) / len(source_genes)) if source_genes else 0.0
        ),
        "mapping_status": "MEASURED_NAMED_EDGE_MAPPING",
    }
    return np.asarray(sorted(mapped), dtype=np.int64).reshape(-1, 2), audit


def _aggregate_mapping_audits(
    audits: Mapping[str, Mapping[str, Any]], sources: Sequence[str], status: str
) -> dict[str, Any]:
    available = [audits[source] for source in sources if source in audits]
    if len(available) != len(sources):
        return {
            "source_edges_total": np.nan,
            "mapped_edges": np.nan,
            "mapping_edge_coverage": np.nan,
            "source_genes_total": np.nan,
            "mapped_source_genes": np.nan,
            "mapping_node_coverage": np.nan,
            "mapping_status": "INCOMPLETE_SOURCE_AUDIT",
        }
    source_edges = sum(int(audit["source_edges_total"]) for audit in available)
    mapped_edges = sum(int(audit["mapped_edges"]) for audit in available)
    source_genes = sum(int(audit["source_genes_total"]) for audit in available)
    mapped_genes = sum(int(audit["mapped_source_genes"]) for audit in available)
    return {
        "source_edges_total": source_edges,
        "mapped_edges": mapped_edges,
        "mapping_edge_coverage": float(mapped_edges / source_edges) if source_edges else 0.0,
        "source_genes_total": source_genes,
        "mapped_source_genes": mapped_genes,
        "mapping_node_coverage": float(mapped_genes / source_genes) if source_genes else 0.0,
        "mapping_status": status,
    }


def _graph_edge_overlap(supports: Mapping[str, GraphSupport]) -> pd.DataFrame:
    edge_sets: dict[str, set[tuple[int, int]]] = {}
    for arm, support in supports.items():
        edge_sets[arm] = {
            tuple(sorted((int(left), int(right))))
            for left, right in support.edge_index.T
            if left != right
        }
    rows: list[dict[str, Any]] = []
    arms = sorted(edge_sets)
    for left_index, left_arm in enumerate(arms):
        for right_arm in arms[left_index + 1 :]:
            intersection = edge_sets[left_arm] & edge_sets[right_arm]
            union = edge_sets[left_arm] | edge_sets[right_arm]
            rows.append(
                {
                    "left_arm": left_arm,
                    "right_arm": right_arm,
                    "intersection_edges": len(intersection),
                    "union_edges": len(union),
                    "edge_jaccard": float(len(intersection) / len(union)) if union else 1.0,
                }
            )
    return pd.DataFrame(rows)


def _graph_diagnostics_payload(
    arm: str,
    support: GraphSupport,
    source_support: GraphSupport | None,
) -> dict[str, Any]:
    """Bind one fitted arm to complete topology-preservation diagnostics."""

    source = source_support or support
    payload: dict[str, Any] = {
        "arm": arm,
        "mode": support.mode.value,
        "n_nodes": support.n_nodes,
        "n_undirected_nonself_edges": support.n_undirected_nonself_edges,
        "degree_sequence_sha256": canonical_sha256(support.undirected_degree_sequence),
        "n_connected_components": support.n_connected_components,
        "component_partition_sha256": support.component_partition_hash,
        "n_isolates": support.n_isolates,
        "support_sha256": support.sha256(),
        "source_arm": "string_go" if source_support is not None else arm,
        "source_support_sha256": source.sha256(),
        "source_edge_sha256": source.source_hash,
        "source_n_nodes": source.n_nodes,
        "source_n_undirected_nonself_edges": source.n_undirected_nonself_edges,
        "source_degree_sequence_sha256": canonical_sha256(source.undirected_degree_sequence),
        "source_n_connected_components": source.n_connected_components,
        "source_component_partition_sha256": source.component_partition_hash,
        "source_n_isolates": source.n_isolates,
        "swapped_edge_fraction": support.swapped_edge_fraction,
        "rewire_seed": support.rewire_seed,
        "cross_dataset_graph_index_pairing": "PROHIBITED_LOCAL_INSTANCE_LABEL",
    }
    payload["diagnostics_sha256"] = canonical_sha256(payload)
    return payload


def _audit_topology_null_ensemble(
    supports: Mapping[str, GraphSupport],
    protocol: Mapping[str, Any],
    *,
    required: bool,
    enforce: bool,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Gate topology-null claims on ten unique, sufficiently rewired supports."""

    topology_protocol = protocol["graph_supports"]["topology_null"]
    seeds = tuple(int(seed) for seed in topology_protocol["replicate_seeds"])
    expected_arms = tuple(f"string_go_rewire_{index:02d}" for index in range(1, len(seeds) + 1))
    minimum_fraction = float(topology_protocol["minimum_swapped_edge_fraction"])
    available = [arm for arm in expected_arms if arm in supports]
    missing = [arm for arm in expected_arms if arm not in supports]
    hashes = [supports[arm].sha256() for arm in available]
    fractions = [
        float(supports[arm].swapped_edge_fraction)
        for arm in available
        if supports[arm].swapped_edge_fraction is not None
    ]
    duplicate_hash_count = len(hashes) - len(set(hashes))
    minimum_observed = min(fractions) if fractions else None
    failures: list[tuple[str, str]] = []
    if required and missing:
        failures.append(
            (
                "INCOMPLETE_TOPOLOGY_NULL_ENSEMBLE",
                f"Missing topology-null supports: {missing}",
            )
        )
    if duplicate_hash_count:
        failures.append(
            (
                "DUPLICATE_TOPOLOGY_NULL_SUPPORT_HASH",
                f"Observed {duplicate_hash_count} duplicate support hashes",
            )
        )
    low_coverage = [
        arm
        for arm in available
        if supports[arm].swapped_edge_fraction is None
        or float(supports[arm].swapped_edge_fraction) < minimum_fraction
    ]
    if low_coverage:
        failures.append(
            (
                "REWIRE_SWAPPED_EDGE_FRACTION_BELOW_MINIMUM",
                f"Required >= {minimum_fraction:.2f}; failing supports: {low_coverage}",
            )
        )
    if not required and not available:
        status = "NOT_REQUESTED"
    elif failures and enforce:
        status = "BLOCKED"
    elif failures:
        status = "TEST_FIXTURE_ONLY_NOT_ENFORCED"
    else:
        status = "PASS"
    audit = {
        "required": required,
        "enforced": enforce,
        "status": status,
        "expected_supports": len(expected_arms),
        "available_supports": len(available),
        "missing_arms": missing,
        "unique_support_hashes": len(set(hashes)),
        "duplicate_support_hashes": duplicate_hash_count,
        "minimum_swapped_edge_fraction_required": minimum_fraction,
        "minimum_swapped_edge_fraction_observed": minimum_observed,
        "failing_low_coverage_arms": low_coverage,
    }
    blockers = [
        {"reason_code": reason_code, "detail": detail}
        for reason_code, detail in failures
        if enforce
    ]
    return audit, blockers


def _validate_derivation_manifest(
    *,
    kind: str,
    manifest_path: Path | None,
    edge_path: Path,
    builder_path: Path | None,
    input_paths: Mapping[str, Path | None],
    expected_release: str,
) -> tuple[dict[str, Any] | None, dict[str, str], list[dict[str, str]]]:
    """Verify that a graph edge list is cryptographically bound to its derivation contract."""

    prefix = kind.upper()
    blockers: list[dict[str, str]] = []
    hashes: dict[str, str] = {}
    missing = []
    if manifest_path is None or not manifest_path.exists():
        missing.append(f"{kind}_derivation_manifest")
    if builder_path is None or not builder_path.exists():
        missing.append(f"{kind}_builder")
    missing.extend(name for name, path in input_paths.items() if path is None or not path.exists())
    if missing:
        _add_blocker(
            blockers,
            f"{prefix}_DERIVATION_INPUT_MISSING",
            f"Missing derivation-contract files: {sorted(missing)}",
        )
        return None, hashes, blockers
    assert manifest_path is not None
    assert builder_path is not None
    resolved_inputs = {name: path for name, path in input_paths.items() if path is not None}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        _add_blocker(blockers, f"{prefix}_DERIVATION_MANIFEST_INVALID", str(error))
        return None, hashes, blockers
    if not isinstance(manifest, dict):
        _add_blocker(
            blockers,
            f"{prefix}_DERIVATION_MANIFEST_INVALID",
            "Derivation manifest root must be a JSON object",
        )
        return None, hashes, blockers

    manifest_hash = file_sha256(manifest_path)
    builder_hash = file_sha256(builder_path)
    edge_hash = file_sha256(edge_path)
    input_hashes = {name: file_sha256(path) for name, path in resolved_inputs.items()}
    hashes[f"{kind}_derivation_manifest"] = manifest_hash
    hashes[f"{kind}_builder_code"] = builder_hash
    hashes.update(
        {f"{kind}_derivation_input_{name}": digest for name, digest in input_hashes.items()}
    )

    def fail(detail: str) -> None:
        _add_blocker(blockers, f"{prefix}_DERIVATION_MANIFEST_MISMATCH", detail)

    if manifest.get("schema_version") != "1.0":
        fail("schema_version must be '1.0'")
    source_name = str(manifest.get("source_name", "")).strip()
    release = str(manifest.get("source_release", "")).strip()
    source_url = str(manifest.get("source_url", "")).strip()
    if not source_name:
        fail("source_name must be non-empty")
    if release != expected_release:
        fail(f"source_release={release!r} does not match {expected_release!r}")
    if not re.fullmatch(r"https?://\S+", source_url):
        fail("source_url must be an explicit HTTP(S) URL")
    if manifest.get("edge_file_sha256") != edge_hash:
        fail("edge_file_sha256 does not match the supplied edge list")

    builder = manifest.get("builder")
    if not isinstance(builder, dict):
        fail("builder must be an object")
        builder = {}
    if not str(builder.get("name", "")).strip():
        fail("builder.name must be non-empty")
    if not str(builder.get("version", "")).strip():
        fail("builder.version must be non-empty")
    if builder.get("code_sha256") != builder_hash:
        fail("builder.code_sha256 does not match the supplied builder file")

    declared_inputs = manifest.get("input_files")
    if not isinstance(declared_inputs, dict):
        fail("input_files must be an object of logical name to SHA-256")
        declared_inputs = {}
    for name, observed_hash in input_hashes.items():
        if declared_inputs.get(name) != observed_hash:
            fail(f"input_files.{name} does not match the supplied raw input")

    mapping = manifest.get("identifier_mapping")
    if not isinstance(mapping, dict):
        fail("identifier_mapping must be an object")
        mapping = {}
    if not str(mapping.get("rule", "")).strip():
        fail("identifier_mapping.rule must be non-empty")
    try:
        mapping_coverage = float(mapping.get("coverage"))
    except (TypeError, ValueError):
        mapping_coverage = float("nan")
    if not np.isfinite(mapping_coverage) or not 0 <= mapping_coverage <= 1:
        fail("identifier_mapping.coverage must be a finite fraction in [0, 1]")

    policy = manifest.get("policy")
    if not isinstance(policy, dict):
        fail("policy must be an object")
        policy = {}
    if kind == "string":
        expected_policy = {
            "species_taxon": 9606,
            "score_field": "combined_score",
            "threshold_operator": ">",
            "threshold_value": 400,
        }
        for field_name, expected in expected_policy.items():
            if policy.get(field_name) != expected:
                fail(f"policy.{field_name} must equal {expected!r}")
    elif kind == "go":
        if policy.get("namespace") != "biological_process":
            fail("policy.namespace must equal 'biological_process'")
        if policy.get("edge_rule") != "shared_annotation":
            fail("policy.edge_rule must equal 'shared_annotation'")
        for field_name in (
            "qualifier_policy",
            "evidence_code_policy",
            "ancestor_propagation_policy",
            "depth_policy",
        ):
            if not str(policy.get(field_name, "")).strip():
                fail(f"policy.{field_name} must be explicitly declared")
    else:
        raise RevisionProtocolError(f"Unknown derivation-manifest kind {kind!r}")

    row = {
        "source": f"{kind}_derivation_manifest",
        "release": release,
        "source_url": source_url,
        "source_id": manifest_path.name,
        "sha256": manifest_hash,
        "builder_name": builder.get("name"),
        "builder_version": builder.get("version"),
        "builder_code_sha256": builder_hash,
        "edge_file_sha256": edge_hash,
        "identifier_mapping_rule": mapping.get("rule"),
        "identifier_mapping_coverage": mapping_coverage,
        "derivation_policy": json.dumps(policy, sort_keys=True),
        "status": "VERIFIED_DERIVATION_CONTRACT" if not blockers else "INVALID_DERIVATION_CONTRACT",
    }
    return row, hashes, blockers


def _graph_provenance_preflight(
    dataset: BenchmarkDataset,
    graph_paths: Mapping[str, Path],
    requested_arms: Sequence[str],
    protocol: Mapping[str, Any],
    fixture_mode: bool,
) -> tuple[pd.DataFrame, dict[str, str], list[dict[str, str]]]:
    requires_curated = any(
        arm in {"combined", "string_go", "string_ppi", "gene_ontology"}
        or arm.startswith("string_go_rewire_")
        for arm in requested_arms
    )
    if not requires_curated:
        return pd.DataFrame(), {}, []
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    blockers: list[dict[str, str]] = []
    if fixture_mode:
        for source in ("string_ppi", "gene_ontology"):
            rows.append(
                {
                    "source": source,
                    "release": "SYNTHETIC_NPZ_FIXTURE",
                    "source_id": dataset.source_path.name,
                    "sha256": dataset.source_hash,
                    "status": "TEST_FIXTURE_ONLY_NOT_SUBMISSION_PROVENANCE",
                }
            )
        return pd.DataFrame(rows), hashes, blockers

    string_path = graph_paths.get("string_ppi")
    if string_path is None or not string_path.exists():
        _add_blocker(
            blockers,
            "STRING_V12_EDGE_FILE_MISSING",
            "Production preflight requires a local STRING v12.0 human edge file",
        )
    else:
        digest = file_sha256(string_path)
        hashes["string_v12_human_edges"] = digest
        separator = "\t" if string_path.suffix.casefold() == ".tsv" else ","
        try:
            string_frame = pd.read_csv(string_path, sep=separator)
        except (OSError, pd.errors.ParserError, UnicodeError) as error:
            _add_blocker(blockers, "STRING_EDGE_FILE_INVALID", str(error))
            string_frame = pd.DataFrame()
        if "combined_score" not in string_frame:
            _add_blocker(
                blockers,
                "STRING_SCORE_COLUMN_MISSING",
                "Production STRING edge input must retain combined_score for threshold audit",
            )
            minimum_score = np.nan
        else:
            scores = pd.to_numeric(string_frame["combined_score"], errors="coerce")
            if scores.isna().any() or not (scores > 400).all():
                _add_blocker(
                    blockers,
                    "STRING_EXCLUSIVE_SCORE_THRESHOLD_FAILED",
                    "Every retained STRING edge must have combined_score > 400",
                )
            minimum_score = float(scores.min())
        rows.append(
            {
                "source": "string_ppi",
                "release": protocol["graph_composition"]["string_ppi"]["source"],
                "species_taxon": 9606,
                "score_rule": "exclusive_gt_400",
                "minimum_retained_combined_score": minimum_score,
                "source_id": string_path.name,
                "sha256": digest,
                "status": "VERIFIED_LOCAL_INPUT",
            }
        )
        string_manifest_row, string_manifest_hashes, string_manifest_blockers = (
            _validate_derivation_manifest(
                kind="string",
                manifest_path=graph_paths.get("string_derivation_manifest"),
                edge_path=string_path,
                builder_path=graph_paths.get("string_builder"),
                input_paths={
                    "string_raw": graph_paths.get("string_raw"),
                    "identifier_map": graph_paths.get("string_identifier_map"),
                },
                expected_release="STRING_v12.0",
            )
        )
        hashes.update(string_manifest_hashes)
        blockers.extend(string_manifest_blockers)
        if string_manifest_row is not None:
            rows.append(string_manifest_row)

    required_go = (
        "gene_ontology",
        "go_gaf",
        "go_obo",
        "go_release_metadata",
        "go_builder",
        "go_derivation_manifest",
    )
    missing_go = [
        name for name in required_go if name not in graph_paths or not graph_paths[name].exists()
    ]
    if missing_go:
        _add_blocker(
            blockers,
            "GO_PROVENANCE_INPUT_MISSING",
            f"Required local GO inputs are missing: {missing_go}",
        )
        return pd.DataFrame(rows), hashes, blockers
    go_hashes = {name: file_sha256(graph_paths[name]) for name in required_go}
    hashes.update({f"go_{name}_sha256": digest for name, digest in go_hashes.items()})
    try:
        metadata = json.loads(graph_paths["go_release_metadata"].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        _add_blocker(blockers, "GO_RELEASE_METADATA_INVALID", str(error))
        return pd.DataFrame(rows), hashes, blockers
    if not isinstance(metadata, dict):
        _add_blocker(
            blockers,
            "GO_RELEASE_METADATA_INVALID",
            "GO release metadata root must be a JSON object",
        )
        return pd.DataFrame(rows), hashes, blockers
    release = str(metadata.get("release", "")).strip()
    if not release:
        _add_blocker(
            blockers,
            "GO_RELEASE_UNVERSIONED",
            "go_release_metadata JSON must contain a non-empty release identifier",
        )
    if metadata.get("gaf_sha256") != go_hashes["go_gaf"]:
        _add_blocker(blockers, "GO_GAF_CHECKSUM_MISMATCH", "GAF checksum does not match metadata")
    if metadata.get("obo_sha256") != go_hashes["go_obo"]:
        _add_blocker(blockers, "GO_OBO_CHECKSUM_MISMATCH", "OBO checksum does not match metadata")
    go_manifest_row, go_manifest_hashes, go_manifest_blockers = _validate_derivation_manifest(
        kind="go",
        manifest_path=graph_paths.get("go_derivation_manifest"),
        edge_path=graph_paths["gene_ontology"],
        builder_path=graph_paths.get("go_builder"),
        input_paths={"go_gaf": graph_paths.get("go_gaf"), "go_obo": graph_paths.get("go_obo")},
        expected_release=release,
    )
    hashes.update(go_manifest_hashes)
    blockers.extend(go_manifest_blockers)
    if go_manifest_row is not None:
        rows.append(go_manifest_row)
    rows.extend(
        [
            {
                "source": "gene_ontology_edges",
                "release": release,
                "source_id": graph_paths["gene_ontology"].name,
                "sha256": go_hashes["gene_ontology"],
                "status": "VERIFIED_DERIVED_EDGE_INPUT",
            },
            {
                "source": "go_gaf",
                "release": release,
                "source_id": graph_paths["go_gaf"].name,
                "sha256": go_hashes["go_gaf"],
                "status": "VERIFIED_LOCAL_INPUT",
            },
            {
                "source": "go_obo",
                "release": release,
                "source_id": graph_paths["go_obo"].name,
                "sha256": go_hashes["go_obo"],
                "status": "VERIFIED_LOCAL_INPUT",
            },
        ]
    )
    return pd.DataFrame(rows), hashes, blockers


def _control_coexpression_edges(
    transformed_controls: np.ndarray | sparse.spmatrix, threshold: float
) -> np.ndarray:
    if not 0 <= threshold <= 1:
        raise RevisionProtocolError("Coexpression threshold must be in [0, 1]")
    if transformed_controls.shape[0] < 2:
        raise RevisionProtocolError("At least two controls are required for coexpression")
    if sparse.issparse(transformed_controls):
        matrix = transformed_controls.tocsr().astype(np.float64)
        n_rows = matrix.shape[0]
        means = np.asarray(matrix.mean(axis=0)).reshape(-1)
        cross_products = (matrix.T @ matrix).toarray()
        covariance = (cross_products - n_rows * np.outer(means, means)) / (n_rows - 1)
        standard_deviations = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        denominator = np.outer(standard_deviations, standard_deviations)
        correlation = np.divide(
            covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=denominator > 0,
        )
    else:
        correlation = np.corrcoef(transformed_controls, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 0.0)
    edges: set[tuple[int, int]] = set()
    for left in range(correlation.shape[0]):
        candidates = np.flatnonzero(np.abs(correlation[left]) > threshold)
        for right in candidates:
            if left != right:
                edges.add(tuple(sorted((left, int(right)))))
    return np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2)


def _default_arms(protocol: Mapping[str, Any], settings: RunnerSettings) -> list[str]:
    block = protocol["analysis_blocks"][settings.analysis_block]
    if settings.analysis_block in {
        "topology_primary",
        "mixed_support_sensitivity",
        "conditional_nonoverlap",
    }:
        if settings.hvg not in block["hvg_scales"]:
            raise RevisionProtocolError(
                f"analysis_block={settings.analysis_block} does not permit hvg={settings.hvg}"
            )
        return list(block["default_arms"])
    by_hvg = block["default_arms_by_hvg"]
    arms = by_hvg.get(settings.hvg, by_hvg.get(str(settings.hvg)))
    if arms is None:
        raise RevisionProtocolError(f"scale_extension does not permit hvg={settings.hvg}")
    return list(arms)


def planned_mandatory_design_matrix(protocol: Mapping[str, Any]) -> pd.DataFrame:
    """Return the authoritative 10,800-fit primary design matrix."""

    rows: list[dict[str, Any]] = []
    conditions = int(protocol["planned_fit_counts"]["conditions_per_panel"])
    seeds = int(protocol["planned_fit_counts"]["seeds"])
    fits_per_cell = conditions * seeds
    datasets = sorted(protocol["datasets"])
    primary_arms = list(protocol["analysis_blocks"]["topology_primary"]["default_arms"])
    mixed_arms = list(protocol["analysis_blocks"]["mixed_support_sensitivity"]["default_arms"])
    scale_by_hvg = protocol["analysis_blocks"]["scale_extension"]["default_arms_by_hvg"]
    for dataset in datasets:
        for arm in primary_arms:
            rows.append(
                {
                    "dataset": dataset,
                    "analysis_block": "topology_primary",
                    "hvg": 200,
                    "arm": arm,
                    "dense_200_reused": False,
                    "model_fits": fits_per_cell,
                }
            )
        for arm in mixed_arms:
            rows.append(
                {
                    "dataset": dataset,
                    "analysis_block": "mixed_support_sensitivity",
                    "hvg": 200,
                    "arm": arm,
                    "dense_200_reused": False,
                    "model_fits": fits_per_cell,
                }
            )
        for hvg_key, arms in scale_by_hvg.items():
            hvg = int(hvg_key)
            for arm in arms:
                rows.append(
                    {
                        "dataset": dataset,
                        "analysis_block": "scale_extension",
                        "hvg": hvg,
                        "arm": arm,
                        "dense_200_reused": hvg == 200,
                        "model_fits": fits_per_cell,
                    }
                )
    matrix = pd.DataFrame(rows)
    if int(matrix["model_fits"].sum()) != 10_800:
        raise RevisionProtocolError("Authoritative mandatory design does not sum to 10,800 fits")
    return matrix


def planned_conditional_nonoverlap_matrix(protocol: Mapping[str, Any]) -> pd.DataFrame:
    """Return the prespecified 1,200-fit conditional sensitivity matrix."""

    conditions = int(protocol["planned_fit_counts"]["conditions_per_panel"])
    seeds = int(protocol["planned_fit_counts"]["seeds"])
    rows = [
        {
            "dataset": dataset,
            "analysis_block": "conditional_nonoverlap",
            "hvg": 200,
            "panel": "sensitivity",
            "arm": arm,
            "model_fits": conditions * seeds,
        }
        for dataset in sorted(protocol["datasets"])
        for arm in protocol["analysis_blocks"]["conditional_nonoverlap"]["default_arms"]
    ]
    matrix = pd.DataFrame(rows)
    if int(matrix["model_fits"].sum()) != 1_200:
        raise RevisionProtocolError("Conditional non-overlap design does not sum to 1,200 fits")
    return matrix


def _profile_tensor(profile: np.ndarray, device: Any) -> Any:
    import torch

    return torch.as_tensor(profile, dtype=torch.float32, device=device)


def _indicator_tensor(encoding: TargetEncoding, n_genes: int, device: Any) -> Any:
    import torch

    return torch.as_tensor(encoding.mask(n_genes), dtype=torch.bool, device=device)


def _set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        return


def _configure_deterministic_backend(torch_module: Any) -> dict[str, Any]:
    torch_module.use_deterministic_algorithms(True)
    cudnn_available = hasattr(torch_module.backends, "cudnn")
    if cudnn_available:
        torch_module.backends.cudnn.benchmark = False
        torch_module.backends.cudnn.deterministic = True
    return {
        "torch_deterministic_algorithms_enabled": bool(
            torch_module.are_deterministic_algorithms_enabled()
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_available": cudnn_available,
        "cudnn_benchmark": bool(torch_module.backends.cudnn.benchmark) if cudnn_available else None,
        "cudnn_deterministic": (
            bool(torch_module.backends.cudnn.deterministic) if cudnn_available else None
        ),
    }


def _code_sha256(package_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(package_dir.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def code_tree_sha256() -> str:
    """Hash the full Python package used by training and postprocessing."""

    return _code_sha256(Path(__file__).resolve().parent)


def _build_git_provenance(*, fixture_mode: bool) -> dict[str, Any]:
    """Record the base commit and a path-redacted worktree-state digest."""

    package_dir = Path(__file__).resolve().parent
    try:
        root_result = subprocess.run(
            ["git", "-C", str(package_dir), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        repository_root = Path(root_result.stdout.strip()).resolve()
        commit_result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        status_result = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as error:
        if not fixture_mode:
            raise RevisionProtocolError("[GIT_PROVENANCE_UNAVAILABLE]") from error
        commit = "0" * 40
        status_lines: list[str] = []
        source = "SYNTHETIC_FIXTURE_NO_GIT"
    else:
        commit = commit_result.stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise RevisionProtocolError("[GIT_COMMIT_INVALID]")
        status_lines = sorted(
            line.rstrip("\r") for line in status_result.stdout.splitlines() if line.strip()
        )
        source = "git_rev_parse_HEAD_and_porcelain_v1"
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "git_commit": commit,
        "git_commit_source": source,
        "git_worktree_status": "DIRTY" if status_lines else "CLEAN",
        "git_status_entry_count": len(status_lines),
        "git_status_hash": canonical_sha256(status_lines),
        "path_disclosure_policy": "status_paths_hashed_not_embedded",
        "code_tree_sha256": code_tree_sha256(),
    }
    payload["provenance_hash"] = canonical_sha256(payload)
    return payload


def _build_environment_manifest(
    environment_lock_path: Path | None,
    *,
    fixture_mode: bool,
    require_h5ad_runtime: bool = False,
) -> dict[str, Any]:
    """Freeze the complete active distribution set and compiled-runtime metadata."""

    versions = _installed_distribution_versions()
    direct_dependencies = _direct_runtime_dependencies()
    missing_direct = sorted(set(direct_dependencies) - set(versions))
    if missing_direct and require_h5ad_runtime and not fixture_mode:
        raise RevisionProtocolError(
            f"[ENVIRONMENT_REQUIRED_DIRECT_DEPENDENCY_MISSING] {missing_direct}"
        )
    cuda_runtime = "NOT_AVAILABLE"
    cudnn_version: str | int = "NOT_AVAILABLE"
    try:
        import torch

        cuda_runtime = str(torch.version.cuda or "NOT_AVAILABLE")
        cudnn_version = torch.backends.cudnn.version() or "NOT_AVAILABLE"
    except ImportError:
        pass
    driver_version = _nvidia_driver_version()
    pyg_extensions = {name: versions.get(name, "NOT_INSTALLED") for name in PYG_EXTENSION_PACKAGES}
    if environment_lock_path is None:
        if not fixture_mode:
            raise RevisionProtocolError("[ENVIRONMENT_LOCK_MISSING]")
        lock_source_id = "NOT_APPLICABLE_SYNTHETIC_FIXTURE"
        lock_hash = canonical_sha256({"status": lock_source_id})
        lock_validation_status = "NOT_APPLICABLE_SYNTHETIC_FIXTURE"
    else:
        if not environment_lock_path.is_file():
            raise RevisionProtocolError("[ENVIRONMENT_LOCK_FILE_MISSING]")
        lock_source_id = environment_lock_path.name
        lock_hash = file_sha256(environment_lock_path)
        _validate_environment_lock(
            environment_lock_path,
            versions,
            python_version=platform.python_version(),
            cuda_runtime=str(cuda_runtime),
            cudnn_version=str(cudnn_version),
            nvidia_driver_version=driver_version,
            direct_dependencies=direct_dependencies,
            pyg_extensions=pyg_extensions,
            require_h5ad_runtime=require_h5ad_runtime,
        )
        lock_validation_status = "EXACT_COMPLETE_ACTIVE_DISTRIBUTION_SET_MATCH"
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "operating_system": platform.system(),
        "operating_system_release": platform.release(),
        "machine_architecture": platform.machine(),
        "package_versions": versions,
        "complete_distribution_count": len(versions),
        "complete_distribution_set_sha256": canonical_sha256(versions),
        "direct_runtime_dependencies": direct_dependencies,
        "missing_direct_runtime_dependencies": missing_direct,
        "h5ad_runtime_required": require_h5ad_runtime,
        "cuda_runtime_version": cuda_runtime,
        "cudnn_version": cudnn_version,
        "nvidia_driver_version": driver_version,
        "pytorch_version": versions.get("torch", "NOT_INSTALLED"),
        "pytorch_geometric_version": versions.get("torch-geometric", "NOT_INSTALLED"),
        "pyg_compiled_extension_versions": pyg_extensions,
        "environment_lock_source_id": lock_source_id,
        "environment_lock_sha256": lock_hash,
        "environment_lock_validation_status": lock_validation_status,
        "container_digest": "NOT_APPLICABLE_BARE_METAL_EXECUTION",
        "fixture_status": "TEST_FIXTURE_ONLY" if fixture_mode else "PRODUCTION",
    }
    payload["manifest_hash"] = canonical_sha256(payload)
    return payload


def _validate_environment_lock(
    path: Path,
    installed_versions: Mapping[str, str],
    *,
    python_version: str,
    cuda_runtime: str,
    cudnn_version: str,
    nvidia_driver_version: str,
    direct_dependencies: Sequence[str],
    pyg_extensions: Mapping[str, str],
    require_h5ad_runtime: bool,
) -> None:
    """Require a complete exact-version freeze of the active interpreter environment."""

    text = path.read_text(encoding="utf-8")
    if re.search(r"REPLACE|TODO|TBD|UNKNOWN|PLACEHOLDER", text, re.IGNORECASE):
        raise RevisionProtocolError("[ENVIRONMENT_LOCK_PLACEHOLDER_PRESENT]")
    headers: dict[str, str] = {}
    locked: dict[str, str] = {}
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            match = re.fullmatch(r"#\s*([a-z0-9_-]+):\s*(\S+)", line, re.IGNORECASE)
            if match is not None:
                name = match.group(1).casefold()
                if name in headers:
                    raise RevisionProtocolError("[ENVIRONMENT_LOCK_DUPLICATE_HEADER]")
                headers[name] = match.group(2)
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if match is None:
            raise RevisionProtocolError(f"[ENVIRONMENT_LOCK_NOT_EXACT] line={line_number}")
        name, version = match.groups()
        normalized = re.sub(r"[-_.]+", "-", name).casefold()
        if normalized in locked:
            raise RevisionProtocolError("[ENVIRONMENT_LOCK_DUPLICATE_PACKAGE]")
        locked[normalized] = version
    expected_headers = _environment_lock_headers(
        installed_versions,
        python_version=python_version,
        cuda_runtime=cuda_runtime,
        cudnn_version=cudnn_version,
        nvidia_driver_version=nvidia_driver_version,
        direct_dependencies=direct_dependencies,
        pyg_extensions=pyg_extensions,
    )
    if headers != expected_headers:
        raise RevisionProtocolError("[ENVIRONMENT_LOCK_RUNTIME_HEADER_MISMATCH]")
    required = {
        re.sub(r"[-_.]+", "-", name).casefold(): version
        for name, version in installed_versions.items()
        if version != "NOT_INSTALLED"
    }
    if set(locked) != set(required):
        raise RevisionProtocolError(
            "[ENVIRONMENT_LOCK_PACKAGE_SET_MISMATCH] "
            f"missing={sorted(set(required)-set(locked))};extra={sorted(set(locked)-set(required))}"
        )
    mismatches = {
        name: (required[name], locked[name]) for name in required if locked[name] != required[name]
    }
    if mismatches:
        raise RevisionProtocolError(
            f"[ENVIRONMENT_LOCK_VERSION_MISMATCH] {json.dumps(mismatches, sort_keys=True)}"
        )
    if require_h5ad_runtime and "h5py" not in locked:
        raise RevisionProtocolError("[ENVIRONMENT_LOCK_H5PY_MISSING]")
    missing_direct = sorted(set(direct_dependencies) - set(locked))
    if require_h5ad_runtime and missing_direct:
        raise RevisionProtocolError(
            f"[ENVIRONMENT_LOCK_DIRECT_DEPENDENCY_MISSING] {missing_direct}"
        )


def _environment_lock_headers(
    installed_versions: Mapping[str, str],
    *,
    python_version: str,
    cuda_runtime: str,
    cudnn_version: str,
    nvidia_driver_version: str,
    direct_dependencies: Sequence[str],
    pyg_extensions: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Derive every exact environment-lock header from inspected runtime state."""

    normalized_versions = dict(sorted(installed_versions.items()))
    compiled_extensions = dict(
        sorted(
            (
                pyg_extensions
                or {
                    name: normalized_versions.get(name, "NOT_INSTALLED")
                    for name in PYG_EXTENSION_PACKAGES
                }
            ).items()
        )
    )
    return {
        "lock-format": ENVIRONMENT_LOCK_FORMAT,
        "python-version": str(python_version),
        "cuda-runtime-version": str(cuda_runtime),
        "cudnn-version": str(cudnn_version),
        "nvidia-driver-version": str(nvidia_driver_version),
        "pytorch-version": normalized_versions.get("torch", "NOT_INSTALLED"),
        "pytorch-geometric-version": normalized_versions.get("torch-geometric", "NOT_INSTALLED"),
        "complete-distribution-set-sha256": canonical_sha256(normalized_versions),
        "direct-runtime-dependencies-sha256": canonical_sha256(list(direct_dependencies)),
        "pyg-extension-versions-sha256": canonical_sha256(compiled_extensions),
    }


def _installed_distribution_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            raise RevisionProtocolError("[ENVIRONMENT_DISTRIBUTION_NAME_MISSING]")
        name = re.sub(r"[-_.]+", "-", str(raw_name)).casefold()
        version = str(distribution.version)
        prior = versions.get(name)
        if prior is not None and prior != version:
            raise RevisionProtocolError(
                f"[ENVIRONMENT_DUPLICATE_DISTRIBUTION_VERSION_CONFLICT] {name}"
            )
        versions[name] = version
    return dict(sorted(versions.items()))


def _direct_runtime_dependencies() -> list[str]:
    try:
        import tomllib
    except ImportError as error:
        raise RevisionProtocolError("[PYPROJECT_TOML_READER_UNAVAILABLE]") from error
    path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        project = tomllib.loads(path.read_text(encoding="utf-8"))["project"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as error:
        raise RevisionProtocolError("[PYPROJECT_DEPENDENCY_CONTRACT_INVALID]") from error
    specifications = list(project.get("dependencies", ()))
    optional = project.get("optional-dependencies", {})
    for group in ("model", "h5ad"):
        specifications.extend(optional.get(group, ()))
    names = []
    for specification in specifications:
        match = re.match(r"[A-Za-z0-9_.-]+", str(specification))
        if match is None:
            raise RevisionProtocolError("[PYPROJECT_DEPENDENCY_CONTRACT_INVALID]")
        names.append(re.sub(r"[-_.]+", "-", match.group(0)).casefold())
    output = sorted(set(names))
    if "h5py" not in output:
        raise RevisionProtocolError("[PYPROJECT_H5PY_DIRECT_DEPENDENCY_MISSING]")
    return output


def _nvidia_driver_version() -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "NOT_AVAILABLE"
    values = sorted({line.strip() for line in completed.stdout.splitlines() if line.strip()})
    return ",".join(values) if completed.returncode == 0 and values else "NOT_AVAILABLE"


def write_complete_environment_lock(
    path: Path, *, require_direct_dependencies: bool = True
) -> Path:
    """Export the complete active environment in the validated CBAC lock format."""

    versions = _installed_distribution_versions()
    direct_dependencies = _direct_runtime_dependencies()
    missing = sorted(set(direct_dependencies) - set(versions))
    if missing and require_direct_dependencies:
        raise RevisionProtocolError(f"[ENVIRONMENT_REQUIRED_DIRECT_DEPENDENCY_MISSING] {missing}")
    try:
        import torch

        cuda_runtime = str(torch.version.cuda or "NOT_AVAILABLE")
        cudnn_version = str(torch.backends.cudnn.version() or "NOT_AVAILABLE")
    except ImportError:
        cuda_runtime = "NOT_AVAILABLE"
        cudnn_version = "NOT_AVAILABLE"
    pyg_extensions = {name: versions.get(name, "NOT_INSTALLED") for name in PYG_EXTENSION_PACKAGES}
    headers = _environment_lock_headers(
        versions,
        python_version=platform.python_version(),
        cuda_runtime=cuda_runtime,
        cudnn_version=cudnn_version,
        nvidia_driver_version=_nvidia_driver_version(),
        direct_dependencies=direct_dependencies,
        pyg_extensions=pyg_extensions,
    )
    lines = [
        *(f"# {name}: {headers[name]}" for name in ENVIRONMENT_LOCK_HEADER_ORDER),
        *(f"{name}=={version}" for name, version in versions.items()),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)
    return path


def _add_blocker(blockers: list[dict[str, str]], reason_code: str, detail: str) -> None:
    record = {"reason_code": reason_code, "detail": detail}
    if record not in blockers:
        blockers.append(record)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unnamed"


def validate_dataset_passport_binding(
    dataset: BenchmarkDataset,
    passport: DatasetPassport,
    *,
    canonical_labels: Sequence[object],
    target_encodings: Mapping[str, TargetEncoding],
    control_label: str,
) -> None:
    """Cross-check every passport declaration against the loaded production dataset."""

    schema = passport.matrix_schema
    if dataset.expression.shape != (schema["n_cells"], schema["n_genes"]):
        raise PreflightBlockedError("[DATASET_PASSPORT_MATRIX_SHAPE_MISMATCH]")
    if schema["condition_column"] != passport.control_evidence.condition_column:
        raise PreflightBlockedError("[DATASET_PASSPORT_CONDITION_COLUMN_MISMATCH]")
    if canonical_sha256(list(dataset.gene_names)) != schema["gene_order_sha256"]:
        raise PreflightBlockedError("[DATASET_PASSPORT_GENE_ORDER_MISMATCH]")

    raw = np.asarray(dataset.condition_labels, dtype=object)
    canonical = np.asarray(canonical_labels, dtype=object)
    if raw.shape != canonical.shape:
        raise PreflightBlockedError("[DATASET_PASSPORT_CONDITION_ROW_ORDER_MISMATCH]")
    observed_raw_counts = {
        label: int(np.sum(raw == label)) for label in sorted(set(str(value) for value in raw))
    }
    if observed_raw_counts != dict(passport.condition_cell_counts):
        raise PreflightBlockedError("[DATASET_PASSPORT_RAW_CONDITION_COUNTS_MISMATCH]")

    observed_mapping: dict[str, str] = {}
    for raw_label in observed_raw_counts:
        mapped = {str(value) for value in canonical[raw == raw_label]}
        if len(mapped) != 1:
            raise PreflightBlockedError("[DATASET_PASSPORT_NONDETERMINISTIC_CONDITION_MAPPING]")
        observed_mapping[raw_label] = next(iter(mapped))
    if observed_mapping != dict(passport.condition_mapping):
        raise PreflightBlockedError("[DATASET_PASSPORT_CONDITION_MAPPING_MISMATCH]")

    observed_canonical_counts = {
        label: int(np.sum(canonical == label))
        for label in sorted(set(str(value) for value in canonical))
    }
    if observed_canonical_counts != dict(passport.canonical_condition_cell_counts):
        raise PreflightBlockedError("[DATASET_PASSPORT_CANONICAL_COUNTS_MISMATCH]")

    observed_targets: dict[str, tuple[str, ...]] = {}
    observed_excluded: dict[str, str] = {}
    for raw_label, canonical_label in observed_mapping.items():
        if canonical_label == control_label:
            continue
        encoding = target_encodings.get(canonical_label)
        if encoding is None:
            raise PreflightBlockedError("[DATASET_PASSPORT_TARGET_ENCODING_MISSING]")
        if encoding.success:
            prior = observed_targets.setdefault(canonical_label, encoding.canonical_targets)
            if prior != encoding.canonical_targets:
                raise PreflightBlockedError("[DATASET_PASSPORT_TARGET_MAPPING_AMBIGUOUS]")
        else:
            observed_excluded[raw_label] = encoding.reason_code or "TARGET_MAPPING_FAILED"
    if observed_targets != dict(passport.target_mapping):
        raise PreflightBlockedError("[DATASET_PASSPORT_TARGET_MAPPING_MISMATCH]")
    if observed_excluded != dict(passport.attrition["excluded_raw_conditions"]):
        raise PreflightBlockedError("[DATASET_PASSPORT_ATTRITION_REASON_MISMATCH]")

    control_cells = sum(
        observed_raw_counts[label] for label in passport.control_evidence.raw_labels
    )
    excluded_cells = sum(observed_raw_counts[label] for label in observed_excluded)
    retained_cells = int(schema["n_cells"]) - control_cells - excluded_cells
    observed_attrition = {
        "raw_condition_count": len(observed_mapping),
        "canonical_condition_count": len(observed_canonical_counts),
        "excluded_raw_condition_count": len(observed_excluded),
        "retained_target_mapped_condition_count": len(observed_targets),
        "raw_cell_count": int(schema["n_cells"]),
        "control_cell_count": control_cells,
        "excluded_cell_count": excluded_cells,
        "retained_target_mapped_cell_count": retained_cells,
    }
    for field, observed in observed_attrition.items():
        if passport.attrition[field] != observed:
            raise PreflightBlockedError(f"[DATASET_PASSPORT_ATTRITION_MISMATCH] {field}")


def _canonicalize_unordered_target_conditions(
    labels: Sequence[object],
    control_label: str,
    encoder: TargetEncoder,
    *,
    pool_target_equivalent_labels: bool = True,
    excluded_endpoint_conditions: Mapping[str, str] | None = None,
) -> tuple[np.ndarray, dict[str, TargetEncoding], dict[str, tuple[str, ...]]]:
    """Collapse target-equivalent labels onto one stable unordered target-set key."""

    input_labels = np.asarray(labels, dtype=object)
    output = input_labels.copy()
    encodings: dict[str, TargetEncoding] = {}
    raw_members: dict[str, set[str]] = {}
    excluded = dict(excluded_endpoint_conditions or {})
    for raw_condition in sorted(set(str(value) for value in input_labels)):
        if raw_condition == control_label:
            continue
        if raw_condition in excluded:
            raw_encoding = TargetEncoding(
                condition=raw_condition,
                success=False,
                canonical_targets=(),
                target_indices=(),
                reason_code=str(excluded[raw_condition]),
                detail="Endpoint excluded by the frozen dataset-specific condition policy",
            )
        else:
            raw_encoding = encoder.encode(raw_condition)
        if raw_encoding.success:
            canonical_target_set = "+".join(
                sorted(raw_encoding.canonical_targets, key=lambda value: value.casefold())
            )
            canonical_condition = (
                canonical_target_set if pool_target_equivalent_labels else raw_condition
            )
            canonical_encoding = TargetEncoding(
                condition=canonical_condition,
                success=True,
                canonical_targets=raw_encoding.canonical_targets,
                target_indices=raw_encoding.target_indices,
                reason_code=None,
                detail=None,
            )
        else:
            canonical_condition = raw_condition
            canonical_encoding = raw_encoding
        prior = encodings.get(canonical_condition)
        if prior is not None and (
            prior.success != canonical_encoding.success
            or set(prior.canonical_targets) != set(canonical_encoding.canonical_targets)
        ):
            raise RevisionProtocolError(
                f"Canonical condition collision for {canonical_condition!r} has conflicting targets"
            )
        encodings[canonical_condition] = canonical_encoding
        raw_members.setdefault(canonical_condition, set()).add(raw_condition)
        output[input_labels == raw_condition] = canonical_condition
    return (
        output,
        encodings,
        {condition: tuple(sorted(members)) for condition, members in raw_members.items()},
    )


def _parse_graph_arguments(values: Sequence[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise RevisionProtocolError("--graph values must use NAME=PATH")
        name, path = value.split("=", 1)
        if not name or name in output:
            raise RevisionProtocolError(f"Invalid or duplicate graph name {name!r}")
        output[name] = Path(path)
    return output


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the dry-run-first command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-passport", type=Path)
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument("--global-trigger-manifest", type=Path)
    parser.add_argument("--precision-registry", type=Path)
    parser.add_argument("--precision-archive-condition-table", type=Path)
    parser.add_argument("--precision-archive-condition-table-sha256")
    parser.add_argument("--precision-canonical-target-map", type=Path)
    parser.add_argument("--precision-canonical-target-map-sha256")
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hvg", type=int, required=True)
    parser.add_argument(
        "--analysis-block",
        choices=(
            "topology_primary",
            "mixed_support_sensitivity",
            "scale_extension",
            "conditional_nonoverlap",
        ),
        default="topology_primary",
    )
    parser.add_argument("--panel", choices=("primary", "sensitivity"), default="primary")
    parser.add_argument("--panel-size", type=int, default=50)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--arm", action="append", dest="arms")
    parser.add_argument("--graph", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--fixture-mode",
        action="store_true",
        help="Permit embedded graph provenance only for a synthetic NPZ test fixture",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Run training after preflight; absent by default and never implied",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicitly request the default preflight-only behavior",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Run preflight by default and train only when ``--execute`` is present."""

    parsed = build_argument_parser().parse_args(arguments)
    settings = RunnerSettings(
        protocol_path=parsed.protocol,
        dataset_name=parsed.dataset,
        data_path=parsed.data,
        output_dir=parsed.output,
        hvg=parsed.hvg,
        analysis_block=parsed.analysis_block,
        panel_name=parsed.panel,
        panel_size=parsed.panel_size,
        seeds=tuple(parsed.seeds or (42, 43, 44)),
        arms=tuple(parsed.arms or ()),
        graph_paths=_parse_graph_arguments(parsed.graph),
        dataset_passport_path=parsed.dataset_passport,
        environment_lock_path=parsed.environment_lock,
        global_trigger_manifest_path=parsed.global_trigger_manifest,
        precision_registry_path=parsed.precision_registry,
        precision_condition_table_path=parsed.precision_archive_condition_table,
        precision_condition_table_sha256=parsed.precision_archive_condition_table_sha256,
        precision_target_map_path=parsed.precision_canonical_target_map,
        precision_target_map_sha256=parsed.precision_canonical_target_map_sha256,
        baseline_root_path=parsed.baseline_root,
        epochs_override=parsed.epochs,
        device=parsed.device,
        fixture_mode=parsed.fixture_mode,
    )
    bundle = preflight_matched_benchmark(settings)
    print(json.dumps(bundle.summary(), indent=2, sort_keys=True))
    if parsed.execute:
        if not bundle.can_execute:
            print("Execution was not started because preflight reported blockers.")
            return 2
        artifacts = execute_matched_benchmark(bundle)
        print(f"Execution completed with {len(artifacts)} lossless fold artifacts.")
    else:
        print("Dry run complete. No model was initialized or trained; pass --execute explicitly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Lossless, hash-addressed fold artifacts for revision experiments."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .errors import ArtifactValidationError

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def replay_early_stopping(
    validation_loss: Sequence[float],
    *,
    minimum_delta: float,
    patience: int,
    epochs_requested: int,
) -> dict[str, Any]:
    """Replay the frozen strict-minimum early-stopping state machine."""

    if isinstance(patience, bool) or not isinstance(patience, (int, np.integer)):
        raise ArtifactValidationError("early-stopping patience must be an integer")
    if int(patience) <= 0:
        raise ArtifactValidationError("early-stopping patience must be positive")
    if isinstance(epochs_requested, bool) or not isinstance(epochs_requested, (int, np.integer)):
        raise ArtifactValidationError("epochs_requested must be an integer")
    if int(epochs_requested) <= 0:
        raise ArtifactValidationError("epochs_requested must be positive")
    if isinstance(minimum_delta, bool):
        raise ArtifactValidationError("early-stopping minimum_delta must be numeric")
    try:
        delta = float(minimum_delta)
    except (TypeError, ValueError) as error:
        raise ArtifactValidationError("early-stopping minimum_delta must be numeric") from error
    if not np.isfinite(delta) or delta < 0:
        raise ArtifactValidationError(
            "early-stopping minimum_delta must be finite and non-negative"
        )

    try:
        losses = tuple(float(value) for value in validation_loss)
    except (TypeError, ValueError) as error:
        raise ArtifactValidationError("validation_loss must be a numeric sequence") from error
    if not losses:
        raise ArtifactValidationError("validation_loss cannot be empty")
    if len(losses) > int(epochs_requested):
        raise ArtifactValidationError("validation_loss contains more epochs than epochs_requested")
    if not np.isfinite(np.asarray(losses, dtype=float)).all():
        raise ArtifactValidationError("validation_loss contains non-finite values")

    best_loss: float | None = None
    best_epoch: int | None = None
    epochs_without_improvement = 0
    trace: list[dict[str, Any]] = []
    patience_exhausted = False

    for epoch, current_loss in enumerate(losses):
        best_loss_before = best_loss
        threshold = None if best_loss is None else best_loss - delta
        qualified_improvement = best_loss is None or current_loss < threshold
        if qualified_improvement:
            best_loss = current_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        patience_exhausted = epochs_without_improvement >= int(patience)
        trace.append(
            {
                "epoch_zero_based": epoch,
                "validation_loss": current_loss,
                "best_loss_before_epoch": best_loss_before,
                "improvement_threshold_exclusive": threshold,
                "qualified_improvement": qualified_improvement,
                "best_epoch_zero_based_after_epoch": best_epoch,
                "best_loss_after_epoch": best_loss,
                "epochs_without_improvement_after_epoch": epochs_without_improvement,
                "stop_triggered_after_epoch": patience_exhausted,
            }
        )
        if patience_exhausted and epoch != len(losses) - 1:
            raise ArtifactValidationError(
                "validation_loss contains epochs after the patience stop condition"
            )

    if best_epoch is None or best_loss is None:
        raise ArtifactValidationError("early stopping did not select a finite best epoch")
    if patience_exhausted:
        stop_reason = "PATIENCE_EXHAUSTED"
    elif len(losses) == int(epochs_requested):
        stop_reason = "MAXIMUM_EPOCHS_REACHED"
    else:
        raise ArtifactValidationError(
            "validation history ended before either patience exhaustion or maximum epochs"
        )
    return {
        "schema_version": "1.0",
        "minimum_delta": delta,
        "patience": int(patience),
        "epochs_requested": int(epochs_requested),
        "epochs_observed": len(losses),
        "best_epoch_zero_based": best_epoch,
        "best_validation_loss": best_loss,
        "stopped_epoch_zero_based": len(losses) - 1,
        "stop_reason": stop_reason,
        "trace": trace,
    }


@dataclass(frozen=True)
class FoldIdentity:
    """Condition-keyed identity of one seed-specific prediction artifact."""

    run_id: str
    dataset: str
    hvg: int
    arm: str
    condition: str
    seed: int


@dataclass(frozen=True)
class RuntimeRecord:
    """Auditable phase timing, retry accounting, and execution environment."""

    started_at_utc: str
    finished_at_utc: str
    wall_seconds: float
    training_wall_seconds: float
    inference_wall_seconds: float
    checkpoint_io_wall_seconds: float
    other_overhead_wall_seconds: float
    useful_compute_seconds: float
    failed_attempt_wall_seconds: float
    retry_overhead_wall_seconds: float
    attempt_index: int
    retry_count: int
    execution_status: str
    device: str
    host: str
    accelerator_model: str
    accelerator_count: int
    accelerator_hours: float
    cpu_model: str
    logical_cpu_count: int
    host_ram_bytes: int
    peak_device_memory_bytes: int | None
    peak_memory_status: str
    device_hours: float
    timing_scope: str


@dataclass(frozen=True)
class FoldArtifact:
    """Complete outputs required to recompute metrics and audit one fold."""

    identity: FoldIdentity
    gene_names: tuple[str, ...]
    y_true: tuple[float, ...]
    y_pred: tuple[float, ...]
    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]
    learning_rate: tuple[float, ...]
    vector_space: str
    config: Mapping[str, Any]
    input_hashes: Mapping[str, str]
    graph_support_hash: str
    preprocessing_state_hash: str
    architecture_hash: str
    initialization_hash: str
    code_hash: str
    checkpoint_path: str
    checkpoint_hash: str
    runtime: RuntimeRecord

    def validate(self) -> None:
        """Reject incomplete, non-finite, or unauditable fold records."""

        identity_values = (
            self.identity.run_id,
            self.identity.dataset,
            self.identity.arm,
            self.identity.condition,
        )
        if any(not value for value in identity_values):
            raise ArtifactValidationError("Fold identity strings must be non-empty")
        if self.identity.hvg <= 0:
            raise ArtifactValidationError("hvg must be positive")
        if not self.gene_names:
            raise ArtifactValidationError("gene_names cannot be empty")
        if len(set(self.gene_names)) != len(self.gene_names):
            raise ArtifactValidationError("gene_names must be unique")
        if len(self.gene_names) != self.identity.hvg:
            raise ArtifactValidationError("gene_names length must equal the declared HVG scale")
        if len(self.y_true) != len(self.gene_names) or len(self.y_pred) != len(self.gene_names):
            raise ArtifactValidationError("gene_names, y_true, and y_pred must have equal lengths")
        for name, values in {
            "y_true": self.y_true,
            "y_pred": self.y_pred,
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "learning_rate": self.learning_rate,
        }.items():
            if not values:
                raise ArtifactValidationError(f"{name} cannot be empty")
            if not np.isfinite(np.asarray(values, dtype=float)).all():
                raise ArtifactValidationError(f"{name} contains non-finite values")
        if not (len(self.train_loss) == len(self.validation_loss) == len(self.learning_rate)):
            raise ArtifactValidationError(
                "train_loss, validation_loss, and learning_rate must have one value per epoch"
            )
        if not self.config:
            raise ArtifactValidationError("config cannot be empty")
        required_config = {
            "protocol_id",
            "panel",
            "analysis_block",
            "train_conditions",
            "validation_conditions",
            "held_out_condition",
            "held_out_canonical_targets",
            "canonical_target_set",
            "gene_panel_contract",
            "epochs_requested",
            "epochs_completed",
            "early_stopping_observation",
            "deterministic_backend",
            "git_commit",
            "git_worktree_status",
            "git_status_hash",
            "model",
            "training",
            "graph_mode",
            "graph_diagnostics",
            "swapped_edge_fraction",
            "topology_null_ensemble_gate_status",
            "topology_null_ensemble_audit_hash",
            "graph_manifest_hash",
            "global_trigger_manifest_hash",
            "spaces",
        }
        missing_config = sorted(required_config - set(self.config))
        if missing_config:
            raise ArtifactValidationError(f"config is missing fields: {missing_config}")
        if self.config["panel"] not in {"primary", "sensitivity"}:
            raise ArtifactValidationError("config.panel is invalid")
        if self.config["analysis_block"] not in {
            "topology_primary",
            "mixed_support_sensitivity",
            "scale_extension",
            "conditional_nonoverlap",
        }:
            raise ArtifactValidationError("config.analysis_block is invalid")
        if not re.fullmatch(r"[0-9a-f]{40}", str(self.config["git_commit"])):
            raise ArtifactValidationError("config.git_commit must be a 40-hex commit")
        if self.config["git_worktree_status"] not in {"CLEAN", "DIRTY"}:
            raise ArtifactValidationError("config.git_worktree_status is invalid")
        _validate_hash("config.git_status_hash", str(self.config["git_status_hash"]))
        if not isinstance(self.config["model"], Mapping) or not self.config["model"]:
            raise ArtifactValidationError("config.model must be a complete mapping")
        if canonical_sha256(self.config["model"]) != self.architecture_hash:
            raise ArtifactValidationError(
                "architecture_hash must equal canonical config.model hash"
            )
        if not isinstance(self.config["training"], Mapping) or not self.config["training"]:
            raise ArtifactValidationError("config.training must be a complete mapping")
        if not str(self.config["graph_mode"]):
            raise ArtifactValidationError("config.graph_mode must be non-empty")
        diagnostics = self.config["graph_diagnostics"]
        diagnostic_fields = {
            "arm",
            "mode",
            "n_nodes",
            "n_undirected_nonself_edges",
            "degree_sequence_sha256",
            "n_connected_components",
            "component_partition_sha256",
            "n_isolates",
            "support_sha256",
            "source_arm",
            "source_support_sha256",
            "source_edge_sha256",
            "source_n_nodes",
            "source_n_undirected_nonself_edges",
            "source_degree_sequence_sha256",
            "source_n_connected_components",
            "source_component_partition_sha256",
            "source_n_isolates",
            "swapped_edge_fraction",
            "rewire_seed",
            "cross_dataset_graph_index_pairing",
            "diagnostics_sha256",
        }
        if not isinstance(diagnostics, Mapping) or set(diagnostics) != diagnostic_fields:
            raise ArtifactValidationError("config.graph_diagnostics schema is invalid")
        unsigned_diagnostics = dict(diagnostics)
        declared_diagnostics_hash = unsigned_diagnostics.pop("diagnostics_sha256")
        if declared_diagnostics_hash != canonical_sha256(unsigned_diagnostics):
            raise ArtifactValidationError("Graph diagnostics self hash is invalid")
        if (
            diagnostics["arm"] != self.identity.arm
            or diagnostics["mode"] != self.config["graph_mode"]
            or diagnostics["support_sha256"] != self.graph_support_hash
            or diagnostics["n_nodes"] != self.identity.hvg
            or diagnostics["cross_dataset_graph_index_pairing"] != "PROHIBITED_LOCAL_INSTANCE_LABEL"
        ):
            raise ArtifactValidationError("Graph diagnostics identity binding is invalid")
        integer_diagnostics = (
            "n_nodes",
            "n_undirected_nonself_edges",
            "n_connected_components",
            "n_isolates",
            "source_n_nodes",
            "source_n_undirected_nonself_edges",
            "source_n_connected_components",
            "source_n_isolates",
        )
        if any(
            isinstance(diagnostics[field], bool)
            or not isinstance(diagnostics[field], int)
            or diagnostics[field] < 0
            for field in integer_diagnostics
        ):
            raise ArtifactValidationError("Graph diagnostic counts are invalid")
        for field in (
            "degree_sequence_sha256",
            "component_partition_sha256",
            "support_sha256",
            "source_support_sha256",
            "source_degree_sequence_sha256",
            "source_component_partition_sha256",
            "diagnostics_sha256",
        ):
            _validate_hash(f"config.graph_diagnostics.{field}", str(diagnostics[field]))
        source_edge_hash = diagnostics["source_edge_sha256"]
        if source_edge_hash is not None:
            _validate_hash("config.graph_diagnostics.source_edge_sha256", str(source_edge_hash))
        for hash_field in ("topology_null_ensemble_audit_hash", "graph_manifest_hash"):
            _validate_hash(f"config.{hash_field}", str(self.config[hash_field]))
        if self.config["topology_null_ensemble_gate_status"] not in {
            "PASS",
            "TEST_FIXTURE_ONLY_NOT_ENFORCED",
            "NOT_REQUESTED",
            "NOT_REQUIRED_FOR_REQUESTED_ARMS",
        }:
            raise ArtifactValidationError("Topology-null gate status is invalid")
        global_trigger_hash = self.config["global_trigger_manifest_hash"]
        if global_trigger_hash is not None:
            _validate_hash("config.global_trigger_manifest_hash", str(global_trigger_hash))
        spaces = self.config["spaces"]
        if (
            not isinstance(spaces, Mapping)
            or set(spaces)
            != {
                "model_control_input",
                "gene_identity",
                "training_target",
                "y_true_and_y_pred",
            }
            or any(not str(value) for value in spaces.values())
        ):
            raise ArtifactValidationError("config.spaces must define every vector space")
        if self.config["held_out_condition"] != self.identity.condition:
            raise ArtifactValidationError("config held-out condition differs from identity")
        train_conditions = tuple(str(value) for value in self.config["train_conditions"])
        validation_conditions = tuple(str(value) for value in self.config["validation_conditions"])
        if (
            not train_conditions
            or not validation_conditions
            or set(train_conditions) & set(validation_conditions)
            or self.identity.condition in set(train_conditions) | set(validation_conditions)
        ):
            raise ArtifactValidationError("Train/validation/held-out condition split is invalid")
        canonical_targets = tuple(str(value) for value in self.config["held_out_canonical_targets"])
        if not canonical_targets or self.config["canonical_target_set"] != "+".join(
            sorted(canonical_targets, key=str.casefold)
        ):
            raise ArtifactValidationError("Canonical target-set binding is invalid")
        gene_panel = self.config["gene_panel_contract"]
        if not isinstance(gene_panel, Mapping) or set(gene_panel) != {
            "nested_manifest_sha256",
            "native_gene_order_sha256",
            "common_evaluation_hvg",
            "common_gene_order",
            "common_gene_order_sha256",
            "nesting_rule",
        }:
            raise ArtifactValidationError("config.gene_panel_contract has an invalid schema")
        _validate_hash(
            "config.gene_panel_contract.nested_manifest_sha256",
            str(gene_panel["nested_manifest_sha256"]),
        )
        if gene_panel["nested_manifest_sha256"] != self.input_hashes.get(
            "nested_gene_panel_manifest"
        ):
            raise ArtifactValidationError("Nested gene-panel manifest is not input-hash bound")
        if gene_panel["native_gene_order_sha256"] != canonical_sha256(self.gene_names):
            raise ArtifactValidationError("Native gene order is not bound to the artifact")
        common_genes = tuple(str(gene) for gene in gene_panel["common_gene_order"])
        common_hvg = gene_panel["common_evaluation_hvg"]
        if (
            isinstance(common_hvg, bool)
            or not isinstance(common_hvg, int)
            or common_hvg <= 0
            or len(common_genes) != common_hvg
            or len(set(common_genes)) != len(common_genes)
            or not set(common_genes) <= set(self.gene_names)
            or gene_panel["common_gene_order_sha256"] != canonical_sha256(common_genes)
        ):
            raise ArtifactValidationError("Common evaluation gene order is invalid")
        try:
            epochs_requested = int(self.config["epochs_requested"])
            epochs_completed = int(self.config["epochs_completed"])
        except (TypeError, ValueError) as error:
            raise ArtifactValidationError("Epoch counts must be integers") from error
        if (
            epochs_requested <= 0
            or epochs_completed <= 0
            or epochs_completed > epochs_requested
            or epochs_completed != len(self.train_loss)
        ):
            raise ArtifactValidationError("Epoch counts do not match saved histories")
        training_early_stopping = self.config["training"].get("early_stopping")
        if not isinstance(training_early_stopping, Mapping):
            raise ArtifactValidationError(
                "config.training.early_stopping must be a complete mapping"
            )
        try:
            expected_early_stopping = replay_early_stopping(
                self.validation_loss,
                minimum_delta=training_early_stopping["minimum_delta"],
                patience=training_early_stopping["patience"],
                epochs_requested=epochs_requested,
            )
        except KeyError as error:
            raise ArtifactValidationError(
                "config.training.early_stopping must bind minimum_delta and patience"
            ) from error
        observed_early_stopping = self.config["early_stopping_observation"]
        if not isinstance(observed_early_stopping, Mapping) or canonical_sha256(
            observed_early_stopping
        ) != canonical_sha256(expected_early_stopping):
            raise ArtifactValidationError(
                "early_stopping_observation does not match deterministic replay"
            )
        deterministic_backend = self.config["deterministic_backend"]
        if (
            not isinstance(deterministic_backend, Mapping)
            or deterministic_backend.get("torch_deterministic_algorithms_enabled") is not True
            or deterministic_backend.get("cublas_workspace_config") != ":4096:8"
            or deterministic_backend.get("cudnn_benchmark") not in {False, None}
            or deterministic_backend.get("cudnn_deterministic") not in {True, None}
        ):
            raise ArtifactValidationError("deterministic backend contract is invalid")
        if self.vector_space != "control_fitted_standardized_delta_expression":
            raise ArtifactValidationError(
                "vector_space must identify the shared standardized delta space"
            )
        _canonical_json(self.config)
        if not self.input_hashes:
            raise ArtifactValidationError("input_hashes cannot be empty")
        required_input_hashes = {
            "dataset",
            "protocol",
            "dataset_passport",
            "condition_panel_manifest",
            "target_encoding_audit",
            "preflight_summary",
            "environment_manifest",
            "environment_lock",
            "git_provenance",
            "nested_gene_panel_manifest",
            "shared_control_cell_evidence",
            "precision_design_registry",
            "precision_archive_condition_table",
            "precision_target_map",
            "hyperparameter_provenance",
        }
        missing_input_hashes = sorted(required_input_hashes - set(self.input_hashes))
        if missing_input_hashes:
            raise ArtifactValidationError(
                f"input_hashes is missing release bindings: {missing_input_hashes}"
            )
        for name, value in self.input_hashes.items():
            _validate_hash(f"input_hashes[{name!r}]", value)
        for name, value in {
            "graph_support_hash": self.graph_support_hash,
            "preprocessing_state_hash": self.preprocessing_state_hash,
            "architecture_hash": self.architecture_hash,
            "initialization_hash": self.initialization_hash,
            "code_hash": self.code_hash,
            "checkpoint_hash": self.checkpoint_hash,
        }.items():
            _validate_hash(name, value)
        if not self.checkpoint_path:
            raise ArtifactValidationError("checkpoint_path cannot be empty")
        checkpoint_source = Path(self.checkpoint_path)
        if checkpoint_source.is_absolute() or ".." in checkpoint_source.parts:
            raise ArtifactValidationError("checkpoint_path must be an artifact-relative path")
        timing_values = {
            "wall_seconds": self.runtime.wall_seconds,
            "training_wall_seconds": self.runtime.training_wall_seconds,
            "inference_wall_seconds": self.runtime.inference_wall_seconds,
            "checkpoint_io_wall_seconds": self.runtime.checkpoint_io_wall_seconds,
            "other_overhead_wall_seconds": self.runtime.other_overhead_wall_seconds,
            "useful_compute_seconds": self.runtime.useful_compute_seconds,
            "failed_attempt_wall_seconds": self.runtime.failed_attempt_wall_seconds,
            "retry_overhead_wall_seconds": self.runtime.retry_overhead_wall_seconds,
        }
        if any(value < 0 or not np.isfinite(value) for value in timing_values.values()):
            raise ArtifactValidationError("runtime timing values must be finite and non-negative")
        phase_total = (
            self.runtime.training_wall_seconds
            + self.runtime.inference_wall_seconds
            + self.runtime.checkpoint_io_wall_seconds
            + self.runtime.other_overhead_wall_seconds
        )
        if not np.isclose(phase_total, self.runtime.wall_seconds, rtol=1e-6, atol=1e-6):
            raise ArtifactValidationError("runtime phase timings must sum to wall_seconds")
        if not np.isclose(
            self.runtime.useful_compute_seconds,
            self.runtime.training_wall_seconds + self.runtime.inference_wall_seconds,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ArtifactValidationError(
                "useful_compute_seconds must equal training plus inference"
            )
        if not all(
            (
                self.runtime.started_at_utc,
                self.runtime.finished_at_utc,
                self.runtime.device,
                self.runtime.host,
                self.runtime.peak_memory_status,
                self.runtime.execution_status,
                self.runtime.accelerator_model,
                self.runtime.cpu_model,
                self.runtime.timing_scope,
            )
        ):
            raise ArtifactValidationError("runtime strings must be non-empty")
        if self.runtime.execution_status != "SUCCESS":
            raise ArtifactValidationError(
                "A released fold artifact requires SUCCESS runtime status"
            )
        if (
            self.runtime.attempt_index <= 0
            or self.runtime.retry_count != self.runtime.attempt_index - 1
        ):
            raise ArtifactValidationError("runtime attempt and retry counts are invalid")
        if self.runtime.failed_attempt_wall_seconds != 0:
            raise ArtifactValidationError(
                "A successful artifact records failed attempts only in the cumulative ledger"
            )
        expected_retry_overhead = (
            self.runtime.other_overhead_wall_seconds if self.runtime.retry_count else 0.0
        )
        if not np.isclose(
            self.runtime.retry_overhead_wall_seconds,
            expected_retry_overhead,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ArtifactValidationError(
                "Artifact retry overhead must equal current-attempt non-useful overhead"
            )
        if (
            self.runtime.accelerator_count < 0
            or self.runtime.logical_cpu_count <= 0
            or self.runtime.host_ram_bytes <= 0
        ):
            raise ArtifactValidationError(
                "runtime hardware counts must be positive or zero as defined"
            )
        if self.runtime.device_hours < 0 or not np.isclose(
            self.runtime.device_hours, self.runtime.wall_seconds / 3600.0, rtol=1e-6, atol=1e-12
        ):
            raise ArtifactValidationError("runtime.device_hours must equal wall_seconds / 3600")
        expected_accelerator_hours = (
            self.runtime.wall_seconds * self.runtime.accelerator_count / 3600.0
        )
        if not np.isclose(
            self.runtime.accelerator_hours,
            expected_accelerator_hours,
            rtol=1e-6,
            atol=1e-12,
        ):
            raise ArtifactValidationError(
                "runtime.accelerator_hours must equal wall_seconds times accelerator_count"
            )
        if self.runtime.peak_device_memory_bytes is not None:
            if self.runtime.peak_device_memory_bytes < 0:
                raise ArtifactValidationError("peak_device_memory_bytes cannot be negative")
        elif not self.runtime.peak_memory_status.startswith("NOT_"):
            raise ArtifactValidationError(
                "Unavailable peak memory requires an explicit reason-coded NOT_* status"
            )

    def to_payload(self) -> dict[str, Any]:
        """Serialize the artifact and add canonical config and artifact hashes."""

        self.validate()
        payload: dict[str, Any] = asdict(self)
        payload["schema_version"] = "1.0"
        payload["config_hash"] = canonical_sha256(self.config)
        payload["artifact_hash"] = canonical_sha256(payload)
        return payload


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value after canonical serialization."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a file without loading it fully into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_fold_artifact(path: Path, artifact: FoldArtifact) -> Path:
    """Atomically write a validated JSON or JSON.GZ artifact."""

    payload = artifact.to_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)
    if path.suffix == ".gz":
        with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.write("\n")
    else:
        temporary.write_text(f"{encoded}\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)
    return path


def read_fold_artifact(path: Path) -> dict[str, Any]:
    """Read an artifact and verify both the canonical config and payload hashes."""

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    expected_config_hash = payload.get("config_hash")
    if expected_config_hash != canonical_sha256(payload.get("config")):
        raise ArtifactValidationError(f"Config hash mismatch in {path}")
    expected_artifact_hash = payload.pop("artifact_hash", None)
    observed_artifact_hash = canonical_sha256(payload)
    payload["artifact_hash"] = expected_artifact_hash
    if expected_artifact_hash != observed_artifact_hash:
        raise ArtifactValidationError(f"Artifact hash mismatch in {path}")
    expected_keys = {
        *FoldArtifact.__dataclass_fields__,
        "schema_version",
        "config_hash",
        "artifact_hash",
    }
    if set(payload) != expected_keys or payload.get("schema_version") != "1.0":
        raise ArtifactValidationError(f"Artifact schema mismatch in {path}")
    try:
        artifact = FoldArtifact(
            identity=FoldIdentity(**payload["identity"]),
            gene_names=tuple(payload["gene_names"]),
            y_true=tuple(payload["y_true"]),
            y_pred=tuple(payload["y_pred"]),
            train_loss=tuple(payload["train_loss"]),
            validation_loss=tuple(payload["validation_loss"]),
            learning_rate=tuple(payload["learning_rate"]),
            vector_space=payload["vector_space"],
            config=payload["config"],
            input_hashes=payload["input_hashes"],
            graph_support_hash=payload["graph_support_hash"],
            preprocessing_state_hash=payload["preprocessing_state_hash"],
            architecture_hash=payload["architecture_hash"],
            initialization_hash=payload["initialization_hash"],
            code_hash=payload["code_hash"],
            checkpoint_path=payload["checkpoint_path"],
            checkpoint_hash=payload["checkpoint_hash"],
            runtime=RuntimeRecord(**payload["runtime"]),
        )
        artifact.validate()
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ArtifactValidationError):
            raise
        raise ArtifactValidationError(
            f"Artifact payload is malformed in {path}: {error}"
        ) from error
    checkpoint = Path(str(payload["checkpoint_path"]))
    if not checkpoint.is_absolute():
        checkpoint = (path.parent / checkpoint).resolve()
    if not checkpoint.exists():
        raise ArtifactValidationError(f"Checkpoint referenced by {path} is missing: {checkpoint}")
    if file_sha256(checkpoint) != payload["checkpoint_hash"]:
        raise ArtifactValidationError(f"Checkpoint hash mismatch in {path}")
    _validate_checkpoint_bindings(checkpoint, payload, path)
    return payload


def _validate_checkpoint_bindings(
    checkpoint_path: Path,
    payload: Mapping[str, Any],
    artifact_path: Path,
) -> None:
    """Verify that the hashed checkpoint is the state promised by its artifact."""

    try:
        import torch
    except ImportError as error:
        raise ArtifactValidationError(
            "Reading released fold artifacts requires PyTorch to validate checkpoint bindings"
        ) from error
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ArtifactValidationError(
            f"Checkpoint cannot be safely decoded for {artifact_path}: {type(error).__name__}"
        ) from error
    if not isinstance(checkpoint, Mapping):
        raise ArtifactValidationError(f"Checkpoint payload is not a mapping in {artifact_path}")
    required = {
        "schema_version",
        "identity",
        "model_state_dict",
        "optimizer_state_dict_at_stop",
        "scheduler_state_dict_at_stop",
        "best_epoch_zero_based",
        "early_stopping_observation",
        "deterministic_backend",
        "architecture",
        "graph_support_hash",
        "preprocessing_state_hash",
        "git_commit",
        "git_worktree_status",
        "git_status_hash",
        "vector_space",
        "artifact_config_hash",
        "input_hashes_hash",
        "code_hash",
        "gene_names_hash",
        "initialization_hash",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ArtifactValidationError(
            f"Checkpoint is missing artifact bindings in {artifact_path}: {missing}"
        )
    expected = {
        "schema_version": "1.0",
        "identity": payload["identity"],
        "architecture": payload["config"]["model"],
        "graph_support_hash": payload["graph_support_hash"],
        "preprocessing_state_hash": payload["preprocessing_state_hash"],
        "git_commit": payload["config"]["git_commit"],
        "git_worktree_status": payload["config"]["git_worktree_status"],
        "git_status_hash": payload["config"]["git_status_hash"],
        "vector_space": payload["vector_space"],
        "artifact_config_hash": payload["config_hash"],
        "input_hashes_hash": canonical_sha256(payload["input_hashes"]),
        "code_hash": payload["code_hash"],
        "gene_names_hash": canonical_sha256(payload["gene_names"]),
        "initialization_hash": payload["initialization_hash"],
        "early_stopping_observation": payload["config"]["early_stopping_observation"],
        "deterministic_backend": payload["config"]["deterministic_backend"],
    }
    mismatches = [
        name
        for name, expected_value in expected.items()
        if canonical_sha256(checkpoint[name]) != canonical_sha256(expected_value)
    ]
    if mismatches:
        raise ArtifactValidationError(
            f"Checkpoint-to-artifact binding mismatch in {artifact_path}: {mismatches}"
        )
    for state_name in (
        "model_state_dict",
        "optimizer_state_dict_at_stop",
        "scheduler_state_dict_at_stop",
    ):
        if not isinstance(checkpoint[state_name], Mapping):
            raise ArtifactValidationError(
                f"Checkpoint {state_name} must be a mapping in {artifact_path}"
            )
    if not checkpoint["model_state_dict"]:
        raise ArtifactValidationError(
            f"Checkpoint model_state_dict cannot be empty in {artifact_path}"
        )
    replayed_early_stopping = replay_early_stopping(
        payload["validation_loss"],
        minimum_delta=payload["config"]["training"]["early_stopping"]["minimum_delta"],
        patience=payload["config"]["training"]["early_stopping"]["patience"],
        epochs_requested=payload["config"]["epochs_requested"],
    )
    best_epoch = checkpoint["best_epoch_zero_based"]
    if (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or best_epoch < 0
        or best_epoch >= len(payload["validation_loss"])
        or best_epoch != replayed_early_stopping["best_epoch_zero_based"]
    ):
        raise ArtifactValidationError(
            f"Checkpoint best epoch is inconsistent with validation history in {artifact_path}"
        )


def metric_row_from_artifact(path: Path) -> dict[str, Any]:
    """Recompute fold metrics from stored vectors, never from a reported scalar."""

    payload = read_fold_artifact(path)
    identity = payload["identity"]
    y_true = np.asarray(payload["y_true"], dtype=float)
    y_pred = np.asarray(payload["y_pred"], dtype=float)
    pearson = _pearson_r(y_true, y_pred)
    spearman = _spearman_r(y_true, y_pred)
    mse = float(np.mean(np.square(y_true - y_pred)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    top20_jaccard = _top_absolute_jaccard(y_true, y_pred, payload["gene_names"], top_k=20)
    gene_panel = payload["config"]["gene_panel_contract"]
    graph_diagnostics = payload["config"]["graph_diagnostics"]
    native_gene_index = {str(gene): index for index, gene in enumerate(payload["gene_names"])}
    common_genes = tuple(str(gene) for gene in gene_panel["common_gene_order"])
    common_indices = np.asarray([native_gene_index[gene] for gene in common_genes], dtype=int)
    common_pearson = _pearson_r(y_true[common_indices], y_pred[common_indices])
    return {
        **identity,
        "panel": payload["config"].get("panel"),
        "analysis_block": payload["config"].get("analysis_block"),
        "epochs_requested": payload["config"].get("epochs_requested"),
        "epochs_completed": payload["config"].get("epochs_completed"),
        "canonical_target_set": payload["config"].get("canonical_target_set"),
        "git_commit": payload["config"].get("git_commit"),
        "git_worktree_status": payload["config"].get("git_worktree_status"),
        "git_status_hash": payload["config"].get("git_status_hash"),
        "pearson_r": pearson,
        "common_200_pearson_r": (
            common_pearson if int(gene_panel["common_evaluation_hvg"]) == 200 else None
        ),
        "common_evaluation_hvg": int(gene_panel["common_evaluation_hvg"]),
        "common_gene_order_hash": str(gene_panel["common_gene_order_sha256"]),
        "nested_gene_panel_manifest_hash": str(gene_panel["nested_manifest_sha256"]),
        "fisher_z_pearson": float(np.arctanh(np.clip(pearson, -1 + 1e-7, 1 - 1e-7))),
        "spearman_r": spearman,
        "mse": mse,
        "mae": mae,
        "top20_absolute_delta_jaccard": top20_jaccard,
        "artifact_hash": payload["artifact_hash"],
        "config_hash": payload["config_hash"],
        "input_hashes_hash": canonical_sha256(payload["input_hashes"]),
        "matched_input_hashes_hash": canonical_sha256(
            {
                key: value
                for key, value in payload["input_hashes"].items()
                if key not in {"preflight_summary", "global_trigger_manifest"}
            }
        ),
        "condition_panel_manifest_hash": payload["input_hashes"].get("condition_panel_manifest"),
        "target_encoding_audit_hash": payload["input_hashes"].get("target_encoding_audit"),
        "preflight_summary_hash": payload["input_hashes"].get("preflight_summary"),
        "dataset_passport_hash": payload["input_hashes"].get("dataset_passport"),
        "dataset_input_hash": payload["input_hashes"].get("dataset"),
        "protocol_file_hash": payload["input_hashes"].get("protocol"),
        "environment_manifest_hash": payload["input_hashes"].get("environment_manifest"),
        "environment_lock_hash": payload["input_hashes"].get("environment_lock"),
        "git_provenance_hash": payload["input_hashes"].get("git_provenance"),
        "shared_control_cell_evidence_hash": payload["input_hashes"].get(
            "shared_control_cell_evidence"
        ),
        "graph_support_hash": payload["graph_support_hash"],
        "preprocessing_state_hash": payload["preprocessing_state_hash"],
        "architecture_hash": payload["architecture_hash"],
        "initialization_hash": payload["initialization_hash"],
        "code_hash": payload["code_hash"],
        "checkpoint_hash": payload["checkpoint_hash"],
        "y_true_hash": canonical_sha256(payload["y_true"]),
        "gene_names_hash": canonical_sha256(payload["gene_names"]),
        "vector_space": payload["vector_space"],
        "split_hash": canonical_sha256(
            {
                "train_conditions": payload["config"].get("train_conditions"),
                "validation_conditions": payload["config"].get("validation_conditions"),
                "held_out_condition": payload["config"].get("held_out_condition"),
            }
        ),
        "topology_null_ensemble_gate_status": payload["config"].get(
            "topology_null_ensemble_gate_status"
        ),
        "topology_null_ensemble_audit_hash": payload["config"].get(
            "topology_null_ensemble_audit_hash"
        ),
        "graph_manifest_hash": payload["config"].get("graph_manifest_hash"),
        "swapped_edge_fraction": payload["config"].get("swapped_edge_fraction"),
        "global_trigger_manifest_hash": payload["config"].get("global_trigger_manifest_hash"),
        **{f"graph_diagnostic_{key}": value for key, value in graph_diagnostics.items()},
        "runtime_wall_seconds": payload["runtime"]["wall_seconds"],
        "runtime_training_wall_seconds": payload["runtime"]["training_wall_seconds"],
        "runtime_inference_wall_seconds": payload["runtime"]["inference_wall_seconds"],
        "runtime_checkpoint_io_wall_seconds": payload["runtime"]["checkpoint_io_wall_seconds"],
        "runtime_other_overhead_wall_seconds": payload["runtime"]["other_overhead_wall_seconds"],
        "runtime_useful_compute_seconds": payload["runtime"]["useful_compute_seconds"],
        "runtime_device_hours": payload["runtime"]["device_hours"],
        "runtime_accelerator_hours": payload["runtime"]["accelerator_hours"],
        "runtime_attempt_index": payload["runtime"]["attempt_index"],
        "runtime_retry_count": payload["runtime"]["retry_count"],
        "runtime_device": payload["runtime"]["device"],
        "runtime_host": payload["runtime"]["host"],
        "runtime_accelerator_model": payload["runtime"]["accelerator_model"],
        "runtime_accelerator_count": payload["runtime"]["accelerator_count"],
        "runtime_cpu_model": payload["runtime"]["cpu_model"],
        "runtime_logical_cpu_count": payload["runtime"]["logical_cpu_count"],
        "runtime_host_ram_bytes": payload["runtime"]["host_ram_bytes"],
        "runtime_peak_device_memory_bytes": payload["runtime"]["peak_device_memory_bytes"],
        "runtime_peak_memory_status": payload["runtime"]["peak_memory_status"],
        "artifact_source_id": f"{payload['artifact_hash'][:16]}_{path.name}",
    }


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ArtifactValidationError(
            f"Value is not canonical-JSON serializable: {error}"
        ) from error


def _validate_hash(name: str, value: str) -> None:
    if not SHA256_RE.fullmatch(str(value)):
        raise ArtifactValidationError(f"{name} must be a lowercase SHA-256 hex digest")


def _pearson_r(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if np.std(left_array) == 0 or np.std(right_array) == 0:
        raise ArtifactValidationError(
            "[DEGENERATE_CORRELATION_ZERO_VARIANCE] Pearson r is undefined for a constant vector"
        )
    return float(np.corrcoef(left_array, right_array)[0, 1])


def _spearman_r(left: Sequence[float], right: Sequence[float]) -> float:
    from scipy.stats import spearmanr

    value = float(
        spearmanr(np.asarray(left, dtype=float), np.asarray(right, dtype=float)).statistic
    )
    if not np.isfinite(value):
        raise ArtifactValidationError(
            "[DEGENERATE_SPEARMAN_ZERO_VARIANCE] Spearman correlation is undefined"
        )
    return value


def _top_absolute_jaccard(
    left: Sequence[float],
    right: Sequence[float],
    gene_names: Sequence[str],
    *,
    top_k: int,
) -> float:
    if top_k <= 0:
        raise ArtifactValidationError("top_k must be positive")
    names = tuple(str(name) for name in gene_names)
    count = min(top_k, len(names))

    def selected(values: Sequence[float]) -> set[str]:
        ordered = sorted(
            zip(names, np.asarray(values, dtype=float), strict=True),
            key=lambda item: (-abs(float(item[1])), item[0]),
        )
        return {name for name, _ in ordered[:count]}

    left_genes = selected(left)
    right_genes = selected(right)
    return float(len(left_genes & right_genes) / len(left_genes | right_genes))

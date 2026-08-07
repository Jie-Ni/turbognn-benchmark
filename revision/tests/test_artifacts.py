from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from cbac_revision.artifacts import (
    FoldArtifact,
    FoldIdentity,
    RuntimeRecord,
    canonical_sha256,
    file_sha256,
    metric_row_from_artifact,
    read_fold_artifact,
    replay_early_stopping,
    write_fold_artifact,
)
from cbac_revision.errors import ArtifactValidationError

HASH = "a" * 64
MODEL = {
    "n_genes": 3,
    "hidden_dim": 8,
    "num_heads": 1,
    "num_layers": 1,
    "dropout": 0.0,
    "gene_identity_dim": 2,
}


def _artifact(checkpoint_path: str = "checkpoint.pt", checkpoint_hash: str = HASH) -> FoldArtifact:
    graph_diagnostics = {
        "arm": "combined",
        "mode": "dense",
        "n_nodes": 3,
        "n_undirected_nonself_edges": 3,
        "degree_sequence_sha256": "7" * 64,
        "n_connected_components": 1,
        "component_partition_sha256": "8" * 64,
        "n_isolates": 0,
        "support_sha256": "c" * 64,
        "source_arm": "dense",
        "source_support_sha256": "c" * 64,
        "source_edge_sha256": None,
        "source_n_nodes": 3,
        "source_n_undirected_nonself_edges": 3,
        "source_degree_sequence_sha256": "7" * 64,
        "source_n_connected_components": 1,
        "source_component_partition_sha256": "8" * 64,
        "source_n_isolates": 0,
        "swapped_edge_fraction": None,
        "rewire_seed": None,
        "cross_dataset_graph_index_pairing": "PROHIBITED_LOCAL_INSTANCE_LABEL",
    }
    graph_diagnostics["diagnostics_sha256"] = canonical_sha256(graph_diagnostics)
    gene_panel_contract = {
        "nested_manifest_sha256": "9" * 64,
        "native_gene_order_sha256": canonical_sha256(("A", "B", "C")),
        "common_evaluation_hvg": 3,
        "common_gene_order": ["A", "B", "C"],
        "common_gene_order_sha256": canonical_sha256(("A", "B", "C")),
        "nesting_rule": "200_subset_500_subset_1000_by_frozen_rank",
    }
    return FoldArtifact(
        identity=FoldIdentity(
            run_id="run-001",
            dataset="adamson",
            hvg=3,
            arm="combined",
            condition="TP53",
            seed=42,
        ),
        gene_names=("A", "B", "C"),
        y_true=(0.0, 1.0, 2.0),
        y_pred=(0.0, 1.0, 1.5),
        train_loss=(1.0, 0.5),
        validation_loss=(1.2, 0.7),
        learning_rate=(0.001, 0.0005),
        vector_space="control_fitted_standardized_delta_expression",
        config={
            "protocol_id": "cbac-kg-prior-major-revision-v2",
            "panel": "primary",
            "analysis_block": "topology_primary",
            "train_conditions": ["A", "B"],
            "validation_conditions": ["C"],
            "held_out_condition": "TP53",
            "held_out_canonical_targets": ["TP53"],
            "canonical_target_set": "TP53",
            "gene_panel_contract": gene_panel_contract,
            "epochs_requested": 2,
            "epochs_completed": 2,
            "early_stopping_observation": replay_early_stopping(
                (1.2, 0.7),
                minimum_delta=0.01,
                patience=10,
                epochs_requested=2,
            ),
            "deterministic_backend": {
                "torch_deterministic_algorithms_enabled": True,
                "cublas_workspace_config": ":4096:8",
                "cudnn_available": False,
                "cudnn_benchmark": None,
                "cudnn_deterministic": None,
            },
            "git_commit": "3" * 40,
            "git_worktree_status": "CLEAN",
            "git_status_hash": "4" * 64,
            "model": MODEL,
            "training": {
                "maximum_epochs": 2,
                "early_stopping": {"minimum_delta": 0.01, "patience": 10},
            },
            "graph_mode": "dense",
            "graph_diagnostics": graph_diagnostics,
            "swapped_edge_fraction": None,
            "topology_null_ensemble_gate_status": "PASS",
            "topology_null_ensemble_audit_hash": "5" * 64,
            "graph_manifest_hash": "6" * 64,
            "global_trigger_manifest_hash": None,
            "spaces": {
                "model_control_input": "control_only_log_normalized_mean_expression",
                "gene_identity": "selected_gene_order_bound_learned_embedding",
                "training_target": "control_fitted_standardized_condition_expression",
                "y_true_and_y_pred": "control_fitted_standardized_delta_expression",
            },
        },
        input_hashes={
            "dataset": HASH,
            "protocol": "b" * 64,
            "dataset_passport": "c" * 64,
            "condition_panel_manifest": "d" * 64,
            "target_encoding_audit": "e" * 64,
            "preflight_summary": "f" * 64,
            "environment_manifest": "1" * 64,
            "environment_lock": "2" * 64,
            "git_provenance": "3" * 64,
            "nested_gene_panel_manifest": "9" * 64,
            "shared_control_cell_evidence": "8" * 64,
            "precision_design_registry": "a" * 64,
            "precision_archive_condition_table": "b" * 64,
            "precision_target_map": "c" * 64,
            "hyperparameter_provenance": "d" * 64,
        },
        graph_support_hash="c" * 64,
        preprocessing_state_hash="d" * 64,
        architecture_hash=canonical_sha256(MODEL),
        initialization_hash="f" * 64,
        code_hash="1" * 64,
        checkpoint_path=checkpoint_path,
        checkpoint_hash=checkpoint_hash,
        runtime=RuntimeRecord(
            started_at_utc="2026-08-06T12:00:00Z",
            finished_at_utc="2026-08-06T12:00:04Z",
            wall_seconds=4.0,
            training_wall_seconds=2.0,
            inference_wall_seconds=1.0,
            checkpoint_io_wall_seconds=0.5,
            other_overhead_wall_seconds=0.5,
            useful_compute_seconds=3.0,
            failed_attempt_wall_seconds=0.0,
            retry_overhead_wall_seconds=0.0,
            attempt_index=1,
            retry_count=0,
            execution_status="SUCCESS",
            device="cpu-test",
            host="test-host",
            accelerator_model="NOT_APPLICABLE_CPU_EXECUTION",
            accelerator_count=0,
            accelerator_hours=0.0,
            cpu_model="test-cpu",
            logical_cpu_count=4,
            host_ram_bytes=8_000_000_000,
            peak_device_memory_bytes=None,
            peak_memory_status="NOT_APPLICABLE_CPU",
            device_hours=4.0 / 3600.0,
            timing_scope="test_fixture",
        ),
    )


def _write_bound_checkpoint(path: Path, artifact: FoldArtifact, **overrides: object) -> None:
    checkpoint = {
        "schema_version": "1.0",
        "identity": asdict(artifact.identity),
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "optimizer_state_dict_at_stop": {},
        "scheduler_state_dict_at_stop": {},
        "best_epoch_zero_based": 1,
        "early_stopping_observation": artifact.config["early_stopping_observation"],
        "deterministic_backend": artifact.config["deterministic_backend"],
        "architecture": artifact.config["model"],
        "graph_support_hash": artifact.graph_support_hash,
        "preprocessing_state_hash": artifact.preprocessing_state_hash,
        "git_commit": artifact.config["git_commit"],
        "git_worktree_status": artifact.config["git_worktree_status"],
        "git_status_hash": artifact.config["git_status_hash"],
        "vector_space": artifact.vector_space,
        "artifact_config_hash": canonical_sha256(artifact.config),
        "input_hashes_hash": canonical_sha256(artifact.input_hashes),
        "code_hash": artifact.code_hash,
        "gene_names_hash": canonical_sha256(artifact.gene_names),
        "initialization_hash": artifact.initialization_hash,
    }
    checkpoint.update(overrides)
    torch.save(checkpoint, path)


def test_artifact_round_trip_preserves_vectors_losses_and_hashes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_bound_checkpoint(checkpoint, _artifact())
    path = write_fold_artifact(
        tmp_path / "fold.json.gz", _artifact("checkpoint.pt", file_sha256(checkpoint))
    )
    payload = read_fold_artifact(path)
    metric = metric_row_from_artifact(path)

    assert payload["y_true"] == [0.0, 1.0, 2.0]
    assert payload["y_pred"] == [0.0, 1.0, 1.5]
    assert payload["train_loss"] == [1.0, 0.5]
    assert payload["validation_loss"] == [1.2, 0.7]
    assert payload["learning_rate"] == [0.001, 0.0005]
    assert len(payload["config_hash"]) == 64
    assert len(payload["artifact_hash"]) == 64
    assert metric["condition"] == "TP53"
    assert metric["mse"] > 0


def test_artifact_rejects_missing_loss_history() -> None:
    artifact = _artifact()
    invalid = FoldArtifact(**{**artifact.__dict__, "validation_loss": ()})

    with pytest.raises(ArtifactValidationError):
        invalid.validate()


def test_metric_rejects_degenerate_correlation_with_reason_code(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_bound_checkpoint(checkpoint, _artifact())
    artifact = _artifact("checkpoint.pt", file_sha256(checkpoint))
    invalid = FoldArtifact(**{**artifact.__dict__, "y_pred": (1.0, 1.0, 1.0)})
    path = write_fold_artifact(tmp_path / "constant.json", invalid)

    with pytest.raises(ArtifactValidationError, match="DEGENERATE_CORRELATION_ZERO_VARIANCE"):
        metric_row_from_artifact(path)


def test_reader_rejects_self_rehashed_semantically_invalid_artifact(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_bound_checkpoint(checkpoint, _artifact())
    path = write_fold_artifact(
        tmp_path / "fold.json", _artifact("checkpoint.pt", file_sha256(checkpoint))
    )
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["runtime"]["wall_seconds"] = 99.0
    payload_without_hash = dict(payload)
    payload_without_hash.pop("artifact_hash")
    payload["artifact_hash"] = canonical_sha256(payload_without_hash)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="phase timings"):
        read_fold_artifact(path)


def test_reader_rejects_hashed_checkpoint_bound_to_a_different_graph(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_bound_checkpoint(checkpoint, _artifact(), graph_support_hash="9" * 64)
    path = write_fold_artifact(
        tmp_path / "fold.json", _artifact("checkpoint.pt", file_sha256(checkpoint))
    )

    with pytest.raises(ArtifactValidationError, match="graph_support_hash"):
        read_fold_artifact(path)


def _artifact_with_subthreshold_raw_minimum() -> FoldArtifact:
    artifact = _artifact()
    validation_loss = (1.2, 1.15)
    training = copy.deepcopy(artifact.config["training"])
    training["early_stopping"] = {"minimum_delta": 0.1, "patience": 10}
    config = {
        **artifact.config,
        "training": training,
        "early_stopping_observation": replay_early_stopping(
            validation_loss,
            minimum_delta=0.1,
            patience=10,
            epochs_requested=2,
        ),
    }
    return FoldArtifact(
        **{
            **artifact.__dict__,
            "validation_loss": validation_loss,
            "config": config,
        }
    )


def test_checkpoint_best_epoch_is_replayed_instead_of_using_raw_argmin(tmp_path: Path) -> None:
    artifact = _artifact_with_subthreshold_raw_minimum()
    checkpoint = tmp_path / "checkpoint.pt"
    _write_bound_checkpoint(checkpoint, artifact, best_epoch_zero_based=0)
    path = write_fold_artifact(
        tmp_path / "fold.json", _artifact_with_checkpoint(artifact, checkpoint)
    )

    payload = read_fold_artifact(path)

    assert payload["config"]["early_stopping_observation"]["best_epoch_zero_based"] == 0
    assert payload["validation_loss"].index(min(payload["validation_loss"])) == 1


def test_checkpoint_rejects_raw_argmin_when_it_differs_from_replayed_selection(
    tmp_path: Path,
) -> None:
    artifact = _artifact_with_subthreshold_raw_minimum()
    checkpoint = tmp_path / "checkpoint.pt"
    _write_bound_checkpoint(checkpoint, artifact, best_epoch_zero_based=1)
    path = write_fold_artifact(
        tmp_path / "fold.json", _artifact_with_checkpoint(artifact, checkpoint)
    )

    with pytest.raises(ArtifactValidationError, match="best epoch"):
        read_fold_artifact(path)


@pytest.mark.parametrize("tampered_field", ["stopped_epoch", "trace"])
def test_artifact_rejects_tampered_early_stopping_observation(
    tampered_field: str,
) -> None:
    artifact = _artifact()
    config = copy.deepcopy(artifact.config)
    if tampered_field == "stopped_epoch":
        config["early_stopping_observation"]["stopped_epoch_zero_based"] = 0
    else:
        config["early_stopping_observation"]["trace"][1]["qualified_improvement"] = False
    invalid = FoldArtifact(**{**artifact.__dict__, "config": config})

    with pytest.raises(ArtifactValidationError, match="deterministic replay"):
        invalid.validate()


def _artifact_with_checkpoint(artifact: FoldArtifact, checkpoint: Path) -> FoldArtifact:
    return FoldArtifact(
        **{
            **artifact.__dict__,
            "checkpoint_path": checkpoint.name,
            "checkpoint_hash": file_sha256(checkpoint),
        }
    )

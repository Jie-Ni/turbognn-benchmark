from __future__ import annotations

import json
from pathlib import Path

from cbac_revision.artifacts import canonical_sha256, file_sha256
from cbac_revision.global_trigger import aggregate_global_trigger, read_global_trigger_manifest
from cbac_revision.protocol import load_protocol


def _summaries(protocol_hash: str, *, triggered_dataset: str | None) -> list[dict]:
    summaries = []
    for dataset in ("adamson", "norman", "replogle_k562", "replogle_rpe1"):
        representativeness_hash = canonical_sha256(
            {"dataset": dataset, "table": "panel_representativeness_long_form"}
        )
        primary = [f"{dataset}_P{index:02d}" for index in range(50)]
        sensitivity = [f"{dataset}_S{index:02d}" for index in range(50)]
        legacy = [f"{dataset}_L{index:02d}" for index in range(50)]
        panel = {
            "dataset": dataset,
            "primary": {"ordered_canonical_condition_ids": primary},
            "sensitivity": {"ordered_canonical_condition_ids": sensitivity},
            "non_overlap_status": "PASS",
            "primary_sensitivity_overlap": [],
            "legacy_first50": {
                "n_conditions": 50,
                "ordered_condition_ids": legacy,
                "binding_source": "hash_verified_dataset_passport",
                "primary_overlap": [],
                "sensitivity_overlap": [],
            },
            "primary_panel_contract_status": "PASS",
            "conditional_panel_contract_status": "PASS",
            "representativeness_table_hash": representativeness_hash,
        }
        panel["manifest_hash"] = canonical_sha256(panel)
        environment = {
            "fixture_status": "PRODUCTION",
            "environment_lock_sha256": "8" * 64,
        }
        environment["manifest_hash"] = canonical_sha256(environment)
        git_provenance = {
            "schema_version": "1.0",
            "git_commit": "7" * 40,
            "git_commit_source": "test_fixture",
            "git_worktree_status": "CLEAN",
            "git_status_entry_count": 0,
            "git_status_hash": canonical_sha256([]),
            "path_disclosure_policy": "status_paths_hashed_not_embedded",
            "code_tree_sha256": "9" * 64,
        }
        git_provenance["provenance_hash"] = canonical_sha256(git_provenance)
        summary = {
            "dataset": dataset,
            "panel": "primary",
            "hvg": 200,
            "analysis_block": "topology_primary",
            "requested_panel_size": 50,
            "n_selected_conditions": 50,
            "selected_conditions": primary,
            "can_execute": True,
            "blockers": [],
            "input_hashes": {
                "protocol": protocol_hash,
                "dataset": "1" * 64,
                "dataset_passport": "2" * 64,
                "condition_panel_manifest": panel["manifest_hash"],
                "panel_representativeness": representativeness_hash,
                "target_encoding_audit": "3" * 64,
                "environment_manifest": environment["manifest_hash"],
                "environment_lock": "8" * 64,
                "git_provenance": git_provenance["provenance_hash"],
                "nested_gene_panel_manifest": "a" * 64,
                "precision_design_registry": "b" * 64,
                "hyperparameter_provenance": "c" * 64,
            },
            "condition_panel_manifest": panel,
            "condition_panel_manifest_hash": panel["manifest_hash"],
            "panel_representativeness_table_hash": representativeness_hash,
            "environment_manifest": environment,
            "git_provenance": git_provenance,
            "code_hash": "9" * 64,
            "conditional_sensitivity_feasible": True,
            "sensitivity_trigger": {
                "sensitivity_triggered": dataset == triggered_dataset,
                "trigger_total_variation_distance": 0.11 if dataset == triggered_dataset else 0.02,
                "trigger_abs_standardized_log1p_cell_count_difference": 0.1,
                "trigger_tv_threshold": 0.10,
                "trigger_cell_count_threshold": 0.25,
                "representativeness_table_hash": representativeness_hash,
                "trigger_maximum_total_variation_variable": "pathway_class",
                "trigger_variable_total_variation_distances": {
                    "pathway_class": (0.11 if dataset == triggered_dataset else 0.02)
                },
            },
        }
        summary["summary_hash"] = canonical_sha256(summary)
        summaries.append(summary)
    return summaries


def test_one_dataset_trigger_requires_all_four_datasets_and_1200_fits(tmp_path: Path) -> None:
    protocol_path = Path(__file__).parents[1] / "protocol.yaml"
    protocol = load_protocol(protocol_path)
    protocol_hash = file_sha256(protocol_path)
    payload = aggregate_global_trigger(
        _summaries(protocol_hash, triggered_dataset="adamson"), protocol, protocol_hash
    )
    path = tmp_path / "global_trigger.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    verified = read_global_trigger_manifest(path, protocol, protocol_hash)

    assert verified["status"] == "PASS"
    assert verified["global_triggered"] is True
    assert verified["conditional_fits_required"] == 1_200
    assert verified["execution_datasets_if_triggered"] == sorted(protocol["datasets"])


def test_no_dataset_trigger_produces_registry_backed_not_required_state() -> None:
    protocol_path = Path(__file__).parents[1] / "protocol.yaml"
    protocol = load_protocol(protocol_path)
    protocol_hash = file_sha256(protocol_path)
    payload = aggregate_global_trigger(
        _summaries(protocol_hash, triggered_dataset=None), protocol, protocol_hash
    )

    assert payload["status"] == "PASS"
    assert payload["global_triggered"] is False
    assert payload["conditional_execution_required"] is False
    assert payload["conditional_fits_required"] == 0


def test_trigger_is_exclusive_at_frozen_threshold_boundary() -> None:
    protocol_path = Path(__file__).parents[1] / "protocol.yaml"
    protocol = load_protocol(protocol_path)
    protocol_hash = file_sha256(protocol_path)
    summaries = _summaries(protocol_hash, triggered_dataset=None)
    for summary in summaries:
        summary.pop("summary_hash")
        summary["sensitivity_trigger"]["trigger_total_variation_distance"] = 0.10
        summary["sensitivity_trigger"][
            "trigger_abs_standardized_log1p_cell_count_difference"
        ] = 0.25
        summary["summary_hash"] = canonical_sha256(summary)

    payload = aggregate_global_trigger(summaries, protocol, protocol_hash)

    assert payload["status"] == "PASS"
    assert payload["global_triggered"] is False


def test_tampered_summary_hash_blocks_global_trigger_release() -> None:
    protocol_path = Path(__file__).parents[1] / "protocol.yaml"
    protocol = load_protocol(protocol_path)
    protocol_hash = file_sha256(protocol_path)
    summaries = _summaries(protocol_hash, triggered_dataset=None)
    summaries[0]["selected_conditions"] = summaries[0]["selected_conditions"][::-1]

    payload = aggregate_global_trigger(summaries, protocol, protocol_hash)

    assert payload["status"] == "BLOCKED"
    assert "PREFLIGHT_SUMMARY_HASH_MISMATCH" in {
        failure["reason_code"] for failure in payload["failures"]
    }


def test_representativeness_hash_tamper_blocks_global_or_release() -> None:
    protocol_path = Path(__file__).parents[1] / "protocol.yaml"
    protocol = load_protocol(protocol_path)
    protocol_hash = file_sha256(protocol_path)
    summaries = _summaries(protocol_hash, triggered_dataset="adamson")
    summaries[0].pop("summary_hash")
    summaries[0]["sensitivity_trigger"]["representativeness_table_hash"] = "f" * 64
    summaries[0]["summary_hash"] = canonical_sha256(summaries[0])

    payload = aggregate_global_trigger(summaries, protocol, protocol_hash)

    assert payload["status"] == "BLOCKED"
    assert "REPRESENTATIVENESS_TABLE_BINDING_MISMATCH" in {
        failure["reason_code"] for failure in payload["failures"]
    }

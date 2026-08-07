from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cbac_revision.artifacts import canonical_sha256, file_sha256
from cbac_revision.statistics import (
    AnalysisReleaseResult,
    build_decision_release_registry,
    build_thirteen_result_lock_registry,
    condition_ranking_overlap,
    mandatory_fit_coverage_release,
    mixed_support_sensitivity_release,
    primary_decision_rule,
    primary_hierarchical_release,
    propagation_control_release,
    scale_interaction_release,
    secondary_metric_release,
    topology_hierarchical_release,
)

PROTOCOL_PATH = Path(__file__).parents[1] / "protocol.yaml"
PROTOCOL_HASH = file_sha256(PROTOCOL_PATH)
CODE_HASH = hashlib.sha256(b"frozen-release-code").hexdigest()
ENVIRONMENT_LOCK_HASH = hashlib.sha256(b"frozen-environment-lock").hexdigest()


def _mini_protocol() -> dict:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol = deepcopy(protocol)
    protocol["planned_fit_counts"]["conditions_per_panel"] = 3
    protocol["planned_fit_counts"]["mandatory_total_fits"] = 648
    protocol["planned_fit_counts"]["conditional_non_overlap_fits_if_triggered"] = 72
    protocol["statistics"]["bootstrap_replicates"] = 200
    return protocol


def _panel_manifest(dataset: str, protocol: dict) -> dict:
    conditions = ["G0", "G1", "G2"]
    entries = [
        {
            "condition": condition,
            "canonical_targets": [condition],
            "raw_condition_members": [condition],
            "target_multiplicity": 1,
        }
        for condition in conditions
    ]
    manifest = {
        "schema_version": "1.0",
        "dataset": dataset,
        "protocol_id": protocol["protocol_id"],
        "protocol_file_sha256": PROTOCOL_HASH,
        "dataset_passport_sha256": hashlib.sha256(
            f"passport:{dataset}".encode("utf-8")
        ).hexdigest(),
        "control_binding": {"binding_source": "hash_verified_dataset_passport"},
        "primary": {
            "ordered_canonical_condition_ids": conditions,
            "entries": entries,
        },
        "sensitivity": {
            "ordered_canonical_condition_ids": [f"S{index}" for index in range(3)],
            "entries": [
                {
                    "condition": f"S{index}",
                    "canonical_targets": [f"S{index}"],
                }
                for index in range(3)
            ],
        },
        "non_overlap_status": "PASS",
        "primary_sensitivity_overlap": [],
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    return manifest


def _environment_manifest() -> dict:
    manifest = {
        "schema_version": "1.0",
        "fixture_status": "PRODUCTION",
        "environment_lock_sha256": ENVIRONMENT_LOCK_HASH,
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    return manifest


def _git_provenance() -> dict:
    provenance = {
        "schema_version": "1.0",
        "git_commit": "5" * 40,
        "git_commit_source": "git_rev_parse_HEAD_and_porcelain_v1",
        "git_worktree_status": "CLEAN",
        "git_status_entry_count": 0,
        "git_status_hash": canonical_sha256([]),
        "path_disclosure_policy": "status_paths_hashed_not_embedded",
        "code_tree_sha256": CODE_HASH,
    }
    provenance["provenance_hash"] = canonical_sha256(provenance)
    return provenance


def _preflight_summaries(protocol: dict) -> list[dict]:
    summaries: list[dict] = []
    block_specs = (
        (200, "topology_primary", protocol["analysis_blocks"]["topology_primary"]["default_arms"]),
        (
            200,
            "mixed_support_sensitivity",
            protocol["analysis_blocks"]["mixed_support_sensitivity"]["default_arms"],
        ),
        (500, "scale_extension", ["dense", "string_go"]),
        (1000, "scale_extension", ["dense", "string_go"]),
    )
    environment = _environment_manifest()
    git_provenance = _git_provenance()
    for dataset in sorted(protocol["datasets"]):
        panel = _panel_manifest(dataset, protocol)
        for hvg, block, arms in block_specs:
            input_hashes = {
                "protocol": PROTOCOL_HASH,
                "dataset": hashlib.sha256(f"dataset:{dataset}".encode()).hexdigest(),
                "dataset_passport": hashlib.sha256(f"passport:{dataset}".encode()).hexdigest(),
                "condition_panel_manifest": panel["manifest_hash"],
                "target_encoding_audit": hashlib.sha256(
                    f"targets:{dataset}:{hvg}".encode()
                ).hexdigest(),
                "environment_manifest": environment["manifest_hash"],
                "environment_lock": ENVIRONMENT_LOCK_HASH,
                "git_provenance": git_provenance["provenance_hash"],
                "nested_gene_panel_manifest": hashlib.sha256(
                    f"nested:{dataset}".encode()
                ).hexdigest(),
                "shared_control_cell_evidence": hashlib.sha256(
                    f"shared-control:{dataset}:{hvg}".encode()
                ).hexdigest(),
                "precision_design_registry": "a" * 64,
                "precision_archive_condition_table": "b" * 64,
                "precision_target_map": "c" * 64,
                "hyperparameter_provenance": "d" * 64,
            }
            summary = {
                "dataset": dataset,
                "hvg": hvg,
                "panel": "primary",
                "analysis_block": block,
                "requested_arms": list(arms),
                "requested_panel_size": 3,
                "n_selected_conditions": 3,
                "selected_conditions": ["G0", "G1", "G2"],
                "can_execute": True,
                "blockers": [],
                "input_hashes": input_hashes,
                "condition_panel_manifest": panel,
                "condition_panel_manifest_hash": panel["manifest_hash"],
                "environment_manifest": environment,
                "git_provenance": git_provenance,
                "code_hash": CODE_HASH,
            }
            summary["summary_hash"] = canonical_sha256(summary)
            summaries.append(summary)
    return summaries


def _complete_metric_frame() -> pd.DataFrame:
    rows: list[dict] = []
    protocol = _mini_protocol()
    summaries = _preflight_summaries(protocol)
    preflight_by_key = {
        (summary["dataset"], summary["hvg"], summary["analysis_block"]): summary
        for summary in summaries
    }
    datasets = ("adamson", "norman", "replogle_k562", "replogle_rpe1")
    conditions = ("G0", "G1", "G2")
    for dataset_index, dataset in enumerate(datasets):
        for condition_index, condition in enumerate(conditions):
            base = 0.30 + dataset_index * 0.02 + condition_index * 0.01
            for seed_index, seed in enumerate((42, 43, 44)):
                seed_offset = seed_index * 0.001
                values: dict[tuple[int, str], float] = {
                    (200, "dense"): base + seed_offset,
                    (200, "combined"): base + 0.020 + seed_offset,
                    (200, "self_loop"): base - 0.010 + seed_offset,
                    (200, "string_go"): base + 0.010 + seed_offset,
                    (500, "dense"): base + seed_offset,
                    (500, "string_go"): base + 0.030 + seed_offset,
                    (1000, "dense"): base + seed_offset,
                    (1000, "string_go"): base + 0.050 + seed_offset,
                }
                for graph_index in range(1, 11):
                    values[(200, f"string_go_rewire_{graph_index:02d}")] = (
                        base - 0.010 + graph_index * 0.0002 + seed_offset
                    )
                for (hvg, arm), value in values.items():

                    def digest(label: str) -> str:
                        return hashlib.sha256(label.encode("utf-8")).hexdigest()

                    if hvg == 200 and arm == "combined":
                        block = "mixed_support_sensitivity"
                    elif hvg == 200:
                        block = "topology_primary"
                    else:
                        block = "scale_extension"
                    summary = preflight_by_key[(dataset, hvg, block)]
                    inputs = summary["input_hashes"]
                    graph_hash = digest(f"graph:{dataset}:{hvg}:{arm}")
                    source_graph_hash = digest(f"graph:{dataset}:200:string_go")
                    source_edge_hash = digest(f"source-edges:{dataset}:200:string_go")
                    is_rewire = arm.startswith("string_go_rewire_")
                    rewire_index = int(arm.rsplit("_", 1)[-1]) if is_rewire else None
                    graph_record = {
                        "arm": arm,
                        "mode": "degree_preserving_rewire" if is_rewire else "curated",
                        "n_nodes": hvg,
                        "n_undirected_nonself_edges": hvg,
                        "degree_sequence_sha256": digest(f"degree:{dataset}:{hvg}"),
                        "n_connected_components": 1,
                        "component_partition_sha256": digest(f"components:{dataset}:{hvg}"),
                        "n_isolates": 0,
                        "support_sha256": graph_hash,
                        "source_arm": "string_go" if is_rewire else arm,
                        "source_support_sha256": source_graph_hash if is_rewire else graph_hash,
                        "source_edge_sha256": source_edge_hash,
                        "source_n_nodes": hvg,
                        "source_n_undirected_nonself_edges": hvg,
                        "source_degree_sequence_sha256": digest(f"degree:{dataset}:{hvg}"),
                        "source_n_connected_components": 1,
                        "source_component_partition_sha256": digest(f"components:{dataset}:{hvg}"),
                        "source_n_isolates": 0,
                        "swapped_edge_fraction": 0.9 if is_rewire else None,
                        "rewire_seed": (
                            protocol["graph_supports"]["topology_null"]["replicate_seeds"][
                                rewire_index - 1
                            ]
                            if is_rewire
                            else None
                        ),
                        "cross_dataset_graph_index_pairing": ("PROHIBITED_LOCAL_INSTANCE_LABEL"),
                    }
                    graph_record["diagnostics_sha256"] = canonical_sha256(graph_record)
                    rows.append(
                        {
                            "dataset": dataset,
                            "hvg": hvg,
                            "panel": "primary",
                            "analysis_block": block,
                            "epochs_requested": 50,
                            "arm": arm,
                            "condition": condition,
                            "seed": seed,
                            "pearson_r": value,
                            "common_200_pearson_r": value,
                            "common_evaluation_hvg": 200,
                            "common_gene_order_hash": digest(f"common-genes:{dataset}"),
                            "nested_gene_panel_manifest_hash": inputs["nested_gene_panel_manifest"],
                            "artifact_hash": digest(
                                f"artifact:{dataset}:{hvg}:{arm}:{condition}:{seed}"
                            ),
                            "architecture_hash": digest(f"architecture:{dataset}:{hvg}"),
                            "initialization_hash": digest(
                                f"initialization:{dataset}:{hvg}:{condition}:{seed}"
                            ),
                            "preprocessing_state_hash": digest(f"preprocessing:{dataset}:{hvg}"),
                            "input_hashes_hash": digest(f"inputs:{dataset}:{hvg}:{block}"),
                            "matched_input_hashes_hash": digest(f"matched-inputs:{dataset}:{hvg}"),
                            "split_hash": digest(f"split:{dataset}:{hvg}:{condition}:{seed}"),
                            "graph_support_hash": graph_hash,
                            "code_hash": CODE_HASH,
                            "y_true_hash": digest(f"ytrue:{dataset}:{hvg}:{condition}:{seed}"),
                            "gene_names_hash": digest(f"genes:{dataset}:{hvg}"),
                            "vector_space": "control_fitted_standardized_delta_expression",
                            "condition_panel_manifest_hash": inputs["condition_panel_manifest"],
                            "target_encoding_audit_hash": inputs["target_encoding_audit"],
                            "preflight_summary_hash": summary["summary_hash"],
                            "dataset_passport_hash": inputs["dataset_passport"],
                            "canonical_target_set": condition,
                            "dataset_input_hash": inputs["dataset"],
                            "protocol_file_hash": PROTOCOL_HASH,
                            "environment_manifest_hash": inputs["environment_manifest"],
                            "environment_lock_hash": inputs["environment_lock"],
                            "git_provenance_hash": inputs["git_provenance"],
                            "shared_control_cell_evidence_hash": inputs[
                                "shared_control_cell_evidence"
                            ],
                            "git_commit": summary["git_provenance"]["git_commit"],
                            "git_worktree_status": summary["git_provenance"]["git_worktree_status"],
                            "git_status_hash": summary["git_provenance"]["git_status_hash"],
                            "topology_null_ensemble_gate_status": "PASS",
                            "topology_null_ensemble_audit_hash": digest(
                                f"topology-audit:{dataset}"
                            ),
                            "swapped_edge_fraction": (0.9 if is_rewire else None),
                            **{
                                f"graph_diagnostic_{key}": diagnostic_value
                                for key, diagnostic_value in graph_record.items()
                            },
                        }
                    )
    return pd.DataFrame(rows)


def test_primary_release_is_equal_dataset_hierarchical_and_deterministic() -> None:
    protocol = _mini_protocol()
    first = primary_hierarchical_release(_complete_metric_frame(), protocol)
    second = primary_hierarchical_release(_complete_metric_frame(), protocol)

    assert first.released
    assert first.registry["estimate"] == pytest.approx(0.010)
    assert first.registry["decision"] == "DIRECTIONAL_POSITIVE"
    assert first.registry["n_conditions_expected"] == 12
    assert first.registry["n_conditions_actual"] == 12
    assert first.registry["bootstrap_index_hash"] == second.registry["bootstrap_index_hash"]


def test_primary_release_withholds_on_one_missing_seed_cell() -> None:
    frame = _complete_metric_frame()
    missing = frame[
        ~(
            (frame["dataset"] == "adamson")
            & (frame["condition"] == "G0")
            & (frame["hvg"] == 200)
            & (frame["arm"] == "dense")
            & (frame["seed"] == 44)
        )
    ]
    result = primary_hierarchical_release(missing, _mini_protocol())

    assert not result.released
    assert "INCOMPLETE_SEED_OR_SUPPORT_CELL" in set(result.failures["reason_code"])


def test_scale_release_uses_paired_arm_by_scale_interactions_and_one_bh_family() -> None:
    result = scale_interaction_release(_complete_metric_frame(), _mini_protocol())

    assert result.released
    contrasts = {row["contrast"]: row for row in result.registry["contrasts"]}
    assert contrasts["500_minus_200_hvg"]["estimate"] == pytest.approx(0.020)
    assert contrasts["1000_minus_200_hvg"]["estimate"] == pytest.approx(0.040)
    assert all("bh_q_two_contrast_family" in row for row in contrasts.values())
    assert result.registry["invariant_condition_seed_support"] == "PASS"


def test_scale_release_withholds_when_one_condition_scale_arm_cell_is_missing() -> None:
    frame = _complete_metric_frame()
    missing = frame[
        ~(
            (frame["dataset"] == "norman")
            & (frame["condition"] == "G1")
            & (frame["hvg"] == 1000)
            & (frame["arm"] == "string_go")
            & (frame["seed"] == 43)
        )
    ]

    assert not scale_interaction_release(missing, _mini_protocol()).released


def test_topology_release_resamples_conditions_and_all_ten_graph_instances() -> None:
    result = topology_hierarchical_release(_complete_metric_frame(), _mini_protocol())

    assert result.released
    assert result.registry["n_graph_instances_actual"] == 10
    assert result.registry["graph_instance_coverage"] == "PASS"
    assert result.registry["local_graph_instance_sd_descriptive"] > 0
    assert result.registry["graph_index_cross_dataset_pairing"] == "PROHIBITED_ARBITRARY_LABELS"
    assert 0 < result.registry["two_sided_centered_null_bootstrap_p"] <= 1
    graph_table = result.detail_tables["graph_instance_estimates"]
    assert len(graph_table) == 40
    assert set(graph_table["cross_dataset_pairing_status"]) == {"NOT_PAIRED_ACROSS_DATASETS"}


def test_topology_release_withholds_if_one_graph_seed_is_missing() -> None:
    frame = _complete_metric_frame()
    missing = frame[
        ~(
            (frame["dataset"] == "replogle_rpe1")
            & (frame["condition"] == "G2")
            & (frame["arm"] == "string_go_rewire_10")
            & (frame["seed"] == 42)
        )
    ]

    assert not topology_hierarchical_release(missing, _mini_protocol()).released


def test_propagation_control_releases_both_contrasts_against_same_self_loop() -> None:
    result = propagation_control_release(_complete_metric_frame(), _mini_protocol())

    assert result.released
    contrasts = {row["contrast"]: row for row in result.registry["contrasts"]}
    assert set(contrasts) == {"string_go_minus_self_loop", "dense_minus_self_loop"}
    assert contrasts["string_go_minus_self_loop"]["estimate"] == pytest.approx(0.020)
    assert contrasts["dense_minus_self_loop"]["estimate"] == pytest.approx(0.010)
    assert result.registry["multiplicity_family"] == [
        "string_go_minus_self_loop",
        "dense_minus_self_loop",
    ]
    assert result.registry["support_scope"] == (
        "primary_panel_200_hvg_complete_three_arm_matched_support"
    )
    assert "identical" in result.registry["common_self_loop_control"]
    assert all("bh_q_two_contrast_family" in row for row in contrasts.values())


def test_propagation_control_withholds_if_dense_member_of_family_is_missing() -> None:
    frame = _complete_metric_frame()
    missing = frame[
        ~(
            (frame["dataset"] == "adamson")
            & (frame["condition"] == "G0")
            & (frame["hvg"] == 200)
            & (frame["arm"] == "dense")
            & (frame["seed"] == 44)
        )
    ]

    result = propagation_control_release(missing, _mini_protocol())

    assert not result.released
    assert "INCOMPLETE_SEED_OR_SUPPORT_CELL" in set(result.failures["reason_code"])


def test_final_propagation_lock_contains_both_same_self_loop_contrasts() -> None:
    frame = _complete_metric_frame()
    protocol = _mini_protocol()
    primary = primary_hierarchical_release(frame, protocol)
    propagation = propagation_control_release(frame, protocol)
    scale = scale_interaction_release(frame, protocol)
    topology = topology_hierarchical_release(frame, protocol)
    mixed_support = mixed_support_sensitivity_release(frame, protocol)
    coverage = mandatory_fit_coverage_release(
        frame,
        protocol,
        _preflight_summaries(protocol),
        protocol_file_hash=PROTOCOL_HASH,
        expected_code_hash=CODE_HASH,
    )
    released_dummy = AnalysisReleaseResult(
        registry={"analysis_id": "DUMMY", "status": "RELEASED"},
        detail_tables={},
        failures=pd.DataFrame(columns=["reason_code", "detail"]),
    )
    trigger = {
        "status": "PASS",
        "global_triggered": False,
        "dataset_triggers": [],
        "thresholds": {},
        "manifest_hash": "1" * 64,
    }

    def external(registry_id: str) -> dict:
        payload = {"registry_id": registry_id, "status": "RELEASED"}
        payload["registry_hash"] = canonical_sha256(payload)
        return payload

    claim_gate = {
        "registry_id": "CLAIM-CONSEQUENCE-GATE",
        "status": "RELEASED",
        "baseline_absolute_skill_eligible": True,
        "ranking_consequence_eligible": True,
    }
    claim_gate["registry_hash"] = canonical_sha256(claim_gate)

    registry = build_thirteen_result_lock_registry(
        primary=primary,
        propagation=propagation,
        scale=scale,
        topology=topology,
        coverage=coverage,
        conditional=released_dummy,
        secondary=released_dummy,
        mixed_support=mixed_support,
        global_trigger_manifest=trigger,
        external_comparator_registry=external("EXTERNAL-COMPARATOR-VALIDATION"),
        measured_compute_registry=external("MEASURED-COMPUTE"),
        claim_consequence_registry=claim_gate,
    )
    lock = registry["result_locks"]["PROPAGATION-CONTROL"]

    assert registry["result_lock_count"] == 13
    assert [row["contrast"] for row in lock["contrasts"]] == [
        "string_go_minus_self_loop",
        "dense_minus_self_loop",
    ]
    assert lock["multiplicity_family"] == [
        "string_go_minus_self_loop",
        "dense_minus_self_loop",
    ]


def test_mixed_support_600_fit_block_has_scientific_release_and_single_test_family() -> None:
    result = mixed_support_sensitivity_release(_complete_metric_frame(), _mini_protocol())

    assert result.released
    assert result.registry["contrast"] == "combined_minus_dense"
    assert result.registry["mixed_support_new_fits_expected"] == 36
    assert result.registry["mixed_support_new_fits_actual"] == 36
    assert result.registry["dense_reference_evaluations_reused"] == 36
    assert result.registry["total_method_evaluations"] == 72
    assert result.registry["multiplicity_family"] == ["combined_minus_dense"]
    assert set(result.detail_tables) == {"condition_contrasts", "dataset_estimates"}


def test_mixed_support_missing_combined_fold_is_withheld() -> None:
    frame = _complete_metric_frame()
    frame = frame[
        ~(
            (frame["dataset"] == "adamson")
            & (frame["condition"] == "G0")
            & (frame["arm"] == "combined")
            & (frame["seed"] == 42)
        )
    ]
    result = mixed_support_sensitivity_release(frame, _mini_protocol())

    assert not result.released
    assert "INCOMPLETE_SEED_OR_SUPPORT_CELL" in set(result.failures["reason_code"])


def test_decision_registry_releases_only_when_all_mandatory_locks_pass() -> None:
    frame = _complete_metric_frame()
    protocol = _mini_protocol()
    primary = primary_hierarchical_release(frame, protocol)
    scale = scale_interaction_release(frame, protocol)
    topology = topology_hierarchical_release(frame, protocol)
    propagation = propagation_control_release(frame, protocol)
    coverage = mandatory_fit_coverage_release(
        frame,
        protocol,
        _preflight_summaries(protocol),
        protocol_file_hash=PROTOCOL_HASH,
        expected_code_hash=CODE_HASH,
    )
    registry = build_decision_release_registry(primary, scale, topology, propagation, coverage)

    assert registry["package_release_status"] == "RELEASED"
    assert registry["primary_decision"] == "DIRECTIONAL_POSITIVE"
    assert len(registry["decision_registry_hash"]) == 64
    incomplete_primary_frame = frame[
        ~(
            (frame["dataset"] == "adamson")
            & (frame["condition"] == "G0")
            & (frame["hvg"] == 200)
            & (frame["arm"] == "dense")
            & (frame["seed"] == 44)
        )
    ]
    withheld = build_decision_release_registry(
        primary_hierarchical_release(incomplete_primary_frame, protocol),
        scale,
        topology,
        propagation,
        coverage,
    )
    assert withheld["package_release_status"] == "WITHHELD"
    assert withheld["primary_decision"] == "WITHHELD_INCOMPLETE_PRIMARY_ANALYSIS"


def test_coverage_withholds_without_hash_bound_preflight_contracts() -> None:
    result = mandatory_fit_coverage_release(_complete_metric_frame(), _mini_protocol())

    assert not result.released
    assert "PREFLIGHT_RELEASE_CONTRACTS_MISSING" in set(result.failures["reason_code"])


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (0.011, 0.020, "DIRECTIONAL_POSITIVE"),
        (-0.020, -0.001, "DIRECTIONAL_NEGATIVE"),
        (0.001, 0.010, "DIRECTIONAL_POSITIVE"),
        (-0.001, 0.009, "INCONCLUSIVE"),
        (-0.001, 0.011, "INCONCLUSIVE"),
        (0.0, 0.01, "INCONCLUSIVE"),
        (-0.01, 0.0, "INCONCLUSIVE"),
    ],
)
def test_primary_decision_rule_order(low: float, high: float, expected: str) -> None:
    assert primary_decision_rule(low, high) == expected


def test_condition_ranking_overlap_freezes_k_and_deterministic_ties() -> None:
    result = condition_ranking_overlap(_complete_metric_frame(), _mini_protocol(), top_k=2)

    assert len(result) == 4
    assert set(result["top_k"]) == {2}
    assert result["jaccard"].between(0, 1).all()


def test_secondary_release_reports_target_level_guide_multiplicity_sensitivity() -> None:
    frame = _complete_metric_frame()
    frame["fisher_z_pearson"] = np.arctanh(frame["pearson_r"])
    frame["spearman_r"] = frame["pearson_r"]
    frame["mse"] = 1.0 - frame["pearson_r"]
    frame["top20_absolute_delta_jaccard"] = frame["pearson_r"]
    frame.loc[frame["condition"].isin(["G0", "G1"]), "canonical_target_set"] = "T01"
    frame.loc[frame["condition"] == "G2", "canonical_target_set"] = "T2"

    result = secondary_metric_release(frame, _mini_protocol())

    assert result.released
    sensitivity = result.registry["target_level_primary_sensitivity"]
    assert sensitivity["resampling_unit"] == "canonical_target_set_within_dataset"
    assert sensitivity["canonical_target_sets_represented"] == 8
    assert sensitivity["guide_conditions_represented"] == 12
    assert sensitivity["guide_multiplicity_distribution"]["maximum"] == 2
    assert len(result.detail_tables["target_level_primary_sensitivity"]) == 8

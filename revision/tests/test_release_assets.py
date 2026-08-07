from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cbac_revision.artifacts import canonical_sha256, file_sha256
from cbac_revision.errors import RevisionProtocolError
from cbac_revision.release_asset_builder import (
    _records_hash,
    _pass_member_diagnostics,
    assemble_release_asset_tables,
    assert_asset_semantics_match,
)
from cbac_revision.release_assets import (
    ASSET_GROUPS,
    INTERVAL_LABEL,
    render_release_assets,
    validate_release_asset_manifest,
)
from cbac_revision.protocol import load_protocol
from cbac_revision.statistics import AnalysisReleaseResult

DATASETS = ("adamson", "norman", "replogle_k562", "replogle_rpe1")
HASH = "a" * 64


def _released_tables() -> dict[str, pd.DataFrame]:
    primary_rows = [
        {
            "dataset": dataset,
            "cell_line": "RPE1" if dataset == "replogle_rpe1" else "K562",
            "weighting": "dataset_descriptive",
            "estimate": 0.01 + index / 1000,
            "uncertainty_interval_95_low": -0.01,
            "uncertainty_interval_95_high": 0.03,
            "interval_label": INTERVAL_LABEL,
            "n_conditions": 50,
        }
        for index, dataset in enumerate(DATASETS)
    ]
    primary_rows.extend(
        [
            {
                "dataset": "EQUAL_DATASET",
                "cell_line": "K562_3_DATASETS_PLUS_RPE1_1_DATASET",
                "weighting": "equal_dataset",
                "estimate": 0.0115,
                "uncertainty_interval_95_low": 0.001,
                "uncertainty_interval_95_high": 0.022,
                "interval_label": INTERVAL_LABEL,
                "n_conditions": 200,
            },
            {
                "dataset": "EQUAL_CELL_LINE",
                "cell_line": "K562_AND_RPE1_EQUAL_WEIGHT",
                "weighting": "equal_cell_line",
                "estimate": 0.012,
                "uncertainty_interval_95_low": 0.001,
                "uncertainty_interval_95_high": 0.024,
                "interval_label": INTERVAL_LABEL,
                "n_conditions": 200,
            },
        ]
    )
    paired = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "condition": f"C{condition:02d}",
                "target_id": f"G{condition:02d}",
                "delta": (condition - 20) / 1000,
                "n_seeds": 3,
            }
            for dataset in DATASETS
            for condition in range(50)
        ]
    )
    fraction = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "numerator_improved": 29,
                "denominator_conditions": 50,
                "fraction_improved": 0.58,
                "uncertainty_interval_95_low": 0.44,
                "uncertainty_interval_95_high": 0.70,
                "interval_label": INTERVAL_LABEL,
            }
            for dataset in DATASETS
        ]
    )
    target = pd.DataFrame(
        [
            {
                "sensitivity_id": "target_cluster_resampling",
                "status": "MEASURED",
                "estimate": 0.012,
                "uncertainty_interval_95_low": -0.002,
                "uncertainty_interval_95_high": 0.025,
                "interval_label": INTERVAL_LABEL,
                "reason_code": None,
            },
            {
                "sensitivity_id": "leave_target_out",
                "status": "SENSITIVITY_RANGE",
                "estimate": 0.012,
                "uncertainty_interval_95_low": 0.008,
                "uncertainty_interval_95_high": 0.016,
                "interval_label": "DESCRIPTIVE_LEAVE_ONE_CLUSTER_OUT_RANGE",
                "reason_code": None,
            },
            {
                "sensitivity_id": "leave_pathway_out",
                "status": "NOT_AVAILABLE",
                "estimate": None,
                "uncertainty_interval_95_low": None,
                "uncertainty_interval_95_high": None,
                "interval_label": None,
                "reason_code": "PATHWAY_METADATA_NOT_AVAILABLE",
            },
            {
                "sensitivity_id": "shared_control_cell_bootstrap",
                "status": "MEASURED",
                "estimate": 0.011,
                "uncertainty_interval_95_low": 0.001,
                "uncertainty_interval_95_high": 0.023,
                "interval_label": INTERVAL_LABEL,
                "reason_code": None,
            },
        ]
    )
    topology = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "rewire_id": f"string_go_rewire_{index:02d}",
                "graph_index": f"{dataset}::{index:02d}",
                "estimate": 0.005 + index / 10000,
                "uncertainty_interval_95_low": None,
                "uncertainty_interval_95_high": None,
                "interval_label": "DESCRIPTIVE_GRAPH_INSTANCE_POINT_ESTIMATE",
                "degree_sequence_match": True,
                "component_membership_match": True,
                "component_count_match": True,
                "isolates_match": True,
                "changed_edge_fraction": 0.9,
                "source_support_sha256": (f"{DATASETS.index(dataset):x}" * 64)[:64],
                "support_sha256": f"{DATASETS.index(dataset)}{index:02d}".ljust(64, "a")[:64],
                "diagnostics_status": "TOPOLOGY_DIAGNOSTICS_PASS",
            }
            for dataset in DATASETS
            for index in range(1, 11)
        ]
    )
    scale_rows = []
    for dataset in (*DATASETS, "EQUAL_DATASET"):
        for hvg in (500, 1000):
            for space in ("native_hvg", "common_200_gene"):
                overall = dataset == "EQUAL_DATASET"
                scale_rows.append(
                    {
                        "dataset": dataset,
                        "hvg": hvg,
                        "evaluation_space": space,
                        "inference_scope": (
                            "overall_two_member_family" if overall else "dataset_descriptive"
                        ),
                        "contrast": f"{hvg}_minus_200_hvg",
                        "estimate": 0.01,
                        "uncertainty_interval_95_low": -0.01 if overall else None,
                        "uncertainty_interval_95_high": 0.03 if overall else None,
                        "two_sided_p": 0.2 if overall else None,
                        "bh_q_two_member_family": 0.3 if overall else None,
                        "interval_label": (
                            INTERVAL_LABEL if overall else "DESCRIPTIVE_POINT_ESTIMATE_NO_INTERVAL"
                        ),
                    }
                )
    propagation_rows = []
    for dataset in (*DATASETS, "EQUAL_DATASET"):
        for contrast in ("string_go_minus_self_loop", "dense_minus_self_loop"):
            overall = dataset == "EQUAL_DATASET"
            propagation_rows.append(
                {
                    "dataset": dataset,
                    "inference_scope": (
                        "overall_two_member_family" if overall else "dataset_descriptive"
                    ),
                    "contrast": contrast,
                    "estimate": 0.02,
                    "uncertainty_interval_95_low": 0.0 if overall else None,
                    "uncertainty_interval_95_high": 0.04 if overall else None,
                    "two_sided_p": 0.1 if overall else None,
                    "bh_q_two_member_family": 0.15 if overall else None,
                    "interval_label": (
                        INTERVAL_LABEL if overall else "DESCRIPTIVE_POINT_ESTIMATE_NO_INTERVAL"
                    ),
                }
            )
    representativeness_rows = []
    metric_values = {
        "combined_selection_stratum_total_variation": (0.05, 0.10),
        "absolute_standardized_log1p_condition_cell_count_difference": (0.10, 0.25),
        "target_mapping_coverage": (1.0, None),
        "condition_coverage": (0.5, None),
        "cell_coverage": (0.5, None),
    }
    for dataset in DATASETS:
        for metric_id, (value, threshold) in metric_values.items():
            representativeness_rows.append(
                {
                    "dataset": dataset,
                    "analysis": "primary_panel_representativeness",
                    "metric_id": metric_id,
                    "status": "MEASURED",
                    "value": value,
                    "threshold": threshold,
                    "trigger_contribution": False,
                    "global_triggered": False,
                    "panel_size": 50,
                    "overlap_with_legacy": 0,
                    "overlap_with_primary": 0,
                    "estimate": None,
                    "uncertainty_interval_95_low": None,
                    "uncertainty_interval_95_high": None,
                    "interval_label": None,
                    "reason_code": None,
                    "source_records_sha256": HASH,
                }
            )
        representativeness_rows.append(
            {
                "dataset": dataset,
                "analysis": "conditional_nonoverlap_panel",
                "metric_id": "conditional_panel_effect",
                "status": "NOT_TRIGGERED",
                "value": None,
                "threshold": None,
                "trigger_contribution": False,
                "global_triggered": False,
                "panel_size": 50,
                "overlap_with_legacy": 0,
                "overlap_with_primary": 0,
                "estimate": None,
                "uncertainty_interval_95_low": None,
                "uncertainty_interval_95_high": None,
                "interval_label": None,
                "reason_code": "GLOBAL_OR_TRIGGER_FALSE",
                "source_records_sha256": HASH,
            }
        )
    secondary_rows = []
    for metric in ("fisher_z_pearson", "spearman_r", "mse", "top20_absolute_delta_jaccard"):
        secondary_rows.append(
            {
                "dataset": "EQUAL_DATASET",
                "inference_scope": "overall_family_test",
                "metric": metric,
                "model": "string_go_minus_dense",
                "estimate": 0.01,
                "rank": 1,
                "two_sided_p": 0.2,
                "bh_q": 0.3,
                "family_label": "SECONDARY_FOUR_METRIC_FAMILY",
                "family_denominator": 4,
                **_ranking_na(),
            }
        )
    for dataset in DATASETS:
        for model, rank in (("string_go", 1), ("dense", 2)):
            secondary_rows.append(
                {
                    "dataset": dataset,
                    "inference_scope": "dataset_descriptive",
                    "metric": "pearson_r",
                    "model": model,
                    "estimate": 0.4 if model == "string_go" else 0.3,
                    "rank": rank,
                    "two_sided_p": None,
                    "bh_q": None,
                    "family_label": "CONDITION_RANKING_DESCRIPTIVE",
                    "family_denominator": 1,
                    "ranking_analysis_id": "CONDITION-RANKING-OVERLAP",
                    "top_k": 10,
                    "shared_condition_count": 50,
                    "top_k_coverage_complete": True,
                    "top_condition_jaccard": 0.5,
                    "shared_rank_spearman": 0.7,
                    "ranking_status": "MEASURED",
                    "ranking_input_artifact_manifest_sha256": HASH,
                }
            )
        secondary_rows.append(
            {
                "dataset": dataset,
                "inference_scope": "dataset_descriptive",
                "metric": "mixed_combined_minus_dense_pearson_r",
                "model": "combined_minus_dense",
                "estimate": 0.01,
                "rank": 1,
                "two_sided_p": None,
                "bh_q": None,
                "family_label": "MIXED_COMBINED_VS_DENSE_200_SENSITIVITY",
                "family_denominator": 1,
                **_ranking_na(),
            }
        )
    secondary_rows.append(
        {
            "dataset": "EQUAL_DATASET",
            "inference_scope": "overall_family_test",
            "metric": "mixed_combined_minus_dense_pearson_r",
            "model": "combined_minus_dense",
            "estimate": 0.01,
            "rank": 1,
            "two_sided_p": 0.2,
            "bh_q": 0.2,
            "family_label": "MIXED_COMBINED_VS_DENSE_200_SENSITIVITY",
            "family_denominator": 1,
            **_ranking_na(),
        }
    )
    baseline_rows = []
    for dataset in DATASETS:
        for hvg in (200, 500, 1000):
            for condition in range(50):
                for model in ("string_go", "dense"):
                    for baseline in (
                        "zero_control_delta",
                        "training_condition_mean",
                        "deterministic_ridge_linear",
                    ):
                        for metric in ("pearson_r", "mse", "mae"):
                            zero = baseline == "zero_control_delta" and metric == "pearson_r"
                            neural = 0.5 if metric == "pearson_r" else 0.2
                            reference = None if zero else (0.3 if metric == "pearson_r" else 0.4)
                            improvement = (
                                None
                                if zero
                                else (
                                    neural - reference
                                    if metric == "pearson_r"
                                    else reference - neural
                                )
                            )
                            baseline_rows.append(
                                {
                                    "dataset": dataset,
                                    "hvg": hvg,
                                    "panel": "primary",
                                    "condition": f"C{condition:02d}",
                                    "neural_model": model,
                                    "baseline": baseline,
                                    "metric": metric,
                                    "neural_absolute_skill": neural,
                                    "neural_skill_status": "MEASURED",
                                    "baseline_absolute_skill": reference,
                                    "baseline_skill_status": (
                                        "UNDEFINED_CONSTANT_VECTOR" if zero else "MEASURED"
                                    ),
                                    "improvement_over_baseline": improvement,
                                    "improvement_status": "NOT_APPLICABLE" if zero else "MEASURED",
                                    "undefined_reason_code": (
                                        "ZERO_DELTA_PEARSON_UNDEFINED" if zero else None
                                    ),
                                    "baseline_artifact_sha256": HASH,
                                }
                            )
    compute = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "model": "string_go",
                "hvg": 200,
                "attempts": 150,
                "successes": 150,
                "failures": 0,
                "training_seconds": 100.0,
                "inference_seconds": 10.0,
                "peak_device_bytes": 1000,
                "status": "MEASURED",
            }
            for dataset in DATASETS
        ]
    )
    external = pd.DataFrame(
        [
            {
                "comparator": comparator,
                "diagnostic_id": "validity_decision",
                "status": "EXCLUDED",
                "value": None,
                "reference": "scientific_validity_contract",
                "tolerance": None,
                "reason_code": "CLAIM_DELETED_NO_VALID_EVIDENCE",
                "evidence_sha256": HASH,
            }
            for comparator in ("GEARS", "scGPT", "Geneformer")
        ]
    )
    return {
        "primary_summary_and_forest": pd.DataFrame(primary_rows),
        "paired_condition_distribution": paired,
        "fraction_improved_interval": fraction,
        "target_sensitivity": target,
        "topology_instances_and_diagnostics": topology,
        "scale_native_and_common200": pd.DataFrame(scale_rows),
        "propagation": pd.DataFrame(propagation_rows),
        "representativeness_and_conditional": pd.DataFrame(representativeness_rows),
        "secondary_ranking_and_multiplicity": pd.DataFrame(secondary_rows),
        "baseline_absolute_skill": pd.DataFrame(baseline_rows),
        "compute_and_failure_denominators": compute,
        "external_diagnostics_and_exclusions": external,
    }


def _ranking_na() -> dict[str, object]:
    return {
        "ranking_analysis_id": None,
        "top_k": None,
        "shared_condition_count": None,
        "top_k_coverage_complete": None,
        "top_condition_jaccard": None,
        "shared_rank_spearman": None,
        "ranking_status": None,
        "ranking_input_artifact_manifest_sha256": None,
    }


def _released_analysis_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    protocol_path = Path(__file__).parents[1] / "protocol.yaml"
    protocol = load_protocol(protocol_path)
    metric_rows = []
    artifact_index = 1
    for dataset in DATASETS:
        for hvg in (200, 500, 1000):
            for condition_index in range(50):
                condition = f"C{condition_index:02d}"
                for arm in ("string_go", "dense"):
                    for seed in (42, 43, 44):
                        string_go = arm == "string_go"
                        metric_rows.append(
                            {
                                "dataset": dataset,
                                "hvg": hvg,
                                "panel": "primary",
                                "arm": arm,
                                "condition": condition,
                                "seed": seed,
                                "pearson_r": 0.50 if string_go else 0.40,
                                "mse": 0.20 if string_go else 0.30,
                                "mae": 0.30 if string_go else 0.40,
                                "artifact_hash": f"{artifact_index:064x}",
                            }
                        )
                        artifact_index += 1
    metrics = pd.DataFrame(metric_rows)
    condition_contrasts = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "condition": f"C{index:02d}",
                "canonical_target_set": f"G{index:02d}",
                "delta": 0.01 + index / 10000,
            }
            for dataset in DATASETS
            for index in range(50)
        ]
    )
    dataset_estimates = [
        {
            "dataset": dataset,
            "estimate": 0.012,
            "uncertainty_interval_95_low": 0.001,
            "uncertainty_interval_95_high": 0.024,
            "n_conditions": 50,
            "fraction_improved_uncertainty_interval_95_low": 0.85,
            "fraction_improved_uncertainty_interval_95_high": 1.0,
        }
        for dataset in DATASETS
    ]
    primary = AnalysisReleaseResult(
        registry={
            "analysis_id": "PRIMARY",
            "status": "RELEASED",
            "estimate": 0.012,
            "uncertainty_interval_95_low": 0.001,
            "uncertainty_interval_95_high": 0.024,
            "dataset_estimates": dataset_estimates,
            "equal_cell_line_sensitivity": {
                "estimate": 0.011,
                "uncertainty_interval_95_low": 0.001,
                "uncertainty_interval_95_high": 0.023,
            },
            "target_cluster_resampling": {
                "estimate": 0.012,
                "uncertainty_interval_95_low": 0.001,
                "uncertainty_interval_95_high": 0.024,
            },
            "leave_target_out": {"minimum_estimate": 0.008, "maximum_estimate": 0.016},
            "leave_pathway_out": {"reason": "PATHWAY_METADATA_NOT_AVAILABLE"},
            "shared_control_cell_bootstrap": {
                "status": "MEASURED",
                "estimate": 0.011,
                "uncertainty_interval_95_low": 0.001,
                "uncertainty_interval_95_high": 0.023,
            },
        },
        detail_tables={
            "condition_contrasts": condition_contrasts,
            "leave_pathway_out": pd.DataFrame([{"status": "NOT_AVAILABLE", "estimate": None}]),
        },
        failures=pd.DataFrame(),
    )
    topology_estimates = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "rewire_arm": f"string_go_rewire_{index:02d}",
                "local_graph_condition_mean_contrast": 0.005 + index / 10000,
            }
            for dataset in DATASETS
            for index in range(1, 11)
        ]
    )
    topology_diagnostics = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "arm": f"string_go_rewire_{index:02d}",
                "local_graph_index": index,
                "degree_sequence_sha256": "a" * 64,
                "source_degree_sequence_sha256": "a" * 64,
                "component_partition_sha256": "b" * 64,
                "source_component_partition_sha256": "b" * 64,
                "n_connected_components": 1,
                "source_n_connected_components": 1,
                "n_isolates": 0,
                "source_n_isolates": 0,
                "swapped_edge_fraction": 0.9,
                "source_support_sha256": f"{DATASETS.index(dataset) + 1:064x}",
                "support_sha256": canonical_sha256({"dataset": dataset, "index": index}),
            }
            for dataset in DATASETS
            for index in range(1, 11)
        ]
    )
    topology = AnalysisReleaseResult(
        registry={"analysis_id": "TOPOLOGY-NULL", "status": "RELEASED"},
        detail_tables={
            "graph_instance_estimates": topology_estimates,
            "graph_diagnostics": topology_diagnostics,
        },
        failures=pd.DataFrame(),
    )
    scale_contrasts = []
    for hvg in (500, 1000):
        for evaluation_space in ("native_hvg", "common_200_gene"):
            scale_contrasts.append(
                {
                    "contrast": f"{hvg}_minus_200_hvg",
                    "evaluation_space": evaluation_space,
                    "estimate": 0.01,
                    "uncertainty_interval_95_low": -0.01,
                    "uncertainty_interval_95_high": 0.03,
                    "two_sided_bootstrap_p": 0.2,
                    "bh_q_two_contrast_family": 0.3,
                    "dataset_estimates": [
                        {"dataset": dataset, "estimate": 0.01} for dataset in DATASETS
                    ],
                }
            )
    scale = AnalysisReleaseResult(
        registry={
            "analysis_id": "SCALE-INTERACTION",
            "status": "RELEASED",
            "contrasts": scale_contrasts,
        },
        detail_tables={},
        failures=pd.DataFrame(),
    )
    propagation = AnalysisReleaseResult(
        registry={
            "analysis_id": "PROPAGATION-CONTROL",
            "status": "RELEASED",
            "contrasts": [
                {
                    "contrast": contrast,
                    "estimate": 0.02,
                    "uncertainty_interval_95_low": 0.0,
                    "uncertainty_interval_95_high": 0.04,
                    "two_sided_bootstrap_p": 0.1,
                    "bh_q_two_contrast_family": 0.15,
                    "dataset_estimates": [
                        {"dataset": dataset, "estimate": 0.02} for dataset in DATASETS
                    ],
                }
                for contrast in ("string_go_minus_self_loop", "dense_minus_self_loop")
            ],
        },
        detail_tables={},
        failures=pd.DataFrame(),
    )
    conditional = AnalysisReleaseResult(
        registry={"analysis_id": "CONDITIONAL-PANEL", "status": "RELEASED"},
        detail_tables={},
        failures=pd.DataFrame(),
    )
    ranking = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "top_k": 10,
                "shared_condition_count": 50,
                "jaccard": 0.5,
                "shared_rank_spearman": 0.7,
                "shared_rank_spearman_status": "MEASURED",
            }
            for dataset in DATASETS
        ]
    )
    secondary = AnalysisReleaseResult(
        registry={
            "analysis_id": "SECONDARY-METRICS",
            "status": "RELEASED",
            "metrics": [
                {
                    "metric": metric,
                    "benefit_direction": "string_go_minus_dense",
                    "estimate": 0.01,
                    "centered_null_bootstrap_p": 0.2,
                    "bh_q_four_metric_family": 0.3,
                }
                for metric in (
                    "fisher_z_pearson",
                    "spearman_r",
                    "mse",
                    "top20_absolute_delta_jaccard",
                )
            ],
        },
        detail_tables={"condition_ranking_overlap": ranking},
        failures=pd.DataFrame(),
    )
    mixed = AnalysisReleaseResult(
        registry={
            "analysis_id": "MIXED-SUPPORT-SENSITIVITY",
            "status": "RELEASED",
            "estimate": 0.01,
            "two_sided_centered_null_bootstrap_p": 0.2,
            "dataset_estimates": [{"dataset": dataset, "estimate": 0.01} for dataset in DATASETS],
        },
        detail_tables={},
        failures=pd.DataFrame(),
    )

    preflight_sources = []
    trigger_rows = []
    for dataset in DATASETS:
        directory = tmp_path / f"preflight_{dataset}"
        directory.mkdir()
        representativeness = pd.DataFrame(
            [
                {
                    "panel": "primary",
                    "variable": variable,
                    "statistic": statistic,
                    "status": "MEASURED",
                    "value": value,
                }
                for variable, statistic, value in (
                    (
                        "combined_selection_stratum",
                        "variable_total_variation_distance",
                        0.05,
                    ),
                    (
                        "log1p_condition_cell_count",
                        "standardized_mean_difference",
                        0.10,
                    ),
                    ("target_mapping", "mapping_coverage", 1.0),
                    ("panel_coverage", "condition_coverage", 0.5),
                    ("panel_coverage", "cell_coverage", 0.5),
                )
            ]
        )
        table_path = directory / "panel_representativeness.csv"
        representativeness.to_csv(table_path, index=False)
        source_hash = _records_hash(pd.read_csv(table_path))
        primary_conditions = [f"P{index:02d}" for index in range(50)]
        sensitivity_conditions = [f"S{index:02d}" for index in range(50)]
        legacy_conditions = [f"L{index:02d}" for index in range(50)]
        summary = {
            "dataset": dataset,
            "summary_hash": canonical_sha256({"dataset": dataset}),
            "panel_representativeness_table_hash": source_hash,
            "condition_panel_manifest": {
                "primary": {"ordered_canonical_condition_ids": primary_conditions},
                "sensitivity": {"ordered_canonical_condition_ids": sensitivity_conditions},
                "legacy_first50": {"ordered_condition_ids": legacy_conditions},
            },
        }
        summary_path = directory / "preflight_summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        preflight_sources.append((summary_path, summary))
        trigger_rows.append(
            {
                "dataset": dataset,
                "trigger_total_variation_distance": 0.05,
                "trigger_abs_standardized_log1p_cell_count_difference": 0.10,
                "panel_representativeness_table_hash": source_hash,
            }
        )
    global_trigger = {
        "global_triggered": False,
        "thresholds": {
            "total_variation_distance_exclusive_gt": 0.10,
            "absolute_standardized_log1p_cell_count_difference_exclusive_gt": 0.25,
        },
        "dataset_triggers": trigger_rows,
    }
    global_trigger["manifest_hash"] = canonical_sha256(global_trigger)

    baseline_payloads: dict[Path, dict[str, object]] = {}
    baseline_paths = []
    for dataset in DATASETS:
        for hvg in (200, 500, 1000):
            for condition_index in range(50):
                condition = f"C{condition_index:02d}"
                path = (tmp_path / "baselines" / f"{dataset}_{hvg}_{condition}.json.gz").resolve()
                baseline_paths.append(path)
                baseline_payloads[path] = {
                    "identity": {
                        "dataset": dataset,
                        "hvg": hvg,
                        "panel": "primary",
                        "condition": condition,
                    },
                    "absolute_skill": {
                        "zero_control_delta": {"pearson_r": None, "mse": 0.6, "mae": 0.7},
                        "training_condition_mean": {
                            "pearson_r": 0.2,
                            "mse": 0.5,
                            "mae": 0.6,
                        },
                        "deterministic_ridge_linear": {
                            "pearson_r": 0.3,
                            "mse": 0.4,
                            "mae": 0.5,
                        },
                    },
                    "artifact_sha256": canonical_sha256(
                        {"dataset": dataset, "hvg": hvg, "condition": condition}
                    ),
                }
    monkeypatch.setattr(
        "cbac_revision.release_asset_builder.read_baseline_artifact",
        lambda path: baseline_payloads[Path(path).resolve()],
    )
    compute_records = [
        {
            "dataset": dataset,
            "model": "string_go",
            "hvg": 200,
            "attempts": 150,
            "successes": 150,
            "failures": 0,
            "training_seconds": 100.0,
            "inference_seconds": 10.0,
            "peak_device_bytes": 1000,
            "status": "MEASURED",
        }
        for dataset in DATASETS
    ]
    compute = {
        "status": "RELEASED",
        "attempt_group_summaries": compute_records,
        "attempt_group_summaries_sha256": canonical_sha256(compute_records),
        "registry_hash": canonical_sha256({"compute": compute_records}),
    }
    excluded_members = {
        comparator: {
            "comparator": comparator,
            "decision": "EXCLUDED",
            "reason_code": "CLAIM_DELETED_NO_VALID_EVIDENCE",
        }
        for comparator in ("GEARS", "scGPT", "Geneformer")
    }
    external = {
        "status": "RELEASED",
        "members": excluded_members,
        "member_decisions": {comparator: "EXCLUDED" for comparator in excluded_members},
    }
    external["registry_hash"] = canonical_sha256(external)
    return {
        "protocol": protocol,
        "protocol_file_sha256": file_sha256(protocol_path),
        "metrics": metrics,
        "primary": primary,
        "scale": scale,
        "topology": topology,
        "propagation": propagation,
        "conditional": conditional,
        "secondary": secondary,
        "mixed_support": mixed,
        "preflight_sources": preflight_sources,
        "global_trigger_manifest": global_trigger,
        "baseline_artifact_paths": baseline_paths,
        "measured_compute_registry": compute,
        "external_registry": external,
    }


def test_released_assets_round_trip_semantics_and_consumer_compile(tmp_path: Path) -> None:
    tables = _released_tables()
    output = tmp_path / "generated"
    manifest = render_release_assets(
        output,
        mode="released",
        tables=tables,
        source_bindings={"analysis_registry": HASH},
    )
    manifest_path = output / "release_asset_manifest.json"
    validated = validate_release_asset_manifest(
        manifest_path,
        expected_manifest_file_sha256=file_sha256(manifest_path),
        expected_source_bindings={"analysis_registry": HASH},
    )
    assert len(validated["entries"]) == 49
    assert set(validated["semantic_records_hashes"]) == set(ASSET_GROUPS)
    assert_asset_semantics_match(tables, validated["semantic_records_hashes"])
    include = (output / "released_results_include.tex").read_text(encoding="utf-8")
    for consumer in ("main", "supplement", "response"):
        assert f"CBACReleaseConsumer{consumer.title()}" in include

    engine = shutil.which("xelatex")
    if engine is None:
        pytest.skip("XeLaTeX unavailable")
    for consumer in ("main", "supplement", "response"):
        source = tmp_path / f"{consumer}.tex"
        source.write_text(
            "\\documentclass{article}\n"
            "\\usepackage[margin=12mm]{geometry}\n"
            "\\usepackage{graphicx}\n"
            "\\def\\CBACReleaseConsumer{" + consumer + "}\n"
            "\\def\\CBACReleaseAssetRoot{" + output.as_posix() + "}\n"
            "\\begin{document}\n"
            "\\input{" + (output / "released_results_include.tex").as_posix() + "}\n"
            "\\end{document}\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [engine, "-interaction=nonstopmode", "-halt-on-error", source.name],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stdout[-4000:]
        log = source.with_suffix(".log").read_text(encoding="utf-8", errors="replace")
        assert "Overfull \\hbox" not in log
        assert "Undefined control sequence" not in log


def test_analysis_registries_deterministically_build_all_reader_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _released_analysis_inputs(tmp_path, monkeypatch)
    tables, bindings = assemble_release_asset_tables(**inputs)
    assert set(tables) == set(ASSET_GROUPS)
    assert len(tables["baseline_absolute_skill"]) == 10_800
    assert set(
        tables["secondary_ranking_and_multiplicity"].loc[
            lambda frame: frame["family_label"] == "MIXED_COMBINED_VS_DENSE_200_SENSITIVITY",
            "dataset",
        ]
    ) == {*DATASETS, "EQUAL_DATASET"}
    output = tmp_path / "analysis_derived"
    manifest = render_release_assets(
        output,
        mode="released",
        tables=tables,
        source_bindings=bindings,
    )
    assert_asset_semantics_match(tables, manifest["semantic_records_hashes"])

    tables["primary_summary_and_forest"].loc[
        tables["primary_summary_and_forest"]["dataset"] == "EQUAL_DATASET", "estimate"
    ] = 0.99
    with pytest.raises(RevisionProtocolError, match="ANALYSIS_SEMANTIC_MISMATCH"):
        assert_asset_semantics_match(tables, manifest["semantic_records_hashes"])


def test_author_review_include_consumes_all_twelve_placeholder_groups(tmp_path: Path) -> None:
    output = tmp_path / "author_review"
    render_release_assets(output, mode="author-review")
    include = (output / "released_results_include.tex").read_text(encoding="utf-8")
    assert include.count(r"\input{\CBACReleaseAssetRoot/") == len(ASSET_GROUPS)
    for group in ASSET_GROUPS:
        assert rf"\input{{\CBACReleaseAssetRoot/release_assets/{group}.tex}}" in include


def test_unrelated_valid_asset_bundle_and_fabricated_external_pass_are_rejected(
    tmp_path: Path,
) -> None:
    expected = _released_tables()
    unrelated = _released_tables()
    unrelated["primary_summary_and_forest"].loc[
        unrelated["primary_summary_and_forest"]["dataset"] == "EQUAL_DATASET", "estimate"
    ] = 0.02
    output = tmp_path / "forged"
    manifest = render_release_assets(
        output,
        mode="released",
        tables=unrelated,
        source_bindings={"analysis_registry": HASH},
    )
    with pytest.raises(RevisionProtocolError, match="ANALYSIS_SEMANTIC_MISMATCH"):
        assert_asset_semantics_match(expected, manifest["semantic_records_hashes"])

    fabricated = _released_tables()
    fabricated["external_diagnostics_and_exclusions"] = pd.DataFrame(
        [
            {
                "comparator": comparator,
                "diagnostic_id": "validity_decision",
                "status": "PASS",
                "value": None,
                "reference": "self_asserted",
                "tolerance": None,
                "reason_code": None,
                "evidence_sha256": HASH,
            }
            for comparator in ("GEARS", "scGPT", "Geneformer")
        ]
    )
    with pytest.raises(RevisionProtocolError, match="PASS_DIAGNOSTICS_INVALID"):
        render_release_assets(
            tmp_path / "fabricated",
            mode="released",
            tables=fabricated,
            source_bindings={"analysis_registry": HASH},
        )


def test_gears_reader_diagnostics_are_derived_and_reject_nonimprovement() -> None:
    candidate = {
        "metrics": {"pearson": 0.5},
        "baseline_metrics": {"pearson": 0.1},
        "directional_baseline_improvements": {"pearson": 0.4},
        "minimum_directional_baseline_improvement": 0.4,
        "training_loss_improvement": 0.9,
        "validation_loss_improvement": 0.8,
        "prediction_standard_deviation": 0.2,
        "truth_standard_deviation": 0.3,
        "loss_history_length": 20,
        "gene_order_sha256": "b" * 64,
        "perturbable_gene_order_sha256": "c" * 64,
        "condition_target_mapping_sha256": "d" * 64,
        "target_audit_sha256": "e" * 64,
        "split_membership_sha256": "f" * 64,
        "vector_row_conditions_sha256": "1" * 64,
        "target_disjoint_status": "NOT_APPLICABLE_TEST_ONLY_POSITIVE_CONTROL_SPLIT",
        "target_disjoint_evidence_sha256": "2" * 64,
        "diagnostic_status": "PASS",
    }
    member = {
        "detached_validator_registry_status": "RELEASED",
        "comparisons": [
            {
                "metric": "pearson",
                "candidate_observed": 0.5,
                "expected": 0.5,
                "allowed_difference": 0.01,
                "status": "PASS",
            }
        ],
        "validated_bindings": {
            "repository_url": "https://github.com/snap-stanford/GEARS",
            "commit": "1" * 40,
            "dataset_sha256": HASH,
            "split_name": "simulation",
            "split_seed": 1,
            "training_config_hash": HASH,
            "metric_source_sha256": HASH,
            "metric_function": "gears.inference.compute_metrics",
            "trust_anchor_sha256": HASH,
        },
        "semantic_diagnostics": {"candidate": candidate},
    }
    rows = _pass_member_diagnostics("GEARS", member, HASH)
    by_id = {row["diagnostic_id"]: row for row in rows}
    for diagnostic in (
        "prediction_nondegeneracy",
        "baseline_superiority",
        "training_loss_improvement",
        "validation_loss_improvement",
    ):
        assert float(by_id[diagnostic]["value"]) > 0
    assert by_id["gene_order_binding"]["value"] == "b" * 64
    assert by_id["target_mapping_binding"]["value"] == "d" * 64
    assert by_id["vector_row_condition_binding"]["value"] == "1" * 64
    assert by_id["split_target_disjoint_binding"]["value"] == "2" * 64
    assert by_id["split_target_disjoint_binding"]["status"] == "MEASURED"

    candidate["validation_loss_improvement"] = 0.0
    with pytest.raises(RevisionProtocolError, match="GEARS_DIAGNOSTIC_GATE_INVALID"):
        _pass_member_diagnostics("GEARS", member, HASH)

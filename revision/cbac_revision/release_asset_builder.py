"""Deterministically derive reader-facing release tables from validated analysis evidence."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .artifacts import canonical_sha256
from .baselines import BASELINE_METHODS, read_baseline_artifact
from .errors import RevisionProtocolError
from .release_assets import ASSET_GROUPS, GROUP_COLUMNS, INTERVAL_LABEL, _validate_group_table


def assemble_release_asset_tables(
    *,
    protocol: Mapping[str, Any],
    protocol_file_sha256: str,
    metrics: pd.DataFrame,
    primary: Any,
    scale: Any,
    topology: Any,
    propagation: Any,
    conditional: Any,
    secondary: Any,
    mixed_support: Any,
    preflight_sources: Sequence[tuple[Path, Mapping[str, Any]]],
    global_trigger_manifest: Mapping[str, Any],
    baseline_artifact_paths: Sequence[Path],
    measured_compute_registry: Mapping[str, Any],
    external_registry: Mapping[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Build all 12 tables from the same registries and retained detail evidence."""

    datasets = tuple(sorted(str(value) for value in protocol["datasets"]))
    if not re.fullmatch(r"[0-9a-f]{64}", protocol_file_sha256):
        raise RevisionProtocolError("[RELEASE_ASSET_PROTOCOL_FILE_HASH_INVALID]")
    analyses = (primary, scale, topology, propagation, conditional, secondary, mixed_support)
    if any(not result.released for result in analyses):
        withheld = [
            result.registry.get("analysis_id") for result in analyses if not result.released
        ]
        raise RevisionProtocolError(f"[RELEASE_ASSET_ANALYSIS_WITHHELD] {withheld}")
    if measured_compute_registry.get("status") != "RELEASED":
        raise RevisionProtocolError("[RELEASE_ASSET_COMPUTE_WITHHELD]")
    if external_registry.get("status") != "RELEASED":
        raise RevisionProtocolError("[RELEASE_ASSET_EXTERNAL_WITHHELD]")

    tables = {
        "primary_summary_and_forest": _primary_table(primary, protocol),
        "paired_condition_distribution": _paired_table(primary),
        "fraction_improved_interval": _fraction_table(primary),
        "target_sensitivity": _target_sensitivity_table(primary),
        "topology_instances_and_diagnostics": _topology_table(topology),
        "scale_native_and_common200": _scale_table(scale),
        "propagation": _propagation_table(propagation),
        "representativeness_and_conditional": _representativeness_table(
            preflight_sources,
            global_trigger_manifest,
            conditional,
            protocol,
        ),
        "secondary_ranking_and_multiplicity": _secondary_table(
            secondary,
            mixed_support,
            metrics,
        ),
        "baseline_absolute_skill": _baseline_table(
            baseline_artifact_paths,
            metrics,
        ),
        "compute_and_failure_denominators": _compute_table(measured_compute_registry),
        "external_diagnostics_and_exclusions": _external_table(external_registry),
    }
    if set(tables) != set(ASSET_GROUPS):
        raise RevisionProtocolError("[RELEASE_ASSET_BUILDER_GROUP_SET_INVALID]")
    ordered: dict[str, pd.DataFrame] = {}
    for group in ASSET_GROUPS:
        frame = tables[group].loc[:, list(GROUP_COLUMNS[group])]
        ordered[group] = _validate_group_table(group, frame, datasets)
    bindings = {
        "protocol": protocol_file_sha256,
        "fold_metrics": _records_hash(metrics.sort_values(_metric_sort_columns(metrics))),
        "global_trigger_manifest": str(global_trigger_manifest["manifest_hash"]),
        "preflight_summaries": canonical_sha256(
            sorted(str(summary.get("summary_hash")) for _, summary in preflight_sources)
        ),
        "baseline_artifacts": canonical_sha256(
            sorted(
                str(read_baseline_artifact(path)["artifact_sha256"])
                for path in baseline_artifact_paths
            )
        ),
        "measured_compute_registry": str(measured_compute_registry["registry_hash"]),
        "external_comparator_registry": str(external_registry["registry_hash"]),
    }
    for result in analyses:
        bindings[f"analysis_{result.registry['analysis_id'].casefold()}"] = canonical_sha256(
            result.registry
        )
    return ordered, dict(sorted(bindings.items()))


def assert_asset_semantics_match(
    tables: Mapping[str, pd.DataFrame], semantic_records_hashes: Mapping[str, str]
) -> None:
    """Reject a validly hashed asset package whose records do not equal computed analyses."""

    if set(tables) != set(ASSET_GROUPS) or set(semantic_records_hashes) != set(ASSET_GROUPS):
        raise RevisionProtocolError("[RELEASE_ASSET_SEMANTIC_GROUP_SET_MISMATCH]")
    mismatches = [
        group
        for group in ASSET_GROUPS
        if _records_hash(tables[group]) != semantic_records_hashes[group]
    ]
    if mismatches:
        raise RevisionProtocolError(f"[RELEASE_ASSET_ANALYSIS_SEMANTIC_MISMATCH] {mismatches}")


def _primary_table(primary: Any, protocol: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    cell_lines = {
        str(dataset): str(values["cell_line"]) for dataset, values in protocol["datasets"].items()
    }
    for record in primary.registry["dataset_estimates"]:
        rows.append(
            {
                "dataset": record["dataset"],
                "cell_line": cell_lines[str(record["dataset"])],
                "weighting": "dataset_descriptive",
                "estimate": record["estimate"],
                "uncertainty_interval_95_low": record["uncertainty_interval_95_low"],
                "uncertainty_interval_95_high": record["uncertainty_interval_95_high"],
                "interval_label": INTERVAL_LABEL,
                "n_conditions": record["n_conditions"],
            }
        )
    total = sum(int(row["n_conditions"]) for row in rows)
    rows.append(
        {
            "dataset": "EQUAL_DATASET",
            "cell_line": "K562_3_DATASETS_PLUS_RPE1_1_DATASET",
            "weighting": "equal_dataset",
            "estimate": primary.registry["estimate"],
            "uncertainty_interval_95_low": primary.registry["uncertainty_interval_95_low"],
            "uncertainty_interval_95_high": primary.registry["uncertainty_interval_95_high"],
            "interval_label": INTERVAL_LABEL,
            "n_conditions": total,
        }
    )
    sensitivity = primary.registry["equal_cell_line_sensitivity"]
    rows.append(
        {
            "dataset": "EQUAL_CELL_LINE",
            "cell_line": "K562_AND_RPE1_EQUAL_WEIGHT",
            "weighting": "equal_cell_line",
            "estimate": sensitivity["estimate"],
            "uncertainty_interval_95_low": sensitivity["uncertainty_interval_95_low"],
            "uncertainty_interval_95_high": sensitivity["uncertainty_interval_95_high"],
            "interval_label": INTERVAL_LABEL,
            "n_conditions": total,
        }
    )
    return pd.DataFrame(rows)


def _paired_table(primary: Any) -> pd.DataFrame:
    frame = primary.detail_tables["condition_contrasts"].copy()
    frame["target_id"] = frame["canonical_target_set"].astype(str)
    frame["n_seeds"] = 3
    return frame[["dataset", "condition", "target_id", "delta", "n_seeds"]]


def _fraction_table(primary: Any) -> pd.DataFrame:
    contrasts = primary.detail_tables["condition_contrasts"]
    intervals = {str(row["dataset"]): row for row in primary.registry["dataset_estimates"]}
    rows = []
    for dataset, group in contrasts.groupby("dataset", sort=True):
        improved = int((group["delta"].astype(float) > 0).sum())
        local = intervals[str(dataset)]
        rows.append(
            {
                "dataset": dataset,
                "numerator_improved": improved,
                "denominator_conditions": len(group),
                "fraction_improved": improved / len(group),
                "uncertainty_interval_95_low": local[
                    "fraction_improved_uncertainty_interval_95_low"
                ],
                "uncertainty_interval_95_high": local[
                    "fraction_improved_uncertainty_interval_95_high"
                ],
                "interval_label": INTERVAL_LABEL,
            }
        )
    return pd.DataFrame(rows)


def _target_sensitivity_table(primary: Any) -> pd.DataFrame:
    registry = primary.registry
    target = registry["target_cluster_resampling"]
    leave_target = registry["leave_target_out"]
    rows = [
        {
            "sensitivity_id": "target_cluster_resampling",
            "status": "MEASURED",
            "estimate": target["estimate"],
            "uncertainty_interval_95_low": target["uncertainty_interval_95_low"],
            "uncertainty_interval_95_high": target["uncertainty_interval_95_high"],
            "interval_label": INTERVAL_LABEL,
            "reason_code": None,
        },
        {
            "sensitivity_id": "leave_target_out",
            "status": "SENSITIVITY_RANGE",
            "estimate": registry["estimate"],
            "uncertainty_interval_95_low": leave_target["minimum_estimate"],
            "uncertainty_interval_95_high": leave_target["maximum_estimate"],
            "interval_label": "DESCRIPTIVE_LEAVE_ONE_CLUSTER_OUT_RANGE",
            "reason_code": None,
        },
    ]
    pathway = primary.detail_tables["leave_pathway_out"]
    measured_pathway = pathway[pathway["status"] == "MEASURED"]
    if len(measured_pathway):
        values = measured_pathway["estimate"].astype(float)
        rows.append(
            {
                "sensitivity_id": "leave_pathway_out",
                "status": "SENSITIVITY_RANGE",
                "estimate": registry["estimate"],
                "uncertainty_interval_95_low": float(values.min()),
                "uncertainty_interval_95_high": float(values.max()),
                "interval_label": "DESCRIPTIVE_LEAVE_ONE_CLUSTER_OUT_RANGE",
                "reason_code": None,
            }
        )
    else:
        rows.append(
            {
                "sensitivity_id": "leave_pathway_out",
                "status": "NOT_AVAILABLE",
                "estimate": None,
                "uncertainty_interval_95_low": None,
                "uncertainty_interval_95_high": None,
                "interval_label": None,
                "reason_code": registry["leave_pathway_out"]["reason"],
            }
        )
    shared = registry["shared_control_cell_bootstrap"]
    if shared.get("status") != "MEASURED":
        raise RevisionProtocolError("[RELEASE_ASSET_SHARED_CONTROL_BOOTSTRAP_NOT_MEASURED]")
    rows.append(
        {
            "sensitivity_id": "shared_control_cell_bootstrap",
            "status": "MEASURED",
            "estimate": shared["estimate"],
            "uncertainty_interval_95_low": shared["uncertainty_interval_95_low"],
            "uncertainty_interval_95_high": shared["uncertainty_interval_95_high"],
            "interval_label": INTERVAL_LABEL,
            "reason_code": None,
        }
    )
    return pd.DataFrame(rows)


def _topology_table(topology: Any) -> pd.DataFrame:
    estimates = topology.detail_tables["graph_instance_estimates"]
    diagnostics = topology.detail_tables["graph_diagnostics"]
    rewires = diagnostics[diagnostics["arm"].str.startswith("string_go_rewire_")].copy()
    merged = estimates.merge(
        rewires,
        left_on=["dataset", "rewire_arm"],
        right_on=["dataset", "arm"],
        validate="one_to_one",
    )
    return pd.DataFrame(
        {
            "dataset": merged["dataset"],
            "rewire_id": merged["rewire_arm"],
            "graph_index": merged.apply(
                lambda row: f"{row['dataset']}::{int(row['local_graph_index']):02d}", axis=1
            ),
            "estimate": merged["local_graph_condition_mean_contrast"],
            "uncertainty_interval_95_low": np.nan,
            "uncertainty_interval_95_high": np.nan,
            "interval_label": "DESCRIPTIVE_GRAPH_INSTANCE_POINT_ESTIMATE",
            "degree_sequence_match": merged["degree_sequence_sha256"]
            == merged["source_degree_sequence_sha256"],
            "component_membership_match": merged["component_partition_sha256"]
            == merged["source_component_partition_sha256"],
            "component_count_match": merged["n_connected_components"]
            == merged["source_n_connected_components"],
            "isolates_match": merged["n_isolates"] == merged["source_n_isolates"],
            "changed_edge_fraction": merged["swapped_edge_fraction"],
            "source_support_sha256": merged["source_support_sha256"],
            "support_sha256": merged["support_sha256"],
            "diagnostics_status": "TOPOLOGY_DIAGNOSTICS_PASS",
        }
    )


def _scale_table(scale: Any) -> pd.DataFrame:
    rows = []
    for contrast in scale.registry["contrasts"]:
        hvg = int(str(contrast["contrast"]).split("_")[0])
        rows.append(
            {
                "dataset": "EQUAL_DATASET",
                "hvg": hvg,
                "evaluation_space": contrast["evaluation_space"],
                "inference_scope": "overall_two_member_family",
                "contrast": contrast["contrast"],
                "estimate": contrast["estimate"],
                "uncertainty_interval_95_low": contrast["uncertainty_interval_95_low"],
                "uncertainty_interval_95_high": contrast["uncertainty_interval_95_high"],
                "two_sided_p": contrast["two_sided_bootstrap_p"],
                "bh_q_two_member_family": contrast["bh_q_two_contrast_family"],
                "interval_label": INTERVAL_LABEL,
            }
        )
        for local in contrast["dataset_estimates"]:
            rows.append(
                {
                    "dataset": local["dataset"],
                    "hvg": hvg,
                    "evaluation_space": contrast["evaluation_space"],
                    "inference_scope": "dataset_descriptive",
                    "contrast": contrast["contrast"],
                    "estimate": local["estimate"],
                    "uncertainty_interval_95_low": None,
                    "uncertainty_interval_95_high": None,
                    "two_sided_p": None,
                    "bh_q_two_member_family": None,
                    "interval_label": "DESCRIPTIVE_POINT_ESTIMATE_NO_INTERVAL",
                }
            )
    return pd.DataFrame(rows)


def _propagation_table(propagation: Any) -> pd.DataFrame:
    rows = []
    for contrast in propagation.registry["contrasts"]:
        rows.append(
            {
                "dataset": "EQUAL_DATASET",
                "inference_scope": "overall_two_member_family",
                "contrast": contrast["contrast"],
                "estimate": contrast["estimate"],
                "uncertainty_interval_95_low": contrast["uncertainty_interval_95_low"],
                "uncertainty_interval_95_high": contrast["uncertainty_interval_95_high"],
                "two_sided_p": contrast["two_sided_bootstrap_p"],
                "bh_q_two_member_family": contrast["bh_q_two_contrast_family"],
                "interval_label": INTERVAL_LABEL,
            }
        )
        for local in contrast["dataset_estimates"]:
            rows.append(
                {
                    "dataset": local["dataset"],
                    "inference_scope": "dataset_descriptive",
                    "contrast": contrast["contrast"],
                    "estimate": local["estimate"],
                    "uncertainty_interval_95_low": None,
                    "uncertainty_interval_95_high": None,
                    "two_sided_p": None,
                    "bh_q_two_member_family": None,
                    "interval_label": "DESCRIPTIVE_POINT_ESTIMATE_NO_INTERVAL",
                }
            )
    return pd.DataFrame(rows)


def _representativeness_table(
    preflight_sources: Sequence[tuple[Path, Mapping[str, Any]]],
    global_trigger: Mapping[str, Any],
    conditional: Any,
    protocol: Mapping[str, Any],
) -> pd.DataFrame:
    by_dataset = {str(summary["dataset"]): (path, summary) for path, summary in preflight_sources}
    trigger_rows = {str(row["dataset"]): row for row in global_trigger["dataset_triggers"]}
    global_value = bool(global_trigger["global_triggered"])
    thresholds = global_trigger["thresholds"]
    conditional_records = {
        str(row["dataset"]): row for row in conditional.registry.get("dataset_estimates", [])
    }
    rows = []
    for dataset in sorted(protocol["datasets"]):
        if dataset not in by_dataset or dataset not in trigger_rows:
            raise RevisionProtocolError("[RELEASE_ASSET_PREFLIGHT_DATASET_MISSING]")
        summary_path, summary = by_dataset[dataset]
        table_path = summary_path.resolve().parent / "panel_representativeness.csv"
        if not table_path.is_file():
            raise RevisionProtocolError("[RELEASE_ASSET_REPRESENTATIVENESS_SOURCE_MISSING]")
        source = pd.read_csv(table_path)
        source_hash = _records_hash(source)
        expected_hash = str(summary["panel_representativeness_table_hash"])
        if (
            source_hash != expected_hash
            or expected_hash != trigger_rows[dataset]["panel_representativeness_table_hash"]
        ):
            raise RevisionProtocolError("[RELEASE_ASSET_REPRESENTATIVENESS_SOURCE_HASH_MISMATCH]")
        panel_manifest = summary["condition_panel_manifest"]
        primary = panel_manifest["primary"]["ordered_canonical_condition_ids"]
        sensitivity = panel_manifest["sensitivity"]["ordered_canonical_condition_ids"]
        legacy = panel_manifest["legacy_first50"]["ordered_condition_ids"]
        overlap_legacy = len(set(sensitivity) & set(legacy))
        overlap_primary = len(set(sensitivity) & set(primary))
        extracted = {
            "combined_selection_stratum_total_variation": _unique_statistic(
                source,
                panel="primary",
                variable="combined_selection_stratum",
                statistic="variable_total_variation_distance",
            ),
            "absolute_standardized_log1p_condition_cell_count_difference": abs(
                _unique_statistic(
                    source,
                    panel="primary",
                    variable="log1p_condition_cell_count",
                    statistic="standardized_mean_difference",
                )
            ),
            "target_mapping_coverage": _unique_statistic(
                source,
                panel="primary",
                variable="target_mapping",
                statistic="mapping_coverage",
            ),
            "condition_coverage": _unique_statistic(
                source,
                panel="primary",
                variable="panel_coverage",
                statistic="condition_coverage",
            ),
            "cell_coverage": _unique_statistic(
                source,
                panel="primary",
                variable="panel_coverage",
                statistic="cell_coverage",
            ),
        }
        declared = trigger_rows[dataset]
        if not np.isclose(
            extracted["combined_selection_stratum_total_variation"],
            float(declared["trigger_total_variation_distance"]),
            rtol=0,
            atol=1e-15,
        ) or not np.isclose(
            extracted["absolute_standardized_log1p_condition_cell_count_difference"],
            float(declared["trigger_abs_standardized_log1p_cell_count_difference"]),
            rtol=0,
            atol=1e-15,
        ):
            raise RevisionProtocolError("[RELEASE_ASSET_GLOBAL_TRIGGER_SOURCE_MISMATCH]")
        for metric_id, value in extracted.items():
            threshold = None
            contribution = False
            if metric_id == "combined_selection_stratum_total_variation":
                threshold = thresholds["total_variation_distance_exclusive_gt"]
                contribution = value > float(threshold)
            elif metric_id == "absolute_standardized_log1p_condition_cell_count_difference":
                threshold = thresholds[
                    "absolute_standardized_log1p_cell_count_difference_exclusive_gt"
                ]
                contribution = value > float(threshold)
            rows.append(
                {
                    "dataset": dataset,
                    "analysis": "primary_panel_representativeness",
                    "metric_id": metric_id,
                    "status": "MEASURED",
                    "value": value,
                    "threshold": threshold,
                    "trigger_contribution": contribution,
                    "global_triggered": global_value,
                    "panel_size": len(primary),
                    "overlap_with_legacy": 0,
                    "overlap_with_primary": 0,
                    "estimate": None,
                    "uncertainty_interval_95_low": None,
                    "uncertainty_interval_95_high": None,
                    "interval_label": None,
                    "reason_code": None,
                    "source_records_sha256": source_hash,
                }
            )
        if global_value:
            result = conditional_records.get(dataset)
            if result is None:
                raise RevisionProtocolError("[RELEASE_ASSET_CONDITIONAL_DATASET_DETAIL_MISSING]")
            conditional_row = {
                "status": "MEASURED",
                "estimate": result["estimate"],
                "low": conditional.registry["uncertainty_interval_95_low"],
                "high": conditional.registry["uncertainty_interval_95_high"],
                "label": INTERVAL_LABEL,
                "reason": None,
            }
        else:
            conditional_row = {
                "status": "NOT_TRIGGERED",
                "estimate": None,
                "low": None,
                "high": None,
                "label": None,
                "reason": "GLOBAL_OR_TRIGGER_FALSE",
            }
        rows.append(
            {
                "dataset": dataset,
                "analysis": "conditional_nonoverlap_panel",
                "metric_id": "conditional_panel_effect",
                "status": conditional_row["status"],
                "value": None,
                "threshold": None,
                "trigger_contribution": False,
                "global_triggered": global_value,
                "panel_size": len(sensitivity),
                "overlap_with_legacy": overlap_legacy,
                "overlap_with_primary": overlap_primary,
                "estimate": conditional_row["estimate"],
                "uncertainty_interval_95_low": conditional_row["low"],
                "uncertainty_interval_95_high": conditional_row["high"],
                "interval_label": conditional_row["label"],
                "reason_code": conditional_row["reason"],
                "source_records_sha256": source_hash,
            }
        )
    return pd.DataFrame(rows)


def _secondary_table(secondary: Any, mixed: Any, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in secondary.registry["metrics"]:
        rows.append(
            {
                "dataset": "EQUAL_DATASET",
                "inference_scope": "overall_family_test",
                "metric": record["metric"],
                "model": record["benefit_direction"],
                "estimate": record["estimate"],
                "rank": 1,
                "two_sided_p": record["centered_null_bootstrap_p"],
                "bh_q": record["bh_q_four_metric_family"],
                "family_label": "SECONDARY_FOUR_METRIC_FAMILY",
                "family_denominator": 4,
                **_empty_ranking(),
            }
        )
    ranking = secondary.detail_tables.get("condition_ranking_overlap")
    if ranking is None:
        raise RevisionProtocolError("[RELEASE_ASSET_CONDITION_RANKING_DETAIL_MISSING]")
    manifest_hash = canonical_sha256(sorted(metrics["artifact_hash"].astype(str)))
    primary_means = (
        metrics[
            (metrics["panel"] == "primary")
            & (metrics["hvg"] == 200)
            & metrics["arm"].isin(["string_go", "dense"])
        ]
        .groupby(["dataset", "arm"], sort=True)["pearson_r"]
        .mean()
    )
    for record in ranking.to_dict(orient="records"):
        for model in ("string_go", "dense"):
            local_values = primary_means.loc[record["dataset"]]
            model_rank = int((-local_values).rank(method="min").loc[model])
            rows.append(
                {
                    "dataset": record["dataset"],
                    "inference_scope": "dataset_descriptive",
                    "metric": "pearson_r",
                    "model": model,
                    "estimate": float(local_values.loc[model]),
                    "rank": model_rank,
                    "two_sided_p": None,
                    "bh_q": None,
                    "family_label": "CONDITION_RANKING_DESCRIPTIVE",
                    "family_denominator": 1,
                    "ranking_analysis_id": "CONDITION-RANKING-OVERLAP",
                    "top_k": int(record["top_k"]),
                    "shared_condition_count": int(record["shared_condition_count"]),
                    "top_k_coverage_complete": True,
                    "top_condition_jaccard": float(record["jaccard"]),
                    "shared_rank_spearman": float(record["shared_rank_spearman"]),
                    "ranking_status": str(record["shared_rank_spearman_status"]),
                    "ranking_input_artifact_manifest_sha256": manifest_hash,
                }
            )
    mixed_p = mixed.registry["two_sided_centered_null_bootstrap_p"]
    for record in mixed.registry["dataset_estimates"]:
        rows.append(
            {
                "dataset": record["dataset"],
                "inference_scope": "dataset_descriptive",
                "metric": "mixed_combined_minus_dense_pearson_r",
                "model": "combined_minus_dense",
                "estimate": record["estimate"],
                "rank": 1,
                "two_sided_p": None,
                "bh_q": None,
                "family_label": "MIXED_COMBINED_VS_DENSE_200_SENSITIVITY",
                "family_denominator": 1,
                **_empty_ranking(),
            }
        )
    rows.append(
        {
            "dataset": "EQUAL_DATASET",
            "inference_scope": "overall_family_test",
            "metric": "mixed_combined_minus_dense_pearson_r",
            "model": "combined_minus_dense",
            "estimate": mixed.registry["estimate"],
            "rank": 1,
            "two_sided_p": mixed_p,
            "bh_q": mixed_p,
            "family_label": "MIXED_COMBINED_VS_DENSE_200_SENSITIVITY",
            "family_denominator": 1,
            **_empty_ranking(),
        }
    )
    return pd.DataFrame(rows)


def _baseline_table(paths: Sequence[Path], metrics: pd.DataFrame) -> pd.DataFrame:
    payloads = [
        read_baseline_artifact(path) for path in sorted({Path(path).resolve() for path in paths})
    ]
    identities = [
        (
            str(item["identity"]["dataset"]),
            int(item["identity"]["hvg"]),
            str(item["identity"]["panel"]),
            str(item["identity"]["condition"]),
        )
        for item in payloads
    ]
    if len(payloads) != 600 or len(set(identities)) != 600:
        raise RevisionProtocolError("[RELEASE_ASSET_BASELINE_REFIT_COUNT_INVALID]")
    neural = metrics[
        (metrics["panel"] == "primary")
        & metrics["hvg"].isin([200, 500, 1000])
        & metrics["arm"].isin(["string_go", "dense"])
    ]
    neural_means = neural.groupby(["dataset", "hvg", "panel", "condition", "arm"], sort=True)[
        ["pearson_r", "mse", "mae"]
    ].mean()
    rows = []
    for payload in payloads:
        identity = payload["identity"]
        for model in ("string_go", "dense"):
            try:
                neural_skill = neural_means.loc[
                    (
                        identity["dataset"],
                        identity["hvg"],
                        identity["panel"],
                        identity["condition"],
                        model,
                    )
                ]
            except KeyError as error:
                raise RevisionProtocolError(
                    "[RELEASE_ASSET_NEURAL_BASELINE_SUPPORT_MISSING]"
                ) from error
            for baseline in BASELINE_METHODS:
                absolute = payload["absolute_skill"][baseline]
                for metric in ("pearson_r", "mse", "mae"):
                    zero_pearson = baseline == "zero_control_delta" and metric == "pearson_r"
                    baseline_value = absolute[metric]
                    neural_value = float(neural_skill[metric])
                    improvement = (
                        None
                        if zero_pearson
                        else (
                            neural_value - float(baseline_value)
                            if metric == "pearson_r"
                            else float(baseline_value) - neural_value
                        )
                    )
                    rows.append(
                        {
                            "dataset": identity["dataset"],
                            "hvg": identity["hvg"],
                            "panel": identity["panel"],
                            "condition": identity["condition"],
                            "neural_model": model,
                            "baseline": baseline,
                            "metric": metric,
                            "neural_absolute_skill": neural_value,
                            "neural_skill_status": "MEASURED",
                            "baseline_absolute_skill": None if zero_pearson else baseline_value,
                            "baseline_skill_status": (
                                "UNDEFINED_CONSTANT_VECTOR" if zero_pearson else "MEASURED"
                            ),
                            "improvement_over_baseline": improvement,
                            "improvement_status": "NOT_APPLICABLE" if zero_pearson else "MEASURED",
                            "undefined_reason_code": (
                                "ZERO_DELTA_PEARSON_UNDEFINED" if zero_pearson else None
                            ),
                            "baseline_artifact_sha256": payload["artifact_sha256"],
                        }
                    )
    return pd.DataFrame(rows)


def _compute_table(registry: Mapping[str, Any]) -> pd.DataFrame:
    records = registry.get("attempt_group_summaries")
    if not isinstance(records, list) or canonical_sha256(records) != registry.get(
        "attempt_group_summaries_sha256"
    ):
        raise RevisionProtocolError("[RELEASE_ASSET_COMPUTE_GROUP_HASH_INVALID]")
    return pd.DataFrame(records).rename(columns={"peak_device_bytes": "peak_device_bytes"})


def _external_table(registry: Mapping[str, Any]) -> pd.DataFrame:
    validity = registry.get("validity_registry", registry)
    performance = registry.get("performance_registry", {})
    members = validity.get("members", {})
    decisions = validity.get("member_decisions", {})
    rows = []
    for comparator in ("GEARS", "scGPT", "Geneformer"):
        member = members.get(comparator, {})
        evidence_hash = canonical_sha256(member)
        decision = decisions.get(comparator)
        rows.append(
            {
                "comparator": comparator,
                "diagnostic_id": "validity_decision",
                "status": decision,
                "value": None,
                "reference": "scientific_validity_contract",
                "tolerance": None,
                "reason_code": _member_reason(member) if decision == "EXCLUDED" else None,
                "evidence_sha256": evidence_hash,
            }
        )
        if decision != "PASS":
            continue
        rows.extend(_pass_member_diagnostics(comparator, member, evidence_hash))
        result = performance.get("adapter_results", {}).get(comparator, {})
        for contrast in result.get("contrasts", []):
            rows.append(
                {
                    "comparator": comparator,
                    "diagnostic_id": f"performance_{int(contrast['hvg'])}_hvg",
                    "status": "MEASURED",
                    "value": contrast["estimate"],
                    "reference": "revision_string_go",
                    "tolerance": contrast["bh_q_three_scale_family"],
                    "reason_code": None,
                    "evidence_sha256": canonical_sha256(contrast),
                }
            )
    return pd.DataFrame(rows)


def _pass_member_diagnostics(
    comparator: str, member: Mapping[str, Any], evidence_hash: str
) -> list[dict[str, Any]]:
    if comparator == "GEARS":
        comparisons = member.get("comparisons")
        bindings = member.get("validated_bindings")
        diagnostics = member.get("semantic_diagnostics")
        required_bindings = {
            "repository_url",
            "commit",
            "dataset_sha256",
            "split_name",
            "split_seed",
            "training_config_hash",
            "metric_source_sha256",
            "metric_function",
            "trust_anchor_sha256",
        }
        if (
            member.get("detached_validator_registry_status") != "RELEASED"
            or not isinstance(comparisons, list)
            or not comparisons
            or any(row.get("status") != "PASS" for row in comparisons)
            or not isinstance(bindings, Mapping)
            or not required_bindings <= set(bindings)
            or not isinstance(diagnostics, Mapping)
            or not isinstance(diagnostics.get("candidate"), Mapping)
        ):
            raise RevisionProtocolError("[RELEASE_ASSET_GEARS_DIAGNOSTIC_EVIDENCE_MISSING]")
        candidate_diagnostics = diagnostics["candidate"]
        required_diagnostics = {
            "metrics",
            "baseline_metrics",
            "directional_baseline_improvements",
            "minimum_directional_baseline_improvement",
            "training_loss_improvement",
            "validation_loss_improvement",
            "prediction_standard_deviation",
            "truth_standard_deviation",
            "loss_history_length",
            "gene_order_sha256",
            "perturbable_gene_order_sha256",
            "condition_target_mapping_sha256",
            "target_audit_sha256",
            "split_membership_sha256",
            "vector_row_conditions_sha256",
            "target_disjoint_status",
            "target_disjoint_evidence_sha256",
            "diagnostic_status",
        }
        positive_fields = (
            "minimum_directional_baseline_improvement",
            "training_loss_improvement",
            "validation_loss_improvement",
            "prediction_standard_deviation",
            "truth_standard_deviation",
        )
        if (
            set(candidate_diagnostics) != required_diagnostics
            or candidate_diagnostics.get("diagnostic_status") != "PASS"
            or any(
                isinstance(candidate_diagnostics.get(field), bool)
                or not isinstance(candidate_diagnostics.get(field), (int, float))
                or not np.isfinite(float(candidate_diagnostics[field]))
                or float(candidate_diagnostics[field]) <= 0
                for field in positive_fields
            )
            or isinstance(candidate_diagnostics.get("loss_history_length"), bool)
            or not isinstance(candidate_diagnostics.get("loss_history_length"), int)
            or int(candidate_diagnostics["loss_history_length"]) < 2
            or any(
                not isinstance(candidate_diagnostics.get(field), str)
                or not re.fullmatch(r"[0-9a-f]{64}", candidate_diagnostics[field])
                for field in (
                    "gene_order_sha256",
                    "perturbable_gene_order_sha256",
                    "condition_target_mapping_sha256",
                    "target_audit_sha256",
                    "split_membership_sha256",
                    "vector_row_conditions_sha256",
                    "target_disjoint_evidence_sha256",
                )
            )
            or candidate_diagnostics.get("target_disjoint_status")
            != "NOT_APPLICABLE_TEST_ONLY_POSITIVE_CONTROL_SPLIT"
        ):
            raise RevisionProtocolError("[RELEASE_ASSET_GEARS_DIAGNOSTIC_GATE_INVALID]")
        values = {
            "prediction_nondegeneracy": candidate_diagnostics["prediction_standard_deviation"],
            "baseline_superiority": candidate_diagnostics[
                "minimum_directional_baseline_improvement"
            ],
            "training_loss_improvement": candidate_diagnostics["training_loss_improvement"],
            "validation_loss_improvement": candidate_diagnostics["validation_loss_improvement"],
            "gene_order_binding": candidate_diagnostics["gene_order_sha256"],
            "target_mapping_binding": candidate_diagnostics["condition_target_mapping_sha256"],
            "vector_row_condition_binding": candidate_diagnostics["vector_row_conditions_sha256"],
            "split_target_disjoint_binding": candidate_diagnostics[
                "target_disjoint_evidence_sha256"
            ],
        }
        references = {
            "prediction_nondegeneracy": "strictly_positive_prediction_standard_deviation",
            "baseline_superiority": (
                "minimum_positive_directional_improvement_over_" "control_mean_zero_change"
            ),
            "training_loss_improvement": "first_epoch_loss_minus_final_epoch_loss",
            "validation_loss_improvement": "first_epoch_loss_minus_final_epoch_loss",
            "gene_order_binding": (
                "detached_candidate_gene_order; perturbable_gene_order_sha256="
                + candidate_diagnostics["perturbable_gene_order_sha256"]
            ),
            "target_mapping_binding": (
                "candidate_condition_target_mapping; target_audit_sha256="
                + candidate_diagnostics["target_audit_sha256"]
            ),
            "vector_row_condition_binding": (
                "candidate_condition_ids_in_exact_vector_row_order; split_membership_sha256="
                + candidate_diagnostics["split_membership_sha256"]
            ),
            "split_target_disjoint_binding": candidate_diagnostics["target_disjoint_status"],
        }
        tolerances = {
            **{key: (0.0 if key in positive_fields else None) for key in values},
        }
        statuses = {
            key: ("MEASURED" if key == "split_target_disjoint_binding" else "PASS")
            for key in values
        }
        reason_codes = {
            key: (
                candidate_diagnostics["target_disjoint_status"]
                if key == "split_target_disjoint_binding"
                else None
            )
            for key in values
        }
    else:
        required = {
            "recomputed_metrics",
            "recomputed_baseline_metrics",
            "expected_metric_comparisons",
            "training_loss_improvement",
            "validation_loss_improvement",
            "minimum_training_loss_improvement",
            "minimum_validation_loss_improvement",
            "gene_order_sha256",
            "target_mapping_manifest_sha256",
            "vector_row_conditions_sha256",
            "split_manifest_sha256",
            "independent_expected_value_provenance_sha256",
        }
        if not required <= set(member):
            raise RevisionProtocolError("[RELEASE_ASSET_EXTERNAL_DIAGNOSTIC_EVIDENCE_MISSING]")
        metrics = member["recomputed_metrics"]
        baseline = member["recomputed_baseline_metrics"]
        comparisons = member["expected_metric_comparisons"]
        if (
            not isinstance(metrics, Mapping)
            or not isinstance(baseline, Mapping)
            or not isinstance(comparisons, list)
            or any(row.get("status") != "PASS" for row in comparisons)
            or any(metrics.get(key) is None for key in ("pearson_r", "mse", "mae"))
            or not (
                float(metrics["mse"]) < float(baseline["mse"])
                and float(metrics["mae"]) < float(baseline["mae"])
                and (
                    baseline.get("pearson_r") is None
                    or float(metrics["pearson_r"]) > float(baseline["pearson_r"])
                )
            )
            or float(member["training_loss_improvement"])
            < float(member["minimum_training_loss_improvement"])
            or float(member["validation_loss_improvement"])
            < float(member["minimum_validation_loss_improvement"])
        ):
            raise RevisionProtocolError("[RELEASE_ASSET_EXTERNAL_DIAGNOSTIC_GATE_INVALID]")
        values = {
            "prediction_nondegeneracy": metrics["pearson_r"],
            "baseline_superiority": float(baseline["mse"]) - float(metrics["mse"]),
            "training_loss_improvement": member["training_loss_improvement"],
            "validation_loss_improvement": member["validation_loss_improvement"],
            "gene_order_binding": member["gene_order_sha256"],
            "target_mapping_binding": member["target_mapping_manifest_sha256"],
            "vector_row_condition_binding": member["vector_row_conditions_sha256"],
            "split_target_disjoint_binding": member["split_manifest_sha256"],
        }
        references = {
            "prediction_nondegeneracy": next(
                row["expected"] for row in comparisons if row["metric"] == "pearson_r"
            ),
            "baseline_superiority": "positive_MSE_improvement_over_bound_baseline",
            "training_loss_improvement": member["minimum_training_loss_improvement"],
            "validation_loss_improvement": member["minimum_validation_loss_improvement"],
            "gene_order_binding": "hash_bound_gene_order",
            "target_mapping_binding": "hash_bound_target_mapping",
            "vector_row_condition_binding": "test_split_exact_order",
            "split_target_disjoint_binding": "independently_recomputed_target_disjoint",
        }
        tolerances = {
            "prediction_nondegeneracy": next(
                row["tolerance"] for row in comparisons if row["metric"] == "pearson_r"
            ),
            "baseline_superiority": 0.0,
            "training_loss_improvement": member["minimum_training_loss_improvement"],
            "validation_loss_improvement": member["minimum_validation_loss_improvement"],
            "gene_order_binding": None,
            "target_mapping_binding": None,
            "vector_row_condition_binding": None,
            "split_target_disjoint_binding": None,
        }
        statuses = {key: "PASS" for key in values}
        reason_codes = {key: None for key in values}
    return [
        {
            "comparator": comparator,
            "diagnostic_id": diagnostic,
            "status": statuses[diagnostic],
            "value": value,
            "reference": references[diagnostic],
            "tolerance": tolerances[diagnostic],
            "reason_code": reason_codes[diagnostic],
            "evidence_sha256": evidence_hash,
        }
        for diagnostic, value in values.items()
    ]


def _member_reason(member: Mapping[str, Any]) -> str:
    reason = member.get("reason_code")
    if isinstance(reason, str) and reason:
        return reason
    reasons = member.get("reason_codes")
    if isinstance(reasons, list) and reasons and isinstance(reasons[0], str):
        return reasons[0]
    return "REASON_CODED_EXCLUSION"


def _empty_ranking() -> dict[str, Any]:
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


def _unique_statistic(frame: pd.DataFrame, *, panel: str, variable: str, statistic: str) -> float:
    selected = frame[
        (frame["panel"] == panel)
        & (frame["variable"] == variable)
        & (frame["statistic"] == statistic)
        & (frame["status"] == "MEASURED")
    ]
    if len(selected) != 1:
        raise RevisionProtocolError(
            f"[RELEASE_ASSET_REPRESENTATIVENESS_METRIC_NOT_UNIQUE] {variable}:{statistic}"
        )
    value = float(selected.iloc[0]["value"])
    if not np.isfinite(value):
        raise RevisionProtocolError("[RELEASE_ASSET_REPRESENTATIVENESS_METRIC_NONFINITE]")
    return value


def _metric_sort_columns(frame: pd.DataFrame) -> list[str]:
    preferred = ["dataset", "hvg", "panel", "arm", "condition", "seed"]
    return [column for column in preferred if column in frame]


def _records_hash(frame: pd.DataFrame) -> str:
    canonical_csv = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="",
    )
    return canonical_sha256({"columns": list(frame.columns), "canonical_csv": canonical_csv})

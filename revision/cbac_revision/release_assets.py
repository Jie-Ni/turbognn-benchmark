"""Generate and validate the complete, hash-bound revision result asset package."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from .artifacts import canonical_sha256, file_sha256
from .errors import RevisionProtocolError

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

DATASETS = ("adamson", "norman", "replogle_k562", "replogle_rpe1")
INTERVAL_LABEL = "95% conditional bootstrap uncertainty interval"
ASSET_GROUPS = (
    "primary_summary_and_forest",
    "paired_condition_distribution",
    "fraction_improved_interval",
    "target_sensitivity",
    "topology_instances_and_diagnostics",
    "scale_native_and_common200",
    "propagation",
    "representativeness_and_conditional",
    "secondary_ranking_and_multiplicity",
    "baseline_absolute_skill",
    "compute_and_failure_denominators",
    "external_diagnostics_and_exclusions",
)
GROUP_LABELS = {
    "primary_summary_and_forest": "Primary effect summary and forest data",
    "paired_condition_distribution": "Paired-condition effect distribution",
    "fraction_improved_interval": "Fraction of conditions improved",
    "target_sensitivity": "Target- and pathway-dependence sensitivity",
    "topology_instances_and_diagnostics": "Topology-null instances and diagnostics",
    "scale_native_and_common200": "Native-space and common-200 scale analysis",
    "propagation": "Propagation control",
    "representativeness_and_conditional": "Representativeness and conditional panel",
    "secondary_ranking_and_multiplicity": "Secondary metrics, ranking and multiplicity",
    "baseline_absolute_skill": "Analytic-baseline absolute and relative skill",
    "compute_and_failure_denominators": "Compute and failure denominators",
    "external_diagnostics_and_exclusions": "External-comparator validity decisions",
}
GROUP_COLUMNS: dict[str, tuple[str, ...]] = {
    "primary_summary_and_forest": (
        "dataset",
        "cell_line",
        "weighting",
        "estimate",
        "uncertainty_interval_95_low",
        "uncertainty_interval_95_high",
        "interval_label",
        "n_conditions",
    ),
    "paired_condition_distribution": (
        "dataset",
        "condition",
        "target_id",
        "delta",
        "n_seeds",
    ),
    "fraction_improved_interval": (
        "dataset",
        "numerator_improved",
        "denominator_conditions",
        "fraction_improved",
        "uncertainty_interval_95_low",
        "uncertainty_interval_95_high",
        "interval_label",
    ),
    "target_sensitivity": (
        "sensitivity_id",
        "status",
        "estimate",
        "uncertainty_interval_95_low",
        "uncertainty_interval_95_high",
        "interval_label",
        "reason_code",
    ),
    "topology_instances_and_diagnostics": (
        "dataset",
        "rewire_id",
        "graph_index",
        "estimate",
        "uncertainty_interval_95_low",
        "uncertainty_interval_95_high",
        "interval_label",
        "degree_sequence_match",
        "component_membership_match",
        "component_count_match",
        "isolates_match",
        "changed_edge_fraction",
        "source_support_sha256",
        "support_sha256",
        "diagnostics_status",
    ),
    "scale_native_and_common200": (
        "dataset",
        "hvg",
        "evaluation_space",
        "inference_scope",
        "contrast",
        "estimate",
        "uncertainty_interval_95_low",
        "uncertainty_interval_95_high",
        "two_sided_p",
        "bh_q_two_member_family",
        "interval_label",
    ),
    "propagation": (
        "dataset",
        "inference_scope",
        "contrast",
        "estimate",
        "uncertainty_interval_95_low",
        "uncertainty_interval_95_high",
        "two_sided_p",
        "bh_q_two_member_family",
        "interval_label",
    ),
    "representativeness_and_conditional": (
        "dataset",
        "analysis",
        "metric_id",
        "status",
        "value",
        "threshold",
        "trigger_contribution",
        "global_triggered",
        "panel_size",
        "overlap_with_legacy",
        "overlap_with_primary",
        "estimate",
        "uncertainty_interval_95_low",
        "uncertainty_interval_95_high",
        "interval_label",
        "reason_code",
        "source_records_sha256",
    ),
    "secondary_ranking_and_multiplicity": (
        "dataset",
        "inference_scope",
        "metric",
        "model",
        "estimate",
        "rank",
        "two_sided_p",
        "bh_q",
        "family_label",
        "family_denominator",
        "ranking_analysis_id",
        "top_k",
        "shared_condition_count",
        "top_k_coverage_complete",
        "top_condition_jaccard",
        "shared_rank_spearman",
        "ranking_status",
        "ranking_input_artifact_manifest_sha256",
    ),
    "baseline_absolute_skill": (
        "dataset",
        "hvg",
        "panel",
        "condition",
        "neural_model",
        "baseline",
        "metric",
        "neural_absolute_skill",
        "neural_skill_status",
        "baseline_absolute_skill",
        "baseline_skill_status",
        "improvement_over_baseline",
        "improvement_status",
        "undefined_reason_code",
        "baseline_artifact_sha256",
    ),
    "compute_and_failure_denominators": (
        "dataset",
        "model",
        "hvg",
        "attempts",
        "successes",
        "failures",
        "training_seconds",
        "inference_seconds",
        "peak_device_bytes",
        "status",
    ),
    "external_diagnostics_and_exclusions": (
        "comparator",
        "diagnostic_id",
        "status",
        "value",
        "reference",
        "tolerance",
        "reason_code",
        "evidence_sha256",
    ),
}
MANIFEST_FIELDS = {
    "schema_version",
    "mode",
    "status",
    "required_asset_groups",
    "source_bindings",
    "source_bindings_hash",
    "semantic_records_hashes",
    "entries",
    "entries_hash",
    "canonical_include_source_id",
    "manifest_sha256",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def render_release_assets(
    output_dir: Path,
    *,
    mode: str,
    tables: Mapping[str, pd.DataFrame] | None = None,
    source_bindings: Mapping[str, str] | None = None,
    expected_datasets: Sequence[str] = DATASETS,
) -> dict[str, Any]:
    """Render all required CSV/TeX/SVG/PDF groups and a canonical include file."""

    if mode not in {"author-review", "released"}:
        raise RevisionProtocolError("[RELEASE_ASSET_MODE_INVALID]")
    bindings = dict(source_bindings or {})
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(value, str)
        or not SHA256_RE.fullmatch(value)
        for name, value in bindings.items()
    ):
        raise RevisionProtocolError("[RELEASE_ASSET_SOURCE_BINDING_INVALID]")
    if mode == "released" and not bindings:
        raise RevisionProtocolError("[RELEASE_ASSET_RELEASED_SOURCE_BINDINGS_MISSING]")
    supplied = dict(tables or {})
    if mode == "released" and set(supplied) != set(ASSET_GROUPS):
        missing = sorted(set(ASSET_GROUPS) - set(supplied))
        extra = sorted(set(supplied) - set(ASSET_GROUPS))
        raise RevisionProtocolError(
            f"[RELEASE_ASSET_GROUP_SET_INVALID] missing={missing}; extra={extra}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    asset_dir = output_dir / "release_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    semantic_hashes: dict[str, str] = {}
    entries: list[dict[str, Any]] = []
    for group in ASSET_GROUPS:
        if mode == "released":
            table = _validate_group_table(group, supplied[group], expected_datasets)
        else:
            table = pd.DataFrame(
                [{"asset_group": group, "status": "AUTHOR_REVIEW_VALUES_WITHHELD"}]
            )
        records_hash = _records_hash(table)
        semantic_hashes[group] = records_hash
        csv_path = asset_dir / f"{group}.csv"
        tex_path = asset_dir / f"{group}.tex"
        svg_path = asset_dir / f"{group}.svg"
        pdf_path = asset_dir / f"{group}.pdf"
        _atomic_write_csv(csv_path, table)
        _atomic_write_text(tex_path, _table_tex(group, table, records_hash, mode))
        _plot_group(table, group, svg_path, pdf_path, records_hash, mode)
        for path, media_type in (
            (csv_path, "text/csv"),
            (tex_path, "application/x-tex"),
            (svg_path, "image/svg+xml"),
            (pdf_path, "application/pdf"),
        ):
            entries.append(_entry(output_dir, path, group, media_type))

    include_path = output_dir / "released_results_include.tex"
    include_text = _canonical_include(mode, semantic_hashes)
    _atomic_write_text(include_path, include_text)
    entries.append(
        _entry(output_dir, include_path, "canonical_released_results_include", "application/x-tex")
    )
    entries.sort(key=lambda row: str(row["source_id"]))
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "mode": mode,
        "status": ("AUTHOR_REVIEW_PLACEHOLDERS" if mode == "author-review" else "RELEASED_ASSETS"),
        "required_asset_groups": list(ASSET_GROUPS),
        "source_bindings": bindings,
        "source_bindings_hash": canonical_sha256(bindings),
        "semantic_records_hashes": semantic_hashes,
        "entries": entries,
        "entries_hash": canonical_sha256(entries),
        "canonical_include_source_id": include_path.relative_to(output_dir).as_posix(),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = output_dir / "release_asset_manifest.json"
    _atomic_write_json(manifest_path, manifest)
    validate_release_asset_manifest(
        manifest_path,
        expected_manifest_file_sha256=file_sha256(manifest_path),
        expected_source_bindings=bindings,
        expected_datasets=expected_datasets,
    )
    return manifest


def validate_release_asset_manifest(
    manifest_path: Path,
    *,
    expected_manifest_file_sha256: str,
    expected_source_bindings: Mapping[str, str] | None = None,
    expected_datasets: Sequence[str] = DATASETS,
) -> dict[str, Any]:
    """Verify hashes, paths, schemas, and semantic round trips for every release asset."""

    if (
        not SHA256_RE.fullmatch(expected_manifest_file_sha256)
        or file_sha256(manifest_path) != expected_manifest_file_sha256
    ):
        raise RevisionProtocolError("[RELEASE_ASSET_CALLER_PINNED_MANIFEST_MISMATCH]")
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RevisionProtocolError("[RELEASE_ASSET_MANIFEST_INVALID]") from error
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_FIELDS:
        raise RevisionProtocolError("[RELEASE_ASSET_MANIFEST_SCHEMA_INVALID]")
    unsigned = dict(manifest)
    declared_hash = unsigned.pop("manifest_sha256", None)
    mode = manifest.get("mode")
    bindings = manifest.get("source_bindings")
    if (
        declared_hash != canonical_sha256(unsigned)
        or mode not in {"author-review", "released"}
        or manifest.get("required_asset_groups") != list(ASSET_GROUPS)
        or not isinstance(bindings, dict)
        or manifest.get("source_bindings_hash") != canonical_sha256(bindings)
        or manifest.get("entries_hash") != canonical_sha256(manifest.get("entries"))
    ):
        raise RevisionProtocolError("[RELEASE_ASSET_MANIFEST_HASH_OR_CONTRACT_INVALID]")
    if expected_source_bindings is not None and bindings != dict(expected_source_bindings):
        raise RevisionProtocolError("[RELEASE_ASSET_SOURCE_BINDING_MISMATCH]")
    if mode == "released" and not bindings:
        raise RevisionProtocolError("[RELEASE_ASSET_RELEASED_SOURCE_BINDINGS_MISSING]")
    root = manifest_path.resolve().parent
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != len(ASSET_GROUPS) * 4 + 1:
        raise RevisionProtocolError("[RELEASE_ASSET_ENTRY_COUNT_INVALID]")
    observed_keys: set[tuple[str, str]] = set()
    entry_by_source: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "asset_group",
            "source_id",
            "media_type",
            "sha256",
            "bytes",
        }:
            raise RevisionProtocolError("[RELEASE_ASSET_ENTRY_SCHEMA_INVALID]")
        source_id = entry["source_id"]
        path = _resolve_relative(root, source_id)
        if (
            path is None
            or not path.is_file()
            or entry["sha256"] != file_sha256(path)
            or isinstance(entry["bytes"], bool)
            or not isinstance(entry["bytes"], int)
            or entry["bytes"] != path.stat().st_size
        ):
            raise RevisionProtocolError("[RELEASE_ASSET_FILE_BINDING_INVALID]")
        key = (str(entry["asset_group"]), str(entry["media_type"]))
        if key in observed_keys or source_id in entry_by_source:
            raise RevisionProtocolError("[RELEASE_ASSET_DUPLICATE_ENTRY]")
        observed_keys.add(key)
        entry_by_source[source_id] = entry
    expected_keys = {
        (group, media_type)
        for group in ASSET_GROUPS
        for media_type in ("text/csv", "application/x-tex", "image/svg+xml", "application/pdf")
    }
    expected_keys.add(("canonical_released_results_include", "application/x-tex"))
    if observed_keys != expected_keys:
        raise RevisionProtocolError("[RELEASE_ASSET_ENTRY_SET_INVALID]")
    semantic_hashes = manifest.get("semantic_records_hashes")
    if not isinstance(semantic_hashes, dict) or set(semantic_hashes) != set(ASSET_GROUPS):
        raise RevisionProtocolError("[RELEASE_ASSET_SEMANTIC_HASH_SET_INVALID]")
    for group in ASSET_GROUPS:
        csv_path = root / "release_assets" / f"{group}.csv"
        tex_path = root / "release_assets" / f"{group}.tex"
        frame = pd.read_csv(
            csv_path,
            dtype={
                column: "string" for column in GROUP_COLUMNS[group] if column.endswith("sha256")
            },
            float_precision="round_trip",
        )
        if mode == "released":
            frame = _validate_group_table(group, frame, expected_datasets)
        elif tuple(frame.columns) != ("asset_group", "status") or frame.to_dict(
            orient="records"
        ) != [{"asset_group": group, "status": "AUTHOR_REVIEW_VALUES_WITHHELD"}]:
            raise RevisionProtocolError("[RELEASE_ASSET_AUTHOR_REVIEW_PLACEHOLDER_INVALID]")
        records_hash = _records_hash(frame)
        if records_hash != semantic_hashes[group]:
            raise RevisionProtocolError("[RELEASE_ASSET_SEMANTIC_ROUND_TRIP_MISMATCH]")
        marker = f"% semantic_records_sha256={records_hash}"
        if marker not in tex_path.read_text(encoding="utf-8"):
            raise RevisionProtocolError("[RELEASE_ASSET_TEX_SEMANTIC_MARKER_MISSING]")
    include_id = manifest.get("canonical_include_source_id")
    include_path = _resolve_relative(root, include_id)
    if include_path is None or include_path.name != "released_results_include.tex":
        raise RevisionProtocolError("[RELEASE_ASSET_CANONICAL_INCLUDE_INVALID]")
    expected_include = _canonical_include(str(mode), semantic_hashes)
    if include_path.read_text(encoding="utf-8") != expected_include:
        raise RevisionProtocolError("[RELEASE_ASSET_INCLUDE_ROUND_TRIP_MISMATCH]")
    return manifest


def build_claim_consequence_gate(
    manifest_path: Path,
    *,
    expected_manifest_file_sha256: str,
    expected_datasets: Sequence[str] = DATASETS,
) -> dict[str, Any]:
    """Bind biological-edge wording to absolute-skill and ranking consequences."""

    manifest = validate_release_asset_manifest(
        manifest_path,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        expected_datasets=expected_datasets,
    )
    if manifest["mode"] != "released":
        registry: dict[str, Any] = {
            "schema_version": "1.0",
            "registry_id": "CLAIM-CONSEQUENCE-GATE",
            "status": "WITHHELD",
            "baseline_absolute_skill_eligible": False,
            "ranking_consequence_eligible": False,
            "reason_codes": ["RELEASED_CONSEQUENCE_ASSETS_UNAVAILABLE"],
            "release_asset_manifest_file_sha256": expected_manifest_file_sha256,
            "release_asset_manifest_self_hash": manifest["manifest_sha256"],
        }
        registry["registry_hash"] = canonical_sha256(registry)
        return registry
    root = manifest_path.resolve().parent / "release_assets"
    baseline = pd.read_csv(root / "baseline_absolute_skill.csv")
    ranking = pd.read_csv(root / "secondary_ranking_and_multiplicity.csv")
    _validate_baseline(baseline, tuple(expected_datasets))
    _validate_secondary(ranking, tuple(expected_datasets))
    required_pairs = {
        ("zero_control_delta", "mse"),
        ("zero_control_delta", "mae"),
        ("training_condition_mean", "pearson_r"),
        ("training_condition_mean", "mse"),
        ("deterministic_ridge_linear", "pearson_r"),
        ("deterministic_ridge_linear", "mse"),
    }
    string_go = baseline[
        (baseline["neural_model"] == "string_go")
        & (baseline["hvg"] == 200)
        & (baseline["panel"] == "primary")
        & baseline[["baseline", "metric"]].apply(tuple, axis=1).isin(required_pairs)
    ]
    baseline_means = (
        string_go.groupby(["dataset", "baseline", "metric"], sort=True)["improvement_over_baseline"]
        .mean()
        .reset_index()
    )
    expected_baseline_cells = {
        (dataset, baseline_name, metric)
        for dataset in expected_datasets
        for baseline_name, metric in required_pairs
    }
    baseline_eligible = (
        set(
            zip(
                baseline_means["dataset"],
                baseline_means["baseline"],
                baseline_means["metric"],
                strict=False,
            )
        )
        == expected_baseline_cells
        and (baseline_means["improvement_over_baseline"].astype(float) > 0).all()
    )
    primary_ranking = ranking[ranking["metric"] == "pearson_r"]
    rank_pivot = primary_ranking.pivot(index="dataset", columns="model", values="rank")
    ranking_evidence_complete = (
        primary_ranking["ranking_analysis_id"].eq("CONDITION-RANKING-OVERLAP").all()
        and primary_ranking["top_k"].eq(10).all()
        and (primary_ranking["shared_condition_count"].astype(int) >= 10).all()
        and primary_ranking["top_k_coverage_complete"].map(_is_true).all()
        and primary_ranking["top_condition_jaccard"].between(0, 1).all()
        and primary_ranking["shared_rank_spearman"].between(-1, 1).all()
        and primary_ranking["ranking_status"].eq("MEASURED").all()
        and primary_ranking["ranking_input_artifact_manifest_sha256"]
        .map(lambda value: isinstance(value, str) and bool(SHA256_RE.fullmatch(value)))
        .all()
    )
    ranking_eligible = (
        set(rank_pivot.index) == set(expected_datasets)
        and {"string_go", "dense"} <= set(rank_pivot.columns)
        and (rank_pivot["string_go"].astype(int) <= rank_pivot["dense"].astype(int)).all()
        and ranking_evidence_complete
    )
    reasons = []
    if not baseline_eligible:
        reasons.append("STRING_GO_NOT_SUPERIOR_TO_ALL_SIMPLE_BASELINES_ON_PRIMARY_SKILL")
    if not ranking_eligible:
        reasons.append("STRING_GO_RANKING_CONSEQUENCE_NOT_SUPPORTIVE")
    registry = {
        "schema_version": "1.0",
        "registry_id": "CLAIM-CONSEQUENCE-GATE",
        "status": "RELEASED",
        "baseline_absolute_skill_eligible": bool(baseline_eligible),
        "ranking_consequence_eligible": bool(ranking_eligible),
        "reason_codes": reasons,
        "baseline_gate_estimand": (
            "positive_within_dataset_mean_string_go_improvement_over_zero_by_mse_and_mae_"
            "and_over_training_mean_and_ridge_by_pearson_and_mse"
        ),
        "ranking_gate_estimand": (
            "string_go_model_rank_not_worse_than_dense_with_condition_top10_"
            "prioritisation_consequence_complete_within_every_dataset"
        ),
        "baseline_gate_records": baseline_means.to_dict(orient="records"),
        "baseline_gate_records_sha256": _records_hash(baseline_means),
        "ranking_gate_records_sha256": _records_hash(primary_ranking),
        "release_asset_manifest_file_sha256": expected_manifest_file_sha256,
        "release_asset_manifest_self_hash": manifest["manifest_sha256"],
    }
    registry["registry_hash"] = canonical_sha256(registry)
    return registry


def _validate_group_table(
    group: str, frame: pd.DataFrame, expected_datasets: Sequence[str]
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != GROUP_COLUMNS[group]:
        raise RevisionProtocolError(f"[RELEASE_ASSET_SCHEMA_INVALID] {group}")
    if frame.empty:
        raise RevisionProtocolError(f"[RELEASE_ASSET_EMPTY] {group}")
    output = frame.copy()
    validators = {
        "primary_summary_and_forest": _validate_primary,
        "paired_condition_distribution": _validate_paired,
        "fraction_improved_interval": _validate_fraction,
        "target_sensitivity": _validate_target_sensitivity,
        "topology_instances_and_diagnostics": _validate_topology,
        "scale_native_and_common200": _validate_scale,
        "propagation": _validate_propagation,
        "representativeness_and_conditional": _validate_representativeness,
        "secondary_ranking_and_multiplicity": _validate_secondary,
        "baseline_absolute_skill": _validate_baseline,
        "compute_and_failure_denominators": _validate_compute,
        "external_diagnostics_and_exclusions": _validate_external,
    }
    validators[group](output, tuple(expected_datasets))
    return output


def _validate_primary(frame: pd.DataFrame, datasets: tuple[str, ...]) -> None:
    expected_rows = set(datasets) | {"EQUAL_DATASET", "EQUAL_CELL_LINE"}
    if set(frame["dataset"]) != expected_rows or frame["dataset"].duplicated().any():
        raise RevisionProtocolError("[RELEASE_ASSET_PRIMARY_ROWS_INVALID]")
    _validate_intervals(frame)
    _positive_integers(frame["n_conditions"], "primary n_conditions")
    if frame.loc[frame["dataset"] == "EQUAL_DATASET", "weighting"].iloc[0] != "equal_dataset":
        raise RevisionProtocolError("[RELEASE_ASSET_PRIMARY_EQUAL_DATASET_WEIGHT_INVALID]")
    if frame.loc[frame["dataset"] == "EQUAL_CELL_LINE", "weighting"].iloc[0] != "equal_cell_line":
        raise RevisionProtocolError("[RELEASE_ASSET_PRIMARY_EQUAL_CELL_LINE_WEIGHT_INVALID]")


def _validate_paired(frame: pd.DataFrame, datasets: tuple[str, ...]) -> None:
    if set(frame["dataset"]) != set(datasets) or frame.duplicated(["dataset", "condition"]).any():
        raise RevisionProtocolError("[RELEASE_ASSET_PAIRED_IDENTITIES_INVALID]")
    _finite(frame["delta"], "paired delta")
    _positive_integers(frame["n_seeds"], "paired seed count")
    if not (frame["n_seeds"].astype(int) == 3).all():
        raise RevisionProtocolError("[RELEASE_ASSET_PAIRED_SEED_COMPLETENESS_INVALID]")


def _validate_fraction(frame: pd.DataFrame, datasets: tuple[str, ...]) -> None:
    if set(frame["dataset"]) != set(datasets) or frame["dataset"].duplicated().any():
        raise RevisionProtocolError("[RELEASE_ASSET_FRACTION_ROWS_INVALID]")
    _nonnegative_integers(frame["numerator_improved"], "fraction numerator")
    _positive_integers(frame["denominator_conditions"], "fraction denominator")
    _validate_intervals(frame, estimate_column="fraction_improved", unit_interval=True)
    observed = frame["numerator_improved"].astype(int) / frame["denominator_conditions"].astype(int)
    if not np.allclose(observed, frame["fraction_improved"].astype(float), rtol=0, atol=1e-15):
        raise RevisionProtocolError("[RELEASE_ASSET_FRACTION_COUNT_MISMATCH]")


def _validate_target_sensitivity(frame: pd.DataFrame, _: tuple[str, ...]) -> None:
    required = {
        "target_cluster_resampling",
        "leave_target_out",
        "leave_pathway_out",
        "shared_control_cell_bootstrap",
    }
    if set(frame["sensitivity_id"]) != required or frame["sensitivity_id"].duplicated().any():
        raise RevisionProtocolError("[RELEASE_ASSET_TARGET_SENSITIVITY_SET_INVALID]")
    shared = frame[frame["sensitivity_id"] == "shared_control_cell_bootstrap"]
    if len(shared) != 1 or shared.iloc[0]["status"] != "MEASURED":
        raise RevisionProtocolError("[RELEASE_ASSET_SHARED_CONTROL_BOOTSTRAP_NOT_MEASURED]")
    for row in frame.itertuples(index=False):
        if row.status == "MEASURED":
            _validate_interval_values(
                row.estimate,
                row.uncertainty_interval_95_low,
                row.uncertainty_interval_95_high,
                row.interval_label,
            )
            if not pd.isna(row.reason_code):
                raise RevisionProtocolError("[RELEASE_ASSET_MEASURED_REASON_INVALID]")
        elif row.status == "SENSITIVITY_RANGE":
            values = _finite_values(
                [
                    row.estimate,
                    row.uncertainty_interval_95_low,
                    row.uncertainty_interval_95_high,
                ],
                "sensitivity range",
            )
            if (
                values[1] > values[0]
                or values[0] > values[2]
                or row.interval_label != "DESCRIPTIVE_LEAVE_ONE_CLUSTER_OUT_RANGE"
                or not pd.isna(row.reason_code)
            ):
                raise RevisionProtocolError("[RELEASE_ASSET_SENSITIVITY_RANGE_INVALID]")
        elif row.status == "FEASIBLE":
            if not all(
                pd.isna(value)
                for value in (
                    row.estimate,
                    row.uncertainty_interval_95_low,
                    row.uncertainty_interval_95_high,
                    row.interval_label,
                    row.reason_code,
                )
            ):
                raise RevisionProtocolError("[RELEASE_ASSET_FEASIBILITY_DETAIL_INVALID]")
        elif row.status == "NOT_AVAILABLE":
            if not all(
                pd.isna(value)
                for value in (
                    row.estimate,
                    row.uncertainty_interval_95_low,
                    row.uncertainty_interval_95_high,
                    row.interval_label,
                )
            ) or not isinstance(row.reason_code, str):
                raise RevisionProtocolError("[RELEASE_ASSET_TARGET_NA_DETAIL_INVALID]")
        else:
            raise RevisionProtocolError("[RELEASE_ASSET_TARGET_STATUS_INVALID]")


def _validate_topology(frame: pd.DataFrame, datasets: tuple[str, ...]) -> None:
    expected = {
        (dataset, f"string_go_rewire_{index:02d}") for dataset in datasets for index in range(1, 11)
    }
    observed = set(zip(frame["dataset"], frame["rewire_id"], strict=False))
    if observed != expected or frame.duplicated(["dataset", "rewire_id"]).any():
        raise RevisionProtocolError("[RELEASE_ASSET_TOPOLOGY_INSTANCE_SET_INVALID]")
    _finite(frame["estimate"], "topology instance estimate")
    if (
        not frame["uncertainty_interval_95_low"].isna().all()
        or not frame["uncertainty_interval_95_high"].isna().all()
        or not frame["interval_label"].eq("DESCRIPTIVE_GRAPH_INSTANCE_POINT_ESTIMATE").all()
    ):
        raise RevisionProtocolError("[RELEASE_ASSET_TOPOLOGY_INSTANCE_INTERVAL_INVALID]")
    for field in (
        "degree_sequence_match",
        "component_membership_match",
        "component_count_match",
        "isolates_match",
    ):
        if not frame[field].map(_is_true).all():
            raise RevisionProtocolError(f"[RELEASE_ASSET_TOPOLOGY_DIAGNOSTIC_INVALID] {field}")
    changed = _finite(frame["changed_edge_fraction"], "changed edge fraction")
    if not ((changed >= 0.8) & (changed <= 1.0)).all():
        raise RevisionProtocolError("[RELEASE_ASSET_TOPOLOGY_EDGE_CHANGE_INVALID]")
    for field in ("source_support_sha256", "support_sha256"):
        if (
            not frame[field]
            .map(lambda value: isinstance(value, str) and bool(SHA256_RE.fullmatch(value)))
            .all()
        ):
            raise RevisionProtocolError("[RELEASE_ASSET_TOPOLOGY_HASH_INVALID]")
    if (
        frame["support_sha256"].duplicated().any()
        or not (frame["diagnostics_status"] == "TOPOLOGY_DIAGNOSTICS_PASS").all()
    ):
        raise RevisionProtocolError("[RELEASE_ASSET_TOPOLOGY_UNIQUENESS_INVALID]")
    for graph_index, group in frame.groupby("graph_index", sort=False):
        if len(group) > 1:
            raise RevisionProtocolError(
                f"[RELEASE_ASSET_TOPOLOGY_CROSS_DATASET_PAIRING_INVALID] {graph_index}"
            )


def _validate_scale(frame: pd.DataFrame, datasets: tuple[str, ...]) -> None:
    expected = {
        (dataset, hvg, space)
        for dataset in (*datasets, "EQUAL_DATASET")
        for hvg in (500, 1000)
        for space in ("native_hvg", "common_200_gene")
    }
    if (
        set(zip(frame["dataset"], frame["hvg"], frame["evaluation_space"], strict=False))
        != expected
    ):
        raise RevisionProtocolError("[RELEASE_ASSET_SCALE_CELL_SET_INVALID]")
    overall = frame[frame["dataset"] == "EQUAL_DATASET"]
    if not overall["inference_scope"].eq("overall_two_member_family").all():
        raise RevisionProtocolError("[RELEASE_ASSET_SCALE_SCOPE_INVALID]")
    _validate_intervals(overall)
    _probabilities(overall["two_sided_p"], "scale p")
    _probabilities(overall["bh_q_two_member_family"], "scale q")
    local = frame[frame["dataset"] != "EQUAL_DATASET"]
    if (
        not local["inference_scope"].eq("dataset_descriptive").all()
        or not local["estimate"].map(_is_finite_number).all()
        or not local[
            [
                "uncertainty_interval_95_low",
                "uncertainty_interval_95_high",
                "two_sided_p",
                "bh_q_two_member_family",
            ]
        ]
        .isna()
        .all()
        .all()
        or not local["interval_label"].eq("DESCRIPTIVE_POINT_ESTIMATE_NO_INTERVAL").all()
    ):
        raise RevisionProtocolError("[RELEASE_ASSET_SCALE_DATASET_DETAIL_INVALID]")
    if set(frame["contrast"]) != {"500_minus_200_hvg", "1000_minus_200_hvg"}:
        raise RevisionProtocolError("[RELEASE_ASSET_SCALE_CONTRAST_INVALID]")


def _validate_propagation(frame: pd.DataFrame, datasets: tuple[str, ...]) -> None:
    expected = {
        (dataset, contrast)
        for dataset in (*datasets, "EQUAL_DATASET")
        for contrast in ("string_go_minus_self_loop", "dense_minus_self_loop")
    }
    if (
        set(zip(frame["dataset"], frame["contrast"], strict=False)) != expected
        or frame.duplicated(["dataset", "contrast"]).any()
    ):
        raise RevisionProtocolError("[RELEASE_ASSET_PROPAGATION_FAMILY_INVALID]")
    overall = frame[frame["dataset"] == "EQUAL_DATASET"]
    if not overall["inference_scope"].eq("overall_two_member_family").all():
        raise RevisionProtocolError("[RELEASE_ASSET_PROPAGATION_SCOPE_INVALID]")
    _validate_intervals(overall)
    _probabilities(overall["two_sided_p"], "propagation p")
    _probabilities(overall["bh_q_two_member_family"], "propagation q")
    local = frame[frame["dataset"] != "EQUAL_DATASET"]
    if (
        not local["inference_scope"].eq("dataset_descriptive").all()
        or not local["estimate"].map(_is_finite_number).all()
        or not local[
            [
                "uncertainty_interval_95_low",
                "uncertainty_interval_95_high",
                "two_sided_p",
                "bh_q_two_member_family",
            ]
        ]
        .isna()
        .all()
        .all()
        or not local["interval_label"].eq("DESCRIPTIVE_POINT_ESTIMATE_NO_INTERVAL").all()
    ):
        raise RevisionProtocolError("[RELEASE_ASSET_PROPAGATION_DATASET_DETAIL_INVALID]")


def _validate_representativeness(frame: pd.DataFrame, datasets: tuple[str, ...]) -> None:
    metric_ids = {
        "combined_selection_stratum_total_variation",
        "absolute_standardized_log1p_condition_cell_count_difference",
        "target_mapping_coverage",
        "condition_coverage",
        "cell_coverage",
        "conditional_panel_effect",
    }
    expected = {(dataset, metric) for dataset in datasets for metric in metric_ids}
    if (
        set(zip(frame["dataset"], frame["metric_id"], strict=False)) != expected
        or frame.duplicated(["dataset", "metric_id"]).any()
    ):
        raise RevisionProtocolError("[RELEASE_ASSET_REPRESENTATIVENESS_SET_INVALID]")
    _nonnegative_integers(frame["overlap_with_legacy"], "legacy overlap")
    _nonnegative_integers(frame["overlap_with_primary"], "primary overlap")
    if (frame["overlap_with_legacy"].astype(int) != 0).any() or (
        frame["overlap_with_primary"].astype(int) != 0
    ).any():
        raise RevisionProtocolError("[RELEASE_ASSET_PANEL_OVERLAP_INVALID]")
    if (
        not frame["source_records_sha256"]
        .map(lambda value: isinstance(value, str) and bool(SHA256_RE.fullmatch(value)))
        .all()
    ):
        raise RevisionProtocolError("[RELEASE_ASSET_REPRESENTATIVENESS_SOURCE_HASH_INVALID]")
    trigger_rows = frame[
        frame["metric_id"].isin(
            {
                "combined_selection_stratum_total_variation",
                "absolute_standardized_log1p_condition_cell_count_difference",
            }
        )
    ]
    for row in trigger_rows.itertuples(index=False):
        value = _finite_values([row.value, row.threshold], "representativeness trigger")
        if _is_true(row.trigger_contribution) != bool(value[0] > value[1]):
            raise RevisionProtocolError("[RELEASE_ASSET_TRIGGER_CONTRIBUTION_INVALID]")
    declared_global = frame["global_triggered"].map(_is_true).unique()
    if len(declared_global) != 1 or bool(declared_global[0]) != bool(
        trigger_rows["trigger_contribution"].map(_is_true).any()
    ):
        raise RevisionProtocolError("[RELEASE_ASSET_GLOBAL_TRIGGER_BINDING_INVALID]")
    for row in frame.itertuples(index=False):
        if row.metric_id == "conditional_panel_effect" and _is_true(row.global_triggered):
            if int(row.panel_size) != 50 or row.status != "MEASURED":
                raise RevisionProtocolError("[RELEASE_ASSET_TRIGGERED_CONDITIONAL_INVALID]")
            _validate_interval_values(
                row.estimate,
                row.uncertainty_interval_95_low,
                row.uncertainty_interval_95_high,
                row.interval_label,
            )
        elif row.metric_id == "conditional_panel_effect":
            if row.status != "NOT_TRIGGERED" or not all(
                pd.isna(value)
                for value in (
                    row.estimate,
                    row.uncertainty_interval_95_low,
                    row.uncertainty_interval_95_high,
                )
            ):
                raise RevisionProtocolError("[RELEASE_ASSET_CONDITIONAL_NOT_TRIGGERED_INVALID]")
        elif row.status != "MEASURED" or not _is_finite_number(row.value):
            raise RevisionProtocolError("[RELEASE_ASSET_REPRESENTATIVENESS_VALUE_INVALID]")


def _validate_secondary(frame: pd.DataFrame, datasets: tuple[str, ...]) -> None:
    if not set(datasets) <= set(frame["dataset"]):
        raise RevisionProtocolError("[RELEASE_ASSET_SECONDARY_DATASET_SET_INVALID]")
    _finite(frame["estimate"], "secondary estimate")
    _positive_integers(frame["rank"], "secondary rank")
    tested = frame["inference_scope"] == "overall_family_test"
    _probabilities(frame.loc[tested, "two_sided_p"], "secondary p")
    _probabilities(frame.loc[tested, "bh_q"], "secondary q")
    if not frame.loc[~tested, ["two_sided_p", "bh_q"]].isna().all().all():
        raise RevisionProtocolError("[RELEASE_ASSET_DESCRIPTIVE_P_VALUE_INVALID]")
    _positive_integers(frame["family_denominator"], "secondary denominator")
    expected_denominators = {
        "SECONDARY_FOUR_METRIC_FAMILY": 4,
        "CONDITION_RANKING_DESCRIPTIVE": 1,
        "MIXED_COMBINED_VS_DENSE_200_SENSITIVITY": 1,
    }
    if set(frame["family_label"]) != set(expected_denominators):
        raise RevisionProtocolError("[RELEASE_ASSET_SECONDARY_FAMILY_SET_INVALID]")
    for family, group in frame.groupby("family_label", sort=False):
        if (
            group["family_denominator"].nunique() != 1
            or int(group["family_denominator"].iloc[0]) != expected_denominators[str(family)]
        ):
            raise RevisionProtocolError("[RELEASE_ASSET_SECONDARY_DENOMINATOR_INVALID]")
    primary = frame[frame["metric"] == "pearson_r"]
    if (
        set(primary["dataset"]) != set(datasets)
        or set(primary["model"]) != {"string_go", "dense"}
        or primary.duplicated(["dataset", "model"]).any()
        or not primary["ranking_analysis_id"].eq("CONDITION-RANKING-OVERLAP").all()
        or not primary["top_k"].eq(10).all()
        or not (primary["shared_condition_count"].astype(int) >= 10).all()
        or not primary["top_k_coverage_complete"].map(_is_true).all()
        or not primary["top_condition_jaccard"].between(0, 1).all()
        or not primary["shared_rank_spearman"].between(-1, 1).all()
        or not primary["ranking_status"].eq("MEASURED").all()
        or not primary["ranking_input_artifact_manifest_sha256"]
        .map(lambda value: isinstance(value, str) and bool(SHA256_RE.fullmatch(value)))
        .all()
    ):
        raise RevisionProtocolError("[RELEASE_ASSET_CONDITION_RANKING_BINDING_INVALID]")
    mixed = frame[frame["family_label"] == "MIXED_COMBINED_VS_DENSE_200_SENSITIVITY"]
    if (
        set(mixed["dataset"]) != {*datasets, "EQUAL_DATASET"}
        or mixed["dataset"].duplicated().any()
        or not mixed["metric"].eq("mixed_combined_minus_dense_pearson_r").all()
        or not mixed["model"].eq("combined_minus_dense").all()
    ):
        raise RevisionProtocolError("[RELEASE_ASSET_MIXED_SUPPORT_DETAIL_INVALID]")


def _validate_baseline(frame: pd.DataFrame, datasets: tuple[str, ...]) -> None:
    if not set(datasets) <= set(frame["dataset"]):
        raise RevisionProtocolError("[RELEASE_ASSET_BASELINE_DATASET_SET_INVALID]")
    if set(frame["baseline"]) != {
        "zero_control_delta",
        "training_condition_mean",
        "deterministic_ridge_linear",
    } or set(frame["metric"]) != {"pearson_r", "mse", "mae"}:
        raise RevisionProtocolError("[RELEASE_ASSET_BASELINE_METHOD_OR_METRIC_INVALID]")
    zero_pearson = (frame["baseline"] == "zero_control_delta") & (frame["metric"] == "pearson_r")
    if (
        not frame.loc[zero_pearson, "baseline_absolute_skill"].isna().all()
        or not frame.loc[zero_pearson, "improvement_over_baseline"].isna().all()
        or not frame.loc[zero_pearson, "baseline_skill_status"]
        .eq("UNDEFINED_CONSTANT_VECTOR")
        .all()
        or not frame.loc[zero_pearson, "improvement_status"].eq("NOT_APPLICABLE").all()
        or not frame.loc[zero_pearson, "undefined_reason_code"]
        .eq("ZERO_DELTA_PEARSON_UNDEFINED")
        .all()
    ):
        raise RevisionProtocolError("[RELEASE_ASSET_ZERO_PEARSON_SEMANTICS_INVALID]")
    measured = ~zero_pearson
    _finite(frame["neural_absolute_skill"], "neural_absolute_skill")
    for field in ("neural_absolute_skill", "baseline_absolute_skill", "improvement_over_baseline"):
        _finite(frame.loc[measured, field], field)
    if (
        not frame["neural_skill_status"].eq("MEASURED").all()
        or not frame.loc[measured, "baseline_skill_status"].eq("MEASURED").all()
        or not frame.loc[measured, "improvement_status"].eq("MEASURED").all()
        or not frame.loc[measured, "undefined_reason_code"].isna().all()
    ):
        raise RevisionProtocolError("[RELEASE_ASSET_BASELINE_STATUS_INVALID]")
    if set(frame["neural_model"]) != {"string_go", "dense"}:
        raise RevisionProtocolError("[RELEASE_ASSET_BASELINE_NEURAL_MODEL_SET_INVALID]")
    expected_conditions = 50
    expected_rows = expected_conditions * 3 * 2 * 3 * 3
    for dataset in datasets:
        group = frame[frame["dataset"] == dataset]
        if (
            set(group["hvg"].astype(int)) != {200, 500, 1000}
            or set(group["panel"]) != {"primary"}
            or any(
                local["condition"].nunique() != expected_conditions
                for _, local in group.groupby("hvg", sort=True)
            )
            or len(group) != expected_rows
            or group.duplicated(
                ["hvg", "panel", "condition", "neural_model", "baseline", "metric"]
            ).any()
        ):
            raise RevisionProtocolError("[RELEASE_ASSET_BASELINE_SUPPORT_INVALID]")
    neural = frame.loc[measured, "neural_absolute_skill"].astype(float)
    baseline = frame.loc[measured, "baseline_absolute_skill"].astype(float)
    expected = np.where(
        frame.loc[measured, "metric"] == "pearson_r", neural - baseline, baseline - neural
    )
    if not np.allclose(
        expected,
        frame.loc[measured, "improvement_over_baseline"].astype(float),
        rtol=0,
        atol=1e-15,
    ):
        raise RevisionProtocolError("[RELEASE_ASSET_BASELINE_IMPROVEMENT_INVALID]")
    if (
        not frame["baseline_artifact_sha256"]
        .map(lambda value: isinstance(value, str) and bool(SHA256_RE.fullmatch(value)))
        .all()
    ):
        raise RevisionProtocolError("[RELEASE_ASSET_BASELINE_HASH_INVALID]")


def _validate_compute(frame: pd.DataFrame, datasets: tuple[str, ...]) -> None:
    if not set(datasets) <= set(frame["dataset"]):
        raise RevisionProtocolError("[RELEASE_ASSET_COMPUTE_DATASET_SET_INVALID]")
    for field in ("hvg", "attempts", "successes", "failures"):
        _nonnegative_integers(frame[field], f"compute {field}")
    if not (
        frame["attempts"].astype(int)
        == frame["successes"].astype(int) + frame["failures"].astype(int)
    ).all():
        raise RevisionProtocolError("[RELEASE_ASSET_COMPUTE_DENOMINATOR_INVALID]")
    for field in ("training_seconds", "inference_seconds"):
        values = _finite(frame[field], field)
        if (values < 0).any():
            raise RevisionProtocolError("[RELEASE_ASSET_COMPUTE_TIME_INVALID]")
    for row in frame.itertuples(index=False):
        if row.status == "MEASURED":
            if not _is_finite_number(row.peak_device_bytes) or int(row.peak_device_bytes) < 0:
                raise RevisionProtocolError("[RELEASE_ASSET_COMPUTE_PEAK_INVALID]")
        elif row.status != "PEAK_MEMORY_NOT_AVAILABLE" or not pd.isna(row.peak_device_bytes):
            raise RevisionProtocolError("[RELEASE_ASSET_COMPUTE_PEAK_INVALID]")


def _validate_external(frame: pd.DataFrame, _: tuple[str, ...]) -> None:
    if (
        set(frame["comparator"]) != {"GEARS", "scGPT", "Geneformer"}
        or frame.duplicated(["comparator", "diagnostic_id"]).any()
    ):
        raise RevisionProtocolError("[RELEASE_ASSET_EXTERNAL_MEMBER_SET_INVALID]")
    for comparator, group in frame.groupby("comparator", sort=True):
        validity = group[group["diagnostic_id"] == "validity_decision"]
        if len(validity) != 1 or validity.iloc[0]["status"] not in {
            "PASS",
            "EXCLUDED",
            "WITHHELD",
        }:
            raise RevisionProtocolError("[RELEASE_ASSET_EXTERNAL_WITHHELD_MEMBER]")
        if validity.iloc[0]["status"] == "PASS":
            required = {
                "validity_decision",
                "prediction_nondegeneracy",
                "baseline_superiority",
                "training_loss_improvement",
                "validation_loss_improvement",
                "gene_order_binding",
                "target_mapping_binding",
                "vector_row_condition_binding",
                "split_target_disjoint_binding",
                "performance_200_hvg",
                "performance_500_hvg",
                "performance_1000_hvg",
            }
            if (
                not required <= set(group["diagnostic_id"])
                or not group["status"].isin(["PASS", "MEASURED"]).all()
            ):
                raise RevisionProtocolError("[RELEASE_ASSET_EXTERNAL_PASS_DIAGNOSTICS_INVALID]")
            positive = group[
                group["diagnostic_id"].isin(
                    {
                        "prediction_nondegeneracy",
                        "baseline_superiority",
                        "training_loss_improvement",
                        "validation_loss_improvement",
                    }
                )
            ]
            values = pd.to_numeric(positive["value"], errors="coerce")
            if len(positive) != 4 or values.isna().any() or not (values > 0).all():
                raise RevisionProtocolError(
                    "[RELEASE_ASSET_EXTERNAL_PASS_DIAGNOSTIC_VALUE_INVALID]"
                )
        elif (
            not isinstance(validity.iloc[0]["reason_code"], str)
            or len(group) != 1
            or any(
                str(value).startswith("performance_")
                for value in group["diagnostic_id"].astype(str)
            )
        ):
            raise RevisionProtocolError("[RELEASE_ASSET_EXTERNAL_EXCLUSION_INVALID]")
    for row in frame.itertuples(index=False):
        if not isinstance(row.evidence_sha256, str) or not SHA256_RE.fullmatch(row.evidence_sha256):
            raise RevisionProtocolError("[RELEASE_ASSET_EXTERNAL_HASH_INVALID]")


def _validate_intervals(
    frame: pd.DataFrame, *, estimate_column: str = "estimate", unit_interval: bool = False
) -> None:
    for row in frame.itertuples(index=False):
        _validate_interval_values(
            getattr(row, estimate_column),
            row.uncertainty_interval_95_low,
            row.uncertainty_interval_95_high,
            row.interval_label,
            unit_interval=unit_interval,
        )


def _validate_interval_values(
    estimate: Any,
    low: Any,
    high: Any,
    label: Any,
    *,
    unit_interval: bool = False,
) -> None:
    values = _finite_values([estimate, low, high], "uncertainty interval")
    if values[1] > values[0] or values[0] > values[2] or label != INTERVAL_LABEL:
        raise RevisionProtocolError("[RELEASE_ASSET_INTERVAL_SEMANTICS_INVALID]")
    if unit_interval and (values[1] < 0 or values[2] > 1):
        raise RevisionProtocolError("[RELEASE_ASSET_INTERVAL_RANGE_INVALID]")


def _finite(series: pd.Series, label: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise RevisionProtocolError(f"[RELEASE_ASSET_NONFINITE] {label}")
    return values


def _finite_values(values: Sequence[Any], label: str) -> np.ndarray:
    if any(isinstance(value, bool) for value in values):
        raise RevisionProtocolError(f"[RELEASE_ASSET_NONNUMERIC] {label}")
    try:
        output = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RevisionProtocolError(f"[RELEASE_ASSET_NONNUMERIC] {label}") from error
    if not np.isfinite(output).all():
        raise RevisionProtocolError(f"[RELEASE_ASSET_NONFINITE] {label}")
    return output


def _positive_integers(series: pd.Series, label: str) -> None:
    _integer_series(series, label, minimum=1)


def _nonnegative_integers(series: pd.Series, label: str) -> None:
    _integer_series(series, label, minimum=0)


def _integer_series(series: pd.Series, label: str, *, minimum: int) -> None:
    values = pd.to_numeric(series, errors="coerce")
    if (
        values.isna().any()
        or not np.isfinite(values.to_numpy(dtype=np.float64)).all()
        or not np.equal(values, np.floor(values)).all()
        or (values < minimum).any()
    ):
        raise RevisionProtocolError(f"[RELEASE_ASSET_INTEGER_INVALID] {label}")


def _probabilities(series: pd.Series, label: str) -> None:
    values = _finite(series, label)
    if ((values < 0) | (values > 1)).any():
        raise RevisionProtocolError(f"[RELEASE_ASSET_PROBABILITY_INVALID] {label}")


def _is_true(value: Any) -> bool:
    return value is True or value == np.bool_(True) or value == "True"


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
    )


def _records_hash(frame: pd.DataFrame) -> str:
    canonical_csv = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="",
    )
    return canonical_sha256({"columns": list(frame.columns), "canonical_csv": canonical_csv})


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n", float_format="%.17g")
    os.replace(temporary, path)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _table_tex(group: str, frame: pd.DataFrame, records_hash: str, mode: str) -> str:
    lines = [
        "% AUTO-GENERATED; DO NOT EDIT",
        f"% asset_group={group}",
        f"% semantic_records_sha256={records_hash}",
    ]
    if mode == "author-review":
        lines.append(r"\relax % author-review values intentionally withheld")
        return "\n".join(lines) + "\n"
    display = _reader_display_frame(group, frame)
    lines.extend(
        [
            r"\begin{center}",
            r"\begin{minipage}{0.98\linewidth}\centering",
            rf"\small\textbf{{{_tex_escape(GROUP_LABELS[group])}}}\par\medskip",
            r"\scriptsize",
            r"\resizebox{\linewidth}{!}{%",
            _simple_tex_tabular(display),
            r"}",
            rf"\par\smallskip\IfFileExists{{\CBACReleaseAssetRoot/release_assets/{group}.pdf}}{{%",
            rf"\includegraphics[width=0.82\linewidth]{{\CBACReleaseAssetRoot/release_assets/{group}.pdf}}%",
            r"}{\PackageError{cbac-release-assets}{Missing generated PDF asset}{}}",
            rf"\par\scriptsize Full validated table: \texttt{{{_tex_escape(group)}.csv}} "
            rf"({len(frame)} records).",
            r"\end{minipage}",
            r"\end{center}",
        ]
    )
    return "\n".join(lines) + "\n"


def _canonical_include(mode: str, semantic_hashes: Mapping[str, str]) -> str:
    lines = [
        "% AUTO-GENERATED RELEASED RESULTS INCLUDE; DO NOT EDIT",
        f"% mode={mode}",
        f"% semantic_manifest_sha256={canonical_sha256(dict(semantic_hashes))}",
    ]
    if mode == "author-review":
        lines.extend(
            [
                r"\providecommand{\CBACReleaseAssetMode}{AUTHOR-REVIEW VALUES WITHHELD}",
                r"\providecommand{\CBACReleaseAssetRoot}{github_repo/revision/generated}",
            ]
        )
        lines.extend(
            [
                r"\begin{center}",
                r"\begin{minipage}{0.96\linewidth}\centering",
                r"\small\textbf{Status of prespecified major-revision result assets}\par\medskip",
                r"\begin{tabular}{p{0.55\linewidth}p{0.34\linewidth}}",
                r"\hline",
                r"Result asset & Author-review status \\",
                r"\hline",
            ]
        )
        for group in ASSET_GROUPS:
            lines.append(
                rf"{_tex_escape(GROUP_LABELS[group])} "
                rf"{{\scriptsize\texttt{{{_tex_escape(group)}}}}} & "
                r"Pending computation and release validation \\"
            )
        lines.extend([r"\hline", r"\end{tabular}", r"\end{minipage}", r"\end{center}"])
        lines.extend(
            rf"\input{{\CBACReleaseAssetRoot/release_assets/{group}.tex}}" for group in ASSET_GROUPS
        )
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            r"\providecommand{\CBACReleaseAssetMode}{RELEASED}",
            r"\providecommand{\CBACReleaseAssetRoot}{github_repo/revision/generated}",
            r"\providecommand{\CBACReleaseConsumer}{supplement}",
            r"\def\CBACReleaseConsumerMain{main}",
            r"\def\CBACReleaseConsumerSupplement{supplement}",
            r"\def\CBACReleaseConsumerResponse{response}",
            r"\ifx\CBACReleaseConsumer\CBACReleaseConsumerMain",
        ]
    )
    lines.extend(
        rf"\input{{\CBACReleaseAssetRoot/release_assets/{group}.tex}}"
        for group in (
            "primary_summary_and_forest",
            "topology_instances_and_diagnostics",
            "secondary_ranking_and_multiplicity",
        )
    )
    lines.append(r"\else\ifx\CBACReleaseConsumer\CBACReleaseConsumerSupplement")
    lines.extend(
        rf"\input{{\CBACReleaseAssetRoot/release_assets/{group}.tex}}" for group in ASSET_GROUPS
    )
    lines.append(r"\else\ifx\CBACReleaseConsumer\CBACReleaseConsumerResponse")
    lines.extend(
        rf"\input{{\CBACReleaseAssetRoot/release_assets/{group}.tex}}"
        for group in (
            "primary_summary_and_forest",
            "topology_instances_and_diagnostics",
            "representativeness_and_conditional",
            "external_diagnostics_and_exclusions",
        )
    )
    lines.extend(
        [
            r"\else",
            r"\PackageError{cbac-release-assets}{Invalid CBACReleaseConsumer}{Use main, supplement, or response}",
            r"\fi\fi\fi",
        ]
    )
    return "\n".join(lines) + "\n"


def _reader_display_frame(group: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Return a compact, human-labelled view; the CSV remains the complete ledger."""

    if group == "primary_summary_and_forest":
        display = frame[
            [
                "dataset",
                "weighting",
                "estimate",
                "uncertainty_interval_95_low",
                "uncertainty_interval_95_high",
                "n_conditions",
            ]
        ]
    elif group == "paired_condition_distribution":
        display = (
            frame.groupby("dataset", sort=True)["delta"]
            .agg(
                n="size",
                mean="mean",
                median="median",
                q1=lambda x: x.quantile(0.25),
                q3=lambda x: x.quantile(0.75),
            )
            .reset_index()
        )
    elif group == "fraction_improved_interval":
        display = frame[
            [
                "dataset",
                "numerator_improved",
                "denominator_conditions",
                "fraction_improved",
                "uncertainty_interval_95_low",
                "uncertainty_interval_95_high",
            ]
        ]
    elif group == "target_sensitivity":
        display = frame[
            [
                "sensitivity_id",
                "status",
                "estimate",
                "uncertainty_interval_95_low",
                "uncertainty_interval_95_high",
                "reason_code",
            ]
        ]
    elif group == "topology_instances_and_diagnostics":
        display = (
            frame.groupby("dataset", sort=True)
            .agg(
                instances=("rewire_id", "size"),
                mean_effect=("estimate", "mean"),
                minimum_effect=("estimate", "min"),
                maximum_effect=("estimate", "max"),
                minimum_changed_edge_fraction=("changed_edge_fraction", "min"),
                all_diagnostics_pass=(
                    "diagnostics_status",
                    lambda x: (x == "TOPOLOGY_DIAGNOSTICS_PASS").all(),
                ),
            )
            .reset_index()
        )
    elif group == "scale_native_and_common200":
        display = frame.loc[
            frame["dataset"] == "EQUAL_DATASET",
            [
                "hvg",
                "evaluation_space",
                "estimate",
                "uncertainty_interval_95_low",
                "uncertainty_interval_95_high",
                "bh_q_two_member_family",
            ],
        ]
    elif group == "propagation":
        display = frame.loc[
            frame["dataset"] == "EQUAL_DATASET",
            [
                "contrast",
                "estimate",
                "uncertainty_interval_95_low",
                "uncertainty_interval_95_high",
                "bh_q_two_member_family",
            ],
        ]
    elif group == "representativeness_and_conditional":
        display = frame[
            [
                "dataset",
                "metric_id",
                "status",
                "value",
                "threshold",
                "trigger_contribution",
            ]
        ]
    elif group == "secondary_ranking_and_multiplicity":
        display = frame.loc[
            (frame["inference_scope"] == "overall_family_test") | (frame["metric"] == "pearson_r"),
            ["dataset", "metric", "model", "estimate", "rank", "bh_q"],
        ]
    elif group == "baseline_absolute_skill":
        selected = frame[
            (frame["hvg"] == 200)
            & ~((frame["baseline"] == "zero_control_delta") & (frame["metric"] == "pearson_r"))
        ]
        display = (
            selected.groupby(["neural_model", "baseline", "metric"], sort=True)[
                "improvement_over_baseline"
            ]
            .mean()
            .rename("mean_improvement")
            .reset_index()
        )
    elif group == "compute_and_failure_denominators":
        display = (
            frame.groupby(["model", "hvg"], sort=True)
            .agg(
                attempts=("attempts", "sum"),
                successes=("successes", "sum"),
                failures=("failures", "sum"),
                training_seconds=("training_seconds", "sum"),
                inference_seconds=("inference_seconds", "sum"),
            )
            .reset_index()
        )
    else:
        display = frame[
            [
                "comparator",
                "diagnostic_id",
                "status",
                "value",
                "reference",
                "reason_code",
            ]
        ]
    return display.head(36).reset_index(drop=True)


def _simple_tex_tabular(frame: pd.DataFrame) -> str:
    labels = [str(column).replace("_", " ").title() for column in frame.columns]
    alignment = "@{}" + "l" * len(labels) + "@{}"
    lines = [rf"\begin{{tabular}}{{{alignment}}}", r"\hline"]
    lines.append(" & ".join(_tex_escape(label) for label in labels) + r" \\")
    lines.append(r"\hline")
    for record in frame.to_dict(orient="records"):
        values = [_reader_value(record[column]) for column in frame.columns]
        lines.append(" & ".join(_tex_escape(value) for value in values) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines)


def _reader_value(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    text = str(value).replace("_", " ")
    return text if len(text) <= 42 else text[:39] + "..."


def _plot_group(
    frame: pd.DataFrame,
    group: str,
    svg_path: Path,
    pdf_path: Path,
    records_hash: str,
    mode: str,
) -> None:
    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")
    if mode == "author-review":
        fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)
        ax.text(
            0.5,
            0.5,
            "AUTHOR REVIEW — numerical results withheld pending released evidence",
            ha="center",
            va="center",
            color="dimgrey",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.set_title(group.replace("_", " ").title(), loc="left", color="dimgrey")
        ax.set_axis_off()
    elif "delta" in frame.columns:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        categories = sorted(frame["dataset"].astype(str).unique())
        for index, category in enumerate(categories):
            values = frame.loc[frame["dataset"].astype(str) == category, "delta"].astype(float)
            offsets = (
                np.linspace(-0.18, 0.18, len(values)) if len(values) > 1 else np.asarray([0.0])
            )
            ax.scatter(
                np.full(len(values), index) + offsets,
                values,
                alpha=0.4,
                color="#4575b4",
                edgecolors="none",
                s=18,
            )
        ax.axhline(0, color="lightgrey", linewidth=0.8)
        ax.set_xticks(range(len(categories)), categories, rotation=20, ha="right")
        ax.set_ylabel("Paired STRING–GO minus dense effect", color="dimgrey")
        ax.grid(False)
    elif (
        "estimate" in frame.columns
        and pd.to_numeric(frame["estimate"], errors="coerce").notna().any()
    ):
        plot = frame.loc[pd.to_numeric(frame["estimate"], errors="coerce").notna()].copy()
        plot["estimate"] = plot["estimate"].astype(float)
        plot["plot_label"] = _plot_labels(plot)
        plot = plot.sort_values("estimate", kind="mergesort").reset_index(drop=True)
        fig_height = max(3.5, 1.0 + len(plot) * 0.35)
        fig, ax = plt.subplots(figsize=(10, fig_height), dpi=300)
        palette = sns.cubehelix_palette(6, rot=-0.25, light=0.7)
        y = np.arange(len(plot))
        if {
            "uncertainty_interval_95_low",
            "uncertainty_interval_95_high",
        } <= set(plot.columns):
            low = pd.to_numeric(plot["uncertainty_interval_95_low"], errors="coerce")
            high = pd.to_numeric(plot["uncertainty_interval_95_high"], errors="coerce")
            measured = low.notna() & high.notna()
            ax.hlines(
                y=y[measured],
                xmin=low[measured],
                xmax=high[measured],
                color="grey",
                alpha=0.4,
                linewidth=3,
                zorder=0,
            )
            ax.scatter(low[measured], y[measured], s=35, color=palette[5], edgecolors="white")
            ax.scatter(high[measured], y[measured], s=35, color=palette[5], edgecolors="white")
        ax.scatter(plot["estimate"], y, s=70, color=palette[2], edgecolors="white", zorder=4)
        ax.set_yticks(y, plot["plot_label"])
        ax.axvline(0, color="lightgrey", linewidth=0.8)
        ax.set_xlabel("Estimate", color="dimgrey")
        ax.grid(False)
    else:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        status_column = next(
            (column for column in ("decision", "status", "metric") if column in frame),
            frame.columns[0],
        )
        counts = frame[status_column].astype(str).value_counts().sort_values()
        ax.barh(counts.index, counts.values, color="#4575b4")
        for index, value in enumerate(counts.values):
            ax.text(value, index, f" {value}", va="center", color="dimgrey")
        ax.set_xlabel("Validated records", color="dimgrey")
        ax.grid(False)
    sns.despine(left=True, bottom=True)
    ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
    ax.set_title(group.replace("_", " ").title(), fontsize=14, loc="left", pad=7, color="dimgrey")
    fig.text(
        0.98,
        0.01,
        f"Validated records: {len(frame)}",
        ha="right",
        va="bottom",
        fontsize=9,
        color="dimgrey",
        style="italic",
    )
    svg_temporary = svg_path.with_name(f".{svg_path.name}.tmp.svg")
    pdf_temporary = pdf_path.with_name(f".{pdf_path.name}.tmp.pdf")
    fig.savefig(
        svg_temporary,
        format="svg",
        dpi=300,
        bbox_inches="tight",
        metadata={"Title": group, "Description": records_hash},
    )
    fig.savefig(
        pdf_temporary,
        format="pdf",
        dpi=300,
        bbox_inches="tight",
        metadata={"Title": group, "Subject": records_hash},
    )
    plt.close(fig)
    os.replace(svg_temporary, svg_path)
    os.replace(pdf_temporary, pdf_path)


def _plot_labels(frame: pd.DataFrame) -> pd.Series:
    fields = [
        field
        for field in (
            "dataset",
            "contrast",
            "rewire_id",
            "evaluation_space",
            "sensitivity_id",
            "analysis",
        )
        if field in frame
    ]
    if not fields:
        return pd.Series([str(index + 1) for index in range(len(frame))])
    return frame[fields].astype(str).agg(" | ".join, axis=1)


def _entry(root: Path, path: Path, asset_group: str, media_type: str) -> dict[str, Any]:
    return {
        "asset_group": asset_group,
        "source_id": path.relative_to(root).as_posix(),
        "media_type": media_type,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _resolve_relative(root: Path, source_id: Any) -> Path | None:
    if not isinstance(source_id, str) or not source_id:
        return None
    relative = Path(source_id)
    if relative.is_absolute():
        return None
    resolved = (root / relative).resolve()
    return resolved if resolved.is_relative_to(root) else None


def _tex_escape(value: str) -> str:
    output = value
    for source, replacement in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
        ("{", r"\{"),
        ("}", r"\}"),
    ):
        output = output.replace(source, replacement)
    return output


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant {value!r} is prohibited")

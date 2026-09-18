"""Cross-dataset OR gate for the conditional non-overlap sensitivity panel."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import canonical_sha256, file_sha256
from .errors import RevisionProtocolError
from .protocol import load_protocol


def aggregate_global_trigger(
    summaries: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], protocol_hash: str
) -> dict[str, Any]:
    """Apply a global OR: one local trigger requires all four datasets to execute."""

    expected_datasets = tuple(sorted(protocol["datasets"]))
    trigger_protocol = protocol["condition_panel"]["conditional_sensitivity_trigger"]
    tv_threshold = float(trigger_protocol["combined_selection_stratum_total_variation_distance_gt"])
    cell_count_threshold = float(
        trigger_protocol["absolute_standardized_log1p_condition_cell_count_difference_gt"]
    )
    by_dataset: dict[str, Mapping[str, Any]] = {}
    failures: list[dict[str, str]] = []
    for summary in summaries:
        dataset = str(summary.get("dataset", ""))
        if dataset in by_dataset:
            failures.append({"reason_code": "DUPLICATE_DATASET_TRIGGER_SUMMARY", "detail": dataset})
        by_dataset[dataset] = summary
    missing = sorted(set(expected_datasets) - set(by_dataset))
    extra = sorted(set(by_dataset) - set(expected_datasets))
    if missing or extra:
        failures.append(
            {
                "reason_code": "GLOBAL_TRIGGER_DATASET_SET_MISMATCH",
                "detail": f"missing={missing}; extra={extra}",
            }
        )
    dataset_rows = []
    expected_panel_size = int(protocol["condition_panel"]["primary_size"])
    for dataset in expected_datasets:
        summary = by_dataset.get(dataset, {})
        summary_copy = dict(summary)
        declared_summary_hash = summary_copy.pop("summary_hash", None)
        if declared_summary_hash != canonical_sha256(summary_copy):
            failures.append({"reason_code": "PREFLIGHT_SUMMARY_HASH_MISMATCH", "detail": dataset})
        if summary.get("input_hashes", {}).get("protocol") != protocol_hash:
            failures.append(
                {
                    "reason_code": "GLOBAL_TRIGGER_PROTOCOL_HASH_MISMATCH",
                    "detail": dataset,
                }
            )
        identity_contract = {
            "panel": "primary",
            "hvg": 200,
            "analysis_block": "topology_primary",
            "requested_panel_size": expected_panel_size,
            "n_selected_conditions": expected_panel_size,
            "can_execute": True,
        }
        for field, expected in identity_contract.items():
            if summary.get(field) != expected:
                failures.append(
                    {
                        "reason_code": "GLOBAL_TRIGGER_PREFLIGHT_IDENTITY_MISMATCH",
                        "detail": f"{dataset}:{field}:expected={expected!r};observed={summary.get(field)!r}",
                    }
                )
        if summary.get("blockers") != []:
            failures.append({"reason_code": "GLOBAL_TRIGGER_PREFLIGHT_BLOCKED", "detail": dataset})
        input_hashes = summary.get("input_hashes", {})
        for required_hash in (
            "dataset",
            "dataset_passport",
            "condition_panel_manifest",
            "panel_representativeness",
            "target_encoding_audit",
            "environment_manifest",
            "environment_lock",
            "git_provenance",
            "nested_gene_panel_manifest",
            "precision_design_registry",
            "hyperparameter_provenance",
        ):
            value = input_hashes.get(required_hash)
            if not isinstance(value, str) or len(value) != 64:
                failures.append(
                    {
                        "reason_code": "GLOBAL_TRIGGER_REQUIRED_INPUT_HASH_MISSING",
                        "detail": f"{dataset}:{required_hash}",
                    }
                )
        panel_manifest = summary.get("condition_panel_manifest")
        panel_manifest_hash = None
        if isinstance(panel_manifest, dict):
            panel_copy = dict(panel_manifest)
            panel_manifest_hash = panel_copy.pop("manifest_hash", None)
            if panel_manifest_hash != canonical_sha256(panel_copy):
                failures.append({"reason_code": "PANEL_MANIFEST_HASH_MISMATCH", "detail": dataset})
            if panel_manifest.get("dataset") != dataset:
                failures.append(
                    {"reason_code": "PANEL_MANIFEST_DATASET_MISMATCH", "detail": dataset}
                )
            primary_conditions = panel_manifest.get("primary", {}).get(
                "ordered_canonical_condition_ids", []
            )
            if (
                len(primary_conditions) != expected_panel_size
                or len(set(primary_conditions)) != expected_panel_size
                or primary_conditions != summary.get("selected_conditions")
            ):
                failures.append(
                    {"reason_code": "PANEL_MANIFEST_PRIMARY_SET_MISMATCH", "detail": dataset}
                )
            if panel_manifest.get("non_overlap_status") != "PASS" or panel_manifest.get(
                "primary_sensitivity_overlap"
            ):
                failures.append(
                    {"reason_code": "PANEL_MANIFEST_NONOVERLAP_FAILED", "detail": dataset}
                )
            legacy = panel_manifest.get("legacy_first50", {})
            if (
                legacy.get("n_conditions") != 50
                or len(set(legacy.get("ordered_condition_ids", []))) != 50
                or legacy.get("binding_source") != "hash_verified_dataset_passport"
                or legacy.get("primary_overlap")
                or legacy.get("sensitivity_overlap")
                or panel_manifest.get("primary_panel_contract_status") != "PASS"
            ):
                failures.append(
                    {"reason_code": "LEGACY_PANEL_EXCLUSION_CONTRACT_FAILED", "detail": dataset}
                )
            sensitivity_conditions = panel_manifest.get("sensitivity", {}).get(
                "ordered_canonical_condition_ids", []
            )
            if panel_manifest.get("sensitivity_trigger", {}).get("sensitivity_triggered") and (
                len(sensitivity_conditions) != expected_panel_size
                or panel_manifest.get("conditional_panel_contract_status") != "PASS"
            ):
                failures.append(
                    {
                        "reason_code": "TRIGGERED_CONDITIONAL_PANEL_INSUFFICIENT",
                        "detail": dataset,
                    }
                )
        else:
            failures.append({"reason_code": "PANEL_MANIFEST_MISSING", "detail": dataset})
        if panel_manifest_hash != summary.get(
            "condition_panel_manifest_hash"
        ) or panel_manifest_hash != input_hashes.get("condition_panel_manifest"):
            failures.append({"reason_code": "PANEL_MANIFEST_BINDING_MISMATCH", "detail": dataset})
        environment_manifest = summary.get("environment_manifest")
        if isinstance(environment_manifest, dict):
            environment_copy = dict(environment_manifest)
            environment_hash = environment_copy.pop("manifest_hash", None)
            if (
                environment_hash != canonical_sha256(environment_copy)
                or environment_hash != input_hashes.get("environment_manifest")
                or environment_manifest.get("environment_lock_sha256")
                != input_hashes.get("environment_lock")
                or environment_manifest.get("fixture_status") != "PRODUCTION"
            ):
                failures.append(
                    {"reason_code": "ENVIRONMENT_MANIFEST_BINDING_MISMATCH", "detail": dataset}
                )
        else:
            failures.append({"reason_code": "ENVIRONMENT_MANIFEST_MISSING", "detail": dataset})
        git_provenance = summary.get("git_provenance")
        git_provenance_hash = None
        if isinstance(git_provenance, dict):
            git_copy = dict(git_provenance)
            git_provenance_hash = git_copy.pop("provenance_hash", None)
            if (
                git_provenance_hash != canonical_sha256(git_copy)
                or git_provenance_hash != input_hashes.get("git_provenance")
                or git_provenance.get("code_tree_sha256") != summary.get("code_hash")
                or git_provenance.get("git_worktree_status") not in {"CLEAN", "DIRTY"}
            ):
                failures.append(
                    {"reason_code": "GIT_PROVENANCE_BINDING_MISMATCH", "detail": dataset}
                )
        else:
            failures.append({"reason_code": "GIT_PROVENANCE_MISSING", "detail": dataset})
        if not isinstance(summary.get("code_hash"), str) or len(summary["code_hash"]) != 64:
            failures.append({"reason_code": "CODE_HASH_MISSING", "detail": dataset})
        trigger = summary.get("sensitivity_trigger", {})
        if "sensitivity_triggered" not in trigger:
            failures.append({"reason_code": "LOCAL_TRIGGER_FIELDS_MISSING", "detail": dataset})
        if (
            trigger.get("trigger_tv_threshold") != tv_threshold
            or trigger.get("trigger_cell_count_threshold") != cell_count_threshold
        ):
            failures.append({"reason_code": "LOCAL_TRIGGER_THRESHOLD_MISMATCH", "detail": dataset})
        representativeness_hash = summary.get("panel_representativeness_table_hash")
        if (
            not isinstance(representativeness_hash, str)
            or len(representativeness_hash) != 64
            or representativeness_hash != input_hashes.get("panel_representativeness")
            or representativeness_hash != trigger.get("representativeness_table_hash")
            or representativeness_hash
            != (panel_manifest.get("representativeness_table_hash") if panel_manifest else None)
        ):
            failures.append(
                {"reason_code": "REPRESENTATIVENESS_TABLE_BINDING_MISMATCH", "detail": dataset}
            )
        try:
            total_variation = float(trigger["trigger_total_variation_distance"])
            cell_count_difference = float(
                trigger["trigger_abs_standardized_log1p_cell_count_difference"]
            )
        except (KeyError, TypeError, ValueError):
            total_variation = math.nan
            cell_count_difference = math.nan
            failures.append(
                {"reason_code": "LOCAL_TRIGGER_NUMERIC_FIELDS_INVALID", "detail": dataset}
            )
        if (
            not math.isfinite(total_variation)
            or not math.isfinite(cell_count_difference)
            or total_variation < 0
            or cell_count_difference < 0
        ):
            failures.append(
                {"reason_code": "LOCAL_TRIGGER_NUMERIC_FIELDS_INVALID", "detail": dataset}
            )
            recomputed_local = False
        else:
            recomputed_local = (
                total_variation > tv_threshold or cell_count_difference > cell_count_threshold
            )
        if trigger.get("sensitivity_triggered") is not recomputed_local:
            failures.append({"reason_code": "LOCAL_TRIGGER_DECISION_MISMATCH", "detail": dataset})
        dataset_rows.append(
            {
                "dataset": dataset,
                "local_triggered": recomputed_local,
                "trigger_total_variation_distance": total_variation,
                "trigger_abs_standardized_log1p_cell_count_difference": cell_count_difference,
                "conditional_sensitivity_feasible": bool(
                    summary.get("conditional_sensitivity_feasible", False)
                ),
                "summary_hash": declared_summary_hash,
                "condition_panel_manifest_hash": panel_manifest_hash,
                "panel_representativeness_table_hash": representativeness_hash,
                "dataset_hash": input_hashes.get("dataset"),
                "dataset_passport_hash": input_hashes.get("dataset_passport"),
                "target_encoding_audit_hash": input_hashes.get("target_encoding_audit"),
                "environment_manifest_hash": input_hashes.get("environment_manifest"),
                "environment_lock_hash": input_hashes.get("environment_lock"),
                "git_provenance_hash": git_provenance_hash,
                "git_commit": (
                    git_provenance.get("git_commit") if isinstance(git_provenance, dict) else None
                ),
                "git_worktree_status": (
                    git_provenance.get("git_worktree_status")
                    if isinstance(git_provenance, dict)
                    else None
                ),
                "git_status_hash": (
                    git_provenance.get("git_status_hash")
                    if isinstance(git_provenance, dict)
                    else None
                ),
                "code_hash": summary.get("code_hash"),
            }
        )
    global_triggered = any(row["local_triggered"] for row in dataset_rows)
    if global_triggered and not all(
        row["conditional_sensitivity_feasible"] for row in dataset_rows
    ):
        failures.append(
            {
                "reason_code": "GLOBAL_TRIGGERED_PANEL_INFEASIBLE",
                "detail": "At least one dataset lacks 50 non-overlap conditions",
            }
        )
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "protocol_id": protocol["protocol_id"],
        "protocol_hash": protocol_hash,
        "rule": "global_OR_across_datasets",
        "thresholds": {
            "total_variation_distance_exclusive_gt": tv_threshold,
            "absolute_standardized_log1p_cell_count_difference_exclusive_gt": (
                cell_count_threshold
            ),
        },
        "dataset_triggers": dataset_rows,
        "global_triggered": global_triggered,
        "conditional_execution_required": global_triggered,
        "execution_datasets_if_triggered": list(expected_datasets),
        "conditional_fits_required": (
            int(protocol["planned_fit_counts"]["conditional_non_overlap_fits_if_triggered"])
            if global_triggered
            else 0
        ),
        "status": "BLOCKED" if failures else "PASS",
        "failures": failures,
    }
    payload["manifest_hash"] = canonical_sha256(payload)
    return payload


def read_global_trigger_manifest(
    path: Path, protocol: Mapping[str, Any], protocol_hash: str
) -> dict[str, Any]:
    """Read and verify a previously aggregated global trigger manifest."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise RevisionProtocolError(f"[GLOBAL_TRIGGER_MANIFEST_INVALID] {error}") from error
    if not isinstance(payload, dict):
        raise RevisionProtocolError("[GLOBAL_TRIGGER_MANIFEST_INVALID] root must be an object")
    expected_hash = payload.pop("manifest_hash", None)
    observed_hash = canonical_sha256(payload)
    payload["manifest_hash"] = expected_hash
    if expected_hash != observed_hash:
        raise RevisionProtocolError("[GLOBAL_TRIGGER_MANIFEST_HASH_MISMATCH]")
    if payload.get("protocol_hash") != protocol_hash:
        raise RevisionProtocolError("[GLOBAL_TRIGGER_PROTOCOL_HASH_MISMATCH]")
    if payload.get("status") != "PASS":
        raise RevisionProtocolError("[GLOBAL_TRIGGER_MANIFEST_BLOCKED]")
    expected_datasets = sorted(protocol["datasets"])
    if payload.get("execution_datasets_if_triggered") != expected_datasets:
        raise RevisionProtocolError("[GLOBAL_TRIGGER_DATASET_SET_MISMATCH]")
    rows = payload.get("dataset_triggers")
    if (
        not isinstance(rows, list)
        or sorted(row.get("dataset") for row in rows) != expected_datasets
    ):
        raise RevisionProtocolError("[GLOBAL_TRIGGER_DATASET_SET_MISMATCH]")
    recomputed_local: list[bool] = []
    trigger_protocol = protocol["condition_panel"]["conditional_sensitivity_trigger"]
    tv_threshold = float(trigger_protocol["combined_selection_stratum_total_variation_distance_gt"])
    cell_count_threshold = float(
        trigger_protocol["absolute_standardized_log1p_condition_cell_count_difference_gt"]
    )
    if payload.get("thresholds") != {
        "total_variation_distance_exclusive_gt": tv_threshold,
        "absolute_standardized_log1p_cell_count_difference_exclusive_gt": cell_count_threshold,
    }:
        raise RevisionProtocolError("[GLOBAL_TRIGGER_THRESHOLD_CONTRACT_MISMATCH]")
    for row in rows:
        try:
            total_variation = float(row["trigger_total_variation_distance"])
            cell_count_difference = float(
                row["trigger_abs_standardized_log1p_cell_count_difference"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RevisionProtocolError("[GLOBAL_TRIGGER_NUMERIC_FIELDS_INVALID]") from error
        if (
            not math.isfinite(total_variation)
            or not math.isfinite(cell_count_difference)
            or total_variation < 0
            or cell_count_difference < 0
        ):
            raise RevisionProtocolError("[GLOBAL_TRIGGER_NUMERIC_FIELDS_INVALID]")
        local = total_variation > tv_threshold or cell_count_difference > cell_count_threshold
        if row.get("local_triggered") is not local:
            raise RevisionProtocolError("[GLOBAL_TRIGGER_LOCAL_DECISION_MISMATCH]")
        for hash_field in (
            "summary_hash",
            "condition_panel_manifest_hash",
            "panel_representativeness_table_hash",
            "dataset_hash",
            "dataset_passport_hash",
            "target_encoding_audit_hash",
            "environment_manifest_hash",
            "environment_lock_hash",
            "git_provenance_hash",
            "git_status_hash",
            "code_hash",
        ):
            value = row.get(hash_field)
            if not isinstance(value, str) or len(value) != 64:
                raise RevisionProtocolError("[GLOBAL_TRIGGER_INPUT_BINDING_INVALID]")
        git_commit = row.get("git_commit")
        if (
            not isinstance(git_commit, str)
            or len(git_commit) != 40
            or any(character not in "0123456789abcdef" for character in git_commit)
            or row.get("git_worktree_status") not in {"CLEAN", "DIRTY"}
        ):
            raise RevisionProtocolError("[GLOBAL_TRIGGER_GIT_BINDING_INVALID]")
        recomputed_local.append(local)
    recomputed_global = any(recomputed_local)
    if payload.get("global_triggered") is not recomputed_global:
        raise RevisionProtocolError("[GLOBAL_TRIGGER_GLOBAL_DECISION_MISMATCH]")
    if payload.get("conditional_execution_required") is not recomputed_global:
        raise RevisionProtocolError("[GLOBAL_TRIGGER_EXECUTION_DECISION_MISMATCH]")
    expected_fits = (
        int(protocol["planned_fit_counts"]["conditional_non_overlap_fits_if_triggered"])
        if recomputed_global
        else 0
    )
    if payload.get("conditional_fits_required") != expected_fits:
        raise RevisionProtocolError("[GLOBAL_TRIGGER_FIT_COUNT_MISMATCH]")
    if recomputed_global and not all(
        bool(row.get("conditional_sensitivity_feasible")) for row in rows
    ):
        raise RevisionProtocolError("[GLOBAL_TRIGGERED_PANEL_INFEASIBLE]")
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight-summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_argument_parser().parse_args(arguments)
    protocol = load_protocol(parsed.protocol)
    protocol_hash = file_sha256(parsed.protocol)
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in parsed.preflight_summary]
    payload = aggregate_global_trigger(summaries, protocol, protocol_hash)
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

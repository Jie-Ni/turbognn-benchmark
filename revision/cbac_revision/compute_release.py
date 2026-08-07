"""Release gate for cumulative success, failure, retry, and hardware accounting."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .artifacts import canonical_sha256, file_sha256

KEY_COLUMNS = ("dataset", "hvg", "panel", "arm", "condition", "seed")
TIMING_COLUMNS = (
    "wall_seconds",
    "training_wall_seconds",
    "inference_wall_seconds",
    "checkpoint_io_wall_seconds",
    "other_overhead_wall_seconds",
    "useful_compute_seconds",
    "failed_attempt_wall_seconds",
    "retry_overhead_wall_seconds",
    "accelerator_hours",
    "device_hours",
)
ACCOUNTING_COLUMNS = (
    "fit_scope_compute_seconds",
    "scheduler_allocation_seconds",
    "failed_preempted_allocation_seconds",
    "retry_allocation_overhead_seconds",
    "serialization_registry_other_nonfit_seconds",
    "unattributed_interrupted_allocation_seconds",
)

RUNTIME_BINDINGS = {
    "wall_seconds": "runtime_wall_seconds",
    "training_wall_seconds": "runtime_training_wall_seconds",
    "inference_wall_seconds": "runtime_inference_wall_seconds",
    "checkpoint_io_wall_seconds": "runtime_checkpoint_io_wall_seconds",
    "other_overhead_wall_seconds": "runtime_other_overhead_wall_seconds",
    "useful_compute_seconds": "runtime_useful_compute_seconds",
    "device_hours": "runtime_device_hours",
    "accelerator_hours": "runtime_accelerator_hours",
    "attempt_index": "runtime_attempt_index",
    "retry_count": "runtime_retry_count",
    "device": "runtime_device",
    "host": "runtime_host",
    "accelerator_model": "runtime_accelerator_model",
    "accelerator_count": "runtime_accelerator_count",
    "cpu_model": "runtime_cpu_model",
    "logical_cpu_count": "runtime_logical_cpu_count",
    "host_ram_bytes": "runtime_host_ram_bytes",
    "peak_device_memory_bytes": "runtime_peak_device_memory_bytes",
    "peak_memory_status": "runtime_peak_memory_status",
    "artifact_hash": "artifact_hash",
    "checkpoint_hash": "checkpoint_hash",
}


def release_measured_compute(
    registry_paths: Sequence[Path], metric_frame: pd.DataFrame
) -> dict[str, Any]:
    """Verify run registries and require exactly one successful attempt per artifact key."""

    failures: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    registry_hashes: list[str] = []
    for path in sorted({Path(value).resolve() for value in registry_paths}):
        try:
            registry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(
                {"reason_code": "COMPUTE_REGISTRY_INVALID", "detail": f"{path.name}:{error}"}
            )
            continue
        if not isinstance(registry, dict):
            failures.append({"reason_code": "COMPUTE_REGISTRY_INVALID", "detail": path.name})
            continue
        payload = dict(registry)
        registry_hash = payload.pop("registry_hash", None)
        if registry_hash != canonical_sha256(payload):
            failures.append({"reason_code": "COMPUTE_REGISTRY_HASH_MISMATCH", "detail": path.name})
            continue
        registry_hashes.append(str(registry_hash))
        source_id = str(registry.get("attempt_ledger_source_id", ""))
        ledger_path = Path(source_id)
        if not source_id or ledger_path.is_absolute() or ".." in ledger_path.parts:
            failures.append({"reason_code": "COMPUTE_LEDGER_SOURCE_INVALID", "detail": path.name})
            continue
        ledger_path = (path.parent / ledger_path).resolve()
        if not ledger_path.is_file():
            failures.append({"reason_code": "COMPUTE_LEDGER_MISSING", "detail": path.name})
            continue
        if file_sha256(ledger_path) != registry.get("attempt_ledger_file_sha256"):
            failures.append(
                {"reason_code": "COMPUTE_LEDGER_FILE_HASH_MISMATCH", "detail": path.name}
            )
            continue
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(
                {"reason_code": "COMPUTE_LEDGER_INVALID", "detail": f"{path.name}:{error}"}
            )
            continue
        if not isinstance(ledger, list) or not all(isinstance(row, dict) for row in ledger):
            failures.append({"reason_code": "COMPUTE_LEDGER_INVALID", "detail": path.name})
            continue
        if canonical_sha256(ledger) != registry.get("attempt_ledger_hash"):
            failures.append(
                {"reason_code": "COMPUTE_LEDGER_CONTENT_HASH_MISMATCH", "detail": path.name}
            )
            continue
        for row in ledger:
            _validate_reconciliation_source(row, path.parent, failures)
        if registry.get("status") != "RELEASED":
            failures.append({"reason_code": "COMPUTE_RUN_REGISTRY_WITHHELD", "detail": path.name})
        rows.extend(dict(row) for row in ledger)
    if not registry_paths:
        failures.append({"reason_code": "COMPUTE_REGISTRIES_MISSING", "detail": "none supplied"})
    missing_metric_columns = sorted(
        (set(KEY_COLUMNS) | set(RUNTIME_BINDINGS.values())) - set(metric_frame.columns)
    )
    if missing_metric_columns:
        failures.append(
            {
                "reason_code": "COMPUTE_METRIC_KEYS_MISSING",
                "detail": str(missing_metric_columns),
            }
        )
    expected_keys = (
        {
            (str(dataset), int(hvg), str(panel), str(arm), str(condition), int(seed))
            for dataset, hvg, panel, arm, condition, seed in metric_frame[
                list(KEY_COLUMNS)
            ].itertuples(index=False, name=None)
        }
        if not missing_metric_columns
        else set()
    )
    metric_by_key = (
        {
            tuple(
                int(value) if name in {"hvg", "seed"} else str(value)
                for name, value in zip(KEY_COLUMNS, key, strict=True)
            ): row
            for key, row in metric_frame.set_index(list(KEY_COLUMNS)).iterrows()
        }
        if not missing_metric_columns
        else {}
    )
    if not missing_metric_columns and metric_frame.duplicated(list(KEY_COLUMNS)).any():
        failures.append(
            {"reason_code": "COMPUTE_METRIC_KEY_DUPLICATE", "detail": "artifact metric frame"}
        )
    by_key: dict[tuple[str, int, str, str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        try:
            key = (
                str(row["dataset"]),
                int(row["hvg"]),
                str(row["panel"]),
                str(row["arm"]),
                str(row["condition"]),
                int(row["seed"]),
            )
            attempt_index = int(row["attempt_index"])
            retry_count = int(row["retry_count"])
            values = [float(row[column]) for column in (*TIMING_COLUMNS, *ACCOUNTING_COLUMNS)]
        except (KeyError, TypeError, ValueError) as error:
            failures.append({"reason_code": "COMPUTE_ATTEMPT_ROW_INVALID", "detail": str(error)})
            continue
        if (
            attempt_index <= 0
            or retry_count != attempt_index - 1
            or any(not math.isfinite(value) or value < 0 for value in values)
        ):
            failures.append({"reason_code": "COMPUTE_ATTEMPT_ROW_INVALID", "detail": str(key)})
        phase_total = sum(float(row[column]) for column in TIMING_COLUMNS[1:5])
        if not math.isclose(phase_total, float(row["wall_seconds"]), rel_tol=1e-6, abs_tol=1e-6):
            failures.append({"reason_code": "COMPUTE_PHASE_TIMING_MISMATCH", "detail": str(key)})
        if not math.isclose(
            float(row["useful_compute_seconds"]),
            float(row["training_wall_seconds"]) + float(row["inference_wall_seconds"]),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            failures.append({"reason_code": "COMPUTE_USEFUL_TIMING_MISMATCH", "detail": str(key)})
        if not math.isclose(
            float(row["fit_scope_compute_seconds"]),
            float(row["useful_compute_seconds"]),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ) or not math.isclose(
            float(row["scheduler_allocation_seconds"]),
            float(row["wall_seconds"]),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            failures.append({"reason_code": "COMPUTE_SCOPE_TIMING_MISMATCH", "detail": str(key)})
        success = row.get("execution_status") == "SUCCESS"
        if (
            row.get("timing_reconciliation_status")
            not in {"FINALIZED_MEASURED", "FINALIZED_SCHEDULER_RECONCILED"}
            or not str(row.get("attempt_started_at_utc", ""))
            or not str(row.get("attempt_finished_at_utc", ""))
            or not str(row.get("scheduler_job_id", ""))
        ):
            failures.append(
                {"reason_code": "COMPUTE_ATTEMPT_TIMING_UNRECONCILED", "detail": str(key)}
            )
        peak_status = str(row.get("peak_memory_status", ""))
        peak_value = row.get("peak_device_memory_bytes")
        if (
            not peak_status
            or (peak_value is None and not peak_status.startswith("NOT_"))
            or (
                peak_value is not None
                and (
                    isinstance(peak_value, bool)
                    or int(peak_value) < 0
                    or not peak_status.startswith("MEASURED_")
                )
            )
        ):
            failures.append(
                {"reason_code": "COMPUTE_PEAK_MEMORY_RECORD_INVALID", "detail": str(key)}
            )
        if success and (
            not _is_sha256(row.get("artifact_hash")) or not _is_sha256(row.get("checkpoint_hash"))
        ):
            failures.append(
                {"reason_code": "COMPUTE_SUCCESS_ARTIFACT_BINDING_MISSING", "detail": str(key)}
            )
        expected_failed = 0.0 if success else float(row["wall_seconds"])
        if not math.isclose(
            float(row["failed_attempt_wall_seconds"]), expected_failed, abs_tol=1e-6
        ) or not math.isclose(
            float(row["failed_preempted_allocation_seconds"]), expected_failed, abs_tol=1e-6
        ):
            failures.append({"reason_code": "COMPUTE_FAILURE_TIMING_MISMATCH", "detail": str(key)})
        by_key.setdefault(key, []).append(row)
    observed_keys = set(by_key)
    if observed_keys != expected_keys:
        failures.append(
            {
                "reason_code": "COMPUTE_FIT_KEY_COVERAGE_MISMATCH",
                "detail": (
                    f"missing={len(expected_keys-observed_keys)};extra={len(observed_keys-expected_keys)}"
                ),
            }
        )
    for key, attempts in by_key.items():
        ordered_attempts = sorted(attempts, key=lambda row: int(row["attempt_index"]))
        indices = [int(row["attempt_index"]) for row in ordered_attempts]
        successes = sum(row.get("execution_status") == "SUCCESS" for row in attempts)
        if (
            indices != list(range(1, len(indices) + 1))
            or successes != 1
            or ordered_attempts[-1].get("execution_status") != "SUCCESS"
        ):
            failures.append(
                {"reason_code": "COMPUTE_ATTEMPT_SEQUENCE_OR_SUCCESS_MISMATCH", "detail": str(key)}
            )
            continue
        metric_row = metric_by_key.get(key)
        if metric_row is None:
            continue
        success_row = ordered_attempts[-1]
        for attempt_field, metric_field in RUNTIME_BINDINGS.items():
            observed = success_row.get(attempt_field)
            expected = metric_row.get(metric_field)
            if _binding_values_differ(observed, expected):
                failures.append(
                    {
                        "reason_code": "COMPUTE_ARTIFACT_RUNTIME_BINDING_MISMATCH",
                        "detail": f"{key}:{attempt_field}",
                    }
                )
    unique_failures = _unique_failures(failures)

    def total(column: str) -> float:
        return float(sum(float(row.get(column, 0.0)) for row in rows))

    successful_rows = [row for row in rows if row.get("execution_status") == "SUCCESS"]
    failed_rows = [row for row in rows if row.get("execution_status") != "SUCCESS"]
    retry_rows = [row for row in rows if int(row.get("attempt_index", 1)) > 1]
    attempt_group_summaries = []
    grouped_rows: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped_rows.setdefault(
            (str(row.get("dataset")), str(row.get("arm")), int(row.get("hvg", 0))), []
        ).append(row)
    for (dataset, arm, hvg), group in sorted(grouped_rows.items()):
        peaks = [
            int(row["peak_device_memory_bytes"])
            for row in group
            if row.get("peak_device_memory_bytes") is not None
        ]
        attempt_group_summaries.append(
            {
                "dataset": dataset,
                "model": arm,
                "hvg": hvg,
                "attempts": len(group),
                "successes": sum(row.get("execution_status") == "SUCCESS" for row in group),
                "failures": sum(row.get("execution_status") != "SUCCESS" for row in group),
                "training_seconds": float(
                    sum(float(row["training_wall_seconds"]) for row in group)
                ),
                "inference_seconds": float(
                    sum(float(row["inference_wall_seconds"]) for row in group)
                ),
                "peak_device_bytes": max(peaks) if peaks else None,
                "status": "MEASURED" if peaks else "PEAK_MEMORY_NOT_AVAILABLE",
            }
        )

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "registry_id": "MEASURED-COMPUTE",
        "status": "RELEASED" if not unique_failures else "WITHHELD",
        "expected_successful_fit_keys": len(expected_keys),
        "observed_fit_keys": len(observed_keys),
        "total_attempts": len(rows),
        "successful_attempts": sum(row.get("execution_status") == "SUCCESS" for row in rows),
        "failed_attempts": sum(row.get("execution_status") != "SUCCESS" for row in rows),
        "retried_attempts": sum(int(row.get("attempt_index", 1)) > 1 for row in rows),
        **{column: total(column) for column in (*TIMING_COLUMNS, *ACCOUNTING_COLUMNS)},
        "successful_fit_phase_seconds": {
            column: _distribution_summary(successful_rows, column)
            for column in (
                "wall_seconds",
                "training_wall_seconds",
                "inference_wall_seconds",
                "checkpoint_io_wall_seconds",
                "useful_compute_seconds",
            )
        },
        "peak_device_memory_bytes": _peak_memory_summary(rows),
        "compute_cost_partition": {
            "successful_fit_work": _cost_summary(successful_rows),
            "failed_attempt_work": _cost_summary(failed_rows),
            "retry_attempt_work_including_successful_retries": _cost_summary(retry_rows),
        },
        "failure_reason_breakdown": _failure_reason_breakdown(failed_rows, len(rows)),
        "attempt_group_summaries": attempt_group_summaries,
        "attempt_group_summaries_sha256": canonical_sha256(attempt_group_summaries),
        "hardware_configurations": sorted(
            {
                (
                    str(row.get("device")),
                    str(row.get("accelerator_model")),
                    int(row.get("accelerator_count", 0)),
                    str(row.get("cpu_model")),
                    int(row.get("logical_cpu_count", 0)),
                    int(row.get("host_ram_bytes", 0)),
                )
                for row in rows
            }
        ),
        "source_registry_hashes": sorted(registry_hashes),
        "source_registry_manifest_hash": canonical_sha256(sorted(registry_hashes)),
        "failures": unique_failures,
    }
    payload["registry_hash"] = canonical_sha256(payload)
    return payload


def _unique_failures(failures: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    output = [dict(values) for values in {tuple(sorted(row.items())) for row in failures}]
    return sorted(output, key=lambda row: (row["reason_code"], row["detail"]))


def _validate_reconciliation_source(
    row: Mapping[str, Any],
    root: Path,
    failures: list[dict[str, str]],
) -> None:
    if row.get("timing_reconciliation_status") != "FINALIZED_SCHEDULER_RECONCILED":
        return
    source_id = str(row.get("scheduler_source_record_source_id", ""))
    expected_hash = str(row.get("scheduler_source_record_sha256", ""))
    source = Path(source_id)
    if (
        not source_id
        or source.is_absolute()
        or ".." in source.parts
        or not _is_sha256(expected_hash)
    ):
        failures.append(
            {"reason_code": "COMPUTE_RECONCILIATION_SOURCE_INVALID", "detail": source_id}
        )
        return
    resolved = (root / source).resolve()
    if (
        not resolved.is_relative_to(root.resolve())
        or not resolved.is_file()
        or file_sha256(resolved) != expected_hash
    ):
        failures.append(
            {"reason_code": "COMPUTE_RECONCILIATION_SOURCE_INVALID", "detail": source_id}
        )


def _binding_values_differ(observed: Any, expected: Any) -> bool:
    if pd.isna(expected) and observed is None:
        return False
    if isinstance(observed, (int, float)) and not isinstance(observed, bool):
        try:
            return not math.isclose(float(observed), float(expected), rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return True
    return observed != expected


def _distribution_summary(rows: Sequence[Mapping[str, Any]], column: str) -> dict[str, Any]:
    values = pd.Series([float(row[column]) for row in rows], dtype=float)
    if values.empty:
        return {"n": 0, "median": None, "q1": None, "q3": None, "iqr": None}
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    return {
        "n": int(len(values)),
        "median": float(values.median()),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }


def _peak_memory_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    measured = [
        int(row["peak_device_memory_bytes"])
        for row in rows
        if row.get("peak_device_memory_bytes") is not None
    ]
    statuses = pd.Series([str(row.get("peak_memory_status")) for row in rows]).value_counts()
    summary = _distribution_summary([{"value": value} for value in measured], "value")
    summary["maximum"] = max(measured) if measured else None
    summary["status_counts"] = {
        str(status): int(count) for status, count in statuses.sort_index().items()
    }
    summary["attempt_denominator"] = len(rows)
    return summary


def _cost_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "attempts": len(rows),
        "wall_seconds": float(sum(float(row["wall_seconds"]) for row in rows)),
        "device_hours": float(sum(float(row["device_hours"]) for row in rows)),
        "accelerator_hours": float(sum(float(row["accelerator_hours"]) for row in rows)),
    }


def _failure_reason_breakdown(
    failed_rows: Sequence[Mapping[str, Any]], all_attempts: int
) -> dict[str, Any]:
    counts = pd.Series(
        [str(row.get("failure_reason", "MISSING_REASON")) for row in failed_rows]
    ).value_counts()
    return {
        "failed_attempt_denominator": len(failed_rows),
        "all_attempt_denominator": all_attempts,
        "reasons": [
            {
                "failure_reason": str(reason),
                "count": int(count),
                "fraction_of_failed_attempts": (
                    float(count / len(failed_rows)) if failed_rows else None
                ),
                "fraction_of_all_attempts": (float(count / all_attempts) if all_attempts else None),
            }
            for reason, count in counts.sort_index().items()
        ],
    }


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)

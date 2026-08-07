"""Atomically reconcile one interrupted compute attempt from a scheduler record."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import canonical_sha256, file_sha256
from .errors import RevisionProtocolError

FINAL_SCHEDULER_STATES = {
    "CANCELLED",
    "FAILED",
    "NODE_FAILURE",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}


def reconcile_interrupted_attempt(
    registry_path: Path,
    *,
    run_id: str,
    dataset: str,
    hvg: int,
    panel: str,
    arm: str,
    condition: str,
    seed: int,
    attempt_index: int,
    scheduler_job_id: str,
    scheduler_state: str,
    started_at_utc: str,
    finished_at_utc: str,
    elapsed_allocation_seconds: float,
    device_count: int,
    exit_or_preemption_reason: str,
    source_record_path: Path,
    source_record_sha256: str,
) -> dict[str, Any]:
    """Replace exactly one STARTED row with a scheduler-grounded finalized failure."""

    registry_path = registry_path.resolve()
    root = registry_path.parent
    registry = _read_self_hashed_registry(registry_path)
    rows = _read_bound_ledger(root, registry)
    source_record_path = source_record_path.resolve()
    if not source_record_path.is_file() or file_sha256(source_record_path) != source_record_sha256:
        raise RevisionProtocolError("[COMPUTE_RECONCILIATION_SOURCE_HASH_MISMATCH]")
    source_record = _read_json_object(source_record_path, "COMPUTE_RECONCILIATION_SOURCE_INVALID")
    scheduler_state = scheduler_state.strip().upper()
    if scheduler_state not in FINAL_SCHEDULER_STATES:
        raise RevisionProtocolError("[COMPUTE_RECONCILIATION_SCHEDULER_STATE_INVALID]")
    if (
        not run_id
        or not dataset
        or hvg <= 0
        or not panel
        or not arm
        or not condition
        or attempt_index <= 0
        or not scheduler_job_id.strip()
        or not exit_or_preemption_reason.strip()
        or isinstance(device_count, bool)
        or device_count < 0
        or not math.isfinite(float(elapsed_allocation_seconds))
        or elapsed_allocation_seconds < 0
    ):
        raise RevisionProtocolError("[COMPUTE_RECONCILIATION_ARGUMENT_INVALID]")
    started = _parse_utc(started_at_utc)
    finished = _parse_utc(finished_at_utc)
    if finished < started:
        raise RevisionProtocolError("[COMPUTE_RECONCILIATION_TIME_ORDER_INVALID]")
    expected_source = {
        "scheduler_job_id": scheduler_job_id,
        "scheduler_state": scheduler_state,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "elapsed_allocation_seconds": float(elapsed_allocation_seconds),
        "device_count": int(device_count),
        "exit_or_preemption_reason": exit_or_preemption_reason,
    }
    if any(source_record.get(key) != value for key, value in expected_source.items()):
        raise RevisionProtocolError("[COMPUTE_RECONCILIATION_SOURCE_CONTENT_MISMATCH]")

    key = (dataset, int(hvg), panel, arm, condition, int(seed))
    matches = [
        index
        for index, row in enumerate(rows)
        if _fit_key(row) == key
        and int(row.get("attempt_index", -1)) == attempt_index
        and row.get("run_id") == run_id
    ]
    if len(matches) != 1:
        raise RevisionProtocolError("[COMPUTE_RECONCILIATION_ATTEMPT_NOT_UNIQUE]")
    index = matches[0]
    started_row = rows[index]
    if (
        started_row.get("execution_status") != "STARTED_UNFINALIZED"
        or started_row.get("scheduler_job_id") != scheduler_job_id
        or started_row.get("attempt_started_at_utc") != started_at_utc
        or registry.get("run_id") != run_id
        or registry.get("dataset") != dataset
        or int(registry.get("hvg", -1)) != hvg
        or registry.get("panel") != panel
    ):
        raise RevisionProtocolError("[COMPUTE_RECONCILIATION_ATTEMPT_BINDING_MISMATCH]")

    retained_source = root / f"scheduler_reconciliation_source.{source_record_sha256}.json"
    _copy_verified(source_record_path, retained_source)
    finalized = dict(started_row)
    elapsed = float(elapsed_allocation_seconds)
    finalized.update(
        {
            "execution_status": f"FAILED_RECONCILED_{scheduler_state}",
            "failure_reason": exit_or_preemption_reason,
            "wall_seconds": elapsed,
            "training_wall_seconds": 0.0,
            "inference_wall_seconds": 0.0,
            "checkpoint_io_wall_seconds": 0.0,
            "other_overhead_wall_seconds": elapsed,
            "useful_compute_seconds": 0.0,
            "fit_scope_compute_seconds": 0.0,
            "scheduler_allocation_seconds": elapsed,
            "failed_attempt_wall_seconds": elapsed,
            "failed_preempted_allocation_seconds": elapsed,
            "retry_overhead_wall_seconds": 0.0,
            "retry_allocation_overhead_seconds": 0.0,
            "serialization_registry_other_nonfit_seconds": 0.0,
            "unattributed_interrupted_allocation_seconds": elapsed,
            "fit_scope_timing_status": "NOT_AVAILABLE_HARD_INTERRUPTION",
            "attempt_finished_at_utc": finished_at_utc,
            "timing_reconciliation_status": "FINALIZED_SCHEDULER_RECONCILED",
            "scheduler_state": scheduler_state,
            "scheduler_exit_or_preemption_reason": exit_or_preemption_reason,
            "scheduler_source_record_source_id": retained_source.name,
            "scheduler_source_record_sha256": source_record_sha256,
            "accelerator_count": int(device_count),
            "accelerator_hours": elapsed * int(device_count) / 3600.0,
            "device_hours": elapsed / 3600.0,
            "artifact_hash": None,
            "checkpoint_hash": None,
            "peak_device_memory_bytes": None,
            "peak_memory_status": "NOT_MEASURED_HARD_INTERRUPTION",
        }
    )
    rows[index] = finalized
    _publish_reconciled_ledger(root, registry_path, registry, rows)
    return finalized


def _read_self_hashed_registry(path: Path) -> dict[str, Any]:
    registry = _read_json_object(path, "COMPUTE_REGISTRY_INVALID")
    payload = dict(registry)
    declared = payload.pop("registry_hash", None)
    if declared != canonical_sha256(payload):
        raise RevisionProtocolError("[COMPUTE_REGISTRY_HASH_MISMATCH]")
    return registry


def _read_bound_ledger(root: Path, registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_id = str(registry.get("attempt_ledger_source_id", ""))
    source = Path(source_id)
    if not source_id or source.is_absolute() or ".." in source.parts:
        raise RevisionProtocolError("[COMPUTE_LEDGER_SOURCE_INVALID]")
    path = (root / source).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise RevisionProtocolError("[COMPUTE_LEDGER_MISSING]")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(rows, list)
        or not all(isinstance(row, dict) for row in rows)
        or file_sha256(path) != registry.get("attempt_ledger_file_sha256")
        or canonical_sha256(rows) != registry.get("attempt_ledger_hash")
    ):
        raise RevisionProtocolError("[COMPUTE_LEDGER_HASH_MISMATCH]")
    return [dict(row) for row in rows]


def _read_json_object(path: Path, code: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RevisionProtocolError(f"[{code}] {error}") from error
    if not isinstance(payload, dict):
        raise RevisionProtocolError(f"[{code}] object required")
    return payload


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RevisionProtocolError("[COMPUTE_RECONCILIATION_TIMESTAMP_INVALID]") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RevisionProtocolError("[COMPUTE_RECONCILIATION_TIMESTAMP_INVALID]")
    return parsed


def _fit_key(row: Mapping[str, Any]) -> tuple[str, int, str, str, str, int]:
    return (
        str(row.get("dataset")),
        int(row.get("hvg", -1)),
        str(row.get("panel")),
        str(row.get("arm")),
        str(row.get("condition")),
        int(row.get("seed", -1)),
    )


def _copy_verified(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    if file_sha256(temporary) != file_sha256(source):
        temporary.unlink(missing_ok=True)
        raise RevisionProtocolError("[COMPUTE_RECONCILIATION_SOURCE_COPY_MISMATCH]")
    os.replace(temporary, destination)


def _publish_reconciled_ledger(
    root: Path,
    registry_path: Path,
    prior_registry: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    ledger_hash = canonical_sha256(list(rows))
    ledger_path = root / f"compute_attempt_ledger.{ledger_hash}.json"
    ledger_temporary = ledger_path.with_name(f".{ledger_path.name}.tmp")
    ledger_temporary.write_text(
        json.dumps(list(rows), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(ledger_temporary, ledger_path)

    def total(field: str) -> float:
        return float(sum(float(row.get(field, 0.0)) for row in rows))

    registry = dict(prior_registry)
    registry.pop("registry_hash", None)
    successes = sum(row.get("execution_status") == "SUCCESS" for row in rows)
    registry.update(
        {
            "status": "WITHHELD",
            "observed_attempts": len(rows),
            "successful_attempts": successes,
            "failed_attempts": len(rows) - successes,
            "retried_attempts": sum(int(row.get("attempt_index", 1)) > 1 for row in rows),
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
            "retry_allocation_overhead_seconds": total("retry_allocation_overhead_seconds"),
            "serialization_registry_other_nonfit_seconds": total(
                "serialization_registry_other_nonfit_seconds"
            ),
            "unattributed_interrupted_allocation_seconds": total(
                "unattributed_interrupted_allocation_seconds"
            ),
            "accelerator_hours": total("accelerator_hours"),
            "device_hours": total("device_hours"),
            "attempt_ledger_hash": ledger_hash,
            "attempt_ledger_source_id": ledger_path.name,
            "attempt_ledger_file_sha256": file_sha256(ledger_path),
            "expected_fit_key_coverage": "FAIL",
        }
    )
    registry["registry_hash"] = canonical_sha256(registry)
    temporary = registry_path.with_name(f".{registry_path.name}.tmp")
    temporary.write_text(
        json.dumps(registry, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, registry_path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--hvg", type=int, required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--attempt-index", type=int, required=True)
    parser.add_argument("--scheduler-job-id", required=True)
    parser.add_argument("--scheduler-state", choices=sorted(FINAL_SCHEDULER_STATES), required=True)
    parser.add_argument("--started-at-utc", required=True)
    parser.add_argument("--finished-at-utc", required=True)
    parser.add_argument("--elapsed-allocation-seconds", type=float, required=True)
    parser.add_argument("--device-count", type=int, required=True)
    parser.add_argument("--exit-or-preemption-reason", required=True)
    parser.add_argument("--source-record", type=Path, required=True)
    parser.add_argument("--source-record-sha256", required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_argument_parser().parse_args(arguments)
    row = reconcile_interrupted_attempt(
        parsed.registry,
        run_id=parsed.run_id,
        dataset=parsed.dataset,
        hvg=parsed.hvg,
        panel=parsed.panel,
        arm=parsed.arm,
        condition=parsed.condition,
        seed=parsed.seed,
        attempt_index=parsed.attempt_index,
        scheduler_job_id=parsed.scheduler_job_id,
        scheduler_state=parsed.scheduler_state,
        started_at_utc=parsed.started_at_utc,
        finished_at_utc=parsed.finished_at_utc,
        elapsed_allocation_seconds=parsed.elapsed_allocation_seconds,
        device_count=parsed.device_count,
        exit_or_preemption_reason=parsed.exit_or_preemption_reason,
        source_record_path=parsed.source_record,
        source_record_sha256=parsed.source_record_sha256,
    )
    print(json.dumps(row, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cbac_revision.artifacts import canonical_sha256, file_sha256
from cbac_revision.compute_release import release_measured_compute

HASH_A = "a" * 64
HASH_B = "b" * 64


def _attempt(
    index: int,
    status: str,
    *,
    artifact_hash: str | None = None,
    checkpoint_hash: str | None = None,
) -> dict:
    wall = 5.0 if status == "SUCCESS" else 2.0
    training = 3.0 if status == "SUCCESS" else 1.5
    inference = 1.0 if status == "SUCCESS" else 0.0
    checkpoint = 0.5 if status == "SUCCESS" else 0.0
    return {
        "dataset": "adamson",
        "hvg": 200,
        "panel": "primary",
        "arm": "dense",
        "condition": "TP53",
        "seed": 42,
        "attempt_index": index,
        "retry_count": index - 1,
        "execution_status": status,
        "failure_reason": "NOT_APPLICABLE" if status == "SUCCESS" else "RuntimeError",
        "wall_seconds": wall,
        "training_wall_seconds": training,
        "inference_wall_seconds": inference,
        "checkpoint_io_wall_seconds": checkpoint,
        "other_overhead_wall_seconds": wall - training - inference - checkpoint,
        "useful_compute_seconds": training + inference,
        "fit_scope_compute_seconds": training + inference,
        "scheduler_allocation_seconds": wall,
        "failed_attempt_wall_seconds": 0.0 if status == "SUCCESS" else wall,
        "failed_preempted_allocation_seconds": 0.0 if status == "SUCCESS" else wall,
        "retry_overhead_wall_seconds": (
            wall - training - inference - checkpoint if index > 1 else 0.0
        ),
        "retry_allocation_overhead_seconds": wall if index > 1 and status != "SUCCESS" else 0.0,
        "serialization_registry_other_nonfit_seconds": wall - training - inference,
        "unattributed_interrupted_allocation_seconds": 0.0,
        "fit_scope_timing_status": "MEASURED_IN_PROCESS",
        "accelerator_hours": 0.0,
        "device_hours": wall / 3600.0,
        "device": "cpu",
        "host": "test-host",
        "accelerator_model": "NOT_APPLICABLE_CPU_EXECUTION",
        "accelerator_count": 0,
        "cpu_model": "test-cpu",
        "logical_cpu_count": 4,
        "host_ram_bytes": 8_000_000_000,
        "peak_device_memory_bytes": None,
        "peak_memory_status": "NOT_APPLICABLE_CPU",
        "artifact_hash": artifact_hash,
        "checkpoint_hash": checkpoint_hash,
        "attempt_started_at_utc": "2026-08-06T12:00:00Z",
        "attempt_finished_at_utc": "2026-08-06T12:00:05Z",
        "scheduler_job_id": "LOCAL_PROCESS:1",
        "timing_reconciliation_status": "FINALIZED_MEASURED",
    }


def _metric(success: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": success["dataset"],
                "hvg": success["hvg"],
                "panel": success["panel"],
                "arm": success["arm"],
                "condition": success["condition"],
                "seed": success["seed"],
                "artifact_hash": success["artifact_hash"],
                "checkpoint_hash": success["checkpoint_hash"],
                **{
                    f"runtime_{field}": success[field]
                    for field in (
                        "wall_seconds",
                        "training_wall_seconds",
                        "inference_wall_seconds",
                        "checkpoint_io_wall_seconds",
                        "other_overhead_wall_seconds",
                        "useful_compute_seconds",
                        "device_hours",
                        "accelerator_hours",
                        "attempt_index",
                        "retry_count",
                        "device",
                        "host",
                        "accelerator_model",
                        "accelerator_count",
                        "cpu_model",
                        "logical_cpu_count",
                        "host_ram_bytes",
                        "peak_device_memory_bytes",
                        "peak_memory_status",
                    )
                },
            }
        ]
    )


def _registry(tmp_path: Path, rows: list[dict]) -> Path:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(rows), encoding="utf-8")
    payload = {
        "schema_version": "1.0",
        "registry_id": "MEASURED-COMPUTE",
        "status": "RELEASED",
        "attempt_ledger_source_id": ledger.name,
        "attempt_ledger_file_sha256": file_sha256(ledger),
        "attempt_ledger_hash": canonical_sha256(rows),
    }
    payload["registry_hash"] = canonical_sha256(payload)
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_compute_release_partitions_retry_cost_and_failure_denominators(tmp_path: Path) -> None:
    failure = _attempt(1, "FAILED_TRAINING_EXCEPTION")
    success = _attempt(2, "SUCCESS", artifact_hash=HASH_A, checkpoint_hash=HASH_B)

    result = release_measured_compute([_registry(tmp_path, [failure, success])], _metric(success))

    assert result["status"] == "RELEASED"
    assert result["successful_fit_phase_seconds"]["training_wall_seconds"]["median"] == 3.0
    assert result["compute_cost_partition"]["failed_attempt_work"]["attempts"] == 1
    assert (
        result["compute_cost_partition"]["retry_attempt_work_including_successful_retries"][
            "attempts"
        ]
        == 1
    )
    assert result["failure_reason_breakdown"]["failed_attempt_denominator"] == 1
    assert result["failure_reason_breakdown"]["all_attempt_denominator"] == 2


def test_success_must_be_final_attempt(tmp_path: Path) -> None:
    success = _attempt(1, "SUCCESS", artifact_hash=HASH_A, checkpoint_hash=HASH_B)
    later_failure = _attempt(2, "FAILED_CHECKPOINT_WRITE_EXCEPTION")

    result = release_measured_compute(
        [_registry(tmp_path, [success, later_failure])], _metric(success)
    )

    assert result["status"] == "WITHHELD"
    assert "COMPUTE_ATTEMPT_SEQUENCE_OR_SUCCESS_MISMATCH" in {
        failure["reason_code"] for failure in result["failures"]
    }


def test_ledger_runtime_must_match_artifact_runtime(tmp_path: Path) -> None:
    success = _attempt(1, "SUCCESS", artifact_hash=HASH_A, checkpoint_hash=HASH_B)
    metric = _metric(success)
    metric.loc[0, "runtime_wall_seconds"] = 100.0

    result = release_measured_compute([_registry(tmp_path, [success])], metric)

    assert result["status"] == "WITHHELD"
    assert "COMPUTE_ARTIFACT_RUNTIME_BINDING_MISMATCH" in {
        failure["reason_code"] for failure in result["failures"]
    }

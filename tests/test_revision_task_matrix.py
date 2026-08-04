from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from slurm.generate_revision_tasks import _runner_record, generate_records
from slurm.validate_revision_launch import LaunchValidationError, validate_launch_files
from turbognn_audit.hashing import sha256_json


@pytest.mark.parametrize(
    ("stage", "configuration_count", "fold_count", "reference_count", "total_count"),
    (
        ("e0", 16, 160, 3, 19),
        ("core", 144, 7200, 0, 144),
        ("sensitivity", 72, 3600, 0, 72),
        ("topology", 360, 18000, 0, 360),
    ),
)
def test_revision_matrix_conserves_frozen_counts(
    stage: str,
    configuration_count: int,
    fold_count: int,
    reference_count: int,
    total_count: int,
) -> None:
    records, references = generate_records(stage)
    assert len(records) == configuration_count
    assert sum(int(row["planned_folds"]) for row in records) == fold_count
    assert len(references) == reference_count
    assert len(records) + len(references) == total_count
    hashes = [str(row["configuration_hash"]) for row in [*records, *references]]
    assert len(hashes) == len(set(hashes))


def test_topology_matrix_uses_every_frozen_instance_seed() -> None:
    records, _ = generate_records("topology")
    by_type = {
        graph_type: {
            int(row["graph_instance_seed"]) for row in records if row["graph_type"] == graph_type
        }
        for graph_type in ("degree_preserving_rewired", "barabasi_albert")
    }
    assert by_type["degree_preserving_rewired"] == {101, 102, 103, 104, 105}
    assert by_type["barabasi_albert"] == {201, 202, 203, 204, 205}


def _launch_fixture(tmp_path: Path) -> tuple[Path, Path]:
    records = [
        _runner_record(
            stage="full",
            dataset="norman",
            hvg=200,
            graph_type=graph_type,
            training_seed=42,
            panel_size=50,
        )
        for graph_type in ("self_loop_gat", "string_ppi")
    ]
    lines = [
        " ".join(
            str(value)
            for value in (
                row["stage"],
                row["dataset"],
                row["graph_type"],
                row["hvg"],
                row["training_seed"],
                row["panel_size"],
                row["fold_start"],
                row["fold_end"],
                "-",
                row["configuration_hash"],
            )
        )
        for row in records
    ]
    task_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    manifest = {
        "configurations": records,
        "runner_task_rows_hash": sha256_json(lines),
        "runner_task_file_sha256": hashlib.sha256(task_bytes).hexdigest(),
        "runner_task_row_count": len(lines),
        "tasks_per_array_element": 4,
        "expected_array_element_count": 1,
    }
    manifest["manifest_content_hash"] = sha256_json(manifest)
    manifest_path = tmp_path / "schedule.json"
    task_path = tmp_path / "tasks.txt"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    task_path.write_bytes(task_bytes)
    return manifest_path, task_path


def test_launch_validator_binds_exact_task_bytes_order_and_count(tmp_path: Path) -> None:
    manifest_path, task_path = _launch_fixture(tmp_path)
    assert validate_launch_files(manifest_path, task_path) == 2
    lines = task_path.read_bytes().splitlines()
    task_path.write_bytes(b"\n".join(reversed(lines)) + b"\n")
    with pytest.raises(LaunchValidationError, match="byte hash mismatch"):
        validate_launch_files(manifest_path, task_path)


def test_revision_job_rejects_incomplete_array_geometry_and_empty_launches() -> None:
    script = (Path(__file__).parents[1] / "slurm" / "job_revision.sh").read_text(encoding="utf-8")
    for required in (
        "validate_revision_launch.py",
        "SLURM_ARRAY_TASK_COUNT",
        "SLURM_ARRAY_TASK_MIN",
        "SLURM_ARRAY_TASK_MAX",
        "SLURM_ARRAY_TASK_STEP",
        'if [ "${#PIDS[@]}" -eq 0 ]',
    ):
        assert required in script

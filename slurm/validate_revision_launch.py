#!/usr/bin/env python
"""Validate exact task-file bytes and schedule rows before a revision array launch."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class LaunchValidationError(ValueError):
    """Raised when a task file is not the one frozen in its schedule manifest."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _expected_runner_lines(configurations: Sequence[object]) -> list[str]:
    lines: list[str] = []
    for raw in configurations:
        if not isinstance(raw, Mapping):
            raise LaunchValidationError("Schedule contains a non-object configuration")
        if raw.get("executor") != "run_benchmark.py":
            continue
        graph_seed = raw.get("graph_instance_seed")
        fields = (
            raw.get("stage"),
            raw.get("dataset"),
            raw.get("graph_type"),
            raw.get("hvg"),
            raw.get("training_seed"),
            raw.get("panel_size"),
            raw.get("fold_start"),
            raw.get("fold_end"),
            "-" if graph_seed is None else graph_seed,
            raw.get("configuration_hash"),
        )
        if any(value is None for value in fields):
            raise LaunchValidationError("Runner configuration lacks a task-row field")
        lines.append(" ".join(str(value) for value in fields))
    return lines


def validate_launch_files(manifest_path: Path, task_path: Path) -> int:
    """Return the frozen row count after exact manifest/task conservation checks."""
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise LaunchValidationError("Schedule manifest must be a JSON object")
    declared_content_hash = manifest.get("manifest_content_hash")
    content = {key: value for key, value in manifest.items() if key != "manifest_content_hash"}
    if declared_content_hash != _sha256_json(content):
        raise LaunchValidationError("Schedule manifest content hash mismatch")
    task_bytes = task_path.read_bytes()
    if hashlib.sha256(task_bytes).hexdigest() != manifest.get("runner_task_file_sha256"):
        raise LaunchValidationError("Task-file byte hash mismatch")
    try:
        task_text = task_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise LaunchValidationError("Task file is not valid UTF-8") from error
    if not task_text.endswith("\n") or "\r" in task_text:
        raise LaunchValidationError("Task file must use exact LF-terminated UTF-8 rows")
    lines = task_text[:-1].split("\n") if task_text else []
    if any(not line for line in lines):
        raise LaunchValidationError("Task file contains a blank row")
    declared_count = manifest.get("runner_task_row_count")
    if isinstance(declared_count, bool) or not isinstance(declared_count, int):
        raise LaunchValidationError("Schedule lacks an integer runner task-row count")
    if len(lines) != declared_count or _sha256_json(lines) != manifest.get("runner_task_rows_hash"):
        raise LaunchValidationError("Task row count or ordered-row hash mismatch")
    configurations = manifest.get("configurations")
    if not isinstance(configurations, list) or lines != _expected_runner_lines(configurations):
        raise LaunchValidationError("Task rows do not exactly encode the frozen configurations")
    tasks_per_element = manifest.get("tasks_per_array_element")
    expected_elements = manifest.get("expected_array_element_count")
    if tasks_per_element != 4 or expected_elements != (declared_count + 3) // 4:
        raise LaunchValidationError("Schedule array geometry is invalid")
    if declared_count < 1 or expected_elements < 1:
        raise LaunchValidationError("Schedule has no executable runner tasks")
    return declared_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    args = parser.parse_args()
    try:
        count = validate_launch_files(args.manifest, args.task_file)
    except (OSError, json.JSONDecodeError, LaunchValidationError) as error:
        parser.error(str(error))
    print(count)


if __name__ == "__main__":
    main()

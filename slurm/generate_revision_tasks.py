#!/usr/bin/env python
"""Generate frozen E0/core/sensitivity/topology configuration manifests and task rows."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from itertools import product
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turbognn_audit.hashing import sha256_file, sha256_json
from turbognn_audit.io import write_json_atomic

DATASETS = ("norman", "adamson", "replogle_k562", "replogle_rpe1")
SCALES = (200, 500, 1000)
TRAINING_SEEDS = (42, 43, 44)
CORE_ARMS = ("self_loop_gat", "string_ppi", "gene_ontology", "string_go_union")
SENSITIVITY_ARMS = ("coexpression", "transformer")
E0_ARMS = (
    "self_loop_gat",
    "string_ppi",
    "gene_ontology",
    "string_go_union",
    "degree_preserving_rewired",
    "barabasi_albert",
    "transformer",
)


def _runner_record(
    *,
    stage: str,
    dataset: str,
    hvg: int,
    graph_type: str,
    training_seed: int,
    panel_size: int,
    graph_instance_seed: int | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "record_type": "strict_lopo_runner",
        "executor": "run_benchmark.py",
        "stage": stage,
        "dataset": dataset,
        "hvg": hvg,
        "graph_type": graph_type,
        "training_seed": training_seed,
        "panel_size": panel_size,
        "fold_start": 0,
        "fold_end": panel_size,
        "planned_folds": panel_size,
        "graph_instance_seed": graph_instance_seed,
    }
    value["configuration_hash"] = sha256_json(value)
    return value


def generate_records(stage: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return strict-LOPO records and separate official-reference records."""
    records: list[dict[str, object]] = []
    references: list[dict[str, object]] = []
    if stage == "e0":
        for dataset, hvg in (("adamson", 200), ("replogle_k562", 1000)):
            for graph_type in E0_ARMS:
                graph_seed = {
                    "degree_preserving_rewired": 101,
                    "barabasi_albert": 201,
                }.get(graph_type)
                records.append(
                    _runner_record(
                        stage="e0",
                        dataset=dataset,
                        hvg=hvg,
                        graph_type=graph_type,
                        training_seed=42,
                        panel_size=10,
                        graph_instance_seed=graph_seed,
                    )
                )
            external = {
                "record_type": "strict_lopo_external",
                "executor": "external_adapter_gears",
                "stage": "e0",
                "dataset": dataset,
                "hvg": hvg,
                "model": "GEARS",
                "training_seed": 42,
                "panel_size": 10,
                "planned_folds": 10,
            }
            external["configuration_hash"] = sha256_json(external)
            records.append(external)
        for seed in TRAINING_SEEDS:
            reference = {
                "record_type": "official_reference",
                "executor": "external_reference_gears",
                "stage": "e0",
                "model": "GEARS",
                "seed": seed,
                "planned_reference_records": 1,
            }
            reference["configuration_hash"] = sha256_json(reference)
            references.append(reference)
        assert len(records) == 16 and sum(int(row["planned_folds"]) for row in records) == 160
        assert len(references) == 3 and len(records) + len(references) == 19
    elif stage == "core":
        for dataset, hvg, graph_type, seed in product(DATASETS, SCALES, CORE_ARMS, TRAINING_SEEDS):
            records.append(
                _runner_record(
                    stage="full",
                    dataset=dataset,
                    hvg=hvg,
                    graph_type=graph_type,
                    training_seed=seed,
                    panel_size=50,
                )
            )
        assert len(records) == 144 and sum(int(row["planned_folds"]) for row in records) == 7200
    elif stage == "sensitivity":
        for dataset, hvg, graph_type, seed in product(
            DATASETS, SCALES, SENSITIVITY_ARMS, TRAINING_SEEDS
        ):
            records.append(
                _runner_record(
                    stage="full",
                    dataset=dataset,
                    hvg=hvg,
                    graph_type=graph_type,
                    training_seed=seed,
                    panel_size=50,
                )
            )
        assert len(records) == 72 and sum(int(row["planned_folds"]) for row in records) == 3600
    elif stage == "topology":
        topology = (
            ("degree_preserving_rewired", (101, 102, 103, 104, 105)),
            ("barabasi_albert", (201, 202, 203, 204, 205)),
        )
        for dataset, hvg, (graph_type, graph_seeds), seed in product(
            DATASETS, SCALES, topology, TRAINING_SEEDS
        ):
            for graph_seed in graph_seeds:
                records.append(
                    _runner_record(
                        stage="full",
                        dataset=dataset,
                        hvg=hvg,
                        graph_type=graph_type,
                        training_seed=seed,
                        panel_size=50,
                        graph_instance_seed=graph_seed,
                    )
                )
        assert len(records) == 360 and sum(int(row["planned_folds"]) for row in records) == 18000
    else:
        raise ValueError(f"Unsupported stage: {stage}")
    hashes = [str(row["configuration_hash"]) for row in [*records, *references]]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("Generated duplicate configuration hashes")
    return records, references


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("e0", "core", "sensitivity", "topology"), required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-tasks", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    for name in (
        "dataset-passport",
        "control-map",
        "target-map",
        "preprocessing-config",
        "model-config",
        "environment-lock",
        "graph-source-config",
        "graph-ensemble-config",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.code_commit) is None:
        parser.error("--code-commit must be an exact lowercase 40-character Git SHA")
    input_paths = {
        name: getattr(args, name.replace("-", "_"))
        for name in (
            "dataset-passport",
            "control-map",
            "target-map",
            "preprocessing-config",
            "model-config",
            "environment-lock",
            "graph-source-config",
            "graph-ensemble-config",
        )
    }
    inputs: dict[str, dict[str, str]] = {}
    for name, path in input_paths.items():
        if not path.is_file():
            parser.error(f"Required frozen input does not exist: {path}")
        inputs[name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    records, references = generate_records(args.stage)
    runner_records = [row for row in records if row["executor"] == "run_benchmark.py"]
    lines = [
        " ".join(
            [
                str(row["stage"]),
                str(row["dataset"]),
                str(row["graph_type"]),
                str(row["hvg"]),
                str(row["training_seed"]),
                str(row["panel_size"]),
                str(row["fold_start"]),
                str(row["fold_end"]),
                str(row["graph_instance_seed"] or "-"),
                str(row["configuration_hash"]),
            ]
        )
        for row in runner_records
    ]
    task_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    tasks_per_array_element = 4
    manifest = {
        "schema_version": "1.0.0",
        "matrix_stage": args.stage,
        "code_commit": args.code_commit,
        "frozen_inputs": inputs,
        "strict_lopo_configuration_count": len(records),
        "strict_lopo_fold_count": sum(int(row["planned_folds"]) for row in records),
        "reference_record_count": len(references),
        "total_scheduled_configuration_count": len(records) + len(references),
        "configurations": records,
        "reference_records": references,
        "runner_task_rows_hash": sha256_json(lines),
        "runner_task_file_sha256": hashlib.sha256(task_bytes).hexdigest(),
        "runner_task_row_count": len(lines),
        "tasks_per_array_element": tasks_per_array_element,
        "expected_array_element_count": (len(lines) + tasks_per_array_element - 1)
        // tasks_per_array_element,
    }
    manifest["manifest_content_hash"] = sha256_json(manifest)
    write_json_atomic(args.output_manifest, manifest)
    args.output_tasks.parent.mkdir(parents=True, exist_ok=True)
    args.output_tasks.write_bytes(task_bytes)
    print(
        f"Wrote {len(records)} strict configurations, "
        f"{manifest['strict_lopo_fold_count']} folds, and {len(references)} references"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Strictly merge seed/fold chunks without aggregate overwrite or duplicate folds.

Accepted split tags are ``__f0-10`` and ``__s42f0-10``. Existing aggregate
files are outputs, not implicit inputs. They can only be included with the
explicit ``--include-aggregates`` flag, and any overlap then fails closed.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from turbognn_audit.hashing import canonical_json, sha256_json
from turbognn_audit.io import write_json_atomic
from turbognn_audit.lineage import LineageValidationError, validate_chunk_lineage
from turbognn_audit.manifest import RUN_IDENTITY_FIELDS

SPLIT_TAG = re.compile(r"^(?P<base>.+)__(?:s(?P<seed>\d+))?f(?P<start>\d+)-(?P<end>\d+)$")


class MergeValidationError(ValueError):
    """Raised when source chunks cannot be merged without ambiguity."""


@dataclass(frozen=True)
class SplitFile:
    """Parsed identity encoded by one split filename."""

    path: Path
    base: str
    seed: int | None
    fold_start: int
    fold_end: int


def parse_split_file(path: Path) -> SplitFile | None:
    """Parse old ``__f`` and seed-coded ``__sNf`` split filenames."""
    match = SPLIT_TAG.fullmatch(path.stem)
    if match is None:
        return None
    start = int(match.group("start"))
    end = int(match.group("end"))
    if start < 0 or end <= start:
        raise MergeValidationError(f"Invalid fold interval in {path.name}")
    seed = int(match.group("seed")) if match.group("seed") is not None else None
    return SplitFile(
        path=path,
        base=match.group("base"),
        seed=seed,
        fold_start=start,
        fold_end=end,
    )


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise MergeValidationError(f"Top-level JSON must be an object: {path}")
    return value


def _metrics(folds: list[Mapping[str, Any]]) -> dict[str, Any]:
    required = (
        "pearson_r",
        "spearman_rho",
        "gene_top20_absolute_delta_jaccard",
        "mse",
    )
    for fold in folds:
        nested = fold.get("metrics")
        available = nested if isinstance(nested, Mapping) else {}
        missing = [name for name in required if name not in available]
        if missing:
            raise MergeValidationError(f"Fold {fold.get('condition')!r} lacks metrics {missing}")
        states = fold.get("metric_states")
        if not isinstance(states, Mapping):
            raise MergeValidationError(
                f"Fold {fold.get('condition')!r} lacks explicit metric_states"
            )
    arrays: dict[str, np.ndarray] = {}
    for name in required:
        values = [
            float(fold["metrics"][name])
            for fold in folds
            if fold["metric_states"].get(name) == "valid"
        ]
        arrays[name] = np.asarray(values, dtype=float)

    def summary(name: str, operation: str) -> float | None:
        values = arrays[name]
        if not len(values):
            return None
        return float(np.mean(values) if operation == "mean" else np.std(values))

    return {
        "pearson_mean": summary("pearson_r", "mean"),
        "pearson_std": summary("pearson_r", "std"),
        "pearson_valid_n": len(arrays["pearson_r"]),
        "spearman_mean": summary("spearman_rho", "mean"),
        "spearman_std": summary("spearman_rho", "std"),
        "spearman_valid_n": len(arrays["spearman_rho"]),
        "gene_top20_absolute_delta_jaccard_mean": summary(
            "gene_top20_absolute_delta_jaccard", "mean"
        ),
        "gene_top20_absolute_delta_jaccard_std": summary(
            "gene_top20_absolute_delta_jaccard", "std"
        ),
        "gene_top20_absolute_delta_jaccard_valid_n": len(
            arrays["gene_top20_absolute_delta_jaccard"]
        ),
        "mse_mean": summary("mse", "mean"),
        "mse_std": summary("mse", "std"),
        "mse_valid_n": len(arrays["mse"]),
    }


def _base_identity(data: Mapping[str, Any], path: Path) -> str:
    try:
        dataset = str(data["dataset"])
        graph_type = str(data["graph_type"])
    except KeyError as error:
        raise MergeValidationError(f"Missing dataset/graph_type in {path}") from error
    graph_instance = data.get("graph_instance")
    if not isinstance(graph_instance, str) or not graph_instance:
        raise MergeValidationError(f"Missing graph_instance in {path}")
    standard_instances = {f"{graph_type}__instance_0", "not_applicable"}
    graph_label = graph_type if graph_instance in standard_instances else graph_instance
    return f"{dataset}__{graph_label}"


def _fold_config(fold: Mapping[str, Any], path: Path) -> dict[str, Any]:
    condition = fold.get("condition")
    metadata = fold.get("metadata")
    if not isinstance(metadata, Mapping):
        raise MergeValidationError(f"Fold {condition!r} lacks revision metadata in {path}")
    dynamic_identity_fields = {
        "condition",
        "split_hash",
        "seed",
        "fold_seed",
        "configuration_universe_hash",
        "schedule_configuration_hash",
    }
    graph_identity_fields = {"graph_type", "graph_instance", "graph_hash"}
    required = tuple(
        name
        for name in RUN_IDENTITY_FIELDS
        if name not in dynamic_identity_fields | graph_identity_fields
    )
    values: dict[str, Any] = {}
    for name in required:
        value = metadata.get(name)
        if not isinstance(value, str) or not value:
            raise MergeValidationError(f"Fold {condition!r} lacks metadata {name!r} in {path}")
        values[name] = value
    planned = metadata.get("evaluated_conditions")
    if not isinstance(planned, list) or not planned or len(set(planned)) != len(planned):
        raise MergeValidationError(f"Fold {condition!r} has an invalid condition panel in {path}")
    graph = metadata.get("graph")
    if not isinstance(graph, Mapping):
        raise MergeValidationError(f"Fold {condition!r} lacks graph provenance in {path}")
    for name in ("graph_type", "graph_instance", "edge_hash"):
        value = graph.get(name)
        if not isinstance(value, str) or not value:
            raise MergeValidationError(
                f"Fold {condition!r} lacks graph provenance {name!r} in {path}"
            )
        values[name] = value
    values["evaluated_conditions"] = list(planned)
    return values


def _validate_fold_position(
    fold: Mapping[str, Any],
    split: SplitFile,
    seed: int,
    seen_indices: set[int],
) -> None:
    condition = fold.get("condition")
    fold_index = fold.get("fold_index")
    run_key = fold.get("run_key")
    metadata = fold.get("metadata")
    if not isinstance(fold_index, int):
        raise MergeValidationError(f"Fold {condition!r} lacks integer fold_index in {split.path}")
    if not split.fold_start <= fold_index < split.fold_end:
        raise MergeValidationError(
            f"Fold index {fold_index} falls outside filename interval "
            f"[{split.fold_start}, {split.fold_end}) in {split.path.name}"
        )
    if fold_index in seen_indices:
        raise MergeValidationError(
            f"Duplicate fold_index {fold_index} within {split.path.name} seed {seed}"
        )
    seen_indices.add(fold_index)
    if not isinstance(run_key, str) or re.fullmatch(r"[0-9a-f]{64}", run_key) is None:
        raise MergeValidationError(f"Fold {condition!r} lacks valid run_key in {split.path}")
    if not isinstance(metadata, Mapping) or metadata.get("seed") != seed:
        raise MergeValidationError(
            f"Fold {condition!r} metadata seed disagrees with seed_{seed} in {split.path}"
        )
    planned = metadata.get("evaluated_conditions")
    if (
        not isinstance(planned, list)
        or fold_index >= len(planned)
        or planned[fold_index] != condition
    ):
        raise MergeValidationError(
            f"Fold {condition!r} disagrees with frozen panel index {fold_index} in {split.path}"
        )


def _collect_fold(
    collected: dict[str, dict[str, dict[str, tuple[Mapping[str, Any], Path]]]],
    base: str,
    seed_key: str,
    fold: Mapping[str, Any],
    source: Path,
) -> None:
    condition = fold.get("condition")
    if not isinstance(condition, str) or not condition:
        raise MergeValidationError(f"Fold without a non-empty condition in {source}")
    prior = collected[base][seed_key].get(condition)
    if prior is not None:
        raise MergeValidationError(
            f"Duplicate fold key ({base}, {seed_key}, {condition!r}) in "
            f"{prior[1].name} and {source.name}"
        )
    collected[base][seed_key][condition] = (fold, source)


def merge_sources(
    split_files: Iterable[SplitFile],
    aggregate_files: Iterable[Path] = (),
) -> dict[str, dict[str, Any]]:
    """Merge sources, rejecting every duplicate ``(base, seed, condition)`` key."""
    collected: dict[str, dict[str, dict[str, tuple[Mapping[str, Any], Path]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    source_files: dict[str, set[str]] = defaultdict(set)
    source_lineages: dict[str, list[dict[str, Any]]] = defaultdict(list)
    base_configs: dict[str, dict[str, Any]] = {}
    run_key_sources: dict[str, Path] = {}

    for split in sorted(split_files, key=lambda value: value.path.name):
        data = _load_json(split.path)
        base = _base_identity(data, split.path)
        if base != split.base:
            raise MergeValidationError(
                f"Filename identity {split.base!r} disagrees with payload {base!r}: {split.path}"
            )
        try:
            lineage = validate_chunk_lineage(split.path, data)
        except (LineageValidationError, ValueError) as error:
            raise MergeValidationError(
                f"Invalid terminal lineage for {split.path}: {error}"
            ) from error
        source_lineages[base].append(lineage.as_dict())
        seeds = data.get("seeds")
        if not isinstance(seeds, dict) or not seeds:
            raise MergeValidationError(f"No seed records in {split.path}")
        if split.seed is not None and set(seeds) != {f"seed_{split.seed}"}:
            raise MergeValidationError(
                f"Seed-coded filename {split.path.name} disagrees with payload "
                f"seeds {sorted(seeds)}"
            )
        for seed_key, seed_data in seeds.items():
            try:
                seed = int(str(seed_key).removeprefix("seed_"))
            except ValueError as error:
                raise MergeValidationError(
                    f"Invalid seed key {seed_key!r} in {split.path}"
                ) from error
            folds = seed_data.get("folds") if isinstance(seed_data, dict) else None
            if not isinstance(folds, list):
                raise MergeValidationError(f"Missing folds array for {seed_key} in {split.path}")
            if len(folds) > split.fold_end - split.fold_start:
                raise MergeValidationError(
                    f"{split.path.name} contains {len(folds)} folds for interval "
                    f"[{split.fold_start}, {split.fold_end})"
                )
            seen_indices: set[int] = set()
            for fold in folds:
                if not isinstance(fold, dict):
                    raise MergeValidationError(f"Non-object fold in {split.path}")
                _validate_fold_position(fold, split, seed, seen_indices)
                config = _fold_config(fold, split.path)
                if config["dataset"] != data["dataset"]:
                    raise MergeValidationError(f"Dataset metadata mismatch in {split.path}")
                if config["graph_type"] != data["graph_type"]:
                    raise MergeValidationError(f"Graph-type metadata mismatch in {split.path}")
                if config["graph_instance"] != data["graph_instance"]:
                    raise MergeValidationError(f"Graph-instance metadata mismatch in {split.path}")
                previous_config = base_configs.setdefault(base, config)
                if canonical_json(previous_config) != canonical_json(config):
                    raise MergeValidationError(
                        f"Immutable configuration differs across chunks for {base}"
                    )
                run_key = str(fold["run_key"])
                previous_source = run_key_sources.get(run_key)
                if previous_source is not None:
                    raise MergeValidationError(
                        f"Duplicate run_key {run_key} in {previous_source.name} and "
                        f"{split.path.name}"
                    )
                run_key_sources[run_key] = split.path
                _collect_fold(collected, base, str(seed_key), fold, split.path)
        source_files[base].add(split.path.name)

    for path in sorted(aggregate_files):
        data = _load_json(path)
        base = _base_identity(data, path)
        seeds = data.get("seeds")
        if not isinstance(seeds, dict):
            raise MergeValidationError(f"Missing seeds object in aggregate {path}")
        for seed_key, seed_data in seeds.items():
            try:
                seed = int(str(seed_key).removeprefix("seed_"))
            except ValueError as error:
                raise MergeValidationError(f"Invalid seed key {seed_key!r} in {path}") from error
            folds = seed_data.get("folds") if isinstance(seed_data, dict) else None
            if not isinstance(folds, list):
                raise MergeValidationError(f"Missing folds array for {seed_key} in {path}")
            for fold in folds:
                if not isinstance(fold, dict):
                    raise MergeValidationError(f"Non-object fold in {path}")
                config = _fold_config(fold, path)
                if config["dataset"] != data["dataset"]:
                    raise MergeValidationError(f"Dataset metadata mismatch in {path}")
                if config["graph_type"] != data["graph_type"]:
                    raise MergeValidationError(f"Graph-type metadata mismatch in {path}")
                if config["graph_instance"] != data["graph_instance"]:
                    raise MergeValidationError(f"Graph-instance metadata mismatch in {path}")
                previous_config = base_configs.setdefault(base, config)
                if canonical_json(previous_config) != canonical_json(config):
                    raise MergeValidationError(
                        f"Immutable configuration differs across sources for {base}"
                    )
                run_key = fold.get("run_key")
                if not isinstance(run_key, str) or re.fullmatch(r"[0-9a-f]{64}", run_key) is None:
                    raise MergeValidationError(f"Fold lacks valid run_key in {path}")
                if fold.get("metadata", {}).get("seed") != seed:
                    raise MergeValidationError(f"Fold metadata seed mismatch in {path}")
                previous_source = run_key_sources.get(run_key)
                if previous_source is not None:
                    raise MergeValidationError(
                        f"Duplicate run_key {run_key} in {previous_source.name} and {path.name}"
                    )
                run_key_sources[run_key] = path
                _collect_fold(collected, base, str(seed_key), fold, path)
        source_files[base].add(path.name)

    merged: dict[str, dict[str, Any]] = {}
    for base, seed_map in sorted(collected.items()):
        if not seed_map:
            continue
        final_seeds: dict[str, dict[str, Any]] = {}
        all_folds: list[Mapping[str, Any]] = []
        for seed_key, condition_map in sorted(seed_map.items()):
            folds = [condition_map[condition][0] for condition in sorted(condition_map)]
            if not folds:
                continue
            final_seeds[seed_key] = {**_metrics(folds), "n_folds": len(folds), "folds": folds}
            all_folds.extend(folds)
        config = base_configs[base]
        merged[base] = {
            "schema_version": "1.0.0",
            "dataset": config["dataset"],
            "graph_type": config["graph_type"],
            "graph_instance": config["graph_instance"],
            "immutable_config_hash": sha256_json(config),
            "overall": {
                **_metrics(all_folds),
                "n_total_folds": len(all_folds),
                "n_seeds": len(final_seeds),
            },
            "merge_provenance": {
                "duplicate_policy": "reject",
                "source_files": sorted(source_files[base]),
                "source_manifests": sorted(
                    source_lineages[base], key=lambda value: str(value["result_file"])
                ),
            },
            "seeds": final_seeds,
        }
    return merged


def merge_directory(results_dir: Path, include_aggregates: bool = False) -> list[Path]:
    """Strictly merge split chunks in one directory and return output paths."""
    split_files = [
        parsed
        for path in results_dir.glob("*.json")
        if (parsed := parse_split_file(path)) is not None
    ]
    aggregate_files: list[Path] = []
    if include_aggregates:
        aggregate_files = [
            path
            for path in results_dir.glob("*.json")
            if parse_split_file(path) is None
            and not path.name.startswith("results_")
            and not path.name.endswith(".manifest.json")
        ]
    merged = merge_sources(split_files, aggregate_files)
    outputs: list[Path] = []
    for base, payload in merged.items():
        output = results_dir / f"{base}.json"
        write_json_atomic(output, payload)
        outputs.append(output)
        print(
            f"  {base}: {payload['overall']['n_seeds']} seeds, "
            f"{payload['overall']['n_total_folds']} unique folds"
        )
    return outputs


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--include-aggregates",
        action="store_true",
        help="Explicitly include existing aggregates; overlap is rejected",
    )
    args = parser.parse_args()
    for results_dir in args.results_dirs:
        if not results_dir.is_dir():
            raise SystemExit(f"Results directory does not exist: {results_dir}")
        print(f"Merging {results_dir}:")
        merge_directory(results_dir, include_aggregates=args.include_aggregates)


if __name__ == "__main__":
    main()

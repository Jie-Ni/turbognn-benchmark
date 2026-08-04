"""Exact joins between terminal run manifests and lossless chunk results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .hashing import sha256_file, sha256_json
from .manifest import RUN_IDENTITY_FIELDS, RunManifest


class LineageValidationError(ValueError):
    """Raised when a result cannot be conserved against its terminal manifest."""


@dataclass(frozen=True)
class ChunkLineage:
    """Verified terminal counts and source identities for one result chunk."""

    result_file: str
    result_hash: str
    manifest_file: str
    manifest_file_sha256: str
    manifest_record_hash: str
    planned_total: int
    status_counts: Mapping[str, int]
    succeeded_run_keys: tuple[str, ...]
    succeeded_run_key_set_sha256: str

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible lineage metadata."""
        return asdict(self)


def manifest_path_for_result(result_path: Path) -> Path:
    """Return the runner's required sibling manifest path."""
    return result_path.with_name(result_path.stem + ".manifest.json")


def fold_run_identity(fold: Mapping[str, Any]) -> dict[str, object]:
    """Reconstruct the complete immutable RunRecord identity from one fold."""
    condition = fold.get("condition")
    metadata = fold.get("metadata")
    if not isinstance(condition, str) or not condition or not isinstance(metadata, Mapping):
        raise LineageValidationError("Fold lacks condition or immutable metadata")
    graph = metadata.get("graph")
    if not isinstance(graph, Mapping):
        raise LineageValidationError(f"Fold {condition!r} lacks graph provenance")
    graph_fields = {
        "graph_type": graph.get("graph_type"),
        "graph_instance": graph.get("graph_instance"),
        "graph_hash": graph.get("edge_hash"),
    }
    fields = {
        name: (
            condition
            if name == "condition"
            else graph_fields[name] if name in graph_fields else metadata.get(name)
        )
        for name in RUN_IDENTITY_FIELDS
    }
    text_names = tuple(name for name in fields if name not in {"seed", "fold_seed"})
    if any(not isinstance(fields[name], str) or not str(fields[name]) for name in text_names):
        raise LineageValidationError(f"Fold {condition!r} has an incomplete run identity")
    if not isinstance(fields["seed"], int) or not isinstance(fields["fold_seed"], int):
        raise LineageValidationError(f"Fold {condition!r} has invalid seed identities")
    return fields


def _folds(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    seeds = result.get("seeds")
    if not isinstance(seeds, Mapping):
        raise LineageValidationError("Result chunk lacks a seeds object")
    folds: list[Mapping[str, Any]] = []
    for seed_data in seeds.values():
        if not isinstance(seed_data, Mapping) or not isinstance(seed_data.get("folds"), list):
            raise LineageValidationError("Result chunk has an invalid seed/folds record")
        for fold in seed_data["folds"]:
            if not isinstance(fold, Mapping):
                raise LineageValidationError("Result chunk contains a non-object fold")
            folds.append(fold)
    return folds


def validate_chunk_lineage(result_path: Path, result: Mapping[str, Any]) -> ChunkLineage:
    """Require exact manifest/result identity, terminal conservation, and byte hash agreement."""
    manifest_path = manifest_path_for_result(result_path)
    if not manifest_path.is_file():
        raise LineageValidationError(f"Missing terminal manifest for {result_path.name}")
    manifest = RunManifest.read(manifest_path)
    manifest.validate(require_terminal=True)
    observed_result_hash = sha256_file(result_path)
    records_by_key = {record.run_key: record for record in manifest.records}
    succeeded = {
        record.run_key: record for record in manifest.records if record.status == "succeeded"
    }
    observed: dict[str, Mapping[str, Any]] = {}
    for fold in _folds(result):
        identity = fold_run_identity(fold)
        computed_key = sha256_json(identity)
        declared_key = fold.get("run_key")
        if declared_key != computed_key:
            raise LineageValidationError(
                f"Fold {fold.get('condition')!r} run_key does not hash its immutable metadata"
            )
        if computed_key in observed:
            raise LineageValidationError(f"Duplicate result run_key: {computed_key}")
        record = records_by_key.get(computed_key)
        if record is None or record.identity != identity:
            raise LineageValidationError(
                f"Fold {fold.get('condition')!r} identity is absent from its manifest"
            )
        if record.status != "succeeded":
            raise LineageValidationError(
                f"Result exists for manifest status {record.status!r}: {computed_key}"
            )
        if record.result_path != result_path.name or record.result_hash != observed_result_hash:
            raise LineageValidationError(
                f"Manifest result path/hash mismatch for run_key {computed_key}"
            )
        observed[computed_key] = fold
    if set(observed) != set(succeeded):
        missing = sorted(set(succeeded) - set(observed))
        extra = sorted(set(observed) - set(succeeded))
        raise LineageValidationError(
            f"Manifest/result conservation failed: missing={missing[:5]}, extra={extra[:5]}"
        )
    return ChunkLineage(
        result_file=result_path.name,
        result_hash=observed_result_hash,
        manifest_file=manifest_path.name,
        manifest_file_sha256=sha256_file(manifest_path),
        manifest_record_hash=manifest.manifest_hash,
        planned_total=len(manifest.records),
        status_counts=manifest.status_counts(),
        succeeded_run_keys=tuple(sorted(succeeded)),
        succeeded_run_key_set_sha256=sha256_json(sorted(succeeded)),
    )

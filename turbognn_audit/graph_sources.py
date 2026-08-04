"""Frozen curated graph-source configuration and deterministic edge-list loading."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np

from .hashing import sha256_file, sha256_json

PLACEHOLDER = re.compile(r"TODO|REPLACE|PLACEHOLDER|EXAMPLE", re.IGNORECASE)


@dataclass(frozen=True)
class FrozenGraphSource:
    """One immutable, projected gene-symbol edge list and its source policy."""

    graph_type: str
    edge_list: Path
    edge_list_sha256: str
    release: str
    source_url: str
    construction_policy: str

    def validate(self) -> None:
        """Reject placeholders, missing files, and byte-level checksum mismatches."""
        text_fields = (self.release, self.source_url, self.construction_policy)
        if any(not value or PLACEHOLDER.search(value) for value in text_fields):
            raise ValueError(f"Graph source {self.graph_type!r} contains placeholder provenance")
        if re.fullmatch(r"[0-9a-f]{64}", self.edge_list_sha256) is None:
            raise ValueError(f"Graph source {self.graph_type!r} lacks a valid SHA-256")
        if not self.edge_list.is_file():
            raise ValueError(f"Graph edge list does not exist: {self.edge_list}")
        observed = sha256_file(self.edge_list)
        if observed != self.edge_list_sha256:
            raise ValueError(
                f"Graph edge-list checksum mismatch for {self.graph_type}: "
                f"{observed} != {self.edge_list_sha256}"
            )


def load_frozen_graph_sources(
    values: Mapping[str, Any],
    *,
    config_path: Path,
    required: Sequence[str],
) -> dict[str, FrozenGraphSource]:
    """Load required graph sources relative to a verified JSON configuration."""
    if values.get("schema_version") != "1.0.0":
        raise ValueError("Unsupported graph-source schema_version")
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "graph_sources.schema.json"
    with schema_path.open(encoding="utf-8") as handle:
        schema = json.load(handle)
    try:
        jsonschema.validate(values, schema)
    except jsonschema.ValidationError as error:
        raise ValueError(f"Graph-source config fails its JSON schema: {error.message}") from error
    sources = values.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("Graph-source config lacks a sources object")
    missing = sorted(set(required) - set(sources))
    if missing:
        raise ValueError(f"Graph-source config lacks required sources: {missing}")
    loaded: dict[str, FrozenGraphSource] = {}
    for graph_type in required:
        raw = sources[graph_type]
        if not isinstance(raw, Mapping):
            raise ValueError(f"Invalid graph-source record for {graph_type!r}")
        edge_path = Path(str(raw.get("edge_list", "")))
        if edge_path.is_absolute() or ".." in edge_path.parts:
            raise ValueError("Graph edge-list paths must be safe paths relative to the config")
        edge_path = (config_path.parent / edge_path).resolve()
        try:
            edge_path.relative_to(config_path.parent.resolve())
        except ValueError as error:
            raise ValueError(
                "Graph edge-list path resolves outside the config directory"
            ) from error
        source = FrozenGraphSource(
            graph_type=graph_type,
            edge_list=edge_path,
            edge_list_sha256=str(raw.get("edge_list_sha256", "")),
            release=str(raw.get("release", "")),
            source_url=str(raw.get("source_url", "")),
            construction_policy=str(raw.get("construction_policy", "")),
        )
        source.validate()
        loaded[graph_type] = source
    return loaded


def build_frozen_edge_graph(
    source: FrozenGraphSource,
    gene_order: Sequence[str],
) -> np.ndarray:
    """Filter a complete frozen gene-symbol edge list to one ordered gene panel."""
    source.validate()
    genes = [str(gene) for gene in gene_order]
    if len(set(genes)) != len(genes):
        raise ValueError("Gene panel contains duplicate symbols")
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    with source.edge_list.open(encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split("\t")
        if header != ["gene_a", "gene_b"]:
            raise ValueError(
                f"Frozen edge list must have exact TSV header gene_a<TAB>gene_b: {source.edge_list}"
            )
        undirected: set[tuple[int, int]] = set()
        for line_number, line in enumerate(handle, start=2):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 2 or not all(fields):
                raise ValueError(f"Malformed edge-list row {line_number} in {source.edge_list}")
            left, right = fields
            if left == right or left not in gene_to_index or right not in gene_to_index:
                continue
            source_index, target_index = sorted((gene_to_index[left], gene_to_index[right]))
            undirected.add((source_index, target_index))
    if not undirected:
        raise ValueError(
            f"Frozen {source.graph_type} edge list has no non-self edges on this gene panel"
        )
    directed = {(node, node) for node in range(len(genes))}
    for source_index, target_index in undirected:
        directed.add((source_index, target_index))
        directed.add((target_index, source_index))
    return np.asarray(sorted(directed), dtype=np.int64).T


def graph_source_hashes(sources: Mapping[str, FrozenGraphSource]) -> str:
    """Hash releases, policies, and edge-list bytes for run-level provenance."""
    return sha256_json(
        {
            graph_type: {
                "edge_list_sha256": source.edge_list_sha256,
                "release": source.release,
                "source_url": source.source_url,
                "construction_policy": source.construction_policy,
            }
            for graph_type, source in sorted(sources.items())
        }
    )

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from turbognn_audit.graph_sources import (
    FrozenGraphSource,
    build_frozen_edge_graph,
    load_frozen_graph_sources,
)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_edge_list_filters_complete_source_and_adds_symmetric_loops(tmp_path) -> None:
    edges = tmp_path / "edges.tsv"
    edges.write_text("gene_a\tgene_b\nA\tB\nB\tC\nA\tOUTSIDE\n", encoding="utf-8")
    source = FrozenGraphSource(
        graph_type="string_ppi",
        edge_list=edges,
        edge_list_sha256=_sha256(edges),
        release="STRING frozen release",
        source_url="https://string-db.org/releases/test",
        construction_policy="Mapped undirected score-filtered edge list",
    )
    edge_index = build_frozen_edge_graph(source, ["A", "B", "C"])
    observed = {tuple(edge) for edge in edge_index.T.tolist()}
    assert observed == {
        (0, 0),
        (1, 1),
        (2, 2),
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
    }
    assert edge_index.dtype == np.int64


def test_graph_source_loader_rejects_checksum_mismatch_and_placeholders(tmp_path) -> None:
    edges = tmp_path / "edges.tsv"
    edges.write_text("gene_a\tgene_b\nA\tB\n", encoding="utf-8")
    values = {
        "schema_version": "1.0.0",
        "sources": {
            "string_ppi": {
                "edge_list": "edges.tsv",
                "edge_list_sha256": "0" * 64,
                "release": "STRING frozen release",
                "source_url": "https://string-db.org/releases/test",
                "construction_policy": "Mapped edge list",
            },
            "gene_ontology": {
                "edge_list": "edges.tsv",
                "edge_list_sha256": _sha256(edges),
                "release": "GO frozen release",
                "source_url": "https://geneontology.org/releases/test",
                "construction_policy": "Mapped edge list",
            },
        },
    }
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_frozen_graph_sources(
            values,
            config_path=tmp_path / "sources.json",
            required=["string_ppi"],
        )
    values["sources"]["string_ppi"]["edge_list_sha256"] = _sha256(edges)
    values["sources"]["string_ppi"]["release"] = "TODO"
    with pytest.raises(ValueError, match="placeholder"):
        load_frozen_graph_sources(
            values,
            config_path=tmp_path / "sources.json",
            required=["string_ppi"],
        )

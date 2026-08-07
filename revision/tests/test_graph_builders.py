from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cbac_revision.artifacts import canonical_sha256, file_sha256
from cbac_revision.graph_builders import build_go_edges, build_string_edges


def test_string_builder_enforces_human_taxon_and_exclusive_score_threshold(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "string.tsv"
    raw.write_text(
        "protein1\tprotein2\tcombined_score\n"
        "9606.P1\t9606.P2\t400\n"
        "9606.P1\t9606.P2\t401\n"
        "9606.P2\t9606.P1\t900\n"
        "10090.P1\t10090.P2\t999\n"
        "9606.P1\t9606.P3\t700\n",
        encoding="utf-8",
    )
    identifier_map = tmp_path / "map.tsv"
    identifier_map.write_text(
        "string_id\tgene_symbol\n" "9606.P1\tTP53\n" "9606.P2\tMYC\n" "9606.P3\tEGFR\n",
        encoding="utf-8",
    )
    output = tmp_path / "string_edges.csv"
    manifest_path = tmp_path / "string_manifest.json"

    manifest = build_string_edges(
        raw_path=raw,
        output_path=output,
        manifest_path=manifest_path,
        source_url="https://string-db.org/cgi/download",
        release="STRING_v12.0",
        taxon=9606,
        source_column="protein1",
        target_column="protein2",
        score_column="combined_score",
        delimiter="tab",
        map_delimiter="tab",
        taxon_column=None,
        protein_taxon_prefix="9606.",
        identifier_map_path=identifier_map,
        map_id_column="string_id",
        map_symbol_column="gene_symbol",
        identifiers_are_gene_symbols=False,
    )
    frame = pd.read_csv(output)

    assert frame.to_dict(orient="records") == [
        {"source": "EGFR", "target": "TP53", "combined_score": 700},
        {"source": "MYC", "target": "TP53", "combined_score": 900},
    ]
    assert manifest["policy"]["threshold_operator"] == ">"
    assert manifest["audit_counts"]["strict_score_passing_rows"] == 3
    payload = dict(manifest)
    declared = payload.pop("manifest_hash")
    assert declared == canonical_sha256(payload)


def _gaf_row(symbol: str, qualifier: str = "") -> str:
    fields = [
        "UniProtKB",
        f"ID_{symbol}",
        symbol,
        qualifier,
        "GO:0000002",
        "PMID:1",
        "EXP",
        "",
        "P",
        symbol,
        "",
        "protein",
        "taxon:9606",
        "20260806",
        "TEST",
        "",
        "",
    ]
    return "\t".join(fields)


def test_go_builder_applies_bp_evidence_not_and_ancestor_policy(tmp_path: Path) -> None:
    obo = tmp_path / "go.obo"
    obo.write_text(
        "format-version: 1.2\n\n"
        "[Term]\n"
        "id: GO:0000001\n"
        "name: root process\n"
        "namespace: biological_process\n\n"
        "[Term]\n"
        "id: GO:0000002\n"
        "name: child process\n"
        "namespace: biological_process\n"
        "is_a: GO:0000001 ! root process\n",
        encoding="utf-8",
    )
    gaf = tmp_path / "go.gaf"
    gaf.write_text(
        "!gaf-version: 2.2\n"
        + _gaf_row("TP53")
        + "\n"
        + _gaf_row("MYC")
        + "\n"
        + _gaf_row("EXCLUDED", "NOT")
        + "\n",
        encoding="utf-8",
    )
    metadata = tmp_path / "release.json"
    metadata.write_text(
        json.dumps(
            {
                "release": "2026-08-01",
                "source_url": "https://current.geneontology.org/",
                "gaf_sha256": file_sha256(gaf),
                "obo_sha256": file_sha256(obo),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "go_edges.csv"
    manifest_path = tmp_path / "go_manifest.json"

    manifest = build_go_edges(
        gaf_path=gaf,
        obo_path=obo,
        release_metadata_path=metadata,
        output_path=output,
        manifest_path=manifest_path,
        evidence_codes=("EXP",),
        include_all_evidence_codes=False,
        ancestor_policy="is_a_transitive",
        minimum_depth=0,
    )

    assert pd.read_csv(output).to_dict(orient="records") == [{"source": "MYC", "target": "TP53"}]
    assert manifest["policy"]["qualifier_policy"] == "exclude_NOT"
    assert manifest["policy"]["evidence_codes"] == ["EXP"]

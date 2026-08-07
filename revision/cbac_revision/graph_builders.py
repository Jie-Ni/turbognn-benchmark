"""Deterministic, hash-emitting builders for STRING and GO gene graphs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, TextIO

from .artifacts import canonical_sha256, file_sha256
from .errors import RevisionProtocolError


def build_string_edges(
    *,
    raw_path: Path,
    output_path: Path,
    manifest_path: Path,
    source_url: str,
    release: str,
    taxon: int,
    source_column: str,
    target_column: str,
    score_column: str,
    delimiter: str,
    map_delimiter: str,
    taxon_column: str | None,
    protein_taxon_prefix: str | None,
    identifier_map_path: Path | None,
    map_id_column: str,
    map_symbol_column: str,
    identifiers_are_gene_symbols: bool,
) -> dict[str, Any]:
    """Filter strict score >400 human STRING links and map endpoints to gene symbols."""

    if release != "STRING_v12.0" or taxon != 9606:
        raise RevisionProtocolError("STRING builder is frozen to STRING_v12.0 and taxon 9606")
    if bool(identifier_map_path) == bool(identifiers_are_gene_symbols):
        raise RevisionProtocolError(
            "Declare exactly one identifier policy: --identifier-map or "
            "--identifiers-are-gene-symbols"
        )
    if bool(taxon_column) == bool(protein_taxon_prefix):
        raise RevisionProtocolError(
            "Declare exactly one taxon policy: --taxon-column or --protein-taxon-prefix"
        )
    if protein_taxon_prefix and protein_taxon_prefix != "9606.":
        raise RevisionProtocolError("The frozen human STRING prefix is exactly '9606.'")
    mapping = (
        _load_identifier_map(
            identifier_map_path,
            id_column=map_id_column,
            symbol_column=map_symbol_column,
            delimiter=map_delimiter,
        )
        if identifier_map_path is not None
        else None
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    database_path = output_path.with_name(f".{output_path.name}.edges.sqlite")
    if database_path.exists():
        raise RevisionProtocolError(f"Temporary database already exists: {database_path.name}")
    connection = sqlite3.connect(database_path)
    total_rows = 0
    human_rows = 0
    passing_rows = 0
    mapped_rows = 0
    try:
        connection.execute(
            "CREATE TABLE edges (source TEXT, target TEXT, combined_score REAL, "
            "PRIMARY KEY (source, target))"
        )
        for row in _iter_named_rows(raw_path, delimiter):
            total_rows += 1
            for column in (source_column, target_column, score_column):
                if column not in row:
                    raise RevisionProtocolError(f"STRING input is missing column {column!r}")
            left_raw = str(row[source_column]).strip()
            right_raw = str(row[target_column]).strip()
            if taxon_column is not None:
                if taxon_column not in row:
                    raise RevisionProtocolError(
                        f"STRING input is missing taxon column {taxon_column!r}"
                    )
                try:
                    is_human = int(str(row[taxon_column]).strip()) == taxon
                except ValueError as error:
                    raise RevisionProtocolError("STRING taxon values must be integers") from error
            else:
                is_human = left_raw.startswith("9606.") and right_raw.startswith("9606.")
            if not is_human:
                continue
            human_rows += 1
            try:
                score = float(row[score_column])
            except ValueError as error:
                raise RevisionProtocolError("STRING combined scores must be numeric") from error
            if score <= 400:
                continue
            passing_rows += 1
            if mapping is None:
                left = left_raw
                right = right_raw
            else:
                left = mapping.get(left_raw)
                right = mapping.get(right_raw)
                if left is None or right is None:
                    continue
            if not left or not right or left == right:
                continue
            mapped_rows += 1
            source, target = sorted((left, right), key=str.casefold)
            connection.execute(
                "INSERT INTO edges(source, target, combined_score) VALUES (?, ?, ?) "
                "ON CONFLICT(source, target) DO UPDATE SET combined_score="
                "MAX(combined_score, excluded.combined_score)",
                (source, target, score),
            )
        connection.commit()
        edge_count = int(connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0])
        if edge_count == 0:
            raise RevisionProtocolError("[STRING_EDGE_OUTPUT_EMPTY]")
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["source", "target", "combined_score"])
            for source, target, score in connection.execute(
                "SELECT source, target, combined_score FROM edges "
                "ORDER BY source COLLATE NOCASE, target COLLATE NOCASE"
            ):
                writer.writerow([source, target, _format_number(score)])
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)
    builder_path = Path(__file__).resolve()
    identifier_rule = (
        "exact_declared_gene_symbols"
        if identifiers_are_gene_symbols
        else "explicit_identifier_map_exact_match"
    )
    inputs = {"string_raw": file_sha256(raw_path)}
    if identifier_map_path is not None:
        inputs["identifier_map"] = file_sha256(identifier_map_path)
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "source_name": "STRING",
        "source_release": release,
        "source_url": source_url,
        "edge_file_sha256": file_sha256(output_path),
        "builder": {
            "name": "cbac_revision.graph_builders:string",
            "version": "1.0",
            "code_sha256": file_sha256(builder_path),
        },
        "input_files": inputs,
        "identifier_mapping": {
            "rule": identifier_rule,
            "coverage": mapped_rows / passing_rows if passing_rows else 0.0,
            "passing_rows": passing_rows,
            "mapped_nonself_rows": mapped_rows,
        },
        "policy": {
            "species_taxon": taxon,
            "taxon_filter": (f"column:{taxon_column}" if taxon_column else "endpoint_prefix:9606."),
            "score_field": score_column,
            "threshold_operator": ">",
            "threshold_value": 400,
            "duplicate_policy": "undirected_pair_keep_max_combined_score",
            "self_edge_policy": "exclude",
        },
        "audit_counts": {
            "raw_rows": total_rows,
            "human_rows": human_rows,
            "strict_score_passing_rows": passing_rows,
            "output_unique_edges": edge_count,
        },
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    _write_json(manifest_path, manifest)
    return manifest


def build_go_edges(
    *,
    gaf_path: Path,
    obo_path: Path,
    release_metadata_path: Path,
    output_path: Path,
    manifest_path: Path,
    evidence_codes: Sequence[str] | None,
    include_all_evidence_codes: bool,
    ancestor_policy: str,
    minimum_depth: int,
) -> dict[str, Any]:
    """Build a GO Biological Process shared-annotation graph from GAF and OBO."""

    if bool(evidence_codes) == bool(include_all_evidence_codes):
        raise RevisionProtocolError(
            "Declare exactly one evidence policy: explicit --evidence-code values or "
            "--include-all-evidence-codes"
        )
    if ancestor_policy not in {"none", "is_a_transitive"}:
        raise RevisionProtocolError("ancestor_policy must be none or is_a_transitive")
    if minimum_depth < 0:
        raise RevisionProtocolError("minimum_depth cannot be negative")
    metadata = _read_json_object(release_metadata_path)
    if metadata.get("gaf_sha256") != file_sha256(gaf_path):
        raise RevisionProtocolError("[GO_GAF_CHECKSUM_MISMATCH]")
    if metadata.get("obo_sha256") != file_sha256(obo_path):
        raise RevisionProtocolError("[GO_OBO_CHECKSUM_MISMATCH]")
    release = str(metadata.get("release", "")).strip()
    source_url = str(metadata.get("source_url", "")).strip()
    if not release or not source_url or "REPLACE" in f"{release}{source_url}".upper():
        raise RevisionProtocolError("GO release metadata requires real release and source_url")
    terms, alt_ids = _parse_obo(obo_path)
    depths = _term_depths(terms)
    allowed_codes = set(str(value).strip() for value in evidence_codes or ())
    observed_codes: set[str] = set()
    annotations: dict[str, set[str]] = defaultdict(set)
    candidate_rows = 0
    retained_rows = 0
    for fields in _iter_gaf(gaf_path):
        candidate_rows += 1
        symbol = fields[2].strip()
        qualifiers = {value for value in fields[3].split("|") if value}
        term_id = alt_ids.get(fields[4].strip(), fields[4].strip())
        evidence = fields[6].strip()
        aspect = fields[8].strip()
        observed_codes.add(evidence)
        if "NOT" in qualifiers or aspect != "P":
            continue
        if not include_all_evidence_codes and evidence not in allowed_codes:
            continue
        term = terms.get(term_id)
        if term is None or term["namespace"] != "biological_process" or term["obsolete"]:
            continue
        retained_terms = {term_id}
        if ancestor_policy == "is_a_transitive":
            retained_terms.update(_ancestors(term_id, terms))
        retained_terms = {
            value
            for value in retained_terms
            if terms[value]["namespace"] == "biological_process"
            and not terms[value]["obsolete"]
            and depths[value] >= minimum_depth
        }
        if not retained_terms or not symbol:
            continue
        retained_rows += 1
        for retained_term in retained_terms:
            annotations[retained_term].add(symbol)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    database_path = output_path.with_name(f".{output_path.name}.edges.sqlite")
    if database_path.exists():
        raise RevisionProtocolError(f"Temporary database already exists: {database_path.name}")
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            "CREATE TABLE edges (source TEXT, target TEXT, PRIMARY KEY(source,target))"
        )
        for term_id in sorted(annotations):
            genes = sorted(annotations[term_id], key=str.casefold)
            for left_index, left in enumerate(genes):
                for right in genes[left_index + 1 :]:
                    source, target = sorted((left, right), key=str.casefold)
                    connection.execute(
                        "INSERT OR IGNORE INTO edges(source,target) VALUES (?,?)",
                        (source, target),
                    )
        connection.commit()
        edge_count = int(connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0])
        if edge_count == 0:
            raise RevisionProtocolError("[GO_EDGE_OUTPUT_EMPTY]")
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["source", "target"])
            writer.writerows(
                connection.execute(
                    "SELECT source,target FROM edges "
                    "ORDER BY source COLLATE NOCASE,target COLLATE NOCASE"
                )
            )
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)
    policy_codes = sorted(observed_codes if include_all_evidence_codes else allowed_codes)
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "source_name": "Gene Ontology",
        "source_release": release,
        "source_url": source_url,
        "edge_file_sha256": file_sha256(output_path),
        "builder": {
            "name": "cbac_revision.graph_builders:go",
            "version": "1.0",
            "code_sha256": file_sha256(Path(__file__).resolve()),
        },
        "input_files": {"go_gaf": file_sha256(gaf_path), "go_obo": file_sha256(obo_path)},
        "identifier_mapping": {
            "rule": "GAF_DB_Object_Symbol_exact",
            "coverage": retained_rows / candidate_rows if candidate_rows else 0.0,
            "candidate_annotation_rows": candidate_rows,
            "retained_annotation_rows": retained_rows,
        },
        "policy": {
            "namespace": "biological_process",
            "edge_rule": "shared_annotation",
            "qualifier_policy": "exclude_NOT",
            "evidence_code_policy": (
                "explicit_include_all_observed_codes"
                if include_all_evidence_codes
                else "explicit_allowlist"
            ),
            "evidence_codes": policy_codes,
            "ancestor_propagation_policy": ancestor_policy,
            "depth_policy": f"minimum_obo_is_a_depth_gte_{minimum_depth}",
            "self_edge_policy": "exclude",
            "duplicate_policy": "collapse_undirected_pairs",
        },
        "audit_counts": {
            "obo_terms": len(terms),
            "annotated_terms_after_policy": len(annotations),
            "output_unique_edges": edge_count,
        },
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    _write_json(manifest_path, manifest)
    return manifest


def _load_identifier_map(
    path: Path, *, id_column: str, symbol_column: str, delimiter: str
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in _iter_named_rows(path, delimiter):
        if id_column not in row or symbol_column not in row:
            raise RevisionProtocolError("Identifier map columns are missing")
        identifier = str(row[id_column]).strip()
        symbol = str(row[symbol_column]).strip()
        if not identifier or not symbol:
            continue
        prior = mapping.get(identifier)
        if prior is not None and prior != symbol:
            raise RevisionProtocolError(f"Ambiguous identifier mapping for {identifier!r}")
        mapping[identifier] = symbol
    if not mapping:
        raise RevisionProtocolError("Identifier map is empty")
    return mapping


def _iter_named_rows(path: Path, delimiter: str) -> Iterator[dict[str, str]]:
    handle = _open_text(path)
    with handle:
        if delimiter == "whitespace":
            header = handle.readline().strip().split()
            if not header:
                raise RevisionProtocolError(f"Input table {path.name} has no header")
            for line in handle:
                if not line.strip():
                    continue
                values = line.strip().split()
                if len(values) != len(header):
                    raise RevisionProtocolError(f"Malformed whitespace row in {path.name}")
                yield dict(zip(header, values, strict=True))
        else:
            token = {"tab": "\t", "comma": ","}.get(delimiter)
            if token is None:
                raise RevisionProtocolError("delimiter must be whitespace, tab, or comma")
            yield from csv.DictReader(handle, delimiter=token)


def _open_text(path: Path) -> TextIO:
    if path.suffix.casefold() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _parse_obo(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    terms: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    with _open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if line == "[Term]":
                if current and current.get("id"):
                    terms[current["id"]] = current
                current = {"parents": [], "alt_ids": [], "obsolete": False}
                continue
            if line.startswith("["):
                if current and current.get("id"):
                    terms[current["id"]] = current
                current = None
                continue
            if current is None or not line or line.startswith("!") or ": " not in line:
                continue
            key, value = line.split(": ", 1)
            if key == "id":
                current["id"] = value
            elif key == "namespace":
                current["namespace"] = value
            elif key == "is_a":
                current["parents"].append(value.split(" ! ", 1)[0])
            elif key == "alt_id":
                current["alt_ids"].append(value)
            elif key == "is_obsolete":
                current["obsolete"] = value.casefold() == "true"
    if current and current.get("id"):
        terms[current["id"]] = current
    if not terms:
        raise RevisionProtocolError("GO OBO contains no terms")
    alt_ids: dict[str, str] = {}
    for term_id, term in terms.items():
        term.setdefault("namespace", "")
        term["parents"] = [parent for parent in term["parents"] if parent in terms]
        for alt_id in term["alt_ids"]:
            if alt_id in alt_ids and alt_ids[alt_id] != term_id:
                raise RevisionProtocolError(f"Ambiguous GO alt_id {alt_id}")
            alt_ids[alt_id] = term_id
    return terms, alt_ids


def _term_depths(terms: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    cache: dict[str, int] = {}

    def depth(term_id: str, active: set[str]) -> int:
        if term_id in cache:
            return cache[term_id]
        if term_id in active:
            raise RevisionProtocolError("GO is_a graph contains a cycle")
        parents = terms[term_id]["parents"]
        value = (
            0 if not parents else 1 + min(depth(parent, active | {term_id}) for parent in parents)
        )
        cache[term_id] = value
        return value

    for term_id in terms:
        depth(term_id, set())
    return cache


def _ancestors(term_id: str, terms: Mapping[str, Mapping[str, Any]]) -> set[str]:
    output: set[str] = set()
    pending = list(terms[term_id]["parents"])
    while pending:
        parent = pending.pop()
        if parent in output:
            continue
        output.add(parent)
        pending.extend(terms[parent]["parents"])
    return output


def _iter_gaf(path: Path) -> Iterator[list[str]]:
    with _open_text(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("!"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 15:
                raise RevisionProtocolError("GAF data rows require at least 15 columns")
            yield fields


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RevisionProtocolError(f"Invalid JSON {path.name}: {error}") from error
    if not isinstance(payload, dict):
        raise RevisionProtocolError(f"JSON {path.name} must contain an object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else format(float(value), ".15g")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="builder", required=True)
    string = subparsers.add_parser("string")
    string.add_argument("--raw", type=Path, required=True)
    string.add_argument("--output", type=Path, required=True)
    string.add_argument("--manifest", type=Path, required=True)
    string.add_argument("--source-url", required=True)
    string.add_argument("--release", default="STRING_v12.0")
    string.add_argument("--taxon", type=int, default=9606)
    string.add_argument("--source-column", required=True)
    string.add_argument("--target-column", required=True)
    string.add_argument("--score-column", default="combined_score")
    string.add_argument("--delimiter", choices=("whitespace", "tab", "comma"), required=True)
    taxon = string.add_mutually_exclusive_group(required=True)
    taxon.add_argument("--taxon-column")
    taxon.add_argument("--protein-taxon-prefix")
    identifier = string.add_mutually_exclusive_group(required=True)
    identifier.add_argument("--identifier-map", type=Path)
    identifier.add_argument("--identifiers-are-gene-symbols", action="store_true")
    string.add_argument("--map-id-column", default="string_id")
    string.add_argument("--map-symbol-column", default="gene_symbol")
    string.add_argument("--map-delimiter", choices=("whitespace", "tab", "comma"), default="tab")

    go = subparsers.add_parser("go")
    go.add_argument("--gaf", type=Path, required=True)
    go.add_argument("--obo", type=Path, required=True)
    go.add_argument("--release-metadata", type=Path, required=True)
    go.add_argument("--output", type=Path, required=True)
    go.add_argument("--manifest", type=Path, required=True)
    evidence = go.add_mutually_exclusive_group(required=True)
    evidence.add_argument("--evidence-code", action="append")
    evidence.add_argument("--include-all-evidence-codes", action="store_true")
    go.add_argument("--ancestor-policy", choices=("none", "is_a_transitive"), required=True)
    go.add_argument("--minimum-depth", type=int, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_argument_parser().parse_args(arguments)
    if parsed.builder == "string":
        payload = build_string_edges(
            raw_path=parsed.raw,
            output_path=parsed.output,
            manifest_path=parsed.manifest,
            source_url=parsed.source_url,
            release=parsed.release,
            taxon=parsed.taxon,
            source_column=parsed.source_column,
            target_column=parsed.target_column,
            score_column=parsed.score_column,
            delimiter=parsed.delimiter,
            map_delimiter=parsed.map_delimiter,
            taxon_column=parsed.taxon_column,
            protein_taxon_prefix=parsed.protein_taxon_prefix,
            identifier_map_path=parsed.identifier_map,
            map_id_column=parsed.map_id_column,
            map_symbol_column=parsed.map_symbol_column,
            identifiers_are_gene_symbols=parsed.identifiers_are_gene_symbols,
        )
    else:
        payload = build_go_edges(
            gaf_path=parsed.gaf,
            obo_path=parsed.obo,
            release_metadata_path=parsed.release_metadata,
            output_path=parsed.output,
            manifest_path=parsed.manifest,
            evidence_codes=parsed.evidence_code,
            include_all_evidence_codes=parsed.include_all_evidence_codes,
            ancestor_policy=parsed.ancestor_policy,
            minimum_depth=parsed.minimum_depth,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

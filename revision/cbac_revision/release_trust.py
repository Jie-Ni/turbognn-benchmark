"""Detached caller-pinned trust boundary for the main scientific release chain."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .artifacts import canonical_sha256, file_sha256, read_fold_artifact
from .baselines import read_baseline_artifact
from .data import load_dataset_passport, load_named_edges
from .errors import RevisionProtocolError
from .precision import read_precision_registry
from .protocol import load_protocol
from .release_assets import validate_release_asset_manifest
from .runner import _build_environment_manifest, _validate_derivation_manifest, code_tree_sha256
from .shared_control import read_shared_control_evidence
from .statistics import (
    mandatory_fit_coverage_release,
    metric_frame_from_artifacts,
    topology_hierarchical_release,
)

SHA256_RE = re.compile(r"[0-9a-f]{64}")
TRUST_CATEGORIES = (
    "dataset_passports",
    "dataset_files",
    "graph_files",
    "string_go_source_and_policy",
    "split_condition_target_order",
    "protocol",
    "entrypoint_environment",
    "raw_predictions_traces",
    "baseline_inputs",
    "precision_simulation_inputs",
    "topology_diagnostics",
    "release_asset_manifest",
)
ENTRY_FIELDS = {"logical_id", "kind", "source_id", "sha256"}
CATEGORY_KINDS: dict[str, set[str]] = {
    "dataset_passports": {"dataset_passport"},
    "dataset_files": {"dataset_file"},
    "graph_files": {
        "string_edges",
        "string_raw",
        "string_identifier_map",
        "string_builder",
        "string_derivation_manifest",
        "go_edges",
        "go_gaf",
        "go_obo",
        "go_release_metadata",
        "go_builder",
        "go_derivation_manifest",
    },
    "string_go_source_and_policy": {"graph_provenance"},
    "split_condition_target_order": {
        "preflight_summary",
        "condition_panel_manifest",
        "target_encoding_audit",
        "nested_gene_panel_manifest",
        "shared_control_cell_evidence",
    },
    "protocol": {"protocol_yaml"},
    "entrypoint_environment": {"entrypoint_python", "environment_lock"},
    "raw_predictions_traces": {"artifact_manifest"},
    "baseline_inputs": {"baseline_manifest"},
    "precision_simulation_inputs": {
        "precision_registry",
        "precision_condition_table",
        "precision_target_map",
    },
    "topology_diagnostics": {"topology_diagnostics"},
    "release_asset_manifest": {"release_asset_manifest"},
}
ANCHOR_FIELDS = {
    "schema_version",
    "anchor_id",
    "role",
    "cryptographic_boundary",
    "categories",
    "anchor_sha256",
}


def validate_main_release_trust_anchor(
    anchor_path: Path,
    *,
    expected_anchor_file_sha256: str,
    evidence_bundle_root: Path,
) -> dict[str, Any]:
    """Validate all raw sources through a detached immutable caller-pinned anchor."""

    anchor, resolved = read_release_trust_anchor(
        anchor_path,
        expected_anchor_file_sha256=expected_anchor_file_sha256,
        evidence_bundle_root=evidence_bundle_root,
    )
    protocol_path = _single(resolved, "protocol", "protocol_yaml")
    protocol = load_protocol(protocol_path)
    protocol_hash = file_sha256(protocol_path)
    datasets = tuple(sorted(protocol["datasets"]))
    _validate_datasets(resolved, protocol, datasets)
    _validate_graph_sources(resolved, protocol)
    preflight_summaries = _validate_preflight_sources(resolved, protocol_hash, datasets)
    _validate_entrypoint_environment(resolved)
    precision_registry = _validate_precision_sources(resolved)

    artifact_paths = _validate_artifact_manifests(resolved)
    metrics = metric_frame_from_artifacts(artifact_paths)
    coverage = mandatory_fit_coverage_release(
        metrics,
        protocol,
        preflight_summaries,
        protocol_file_hash=protocol_hash,
        expected_code_hash=code_tree_sha256(),
    )
    if not coverage.released:
        raise RevisionProtocolError("[MAIN_TRUST_MANDATORY_FIT_COVERAGE_FAILED]")
    baseline_summary = _validate_baseline_manifests(resolved, protocol)
    topology = topology_hierarchical_release(metrics, protocol)
    if not topology.released:
        raise RevisionProtocolError("[MAIN_TRUST_TOPOLOGY_DIAGNOSTICS_FAILED]")
    _validate_topology_table_binding(resolved, topology.detail_tables["graph_diagnostics"])
    release_manifest_path = _single(resolved, "release_asset_manifest", "release_asset_manifest")
    release_manifest = validate_release_asset_manifest(
        release_manifest_path,
        expected_manifest_file_sha256=file_sha256(release_manifest_path),
    )
    if release_manifest["mode"] != "released" or release_manifest["status"] != "RELEASED_ASSETS":
        raise RevisionProtocolError("[MAIN_TRUST_RELEASE_ASSETS_NOT_RELEASED]")
    _cross_bind_artifacts(
        artifact_paths,
        protocol_hash=protocol_hash,
        precision_registry_sha256=precision_registry["registry_sha256"],
        anchored_target_audit_hashes=_anchored_target_audit_record_hashes(resolved),
        anchored_shared_control_hashes=_anchored_shared_control_hashes(resolved),
    )
    registry: dict[str, Any] = {
        "schema_version": "1.0",
        "registry_id": "MAIN-RELEASE-TRUST-ANCHOR",
        "status": "RELEASED",
        "anchor_file_sha256": expected_anchor_file_sha256,
        "anchor_self_hash": anchor["anchor_sha256"],
        "protocol_file_sha256": protocol_hash,
        "code_tree_sha256": code_tree_sha256(),
        "artifact_count": len(artifact_paths),
        "artifact_content_manifest_hash": canonical_sha256(
            sorted(read_fold_artifact(path)["artifact_hash"] for path in artifact_paths)
        ),
        "ridge_refits": baseline_summary["ridge_refits"],
        "analytic_baseline_evaluations": baseline_summary["analytic_baseline_evaluations"],
        "precision_registry_sha256": precision_registry["registry_sha256"],
        "topology_diagnostic_records_sha256": _records_hash(
            topology.detail_tables["graph_diagnostics"]
        ),
        "release_asset_manifest_file_sha256": file_sha256(release_manifest_path),
        "cryptographic_boundary": (
            "valid_only_while_caller_pinned_anchor_file_sha256_is_immutable_to_the_"
            "evidence_bundle_attacker"
        ),
    }
    registry["registry_hash"] = canonical_sha256(registry)
    return registry


def read_release_trust_anchor(
    anchor_path: Path,
    *,
    expected_anchor_file_sha256: str,
    evidence_bundle_root: Path,
) -> tuple[dict[str, Any], dict[str, list[tuple[dict[str, str], Path]]]]:
    """Verify the detached anchor, every source hash, and exact category/kind contracts."""

    if not SHA256_RE.fullmatch(expected_anchor_file_sha256):
        raise RevisionProtocolError("[MAIN_TRUST_CALLER_PINNED_SHA256_INVALID]")
    anchor_resolved = anchor_path.resolve()
    bundle_resolved = evidence_bundle_root.resolve()
    if anchor_resolved.is_relative_to(bundle_resolved):
        raise RevisionProtocolError("[MAIN_TRUST_ANCHOR_NOT_DETACHED]")
    if not anchor_path.is_file() or file_sha256(anchor_path) != expected_anchor_file_sha256:
        raise RevisionProtocolError("[MAIN_TRUST_CALLER_PINNED_ANCHOR_MISMATCH]")
    try:
        anchor = json.loads(
            anchor_path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RevisionProtocolError("[MAIN_TRUST_ANCHOR_INVALID]") from error
    if not isinstance(anchor, dict) or set(anchor) != ANCHOR_FIELDS:
        raise RevisionProtocolError("[MAIN_TRUST_ANCHOR_SCHEMA_INVALID]")
    unsigned = dict(anchor)
    declared_hash = unsigned.pop("anchor_sha256")
    if (
        anchor["schema_version"] != "1.0"
        or anchor["anchor_id"] != "CBAC-MAIN-RELEASE-TRUST-V1"
        or anchor["role"] != "CALLER_PINNED_DETACHED_TRUST_ANCHOR"
        or anchor["cryptographic_boundary"]
        != "caller_pinned_file_sha256_must_be_immutable_to_bundle_attacker"
        or declared_hash != canonical_sha256(unsigned)
        or not isinstance(anchor["categories"], dict)
        or set(anchor["categories"]) != set(TRUST_CATEGORIES)
    ):
        raise RevisionProtocolError("[MAIN_TRUST_ANCHOR_HASH_OR_CONTRACT_INVALID]")
    resolved: dict[str, list[tuple[dict[str, str], Path]]] = {}
    root = anchor_path.resolve().parent
    all_logical_ids: set[tuple[str, str, str]] = set()
    for category in TRUST_CATEGORIES:
        entries = anchor["categories"][category]
        if not isinstance(entries, list) or not entries:
            raise RevisionProtocolError(f"[MAIN_TRUST_CATEGORY_EMPTY] {category}")
        category_rows: list[tuple[dict[str, str], Path]] = []
        observed_kinds: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != ENTRY_FIELDS:
                raise RevisionProtocolError(f"[MAIN_TRUST_ENTRY_SCHEMA_INVALID] {category}")
            if any(not isinstance(entry[field], str) or not entry[field] for field in ENTRY_FIELDS):
                raise RevisionProtocolError(f"[MAIN_TRUST_ENTRY_VALUE_INVALID] {category}")
            if entry["kind"] not in CATEGORY_KINDS[category]:
                raise RevisionProtocolError(f"[MAIN_TRUST_ENTRY_KIND_INVALID] {category}")
            key = (category, entry["kind"], entry["logical_id"])
            if key in all_logical_ids:
                raise RevisionProtocolError(f"[MAIN_TRUST_DUPLICATE_LOGICAL_ID] {category}")
            all_logical_ids.add(key)
            source = _resolve_source(root, entry["source_id"])
            if (
                source is None
                or not source.is_file()
                or not SHA256_RE.fullmatch(entry["sha256"])
                or file_sha256(source) != entry["sha256"]
            ):
                raise RevisionProtocolError(
                    f"[MAIN_TRUST_SOURCE_BINDING_INVALID] {category}/{entry['logical_id']}"
                )
            observed_kinds.add(entry["kind"])
            category_rows.append((dict(entry), source))
        if observed_kinds != CATEGORY_KINDS[category]:
            missing = sorted(CATEGORY_KINDS[category] - observed_kinds)
            raise RevisionProtocolError(f"[MAIN_TRUST_REQUIRED_KIND_MISSING] {category}:{missing}")
        resolved[category] = category_rows
    return anchor, resolved


def _validate_datasets(
    resolved: Mapping[str, Sequence[tuple[dict[str, str], Path]]],
    protocol: Mapping[str, Any],
    datasets: Sequence[str],
) -> None:
    passports = _by_logical_id(resolved["dataset_passports"], "dataset_passport")
    files = _by_logical_id(resolved["dataset_files"], "dataset_file")
    if set(passports) != set(datasets) or set(files) != set(datasets):
        raise RevisionProtocolError("[MAIN_TRUST_DATASET_SET_INVALID]")
    for dataset in datasets:
        dataset_protocol = protocol["datasets"][dataset]
        expected_condition_column = dataset_protocol["condition_column"]
        if expected_condition_column == "DATASET_PASSPORT_REQUIRED":
            expected_condition_column = None
        load_dataset_passport(
            passports[dataset],
            dataset_name=dataset,
            data_path=files[dataset],
            expected_scale=protocol["preprocessing"]["input_scale"],
            expected_condition_column=expected_condition_column,
        )


def _validate_graph_sources(
    resolved: Mapping[str, Sequence[tuple[dict[str, str], Path]]],
    protocol: Mapping[str, Any],
) -> None:
    graph_entries = resolved["graph_files"]
    by_kind = {entry["kind"]: path for entry, path in graph_entries}
    for kind in ("string_edges", "go_edges"):
        if not load_named_edges(by_kind[kind]):
            raise RevisionProtocolError(f"[MAIN_TRUST_GRAPH_EDGE_FILE_EMPTY] {kind}")
    go_release = json.loads(by_kind["go_release_metadata"].read_text(encoding="utf-8"))
    if (
        not isinstance(go_release, dict)
        or not str(go_release.get("release", "")).strip()
        or go_release.get("gaf_sha256") != file_sha256(by_kind["go_gaf"])
        or go_release.get("obo_sha256") != file_sha256(by_kind["go_obo"])
    ):
        raise RevisionProtocolError("[MAIN_TRUST_GRAPH_PROVENANCE_SEMANTICS_INVALID]")
    _, _, string_failures = _validate_derivation_manifest(
        kind="string",
        manifest_path=by_kind["string_derivation_manifest"],
        edge_path=by_kind["string_edges"],
        builder_path=by_kind["string_builder"],
        input_paths={
            "string_raw": by_kind["string_raw"],
            "identifier_map": by_kind["string_identifier_map"],
        },
        expected_release=protocol["graph_composition"]["string_ppi"]["source"],
    )
    _, _, go_failures = _validate_derivation_manifest(
        kind="go",
        manifest_path=by_kind["go_derivation_manifest"],
        edge_path=by_kind["go_edges"],
        builder_path=by_kind["go_builder"],
        input_paths={"go_gaf": by_kind["go_gaf"], "go_obo": by_kind["go_obo"]},
        expected_release=str(go_release["release"]),
    )
    if string_failures or go_failures:
        raise RevisionProtocolError(
            f"[MAIN_TRUST_GRAPH_DERIVATION_INVALID] {string_failures + go_failures}"
        )
    anchored_hashes = {path.name: file_sha256(path) for _, path in graph_entries}
    for _, path in resolved["string_go_source_and_policy"]:
        frame = pd.read_csv(path)
        required = {"source", "release", "source_id", "sha256", "status"}
        if frame.empty or not required <= set(frame.columns):
            raise RevisionProtocolError("[MAIN_TRUST_GRAPH_PROVENANCE_TABLE_INVALID]")
        for row in frame.itertuples(index=False):
            if row.source_id in anchored_hashes and row.sha256 != anchored_hashes[row.source_id]:
                raise RevisionProtocolError("[MAIN_TRUST_GRAPH_PROVENANCE_HASH_MISMATCH]")


def _validate_preflight_sources(
    resolved: Mapping[str, Sequence[tuple[dict[str, str], Path]]],
    protocol_hash: str,
    datasets: Sequence[str],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    summaries_by_id: dict[str, dict[str, Any]] = {}
    shared_control_datasets: set[str] = set()
    for entry, path in resolved["split_condition_target_order"]:
        if entry["kind"] != "preflight_summary":
            continue
        payload = _read_json(path)
        unsigned = dict(payload)
        declared_hash = unsigned.pop("summary_hash", None)
        if (
            declared_hash != canonical_sha256(unsigned)
            or payload.get("dataset") not in datasets
            or payload.get("input_hashes", {}).get("protocol") != protocol_hash
            or payload.get("n_selected_conditions") != 50
            or len(payload.get("selected_conditions", ())) != 50
            or len(set(payload.get("selected_conditions", ()))) != 50
            or payload.get("can_execute") is not True
        ):
            raise RevisionProtocolError("[MAIN_TRUST_PREFLIGHT_SUMMARY_INVALID]")
        summaries.append(payload)
        if entry["logical_id"] in summaries_by_id:
            raise RevisionProtocolError("[MAIN_TRUST_PREFLIGHT_LOGICAL_ID_DUPLICATE]")
        summaries_by_id[entry["logical_id"]] = payload
    if not summaries or set(summary["dataset"] for summary in summaries) != set(datasets):
        raise RevisionProtocolError("[MAIN_TRUST_PREFLIGHT_DATASET_COVERAGE_INVALID]")
    for entry, path in resolved["split_condition_target_order"]:
        if entry["kind"] == "condition_panel_manifest":
            _validate_named_self_hash(path, ("manifest_hash",))
        elif entry["kind"] == "nested_gene_panel_manifest":
            _validate_named_self_hash(path, ("manifest_sha256",))
        elif entry["kind"] == "target_encoding_audit":
            summary = summaries_by_id.get(entry["logical_id"])
            if summary is None:
                raise RevisionProtocolError("[MAIN_TRUST_TARGET_AUDIT_SUMMARY_BINDING_MISSING]")
            _validate_target_audit(path, summary)
        elif entry["kind"] == "shared_control_cell_evidence":
            summary = summaries_by_id.get(entry["logical_id"])
            if summary is None:
                raise RevisionProtocolError("[MAIN_TRUST_SHARED_CONTROL_SUMMARY_BINDING_MISSING]")
            manifest, _ = read_shared_control_evidence(path)
            if (
                manifest["dataset"] != summary["dataset"]
                or manifest["hvg"] != 200
                or manifest["manifest_sha256"] != summary.get("shared_control_cell_evidence_hash")
                or manifest["dataset_sha256"] != summary["input_hashes"]["dataset"]
                or manifest["preprocessing_state_sha256"] != summary["preprocessing_state_hash"]
            ):
                raise RevisionProtocolError("[MAIN_TRUST_SHARED_CONTROL_BINDING_INVALID]")
            shared_control_datasets.add(str(manifest["dataset"]))
    if shared_control_datasets != set(datasets):
        raise RevisionProtocolError("[MAIN_TRUST_SHARED_CONTROL_DATASET_COVERAGE_INVALID]")
    return summaries


def _validate_target_audit(path: Path, summary: Mapping[str, Any]) -> None:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "condition",
        "status",
        "canonical_targets",
        "target_indices",
        "target_forced_in_current_run",
    }
    if frame.empty or not required <= set(frame.columns):
        raise RevisionProtocolError("[MAIN_TRUST_TARGET_AUDIT_SCHEMA_INVALID]")
    genes = tuple(str(value) for value in summary["selected_genes"])
    gene_index = {gene.casefold(): index for index, gene in enumerate(genes)}
    selected = frame[frame["condition"].isin(summary["selected_conditions"])]
    if len(selected) != 50 or set(selected["condition"]) != set(summary["selected_conditions"]):
        raise RevisionProtocolError("[MAIN_TRUST_TARGET_AUDIT_PANEL_COVERAGE_INVALID]")
    for row in selected.itertuples(index=False):
        targets = tuple(value for value in str(row.canonical_targets).split("|") if value)
        try:
            indices = tuple(int(value) for value in str(row.target_indices).split("|") if value)
        except ValueError as error:
            raise RevisionProtocolError("[MAIN_TRUST_TARGET_INDEX_TYPE_INVALID]") from error
        recomputed = tuple(
            sorted(
                {
                    gene_index[target.casefold()]
                    for target in targets
                    if target.casefold() in gene_index
                }
            )
        )
        if (
            row.status != "SELECTED_PANEL"
            or not targets
            or not indices
            or indices != recomputed
            or any(index < 0 or index >= len(genes) for index in indices)
        ):
            raise RevisionProtocolError("[MAIN_TRUST_TARGET_MAPPING_RECOMPUTATION_FAILED]")


def _validate_entrypoint_environment(
    resolved: Mapping[str, Sequence[tuple[dict[str, str], Path]]],
) -> None:
    environment_paths = []
    for entry, path in resolved["entrypoint_environment"]:
        if entry["kind"] == "entrypoint_python":
            try:
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
            except (OSError, SyntaxError, UnicodeDecodeError) as error:
                raise RevisionProtocolError("[MAIN_TRUST_ENTRYPOINT_INVALID]") from error
        else:
            environment_paths.append(path)
    if len(environment_paths) != 1:
        raise RevisionProtocolError("[MAIN_TRUST_ENVIRONMENT_LOCK_COUNT_INVALID]")
    _build_environment_manifest(environment_paths[0], fixture_mode=False, require_h5ad_runtime=True)


def _validate_precision_sources(
    resolved: Mapping[str, Sequence[tuple[dict[str, str], Path]]],
) -> dict[str, Any]:
    registry = _single(resolved, "precision_simulation_inputs", "precision_registry")
    condition_table = _single(resolved, "precision_simulation_inputs", "precision_condition_table")
    target_map = _single(resolved, "precision_simulation_inputs", "precision_target_map")
    return read_precision_registry(
        registry,
        condition_table_path=condition_table,
        target_map_path=target_map,
        expected_condition_table_sha256=file_sha256(condition_table),
        expected_target_map_sha256=file_sha256(target_map),
    )


def _validate_artifact_manifests(
    resolved: Mapping[str, Sequence[tuple[dict[str, str], Path]]],
) -> list[Path]:
    paths: list[Path] = []
    seen_identity: set[tuple[Any, ...]] = set()
    for _, manifest_path in resolved["raw_predictions_traces"]:
        frame = pd.read_csv(manifest_path)
        if tuple(frame.columns) != ("source_id", "file_sha256", "artifact_sha256") or frame.empty:
            raise RevisionProtocolError("[MAIN_TRUST_ARTIFACT_MANIFEST_SCHEMA_INVALID]")
        for row in frame.itertuples(index=False):
            artifact_path = _resolve_source(manifest_path.resolve().parent, row.source_id)
            if artifact_path is None or file_sha256(artifact_path) != row.file_sha256:
                raise RevisionProtocolError("[MAIN_TRUST_ARTIFACT_FILE_HASH_MISMATCH]")
            payload = read_fold_artifact(artifact_path)
            if payload["artifact_hash"] != row.artifact_sha256:
                raise RevisionProtocolError("[MAIN_TRUST_ARTIFACT_CONTENT_HASH_MISMATCH]")
            identity = payload["identity"]
            key = tuple(
                identity[field] for field in ("dataset", "hvg", "panel", "arm", "condition", "seed")
            )
            if key in seen_identity:
                raise RevisionProtocolError("[MAIN_TRUST_DUPLICATE_FOLD_IDENTITY]")
            seen_identity.add(key)
            paths.append(artifact_path)
    if len(paths) != 10800:
        raise RevisionProtocolError(f"[MAIN_TRUST_ARTIFACT_COUNT_INVALID] observed={len(paths)}")
    return paths


def _validate_baseline_manifests(
    resolved: Mapping[str, Sequence[tuple[dict[str, str], Path]]],
    protocol: Mapping[str, Any],
) -> dict[str, int]:
    expected_columns = (
        "source_id",
        "file_sha256",
        "artifact_sha256",
        "dataset",
        "hvg",
        "panel",
        "condition",
        "ridge_refits",
        "analytic_baseline_evaluations",
    )
    identities: set[tuple[Any, ...]] = set()
    ridge_refits = 0
    evaluations = 0
    for _, manifest_path in resolved["baseline_inputs"]:
        frame = pd.read_csv(manifest_path)
        if tuple(frame.columns) != expected_columns or frame.empty:
            raise RevisionProtocolError("[MAIN_TRUST_BASELINE_MANIFEST_SCHEMA_INVALID]")
        for row in frame.itertuples(index=False):
            path = _resolve_source(manifest_path.resolve().parent, row.source_id)
            if path is None or file_sha256(path) != row.file_sha256:
                raise RevisionProtocolError("[MAIN_TRUST_BASELINE_FILE_HASH_MISMATCH]")
            payload = read_baseline_artifact(path)
            identity = payload["identity"]
            key = tuple(identity[field] for field in ("dataset", "hvg", "panel", "condition"))
            if key in identities or identity != {
                "dataset": row.dataset,
                "hvg": int(row.hvg),
                "panel": row.panel,
                "condition": row.condition,
            }:
                raise RevisionProtocolError("[MAIN_TRUST_BASELINE_IDENTITY_INVALID]")
            identities.add(key)
            if payload["artifact_sha256"] != row.artifact_sha256:
                raise RevisionProtocolError("[MAIN_TRUST_BASELINE_CONTENT_HASH_MISMATCH]")
            ridge_refits += int(payload["accounting"]["ridge_refits"])
            evaluations += int(payload["accounting"]["analytic_baseline_evaluations"])
    expected = protocol["planned_fit_counts"]
    if (
        ridge_refits != expected["analytic_ridge_refits"]
        or evaluations != expected["analytic_baseline_evaluations"]
    ):
        raise RevisionProtocolError(
            "[MAIN_TRUST_BASELINE_ACCOUNTING_INVALID] "
            f"ridge={ridge_refits}; evaluations={evaluations}"
        )
    return {"ridge_refits": ridge_refits, "analytic_baseline_evaluations": evaluations}


def _validate_topology_table_binding(
    resolved: Mapping[str, Sequence[tuple[dict[str, str], Path]]],
    recomputed: pd.DataFrame,
) -> None:
    path = _single(resolved, "topology_diagnostics", "topology_diagnostics")
    retained = pd.read_csv(path)
    if tuple(retained.columns) != tuple(recomputed.columns) or _records_hash(
        retained
    ) != _records_hash(recomputed):
        raise RevisionProtocolError("[MAIN_TRUST_TOPOLOGY_TABLE_RECOMPUTATION_MISMATCH]")


def _cross_bind_artifacts(
    paths: Sequence[Path],
    *,
    protocol_hash: str,
    precision_registry_sha256: str,
    anchored_target_audit_hashes: set[str],
    anchored_shared_control_hashes: set[str],
) -> None:
    for path in paths:
        payload = read_fold_artifact(path)
        hashes = payload["input_hashes"]
        if (
            hashes.get("protocol") != protocol_hash
            or hashes.get("precision_design_registry") != precision_registry_sha256
            or hashes.get("target_encoding_audit") not in anchored_target_audit_hashes
            or hashes.get("shared_control_cell_evidence") not in anchored_shared_control_hashes
        ):
            raise RevisionProtocolError("[MAIN_TRUST_ARTIFACT_UPSTREAM_BINDING_INVALID]")


def _anchored_target_audit_record_hashes(
    resolved: Mapping[str, Sequence[tuple[dict[str, str], Path]]],
) -> set[str]:
    return {
        canonical_sha256(pd.read_csv(path).fillna("NOT_APPLICABLE").to_dict(orient="records"))
        for entry, path in resolved["split_condition_target_order"]
        if entry["kind"] == "target_encoding_audit"
    }


def _anchored_shared_control_hashes(
    resolved: Mapping[str, Sequence[tuple[dict[str, str], Path]]],
) -> set[str]:
    return {
        read_shared_control_evidence(path)[0]["manifest_sha256"]
        for entry, path in resolved["split_condition_target_order"]
        if entry["kind"] == "shared_control_cell_evidence"
    }


def _single(
    resolved: Mapping[str, Sequence[tuple[dict[str, str], Path]]],
    category: str,
    kind: str,
) -> Path:
    matches = [path for entry, path in resolved[category] if entry["kind"] == kind]
    if len(matches) != 1:
        raise RevisionProtocolError(f"[MAIN_TRUST_SINGLETON_COUNT_INVALID] {category}/{kind}")
    return matches[0]


def _by_logical_id(entries: Sequence[tuple[dict[str, str], Path]], kind: str) -> dict[str, Path]:
    return {entry["logical_id"]: path for entry, path in entries if entry["kind"] == kind}


def _resolve_source(root: Path, source_id: Any) -> Path | None:
    if not isinstance(source_id, str) or not source_id:
        return None
    relative = Path(source_id)
    if relative.is_absolute():
        return None
    return (root / relative).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RevisionProtocolError(f"[MAIN_TRUST_JSON_INVALID] {path.name}") from error
    if not isinstance(payload, dict):
        raise RevisionProtocolError(f"[MAIN_TRUST_JSON_ROOT_INVALID] {path.name}")
    return payload


def _read_self_hashed_json(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    for field in ("manifest_sha256", "manifest_hash", "derivation_manifest_sha256"):
        if field in payload:
            unsigned = dict(payload)
            declared = unsigned.pop(field)
            if declared != canonical_sha256(unsigned):
                raise RevisionProtocolError(f"[MAIN_TRUST_JSON_SELF_HASH_INVALID] {path.name}")
            return payload
    raise RevisionProtocolError(f"[MAIN_TRUST_JSON_SELF_HASH_MISSING] {path.name}")


def _validate_named_self_hash(path: Path, candidates: Sequence[str]) -> None:
    payload = _read_json(path)
    for field in candidates:
        if field in payload:
            unsigned = dict(payload)
            declared = unsigned.pop(field)
            if declared == canonical_sha256(unsigned):
                return
    raise RevisionProtocolError(f"[MAIN_TRUST_NAMED_SELF_HASH_INVALID] {path.name}")


def _records_hash(frame: pd.DataFrame) -> str:
    return canonical_sha256(
        frame.astype(object).where(pd.notna(frame), "NOT_APPLICABLE").to_dict(orient="records")
    )


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant {value!r} is prohibited")

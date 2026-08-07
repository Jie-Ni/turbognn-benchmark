"""Fail-closed validator for an official GEARS positive-control reproduction."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import canonical_sha256, file_sha256
from .errors import RevisionProtocolError
from .runner import (
    ENVIRONMENT_LOCK_FORMAT,
    ENVIRONMENT_LOCK_HEADER_ORDER,
    PYG_EXTENSION_PACKAGES,
    _direct_runtime_dependencies,
)

COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDER_RE = re.compile(r"REPLACE|TODO|TBD|UNKNOWN|PLACEHOLDER", re.IGNORECASE)
TRUST_ANCHOR_ID = "GEARS-DETACHED-TRUST-ANCHOR"
TRUST_MODEL = "caller_pinned_sha256_over_detached_anchor_and_external_sources"
BOUND_FILE_FIELDS = (
    "environment_lock",
    "dataset",
    "official_metric_source",
    "execution_plan",
    "official_reference_execution_audit",
    "candidate_execution_audit",
    "official_reference_raw_evidence",
    "candidate_raw_evidence",
    "official_reference_entrypoint",
    "candidate_entrypoint",
    "official_reference_expected_value_provenance",
    "candidate_expected_value_provenance",
    "official_reference_initialization_state",
    "official_reference_checkpoint_state",
    "official_reference_graph_state",
    "candidate_initialization_state",
    "candidate_checkpoint_state",
    "candidate_graph_state",
    "official_reference_output",
    "candidate_output",
)


def create_gears_trust_anchor(manifest_path: Path, anchor_path: Path) -> Path:
    """Create detached trusted copies and a caller-pinnable anchor file."""

    manifest_path = manifest_path.resolve()
    evidence_root = manifest_path.parent
    anchor_path = anchor_path.resolve()
    if anchor_path.is_relative_to(evidence_root):
        raise RevisionProtocolError("[GEARS_TRUST_ANCHOR_INSIDE_EVIDENCE_BUNDLE]")
    if anchor_path.exists():
        raise RevisionProtocolError("[GEARS_TRUST_ANCHOR_ALREADY_EXISTS]")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RevisionProtocolError("[GEARS_TRUST_ANCHOR_MANIFEST_INVALID]") from error
    if not isinstance(manifest, dict):
        raise RevisionProtocolError("[GEARS_TRUST_ANCHOR_MANIFEST_INVALID]")
    manifest_payload = dict(manifest)
    declared_manifest_hash = manifest_payload.pop("manifest_hash", None)
    if declared_manifest_hash != canonical_sha256(manifest_payload):
        raise RevisionProtocolError("[GEARS_TRUST_ANCHOR_MANIFEST_INVALID]")

    resolution_failures: list[dict[str, str]] = []
    source_paths: dict[str, Path] = {}
    for field in BOUND_FILE_FIELDS:
        binding = manifest.get(field)
        if not isinstance(binding, Mapping):
            resolution_failures.append(
                {"reason_code": "GEARS_FILE_BINDING_MISSING", "detail": field}
            )
            continue
        resolved = _resolve_bound_file(evidence_root, binding, field, resolution_failures)
        if resolved is not None:
            source_paths[field] = resolved
    if resolution_failures or set(source_paths) != set(BOUND_FILE_FIELDS):
        raise RevisionProtocolError(
            "[GEARS_TRUST_ANCHOR_SOURCE_INVALID] " + json.dumps(resolution_failures, sort_keys=True)
        )

    anchor_path.parent.mkdir(parents=True, exist_ok=True)
    trusted_source_root = anchor_path.parent / f"{anchor_path.stem}.sources"
    if trusted_source_root.exists():
        raise RevisionProtocolError("[GEARS_TRUST_ANCHOR_SOURCE_STORE_ALREADY_EXISTS]")
    trusted_source_root.mkdir()
    trusted_files: dict[str, dict[str, str]] = {}
    trusted_paths: dict[str, Path] = {}
    for field, source in source_paths.items():
        suffix = source.suffix if source.suffix else ".bin"
        destination = trusted_source_root / f"{field}{suffix}"
        shutil.copyfile(source, destination)
        if file_sha256(destination) != file_sha256(source):
            raise RevisionProtocolError(f"[GEARS_TRUST_ANCHOR_COPY_MISMATCH] {field}")
        trusted_paths[field] = destination
        trusted_files[field] = {
            "source_id": destination.relative_to(anchor_path.parent).as_posix(),
            "sha256": file_sha256(destination),
        }

    semantic_commitments = _compute_trusted_semantic_commitments(trusted_paths)
    anchor: dict[str, Any] = {
        "schema_version": "1.0",
        "anchor_id": TRUST_ANCHOR_ID,
        "trust_model": TRUST_MODEL,
        "evidence_manifest_sha256": file_sha256(manifest_path),
        "evidence_manifest_hash": declared_manifest_hash,
        "trusted_files": trusted_files,
        "semantic_commitments": semantic_commitments,
    }
    anchor["anchor_hash"] = canonical_sha256(anchor)
    temporary = anchor_path.with_name(f".{anchor_path.name}.tmp")
    temporary.write_text(
        json.dumps(anchor, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(anchor_path)
    return anchor_path


def _validate_detached_trust_anchor(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    bound_files: Mapping[str, Path],
    trust_anchor_path: Path | None,
    expected_trust_anchor_sha256: str | None,
    failures: list[dict[str, str]],
) -> dict[str, str]:
    if trust_anchor_path is None or expected_trust_anchor_sha256 is None:
        failures.append(
            {
                "reason_code": "GEARS_TRUST_ANCHOR_MISSING",
                "detail": "detached anchor path and caller-pinned SHA256 are both required",
            }
        )
        return {}
    if not isinstance(expected_trust_anchor_sha256, str) or not SHA256_RE.fullmatch(
        expected_trust_anchor_sha256
    ):
        failures.append(
            {
                "reason_code": "GEARS_TRUST_ANCHOR_EXPECTED_SHA256_INVALID",
                "detail": "64 lowercase hexadecimal characters required",
            }
        )
        return {}

    manifest_path = manifest_path.resolve()
    evidence_root = manifest_path.parent
    anchor_path = trust_anchor_path.resolve()
    bindings = {
        "trust_anchor_id": TRUST_ANCHOR_ID,
        "trust_anchor_sha256": expected_trust_anchor_sha256,
        "trust_model": TRUST_MODEL,
    }
    if anchor_path.is_relative_to(evidence_root):
        failures.append(
            {
                "reason_code": "GEARS_TRUST_ANCHOR_INSIDE_EVIDENCE_BUNDLE",
                "detail": str(anchor_path),
            }
        )
        return bindings
    if not anchor_path.is_file():
        failures.append({"reason_code": "GEARS_TRUST_ANCHOR_MISSING", "detail": str(anchor_path)})
        return bindings
    observed_anchor_sha256 = file_sha256(anchor_path)
    if observed_anchor_sha256 != expected_trust_anchor_sha256:
        failures.append(
            {
                "reason_code": "GEARS_TRUST_ANCHOR_SHA256_MISMATCH",
                "detail": "anchor bytes differ from the caller-pinned SHA256",
            }
        )
        return bindings
    try:
        anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        failures.append({"reason_code": "GEARS_TRUST_ANCHOR_INVALID", "detail": str(error)})
        return bindings
    required_fields = {
        "schema_version",
        "anchor_id",
        "trust_model",
        "evidence_manifest_sha256",
        "evidence_manifest_hash",
        "trusted_files",
        "semantic_commitments",
        "anchor_hash",
    }
    if (
        not isinstance(anchor, dict)
        or set(anchor) != required_fields
        or _contains_placeholder(anchor)
    ):
        failures.append({"reason_code": "GEARS_TRUST_ANCHOR_INVALID", "detail": "root schema"})
        return bindings
    anchor_payload = dict(anchor)
    declared_anchor_hash = anchor_payload.pop("anchor_hash", None)
    if (
        anchor.get("schema_version") != "1.0"
        or anchor.get("anchor_id") != TRUST_ANCHOR_ID
        or anchor.get("trust_model") != TRUST_MODEL
        or declared_anchor_hash != canonical_sha256(anchor_payload)
    ):
        failures.append(
            {"reason_code": "GEARS_TRUST_ANCHOR_INVALID", "detail": "identity/self-hash"}
        )
    if anchor.get("evidence_manifest_sha256") != file_sha256(manifest_path):
        failures.append(
            {
                "reason_code": "GEARS_TRUSTED_MANIFEST_SHA256_MISMATCH",
                "detail": "evidence manifest differs from detached trust anchor",
            }
        )
    if anchor.get("evidence_manifest_hash") != manifest.get("manifest_hash"):
        failures.append(
            {
                "reason_code": "GEARS_TRUSTED_MANIFEST_HASH_MISMATCH",
                "detail": "manifest_hash differs from detached trust anchor",
            }
        )

    trusted_bindings = anchor.get("trusted_files")
    if not isinstance(trusted_bindings, Mapping) or set(trusted_bindings) != set(BOUND_FILE_FIELDS):
        failures.append(
            {
                "reason_code": "GEARS_TRUSTED_FILE_MANIFEST_INVALID",
                "detail": "exact trusted file set required",
            }
        )
        return bindings
    trusted_paths: dict[str, Path] = {}
    anchor_root = anchor_path.parent
    for field in BOUND_FILE_FIELDS:
        binding = trusted_bindings.get(field)
        if not isinstance(binding, Mapping) or set(binding) != {"source_id", "sha256"}:
            failures.append({"reason_code": "GEARS_TRUSTED_FILE_BINDING_INVALID", "detail": field})
            continue
        source_id = binding.get("source_id")
        expected_hash = binding.get("sha256")
        source_path = Path(str(source_id))
        if (
            not isinstance(source_id, str)
            or not source_id
            or source_path.is_absolute()
            or not isinstance(expected_hash, str)
            or not SHA256_RE.fullmatch(expected_hash)
        ):
            failures.append({"reason_code": "GEARS_TRUSTED_FILE_BINDING_INVALID", "detail": field})
            continue
        resolved = (anchor_root / source_path).resolve()
        if (
            not resolved.is_relative_to(anchor_root)
            or resolved.is_relative_to(evidence_root)
            or not resolved.is_file()
        ):
            failures.append(
                {
                    "reason_code": "GEARS_TRUSTED_FILE_MISSING_OR_UNTRUSTED_LOCATION",
                    "detail": field,
                }
            )
            continue
        if file_sha256(resolved) != expected_hash:
            failures.append({"reason_code": "GEARS_TRUSTED_FILE_SHA256_MISMATCH", "detail": field})
            continue
        trusted_paths[field] = resolved
        package_path = bound_files.get(field)
        if package_path is None or file_sha256(package_path) != expected_hash:
            failures.append(
                {
                    "reason_code": "GEARS_PACKAGE_DIFFERS_FROM_TRUSTED_SOURCE",
                    "detail": field,
                }
            )
    if set(trusted_paths) == set(BOUND_FILE_FIELDS):
        try:
            observed_commitments = _compute_trusted_semantic_commitments(trusted_paths)
        except RevisionProtocolError as error:
            failures.append(
                {
                    "reason_code": "GEARS_TRUSTED_SEMANTIC_SOURCE_INVALID",
                    "detail": str(error),
                }
            )
        else:
            if anchor.get("semantic_commitments") != observed_commitments:
                failures.append(
                    {
                        "reason_code": "GEARS_TRUSTED_SEMANTIC_COMMITMENT_MISMATCH",
                        "detail": "independent environment/plan/dataset/target/graph recomputation",
                    }
                )
    return bindings


def _compute_trusted_semantic_commitments(paths: Mapping[str, Path]) -> dict[str, Any]:
    if set(paths) != set(BOUND_FILE_FIELDS):
        raise RevisionProtocolError("[GEARS_TRUSTED_FILE_MANIFEST_INVALID]")
    try:
        plan = json.loads(paths["execution_plan"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RevisionProtocolError("[GEARS_TRUSTED_EXECUTION_PLAN_INVALID]") from error
    if not isinstance(plan, dict):
        raise RevisionProtocolError("[GEARS_TRUSTED_EXECUTION_PLAN_INVALID]")
    plan_payload = dict(plan)
    declared_plan_hash = plan_payload.pop("plan_hash", None)
    if declared_plan_hash != canonical_sha256(plan_payload):
        raise RevisionProtocolError("[GEARS_TRUSTED_EXECUTION_PLAN_INVALID]")
    execution = plan.get("execution")
    expected_metrics = plan.get("expected_metrics")
    if (
        plan.get("schema_version") != "1.0"
        or not isinstance(execution, Mapping)
        or set(execution) != {"official_reference", "candidate"}
        or not isinstance(expected_metrics, Mapping)
        or not expected_metrics
    ):
        raise RevisionProtocolError("[GEARS_TRUSTED_EXECUTION_PLAN_INVALID]")
    environment_failures: list[dict[str, str]] = []
    environment = _read_complete_environment_lock(
        paths["environment_lock"],
        require_h5py=str(plan.get("dataset", {}).get("source_id", "")).casefold().endswith(".h5ad"),
        failures=environment_failures,
    )
    if environment is None or environment_failures:
        raise RevisionProtocolError(
            "[GEARS_TRUSTED_ENVIRONMENT_LOCK_INVALID] "
            + json.dumps(environment_failures, sort_keys=True)
        )

    from .gears_reproduction import (
        _load_npz_manifest,
        _read_expected_value_provenance,
        _validate_raw_evidence,
    )

    commitments: dict[str, Any] = {
        "environment": {
            "file_sha256": file_sha256(paths["environment_lock"]),
            "structured_sha256": canonical_sha256(environment),
        },
        "execution_plan": {
            "file_sha256": file_sha256(paths["execution_plan"]),
            "plan_hash": declared_plan_hash,
        },
        "dataset": {
            "file_sha256": file_sha256(paths["dataset"]),
            "bytes": paths["dataset"].stat().st_size,
        },
        "official_metric_source": {
            "file_sha256": file_sha256(paths["official_metric_source"]),
            "function": str(plan.get("official_metric_source", {}).get("function", "")),
        },
        "phases": {},
    }
    for phase in ("official_reference", "candidate"):
        specification = execution[phase]
        if not isinstance(specification, Mapping):
            raise RevisionProtocolError(f"[GEARS_TRUSTED_PHASE_SPEC_INVALID] {phase}")
        sidecars = {
            name: _load_npz_manifest(paths[f"{phase}_{name}"], f"trusted.{phase}.{name}")
            for name in ("initialization_state", "checkpoint_state", "graph_state")
        }
        try:
            raw = json.loads(paths[f"{phase}_raw_evidence"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RevisionProtocolError(f"[GEARS_TRUSTED_RAW_EVIDENCE_INVALID] {phase}") from error
        evidence, _ = _validate_raw_evidence(
            raw,
            phase=phase,
            expected_metric_names={str(name) for name in expected_metrics},
            metric_directions=plan.get("metric_directions"),
            expected_split=plan.get("split", {}),
            training_config=plan.get("training_config", {}),
            sidecar_manifests=sidecars,
        )
        provenance = _read_expected_value_provenance(
            paths[f"{phase}_expected_value_provenance"],
            phase=phase,
            specification=specification,
            plan=plan,
            entrypoint_sha256=file_sha256(paths[f"{phase}_entrypoint"]),
            expected_metric_names={str(name) for name in expected_metrics},
        )
        graph_manifest = sidecars["graph_state"]
        commitments["phases"][phase] = {
            "phase_role": specification.get("phase_role"),
            "entrypoint_sha256": file_sha256(paths[f"{phase}_entrypoint"]),
            "expected_value_provenance_sha256": file_sha256(
                paths[f"{phase}_expected_value_provenance"]
            ),
            "expected_value_record_hash": provenance.get("record_hash"),
            "raw_evidence_sha256": file_sha256(paths[f"{phase}_raw_evidence"]),
            "gene_order_hash": canonical_sha256(evidence["gene_names"]),
            "perturbable_gene_order_hash": canonical_sha256(evidence["perturbable_gene_order"]),
            "condition_target_mapping_hash": canonical_sha256(evidence["condition_target_mapping"]),
            "target_audit_hash": canonical_sha256(evidence["target_audit"]),
            "graph_state_sha256": graph_manifest["file_sha256"],
            "graph_array_manifest_hash": canonical_sha256(graph_manifest),
            "graph_semantic_bindings_hash": canonical_sha256(graph_manifest["semantic_bindings"]),
        }
    return commitments


def validate_gears_positive_control(
    manifest_path: Path,
    *,
    trust_anchor_path: Path | None = None,
    expected_trust_anchor_sha256: str | None = None,
) -> dict[str, Any]:
    """Resolve GEARS eligibility while separating invalid evidence from a valid failed run."""

    failures: list[dict[str, str]] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return _registry(None, [{"reason_code": "GEARS_MANIFEST_INVALID", "detail": str(error)}])
    if not isinstance(manifest, dict):
        return _registry(
            None,
            [{"reason_code": "GEARS_MANIFEST_INVALID", "detail": "root must be an object"}],
        )
    if _contains_placeholder(manifest):
        failures.append(
            {
                "reason_code": "GEARS_MANIFEST_PLACEHOLDER_PRESENT",
                "detail": "All REPLACE/TODO/TBD/UNKNOWN fields must be resolved",
            }
        )
    manifest_copy = dict(manifest)
    declared_manifest_hash = manifest_copy.pop("manifest_hash", None)
    if declared_manifest_hash != canonical_sha256(manifest_copy):
        failures.append({"reason_code": "GEARS_MANIFEST_HASH_MISMATCH", "detail": "manifest_hash"})
    if manifest.get("schema_version") != "1.0":
        failures.append({"reason_code": "GEARS_SCHEMA_VERSION_INVALID", "detail": "expected 1.0"})
    repository = manifest.get("official_repository")
    if not isinstance(repository, Mapping):
        failures.append(
            {"reason_code": "GEARS_REPOSITORY_BINDING_MISSING", "detail": "object required"}
        )
        repository = {}
    repository_url = str(repository.get("url", ""))
    commit = str(repository.get("commit", ""))
    if not re.fullmatch(r"https://github\.com/[^/]+/[^/]+/?", repository_url):
        failures.append(
            {"reason_code": "GEARS_OFFICIAL_REPOSITORY_URL_INVALID", "detail": repository_url}
        )
    if not COMMIT_RE.fullmatch(commit):
        failures.append(
            {"reason_code": "GEARS_COMMIT_INVALID", "detail": "40 lowercase hex required"}
        )
    root = manifest_path.resolve().parent
    bound_files: dict[str, Path] = {}
    for field in BOUND_FILE_FIELDS:
        binding = manifest.get(field)
        if not isinstance(binding, Mapping):
            failures.append({"reason_code": "GEARS_FILE_BINDING_MISSING", "detail": field})
            continue
        path = _resolve_bound_file(root, binding, field, failures)
        if path is not None:
            bound_files[field] = path
    trust_bindings = _validate_detached_trust_anchor(
        manifest_path=manifest_path,
        manifest=manifest,
        bound_files=bound_files,
        trust_anchor_path=trust_anchor_path,
        expected_trust_anchor_sha256=expected_trust_anchor_sha256,
        failures=failures,
    )
    metric_binding = manifest.get("official_metric_source", {})
    if not str(metric_binding.get("function", "")).strip():
        failures.append({"reason_code": "GEARS_METRIC_FUNCTION_MISSING", "detail": "function"})
    locked_environment = _read_complete_environment_lock(
        bound_files.get("environment_lock"),
        require_h5py=str(manifest.get("dataset", {}).get("source_id", ""))
        .casefold()
        .endswith(".h5ad"),
        failures=failures,
    )
    split = manifest.get("split")
    if (
        not isinstance(split, Mapping)
        or not str(split.get("name", "")).strip()
        or not isinstance(split.get("seed"), int)
        or isinstance(split.get("seed"), bool)
    ):
        failures.append(
            {"reason_code": "GEARS_SPLIT_CONTRACT_INVALID", "detail": "name and integer seed"}
        )
        split = {}
    training_config = manifest.get("training_config")
    if not isinstance(training_config, Mapping) or not training_config:
        failures.append(
            {"reason_code": "GEARS_TRAINING_CONFIG_MISSING", "detail": "full mapping required"}
        )
        training_config = {}
    expected_metrics = manifest.get("expected_metrics")
    tolerances = manifest.get("tolerances")
    if not isinstance(expected_metrics, Mapping) or not expected_metrics:
        failures.append(
            {"reason_code": "GEARS_EXPECTED_METRICS_MISSING", "detail": "mapping required"}
        )
        expected_metrics = {}
    if not isinstance(tolerances, Mapping) or set(tolerances) != set(expected_metrics):
        failures.append(
            {"reason_code": "GEARS_TOLERANCE_FAMILY_MISMATCH", "detail": "one per metric"}
        )
        tolerances = {}
    parsed_expected: dict[str, float] = {}
    parsed_tolerances: dict[str, tuple[float, float, str]] = {}
    for metric, value in expected_metrics.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = math.nan
        if not math.isfinite(numeric):
            failures.append({"reason_code": "GEARS_EXPECTED_METRIC_INVALID", "detail": str(metric)})
        parsed_expected[str(metric)] = numeric
        tolerance = tolerances.get(metric, {}) if isinstance(tolerances, Mapping) else {}
        try:
            absolute = float(tolerance.get("absolute"))
            relative = float(tolerance.get("relative"))
        except (TypeError, ValueError):
            absolute = math.nan
            relative = math.nan
        justification = str(tolerance.get("justification", "")).strip()
        if (
            not math.isfinite(absolute)
            or not math.isfinite(relative)
            or absolute < 0
            or relative < 0
            or not justification
        ):
            failures.append({"reason_code": "GEARS_TOLERANCE_INVALID", "detail": str(metric)})
        parsed_tolerances[str(metric)] = (absolute, relative, justification)
    metric_directions = manifest.get("metric_directions")
    if (
        not isinstance(metric_directions, Mapping)
        or set(metric_directions) != set(parsed_expected)
        or any(
            value not in {"higher_is_better", "lower_is_better"}
            for value in metric_directions.values()
        )
    ):
        failures.append(
            {"reason_code": "GEARS_METRIC_DIRECTION_CONTRACT_INVALID", "detail": "mapping"}
        )
    reference_metrics: dict[str, float] = {}
    candidate_metrics: dict[str, float] = {}
    common_binding = {
        "repository_url": repository_url,
        "commit": commit,
        "dataset_sha256": str(manifest.get("dataset", {}).get("sha256", "")),
        "split_name": str(split.get("name", "")),
        "split_seed": split.get("seed"),
        "training_config_hash": canonical_sha256(training_config),
        "metric_source_sha256": str(metric_binding.get("sha256", "")),
        "metric_function": str(metric_binding.get("function", "")),
    }
    packaged_plan = _validate_packaged_execution_plan(
        bound_files.get("execution_plan"), manifest, failures
    )
    phase_contracts = _validate_packaged_phase_contracts(
        packaged_plan,
        bound_files=bound_files,
        expected_metric_names=set(parsed_expected),
        failures=failures,
    )
    audit_hashes: dict[str, str] = {}
    for field, phase in (
        ("official_reference_execution_audit", "official_reference"),
        ("candidate_execution_audit", "candidate"),
    ):
        audit_path = bound_files.get(field)
        if audit_path is None:
            continue
        if _validate_execution_audit(
            audit_path,
            phase=phase,
            common_binding=common_binding,
            environment_lock_sha256=str(manifest.get("environment_lock", {}).get("sha256", "")),
            raw_evidence_sha256=str(manifest.get(f"{phase}_raw_evidence", {}).get("sha256", "")),
            locked_environment=locked_environment,
            phase_contract=phase_contracts.get(phase),
            failures=failures,
        ):
            audit_hashes[phase] = file_sha256(audit_path)
    semantic_metrics: dict[str, dict[str, float]] = {}
    semantic_diagnostics: dict[str, dict[str, Any]] = {}
    for phase in ("official_reference", "candidate"):
        diagnostics = _validate_semantic_evidence(
            phase=phase,
            bound_files=bound_files,
            expected_metric_names=set(parsed_expected),
            metric_directions=metric_directions,
            split=split,
            training_config=training_config,
            packaged_plan=packaged_plan,
            failures=failures,
        )
        if diagnostics is not None:
            semantic_diagnostics[phase] = diagnostics
            semantic_metrics[phase] = diagnostics["metrics"]
    for field, destination in (
        ("official_reference_output", reference_metrics),
        ("candidate_output", candidate_metrics),
    ):
        if field not in bound_files:
            continue
        output = _read_output(bound_files[field], field, failures)
        if output is None:
            continue
        metadata = output.get("metadata")
        phase = "official_reference" if field == "official_reference_output" else "candidate"
        if (
            not isinstance(metadata, Mapping)
            or any(metadata.get(key) != value for key, value in common_binding.items())
            or metadata.get("execution_phase") != phase
            or metadata.get("phase_role")
            != phase_contracts.get(phase, {}).get("record", {}).get("phase_role")
            or metadata.get("phase_task_hash")
            != canonical_sha256(phase_contracts.get(phase, {}).get("record", {}))
            or metadata.get("execution_audit_sha256") != audit_hashes.get(phase)
        ):
            failures.append({"reason_code": "GEARS_OUTPUT_PROVENANCE_MISMATCH", "detail": field})
        metrics = output.get("metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != set(parsed_expected):
            failures.append({"reason_code": "GEARS_OUTPUT_METRIC_SET_MISMATCH", "detail": field})
            continue
        for metric, value in metrics.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = math.nan
            if not math.isfinite(numeric):
                failures.append(
                    {"reason_code": "GEARS_OUTPUT_METRIC_INVALID", "detail": f"{field}:{metric}"}
                )
            destination[str(metric)] = numeric
        semantic = semantic_metrics.get(phase)
        if semantic is not None and destination != semantic:
            failures.append(
                {"reason_code": "GEARS_OUTPUT_SEMANTIC_METRIC_MISMATCH", "detail": field}
            )
    comparisons: list[dict[str, Any]] = []
    outcome_reason_codes: list[str] = []
    for metric, expected in sorted(parsed_expected.items()):
        reference = reference_metrics.get(metric, math.nan)
        observed = candidate_metrics.get(metric, math.nan)
        if math.isfinite(reference) and not math.isclose(
            reference, expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            failures.append(
                {"reason_code": "GEARS_REFERENCE_EXPECTATION_MISMATCH", "detail": metric}
            )
        absolute, relative, justification = parsed_tolerances.get(metric, (math.nan, math.nan, ""))
        allowed = absolute + relative * abs(expected)
        difference = abs(observed - expected) if math.isfinite(observed) else math.inf
        passed = math.isfinite(allowed) and difference <= allowed
        if not passed and math.isfinite(observed) and math.isfinite(allowed):
            outcome_reason_codes.append(f"GEARS_METRIC_OUTSIDE_TOLERANCE:{metric}")
        comparisons.append(
            {
                "metric": metric,
                "expected": _finite_or_none(expected),
                "official_reference": _finite_or_none(reference),
                "candidate_observed": _finite_or_none(observed),
                "absolute_difference": _finite_or_none(difference),
                "allowed_difference": _finite_or_none(allowed),
                "absolute_tolerance": _finite_or_none(absolute),
                "relative_tolerance": _finite_or_none(relative),
                "tolerance_justification": justification,
                "status": "PASS" if passed else "FAIL",
            }
        )
    return _registry(
        declared_manifest_hash,
        failures,
        comparisons=comparisons,
        bindings={**common_binding, **trust_bindings},
        semantic_diagnostics=semantic_diagnostics,
        outcome_reason_codes=outcome_reason_codes,
    )


def _validate_packaged_execution_plan(
    path: Path | None,
    manifest: Mapping[str, Any],
    failures: list[dict[str, str]],
) -> dict[str, Any] | None:
    if path is None:
        return None
    plan = _read_output(path, "execution_plan", failures)
    if plan is None:
        return None
    payload = dict(plan)
    declared_hash = payload.pop("plan_hash", None)
    required = {
        "schema_version",
        "official_repository",
        "environment_lock",
        "dataset",
        "split",
        "training_config",
        "official_metric_source",
        "expected_metrics",
        "tolerances",
        "metric_directions",
        "execution",
        "plan_hash",
    }
    valid = (
        set(plan) == required
        and plan.get("schema_version") == "1.0"
        and declared_hash == canonical_sha256(payload)
        and declared_hash == manifest.get("execution_plan_hash")
        and plan.get("official_repository") == manifest.get("official_repository")
        and plan.get("split") == manifest.get("split")
        and plan.get("training_config") == manifest.get("training_config")
        and plan.get("expected_metrics") == manifest.get("expected_metrics")
        and plan.get("tolerances") == manifest.get("tolerances")
        and plan.get("metric_directions") == manifest.get("metric_directions")
        and isinstance(plan.get("execution"), Mapping)
        and set(plan.get("execution", {})) == {"official_reference", "candidate"}
    )
    for source_field in ("environment_lock", "dataset", "official_metric_source"):
        plan_binding = plan.get(source_field)
        manifest_binding = manifest.get(source_field)
        if (
            not isinstance(plan_binding, Mapping)
            or not isinstance(manifest_binding, Mapping)
            or plan_binding.get("sha256") != manifest_binding.get("sha256")
        ):
            valid = False
    if not valid:
        failures.append({"reason_code": "GEARS_EXECUTION_PLAN_BINDING_MISMATCH", "detail": "plan"})
        return None
    return plan


def _validate_packaged_phase_contracts(
    plan: Mapping[str, Any] | None,
    *,
    bound_files: Mapping[str, Path],
    expected_metric_names: set[str],
    failures: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    if plan is None:
        return {}
    try:
        from .gears_reproduction import (
            DISTINCT_WORKFLOW_POLICY,
            PHASE_ROLES,
            PINNED_WORKFLOW_REUSE_POLICY,
            _read_expected_value_provenance,
        )

        execution = plan.get("execution")
        if not isinstance(execution, Mapping) or set(execution) != {
            "official_reference",
            "candidate",
        }:
            raise RevisionProtocolError("[GEARS_EXECUTION_SPEC_INVALID]")
        contracts: dict[str, dict[str, Any]] = {}
        for phase in ("official_reference", "candidate"):
            specification = execution.get(phase)
            if not isinstance(specification, Mapping) or set(specification) != {
                "argv",
                "entrypoint",
                "phase_role",
                "expected_value_provenance",
                "workflow_reuse_policy",
            }:
                raise RevisionProtocolError(f"[GEARS_EXECUTION_SPEC_INVALID] {phase}")
            if specification.get("phase_role") != PHASE_ROLES[phase] or specification.get(
                "workflow_reuse_policy"
            ) not in {PINNED_WORKFLOW_REUSE_POLICY, DISTINCT_WORKFLOW_POLICY}:
                raise RevisionProtocolError(f"[GEARS_EXECUTION_PHASE_ROLE_INVALID] {phase}")
            entrypoint = specification.get("entrypoint")
            provenance = specification.get("expected_value_provenance")
            argv = specification.get("argv")
            entrypoint_path = bound_files.get(f"{phase}_entrypoint")
            provenance_path = bound_files.get(f"{phase}_expected_value_provenance")
            if (
                not isinstance(entrypoint, Mapping)
                or set(entrypoint) != {"scope", "source_id", "sha256"}
                or not isinstance(provenance, Mapping)
                or set(provenance) != {"source_id", "sha256"}
                or not isinstance(argv, list)
                or len(argv) < 2
                or entrypoint_path is None
                or provenance_path is None
                or entrypoint.get("sha256") != file_sha256(entrypoint_path)
                or provenance.get("sha256") != file_sha256(provenance_path)
            ):
                raise RevisionProtocolError(f"[GEARS_PHASE_TASK_BINDING_INVALID] {phase}")
            record = _read_expected_value_provenance(
                provenance_path,
                phase=phase,
                specification=specification,
                plan=plan,
                entrypoint_sha256=file_sha256(entrypoint_path),
                expected_metric_names=expected_metric_names,
            )
            contracts[phase] = {
                "record": record,
                "source_sha256": file_sha256(provenance_path),
            }
        reference = execution["official_reference"]
        candidate = execution["candidate"]
        same_workflow = (
            reference["entrypoint"]["sha256"] == candidate["entrypoint"]["sha256"]
            and reference["argv"] == candidate["argv"]
        )
        if same_workflow and any(
            specification.get("workflow_reuse_policy") != PINNED_WORKFLOW_REUSE_POLICY
            for specification in (reference, candidate)
        ):
            raise RevisionProtocolError("[GEARS_IDENTICAL_PHASE_WORKFLOW_UNDECLARED]")
        if (
            contracts["official_reference"]["source_sha256"]
            == contracts["candidate"]["source_sha256"]
            or contracts["official_reference"]["record"]["source_id"]
            == contracts["candidate"]["record"]["source_id"]
        ):
            raise RevisionProtocolError("[GEARS_PHASE_PROVENANCE_NOT_INDEPENDENTLY_BOUND]")
        return contracts
    except RevisionProtocolError as error:
        failures.append(
            {
                "reason_code": "GEARS_PHASE_CONTRACT_INVALID",
                "detail": str(error),
            }
        )
        return {}


def _validate_semantic_evidence(
    *,
    phase: str,
    bound_files: Mapping[str, Path],
    expected_metric_names: set[str],
    metric_directions: Any,
    split: Mapping[str, Any],
    training_config: Mapping[str, Any],
    packaged_plan: Mapping[str, Any] | None,
    failures: list[dict[str, str]],
) -> dict[str, Any] | None:
    required_fields = {
        "raw": f"{phase}_raw_evidence",
        "audit": f"{phase}_execution_audit",
        "entrypoint": f"{phase}_entrypoint",
        "initialization_state": f"{phase}_initialization_state",
        "checkpoint_state": f"{phase}_checkpoint_state",
        "graph_state": f"{phase}_graph_state",
        "metric_source": "official_metric_source",
    }
    if any(field not in bound_files for field in required_fields.values()):
        return None
    try:
        from .gears_reproduction import (
            _load_metric_callable,
            _load_npz_manifest,
            _numeric_metrics,
            _validate_raw_evidence,
        )

        raw = json.loads(bound_files[required_fields["raw"]].read_text(encoding="utf-8"))
        sidecar_manifests = {
            name: _load_npz_manifest(
                bound_files[required_fields[name]], f"validation.{phase}.{name}"
            )
            for name in ("initialization_state", "checkpoint_state", "graph_state")
        }
        evidence, results = _validate_raw_evidence(
            raw,
            phase=phase,
            expected_metric_names=expected_metric_names,
            metric_directions=metric_directions,
            expected_split=split,
            training_config=training_config,
            sidecar_manifests=sidecar_manifests,
        )
        if packaged_plan is None:
            raise RevisionProtocolError("[GEARS_EXECUTION_PLAN_BINDING_MISMATCH]")
        phase_specification = packaged_plan["execution"].get(phase)
        if not isinstance(phase_specification, Mapping):
            raise RevisionProtocolError("[GEARS_EXECUTION_SPEC_INVALID]")
        entrypoint_specification = phase_specification.get("entrypoint")
        if not isinstance(entrypoint_specification, Mapping) or entrypoint_specification.get(
            "sha256"
        ) != file_sha256(bound_files[required_fields["entrypoint"]]):
            raise RevisionProtocolError("[GEARS_EXECUTION_ENTRYPOINT_BINDING_MISMATCH]")
        audit = json.loads(bound_files[required_fields["audit"]].read_text(encoding="utf-8"))
        if not isinstance(audit, Mapping):
            raise RevisionProtocolError("[GEARS_EXECUTION_AUDIT_INVALID]")
        expected_hashes = {
            "prediction_vector_hash": canonical_sha256(evidence["results"]["pred"]),
            "truth_vector_hash": canonical_sha256(evidence["results"]["truth"]),
            "gene_order_hash": canonical_sha256(evidence["gene_names"]),
            "perturbable_gene_order_hash": canonical_sha256(evidence["perturbable_gene_order"]),
            "training_loss_hash": canonical_sha256(evidence["training_loss"]),
            "validation_loss_hash": canonical_sha256(evidence["validation_loss"]),
            "target_audit_hash": canonical_sha256(evidence["target_audit"]),
            "split_membership_hash": canonical_sha256(evidence["split_membership"]),
            "condition_target_mapping_hash": canonical_sha256(evidence["condition_target_mapping"]),
            "baseline_prediction_vector_hash": canonical_sha256(
                evidence["baselines"]["control_mean_zero_change"]["results"]["pred"]
            ),
            "baseline_de_prediction_vector_hash": canonical_sha256(
                evidence["baselines"]["control_mean_zero_change"]["results"]["pred_de"]
            ),
            "entrypoint_sha256": file_sha256(bound_files[required_fields["entrypoint"]]),
            "initialization_state_sha256": sidecar_manifests["initialization_state"]["file_sha256"],
            "checkpoint_state_sha256": sidecar_manifests["checkpoint_state"]["file_sha256"],
            "graph_state_sha256": sidecar_manifests["graph_state"]["file_sha256"],
            "initialization_array_manifest_hash": canonical_sha256(
                sidecar_manifests["initialization_state"]
            ),
            "checkpoint_array_manifest_hash": canonical_sha256(
                sidecar_manifests["checkpoint_state"]
            ),
            "graph_array_manifest_hash": canonical_sha256(sidecar_manifests["graph_state"]),
        }
        if audit.get("normalized_command_argv") != phase_specification.get("argv") or any(
            audit.get(field) != expected for field, expected in expected_hashes.items()
        ):
            raise RevisionProtocolError("[GEARS_SEMANTIC_AUDIT_RECOMPUTATION_MISMATCH]")
        metric_binding = packaged_plan["official_metric_source"]
        metric_callable = _load_metric_callable(
            bound_files[required_fields["metric_source"]], str(metric_binding["function"])
        )
        computed = metric_callable(results)
        if isinstance(computed, tuple):
            computed = computed[0]
        metrics = _numeric_metrics(computed, f"validation.{phase}.metrics")
        reported = _numeric_metrics(
            evidence["reported_metrics"], f"validation.{phase}.reported_metrics"
        )
        if metrics != reported or set(metrics) != expected_metric_names:
            raise RevisionProtocolError("[GEARS_REPORTED_METRIC_RECOMPUTATION_MISMATCH]")
        baseline_payload = evidence["baselines"]["control_mean_zero_change"]
        baseline_results = {
            name: (
                np.asarray(value, dtype=str)
                if name == "pert_cat"
                else np.asarray(value, dtype=float)
            )
            for name, value in baseline_payload["results"].items()
        }
        baseline_computed = metric_callable(baseline_results)
        if isinstance(baseline_computed, tuple):
            baseline_computed = baseline_computed[0]
        baseline_metrics = _numeric_metrics(
            baseline_computed, f"validation.{phase}.baseline_metrics"
        )
        baseline_reported = _numeric_metrics(
            baseline_payload["reported_metrics"],
            f"validation.{phase}.baseline_reported_metrics",
        )
        if (
            baseline_metrics != baseline_reported
            or audit.get("baseline_metrics") != baseline_metrics
        ):
            raise RevisionProtocolError("[GEARS_BASELINE_METRIC_RECOMPUTATION_MISMATCH]")
        directional_improvements: dict[str, float] = {}
        for metric in sorted(metrics):
            direction = metric_directions[metric]
            if direction == "higher_is_better":
                improvement = metrics[metric] - baseline_metrics[metric]
            elif direction == "lower_is_better":
                improvement = baseline_metrics[metric] - metrics[metric]
            else:
                raise RevisionProtocolError("[GEARS_METRIC_DIRECTION_CONTRACT_INVALID]")
            if not math.isfinite(improvement) or improvement <= 0:
                raise RevisionProtocolError(
                    f"[GEARS_BASELINE_POSITIVE_CONTROL_FAILED] {phase}:{metric}"
                )
            directional_improvements[metric] = float(improvement)
        training_losses = np.asarray(evidence["training_loss"], dtype=float)
        validation_losses = np.asarray(evidence["validation_loss"], dtype=float)
        training_improvement = float(training_losses[0] - training_losses[-1])
        validation_improvement = float(validation_losses[0] - validation_losses[-1])
        if training_improvement <= 0 or validation_improvement <= 0:
            raise RevisionProtocolError(f"[GEARS_LOSS_DID_NOT_IMPROVE] {phase}")
        prediction_standard_deviation = float(np.std(results["pred"]))
        truth_standard_deviation = float(np.std(results["truth"]))
        if prediction_standard_deviation <= 0 or truth_standard_deviation <= 0:
            raise RevisionProtocolError(f"[GEARS_RAW_VECTOR_DEGENERATE_OR_INVALID] {phase}")
        return {
            "metrics": metrics,
            "baseline_metrics": baseline_metrics,
            "directional_baseline_improvements": directional_improvements,
            "minimum_directional_baseline_improvement": min(directional_improvements.values()),
            "training_loss_improvement": training_improvement,
            "validation_loss_improvement": validation_improvement,
            "prediction_standard_deviation": prediction_standard_deviation,
            "truth_standard_deviation": truth_standard_deviation,
            "loss_history_length": len(training_losses),
            "gene_order_sha256": canonical_sha256(evidence["gene_names"]),
            "perturbable_gene_order_sha256": canonical_sha256(evidence["perturbable_gene_order"]),
            "condition_target_mapping_sha256": canonical_sha256(
                evidence["condition_target_mapping"]
            ),
            "target_audit_sha256": canonical_sha256(evidence["target_audit"]),
            "split_membership_sha256": canonical_sha256(evidence["split_membership"]),
            "vector_row_conditions_sha256": canonical_sha256(
                evidence["split_membership"]["condition_ids"]
            ),
            "target_disjoint_status": ("NOT_APPLICABLE_TEST_ONLY_POSITIVE_CONTROL_SPLIT"),
            "target_disjoint_evidence_sha256": canonical_sha256(
                {
                    "split_membership": evidence["split_membership"],
                    "condition_target_mapping": evidence["condition_target_mapping"],
                }
            ),
            "diagnostic_status": "PASS",
        }
    except (OSError, json.JSONDecodeError, RevisionProtocolError, KeyError, TypeError) as error:
        detail = str(error) or type(error).__name__
        failures.append(
            {"reason_code": "GEARS_SEMANTIC_EVIDENCE_INVALID", "detail": f"{phase}:{detail}"}
        )
        return None


def _resolve_bound_file(
    root: Path,
    binding: Mapping[str, Any],
    field: str,
    failures: list[dict[str, str]],
) -> Path | None:
    source_id = str(binding.get("source_id", "")).strip()
    expected_hash = str(binding.get("sha256", ""))
    path = Path(source_id)
    if not source_id or path.is_absolute() or not SHA256_RE.fullmatch(expected_hash):
        failures.append({"reason_code": "GEARS_FILE_BINDING_INVALID", "detail": field})
        return None
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        failures.append(
            {"reason_code": "GEARS_BOUND_FILE_MISSING_OR_ESCAPES_ROOT", "detail": field}
        )
        return None
    if file_sha256(resolved) != expected_hash:
        failures.append({"reason_code": "GEARS_BOUND_FILE_HASH_MISMATCH", "detail": field})
        return None
    return resolved


def _read_complete_environment_lock(
    path: Path | None,
    *,
    require_h5py: bool,
    failures: list[dict[str, str]],
) -> dict[str, Any] | None:
    if path is None:
        return None
    packages: dict[str, str] = {}
    headers: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        failures.append({"reason_code": "GEARS_ENVIRONMENT_LOCK_INVALID", "detail": str(error)})
        return None
    if PLACEHOLDER_RE.search(text):
        failures.append({"reason_code": "GEARS_ENVIRONMENT_LOCK_INVALID", "detail": "placeholder"})
        return None
    lines = text.splitlines()
    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            match = re.fullmatch(r"#\s*([a-z0-9_-]+):\s*(\S+)", line, re.IGNORECASE)
            if match is not None:
                name = match.group(1).casefold()
                if name in headers:
                    failures.append(
                        {
                            "reason_code": "GEARS_ENVIRONMENT_LOCK_DUPLICATE_HEADER",
                            "detail": name,
                        }
                    )
                    return None
                headers[name] = match.group(2)
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if match is None:
            failures.append(
                {
                    "reason_code": "GEARS_ENVIRONMENT_LOCK_INVALID",
                    "detail": f"line={line_number}",
                }
            )
            return None
        name = re.sub(r"[-_.]+", "-", match.group(1)).casefold()
        if name in packages:
            failures.append(
                {"reason_code": "GEARS_ENVIRONMENT_LOCK_DUPLICATE_PACKAGE", "detail": name}
            )
            return None
        packages[name] = match.group(2)
    if not packages or set(headers) != set(ENVIRONMENT_LOCK_HEADER_ORDER):
        failures.append(
            {"reason_code": "GEARS_ENVIRONMENT_LOCK_INCOMPLETE", "detail": "headers/package set"}
        )
        return None
    packages = dict(sorted(packages.items()))
    direct_dependencies = _direct_runtime_dependencies()
    pyg_extensions = {name: packages.get(name, "NOT_INSTALLED") for name in PYG_EXTENSION_PACKAGES}
    derived_headers = {
        "lock-format": ENVIRONMENT_LOCK_FORMAT,
        "pytorch-version": packages.get("torch", "NOT_INSTALLED"),
        "pytorch-geometric-version": packages.get("torch-geometric", "NOT_INSTALLED"),
        "complete-distribution-set-sha256": canonical_sha256(packages),
        "direct-runtime-dependencies-sha256": canonical_sha256(direct_dependencies),
        "pyg-extension-versions-sha256": canonical_sha256(pyg_extensions),
    }
    if any(headers.get(name) != value for name, value in derived_headers.items()):
        failures.append(
            {
                "reason_code": "GEARS_ENVIRONMENT_LOCK_RUNTIME_HEADER_MISMATCH",
                "detail": "package-derived header",
            }
        )
        return None
    missing_direct = sorted(set(direct_dependencies) - set(packages))
    if require_h5py and ("h5py" not in packages or missing_direct):
        failures.append(
            {
                "reason_code": "GEARS_ENVIRONMENT_LOCK_INCOMPLETE",
                "detail": f"missing direct dependencies: {missing_direct}",
            }
        )
        return None
    return {"packages": packages, "headers": headers}


def _read_output(path: Path, field: str, failures: list[dict[str, str]]) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        failures.append({"reason_code": "GEARS_OUTPUT_INVALID", "detail": f"{field}:{error}"})
        return None
    if not isinstance(payload, dict):
        failures.append(
            {"reason_code": "GEARS_OUTPUT_INVALID", "detail": f"{field}:object required"}
        )
        return None
    return payload


def _validate_execution_audit(
    path: Path,
    *,
    phase: str,
    common_binding: Mapping[str, Any],
    environment_lock_sha256: str,
    raw_evidence_sha256: str,
    locked_environment: Mapping[str, Any] | None,
    phase_contract: Mapping[str, Any] | None,
    failures: list[dict[str, str]],
) -> bool:
    audit = _read_output(path, f"{phase}_execution_audit", failures)
    if audit is None:
        return False
    payload = dict(audit)
    declared_hash = payload.pop("audit_hash", None)
    required = {
        "schema_version",
        "phase",
        "phase_role",
        "phase_task_hash",
        "expected_value_provenance_sha256",
        "expected_value_record_hash",
        "repository_url",
        "commit",
        "normalized_command_argv",
        "returncode",
        "wall_seconds",
        "stdout_sha256",
        "stderr_sha256",
        "raw_metrics_sha256",
        "entrypoint_sha256",
        "official_metric_recomputed",
        "prediction_vector_hash",
        "truth_vector_hash",
        "gene_order_hash",
        "perturbable_gene_order_hash",
        "training_loss_hash",
        "validation_loss_hash",
        "target_audit_hash",
        "split_membership_hash",
        "condition_target_mapping_hash",
        "baseline_prediction_vector_hash",
        "baseline_de_prediction_vector_hash",
        "baseline_metrics",
        "initialization_state_sha256",
        "checkpoint_state_sha256",
        "graph_state_sha256",
        "initialization_array_manifest_hash",
        "checkpoint_array_manifest_hash",
        "graph_array_manifest_hash",
        "environment_lock_sha256",
        "dataset_sha256",
        "metric_source_sha256",
        "split_name",
        "split_seed",
        "training_config_hash",
        "installed_locked_versions",
        "runtime_environment_headers",
        "audit_hash",
    }
    if set(audit) != required or declared_hash != canonical_sha256(payload):
        failures.append({"reason_code": "GEARS_EXECUTION_AUDIT_INVALID", "detail": phase})
        return False
    expected = {
        "repository_url": common_binding["repository_url"],
        "commit": common_binding["commit"],
        "dataset_sha256": common_binding["dataset_sha256"],
        "metric_source_sha256": common_binding["metric_source_sha256"],
        "split_name": common_binding["split_name"],
        "split_seed": common_binding["split_seed"],
        "training_config_hash": common_binding["training_config_hash"],
        "environment_lock_sha256": environment_lock_sha256,
    }
    try:
        wall_seconds = float(audit.get("wall_seconds"))
    except (TypeError, ValueError):
        wall_seconds = math.nan
    valid = (
        audit.get("schema_version") == "2.0"
        and audit.get("phase") == phase
        and phase_contract is not None
        and audit.get("phase_role") == phase_contract.get("record", {}).get("phase_role")
        and audit.get("phase_task_hash") == canonical_sha256(phase_contract.get("record", {}))
        and audit.get("expected_value_provenance_sha256") == phase_contract.get("source_sha256")
        and audit.get("expected_value_record_hash")
        == phase_contract.get("record", {}).get("record_hash")
        and audit.get("returncode") == 0
        and audit.get("official_metric_recomputed") is True
        and audit.get("raw_metrics_sha256") == raw_evidence_sha256
        and math.isfinite(wall_seconds)
        and wall_seconds >= 0
        and isinstance(audit.get("normalized_command_argv"), list)
        and bool(audit.get("normalized_command_argv"))
        and isinstance(audit.get("installed_locked_versions"), Mapping)
        and bool(audit.get("installed_locked_versions"))
        and isinstance(audit.get("runtime_environment_headers"), Mapping)
        and set(audit.get("runtime_environment_headers", {})) == set(ENVIRONMENT_LOCK_HEADER_ORDER)
        and locked_environment is not None
        and audit.get("installed_locked_versions") == locked_environment.get("packages")
        and audit.get("runtime_environment_headers") == locked_environment.get("headers")
        and all(audit.get(key) == value for key, value in expected.items())
        and all(
            SHA256_RE.fullmatch(str(audit.get(field, "")))
            for field in (
                "stdout_sha256",
                "stderr_sha256",
                "raw_metrics_sha256",
                "entrypoint_sha256",
                "prediction_vector_hash",
                "truth_vector_hash",
                "gene_order_hash",
                "perturbable_gene_order_hash",
                "training_loss_hash",
                "validation_loss_hash",
                "target_audit_hash",
                "split_membership_hash",
                "condition_target_mapping_hash",
                "baseline_prediction_vector_hash",
                "baseline_de_prediction_vector_hash",
                "initialization_state_sha256",
                "checkpoint_state_sha256",
                "graph_state_sha256",
                "initialization_array_manifest_hash",
                "checkpoint_array_manifest_hash",
                "graph_array_manifest_hash",
                "phase_task_hash",
                "expected_value_provenance_sha256",
                "expected_value_record_hash",
            )
        )
    )
    if not valid:
        failures.append({"reason_code": "GEARS_EXECUTION_AUDIT_CONTRACT_MISMATCH", "detail": phase})
    return valid


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return bool(PLACEHOLDER_RE.search(value))
    if isinstance(value, Mapping):
        return any(
            _contains_placeholder(key) or _contains_placeholder(item) for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_placeholder(item) for item in value)
    return False


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _registry(
    manifest_hash: str | None,
    failures: Sequence[Mapping[str, str]],
    *,
    comparisons: Sequence[Mapping[str, Any]] = (),
    bindings: Mapping[str, Any] | None = None,
    semantic_diagnostics: Mapping[str, Mapping[str, Any]] | None = None,
    outcome_reason_codes: Sequence[str] = (),
) -> dict[str, Any]:
    unique_failures = [dict(item) for item in {tuple(sorted(item.items())) for item in failures}]
    unique_failures.sort(key=lambda item: (item.get("reason_code", ""), item.get("detail", "")))
    evidence_valid = not unique_failures
    positive_control_passed = (
        evidence_valid
        and bool(comparisons)
        and all(comparison.get("status") == "PASS" for comparison in comparisons)
    )
    if not evidence_valid:
        eligibility_decision = "WITHHELD_INVALID_OR_INCOMPLETE_EVIDENCE"
        execution_outcome = "NOT_INTERPRETABLE"
    elif positive_control_passed:
        eligibility_decision = "ELIGIBLE_POSITIVE_CONTROL_PASSED"
        execution_outcome = "PASS"
    else:
        eligibility_decision = "EXCLUDED_FAILED_POSITIVE_CONTROL"
        execution_outcome = "FAIL"
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "registry_id": "GEARS-POSITIVE-CONTROL-VALIDATION",
        "status": "RELEASED" if evidence_valid else "WITHHELD",
        "evidence_status": "VALID" if evidence_valid else "INVALID_OR_INCOMPLETE",
        "execution_outcome": execution_outcome,
        "eligibility_decision": eligibility_decision,
        "ranking_eligibility": eligibility_decision == "ELIGIBLE_POSITIVE_CONTROL_PASSED",
        "manifest_hash": manifest_hash,
        "comparisons": list(comparisons),
        "provenance_bindings": dict(bindings or {}),
        "semantic_diagnostics": {
            str(phase): dict(values)
            for phase, values in sorted((semantic_diagnostics or {}).items())
        },
        "outcome_reason_codes": sorted(set(str(value) for value in outcome_reason_codes)),
        "failures": unique_failures,
    }
    payload["registry_hash"] = canonical_sha256(payload)
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trust-anchor", type=Path, required=True)
    parser.add_argument("--expected-trust-anchor-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_argument_parser().parse_args(arguments)
    registry = validate_gears_positive_control(
        parsed.manifest,
        trust_anchor_path=parsed.trust_anchor,
        expected_trust_anchor_sha256=parsed.expected_trust_anchor_sha256,
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(
        json.dumps(registry, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(registry, indent=2, sort_keys=True))
    return 0 if registry["status"] == "RELEASED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

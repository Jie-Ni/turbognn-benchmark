"""Execute and package a hash-bound GEARS positive-control reproduction."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import canonical_sha256, file_sha256
from .errors import RevisionProtocolError
from .gears_validation import create_gears_trust_anchor, validate_gears_positive_control
from .runner import (
    _direct_runtime_dependencies,
    _environment_lock_headers,
)

OFFICIAL_REPOSITORY = "https://github.com/snap-stanford/GEARS"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LOCK_LINE_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)$")
PLACEHOLDERS = {
    "{python}",
    "{repository_root}",
    "{dataset}",
    "{output}",
    "{entrypoint}",
}
PHASE_ROLES = {
    "official_reference": "official_reference",
    "candidate": "heldout_adaptation",
}
PINNED_WORKFLOW_REUSE_POLICY = "PINNED_OFFICIAL_WORKFLOW_REUSE_ALLOWED"
DISTINCT_WORKFLOW_POLICY = "DISTINCT_PHASE_WORKFLOW_REQUIRED"


def execute_gears_reproduction(
    plan_path: Path,
    repository_root: Path,
    output_dir: Path,
    *,
    python_executable: Path = Path(sys.executable),
    trust_anchor_output_path: Path | None = None,
) -> dict[str, Any]:
    """Run both declared phases and emit a self-contained validation package."""

    plan = _read_plan(plan_path)
    plan_root = plan_path.resolve().parent
    repository_root = repository_root.resolve()
    output_dir = output_dir.resolve()
    if trust_anchor_output_path is None:
        trust_anchor_output_path = output_dir.parent / f"{output_dir.name}.gears_trust_anchor.json"
    trust_anchor_output_path = trust_anchor_output_path.resolve()
    if trust_anchor_output_path.is_relative_to(output_dir):
        raise RevisionProtocolError("[GEARS_TRUST_ANCHOR_INSIDE_EVIDENCE_BUNDLE]")
    python_executable = python_executable.resolve()
    repository = _verify_repository(repository_root, plan["official_repository"])
    environment_source = _resolve_bound_source(
        plan_root, plan["environment_lock"], "environment_lock"
    )
    dataset_source = _resolve_bound_source(plan_root, plan["dataset"], "dataset")
    metric_source = _resolve_repository_source(
        repository_root, plan["official_metric_source"], "official_metric_source"
    )
    installed_versions, runtime_environment_headers = _verify_environment(
        python_executable,
        environment_source,
        require_h5py=dataset_source.suffix.casefold() == ".h5ad",
    )
    expected_metrics = _numeric_metrics(plan.get("expected_metrics"), "expected_metrics")
    phase_contracts = _validate_execution_specs(
        plan.get("execution"),
        set(expected_metrics),
        plan_root,
        repository_root,
        plan,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    copied_environment = inputs_dir / "environment.lock"
    copied_dataset = inputs_dir / f"dataset{dataset_source.suffix}"
    copied_metric = inputs_dir / f"official_metric_source{metric_source.suffix or '.py'}"
    copied_plan = inputs_dir / "execution_plan.json"
    for source, destination in (
        (environment_source, copied_environment),
        (dataset_source, copied_dataset),
        (metric_source, copied_metric),
        (plan_path, copied_plan),
    ):
        _copy_verified(source, destination)

    copied_entrypoints: dict[str, Path] = {}
    copied_expected_value_provenance: dict[str, Path] = {}
    for phase in ("official_reference", "candidate"):
        binding = plan["execution"][phase]["entrypoint"]
        source_root = repository_root if binding["scope"] == "repository" else plan_root
        source = _resolve_bound_source(source_root, binding, f"execution.{phase}.entrypoint")
        destination = inputs_dir / f"{phase}_entrypoint{source.suffix or '.py'}"
        _copy_verified(source, destination)
        copied_entrypoints[phase] = destination
        provenance_source = phase_contracts[phase]["source_path"]
        provenance_destination = inputs_dir / f"{phase}_expected_value_provenance.json"
        _copy_verified(provenance_source, provenance_destination)
        copied_expected_value_provenance[phase] = provenance_destination

    bindings = _common_bindings(plan, repository)
    execution_audits: dict[str, Path] = {}
    wrapped_outputs: dict[str, Path] = {}
    raw_evidence: dict[str, Path] = {}
    state_sidecars: dict[str, dict[str, Path]] = {}
    for phase in ("official_reference", "candidate"):
        raw_output = output_dir / f".{phase}.raw.json"
        audit_path = output_dir / f"{phase}_execution_audit.json"
        wrapped_path = output_dir / f"{phase}_output.json"
        metrics, audit, generated_sidecars = _execute_phase(
            phase=phase,
            specification=plan["execution"][phase],
            repository_root=repository_root,
            python_executable=python_executable,
            dataset_source=dataset_source,
            raw_output=raw_output,
            expected_metric_names=set(expected_metrics),
            metric_source=metric_source,
            metric_function=str(plan["official_metric_source"]["function"]),
            metric_directions=plan["metric_directions"],
            plan_root=plan_root,
            repository=repository,
            bindings=bindings,
            installed_versions=installed_versions,
            runtime_environment_headers=runtime_environment_headers,
            phase_contract=phase_contracts[phase],
            split=plan["split"],
            training_config=plan["training_config"],
        )
        _write_self_hashed_json(audit_path, audit, "audit_hash")
        audit_file_hash = file_sha256(audit_path)
        wrapped = {
            "schema_version": "1.0",
            "metadata": {
                **bindings,
                "execution_phase": phase,
                "phase_role": plan["execution"][phase]["phase_role"],
                "phase_task_hash": canonical_sha256(phase_contracts[phase]["record"]),
                "execution_audit_sha256": audit_file_hash,
            },
            "metrics": metrics,
        }
        _atomic_write_json(wrapped_path, wrapped)
        evidence_path = output_dir / f"{phase}_raw_evidence.json"
        os.replace(raw_output, evidence_path)
        retained_sidecars: dict[str, Path] = {}
        for kind, generated in generated_sidecars.items():
            retained = output_dir / f"{phase}_{kind}.npz"
            os.replace(generated, retained)
            retained_sidecars[kind] = retained
        execution_audits[phase] = audit_path
        wrapped_outputs[phase] = wrapped_path
        raw_evidence[phase] = evidence_path
        state_sidecars[phase] = retained_sidecars

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "official_repository": repository,
        "environment_lock": _package_binding(output_dir, copied_environment),
        "dataset": {
            "name": str(plan["dataset"]["name"]),
            **_package_binding(output_dir, copied_dataset),
        },
        "split": dict(plan["split"]),
        "training_config": dict(plan["training_config"]),
        "official_metric_source": {
            **_package_binding(output_dir, copied_metric),
            "function": str(plan["official_metric_source"]["function"]),
            "repository_source_id": str(plan["official_metric_source"]["source_id"]),
        },
        "execution_plan": _package_binding(output_dir, copied_plan),
        "official_reference_execution_audit": _package_binding(
            output_dir, execution_audits["official_reference"]
        ),
        "candidate_execution_audit": _package_binding(output_dir, execution_audits["candidate"]),
        "official_reference_raw_evidence": _package_binding(
            output_dir, raw_evidence["official_reference"]
        ),
        "candidate_raw_evidence": _package_binding(output_dir, raw_evidence["candidate"]),
        "official_reference_entrypoint": _package_binding(
            output_dir, copied_entrypoints["official_reference"]
        ),
        "candidate_entrypoint": _package_binding(output_dir, copied_entrypoints["candidate"]),
        "official_reference_expected_value_provenance": _package_binding(
            output_dir, copied_expected_value_provenance["official_reference"]
        ),
        "candidate_expected_value_provenance": _package_binding(
            output_dir, copied_expected_value_provenance["candidate"]
        ),
        "official_reference_initialization_state": _package_binding(
            output_dir, state_sidecars["official_reference"]["initialization_state"]
        ),
        "official_reference_checkpoint_state": _package_binding(
            output_dir, state_sidecars["official_reference"]["checkpoint_state"]
        ),
        "official_reference_graph_state": _package_binding(
            output_dir, state_sidecars["official_reference"]["graph_state"]
        ),
        "candidate_initialization_state": _package_binding(
            output_dir, state_sidecars["candidate"]["initialization_state"]
        ),
        "candidate_checkpoint_state": _package_binding(
            output_dir, state_sidecars["candidate"]["checkpoint_state"]
        ),
        "candidate_graph_state": _package_binding(
            output_dir, state_sidecars["candidate"]["graph_state"]
        ),
        "official_reference_output": _package_binding(
            output_dir, wrapped_outputs["official_reference"]
        ),
        "candidate_output": _package_binding(output_dir, wrapped_outputs["candidate"]),
        "expected_metrics": expected_metrics,
        "tolerances": dict(plan["tolerances"]),
        "metric_directions": dict(plan["metric_directions"]),
        "execution_plan_hash": str(plan["plan_hash"]),
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    manifest_path = output_dir / "gears_positive_control_manifest.json"
    _atomic_write_json(manifest_path, manifest)
    create_gears_trust_anchor(manifest_path, trust_anchor_output_path)
    registry = validate_gears_positive_control(
        manifest_path,
        trust_anchor_path=trust_anchor_output_path,
        expected_trust_anchor_sha256=file_sha256(trust_anchor_output_path),
    )
    _atomic_write_json(output_dir / "gears_validation_registry.json", registry)
    return registry


def _read_plan(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RevisionProtocolError(f"[GEARS_EXECUTION_PLAN_INVALID] {error}") from error
    if not isinstance(payload, dict):
        raise RevisionProtocolError("[GEARS_EXECUTION_PLAN_INVALID] root must be an object")
    copy = dict(payload)
    declared_hash = copy.pop("plan_hash", None)
    if declared_hash != canonical_sha256(copy):
        raise RevisionProtocolError("[GEARS_EXECUTION_PLAN_HASH_MISMATCH]")
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
    if set(payload) != required or payload.get("schema_version") != "1.0":
        raise RevisionProtocolError("[GEARS_EXECUTION_PLAN_SCHEMA_MISMATCH]")
    if _contains_placeholder(payload):
        raise RevisionProtocolError("[GEARS_EXECUTION_PLAN_PLACEHOLDER_PRESENT]")
    return payload


def _verify_repository(root: Path, binding: Mapping[str, Any]) -> dict[str, str]:
    url = str(binding.get("url", "")).rstrip("/").removesuffix(".git")
    commit = str(binding.get("commit", ""))
    if url != OFFICIAL_REPOSITORY or not COMMIT_RE.fullmatch(commit):
        raise RevisionProtocolError("[GEARS_OFFICIAL_REPOSITORY_BINDING_INVALID]")
    if not root.is_dir():
        raise RevisionProtocolError("[GEARS_REPOSITORY_MISSING]")
    observed_commit = _git(root, "rev-parse", "HEAD")
    observed_url = _git(root, "remote", "get-url", "origin").rstrip("/").removesuffix(".git")
    if observed_commit != commit or observed_url != url:
        raise RevisionProtocolError("[GEARS_REPOSITORY_CHECKOUT_MISMATCH]")
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise RevisionProtocolError("[GEARS_REPOSITORY_TRACKED_WORKTREE_DIRTY]")
    return {"url": url, "commit": commit}


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode:
        raise RevisionProtocolError(
            f"[GEARS_GIT_COMMAND_FAILED] {' '.join(arguments)}:{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _resolve_bound_source(root: Path, binding: Mapping[str, Any], field: str) -> Path:
    if not isinstance(binding, Mapping):
        raise RevisionProtocolError(f"[GEARS_PLAN_BINDING_INVALID] {field}")
    source_id = str(binding.get("source_id", ""))
    expected_hash = str(binding.get("sha256", ""))
    source = Path(source_id)
    if not source_id or source.is_absolute() or ".." in source.parts:
        raise RevisionProtocolError(f"[GEARS_PLAN_BINDING_INVALID] {field}")
    resolved = (root / source).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise RevisionProtocolError(f"[GEARS_PLAN_BOUND_FILE_MISSING] {field}")
    if not SHA256_RE.fullmatch(expected_hash) or file_sha256(resolved) != expected_hash:
        raise RevisionProtocolError(f"[GEARS_PLAN_BOUND_FILE_HASH_MISMATCH] {field}")
    return resolved


def _resolve_repository_source(root: Path, binding: Mapping[str, Any], field: str) -> Path:
    return _resolve_bound_source(root, binding, field)


def _verify_environment(
    python_executable: Path, lock_path: Path, *, require_h5py: bool
) -> tuple[dict[str, str], dict[str, str]]:
    if not python_executable.is_file():
        raise RevisionProtocolError("[GEARS_PYTHON_EXECUTABLE_MISSING]")
    locked: dict[str, str] = {}
    headers: dict[str, str] = {}
    text = lock_path.read_text(encoding="utf-8")
    if re.search(r"REPLACE|TODO|TBD|UNKNOWN|PLACEHOLDER", text, re.IGNORECASE):
        raise RevisionProtocolError("[GEARS_ENVIRONMENT_LOCK_PLACEHOLDER_PRESENT]")
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            if line.startswith("#"):
                header = re.fullmatch(r"#\s*([a-z0-9_-]+):\s*(\S+)", line, re.IGNORECASE)
                if header is not None:
                    name = header.group(1).casefold()
                    if name in headers:
                        raise RevisionProtocolError("[GEARS_ENVIRONMENT_LOCK_DUPLICATE_HEADER]")
                    headers[name] = header.group(2)
            continue
        match = LOCK_LINE_RE.fullmatch(line)
        if match is None:
            raise RevisionProtocolError(f"[GEARS_ENVIRONMENT_LOCK_NOT_EXACT] line={line_number}")
        name, version = match.groups()
        normalized = re.sub(r"[-_.]+", "-", name).casefold()
        if normalized in locked:
            raise RevisionProtocolError("[GEARS_ENVIRONMENT_LOCK_DUPLICATE_PACKAGE]")
        locked[normalized] = version
    if not locked:
        raise RevisionProtocolError("[GEARS_ENVIRONMENT_LOCK_EMPTY]")
    script = """
import importlib.metadata as metadata
import json
import platform
import subprocess

packages = []
for distribution in metadata.distributions():
    name = distribution.metadata.get("Name")
    if not name:
        raise RuntimeError("distribution name missing")
    packages.append([str(name), str(distribution.version)])
try:
    import torch
except ImportError:
    cuda_runtime = "NOT_AVAILABLE"
    cudnn_version = "NOT_AVAILABLE"
else:
    cuda_runtime = str(torch.version.cuda or "NOT_AVAILABLE")
    cudnn_version = str(torch.backends.cudnn.version() or "NOT_AVAILABLE")
try:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )
except (OSError, subprocess.TimeoutExpired):
    driver_version = "NOT_AVAILABLE"
else:
    values = sorted({line.strip() for line in completed.stdout.splitlines() if line.strip()})
    driver_version = ",".join(values) if completed.returncode == 0 and values else "NOT_AVAILABLE"
print(json.dumps({
    "python": platform.python_version(),
    "cuda_runtime": cuda_runtime,
    "cudnn_version": cudnn_version,
    "nvidia_driver_version": driver_version,
    "packages": packages,
}, sort_keys=True))
"""
    try:
        completed = subprocess.run(
            [str(python_executable), "-c", script],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RevisionProtocolError("[GEARS_ENVIRONMENT_INSPECTION_FAILED]") from error
    if completed.returncode:
        raise RevisionProtocolError("[GEARS_ENVIRONMENT_INSPECTION_FAILED]")
    try:
        inspection = json.loads(completed.stdout)
        installed_raw = inspection["packages"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RevisionProtocolError("[GEARS_ENVIRONMENT_INSPECTION_FAILED]") from error
    installed: dict[str, str] = {}
    for item in installed_raw:
        if not isinstance(item, list) or len(item) != 2:
            raise RevisionProtocolError("[GEARS_ENVIRONMENT_INSPECTION_FAILED]")
        normalized = re.sub(r"[-_.]+", "-", str(item[0])).casefold()
        version = str(item[1])
        prior = installed.get(normalized)
        if prior is not None and prior != version:
            raise RevisionProtocolError(
                f"[GEARS_ENVIRONMENT_DUPLICATE_DISTRIBUTION_CONFLICT] {normalized}"
            )
        installed[normalized] = version
    installed = dict(sorted(installed.items()))
    direct_dependencies = _direct_runtime_dependencies()
    expected_headers = _environment_lock_headers(
        installed,
        python_version=str(inspection.get("python", "")),
        cuda_runtime=str(inspection.get("cuda_runtime", "")),
        cudnn_version=str(inspection.get("cudnn_version", "")),
        nvidia_driver_version=str(inspection.get("nvidia_driver_version", "")),
        direct_dependencies=direct_dependencies,
    )
    if headers != expected_headers:
        raise RevisionProtocolError("[GEARS_ENVIRONMENT_LOCK_RUNTIME_HEADER_MISMATCH]")
    if set(locked) != set(installed):
        raise RevisionProtocolError(
            "[GEARS_ENVIRONMENT_LOCK_PACKAGE_SET_MISMATCH] "
            f"missing={sorted(set(installed)-set(locked))};"
            f"extra={sorted(set(locked)-set(installed))}"
        )
    if require_h5py and "h5py" not in locked:
        raise RevisionProtocolError("[GEARS_ENVIRONMENT_LOCK_H5PY_MISSING]")
    missing_direct = sorted(set(direct_dependencies) - set(locked))
    if require_h5py and missing_direct:
        raise RevisionProtocolError(
            f"[GEARS_ENVIRONMENT_LOCK_DIRECT_DEPENDENCY_MISSING] {missing_direct}"
        )
    mismatches = {
        name: {"locked": version, "installed": installed.get(name)}
        for name, version in locked.items()
        if installed.get(name) != version
    }
    if mismatches:
        raise RevisionProtocolError(
            f"[GEARS_ENVIRONMENT_LOCK_MISMATCH] {json.dumps(mismatches, sort_keys=True)}"
        )
    return installed, expected_headers


def _validate_execution_specs(
    value: Any,
    expected_metrics: set[str],
    plan_root: Path,
    repository_root: Path,
    plan: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {"official_reference", "candidate"}:
        raise RevisionProtocolError("[GEARS_EXECUTION_SPEC_INVALID]")
    contracts: dict[str, dict[str, Any]] = {}
    for phase, specification in value.items():
        if not isinstance(specification, Mapping) or set(specification) != {
            "argv",
            "entrypoint",
            "phase_role",
            "expected_value_provenance",
            "workflow_reuse_policy",
        }:
            raise RevisionProtocolError(f"[GEARS_EXECUTION_SPEC_INVALID] {phase}")
        if specification.get("phase_role") != PHASE_ROLES[phase]:
            raise RevisionProtocolError(f"[GEARS_EXECUTION_PHASE_ROLE_INVALID] {phase}")
        if specification.get("workflow_reuse_policy") not in {
            PINNED_WORKFLOW_REUSE_POLICY,
            DISTINCT_WORKFLOW_POLICY,
        }:
            raise RevisionProtocolError(f"[GEARS_EXECUTION_WORKFLOW_POLICY_INVALID] {phase}")
        entrypoint = specification["entrypoint"]
        if not isinstance(entrypoint, Mapping) or set(entrypoint) != {
            "scope",
            "source_id",
            "sha256",
        }:
            raise RevisionProtocolError(f"[GEARS_EXECUTION_ENTRYPOINT_INVALID] {phase}")
        scope = str(entrypoint["scope"])
        scope_root = repository_root if scope == "repository" else plan_root
        if scope not in {"repository", "plan"}:
            raise RevisionProtocolError(f"[GEARS_EXECUTION_ENTRYPOINT_INVALID] {phase}")
        entrypoint_path = _resolve_bound_source(
            scope_root, entrypoint, f"execution.{phase}.entrypoint"
        )
        argv = specification["argv"]
        if (
            not isinstance(argv, list)
            or len(argv) < 2
            or any(not isinstance(token, str) or not token for token in argv)
            or argv[0] != "{python}"
            or "{output}" not in argv
            or "{entrypoint}" not in argv
            or "-c" in argv
        ):
            raise RevisionProtocolError(f"[GEARS_EXECUTION_ARGV_INVALID] {phase}")
        for token in argv:
            unknown = set(re.findall(r"\{[^{}]+\}", token)) - PLACEHOLDERS
            if unknown:
                raise RevisionProtocolError(
                    f"[GEARS_EXECUTION_UNKNOWN_PLACEHOLDER] {phase}:{sorted(unknown)}"
                )
        provenance_binding = specification.get("expected_value_provenance")
        provenance_path = _resolve_bound_source(
            plan_root,
            provenance_binding,
            f"execution.{phase}.expected_value_provenance",
        )
        record = _read_expected_value_provenance(
            provenance_path,
            phase=phase,
            specification=specification,
            plan=plan,
            entrypoint_sha256=file_sha256(entrypoint_path),
            expected_metric_names=expected_metrics,
        )
        contracts[phase] = {
            "record": record,
            "source_path": provenance_path,
            "source_sha256": file_sha256(provenance_path),
        }
    if not expected_metrics:
        raise RevisionProtocolError("[GEARS_EXPECTED_METRICS_EMPTY]")
    reference_specification = value["official_reference"]
    candidate_specification = value["candidate"]
    same_workflow = (
        reference_specification["entrypoint"]["sha256"]
        == candidate_specification["entrypoint"]["sha256"]
        and reference_specification["argv"] == candidate_specification["argv"]
    )
    if same_workflow and any(
        specification.get("workflow_reuse_policy") != PINNED_WORKFLOW_REUSE_POLICY
        for specification in (reference_specification, candidate_specification)
    ):
        raise RevisionProtocolError("[GEARS_IDENTICAL_PHASE_WORKFLOW_UNDECLARED]")
    if (
        contracts["official_reference"]["source_sha256"] == contracts["candidate"]["source_sha256"]
        or contracts["official_reference"]["record"]["source_id"]
        == contracts["candidate"]["record"]["source_id"]
    ):
        raise RevisionProtocolError("[GEARS_PHASE_PROVENANCE_NOT_INDEPENDENTLY_BOUND]")
    return contracts


def _read_expected_value_provenance(
    path: Path,
    *,
    phase: str,
    specification: Mapping[str, Any],
    plan: Mapping[str, Any],
    entrypoint_sha256: str,
    expected_metric_names: set[str],
) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RevisionProtocolError(f"[GEARS_EXPECTED_VALUE_PROVENANCE_INVALID] {phase}") from error
    required = {
        "schema_version",
        "phase_role",
        "source_kind",
        "source_id",
        "frozen_before_execution",
        "repository_url",
        "commit",
        "dataset_sha256",
        "split",
        "entrypoint_sha256",
        "argv_sha256",
        "expected_metrics",
        "record_hash",
    }
    if not isinstance(record, dict) or set(record) != required or _contains_placeholder(record):
        raise RevisionProtocolError(f"[GEARS_EXPECTED_VALUE_PROVENANCE_INVALID] {phase}")
    payload = dict(record)
    declared_hash = payload.pop("record_hash", None)
    try:
        record_metrics = _numeric_metrics(
            record.get("expected_metrics"), f"execution.{phase}.expected_metrics"
        )
    except RevisionProtocolError as error:
        raise RevisionProtocolError(f"[GEARS_EXPECTED_VALUE_PROVENANCE_INVALID] {phase}") from error
    expected_source_kind = {
        "official_reference": {
            "published_official_benchmark",
            "pinned_official_reference_run",
        },
        "candidate": {"predeclared_heldout_adaptation_target"},
    }[phase]
    valid = (
        record.get("schema_version") == "1.0"
        and declared_hash == canonical_sha256(payload)
        and record.get("phase_role") == PHASE_ROLES[phase]
        and record.get("source_kind") in expected_source_kind
        and bool(str(record.get("source_id", "")).strip())
        and record.get("frozen_before_execution") is True
        and record.get("repository_url") == plan["official_repository"]["url"]
        and record.get("commit") == plan["official_repository"]["commit"]
        and record.get("dataset_sha256") == plan["dataset"]["sha256"]
        and record.get("split") == plan["split"]
        and record.get("entrypoint_sha256") == entrypoint_sha256
        and record.get("argv_sha256") == canonical_sha256(specification["argv"])
        and set(record_metrics) == expected_metric_names
        and record_metrics == _numeric_metrics(plan["expected_metrics"], "expected_metrics")
    )
    if not valid:
        raise RevisionProtocolError(f"[GEARS_EXPECTED_VALUE_PROVENANCE_INVALID] {phase}")
    return record


def _execute_phase(
    *,
    phase: str,
    specification: Mapping[str, Any],
    repository_root: Path,
    python_executable: Path,
    dataset_source: Path,
    raw_output: Path,
    expected_metric_names: set[str],
    metric_source: Path,
    metric_function: str,
    metric_directions: Mapping[str, Any],
    plan_root: Path,
    repository: Mapping[str, str],
    bindings: Mapping[str, Any],
    installed_versions: Mapping[str, str],
    runtime_environment_headers: Mapping[str, str],
    phase_contract: Mapping[str, Any],
    split: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any], dict[str, Path]]:
    entrypoint_binding = specification["entrypoint"]
    entrypoint_root = repository_root if entrypoint_binding["scope"] == "repository" else plan_root
    entrypoint = _resolve_bound_source(
        entrypoint_root, entrypoint_binding, f"execution.{phase}.entrypoint"
    )
    replacements = {
        "{python}": str(python_executable),
        "{repository_root}": str(repository_root),
        "{dataset}": str(dataset_source),
        "{output}": str(raw_output),
        "{entrypoint}": str(entrypoint),
    }
    template = list(specification["argv"])
    argv = [_replace_placeholders(token, replacements) for token in template]
    raw_output.unlink(missing_ok=True)
    sidecars = _sidecar_paths(raw_output)
    for sidecar in sidecars.values():
        sidecar.unlink(missing_ok=True)
    started = time.perf_counter()
    completed = subprocess.run(
        argv,
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=False,
        env={
            **os.environ,
            "PYTHONHASHSEED": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
    )
    wall_seconds = float(time.perf_counter() - started)
    if completed.returncode:
        raise RevisionProtocolError(
            f"[GEARS_EXECUTION_FAILED] {phase}:returncode={completed.returncode};"
            f"stderr_sha256={_bytes_sha256(completed.stderr)}"
        )
    if not raw_output.is_file():
        raise RevisionProtocolError(f"[GEARS_EXECUTION_OUTPUT_MISSING] {phase}")
    missing_sidecars = [name for name, path in sidecars.items() if not path.is_file()]
    if missing_sidecars:
        raise RevisionProtocolError(
            f"[GEARS_EXECUTION_STATE_SIDECAR_MISSING] {phase}:{','.join(missing_sidecars)}"
        )
    sidecar_manifests = {
        name: _load_npz_manifest(path, f"execution.{phase}.{name}")
        for name, path in sidecars.items()
    }
    try:
        raw_payload = json.loads(raw_output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RevisionProtocolError(f"[GEARS_EXECUTION_OUTPUT_INVALID] {phase}:{error}") from error
    evidence, results = _validate_raw_evidence(
        raw_payload,
        phase=phase,
        expected_metric_names=expected_metric_names,
        metric_directions=metric_directions,
        expected_split=split,
        training_config=training_config,
        sidecar_manifests=sidecar_manifests,
    )
    metric_callable = _load_metric_callable(metric_source, metric_function)
    computed = metric_callable(results)
    if isinstance(computed, tuple):
        computed = computed[0]
    metrics = _numeric_metrics(computed, f"execution.{phase}.official_recomputed_metrics")
    if set(metrics) != expected_metric_names:
        raise RevisionProtocolError(f"[GEARS_EXECUTION_METRIC_SET_MISMATCH] {phase}")
    reported = _numeric_metrics(evidence["reported_metrics"], f"execution.{phase}.reported_metrics")
    if reported != metrics:
        raise RevisionProtocolError(f"[GEARS_REPORTED_METRIC_RECOMPUTATION_MISMATCH] {phase}")
    baseline_results = {
        key: np.asarray(value, dtype=str if key == "pert_cat" else float)
        for key, value in evidence["baselines"]["control_mean_zero_change"]["results"].items()
    }
    computed_baseline = metric_callable(baseline_results)
    if isinstance(computed_baseline, tuple):
        computed_baseline = computed_baseline[0]
    baseline_metrics = _numeric_metrics(
        computed_baseline, f"execution.{phase}.official_recomputed_baseline_metrics"
    )
    retained_baseline = _numeric_metrics(
        evidence["baselines"]["control_mean_zero_change"]["reported_metrics"],
        f"execution.{phase}.retained_baseline_metrics",
    )
    if baseline_metrics != retained_baseline:
        raise RevisionProtocolError(f"[GEARS_BASELINE_METRIC_RECOMPUTATION_MISMATCH] {phase}")
    audit = {
        "schema_version": "2.0",
        "phase": phase,
        "phase_role": specification["phase_role"],
        "phase_task_hash": canonical_sha256(phase_contract["record"]),
        "expected_value_provenance_sha256": phase_contract["source_sha256"],
        "expected_value_record_hash": phase_contract["record"]["record_hash"],
        "repository_url": repository["url"],
        "commit": repository["commit"],
        "normalized_command_argv": template,
        "returncode": completed.returncode,
        "wall_seconds": wall_seconds,
        "stdout_sha256": _bytes_sha256(completed.stdout),
        "stderr_sha256": _bytes_sha256(completed.stderr),
        "raw_metrics_sha256": file_sha256(raw_output),
        "entrypoint_sha256": file_sha256(entrypoint),
        "official_metric_recomputed": True,
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
        "baseline_metrics": retained_baseline,
        "initialization_state_sha256": file_sha256(sidecars["initialization_state"]),
        "checkpoint_state_sha256": file_sha256(sidecars["checkpoint_state"]),
        "graph_state_sha256": file_sha256(sidecars["graph_state"]),
        "initialization_array_manifest_hash": canonical_sha256(
            sidecar_manifests["initialization_state"]
        ),
        "checkpoint_array_manifest_hash": canonical_sha256(sidecar_manifests["checkpoint_state"]),
        "graph_array_manifest_hash": canonical_sha256(sidecar_manifests["graph_state"]),
        "environment_lock_sha256": bindings["environment_lock_sha256"],
        "dataset_sha256": bindings["dataset_sha256"],
        "metric_source_sha256": bindings["metric_source_sha256"],
        "split_name": bindings["split_name"],
        "split_seed": bindings["split_seed"],
        "training_config_hash": bindings["training_config_hash"],
        "installed_locked_versions": dict(installed_versions),
        "runtime_environment_headers": dict(runtime_environment_headers),
    }
    return metrics, audit, sidecars


def _common_bindings(plan: Mapping[str, Any], repository: Mapping[str, str]) -> dict[str, Any]:
    return {
        "repository_url": repository["url"],
        "commit": repository["commit"],
        "dataset_sha256": str(plan["dataset"]["sha256"]),
        "split_name": str(plan["split"]["name"]),
        "split_seed": plan["split"]["seed"],
        "training_config_hash": canonical_sha256(plan["training_config"]),
        "metric_source_sha256": str(plan["official_metric_source"]["sha256"]),
        "metric_function": str(plan["official_metric_source"]["function"]),
        "environment_lock_sha256": str(plan["environment_lock"]["sha256"]),
    }


def _validate_raw_evidence(
    value: Any,
    *,
    phase: str,
    expected_metric_names: set[str],
    metric_directions: Mapping[str, Any],
    expected_split: Mapping[str, Any],
    training_config: Mapping[str, Any],
    sidecar_manifests: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    required = {
        "schema_version",
        "results",
        "reported_metrics",
        "baselines",
        "training_loss",
        "validation_loss",
        "gene_names",
        "perturbable_gene_order",
        "split_membership",
        "condition_target_mapping",
        "identifier_coverage",
        "target_audit",
        "model_identity",
        "graph_provenance",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema_version") != "2.0"
    ):
        raise RevisionProtocolError(f"[GEARS_RAW_EVIDENCE_SCHEMA_MISMATCH] {phase}")
    results = _coerce_gears_results(value.get("results"), phase, "model")
    if (
        results["pred"].ndim != 2
        or results["pred"].shape != results["truth"].shape
        or results["pred_de"].ndim != 2
        or results["pred_de"].shape != results["truth_de"].shape
        or len(results["pert_cat"]) != results["pred"].shape[0]
        or len(results["pert_cat"]) != results["pred_de"].shape[0]
        or results["pred"].shape[0] < 2
        or results["pred"].shape[1] < 2
        or not all(
            np.isfinite(array).all() for name, array in results.items() if name != "pert_cat"
        )
        or np.std(results["pred"]) == 0
        or np.std(results["truth"]) == 0
    ):
        raise RevisionProtocolError(f"[GEARS_RAW_VECTOR_DEGENERATE_OR_INVALID] {phase}")
    gene_names = value.get("gene_names")
    if (
        not isinstance(gene_names, list)
        or len(gene_names) != results["pred"].shape[1]
        or len(set(str(name) for name in gene_names)) != len(gene_names)
    ):
        raise RevisionProtocolError(f"[GEARS_GENE_ORDER_INVALID] {phase}")
    perturbable_gene_order = value.get("perturbable_gene_order")
    if (
        not isinstance(perturbable_gene_order, list)
        or not perturbable_gene_order
        or len({str(name) for name in perturbable_gene_order}) != len(perturbable_gene_order)
        or not set(str(name) for name in perturbable_gene_order).issubset(
            {str(name) for name in gene_names}
        )
    ):
        raise RevisionProtocolError(f"[GEARS_PERTURBABLE_GENE_ORDER_INVALID] {phase}")
    perturbable_gene_order = [str(name) for name in perturbable_gene_order]
    perturbable_index = {gene: index for index, gene in enumerate(perturbable_gene_order)}
    for field in ("training_loss", "validation_loss"):
        losses = np.asarray(value.get(field), dtype=float)
        expected_epochs = training_config.get("epochs")
        if (
            losses.ndim != 1
            or not isinstance(expected_epochs, int)
            or isinstance(expected_epochs, bool)
            or expected_epochs <= 0
            or len(losses) != expected_epochs
            or not np.isfinite(losses).all()
        ):
            raise RevisionProtocolError(f"[GEARS_LOSS_HISTORY_INVALID] {phase}:{field}")

    split_membership = value.get("split_membership")
    if not isinstance(split_membership, Mapping) or set(split_membership) != {
        "name",
        "seed",
        "row_ids",
        "condition_ids",
        "eligible_test_conditions",
    }:
        raise RevisionProtocolError(f"[GEARS_SPLIT_MEMBERSHIP_INVALID] {phase}")
    row_ids = split_membership.get("row_ids")
    condition_ids = split_membership.get("condition_ids")
    eligible_conditions = split_membership.get("eligible_test_conditions")
    if (
        split_membership.get("name") != expected_split.get("name")
        or split_membership.get("seed") != expected_split.get("seed")
        or not isinstance(row_ids, list)
        or len(row_ids) != len(results["pert_cat"])
        or len({str(item) for item in row_ids}) != len(row_ids)
        or not all(str(item).strip() for item in row_ids)
        or not isinstance(condition_ids, list)
        or [str(item) for item in condition_ids] != results["pert_cat"].tolist()
        or not isinstance(eligible_conditions, list)
        or not eligible_conditions
        or len({str(item) for item in eligible_conditions}) != len(eligible_conditions)
        or not set(results["pert_cat"].tolist()).issubset(
            {str(item) for item in eligible_conditions}
        )
    ):
        raise RevisionProtocolError(f"[GEARS_SPLIT_MEMBERSHIP_INVALID] {phase}")

    unique_conditions = sorted(set(results["pert_cat"].tolist()))
    if len(unique_conditions) < 2:
        raise RevisionProtocolError(f"[GEARS_CONDITION_LEVEL_DEGENERACY] {phase}")
    condition_means: list[np.ndarray] = []
    for condition in unique_conditions:
        subset = results["pred"][results["pert_cat"] == condition]
        if subset.size == 0 or float(np.std(subset)) == 0.0:
            raise RevisionProtocolError(f"[GEARS_CONDITION_LEVEL_DEGENERACY] {phase}:{condition}")
        condition_means.append(np.mean(subset, axis=0))
    if float(np.std(np.stack(condition_means), axis=0).max()) == 0.0:
        raise RevisionProtocolError(f"[GEARS_BETWEEN_CONDITION_DEGENERACY] {phase}")

    mapping = value.get("condition_target_mapping")
    if not isinstance(mapping, list) or len(mapping) != len(unique_conditions):
        raise RevisionProtocolError(f"[GEARS_CONDITION_TARGET_MAPPING_INVALID] {phase}")
    normalized_mapping: list[dict[str, Any]] = []
    for entry in mapping:
        if not isinstance(entry, Mapping) or set(entry) != {
            "condition",
            "targets",
            "target_indices",
            "target_indicator",
            "mapped",
        }:
            raise RevisionProtocolError(f"[GEARS_CONDITION_TARGET_MAPPING_INVALID] {phase}")
        condition = str(entry.get("condition", ""))
        targets = entry.get("targets")
        expected_targets = [target for target in condition.split("+") if target != "ctrl"]
        expected_indices = [
            perturbable_index[target] for target in expected_targets if target in perturbable_index
        ]
        expected_indicator = [0] * len(perturbable_gene_order)
        for index in expected_indices:
            expected_indicator[index] = 1
        expected_mapped = bool(expected_targets) and len(expected_indices) == len(expected_targets)
        if (
            not condition
            or not isinstance(targets, list)
            or not expected_targets
            or len(set(expected_targets)) != len(expected_targets)
            or [str(target) for target in targets] != expected_targets
            or entry.get("target_indices") != expected_indices
            or entry.get("target_indicator") != expected_indicator
            or entry.get("mapped") is not expected_mapped
        ):
            raise RevisionProtocolError(f"[GEARS_CONDITION_TARGET_MAPPING_INVALID] {phase}")
        normalized_mapping.append(
            {
                "condition": condition,
                "targets": expected_targets,
                "target_indices": expected_indices,
                "target_indicator": expected_indicator,
                "mapped": expected_mapped,
            }
        )
    if [entry["condition"] for entry in normalized_mapping] != unique_conditions:
        raise RevisionProtocolError(f"[GEARS_CONDITION_TARGET_MAPPING_INVALID] {phase}")

    try:
        identifier_coverage = float(value.get("identifier_coverage"))
    except (TypeError, ValueError):
        identifier_coverage = math.nan
    audit = value.get("target_audit")
    mapped_conditions = sum(bool(entry["mapped"]) for entry in normalized_mapping)
    unmapped_conditions = [
        entry["condition"] for entry in normalized_mapping if not entry["mapped"]
    ]
    if (
        identifier_coverage != 1.0
        or not isinstance(audit, Mapping)
        or set(audit)
        != {
            "total_conditions",
            "mapped_conditions",
            "unmapped_conditions",
            "perturbable_gene_count",
        }
        or audit.get("total_conditions") != len(unique_conditions)
        or audit.get("mapped_conditions") != mapped_conditions
        or audit.get("unmapped_conditions") != unmapped_conditions
        or audit.get("perturbable_gene_count") != len(perturbable_gene_order)
        or mapped_conditions != len(unique_conditions)
    ):
        raise RevisionProtocolError(f"[GEARS_IDENTIFIER_OR_TARGET_AUDIT_FAILED] {phase}")

    baselines = value.get("baselines")
    if not isinstance(baselines, Mapping) or set(baselines) != {"control_mean_zero_change"}:
        raise RevisionProtocolError(f"[GEARS_BASELINE_EVIDENCE_SCHEMA_MISMATCH] {phase}")
    baseline = baselines["control_mean_zero_change"]
    if not isinstance(baseline, Mapping) or set(baseline) != {"results", "reported_metrics"}:
        raise RevisionProtocolError(f"[GEARS_BASELINE_EVIDENCE_SCHEMA_MISMATCH] {phase}")
    baseline_results = _coerce_gears_results(
        baseline.get("results"), phase, "control_mean_zero_change"
    )
    for field in ("pert_cat", "truth", "truth_de"):
        if not np.array_equal(baseline_results[field], results[field]):
            raise RevisionProtocolError(f"[GEARS_BASELINE_ALIGNMENT_MISMATCH] {phase}:{field}")
    if (
        baseline_results["pred"].shape != results["pred"].shape
        or baseline_results["pred_de"].shape != results["pred_de"].shape
        or not np.isfinite(baseline_results["pred"]).all()
        or not np.isfinite(baseline_results["pred_de"]).all()
    ):
        raise RevisionProtocolError(f"[GEARS_BASELINE_VECTOR_INVALID] {phase}")

    model_identity = value.get("model_identity")
    graph_provenance = value.get("graph_provenance")
    expected_state_fields = {"sha256", "arrays"}
    if not isinstance(model_identity, Mapping) or set(model_identity) != {
        "initialization_state",
        "checkpoint_state",
    }:
        raise RevisionProtocolError(f"[GEARS_MODEL_IDENTITY_INVALID] {phase}")
    for raw_name, sidecar_name in (
        ("initialization_state", "initialization_state"),
        ("checkpoint_state", "checkpoint_state"),
    ):
        identity = model_identity.get(raw_name)
        manifest = sidecar_manifests.get(sidecar_name)
        if (
            not isinstance(identity, Mapping)
            or set(identity) != expected_state_fields
            or not isinstance(manifest, Mapping)
            or identity.get("sha256") != manifest.get("file_sha256")
            or identity.get("arrays") != manifest.get("arrays")
        ):
            raise RevisionProtocolError(f"[GEARS_MODEL_IDENTITY_INVALID] {phase}:{raw_name}")
    if (
        model_identity["initialization_state"]["arrays"]
        == model_identity["checkpoint_state"]["arrays"]
    ):
        raise RevisionProtocolError(f"[GEARS_MODEL_STATE_DID_NOT_CHANGE] {phase}")
    graph_manifest = sidecar_manifests.get("graph_state")
    if (
        not isinstance(graph_provenance, Mapping)
        or set(graph_provenance) != {"source", "sha256", "arrays"}
        or not str(graph_provenance.get("source", "")).strip()
        or not isinstance(graph_manifest, Mapping)
        or graph_provenance.get("sha256") != graph_manifest.get("file_sha256")
        or graph_provenance.get("arrays") != graph_manifest.get("arrays")
    ):
        raise RevisionProtocolError(f"[GEARS_GRAPH_PROVENANCE_INVALID] {phase}")
    if graph_manifest.get("semantic_bindings", {}).get("gene_order") != [
        str(name) for name in gene_names
    ]:
        raise RevisionProtocolError(f"[GEARS_GENE_ORDER_SIDECAR_MISMATCH] {phase}")
    if graph_manifest.get("semantic_bindings", {}).get("perturbable_gene_order") != (
        perturbable_gene_order
    ):
        raise RevisionProtocolError(f"[GEARS_PERTURBABLE_GENE_ORDER_SIDECAR_MISMATCH] {phase}")

    reported = _numeric_metrics(value.get("reported_metrics"), f"{phase}.reported_metrics")
    baseline_metrics = _numeric_metrics(
        baseline.get("reported_metrics"), f"{phase}.baseline_metrics"
    )
    if set(reported) != expected_metric_names or set(baseline_metrics) != expected_metric_names:
        raise RevisionProtocolError(f"[GEARS_RAW_METRIC_SET_MISMATCH] {phase}")
    if (
        not isinstance(metric_directions, Mapping)
        or set(metric_directions) != expected_metric_names
    ):
        raise RevisionProtocolError("[GEARS_METRIC_DIRECTION_CONTRACT_INVALID]")
    for metric in sorted(expected_metric_names):
        direction = metric_directions[metric]
        if direction == "higher_is_better":
            better = reported[metric] > baseline_metrics[metric]
        elif direction == "lower_is_better":
            better = reported[metric] < baseline_metrics[metric]
        else:
            raise RevisionProtocolError("[GEARS_METRIC_DIRECTION_CONTRACT_INVALID]")
        if not better:
            raise RevisionProtocolError(
                f"[GEARS_BASELINE_POSITIVE_CONTROL_FAILED] {phase}:{metric}"
            )
    return value, results


def _coerce_gears_results(value: Any, phase: str, source: str) -> dict[str, np.ndarray]:
    fields = {"pert_cat", "pred", "truth", "pred_de", "truth_de"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RevisionProtocolError(f"[GEARS_RAW_RESULT_SCHEMA_MISMATCH] {phase}:{source}")
    try:
        return {
            "pert_cat": np.asarray(value["pert_cat"], dtype=str),
            "pred": np.asarray(value["pred"], dtype=float),
            "truth": np.asarray(value["truth"], dtype=float),
            "pred_de": np.asarray(value["pred_de"], dtype=float),
            "truth_de": np.asarray(value["truth_de"], dtype=float),
        }
    except (TypeError, ValueError) as error:
        raise RevisionProtocolError(
            f"[GEARS_RAW_RESULT_SCHEMA_MISMATCH] {phase}:{source}"
        ) from error


def _sidecar_paths(raw_output: Path) -> dict[str, Path]:
    return {
        "initialization_state": Path(f"{raw_output}.initialization.npz"),
        "checkpoint_state": Path(f"{raw_output}.checkpoint.npz"),
        "graph_state": Path(f"{raw_output}.graph.npz"),
    }


def _load_npz_manifest(path: Path, field: str) -> dict[str, Any]:
    semantic_bindings: dict[str, Any] = {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            if not archive.files:
                raise RevisionProtocolError(f"[GEARS_STATE_ARCHIVE_EMPTY] {field}")
            arrays: dict[str, dict[str, Any]] = {}
            for name in sorted(archive.files):
                array = np.ascontiguousarray(archive[name])
                if array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
                    raise RevisionProtocolError(f"[GEARS_STATE_ARCHIVE_UNSAFE] {field}:{name}")
                if not np.isfinite(array).all():
                    raise RevisionProtocolError(f"[GEARS_STATE_ARCHIVE_NONFINITE] {field}:{name}")
                arrays[name] = {
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                }
                semantic_name = {
                    "__cbac_gene_order_utf8": "gene_order",
                    "__cbac_perturbable_gene_order_utf8": "perturbable_gene_order",
                }.get(name)
                if semantic_name is not None:
                    if array.ndim != 1 or array.dtype != np.dtype(np.uint8):
                        raise RevisionProtocolError(
                            f"[GEARS_STATE_ARCHIVE_SEMANTIC_BINDING_INVALID] {field}:{name}"
                        )
                    try:
                        decoded = bytes(array.tolist()).decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise RevisionProtocolError(
                            f"[GEARS_STATE_ARCHIVE_SEMANTIC_BINDING_INVALID] {field}:{name}"
                        ) from error
                    semantic_bindings[semantic_name] = decoded.split("\0") if decoded else []
    except RevisionProtocolError:
        raise
    except Exception as error:
        raise RevisionProtocolError(f"[GEARS_STATE_ARCHIVE_INVALID] {field}") from error
    return {
        "file_sha256": file_sha256(path),
        "arrays": arrays,
        "semantic_bindings": semantic_bindings,
    }


def _load_metric_callable(path: Path, qualified_function: str) -> Any:
    function_name = qualified_function.rsplit(".", 1)[-1]
    if not function_name or not function_name.isidentifier():
        raise RevisionProtocolError("[GEARS_METRIC_FUNCTION_INVALID]")
    module_name = f"_cbac_gears_metric_{file_sha256(path)[:16]}"
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise RevisionProtocolError("[GEARS_METRIC_SOURCE_IMPORT_FAILED]")
    module = importlib.util.module_from_spec(specification)
    try:
        specification.loader.exec_module(module)
    except Exception as error:
        raise RevisionProtocolError(
            f"[GEARS_METRIC_SOURCE_IMPORT_FAILED] {type(error).__name__}"
        ) from error
    function = getattr(module, function_name, None)
    if not callable(function):
        raise RevisionProtocolError("[GEARS_METRIC_FUNCTION_MISSING]")
    return function


def _numeric_metrics(value: Any, field: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise RevisionProtocolError(f"[GEARS_NUMERIC_METRICS_INVALID] {field}")
    output: dict[str, float] = {}
    for name, raw in value.items():
        try:
            numeric = float(raw)
        except (TypeError, ValueError) as error:
            raise RevisionProtocolError(
                f"[GEARS_NUMERIC_METRICS_INVALID] {field}:{name}"
            ) from error
        if not str(name).strip() or not math.isfinite(numeric):
            raise RevisionProtocolError(f"[GEARS_NUMERIC_METRICS_INVALID] {field}:{name}")
        output[str(name)] = numeric
    return output


def _replace_placeholders(token: str, replacements: Mapping[str, str]) -> str:
    result = token
    for placeholder, replacement in replacements.items():
        result = result.replace(placeholder, replacement)
    return result


def _copy_verified(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    if file_sha256(temporary) != file_sha256(source):
        temporary.unlink(missing_ok=True)
        raise RevisionProtocolError("[GEARS_PACKAGE_COPY_HASH_MISMATCH]")
    os.replace(temporary, destination)


def _package_binding(root: Path, path: Path) -> dict[str, str]:
    return {
        "source_id": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
    }


def _write_self_hashed_json(path: Path, payload: dict[str, Any], hash_field: str) -> None:
    payload[hash_field] = canonical_sha256(payload)
    _atomic_write_json(path, payload)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _bytes_sha256(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return bool(re.search(r"REPLACE|TODO|TBD|UNKNOWN|PLACEHOLDER", value, re.IGNORECASE))
    if isinstance(value, Mapping):
        return any(
            _contains_placeholder(key) or _contains_placeholder(item) for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_placeholder(item) for item in value)
    return False


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trust-anchor-output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required acknowledgement because both declared GEARS commands will run.",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_argument_parser().parse_args(arguments)
    if not parsed.execute:
        raise RevisionProtocolError("[GEARS_EXECUTION_REQUIRES_EXPLICIT_EXECUTE_FLAG]")
    registry = execute_gears_reproduction(
        parsed.plan,
        parsed.repository_root,
        parsed.output,
        python_executable=parsed.python,
        trust_anchor_output_path=parsed.trust_anchor_output,
    )
    print(json.dumps(registry, indent=2, sort_keys=True))
    return 0 if registry["status"] == "RELEASED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

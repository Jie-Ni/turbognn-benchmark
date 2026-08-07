from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from cbac_revision.artifacts import canonical_sha256, file_sha256
from cbac_revision.errors import RevisionProtocolError
from cbac_revision.gears_reproduction import (
    _load_npz_manifest,
    _verify_environment,
    execute_gears_reproduction,
)
from cbac_revision.gears_validation import validate_gears_positive_control
from cbac_revision.runner import ENVIRONMENT_LOCK_HEADER_ORDER, write_complete_environment_lock
from cbac_revision.statistics import _external_lock


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _reproduction_fixture(tmp_path: Path, candidate_value: float) -> tuple[Path, Path]:
    repository = tmp_path / "GEARS"
    repository.mkdir()
    _git(repository, "init", "-b", "master")
    _git(repository, "config", "user.email", "test@example.org")
    _git(repository, "config", "user.name", "Test Fixture")
    metric_source = repository / "metric.py"
    metric_source.write_text(
        "def compute_metrics(results):\n    return {'pearson': float(results['pred'][0][0])}\n",
        encoding="utf-8",
    )
    executor = repository / "execute.py"
    executor.write_text(
        """import argparse, hashlib, json
from pathlib import Path
import numpy as np

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def manifest(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {}
        for name in sorted(archive.files):
            value = np.ascontiguousarray(archive[name])
            arrays[name] = {
                'dtype': value.dtype.str,
                'shape': list(value.shape),
                'sha256': hashlib.sha256(value.tobytes(order='C')).hexdigest(),
            }
    return {'sha256': sha256(path), 'arrays': arrays}

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--value', type=float, required=True)
args = parser.parse_args()
initialization = Path(f'{args.output}.initialization.npz')
checkpoint = Path(f'{args.output}.checkpoint.npz')
graph = Path(f'{args.output}.graph.npz')
np.savez_compressed(initialization, weight=np.asarray([0.0, 1.0]))
np.savez_compressed(checkpoint, weight=np.asarray([args.value, 1.0]))
np.savez_compressed(
    graph,
    edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
    __cbac_gene_order_utf8=np.frombuffer(b'A\\0B', dtype=np.uint8),
    __cbac_perturbable_gene_order_utf8=np.frombuffer(b'A\\0B', dtype=np.uint8),
)
baseline_results = {
    'pert_cat': ['A', 'B'],
    'pred': [[0.1, 0.1], [0.1, 0.1]],
    'truth': [[0.5, 0.1], [0.2, 0.6]],
    'pred_de': [[0.1, 0.1], [0.1, 0.1]],
    'truth_de': [[0.5, 0.1], [0.2, 0.6]],
}
payload = {
    'schema_version': '2.0',
    'results': {
        'pert_cat': ['A', 'B'],
        'pred': [[args.value, 0.0], [0.2, 0.4]],
        'truth': [[0.5, 0.1], [0.2, 0.6]],
        'pred_de': [[args.value, 0.0], [0.2, 0.4]],
        'truth_de': [[0.5, 0.1], [0.2, 0.6]],
    },
    'reported_metrics': {'pearson': args.value},
    'baselines': {
        'control_mean_zero_change': {
            'results': baseline_results,
            'reported_metrics': {'pearson': 0.1},
        },
    },
    'training_loss': [float(1.0 / (index + 1)) for index in range(20)],
    'validation_loss': [float(1.1 / (index + 1)) for index in range(20)],
    'gene_names': ['A', 'B'],
    'perturbable_gene_order': ['A', 'B'],
    'split_membership': {
        'name': 'simulation',
        'seed': 1,
        'row_ids': ['row:0', 'row:1'],
        'condition_ids': ['A', 'B'],
        'eligible_test_conditions': ['A', 'B'],
    },
    'condition_target_mapping': [
        {
            'condition': 'A',
            'targets': ['A'],
            'target_indices': [0],
            'target_indicator': [1, 0],
            'mapped': True,
        },
        {
            'condition': 'B',
            'targets': ['B'],
            'target_indices': [1],
            'target_indicator': [0, 1],
            'mapped': True,
        },
    ],
    'identifier_coverage': 1.0,
    'target_audit': {
        'total_conditions': 2,
        'mapped_conditions': 2,
        'unmapped_conditions': [],
        'perturbable_gene_count': 2,
    },
    'model_identity': {
        'initialization_state': manifest(initialization),
        'checkpoint_state': manifest(checkpoint),
    },
    'graph_provenance': {
        'source': 'toy pinned graph fixture',
        **manifest(graph),
    },
}
args.output.write_text(json.dumps(payload), encoding='utf-8')
""",
        encoding="utf-8",
    )
    _git(repository, "add", "metric.py", "execute.py")
    _git(repository, "commit", "-m", "toy pinned GEARS fixture")
    commit = _git(repository, "rev-parse", "HEAD")
    _git(repository, "remote", "add", "origin", "https://github.com/snap-stanford/GEARS.git")

    plan_root = tmp_path / "plan"
    plan_root.mkdir()
    lock = plan_root / "environment.lock"
    write_complete_environment_lock(lock, require_direct_dependencies=False)
    dataset = plan_root / "positive_control.npz"
    dataset.write_bytes(b"hash-bound toy positive-control dataset")
    plan = {
        "schema_version": "1.0",
        "official_repository": {
            "url": "https://github.com/snap-stanford/GEARS",
            "commit": commit,
        },
        "environment_lock": {
            "source_id": lock.name,
            "sha256": file_sha256(lock),
        },
        "dataset": {
            "name": "toy_positive_control",
            "source_id": dataset.name,
            "sha256": file_sha256(dataset),
        },
        "split": {"name": "simulation", "seed": 1},
        "training_config": {
            "hidden_size": 64,
            "epochs": 20,
            "batch_size": 32,
            "test_batch_size": 128,
            "device_policy": "explicit_cli_interpreter",
        },
        "official_metric_source": {
            "source_id": metric_source.name,
            "sha256": file_sha256(metric_source),
            "function": "metric.compute_metrics",
        },
        "expected_metrics": {"pearson": 0.5},
        "tolerances": {
            "pearson": {
                "absolute": 0.01,
                "relative": 0.0,
                "justification": "Toy test freezes an absolute numerical tolerance.",
            }
        },
        "metric_directions": {"pearson": "higher_is_better"},
        "execution": {
            "official_reference": {
                "entrypoint": {
                    "scope": "repository",
                    "source_id": executor.name,
                    "sha256": file_sha256(executor),
                },
                "argv": [
                    "{python}",
                    "{entrypoint}",
                    "--output",
                    "{output}",
                    "--value",
                    "0.5",
                ],
            },
            "candidate": {
                "entrypoint": {
                    "scope": "repository",
                    "source_id": executor.name,
                    "sha256": file_sha256(executor),
                },
                "argv": [
                    "{python}",
                    "{entrypoint}",
                    "--output",
                    "{output}",
                    "--value",
                    str(candidate_value),
                ],
            },
        },
    }
    phase_metadata = {
        "official_reference": (
            "official_reference",
            "pinned_official_reference_run",
            "toy_official_reference_expected_values",
        ),
        "candidate": (
            "heldout_adaptation",
            "predeclared_heldout_adaptation_target",
            "toy_heldout_adaptation_expected_values",
        ),
    }
    for phase, (role, source_kind, source_id) in phase_metadata.items():
        specification = plan["execution"][phase]
        specification["phase_role"] = role
        specification["workflow_reuse_policy"] = "PINNED_OFFICIAL_WORKFLOW_REUSE_ALLOWED"
        provenance = {
            "schema_version": "1.0",
            "phase_role": role,
            "source_kind": source_kind,
            "source_id": source_id,
            "frozen_before_execution": True,
            "repository_url": plan["official_repository"]["url"],
            "commit": commit,
            "dataset_sha256": plan["dataset"]["sha256"],
            "split": plan["split"],
            "entrypoint_sha256": specification["entrypoint"]["sha256"],
            "argv_sha256": canonical_sha256(specification["argv"]),
            "expected_metrics": plan["expected_metrics"],
        }
        provenance["record_hash"] = canonical_sha256(provenance)
        provenance_path = plan_root / f"{phase}_expected_value_provenance.json"
        _write_json(provenance_path, provenance)
        specification["expected_value_provenance"] = {
            "source_id": provenance_path.name,
            "sha256": file_sha256(provenance_path),
        }
    plan["plan_hash"] = canonical_sha256(plan)
    plan_path = plan_root / "gears_execution_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan_path, repository


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _trust_anchor_path(package: Path) -> Path:
    return package.parent / f"{package.name}.gears_trust_anchor.json"


def _validate_with_pinned_trust(package: Path) -> dict[str, Any]:
    anchor = _trust_anchor_path(package)
    return validate_gears_positive_control(
        package / "gears_positive_control_manifest.json",
        trust_anchor_path=anchor,
        expected_trust_anchor_sha256=file_sha256(anchor),
    )


def _refresh_outer_hashes(package: Path) -> None:
    manifest_path = package / "gears_positive_control_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_path = package / "candidate_raw_evidence.json"
    manifest["candidate_raw_evidence"]["sha256"] = file_sha256(raw_path)
    audit_path = package / "candidate_execution_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["raw_metrics_sha256"] = file_sha256(raw_path)
    audit.pop("audit_hash")
    audit["audit_hash"] = canonical_sha256(audit)
    _write_json(audit_path, audit)
    manifest["candidate_execution_audit"]["sha256"] = file_sha256(audit_path)
    output_path = package / "candidate_output.json"
    output = json.loads(output_path.read_text(encoding="utf-8"))
    output["metadata"]["execution_audit_sha256"] = file_sha256(audit_path)
    _write_json(output_path, output)
    manifest["candidate_output"]["sha256"] = file_sha256(output_path)
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = canonical_sha256(manifest)
    _write_json(manifest_path, manifest)


def _refresh_candidate_audit_bindings(package: Path) -> None:
    manifest_path = package / "gears_positive_control_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit_path = package / "candidate_execution_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.pop("audit_hash", None)
    audit["audit_hash"] = canonical_sha256(audit)
    _write_json(audit_path, audit)
    manifest["candidate_execution_audit"]["sha256"] = file_sha256(audit_path)
    output_path = package / "candidate_output.json"
    output = json.loads(output_path.read_text(encoding="utf-8"))
    output["metadata"]["execution_audit_sha256"] = file_sha256(audit_path)
    _write_json(output_path, output)
    manifest["candidate_output"]["sha256"] = file_sha256(output_path)
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = canonical_sha256(manifest)
    _write_json(manifest_path, manifest)


def _refresh_environment_outer_hashes(package: Path) -> None:
    manifest_path = package / "gears_positive_control_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lock_path = package / "inputs" / "environment.lock"
    lock_hash = file_sha256(lock_path)
    plan_path = package / "inputs" / "execution_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["environment_lock"]["sha256"] = lock_hash
    plan.pop("plan_hash", None)
    plan["plan_hash"] = canonical_sha256(plan)
    _write_json(plan_path, plan)
    manifest["environment_lock"]["sha256"] = lock_hash
    manifest["execution_plan"]["sha256"] = file_sha256(plan_path)
    manifest["execution_plan_hash"] = plan["plan_hash"]
    for phase in ("official_reference", "candidate"):
        audit_path = package / f"{phase}_execution_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["environment_lock_sha256"] = lock_hash
        audit.pop("audit_hash", None)
        audit["audit_hash"] = canonical_sha256(audit)
        _write_json(audit_path, audit)
        manifest[f"{phase}_execution_audit"]["sha256"] = file_sha256(audit_path)
        output_path = package / f"{phase}_output.json"
        output = json.loads(output_path.read_text(encoding="utf-8"))
        output["metadata"]["environment_lock_sha256"] = lock_hash
        output["metadata"]["execution_audit_sha256"] = file_sha256(audit_path)
        _write_json(output_path, output)
        manifest[f"{phase}_output"]["sha256"] = file_sha256(output_path)
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = canonical_sha256(manifest)
    _write_json(manifest_path, manifest)


def _forge_self_consistent_environment_bundle(package: Path) -> None:
    lock_path = package / "inputs" / "environment.lock"
    lines = lock_path.read_text(encoding="utf-8").splitlines()
    lines.append("attacker-added-distribution==1.0")
    packages: dict[str, str] = {}
    for line in lines:
        if line and not line.startswith("#"):
            name, version = line.split("==", 1)
            packages[name.casefold().replace("_", "-").replace(".", "-")] = version
    package_set_hash = canonical_sha256(dict(sorted(packages.items())))
    header_prefix = "# complete-distribution-set-sha256: "
    lines = [
        f"{header_prefix}{package_set_hash}" if line.startswith(header_prefix) else line
        for line in lines
    ]
    lock_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    headers = {
        line[2:].split(": ", 1)[0]: line[2:].split(": ", 1)[1]
        for line in lines
        if line.startswith("# ") and ": " in line
    }

    manifest_path = package / "gears_positive_control_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lock_hash = file_sha256(lock_path)
    plan_path = package / "inputs" / "execution_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["environment_lock"]["sha256"] = lock_hash
    plan.pop("plan_hash", None)
    plan["plan_hash"] = canonical_sha256(plan)
    _write_json(plan_path, plan)
    manifest["environment_lock"]["sha256"] = lock_hash
    manifest["execution_plan"]["sha256"] = file_sha256(plan_path)
    manifest["execution_plan_hash"] = plan["plan_hash"]
    for phase in ("official_reference", "candidate"):
        audit_path = package / f"{phase}_execution_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["environment_lock_sha256"] = lock_hash
        audit["installed_locked_versions"] = dict(sorted(packages.items()))
        audit["runtime_environment_headers"] = headers
        audit.pop("audit_hash", None)
        audit["audit_hash"] = canonical_sha256(audit)
        _write_json(audit_path, audit)
        manifest[f"{phase}_execution_audit"]["sha256"] = file_sha256(audit_path)
        output_path = package / f"{phase}_output.json"
        output = json.loads(output_path.read_text(encoding="utf-8"))
        output["metadata"]["environment_lock_sha256"] = lock_hash
        output["metadata"]["execution_audit_sha256"] = file_sha256(audit_path)
        _write_json(output_path, output)
        manifest[f"{phase}_output"]["sha256"] = file_sha256(output_path)
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = canonical_sha256(manifest)
    _write_json(manifest_path, manifest)


def _forge_self_consistent_target_and_graph_bundle(package: Path) -> None:
    phase = "candidate"
    raw_path = package / f"{phase}_raw_evidence.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["perturbable_gene_order"].reverse()
    perturbable_index = {gene: index for index, gene in enumerate(raw["perturbable_gene_order"])}
    for mapping in raw["condition_target_mapping"]:
        mapping["target_indices"] = [perturbable_index[target] for target in mapping["targets"]]
        mapping["target_indicator"] = [0] * len(raw["perturbable_gene_order"])
        for index in mapping["target_indices"]:
            mapping["target_indicator"][index] = 1

    graph_path = package / f"{phase}_graph_state.npz"
    with np.load(graph_path, allow_pickle=False) as archive:
        graph_arrays = {name: archive[name] for name in archive.files}
    graph_arrays["__cbac_perturbable_gene_order_utf8"] = np.frombuffer(
        "\0".join(raw["perturbable_gene_order"]).encode("utf-8"), dtype=np.uint8
    )
    np.savez_compressed(graph_path, **graph_arrays)
    graph_manifest = _load_npz_manifest(graph_path, "attack.candidate.graph_state")
    raw["graph_provenance"] = {
        "source": raw["graph_provenance"]["source"],
        "sha256": graph_manifest["file_sha256"],
        "arrays": graph_manifest["arrays"],
    }
    _write_json(raw_path, raw)

    manifest_path = package / "gears_positive_control_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[f"{phase}_raw_evidence"]["sha256"] = file_sha256(raw_path)
    manifest[f"{phase}_graph_state"]["sha256"] = file_sha256(graph_path)
    audit_path = package / f"{phase}_execution_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["raw_metrics_sha256"] = file_sha256(raw_path)
    audit["perturbable_gene_order_hash"] = canonical_sha256(raw["perturbable_gene_order"])
    audit["condition_target_mapping_hash"] = canonical_sha256(raw["condition_target_mapping"])
    audit["graph_state_sha256"] = graph_manifest["file_sha256"]
    audit["graph_array_manifest_hash"] = canonical_sha256(graph_manifest)
    audit.pop("audit_hash", None)
    audit["audit_hash"] = canonical_sha256(audit)
    _write_json(audit_path, audit)
    manifest[f"{phase}_execution_audit"]["sha256"] = file_sha256(audit_path)
    output_path = package / f"{phase}_output.json"
    output = json.loads(output_path.read_text(encoding="utf-8"))
    output["metadata"]["execution_audit_sha256"] = file_sha256(audit_path)
    _write_json(output_path, output)
    manifest[f"{phase}_output"]["sha256"] = file_sha256(output_path)
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = canonical_sha256(manifest)
    _write_json(manifest_path, manifest)


def _empty_raw(_: dict[str, Any]) -> dict[str, Any]:
    return {}


def _change_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    payload["results"]["pred"][0][0] = 0.7
    return payload


def _reorder_genes(payload: dict[str, Any]) -> dict[str, Any]:
    payload["gene_names"].reverse()
    return payload


def _remove_condition(payload: dict[str, Any]) -> dict[str, Any]:
    for field in payload["results"]:
        payload["results"][field] = payload["results"][field][:-1]
    for field in payload["baselines"]["control_mean_zero_change"]["results"]:
        payload["baselines"]["control_mean_zero_change"]["results"][field] = payload["baselines"][
            "control_mean_zero_change"
        ]["results"][field][:-1]
    payload["split_membership"]["row_ids"] = payload["split_membership"]["row_ids"][:-1]
    payload["split_membership"]["condition_ids"] = payload["split_membership"]["condition_ids"][:-1]
    return payload


def _change_target_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    payload["condition_target_mapping"][0]["targets"] = ["B"]
    return payload


def _change_target_indices(payload: dict[str, Any]) -> dict[str, Any]:
    payload["condition_target_mapping"][0]["target_indices"] = [1]
    return payload


def _change_target_indicator(payload: dict[str, Any]) -> dict[str, Any]:
    payload["condition_target_mapping"][0]["target_indicator"] = [0, 1]
    return payload


def _change_perturbable_gene_order(payload: dict[str, Any]) -> dict[str, Any]:
    payload["perturbable_gene_order"].reverse()
    return payload


def _constant_between_condition_predictions(payload: dict[str, Any]) -> dict[str, Any]:
    payload["results"]["pred"][1] = payload["results"]["pred"][0]
    return payload


def _shorten_loss(payload: dict[str, Any]) -> dict[str, Any]:
    payload["validation_loss"] = payload["validation_loss"][:-1]
    return payload


def _alter_split_member(payload: dict[str, Any]) -> dict[str, Any]:
    payload["split_membership"]["condition_ids"][0] = "B"
    return payload


def _alter_baseline_vector(payload: dict[str, Any]) -> dict[str, Any]:
    payload["baselines"]["control_mean_zero_change"]["results"]["pred"][0][0] = 0.3
    return payload


def test_executed_positive_control_pass_releases_eligible_branch(tmp_path: Path) -> None:
    plan, repository = _reproduction_fixture(tmp_path, candidate_value=0.505)

    registry = execute_gears_reproduction(
        plan,
        repository,
        tmp_path / "package",
        python_executable=Path(sys.executable),
    )

    assert registry["status"] == "RELEASED"
    assert registry["execution_outcome"] == "PASS"
    assert registry["eligibility_decision"] == "ELIGIBLE_POSITIVE_CONTROL_PASSED"
    assert registry["ranking_eligibility"] is True
    assert registry["failures"] == []


def test_valid_executed_failure_is_resolved_exclusion_not_withheld(tmp_path: Path) -> None:
    plan, repository = _reproduction_fixture(tmp_path, candidate_value=0.8)

    registry = execute_gears_reproduction(
        plan,
        repository,
        tmp_path / "package",
        python_executable=Path(sys.executable),
    )
    lock = _external_lock("GEARS-POSITIVE-CONTROL-VALIDATION", registry)

    assert registry["status"] == "RELEASED"
    assert registry["execution_outcome"] == "FAIL"
    assert registry["eligibility_decision"] == "EXCLUDED_FAILED_POSITIVE_CONTROL"
    assert registry["ranking_eligibility"] is False
    assert registry["failures"] == []
    assert registry["outcome_reason_codes"] == ["GEARS_METRIC_OUTSIDE_TOLERANCE:pearson"]
    assert lock["status"] == "RELEASED"


def test_final_validator_requires_detached_anchor_and_caller_pinned_sha256(
    tmp_path: Path,
) -> None:
    plan, repository = _reproduction_fixture(tmp_path, candidate_value=0.5)
    package = tmp_path / "package"
    execute_gears_reproduction(
        plan,
        repository,
        package,
        python_executable=Path(sys.executable),
    )

    registry = validate_gears_positive_control(package / "gears_positive_control_manifest.json")

    assert registry["status"] == "WITHHELD"
    assert "GEARS_TRUST_ANCHOR_MISSING" in {
        failure["reason_code"] for failure in registry["failures"]
    }


def test_clean_package_revalidates_against_detached_pinned_trust(tmp_path: Path) -> None:
    plan, repository = _reproduction_fixture(tmp_path, candidate_value=0.5)
    package = tmp_path / "package"
    execute_gears_reproduction(
        plan,
        repository,
        package,
        python_executable=Path(sys.executable),
    )

    registry = _validate_with_pinned_trust(package)

    assert registry["status"] == "RELEASED"
    assert registry["provenance_bindings"]["trust_model"] == (
        "caller_pinned_sha256_over_detached_anchor_and_external_sources"
    )
    diagnostics = registry["semantic_diagnostics"]["candidate"]
    assert diagnostics["minimum_directional_baseline_improvement"] > 0
    assert diagnostics["training_loss_improvement"] > 0
    assert diagnostics["validation_loss_improvement"] > 0


def test_synchronized_environment_bundle_forgery_fails_detached_trust(
    tmp_path: Path,
) -> None:
    plan, repository = _reproduction_fixture(tmp_path, candidate_value=0.5)
    package = tmp_path / "package"
    execute_gears_reproduction(
        plan,
        repository,
        package,
        python_executable=Path(sys.executable),
    )
    _forge_self_consistent_environment_bundle(package)

    registry = _validate_with_pinned_trust(package)
    reasons = {failure["reason_code"] for failure in registry["failures"]}

    assert registry["status"] == "WITHHELD"
    assert "GEARS_TRUSTED_MANIFEST_SHA256_MISMATCH" in reasons
    assert "GEARS_PACKAGE_DIFFERS_FROM_TRUSTED_SOURCE" in reasons
    assert reasons <= {
        "GEARS_TRUSTED_MANIFEST_SHA256_MISMATCH",
        "GEARS_TRUSTED_MANIFEST_HASH_MISMATCH",
        "GEARS_PACKAGE_DIFFERS_FROM_TRUSTED_SOURCE",
    }


def test_synchronized_target_and_graph_forgery_fails_detached_trust(
    tmp_path: Path,
) -> None:
    plan, repository = _reproduction_fixture(tmp_path, candidate_value=0.5)
    package = tmp_path / "package"
    execute_gears_reproduction(
        plan,
        repository,
        package,
        python_executable=Path(sys.executable),
    )
    _forge_self_consistent_target_and_graph_bundle(package)

    registry = _validate_with_pinned_trust(package)
    reasons = {failure["reason_code"] for failure in registry["failures"]}

    assert registry["status"] == "WITHHELD"
    assert "GEARS_TRUSTED_MANIFEST_SHA256_MISMATCH" in reasons
    assert "GEARS_PACKAGE_DIFFERS_FROM_TRUSTED_SOURCE" in reasons
    assert reasons <= {
        "GEARS_TRUSTED_MANIFEST_SHA256_MISMATCH",
        "GEARS_TRUSTED_MANIFEST_HASH_MISMATCH",
        "GEARS_PACKAGE_DIFFERS_FROM_TRUSTED_SOURCE",
    }


def test_attacker_cannot_rewrite_anchor_when_caller_retains_original_sha256(
    tmp_path: Path,
) -> None:
    plan, repository = _reproduction_fixture(tmp_path, candidate_value=0.5)
    package = tmp_path / "package"
    execute_gears_reproduction(
        plan,
        repository,
        package,
        python_executable=Path(sys.executable),
    )
    anchor_path = _trust_anchor_path(package)
    caller_pinned_sha256 = file_sha256(anchor_path)
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchor["evidence_manifest_sha256"] = "0" * 64
    anchor.pop("anchor_hash")
    anchor["anchor_hash"] = canonical_sha256(anchor)
    _write_json(anchor_path, anchor)

    registry = validate_gears_positive_control(
        package / "gears_positive_control_manifest.json",
        trust_anchor_path=anchor_path,
        expected_trust_anchor_sha256=caller_pinned_sha256,
    )

    assert registry["status"] == "WITHHELD"
    assert "GEARS_TRUST_ANCHOR_SHA256_MISMATCH" in {
        failure["reason_code"] for failure in registry["failures"]
    }


def test_anchor_stored_inside_evidence_bundle_is_untrusted(tmp_path: Path) -> None:
    plan, repository = _reproduction_fixture(tmp_path, candidate_value=0.5)
    package = tmp_path / "package"
    execute_gears_reproduction(
        plan,
        repository,
        package,
        python_executable=Path(sys.executable),
    )
    external_anchor = _trust_anchor_path(package)
    internal_anchor = package / "attacker_controlled_anchor.json"
    internal_anchor.write_bytes(external_anchor.read_bytes())

    registry = validate_gears_positive_control(
        package / "gears_positive_control_manifest.json",
        trust_anchor_path=internal_anchor,
        expected_trust_anchor_sha256=file_sha256(internal_anchor),
    )

    assert registry["status"] == "WITHHELD"
    assert "GEARS_TRUST_ANCHOR_INSIDE_EVIDENCE_BUNDLE" in {
        failure["reason_code"] for failure in registry["failures"]
    }


def test_executor_rejects_anchor_inside_bundle_before_running_phases(tmp_path: Path) -> None:
    plan, repository = _reproduction_fixture(tmp_path, candidate_value=0.5)
    package = tmp_path / "package"

    with pytest.raises(
        RevisionProtocolError,
        match="GEARS_TRUST_ANCHOR_INSIDE_EVIDENCE_BUNDLE",
    ):
        execute_gears_reproduction(
            plan,
            repository,
            package,
            python_executable=Path(sys.executable),
            trust_anchor_output_path=package / "anchor.json",
        )

    assert not package.exists()


def test_changed_external_trusted_source_is_withheld(tmp_path: Path) -> None:
    plan, repository = _reproduction_fixture(tmp_path, candidate_value=0.5)
    package = tmp_path / "package"
    execute_gears_reproduction(
        plan,
        repository,
        package,
        python_executable=Path(sys.executable),
    )
    anchor_path = _trust_anchor_path(package)
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    trusted_dataset = anchor_path.parent / anchor["trusted_files"]["dataset"]["source_id"]
    trusted_dataset.write_bytes(b"attacker changed detached dataset copy")

    registry = validate_gears_positive_control(
        package / "gears_positive_control_manifest.json",
        trust_anchor_path=anchor_path,
        expected_trust_anchor_sha256=file_sha256(anchor_path),
    )

    assert registry["status"] == "WITHHELD"
    assert "GEARS_TRUSTED_FILE_SHA256_MISMATCH" in {
        failure["reason_code"] for failure in registry["failures"]
    }


def test_tampered_candidate_output_withholds_invalid_evidence(tmp_path: Path) -> None:
    plan, repository = _reproduction_fixture(tmp_path, candidate_value=0.5)
    package = tmp_path / "package"
    execute_gears_reproduction(
        plan,
        repository,
        package,
        python_executable=Path(sys.executable),
    )
    candidate = package / "candidate_output.json"
    candidate.write_text("{}", encoding="utf-8")

    registry = _validate_with_pinned_trust(package)

    assert registry["status"] == "WITHHELD"
    assert registry["eligibility_decision"] == "WITHHELD_INVALID_OR_INCOMPLETE_EVIDENCE"
    assert "GEARS_BOUND_FILE_HASH_MISMATCH" in {
        failure["reason_code"] for failure in registry["failures"]
    }


@pytest.mark.parametrize(
    "mutation",
    [
        _empty_raw,
        _change_prediction,
        _reorder_genes,
        _remove_condition,
        _change_target_mapping,
        _change_target_indices,
        _change_target_indicator,
        _change_perturbable_gene_order,
        _constant_between_condition_predictions,
        _shorten_loss,
        _alter_split_member,
        _alter_baseline_vector,
    ],
    ids=[
        "empty-json",
        "changed-prediction",
        "reordered-genes",
        "removed-condition",
        "changed-target-mapping",
        "changed-target-indices",
        "changed-target-indicator",
        "changed-perturbable-gene-order",
        "constant-between-condition-predictions",
        "shortened-loss",
        "altered-split-member",
        "altered-baseline-vector",
    ],
)
def test_semantic_tampering_with_refreshed_outer_hashes_is_withheld(
    tmp_path: Path, mutation: Callable[[dict[str, Any]], dict[str, Any]]
) -> None:
    plan, repository = _reproduction_fixture(tmp_path, candidate_value=0.5)
    package = tmp_path / "package"
    execute_gears_reproduction(
        plan,
        repository,
        package,
        python_executable=Path(sys.executable),
    )
    raw_path = package / "candidate_raw_evidence.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    _write_json(raw_path, mutation(raw))
    _refresh_outer_hashes(package)

    registry = _validate_with_pinned_trust(package)

    assert registry["status"] == "WITHHELD"
    assert registry["evidence_status"] == "INVALID_OR_INCOMPLETE"
    assert registry["ranking_eligibility"] is False


def test_placeholder_example_is_intentionally_withheld() -> None:
    example = Path(__file__).parents[1] / "manifests" / "gears_positive_control.example.json"

    registry = validate_gears_positive_control(example)

    assert registry["status"] == "WITHHELD"
    assert "GEARS_MANIFEST_PLACEHOLDER_PRESENT" in {
        failure["reason_code"] for failure in registry["failures"]
    }


def test_gears_h5ad_runtime_requires_h5py_in_complete_lock(tmp_path: Path) -> None:
    lock = tmp_path / "complete.lock"
    write_complete_environment_lock(lock, require_direct_dependencies=False)
    lock.write_text(
        "\n".join(
            line
            for line in lock.read_text(encoding="utf-8").splitlines()
            if not line.startswith("h5py==")
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RevisionProtocolError,
        match="GEARS_ENVIRONMENT_LOCK_(PACKAGE_SET_MISMATCH|H5PY_MISSING)",
    ):
        _verify_environment(Path(sys.executable), lock, require_h5py=True)


def _forge_lock_header(path: Path, header: str) -> None:
    prefix = f"# {header}: "
    lines = path.read_text(encoding="utf-8").splitlines()
    matches = [index for index, line in enumerate(lines) if line.startswith(prefix)]
    assert len(matches) == 1
    lines[matches[0]] = f"{prefix}FORGED"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.parametrize("header", ENVIRONMENT_LOCK_HEADER_ORDER)
def test_gears_execution_rejects_every_forged_environment_header(
    tmp_path: Path, header: str
) -> None:
    lock = tmp_path / "complete.lock"
    write_complete_environment_lock(lock, require_direct_dependencies=False)
    _forge_lock_header(lock, header)

    with pytest.raises(
        RevisionProtocolError,
        match="GEARS_ENVIRONMENT_LOCK_RUNTIME_HEADER_MISMATCH",
    ):
        _verify_environment(Path(sys.executable), lock, require_h5py=False)


@pytest.mark.parametrize("header", ENVIRONMENT_LOCK_HEADER_ORDER)
def test_gears_final_validator_rejects_every_forged_environment_header_with_outer_hashes(
    tmp_path: Path, header: str
) -> None:
    plan, repository = _reproduction_fixture(tmp_path, candidate_value=0.5)
    package = tmp_path / "package"
    execute_gears_reproduction(
        plan,
        repository,
        package,
        python_executable=Path(sys.executable),
    )
    _forge_lock_header(package / "inputs" / "environment.lock", header)
    _refresh_environment_outer_hashes(package)

    registry = _validate_with_pinned_trust(package)

    assert registry["status"] == "WITHHELD"
    assert registry["evidence_status"] == "INVALID_OR_INCOMPLETE"
    assert registry["ranking_eligibility"] is False


def test_gears_final_validator_rejects_forged_perturbable_order_hash_with_outer_hashes(
    tmp_path: Path,
) -> None:
    plan, repository = _reproduction_fixture(tmp_path, candidate_value=0.5)
    package = tmp_path / "package"
    execute_gears_reproduction(
        plan,
        repository,
        package,
        python_executable=Path(sys.executable),
    )
    audit_path = package / "candidate_execution_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["perturbable_gene_order_hash"] = "0" * 64
    _write_json(audit_path, audit)
    _refresh_candidate_audit_bindings(package)

    registry = _validate_with_pinned_trust(package)

    assert registry["status"] == "WITHHELD"
    assert registry["evidence_status"] == "INVALID_OR_INCOMPLETE"
    assert registry["ranking_eligibility"] is False


def test_gears_final_validator_rejects_coordinated_raw_perturbable_list_forgery(
    tmp_path: Path,
) -> None:
    plan, repository = _reproduction_fixture(tmp_path, candidate_value=0.5)
    package = tmp_path / "package"
    execute_gears_reproduction(
        plan,
        repository,
        package,
        python_executable=Path(sys.executable),
    )
    raw_path = package / "candidate_raw_evidence.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["perturbable_gene_order"].reverse()
    perturbable_index = {gene: index for index, gene in enumerate(raw["perturbable_gene_order"])}
    for mapping in raw["condition_target_mapping"]:
        mapping["target_indices"] = [perturbable_index[target] for target in mapping["targets"]]
        mapping["target_indicator"] = [0] * len(raw["perturbable_gene_order"])
        for index in mapping["target_indices"]:
            mapping["target_indicator"][index] = 1
    _write_json(raw_path, raw)
    _refresh_outer_hashes(package)
    audit_path = package / "candidate_execution_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["perturbable_gene_order_hash"] = canonical_sha256(raw["perturbable_gene_order"])
    audit["condition_target_mapping_hash"] = canonical_sha256(raw["condition_target_mapping"])
    _write_json(audit_path, audit)
    _refresh_candidate_audit_bindings(package)

    registry = _validate_with_pinned_trust(package)

    assert registry["status"] == "WITHHELD"
    assert registry["evidence_status"] == "INVALID_OR_INCOMPLETE"
    assert registry["ranking_eligibility"] is False


def test_gears_rejects_identical_phase_workflow_without_explicit_reuse_policy(
    tmp_path: Path,
) -> None:
    plan_path, repository = _reproduction_fixture(tmp_path, candidate_value=0.5)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["execution"]["candidate"]["workflow_reuse_policy"] = "DISTINCT_PHASE_WORKFLOW_REQUIRED"
    plan.pop("plan_hash", None)
    plan["plan_hash"] = canonical_sha256(plan)
    _write_json(plan_path, plan)

    with pytest.raises(
        RevisionProtocolError,
        match="GEARS_IDENTICAL_PHASE_WORKFLOW_UNDECLARED",
    ):
        execute_gears_reproduction(
            plan_path,
            repository,
            tmp_path / "package",
            python_executable=Path(sys.executable),
        )


def test_gears_rejects_reused_expected_value_provenance_across_phase_roles(
    tmp_path: Path,
) -> None:
    plan_path, repository = _reproduction_fixture(tmp_path, candidate_value=0.5)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["execution"]["candidate"]["expected_value_provenance"] = plan["execution"][
        "official_reference"
    ]["expected_value_provenance"]
    plan.pop("plan_hash", None)
    plan["plan_hash"] = canonical_sha256(plan)
    _write_json(plan_path, plan)

    with pytest.raises(
        RevisionProtocolError,
        match="GEARS_EXPECTED_VALUE_PROVENANCE_INVALID",
    ):
        execute_gears_reproduction(
            plan_path,
            repository,
            tmp_path / "package",
            python_executable=Path(sys.executable),
        )

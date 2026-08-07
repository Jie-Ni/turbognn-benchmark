from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cbac_revision.artifacts import canonical_sha256, file_sha256
from cbac_revision.errors import RevisionProtocolError
from cbac_revision.external_validation import (
    ComparatorScientificGateFailed,
    read_external_comparator_family,
    validate_generic_comparator_manifest,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )


def _self_hash(payload: dict[str, Any], field: str) -> dict[str, Any]:
    payload[field] = canonical_sha256(payload)
    return payload


def _binding(path: Path) -> dict[str, str]:
    return {"source_id": path.name, "sha256": file_sha256(path)}


def _generic_manifest(
    root: Path,
    comparator: str = "scGPT",
    *,
    scientifically_valid: bool = True,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    prefix = comparator.casefold()
    genes = ["A", "B", "C", "D"]
    train_conditions = ["train_A"]
    validation_conditions = ["validation_B"]
    test_conditions = ["test_C", "test_D"]
    truth = np.asarray([[1.0, -0.2, 0.5, -0.7], [0.1, 0.9, -0.4, 0.8]], dtype=np.float64)
    if scientifically_valid:
        prediction = truth + np.asarray(
            [[0.02, -0.01, 0.01, 0.02], [-0.01, 0.02, 0.01, -0.02]],
            dtype=np.float64,
        )
    else:
        prediction = truth + np.asarray(
            [[8.0, -7.0, 9.0, -8.0], [-9.0, 8.0, -7.0, 9.0]], dtype=np.float64
        )
    baseline = np.zeros_like(truth)
    training_vectors = np.zeros((1, len(genes)), dtype=np.float64)
    training_loss = np.asarray([1.0, 0.7, 0.4], dtype=np.float64)
    validation_loss = np.asarray([1.2, 0.8, 0.5], dtype=np.float64)

    paths: dict[str, Path] = {}
    for role, values in (
        ("prediction_vectors", prediction),
        ("truth_vectors", truth),
        ("baseline_vectors", baseline),
        ("baseline_training_vectors", training_vectors),
        ("training_loss", training_loss),
        ("validation_loss", validation_loss),
    ):
        path = root / f"{prefix}_{role}.npy"
        np.save(path, values, allow_pickle=False)
        paths[role] = path

    paths["gene_order"] = root / f"{prefix}_gene_order.json"
    _write_json(paths["gene_order"], genes)
    paths["vector_row_conditions"] = root / f"{prefix}_vector_row_conditions.json"
    _write_json(paths["vector_row_conditions"], test_conditions)
    paths["baseline_training_row_conditions"] = (
        root / f"{prefix}_baseline_training_row_conditions.json"
    )
    _write_json(paths["baseline_training_row_conditions"], train_conditions)

    split = _self_hash(
        {
            "schema_version": "1.0",
            "split_id": f"{prefix}_target_disjoint_split",
            "training_conditions": train_conditions,
            "validation_conditions": validation_conditions,
            "test_conditions": test_conditions,
            "target_disjoint": True,
        },
        "manifest_sha256",
    )
    paths["split"] = root / f"{prefix}_split.json"
    _write_json(paths["split"], split)
    target_mapping = _self_hash(
        {
            "schema_version": "1.0",
            "condition_targets": {
                "train_A": ["A"],
                "validation_B": ["B"],
                "test_C": ["C"],
                "test_D": ["D"],
            },
            "gene_order_sha256": canonical_sha256(tuple(genes)),
        },
        "manifest_sha256",
    )
    paths["target_mapping"] = root / f"{prefix}_target_mapping.json"
    _write_json(paths["target_mapping"], target_mapping)
    dataset = _self_hash(
        {
            "schema_version": "1.0",
            "dataset_id": f"{prefix}_toy_dataset",
            "cell_line": "K562",
            "assay": "Perturb-seq",
            "vector_shape": list(truth.shape),
            "test_row_conditions": test_conditions,
            "test_row_conditions_sha256": canonical_sha256(tuple(test_conditions)),
            "gene_order_sha256": canonical_sha256(tuple(genes)),
            "truth_vectors_file_sha256": file_sha256(paths["truth_vectors"]),
            "target_mapping_manifest_sha256": target_mapping["manifest_sha256"],
        },
        "manifest_sha256",
    )
    paths["dataset"] = root / f"{prefix}_dataset.json"
    _write_json(paths["dataset"], dataset)
    baseline_provenance = _self_hash(
        {
            "schema_version": "1.0",
            "baseline_type": "zero_control_delta",
            "fit_scope": "training_conditions_only_no_test_outcome",
            "training_vectors_file_sha256": file_sha256(paths["baseline_training_vectors"]),
            "training_row_conditions_file_sha256": file_sha256(
                paths["baseline_training_row_conditions"]
            ),
            "gene_order_sha256": canonical_sha256(tuple(genes)),
            "split_file_sha256": file_sha256(paths["split"]),
            "expected_baseline_vectors_file_sha256": file_sha256(paths["baseline_vectors"]),
        },
        "manifest_sha256",
    )
    paths["baseline_provenance"] = root / f"{prefix}_baseline_provenance.json"
    _write_json(paths["baseline_provenance"], baseline_provenance)

    flat_truth = truth.reshape(-1)
    flat_prediction = prediction.reshape(-1)
    residual = flat_truth - flat_prediction
    metrics = {
        "pearson_r": float(np.corrcoef(flat_truth, flat_prediction)[0, 1]),
        "mse": float(np.mean(np.square(residual))),
        "mae": float(np.mean(np.abs(residual))),
    }
    commit = "1" * 40
    repository = f"https://example.org/{prefix}/official"
    expected_provenance = {
        "schema_version": "1.0",
        "comparator": comparator,
        "repository_url": repository,
        "commit": commit,
        "dataset_sha256": file_sha256(paths["dataset"]),
        "split_sha256": file_sha256(paths["split"]),
        "target_mapping_sha256": file_sha256(paths["target_mapping"]),
        "prediction_vectors_sha256": file_sha256(paths["prediction_vectors"]),
        "truth_vectors_sha256": file_sha256(paths["truth_vectors"]),
        "vector_row_conditions_sha256": file_sha256(paths["vector_row_conditions"]),
        "training_loss_sha256": file_sha256(paths["training_loss"]),
        "validation_loss_sha256": file_sha256(paths["validation_loss"]),
        "gene_order_sha256": file_sha256(paths["gene_order"]),
        "baseline_vectors_sha256": file_sha256(paths["baseline_vectors"]),
        "baseline_provenance_sha256": file_sha256(paths["baseline_provenance"]),
        "baseline_training_vectors_sha256": file_sha256(paths["baseline_training_vectors"]),
        "baseline_training_row_conditions_sha256": file_sha256(
            paths["baseline_training_row_conditions"]
        ),
        "expected_metrics": {
            metric: {
                "expected_value": value,
                "direction": "higher_is_better" if metric == "pearson_r" else "lower_is_better",
                "absolute_tolerance": 1e-12,
                "relative_tolerance": 0.0,
                "justification": "Frozen toy expected value for direct validator testing.",
                "official_independent_source": "toy_archived_reference",
            }
            for metric, value in metrics.items()
        },
        "loss_contract": {
            "minimum_history_length": 3,
            "minimum_training_improvement": 0.1,
            "minimum_validation_improvement": 0.1,
        },
        "source_id": f"{prefix}_frozen_expected_values",
        "source_kind": "official_release_or_independent_archived_reference",
        "frozen_before_execution": True,
    }
    _self_hash(expected_provenance, "provenance_sha256")
    paths["independent_expected_value_provenance"] = root / f"{prefix}_expected_provenance.json"
    _write_json(paths["independent_expected_value_provenance"], expected_provenance)

    manifest = _self_hash(
        {
            "schema_version": "1.0",
            "comparator": comparator,
            "official_repository": repository,
            "pinned_commit": commit,
            "bindings": {role: _binding(paths[role]) for role in sorted(paths)},
        },
        "manifest_sha256",
    )
    path = root / f"{prefix}_manifest.json"
    _write_json(path, manifest)
    return path


def _exclusion(root: Path, comparator: str) -> Path:
    payload = _self_hash(
        {
            "schema_version": "1.0",
            "comparator": comparator,
            "decision": "EXCLUDED",
            "claim_deleted": True,
            "reason_code": "NO_VALID_ARCHIVED_VECTORS",
            "rationale": "No complete frozen vector evidence is available.",
            "claim_locations_removed": ["main", "supplement", "response"],
            "frozen_date": "2026-08-07",
        },
        "manifest_sha256",
    )
    path = root / f"{comparator.casefold()}_exclusion.json"
    _write_json(path, payload)
    return path


def _family(root: Path, specifications: dict[str, dict[str, Any]]) -> Path:
    payload = _self_hash(
        {"schema_version": "1.0", "members": specifications},
        "manifest_sha256",
    )
    path = root / "external_family.json"
    _write_json(path, payload)
    return path


def _exclude_spec(path: Path) -> dict[str, str]:
    return {"mode": "exclude", **_binding(path)}


def test_generic_full_vector_evidence_passes_and_tamper_is_fail_closed(tmp_path: Path) -> None:
    manifest = _generic_manifest(tmp_path)
    member = validate_generic_comparator_manifest(manifest, "scGPT")
    assert member["decision"] == "PASS"
    assert member["recomputed_metrics"]["mse"] < member["recomputed_baseline_metrics"]["mse"]

    prediction = tmp_path / "scgpt_prediction_vectors.npy"
    values = np.load(prediction, allow_pickle=False)
    values[0, 0] += 1.0
    np.save(prediction, values, allow_pickle=False)
    with pytest.raises(RevisionProtocolError, match="failed hash validation"):
        validate_generic_comparator_manifest(manifest, "scGPT")


def test_generic_valid_evidence_with_failed_scientific_gate_is_excluded(tmp_path: Path) -> None:
    generic = _generic_manifest(tmp_path, scientifically_valid=False)
    with pytest.raises(ComparatorScientificGateFailed, match="baseline"):
        validate_generic_comparator_manifest(generic, "scGPT")

    gears = _exclusion(tmp_path, "GEARS")
    geneformer = _exclusion(tmp_path, "Geneformer")
    family = _family(
        tmp_path,
        {
            "GEARS": _exclude_spec(gears),
            "scGPT": {"mode": "validate", **_binding(generic)},
            "Geneformer": _exclude_spec(geneformer),
        },
    )
    registry = read_external_comparator_family(family, expected_manifest_sha256=file_sha256(family))
    assert registry["status"] == "RELEASED"
    assert registry["member_decisions"]["scGPT"] == "EXCLUDED"
    assert registry["scientific_gate_failed_members"] == ["scGPT"]
    assert "scGPT" not in registry["claim_deleted_members"]


def test_reason_coded_claim_deletions_complete_family_but_naked_missing_withholds(
    tmp_path: Path,
) -> None:
    exclusions = {
        comparator: _exclusion(tmp_path, comparator)
        for comparator in ("GEARS", "scGPT", "Geneformer")
    }
    complete = _family(
        tmp_path,
        {comparator: _exclude_spec(path) for comparator, path in exclusions.items()},
    )
    released = read_external_comparator_family(
        complete, expected_manifest_sha256=file_sha256(complete)
    )
    assert released["status"] == "RELEASED"
    assert released["claim_deleted_members"] == ["GEARS", "scGPT", "Geneformer"]

    missing = _family(
        tmp_path,
        {
            "GEARS": _exclude_spec(exclusions["GEARS"]),
            "scGPT": {"mode": "validate", "source_id": "missing.json", "sha256": "0" * 64},
            "Geneformer": _exclude_spec(exclusions["Geneformer"]),
        },
    )
    withheld = read_external_comparator_family(
        missing, expected_manifest_sha256=file_sha256(missing)
    )
    assert withheld["status"] == "WITHHELD"
    assert withheld["member_decisions"]["scGPT"] == "WITHHELD"


def test_gears_family_route_passes_detached_anchor_and_caller_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gears_source = tmp_path / "gears_manifest.json"
    _write_json(gears_source, {"toy": "bound"})
    anchor = tmp_path / "gears_anchor.json"
    _write_json(anchor, {"detached": True})
    scgpt = _exclusion(tmp_path, "scGPT")
    geneformer = _exclusion(tmp_path, "Geneformer")
    observed: dict[str, Any] = {}

    def fake_gears(
        path: Path, *, trust_anchor_path: Path, expected_trust_anchor_sha256: str
    ) -> dict[str, Any]:
        observed.update(
            path=path,
            trust_anchor_path=trust_anchor_path,
            expected_trust_anchor_sha256=expected_trust_anchor_sha256,
        )
        return {
            "comparator": "GEARS",
            "decision": "PASS",
            "scientific_validity": "VALID",
            "claim_deleted": False,
        }

    monkeypatch.setattr("cbac_revision.external_validation.validate_gears_member", fake_gears)
    family = _family(
        tmp_path,
        {
            "GEARS": {
                "mode": "validate",
                **_binding(gears_source),
                "trust_anchor_source_id": anchor.name,
                "trust_anchor_sha256": file_sha256(anchor),
            },
            "scGPT": _exclude_spec(scgpt),
            "Geneformer": _exclude_spec(geneformer),
        },
    )
    registry = read_external_comparator_family(family, expected_manifest_sha256=file_sha256(family))
    assert registry["status"] == "RELEASED"
    assert registry["member_decisions"]["GEARS"] == "PASS"
    assert observed == {
        "path": gears_source.resolve(),
        "trust_anchor_path": anchor.resolve(),
        "expected_trust_anchor_sha256": file_sha256(anchor),
    }

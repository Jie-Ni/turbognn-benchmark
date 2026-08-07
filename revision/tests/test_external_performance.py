from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cbac_revision.artifacts import canonical_sha256, file_sha256
from cbac_revision.external_performance import (
    DATASETS,
    ESTIMAND,
    EVIDENCE_FILE_ROLES,
    HVG_SCALES,
    METRIC_DEFINITION,
    PRIMARY_PANEL_ID,
    REFERENCE_MODEL_ID,
    combine_external_validity_and_performance,
    release_external_performance_family,
)

PASSING_COMPARATOR = "scGPT"
LINKED_DATASET = "adamson"
LINKED_HVG = 1000


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def _binding(path: Path) -> dict[str, str]:
    return {"source_id": path.name, "sha256": file_sha256(path)}


def _finalize(payload: dict[str, Any], field: str) -> dict[str, Any]:
    payload[field] = canonical_sha256(payload)
    return payload


def _make_raw_scale_evidence(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    scale_evidence: dict[str, dict[str, dict[str, Any]]] = {}
    for hvg in HVG_SCALES:
        scale_evidence[str(hvg)] = {}
        for dataset_index, dataset in enumerate(DATASETS):
            test_conditions = [f"{dataset}::condition_{index:02d}" for index in range(50)]
            training_conditions = [f"{dataset}::training_condition"]
            validation_conditions = [f"{dataset}::validation_condition"]
            row_path = root / f"{dataset}_row_conditions.json"
            if not row_path.exists():
                _write_json(row_path, test_conditions)
            split_path = root / f"{dataset}_split.json"
            if not split_path.exists():
                split_payload = _finalize(
                    {
                        "schema_version": "1.0",
                        "split_id": f"{dataset}::frozen_target_disjoint_split",
                        "training_conditions": training_conditions,
                        "validation_conditions": validation_conditions,
                        "test_conditions": test_conditions,
                        "target_disjoint": True,
                    },
                    "manifest_sha256",
                )
                _write_json(split_path, split_payload)

            gene_order = [f"{dataset}_gene_{index:04d}" for index in range(hvg)]
            gene_path = root / f"{dataset}_{hvg}_gene_order.json"
            _write_json(gene_path, gene_order)
            condition_targets = {
                training_conditions[0]: [gene_order[0]],
                validation_conditions[0]: [gene_order[1]],
                **{
                    condition: [gene_order[index + 2]]
                    for index, condition in enumerate(test_conditions)
                },
            }
            target_path = root / f"{dataset}_{hvg}_target_mapping.json"
            target_payload = _finalize(
                {
                    "schema_version": "1.0",
                    "condition_targets": condition_targets,
                    "gene_order_sha256": canonical_sha256(tuple(gene_order)),
                },
                "manifest_sha256",
            )
            _write_json(target_path, target_payload)

            gene_axis = np.linspace(-2.0, 2.0, hvg, dtype=np.float64)
            truth = np.vstack(
                [
                    gene_axis + 0.08 * np.sin((index + 1) * gene_axis / 7.0) + dataset_index / 50.0
                    for index in range(50)
                ]
            )
            candidate = np.vstack(
                [truth[index] + 0.04 * np.cos(gene_axis * 2.0 + index / 9.0) for index in range(50)]
            )
            reference = np.vstack(
                [truth[index] + 0.24 * np.cos(gene_axis * 2.0 + index / 9.0) for index in range(50)]
            )
            candidate_path = root / f"{PASSING_COMPARATOR}_{dataset}_{hvg}_candidate.npy"
            truth_path = root / f"{dataset}_{hvg}_truth.npy"
            reference_path = root / f"{dataset}_{hvg}_revision_string_go.npy"
            np.save(candidate_path, candidate)
            np.save(truth_path, truth)
            np.save(reference_path, reference)
            scale_evidence[str(hvg)][dataset] = {
                "candidate_vectors": _binding(candidate_path),
                "truth_vectors": _binding(truth_path),
                "revision_string_go_vectors": _binding(reference_path),
                "row_conditions": _binding(row_path),
                "gene_order": _binding(gene_path),
                "split": _binding(split_path),
                "target_mapping": _binding(target_path),
            }
    return scale_evidence


def _rebuild_reference_chain(
    root: Path, scale_evidence: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, str]:
    entries: dict[str, dict[str, dict[str, str]]] = {}
    for hvg in HVG_SCALES:
        entries[str(hvg)] = {}
        for dataset in DATASETS:
            bundle = scale_evidence[str(hvg)][dataset]
            for role, binding in list(bundle.items()):
                if role == "revision_string_go_artifact":
                    continue
                binding["sha256"] = file_sha256(root / binding["source_id"])
            gene_order = json.loads((root / bundle["gene_order"]["source_id"]).read_text())
            split = json.loads((root / bundle["split"]["source_id"]).read_text())
            targets = json.loads((root / bundle["target_mapping"]["source_id"]).read_text())
            artifact = _finalize(
                {
                    "schema_version": "1.0",
                    "artifact_id": f"revision_string_go::{dataset}::{hvg}",
                    "reference_model_id": REFERENCE_MODEL_ID,
                    "dataset": dataset,
                    "hvg": hvg,
                    "primary_panel_id": PRIMARY_PANEL_ID,
                    "gene_order_sha256": canonical_sha256(tuple(gene_order)),
                    "row_conditions_file_sha256": bundle["row_conditions"]["sha256"],
                    "split_manifest_sha256": split["manifest_sha256"],
                    "target_mapping_manifest_sha256": targets["manifest_sha256"],
                    "truth_vectors_file_sha256": bundle["truth_vectors"]["sha256"],
                    "prediction_vectors_file_sha256": bundle["revision_string_go_vectors"][
                        "sha256"
                    ],
                },
                "artifact_sha256",
            )
            artifact_path = root / f"{dataset}_{hvg}_revision_string_go_artifact.json"
            _write_json(artifact_path, artifact)
            bundle["revision_string_go_artifact"] = _binding(artifact_path)
            entries[str(hvg)][dataset] = {
                "artifact_sha256": artifact["artifact_sha256"],
                "artifact_file_sha256": bundle["revision_string_go_artifact"]["sha256"],
                "revision_string_go_vectors_file_sha256": bundle["revision_string_go_vectors"][
                    "sha256"
                ],
                "truth_vectors_file_sha256": bundle["truth_vectors"]["sha256"],
                "row_conditions_file_sha256": bundle["row_conditions"]["sha256"],
                "gene_order_file_sha256": bundle["gene_order"]["sha256"],
                "gene_order_sha256": canonical_sha256(tuple(gene_order)),
                "split_manifest_sha256": split["manifest_sha256"],
                "target_mapping_manifest_sha256": targets["manifest_sha256"],
            }
    registry = _finalize(
        {
            "schema_version": "1.0",
            "registry_id": "REVISION-STRING-GO-REFERENCE",
            "status": "RELEASED",
            "reference_model_id": REFERENCE_MODEL_ID,
            "estimand": ESTIMAND,
            "primary_panel_id": PRIMARY_PANEL_ID,
            "entries": entries,
        },
        "registry_sha256",
    )
    registry_path = root / "revision_string_go_reference_registry.json"
    _write_json(registry_path, registry)
    return _binding(registry_path)


def _validity(root: Path, scale_evidence: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    linked = scale_evidence[str(LINKED_HVG)][LINKED_DATASET]
    split = json.loads((root / linked["split"]["source_id"]).read_text())
    targets = json.loads((root / linked["target_mapping"]["source_id"]).read_text())
    members = {
        "GEARS": {
            "comparator": "GEARS",
            "decision": "EXCLUDED",
            "claim_deleted": True,
            "reason_code": "CLAIM_DELETED",
        },
        PASSING_COMPARATOR: {
            "comparator": PASSING_COMPARATOR,
            "decision": "PASS",
            "claim_deleted": False,
            "scientific_validity": "FULL_VECTOR_EVIDENCE_VALID",
            "n_genes": LINKED_HVG,
            "prediction_vectors_sha256": linked["candidate_vectors"]["sha256"],
            "truth_vectors_sha256": linked["truth_vectors"]["sha256"],
            "vector_row_conditions_sha256": linked["row_conditions"]["sha256"],
            "split_manifest_sha256": split["manifest_sha256"],
            "target_mapping_manifest_sha256": targets["manifest_sha256"],
        },
        "Geneformer": {
            "comparator": "Geneformer",
            "decision": "EXCLUDED",
            "claim_deleted": True,
            "reason_code": "CLAIM_DELETED",
        },
    }
    return _finalize(
        {
            "schema_version": "1.0",
            "registry_id": "EXTERNAL-COMPARATOR-VALIDATION",
            "status": "RELEASED",
            "members": members,
            "member_decisions": {
                comparator: members[comparator]["decision"] for comparator in members
            },
        },
        "registry_hash",
    )


def _links() -> dict[str, dict[str, Any]]:
    role_by_field = {
        "prediction_vectors_sha256": "candidate_vectors",
        "truth_vectors_sha256": "truth_vectors",
        "vector_row_conditions_sha256": "row_conditions",
        "split_manifest_sha256": "split",
        "target_mapping_manifest_sha256": "target_mapping",
    }
    return {
        field: {
            "hvg": LINKED_HVG,
            "dataset": LINKED_DATASET,
            "role": role,
        }
        for field, role in role_by_field.items()
    }


def _write_manifest(
    root: Path,
    validity: dict[str, Any],
    scale_evidence: dict[str, dict[str, dict[str, Any]]],
    reference_registry: dict[str, str],
) -> Path:
    members = {}
    for comparator in ("GEARS", PASSING_COMPARATOR, "Geneformer"):
        member = validity["members"][comparator]
        if comparator == PASSING_COMPARATOR:
            members[comparator] = {
                "mode": "validate_raw_vectors",
                "validity_member_payload_sha256": canonical_sha256(member),
                "validity_evidence_links": _links(),
                "scale_evidence": scale_evidence,
            }
        else:
            members[comparator] = {
                "mode": "not_applicable_excluded",
                "reason_code": member["reason_code"],
                "validity_member_payload_sha256": canonical_sha256(member),
            }
    payload = _finalize(
        {
            "schema_version": "2.0",
            "validity_registry_hash": validity["registry_hash"],
            "metric_definition": METRIC_DEFINITION,
            "estimand": ESTIMAND,
            "reference_model_id": REFERENCE_MODEL_ID,
            "reference_registry": reference_registry,
            "members": members,
        },
        "manifest_sha256",
    )
    path = root / "performance_manifest.json"
    _write_json(path, payload)
    return path


def _package(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    scale_evidence = _make_raw_scale_evidence(tmp_path)
    reference_registry = _rebuild_reference_chain(tmp_path, scale_evidence)
    validity = _validity(tmp_path, scale_evidence)
    manifest = _write_manifest(tmp_path, validity, scale_evidence, reference_registry)
    return validity, manifest


def _rewrite_manifest(path: Path, payload: dict[str, Any]) -> None:
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = canonical_sha256(payload)
    _write_json(path, payload)


def _refresh_coordinated_bundle(root: Path, manifest: Path, payload: dict[str, Any]) -> None:
    scale_evidence = payload["members"][PASSING_COMPARATOR]["scale_evidence"]
    payload["reference_registry"] = _rebuild_reference_chain(root, scale_evidence)
    _rewrite_manifest(manifest, payload)


def _release(validity: dict[str, Any], manifest: Path) -> dict[str, Any]:
    return release_external_performance_family(
        validity,
        manifest_path=manifest,
        expected_manifest_sha256=file_sha256(manifest),
        bootstrap_replicates=100,
    )


def test_external_performance_recomputes_three_scale_raw_vector_family(
    tmp_path: Path,
) -> None:
    validity, manifest = _package(tmp_path)
    performance = _release(validity, manifest)
    combined = combine_external_validity_and_performance(validity, performance)

    assert performance["status"] == "RELEASED"
    assert performance["estimand"] == ESTIMAND
    assert performance["reference_model_id"] == REFERENCE_MODEL_ID
    assert performance["metric_definition"] == METRIC_DEFINITION
    assert performance["cross_adapter_pooling"] == "PROHIBITED"
    result = performance["adapter_results"][PASSING_COMPARATOR]
    contrasts = result["contrasts"]
    assert [row["hvg"] for row in contrasts] == [200, 500, 1000]
    assert all(row["contrast"] == "scGPT_minus_revision_string_go" for row in contrasts)
    assert all(0 <= row["bh_q_three_scale_family"] <= 1 for row in contrasts)
    assert result["family_denominator"] == 3
    assert len(result["condition_level_metrics"]) == 4 * 3 * 50
    assert len(result["validity_evidence_links_verified"]) == 6
    assert any(
        row["validity_field"] == "gene_order_via_target_mapping_manifest_sha256_and_n_genes"
        for row in result["validity_evidence_links_verified"]
    )
    first = result["condition_level_metrics"][0]
    candidate = np.load(tmp_path / "scGPT_adamson_200_candidate.npy")[0]
    truth = np.load(tmp_path / "adamson_200_truth.npy")[0]
    reference = np.load(tmp_path / "adamson_200_revision_string_go.npy")[0]
    assert first["candidate_pearson_r"] == pytest.approx(np.corrcoef(candidate, truth)[0, 1])
    assert first["revision_string_go_pearson_r"] == pytest.approx(
        np.corrcoef(reference, truth)[0, 1]
    )
    assert combined["status"] == "RELEASED"


def test_caller_scalar_metric_cannot_enter_raw_vector_evidence(tmp_path: Path) -> None:
    validity, manifest = _package(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["members"][PASSING_COMPARATOR]["scale_evidence"]["200"]["adamson"][
        "candidate_metric"
    ] = 0.999
    _rewrite_manifest(manifest, payload)

    result = _release(validity, manifest)
    assert result["status"] == "WITHHELD"
    assert result["reason_codes"] == ["PERFORMANCE_EVIDENCE_INVALID"]


def test_coordinated_candidate_vector_rewrite_cannot_escape_validity_member_hash(
    tmp_path: Path,
) -> None:
    validity, manifest = _package(tmp_path)
    payload = json.loads(manifest.read_text())
    bundle = payload["members"][PASSING_COMPARATOR]["scale_evidence"]["1000"][LINKED_DATASET]
    candidate_path = tmp_path / bundle["candidate_vectors"]["source_id"]
    candidate = np.load(candidate_path)
    candidate[0] += np.linspace(0.0, 0.01, candidate.shape[1])
    np.save(candidate_path, candidate)
    _refresh_coordinated_bundle(tmp_path, manifest, payload)

    result = _release(validity, manifest)
    assert result["status"] == "WITHHELD"
    assert "prediction_vectors_sha256" in result["detail"]


@pytest.mark.parametrize("tamper", ["condition_order", "target", "gene", "split"])
def test_semantic_identity_tamper_is_rejected_after_outer_hashes_are_refreshed(
    tmp_path: Path, tamper: str
) -> None:
    validity, manifest = _package(tmp_path)
    payload = json.loads(manifest.read_text())
    bundle = payload["members"][PASSING_COMPARATOR]["scale_evidence"]["1000"][LINKED_DATASET]
    if tamper == "condition_order":
        row_path = tmp_path / bundle["row_conditions"]["source_id"]
        rows = json.loads(row_path.read_text())
        rows[0], rows[1] = rows[1], rows[0]
        _write_json(row_path, rows)
        for role in (
            "candidate_vectors",
            "truth_vectors",
            "revision_string_go_vectors",
        ):
            vector_path = tmp_path / bundle[role]["source_id"]
            vectors = np.load(vector_path)
            vectors[[0, 1]] = vectors[[1, 0]]
            np.save(vector_path, vectors)
        split_path = tmp_path / bundle["split"]["source_id"]
        split = json.loads(split_path.read_text())
        split["test_conditions"] = rows
        split.pop("manifest_sha256")
        split["manifest_sha256"] = canonical_sha256(split)
        _write_json(split_path, split)
    elif tamper == "target":
        target_path = tmp_path / bundle["target_mapping"]["source_id"]
        targets = json.loads(target_path.read_text())
        condition = next(
            iter(json.loads((tmp_path / bundle["row_conditions"]["source_id"]).read_text()))
        )
        targets["condition_targets"][condition] = ["coordinated_forged_target"]
        targets.pop("manifest_sha256")
        targets["manifest_sha256"] = canonical_sha256(targets)
        _write_json(target_path, targets)
    elif tamper == "gene":
        gene_path = tmp_path / bundle["gene_order"]["source_id"]
        genes = json.loads(gene_path.read_text())
        genes[-1] = "coordinated_forged_gene"
        _write_json(gene_path, genes)
        target_path = tmp_path / bundle["target_mapping"]["source_id"]
        targets = json.loads(target_path.read_text())
        targets["gene_order_sha256"] = canonical_sha256(tuple(genes))
        targets.pop("manifest_sha256")
        targets["manifest_sha256"] = canonical_sha256(targets)
        _write_json(target_path, targets)
    else:
        split_path = tmp_path / bundle["split"]["source_id"]
        split = json.loads(split_path.read_text())
        split["split_id"] = "coordinated_forged_split"
        split.pop("manifest_sha256")
        split["manifest_sha256"] = canonical_sha256(split)
        _write_json(split_path, split)
    _refresh_coordinated_bundle(tmp_path, manifest, payload)

    result = _release(validity, manifest)
    assert result["status"] == "WITHHELD"
    assert result["reason_codes"] == ["PERFORMANCE_EVIDENCE_INVALID"]


def test_revision_string_go_vector_requires_matching_artifact_and_registry_chain(
    tmp_path: Path,
) -> None:
    validity, manifest = _package(tmp_path)
    payload = json.loads(manifest.read_text())
    bundle = payload["members"][PASSING_COMPARATOR]["scale_evidence"]["500"]["norman"]
    vector_path = tmp_path / bundle["revision_string_go_vectors"]["source_id"]
    vectors = np.load(vector_path)
    vectors[0] += np.linspace(0.0, 0.02, vectors.shape[1])
    np.save(vector_path, vectors)
    bundle["revision_string_go_vectors"]["sha256"] = file_sha256(vector_path)
    _rewrite_manifest(manifest, payload)

    result = _release(validity, manifest)
    assert result["status"] == "WITHHELD"
    assert "artifact binding" in result["detail"]


def test_immutable_caller_pin_rejects_coordinated_reference_bundle_rewrite(
    tmp_path: Path,
) -> None:
    validity, manifest = _package(tmp_path)
    pinned = file_sha256(manifest)
    payload = json.loads(manifest.read_text())
    bundle = payload["members"][PASSING_COMPARATOR]["scale_evidence"]["500"]["norman"]
    vector_path = tmp_path / bundle["revision_string_go_vectors"]["source_id"]
    vectors = np.load(vector_path)
    vectors[0] += np.linspace(0.0, 0.02, vectors.shape[1])
    np.save(vector_path, vectors)
    _refresh_coordinated_bundle(tmp_path, manifest, payload)

    result = release_external_performance_family(
        validity,
        manifest_path=manifest,
        expected_manifest_sha256=pinned,
        bootstrap_replicates=100,
    )
    assert result["status"] == "WITHHELD"
    assert result["reason_codes"] == ["PERFORMANCE_MANIFEST_CALLER_PIN_MISMATCH"]


def test_all_reason_coded_excluded_members_need_no_performance_hypotheses(
    tmp_path: Path,
) -> None:
    validity, _ = _package(tmp_path)
    validity = copy.deepcopy(validity)
    member = validity["members"][PASSING_COMPARATOR]
    member.clear()
    member.update(
        {
            "comparator": PASSING_COMPARATOR,
            "decision": "EXCLUDED",
            "claim_deleted": True,
            "reason_code": "CLAIM_DELETED",
        }
    )
    validity["member_decisions"][PASSING_COMPARATOR] = "EXCLUDED"
    validity.pop("registry_hash")
    validity["registry_hash"] = canonical_sha256(validity)

    registry = release_external_performance_family(
        validity,
        manifest_path=None,
        expected_manifest_sha256=None,
    )
    assert registry["status"] == "RELEASED"
    assert registry["execution_status"] == "NO_PASSING_ADAPTERS_NO_PERFORMANCE_HYPOTHESES"
    assert registry["estimand"] == ESTIMAND


def test_fixture_evidence_roles_are_exact_and_do_not_expose_scalar_columns(
    tmp_path: Path,
) -> None:
    _, manifest = _package(tmp_path)
    payload = json.loads(manifest.read_text())
    bundle = payload["members"][PASSING_COMPARATOR]["scale_evidence"]["200"]["adamson"]
    assert set(bundle) == set(EVIDENCE_FILE_ROLES)
    assert not any("metric" in field for field in bundle)

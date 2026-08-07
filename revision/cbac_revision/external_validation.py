"""Fail-closed external-comparator validity family."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import canonical_sha256, file_sha256
from .errors import RevisionProtocolError

COMPARATORS = ("GEARS", "scGPT", "Geneformer")
DECISIONS = ("PASS", "EXCLUDED", "WITHHELD")
GENERIC_FILE_BINDINGS = (
    "dataset",
    "split",
    "target_mapping",
    "prediction_vectors",
    "truth_vectors",
    "vector_row_conditions",
    "training_loss",
    "validation_loss",
    "gene_order",
    "baseline_vectors",
    "baseline_provenance",
    "baseline_training_vectors",
    "baseline_training_row_conditions",
    "independent_expected_value_provenance",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class ComparatorScientificGateFailed(RevisionProtocolError):
    """Raised when valid evidence shows a comparator fails its frozen scientific gate."""


def read_external_comparator_family(
    manifest_path: Path, *, expected_manifest_sha256: str
) -> dict[str, Any]:
    """Validate a caller-pinned family manifest and all comparator evidence."""

    if not SHA256_RE.fullmatch(expected_manifest_sha256):
        raise RevisionProtocolError("External-family trusted SHA-256 is invalid")
    if file_sha256(manifest_path) != expected_manifest_sha256:
        raise RevisionProtocolError("External-family manifest differs from caller-pinned SHA-256")
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RevisionProtocolError(f"External-family manifest is invalid: {error}") from error
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "members",
        "manifest_sha256",
    }:
        raise RevisionProtocolError("External-family manifest schema is invalid")
    unsigned = dict(manifest)
    declared_hash = unsigned.pop("manifest_sha256")
    if declared_hash != canonical_sha256(unsigned):
        raise RevisionProtocolError("External-family manifest self hash is invalid")
    if manifest["schema_version"] != "1.0" or set(manifest["members"]) != set(COMPARATORS):
        raise RevisionProtocolError("External-family members must be GEARS/scGPT/Geneformer")

    member_results: dict[str, dict[str, Any]] = {}
    root = manifest_path.resolve().parent
    for comparator in COMPARATORS:
        specification = manifest["members"].get(comparator)
        if not isinstance(specification, dict):
            member_results[comparator] = _withheld(
                comparator, "MEMBER_SPECIFICATION_MISSING_OR_INVALID"
            )
            continue
        mode = specification.get("mode")
        expected_specification_fields = {"mode", "source_id", "sha256"}
        if mode == "validate" and comparator == "GEARS":
            expected_specification_fields |= {
                "trust_anchor_source_id",
                "trust_anchor_sha256",
            }
        if set(specification) != expected_specification_fields:
            member_results[comparator] = _withheld(
                comparator, "MEMBER_SPECIFICATION_MISSING_OR_INVALID"
            )
            continue
        source = _resolve_bound_file(root, specification.get("source_id"))
        expected_hash = specification.get("sha256")
        if (
            source is None
            or not isinstance(expected_hash, str)
            or not SHA256_RE.fullmatch(expected_hash)
            or not source.is_file()
            or file_sha256(source) != expected_hash
        ):
            member_results[comparator] = _withheld(comparator, "MEMBER_SOURCE_BINDING_INVALID")
            continue
        try:
            if mode == "exclude":
                member_results[comparator] = validate_exclusion_manifest(source, comparator)
            elif mode == "validate" and comparator == "GEARS":
                trust_anchor = _resolve_bound_file(
                    root, specification.get("trust_anchor_source_id")
                )
                trust_anchor_hash = specification.get("trust_anchor_sha256")
                if (
                    trust_anchor is None
                    or not trust_anchor.is_file()
                    or not isinstance(trust_anchor_hash, str)
                    or not SHA256_RE.fullmatch(trust_anchor_hash)
                    or file_sha256(trust_anchor) != trust_anchor_hash
                ):
                    raise RevisionProtocolError("GEARS detached trust-anchor binding is invalid")
                member_results[comparator] = validate_gears_member(
                    source,
                    trust_anchor_path=trust_anchor,
                    expected_trust_anchor_sha256=trust_anchor_hash,
                )
            elif mode == "validate":
                member_results[comparator] = validate_generic_comparator_manifest(
                    source, comparator
                )
            else:
                member_results[comparator] = _withheld(comparator, "MEMBER_MODE_INVALID")
        except ComparatorScientificGateFailed as error:
            member_results[comparator] = {
                "comparator": comparator,
                "decision": "EXCLUDED",
                "scientific_validity": "VALID_EVIDENCE_FAILED_SCIENTIFIC_GATE",
                "claim_deleted": False,
                "reason_code": "SCIENTIFIC_GATE_FAILED",
                "detail": str(error),
            }
        except RevisionProtocolError as error:
            member_results[comparator] = _withheld(
                comparator, "SCIENTIFIC_VALIDITY_EVIDENCE_INVALID", detail=str(error)
            )
    decisions = {member: result["decision"] for member, result in member_results.items()}
    family_status = "RELEASED" if "WITHHELD" not in decisions.values() else "WITHHELD"
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "registry_id": "EXTERNAL-COMPARATOR-VALIDATION",
        "status": family_status,
        "family": "EXTERNAL_COMPARATOR_VALIDITY",
        "parse_valid_scalar_is_scientific_validity": False,
        "trusted_manifest_file_sha256": expected_manifest_sha256,
        "members": member_results,
        "member_decisions": decisions,
        "ranking_eligible_members": [
            member for member in COMPARATORS if decisions[member] == "PASS"
        ],
        "claim_deleted_members": [
            member for member in COMPARATORS if member_results[member].get("claim_deleted") is True
        ],
        "scientific_gate_failed_members": [
            member
            for member in COMPARATORS
            if member_results[member].get("reason_code") == "SCIENTIFIC_GATE_FAILED"
        ],
        "family_completion_rule": "PASS_OR_REASON_CODED_EXCLUDED_FOR_EACH_MEMBER",
    }
    payload["registry_hash"] = canonical_sha256(payload)
    return payload


def validate_exclusion_manifest(path: Path, comparator: str) -> dict[str, Any]:
    """Accept claim deletion only through an explicit self-hashed reason-coded manifest."""

    payload = _read_json(path)
    expected_fields = {
        "schema_version",
        "comparator",
        "decision",
        "claim_deleted",
        "reason_code",
        "rationale",
        "claim_locations_removed",
        "frozen_date",
        "manifest_sha256",
    }
    if set(payload) != expected_fields:
        raise RevisionProtocolError("Comparator exclusion manifest schema is invalid")
    unsigned = dict(payload)
    declared_hash = unsigned.pop("manifest_sha256")
    if declared_hash != canonical_sha256(unsigned):
        raise RevisionProtocolError("Comparator exclusion manifest self hash is invalid")
    if (
        payload["schema_version"] != "1.0"
        or payload["comparator"] != comparator
        or payload["decision"] != "EXCLUDED"
        or payload["claim_deleted"] is not True
        or not isinstance(payload["reason_code"], str)
        or not re.fullmatch(r"[A-Z][A-Z0-9_]{4,}", payload["reason_code"])
        or not isinstance(payload["rationale"], str)
        or not payload["rationale"].strip()
        or not isinstance(payload["claim_locations_removed"], list)
        or not payload["claim_locations_removed"]
        or not all(
            isinstance(value, str) and value.strip() for value in payload["claim_locations_removed"]
        )
        or not isinstance(payload["frozen_date"], str)
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", payload["frozen_date"])
    ):
        raise RevisionProtocolError("Comparator exclusion declaration is incomplete")
    return {
        "comparator": comparator,
        "decision": "EXCLUDED",
        "scientific_validity": "NOT_CLAIMED",
        "claim_deleted": True,
        "reason_code": payload["reason_code"],
        "exclusion_manifest_sha256": file_sha256(path),
    }


def validate_gears_member(
    path: Path,
    *,
    trust_anchor_path: Path,
    expected_trust_anchor_sha256: str,
) -> dict[str, Any]:
    """Rerun the detached-anchor validator from raw GEARS evidence."""

    from .gears_validation import validate_gears_positive_control

    payload = validate_gears_positive_control(
        path,
        trust_anchor_path=trust_anchor_path,
        expected_trust_anchor_sha256=expected_trust_anchor_sha256,
    )
    declared_hash = payload.get("registry_hash")
    unsigned = dict(payload)
    unsigned.pop("registry_hash", None)
    if declared_hash != canonical_sha256(unsigned):
        raise RevisionProtocolError("Recomputed GEARS registry self hash is invalid")
    eligibility = payload.get("eligibility_decision")
    if payload.get("status") == "RELEASED" and eligibility == "ELIGIBLE_POSITIVE_CONTROL_PASSED":
        decision = "PASS"
    elif payload.get("status") == "RELEASED" and eligibility == "EXCLUDED_FAILED_POSITIVE_CONTROL":
        decision = "EXCLUDED"
    else:
        decision = "WITHHELD"
    return {
        "comparator": "GEARS",
        "decision": decision,
        "scientific_validity": payload.get("evidence_status"),
        "claim_deleted": False,
        "source_registry_hash": declared_hash,
        "evidence_manifest_file_sha256": file_sha256(path),
        "detached_trust_anchor_file_sha256": file_sha256(trust_anchor_path),
        "reason_code": eligibility,
        "comparisons": payload.get("comparisons"),
        "validated_bindings": payload.get("provenance_bindings"),
        "semantic_diagnostics": payload.get("semantic_diagnostics"),
        "detached_validator_registry_status": payload.get("status"),
    }


def validate_generic_comparator_manifest(path: Path, comparator: str) -> dict[str, Any]:
    """Require full vectors and independent expected-value provenance, not a scalar."""

    payload = _read_json(path)
    if set(payload) != {
        "schema_version",
        "comparator",
        "official_repository",
        "pinned_commit",
        "bindings",
        "manifest_sha256",
    }:
        raise RevisionProtocolError("Generic comparator manifest schema is invalid")
    unsigned = dict(payload)
    declared_hash = unsigned.pop("manifest_sha256")
    if declared_hash != canonical_sha256(unsigned):
        raise RevisionProtocolError("Generic comparator manifest self hash is invalid")
    repository = payload.get("official_repository")
    commit = payload.get("pinned_commit")
    if (
        payload.get("schema_version") != "1.0"
        or payload.get("comparator") != comparator
        or not isinstance(repository, str)
        or not repository.startswith("https://")
        or not isinstance(commit, str)
        or not COMMIT_RE.fullmatch(commit)
    ):
        raise RevisionProtocolError("Generic comparator repository binding is invalid")
    bindings = payload.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != set(GENERIC_FILE_BINDINGS):
        raise RevisionProtocolError("Generic comparator file bindings are incomplete")
    root = path.resolve().parent
    bound: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name in GENERIC_FILE_BINDINGS:
        binding = bindings[name]
        if not isinstance(binding, dict) or set(binding) != {"source_id", "sha256"}:
            raise RevisionProtocolError(f"Comparator binding {name} is invalid")
        source = _resolve_bound_file(root, binding["source_id"])
        expected_hash = binding["sha256"]
        if (
            source is None
            or not isinstance(expected_hash, str)
            or not SHA256_RE.fullmatch(expected_hash)
            or not source.is_file()
            or file_sha256(source) != expected_hash
        ):
            raise RevisionProtocolError(f"Comparator binding {name} failed hash validation")
        bound[name] = source
        hashes[name] = expected_hash
    prediction = _numeric_array(bound["prediction_vectors"])
    truth = _numeric_array(bound["truth_vectors"])
    baseline = _numeric_array(bound["baseline_vectors"])
    if (
        prediction.ndim != 2
        or prediction.shape != truth.shape
        or baseline.shape != truth.shape
        or truth.shape[0] < 1
        or truth.shape[1] < 2
    ):
        raise RevisionProtocolError("Comparator prediction/truth/baseline vector shapes differ")
    training_loss = _numeric_array(bound["training_loss"])
    validation_loss = _numeric_array(bound["validation_loss"])
    if not training_loss.size or not validation_loss.size:
        raise RevisionProtocolError("Comparator loss histories are empty")
    gene_order = _read_gene_order(bound["gene_order"])
    if len(gene_order) != truth.shape[-1]:
        raise RevisionProtocolError("Comparator gene order does not match vector space")
    vector_row_conditions = _read_ordered_identifiers(
        bound["vector_row_conditions"], label="vector row conditions"
    )
    if len(vector_row_conditions) != truth.shape[0]:
        raise RevisionProtocolError("Comparator vector row identities do not match vector rows")
    split_payload = _read_json(bound["split"])
    target_payload = _read_json(bound["target_mapping"])
    split = _validate_split(split_payload)
    if tuple(split["test_conditions"]) != vector_row_conditions:
        raise RevisionProtocolError(
            "Comparator vector row conditions differ from the frozen test split order"
        )
    target_mapping = _validate_target_mapping(target_payload, split=split, gene_order=gene_order)
    _validate_dataset_evidence(
        _read_json(bound["dataset"]),
        gene_order=gene_order,
        vector_row_conditions=vector_row_conditions,
        truth_vectors_sha256=hashes["truth_vectors"],
        target_mapping_manifest_sha256=target_mapping["manifest_sha256"],
    )
    _validate_and_recompute_baseline(
        baseline=baseline,
        truth_shape=truth.shape,
        provenance=_read_json(bound["baseline_provenance"]),
        training_vectors=_numeric_array(bound["baseline_training_vectors"]),
        training_row_conditions=_read_ordered_identifiers(
            bound["baseline_training_row_conditions"],
            label="baseline training row conditions",
        ),
        split=split,
        gene_order=gene_order,
        hashes=hashes,
    )
    provenance = _read_json(bound["independent_expected_value_provenance"])
    expected_contract = _validate_expected_provenance(
        provenance,
        comparator=comparator,
        repository=repository,
        commit=commit,
        hashes=hashes,
    )
    observed_metrics = _recomputed_metrics(truth, prediction, require_prediction_variance=True)
    baseline_metrics = _recomputed_metrics(truth, baseline, require_prediction_variance=False)
    comparisons: list[dict[str, Any]] = []
    for metric in ("pearson_r", "mse", "mae"):
        contract = expected_contract["expected_metrics"][metric]
        expected_value = float(contract["expected_value"])
        observed_value = float(observed_metrics[metric])
        tolerance = max(
            float(contract["absolute_tolerance"]),
            abs(expected_value) * float(contract["relative_tolerance"]),
        )
        passed = abs(observed_value - expected_value) <= tolerance
        comparisons.append(
            {
                "metric": metric,
                "observed": observed_value,
                "expected": expected_value,
                "tolerance": tolerance,
                "status": "PASS" if passed else "FAIL",
            }
        )
    if any(row["status"] != "PASS" for row in comparisons):
        raise ComparatorScientificGateFailed("Recomputed metrics are outside frozen tolerances")
    if not (
        observed_metrics["mse"] < baseline_metrics["mse"]
        and observed_metrics["mae"] < baseline_metrics["mae"]
        and (
            baseline_metrics["pearson_r"] is None
            or observed_metrics["pearson_r"] > baseline_metrics["pearson_r"]
        )
    ):
        raise ComparatorScientificGateFailed("Comparator does not outperform its bound baseline")
    loss_contract = expected_contract["loss_contract"]
    required_epochs = int(loss_contract["minimum_history_length"])
    if training_loss.ndim != 1 or validation_loss.ndim != 1:
        raise RevisionProtocolError("Comparator loss histories must be one-dimensional")
    training_improvement = float(training_loss[0] - training_loss[-1])
    validation_improvement = float(validation_loss[0] - validation_loss[-1])
    if (
        len(training_loss) < required_epochs
        or len(validation_loss) < required_epochs
        or training_improvement < float(loss_contract["minimum_training_improvement"])
        or validation_improvement < float(loss_contract["minimum_validation_improvement"])
    ):
        raise ComparatorScientificGateFailed("Loss-history improvement gate failed")
    return {
        "comparator": comparator,
        "decision": "PASS",
        "scientific_validity": "FULL_VECTOR_EVIDENCE_VALID",
        "claim_deleted": False,
        "repository": repository,
        "commit": commit,
        "manifest_sha256": declared_hash,
        "source_file_sha256": file_sha256(path),
        "n_vector_values": int(truth.size),
        "n_genes": len(gene_order),
        "prediction_vectors_sha256": hashes["prediction_vectors"],
        "truth_vectors_sha256": hashes["truth_vectors"],
        "baseline_vectors_sha256": hashes["baseline_vectors"],
        "vector_row_conditions_sha256": hashes["vector_row_conditions"],
        "recomputed_metrics": observed_metrics,
        "recomputed_baseline_metrics": baseline_metrics,
        "expected_metric_comparisons": comparisons,
        "training_loss_improvement": training_improvement,
        "validation_loss_improvement": validation_improvement,
        "minimum_training_loss_improvement": float(loss_contract["minimum_training_improvement"]),
        "minimum_validation_loss_improvement": float(
            loss_contract["minimum_validation_improvement"]
        ),
        "gene_order_sha256": hashes["gene_order"],
        "dataset_file_sha256": hashes["dataset"],
        "independent_expected_value_provenance_sha256": hashes[
            "independent_expected_value_provenance"
        ],
        "split_manifest_sha256": split["manifest_sha256"],
        "target_mapping_manifest_sha256": target_mapping["manifest_sha256"],
    }


def _validate_expected_provenance(
    payload: Mapping[str, Any],
    *,
    comparator: str,
    repository: str,
    commit: str,
    hashes: Mapping[str, str],
) -> Mapping[str, Any]:
    expected_fields = {
        "schema_version",
        "comparator",
        "repository_url",
        "commit",
        "dataset_sha256",
        "split_sha256",
        "target_mapping_sha256",
        "prediction_vectors_sha256",
        "truth_vectors_sha256",
        "vector_row_conditions_sha256",
        "training_loss_sha256",
        "validation_loss_sha256",
        "gene_order_sha256",
        "baseline_vectors_sha256",
        "baseline_provenance_sha256",
        "baseline_training_vectors_sha256",
        "baseline_training_row_conditions_sha256",
        "expected_metrics",
        "loss_contract",
        "source_id",
        "source_kind",
        "frozen_before_execution",
        "provenance_sha256",
    }
    if set(payload) != expected_fields:
        raise RevisionProtocolError("Independent expected-value provenance schema is invalid")
    unsigned = dict(payload)
    declared_hash = unsigned.pop("provenance_sha256")
    if declared_hash != canonical_sha256(unsigned):
        raise RevisionProtocolError("Expected-value provenance self hash is invalid")
    cross_bindings = {
        "dataset_sha256": hashes["dataset"],
        "split_sha256": hashes["split"],
        "target_mapping_sha256": hashes["target_mapping"],
        "prediction_vectors_sha256": hashes["prediction_vectors"],
        "truth_vectors_sha256": hashes["truth_vectors"],
        "vector_row_conditions_sha256": hashes["vector_row_conditions"],
        "training_loss_sha256": hashes["training_loss"],
        "validation_loss_sha256": hashes["validation_loss"],
        "gene_order_sha256": hashes["gene_order"],
        "baseline_vectors_sha256": hashes["baseline_vectors"],
        "baseline_provenance_sha256": hashes["baseline_provenance"],
        "baseline_training_vectors_sha256": hashes["baseline_training_vectors"],
        "baseline_training_row_conditions_sha256": hashes["baseline_training_row_conditions"],
    }
    metrics = payload.get("expected_metrics")
    loss_contract = payload.get("loss_contract")
    if (
        payload.get("schema_version") != "1.0"
        or payload.get("comparator") != comparator
        or payload.get("repository_url") != repository
        or payload.get("commit") != commit
        or payload.get("frozen_before_execution") is not True
        or not isinstance(payload.get("source_id"), str)
        or not payload["source_id"].strip()
        or payload.get("source_kind") != "official_release_or_independent_archived_reference"
        or any(payload.get(field) != value for field, value in cross_bindings.items())
        or not isinstance(metrics, dict)
        or set(metrics) != {"pearson_r", "mse", "mae"}
        or not isinstance(loss_contract, dict)
        or set(loss_contract)
        != {
            "minimum_history_length",
            "minimum_training_improvement",
            "minimum_validation_improvement",
        }
    ):
        raise RevisionProtocolError("Expected-value provenance cross-binding is invalid")
    for metric, expected_direction in {
        "pearson_r": "higher_is_better",
        "mse": "lower_is_better",
        "mae": "lower_is_better",
    }.items():
        contract = metrics[metric]
        if not isinstance(contract, dict) or set(contract) != {
            "expected_value",
            "direction",
            "absolute_tolerance",
            "relative_tolerance",
            "justification",
            "official_independent_source",
        }:
            raise RevisionProtocolError(f"Expected metric contract {metric} is invalid")
        numeric_values = [
            contract["expected_value"],
            contract["absolute_tolerance"],
            contract["relative_tolerance"],
        ]
        if (
            contract["direction"] != expected_direction
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in numeric_values
            )
            or float(contract["absolute_tolerance"]) < 0
            or float(contract["relative_tolerance"]) < 0
            or not isinstance(contract["justification"], str)
            or not contract["justification"].strip()
            or not isinstance(contract["official_independent_source"], str)
            or not contract["official_independent_source"].strip()
        ):
            raise RevisionProtocolError(f"Expected metric contract {metric} is invalid")
    if (
        isinstance(loss_contract["minimum_history_length"], bool)
        or not isinstance(loss_contract["minimum_history_length"], int)
        or loss_contract["minimum_history_length"] < 2
        or any(
            isinstance(loss_contract[field], bool)
            or not isinstance(loss_contract[field], (int, float))
            or not math.isfinite(float(loss_contract[field]))
            or float(loss_contract[field]) < 0
            for field in (
                "minimum_training_improvement",
                "minimum_validation_improvement",
            )
        )
    ):
        raise RevisionProtocolError("Expected loss contract is invalid")
    return payload


def _numeric_array(path: Path) -> np.ndarray:
    suffix = path.suffix.casefold()
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"values"}:
                raise RevisionProtocolError("Comparator NPZ must contain only values")
            array = archive["values"]
    else:
        raise RevisionProtocolError("Comparator numeric evidence must be NPY or NPZ")
    output = np.asarray(array, dtype=np.float64)
    if not output.size or not np.isfinite(output).all():
        raise RevisionProtocolError("Comparator numeric evidence is empty or non-finite")
    return output


def _validate_split(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "split_id",
        "training_conditions",
        "validation_conditions",
        "test_conditions",
        "target_disjoint",
        "manifest_sha256",
    }:
        raise RevisionProtocolError("Comparator split manifest schema is invalid")
    unsigned = dict(payload)
    declared_hash = unsigned.pop("manifest_sha256")
    if declared_hash != canonical_sha256(unsigned):
        raise RevisionProtocolError("Comparator split manifest self hash is invalid")
    sets: list[set[str]] = []
    aliases: list[set[str]] = []
    for field in ("training_conditions", "validation_conditions", "test_conditions"):
        values = payload[field]
        if (
            not isinstance(values, list)
            or not values
            or not all(
                isinstance(value, str) and value and value == value.strip() for value in values
            )
            or len(set(values)) != len(values)
        ):
            raise RevisionProtocolError(f"Comparator split {field} is invalid")
        sets.append(set(values))
        aliases.append({_condition_alias(value) for value in values})
    if (
        payload["schema_version"] != "1.0"
        or not isinstance(payload["split_id"], str)
        or not payload["split_id"].strip()
        or payload["target_disjoint"] is not True
        or any(sets[left] & sets[right] for left in range(3) for right in range(left + 1, 3))
        or any(aliases[left] & aliases[right] for left in range(3) for right in range(left + 1, 3))
    ):
        raise RevisionProtocolError("Comparator split semantics are invalid")
    return dict(payload)


def _validate_target_mapping(
    payload: Any, *, split: Mapping[str, Any], gene_order: Sequence[str]
) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "condition_targets",
        "gene_order_sha256",
        "manifest_sha256",
    }:
        raise RevisionProtocolError("Comparator target-mapping schema is invalid")
    unsigned = dict(payload)
    declared_hash = unsigned.pop("manifest_sha256")
    if declared_hash != canonical_sha256(unsigned):
        raise RevisionProtocolError("Comparator target-mapping self hash is invalid")
    mapping = payload["condition_targets"]
    expected_conditions = set().union(
        *(
            set(split[field])
            for field in ("training_conditions", "validation_conditions", "test_conditions")
        )
    )
    gene_set = set(gene_order)
    if (
        payload["schema_version"] != "1.0"
        or payload["gene_order_sha256"] != canonical_sha256(tuple(gene_order))
        or not isinstance(mapping, dict)
        or set(mapping) != expected_conditions
    ):
        raise RevisionProtocolError("Comparator target-mapping bindings are invalid")
    for condition, targets in mapping.items():
        if (
            not isinstance(condition, str)
            or not isinstance(targets, list)
            or not targets
            or len(set(targets)) != len(targets)
            or any(not isinstance(target, str) or target not in gene_set for target in targets)
        ):
            raise RevisionProtocolError("Comparator target mapping contains an invalid target")
    partition_targets = {
        partition: set().union(*(set(mapping[condition]) for condition in split[partition]))
        for partition in ("training_conditions", "validation_conditions", "test_conditions")
    }
    partitions = tuple(partition_targets)
    if any(
        partition_targets[partitions[left]] & partition_targets[partitions[right]]
        for left in range(len(partitions))
        for right in range(left + 1, len(partitions))
    ):
        raise RevisionProtocolError(
            "Comparator target-disjoint declaration conflicts with condition targets"
        )
    return {
        "manifest_sha256": declared_hash,
        "partition_target_sha256": {
            partition: canonical_sha256(tuple(sorted(targets)))
            for partition, targets in partition_targets.items()
        },
    }


def _validate_dataset_evidence(
    payload: Any,
    *,
    gene_order: Sequence[str],
    vector_row_conditions: Sequence[str],
    truth_vectors_sha256: str,
    target_mapping_manifest_sha256: str,
) -> None:
    expected_fields = {
        "schema_version",
        "dataset_id",
        "cell_line",
        "assay",
        "vector_shape",
        "test_row_conditions",
        "test_row_conditions_sha256",
        "gene_order_sha256",
        "truth_vectors_file_sha256",
        "target_mapping_manifest_sha256",
        "manifest_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise RevisionProtocolError("Comparator dataset evidence schema is invalid")
    unsigned = dict(payload)
    declared_hash = unsigned.pop("manifest_sha256")
    if declared_hash != canonical_sha256(unsigned):
        raise RevisionProtocolError("Comparator dataset evidence self hash is invalid")
    row_conditions = payload["test_row_conditions"]
    shape = payload["vector_shape"]
    if (
        payload["schema_version"] != "1.0"
        or any(
            not isinstance(payload[field], str) or not payload[field].strip()
            for field in ("dataset_id", "cell_line", "assay")
        )
        or not isinstance(shape, list)
        or len(shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in shape
        )
        or shape != [len(vector_row_conditions), len(gene_order)]
        or row_conditions != list(vector_row_conditions)
        or payload["test_row_conditions_sha256"] != canonical_sha256(tuple(vector_row_conditions))
        or payload["gene_order_sha256"] != canonical_sha256(tuple(gene_order))
        or payload["truth_vectors_file_sha256"] != truth_vectors_sha256
        or payload["target_mapping_manifest_sha256"] != target_mapping_manifest_sha256
    ):
        raise RevisionProtocolError("Comparator dataset row/gene identity binding is invalid")


def _validate_and_recompute_baseline(
    *,
    baseline: np.ndarray,
    truth_shape: tuple[int, ...],
    provenance: Any,
    training_vectors: np.ndarray,
    training_row_conditions: Sequence[str],
    split: Mapping[str, Any],
    gene_order: Sequence[str],
    hashes: Mapping[str, str],
) -> None:
    expected_fields = {
        "schema_version",
        "baseline_type",
        "fit_scope",
        "training_vectors_file_sha256",
        "training_row_conditions_file_sha256",
        "gene_order_sha256",
        "split_file_sha256",
        "expected_baseline_vectors_file_sha256",
        "manifest_sha256",
    }
    if not isinstance(provenance, dict) or set(provenance) != expected_fields:
        raise RevisionProtocolError("Comparator baseline provenance schema is invalid")
    unsigned = dict(provenance)
    declared_hash = unsigned.pop("manifest_sha256")
    if declared_hash != canonical_sha256(unsigned):
        raise RevisionProtocolError("Comparator baseline provenance self hash is invalid")
    baseline_type = provenance["baseline_type"]
    if (
        provenance["schema_version"] != "1.0"
        or baseline_type not in {"zero_control_delta", "training_condition_mean"}
        or provenance["fit_scope"] != "training_conditions_only_no_test_outcome"
        or tuple(training_row_conditions) != tuple(split["training_conditions"])
        or training_vectors.ndim != 2
        or training_vectors.shape != (len(training_row_conditions), len(gene_order))
        or provenance["training_vectors_file_sha256"] != hashes["baseline_training_vectors"]
        or provenance["training_row_conditions_file_sha256"]
        != hashes["baseline_training_row_conditions"]
        or provenance["gene_order_sha256"] != canonical_sha256(tuple(gene_order))
        or provenance["split_file_sha256"] != hashes["split"]
        or provenance["expected_baseline_vectors_file_sha256"] != hashes["baseline_vectors"]
    ):
        raise RevisionProtocolError("Comparator baseline provenance binding is invalid")
    if baseline_type == "zero_control_delta":
        recomputed = np.zeros(truth_shape, dtype=np.float64)
    else:
        recomputed = np.repeat(training_vectors.mean(axis=0)[None, :], truth_shape[0], axis=0)
    if baseline.shape != truth_shape or not np.array_equal(baseline, recomputed):
        raise RevisionProtocolError(
            "Comparator baseline vectors do not equal the independently recomputed baseline"
        )


def _recomputed_metrics(
    truth: np.ndarray, prediction: np.ndarray, *, require_prediction_variance: bool
) -> dict[str, float | None]:
    left = truth.reshape(-1)
    right = prediction.reshape(-1)
    if float(np.std(left)) <= 0:
        raise RevisionProtocolError("Comparator truth vector is degenerate")
    if float(np.std(right)) <= 0:
        if require_prediction_variance:
            raise ComparatorScientificGateFailed("Comparator prediction vector is degenerate")
        pearson: float | None = None
    else:
        pearson = float(np.corrcoef(left, right)[0, 1])
        if not math.isfinite(pearson):
            raise RevisionProtocolError("Comparator Pearson recomputation is non-finite")
    residual = left - right
    return {
        "pearson_r": pearson,
        "mse": float(np.mean(np.square(residual))),
        "mae": float(np.mean(np.abs(residual))),
    }


def _read_gene_order(path: Path) -> tuple[str, ...]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise RevisionProtocolError("Comparator gene order must be a JSON list")
    genes = tuple(str(value) for value in payload)
    if not genes or len(set(genes)) != len(genes) or any(not value for value in genes):
        raise RevisionProtocolError("Comparator gene order is invalid")
    return genes


def _read_ordered_identifiers(path: Path, *, label: str) -> tuple[str, ...]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise RevisionProtocolError(f"Comparator {label} must be a JSON list")
    identifiers = tuple(payload)
    aliases = tuple(_condition_alias(value) for value in identifiers if isinstance(value, str))
    if (
        not identifiers
        or not all(
            isinstance(value, str) and value and value == value.strip() for value in identifiers
        )
        or len(set(identifiers)) != len(identifiers)
        or len(aliases) != len(identifiers)
        or len(set(aliases)) != len(aliases)
    ):
        raise RevisionProtocolError(f"Comparator {label} are invalid or aliased")
    return identifiers


def _condition_alias(value: str) -> str:
    return value.casefold().strip()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RevisionProtocolError(f"Invalid JSON evidence {path.name}: {error}") from error


def _resolve_bound_file(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value or "REPLACE" in value.upper():
        return None
    relative = Path(value)
    if relative.is_absolute():
        return None
    resolved = (root / relative).resolve()
    return resolved if resolved.is_relative_to(root) else None


def _withheld(comparator: str, reason_code: str, *, detail: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "comparator": comparator,
        "decision": "WITHHELD",
        "scientific_validity": "NOT_ESTABLISHED",
        "claim_deleted": False,
        "reason_code": reason_code,
    }
    if detail is not None:
        payload["detail"] = detail
    return payload


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant {value!r} is prohibited")

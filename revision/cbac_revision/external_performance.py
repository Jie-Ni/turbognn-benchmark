"""Raw-vector external performance gate against the frozen revision reference."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .artifacts import canonical_sha256, file_sha256
from .errors import RevisionProtocolError
from .external_validation import COMPARATORS
from .statistics import benjamini_hochberg

DATASETS = ("adamson", "norman", "replogle_k562", "replogle_rpe1")
HVG_SCALES = (200, 500, 1000)
REFERENCE_MODEL_ID = "revision_string_go"
PRIMARY_PANEL_ID = "primary_frozen_hvg_panel"
ESTIMAND = "adapter_minus_revision_string_go"
METRIC_DEFINITION = "per_condition_pearson_r_recomputed_from_raw_vectors"
EVIDENCE_FILE_ROLES = (
    "candidate_vectors",
    "truth_vectors",
    "revision_string_go_vectors",
    "row_conditions",
    "gene_order",
    "split",
    "target_mapping",
    "revision_string_go_artifact",
)
GENERIC_VALIDITY_LINK_ROLES = {
    "prediction_vectors_sha256": "candidate_vectors",
    "truth_vectors_sha256": "truth_vectors",
    "vector_row_conditions_sha256": "row_conditions",
    "split_manifest_sha256": "split",
    "target_mapping_manifest_sha256": "target_mapping",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def release_external_performance_family(
    validity_registry: Mapping[str, Any],
    *,
    manifest_path: Path | None,
    expected_manifest_sha256: str | None,
    bootstrap_replicates: int = 20_000,
    bootstrap_random_seed: int = 20260806,
) -> dict[str, Any]:
    """Release three adapter-minus-string_go contrasts from bound raw vectors."""

    _validate_validity_registry(validity_registry)
    decisions = dict(validity_registry["member_decisions"])
    validity_members = dict(validity_registry["members"])
    passing = [member for member in COMPARATORS if decisions[member] == "PASS"]
    if not passing and manifest_path is None and expected_manifest_sha256 is None:
        registry: dict[str, Any] = {
            "schema_version": "2.0",
            "registry_id": "EXTERNAL-COMPARATOR-SCALE-PERFORMANCE",
            "status": "RELEASED",
            "execution_status": "NO_PASSING_ADAPTERS_NO_PERFORMANCE_HYPOTHESES",
            "validity_registry_hash": validity_registry["registry_hash"],
            "passing_adapters": [],
            "adapter_results": {},
            "metric_definition": METRIC_DEFINITION,
            "estimand": ESTIMAND,
            "reference_model_id": REFERENCE_MODEL_ID,
            "multiplicity": "three_scale_tests_BH_within_each_passing_adapter",
            "cross_adapter_pooling": "PROHIBITED",
        }
        registry["registry_hash"] = canonical_sha256(registry)
        return registry
    if manifest_path is None or expected_manifest_sha256 is None:
        return _withheld(
            validity_registry,
            "PASSING_ADAPTER_PERFORMANCE_MANIFEST_OR_CALLER_PINNED_HASH_MISSING",
        )
    if (
        not SHA256_RE.fullmatch(expected_manifest_sha256)
        or not manifest_path.is_file()
        or file_sha256(manifest_path) != expected_manifest_sha256
    ):
        return _withheld(validity_registry, "PERFORMANCE_MANIFEST_CALLER_PIN_MISMATCH")
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return _withheld(validity_registry, "PERFORMANCE_MANIFEST_INVALID", str(error))
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "validity_registry_hash",
        "metric_definition",
        "estimand",
        "reference_model_id",
        "reference_registry",
        "members",
        "manifest_sha256",
    }:
        return _withheld(validity_registry, "PERFORMANCE_MANIFEST_SCHEMA_INVALID")
    unsigned = dict(manifest)
    declared_hash = unsigned.pop("manifest_sha256", None)
    if (
        declared_hash != canonical_sha256(unsigned)
        or manifest["schema_version"] != "2.0"
        or manifest["validity_registry_hash"] != validity_registry["registry_hash"]
        or manifest["metric_definition"] != METRIC_DEFINITION
        or manifest["estimand"] != ESTIMAND
        or manifest["reference_model_id"] != REFERENCE_MODEL_ID
        or not isinstance(manifest["members"], dict)
        or set(manifest["members"]) != set(COMPARATORS)
    ):
        return _withheld(validity_registry, "PERFORMANCE_MANIFEST_BINDING_INVALID")

    root = manifest_path.resolve().parent
    adapter_results: dict[str, Any] = {}
    try:
        reference_registry, reference_registry_binding = _read_reference_registry(
            root, manifest["reference_registry"]
        )
        for comparator in COMPARATORS:
            specification = manifest["members"][comparator]
            validity_member = validity_members[comparator]
            if not isinstance(specification, dict):
                raise RevisionProtocolError("External performance member specification invalid")
            member_payload_hash = canonical_sha256(dict(validity_member))
            if decisions[comparator] == "EXCLUDED":
                if (
                    set(specification) != {"mode", "reason_code", "validity_member_payload_sha256"}
                    or specification.get("mode") != "not_applicable_excluded"
                    or specification.get("validity_member_payload_sha256") != member_payload_hash
                    or specification.get("reason_code") != validity_member.get("reason_code")
                ):
                    raise RevisionProtocolError(
                        "Excluded comparator requires validity-bound reason-coded N/A performance"
                    )
                adapter_results[comparator] = {
                    "status": "NOT_APPLICABLE_EXCLUDED",
                    "reason_code": specification["reason_code"],
                    "validity_member_payload_sha256": member_payload_hash,
                }
                continue
            if decisions[comparator] != "PASS":
                raise RevisionProtocolError("WITHHELD comparator cannot enter performance family")
            if (
                set(specification)
                != {
                    "mode",
                    "validity_member_payload_sha256",
                    "validity_evidence_links",
                    "scale_evidence",
                }
                or specification.get("mode") != "validate_raw_vectors"
                or specification.get("validity_member_payload_sha256") != member_payload_hash
            ):
                raise RevisionProtocolError(
                    "PASS adapter requires validity-bound raw-vector evidence"
                )
            scale_evidence = specification.get("scale_evidence")
            if not isinstance(scale_evidence, dict) or set(scale_evidence) != {
                str(scale) for scale in HVG_SCALES
            }:
                raise RevisionProtocolError("PASS adapter requires exactly 200/500/1000 evidence")
            bundles: dict[tuple[int, str], dict[str, Any]] = {}
            condition_records: list[dict[str, Any]] = []
            file_bindings: list[dict[str, Any]] = []
            for hvg in HVG_SCALES:
                dataset_evidence = scale_evidence[str(hvg)]
                if not isinstance(dataset_evidence, dict) or set(dataset_evidence) != set(DATASETS):
                    raise RevisionProtocolError(
                        "Each external scale requires exactly the four frozen datasets"
                    )
                for dataset in DATASETS:
                    bundle = _read_bundle(
                        root,
                        dataset_evidence[dataset],
                        comparator=comparator,
                        dataset=dataset,
                        hvg=hvg,
                        reference_registry=reference_registry,
                    )
                    bundles[(hvg, dataset)] = bundle
                    condition_records.extend(bundle["condition_records"])
                    file_bindings.extend(bundle["file_bindings"])
            _validate_cross_scale_identity(bundles)
            verified_links = _validate_validity_evidence_links(
                comparator=comparator,
                member=validity_member,
                links=specification["validity_evidence_links"],
                bundles=bundles,
            )
            adapter_result = _summarize_adapter(
                condition_records,
                comparator=comparator,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_random_seed=bootstrap_random_seed,
                file_bindings=file_bindings,
            )
            adapter_result["validity_member_payload_sha256"] = member_payload_hash
            adapter_result["validity_evidence_links_verified"] = verified_links
            adapter_results[comparator] = adapter_result
    except RevisionProtocolError as error:
        return _withheld(validity_registry, "PERFORMANCE_EVIDENCE_INVALID", str(error))

    registry = {
        "schema_version": "2.0",
        "registry_id": "EXTERNAL-COMPARATOR-SCALE-PERFORMANCE",
        "status": "RELEASED",
        "execution_status": "PASS_ADAPTERS_ANALYSED_OR_EXCLUDED_NA",
        "validity_registry_hash": validity_registry["registry_hash"],
        "trusted_manifest_file_sha256": expected_manifest_sha256,
        "trusted_manifest_self_hash": manifest["manifest_sha256"],
        "metric_definition": METRIC_DEFINITION,
        "estimand": ESTIMAND,
        "reference_model_id": REFERENCE_MODEL_ID,
        "primary_panel_id": PRIMARY_PANEL_ID,
        "trusted_revision_string_go_registry_file_sha256": reference_registry_binding["sha256"],
        "trusted_revision_string_go_registry_self_hash": reference_registry["registry_sha256"],
        "passing_adapters": passing,
        "adapter_results": adapter_results,
        "multiplicity": "three_scale_tests_BH_within_each_passing_adapter",
        "cross_adapter_pooling": "PROHIBITED",
    }
    registry["registry_hash"] = canonical_sha256(registry)
    return registry


def combine_external_validity_and_performance(
    validity_registry: Mapping[str, Any], performance_registry: Mapping[str, Any]
) -> dict[str, Any]:
    """Create the headline external registry only when validity and performance cohere."""

    _validate_validity_registry(validity_registry)
    performance_payload = dict(performance_registry)
    performance_hash = performance_payload.pop("registry_hash", None)
    performance_valid = (
        performance_registry.get("registry_id") == "EXTERNAL-COMPARATOR-SCALE-PERFORMANCE"
        and performance_registry.get("status") == "RELEASED"
        and performance_hash == canonical_sha256(performance_payload)
        and performance_registry.get("validity_registry_hash") == validity_registry["registry_hash"]
        and performance_registry.get("estimand") == ESTIMAND
        and performance_registry.get("reference_model_id") == REFERENCE_MODEL_ID
        and performance_registry.get("metric_definition") == METRIC_DEFINITION
    )
    registry: dict[str, Any] = {
        "schema_version": "2.0",
        "registry_id": "EXTERNAL-COMPARATOR-VALIDATION",
        "status": "RELEASED" if performance_valid else "WITHHELD",
        "member_decisions": dict(validity_registry["member_decisions"]),
        "validity_registry": dict(validity_registry),
        "validity_registry_hash": validity_registry["registry_hash"],
        "performance_registry": dict(performance_registry),
        "performance_registry_hash": performance_hash,
        "estimand": ESTIMAND,
        "reference_model_id": REFERENCE_MODEL_ID,
        "reason_codes": [] if performance_valid else ["EXTERNAL_PERFORMANCE_FAMILY_WITHHELD"],
    }
    registry["registry_hash"] = canonical_sha256(registry)
    return registry


def _read_reference_registry(root: Path, binding: Any) -> tuple[dict[str, Any], dict[str, str]]:
    path, normalized_binding = _bound_file(root, binding, label="reference registry")
    payload = _read_json(path, label="revision string_go reference registry")
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "registry_id",
        "status",
        "reference_model_id",
        "estimand",
        "primary_panel_id",
        "entries",
        "registry_sha256",
    }:
        raise RevisionProtocolError("Revision string_go reference registry schema is invalid")
    unsigned = dict(payload)
    declared_hash = unsigned.pop("registry_sha256")
    entries = payload.get("entries")
    if (
        declared_hash != canonical_sha256(unsigned)
        or payload["schema_version"] != "1.0"
        or payload["registry_id"] != "REVISION-STRING-GO-REFERENCE"
        or payload["status"] != "RELEASED"
        or payload["reference_model_id"] != REFERENCE_MODEL_ID
        or payload["estimand"] != ESTIMAND
        or payload["primary_panel_id"] != PRIMARY_PANEL_ID
        or not isinstance(entries, dict)
        or set(entries) != {str(scale) for scale in HVG_SCALES}
        or any(
            not isinstance(entries[str(scale)], dict) or set(entries[str(scale)]) != set(DATASETS)
            for scale in HVG_SCALES
        )
    ):
        raise RevisionProtocolError("Revision string_go reference registry binding is invalid")
    return payload, normalized_binding


def _read_bundle(
    root: Path,
    specification: Any,
    *,
    comparator: str,
    dataset: str,
    hvg: int,
    reference_registry: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(specification, dict) or set(specification) != set(EVIDENCE_FILE_ROLES):
        raise RevisionProtocolError("Raw-vector evidence bundle schema is invalid")
    paths: dict[str, Path] = {}
    bindings: dict[str, dict[str, str]] = {}
    for role in EVIDENCE_FILE_ROLES:
        paths[role], bindings[role] = _bound_file(
            root, specification[role], label=f"{dataset}/{hvg}/{role}"
        )

    candidate = _numeric_matrix(paths["candidate_vectors"], label="candidate vectors")
    truth = _numeric_matrix(paths["truth_vectors"], label="truth vectors")
    reference = _numeric_matrix(
        paths["revision_string_go_vectors"], label="revision string_go vectors"
    )
    if candidate.shape != truth.shape or reference.shape != truth.shape or truth.shape != (50, hvg):
        raise RevisionProtocolError(
            "Candidate, truth, and revision string_go vectors must share frozen 50-by-HVG support"
        )
    row_conditions = _ordered_identifiers(
        _read_json(paths["row_conditions"], label="row conditions"),
        label="row conditions",
    )
    if len(row_conditions) != truth.shape[0]:
        raise RevisionProtocolError("Row-condition identities do not match vector rows")
    gene_order = _ordered_identifiers(
        _read_json(paths["gene_order"], label="gene order"), label="gene order"
    )
    if len(gene_order) != hvg:
        raise RevisionProtocolError("Gene order does not match the declared HVG scale")
    split = _validate_split(_read_json(paths["split"], label="split manifest"))
    if tuple(split["test_conditions"]) != row_conditions:
        raise RevisionProtocolError("Vector row order differs from the frozen test split order")
    target_mapping = _validate_target_mapping(
        _read_json(paths["target_mapping"], label="target mapping"),
        split=split,
        gene_order=gene_order,
    )
    artifact = _validate_reference_artifact(
        _read_json(paths["revision_string_go_artifact"], label="reference artifact"),
        dataset=dataset,
        hvg=hvg,
        bindings=bindings,
        gene_order=gene_order,
        split=split,
        target_mapping=target_mapping,
    )
    registry_entry = reference_registry["entries"][str(hvg)][dataset]
    _validate_reference_registry_entry(
        registry_entry,
        bindings=bindings,
        gene_order=gene_order,
        split=split,
        target_mapping=target_mapping,
        artifact=artifact,
    )

    condition_records = []
    for index, condition in enumerate(row_conditions):
        candidate_metric = _pearson(candidate[index], truth[index])
        reference_metric = _pearson(reference[index], truth[index])
        condition_records.append(
            {
                "dataset": dataset,
                "hvg": hvg,
                "condition": condition,
                "candidate_pearson_r": candidate_metric,
                "revision_string_go_pearson_r": reference_metric,
                "adapter_minus_revision_string_go": candidate_metric - reference_metric,
                "condition_target_sha256": canonical_sha256(
                    tuple(target_mapping["condition_targets"][condition])
                ),
                "gene_order_sha256": canonical_sha256(gene_order),
                "split_manifest_sha256": split["manifest_sha256"],
                "candidate_vectors_file_sha256": bindings["candidate_vectors"]["sha256"],
                "truth_vectors_file_sha256": bindings["truth_vectors"]["sha256"],
                "revision_string_go_vectors_file_sha256": bindings["revision_string_go_vectors"][
                    "sha256"
                ],
                "revision_string_go_artifact_sha256": artifact["artifact_sha256"],
                "revision_string_go_registry_sha256": reference_registry["registry_sha256"],
            }
        )
    file_bindings = [
        {
            "comparator": comparator,
            "dataset": dataset,
            "hvg": hvg,
            "role": role,
            **bindings[role],
        }
        for role in EVIDENCE_FILE_ROLES
    ]
    return {
        "dataset": dataset,
        "hvg": hvg,
        "row_conditions": row_conditions,
        "gene_order": gene_order,
        "split": split,
        "target_mapping": target_mapping,
        "file_hashes": {role: bindings[role]["sha256"] for role in EVIDENCE_FILE_ROLES},
        "condition_records": condition_records,
        "file_bindings": file_bindings,
    }


def _validate_reference_artifact(
    payload: Any,
    *,
    dataset: str,
    hvg: int,
    bindings: Mapping[str, Mapping[str, str]],
    gene_order: Sequence[str],
    split: Mapping[str, Any],
    target_mapping: Mapping[str, Any],
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "artifact_id",
        "reference_model_id",
        "dataset",
        "hvg",
        "primary_panel_id",
        "gene_order_sha256",
        "row_conditions_file_sha256",
        "split_manifest_sha256",
        "target_mapping_manifest_sha256",
        "truth_vectors_file_sha256",
        "prediction_vectors_file_sha256",
        "artifact_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise RevisionProtocolError("Revision string_go artifact schema is invalid")
    unsigned = dict(payload)
    declared_hash = unsigned.pop("artifact_sha256")
    if (
        declared_hash != canonical_sha256(unsigned)
        or payload["schema_version"] != "1.0"
        or payload["artifact_id"] != f"revision_string_go::{dataset}::{hvg}"
        or payload["reference_model_id"] != REFERENCE_MODEL_ID
        or payload["dataset"] != dataset
        or payload["hvg"] != hvg
        or payload["primary_panel_id"] != PRIMARY_PANEL_ID
        or payload["gene_order_sha256"] != canonical_sha256(tuple(gene_order))
        or payload["row_conditions_file_sha256"] != bindings["row_conditions"]["sha256"]
        or payload["split_manifest_sha256"] != split["manifest_sha256"]
        or payload["target_mapping_manifest_sha256"] != target_mapping["manifest_sha256"]
        or payload["truth_vectors_file_sha256"] != bindings["truth_vectors"]["sha256"]
        or payload["prediction_vectors_file_sha256"]
        != bindings["revision_string_go_vectors"]["sha256"]
    ):
        raise RevisionProtocolError("Revision string_go artifact binding is invalid")
    return dict(payload)


def _validate_reference_registry_entry(
    entry: Any,
    *,
    bindings: Mapping[str, Mapping[str, str]],
    gene_order: Sequence[str],
    split: Mapping[str, Any],
    target_mapping: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> None:
    expected = {
        "artifact_sha256": artifact["artifact_sha256"],
        "artifact_file_sha256": bindings["revision_string_go_artifact"]["sha256"],
        "revision_string_go_vectors_file_sha256": bindings["revision_string_go_vectors"]["sha256"],
        "truth_vectors_file_sha256": bindings["truth_vectors"]["sha256"],
        "row_conditions_file_sha256": bindings["row_conditions"]["sha256"],
        "gene_order_file_sha256": bindings["gene_order"]["sha256"],
        "gene_order_sha256": canonical_sha256(tuple(gene_order)),
        "split_manifest_sha256": split["manifest_sha256"],
        "target_mapping_manifest_sha256": target_mapping["manifest_sha256"],
    }
    if not isinstance(entry, dict) or entry != expected:
        raise RevisionProtocolError("Revision string_go registry entry does not bind the bundle")


def _validate_cross_scale_identity(bundles: Mapping[tuple[int, str], Mapping[str, Any]]) -> None:
    for dataset in DATASETS:
        ordered = [bundles[(hvg, dataset)] for hvg in HVG_SCALES]
        if any(bundle["row_conditions"] != ordered[0]["row_conditions"] for bundle in ordered[1:]):
            raise RevisionProtocolError("External condition identities differ across HVG scales")
        if any(
            bundle["split"]["manifest_sha256"] != ordered[0]["split"]["manifest_sha256"]
            for bundle in ordered[1:]
        ):
            raise RevisionProtocolError("External split identity differs across HVG scales")
        if any(
            bundle["target_mapping"]["condition_targets"]
            != ordered[0]["target_mapping"]["condition_targets"]
            for bundle in ordered[1:]
        ):
            raise RevisionProtocolError("External condition targets differ across HVG scales")
        genes_200, genes_500, genes_1000 = [bundle["gene_order"] for bundle in ordered]
        if genes_500[:200] != genes_200 or genes_1000[:500] != genes_500:
            raise RevisionProtocolError(
                "External primary HVG panels are not nested in frozen order"
            )


def _validate_validity_evidence_links(
    *,
    comparator: str,
    member: Mapping[str, Any],
    links: Any,
    bundles: Mapping[tuple[int, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if comparator == "GEARS":
        if links != {}:
            raise RevisionProtocolError(
                "GEARS validity payload exposes no raw-vector hashes for an invented cross-link"
            )
        return []
    expected_fields = set(GENERIC_VALIDITY_LINK_ROLES)
    if "gene_order_sha256" in member:
        expected_fields.add("gene_order_sha256")
    if not isinstance(links, dict) or set(links) != expected_fields:
        raise RevisionProtocolError("Generic comparator validity evidence links are incomplete")
    verified: list[dict[str, Any]] = []
    locations: set[tuple[int, str]] = set()
    for field in sorted(expected_fields):
        locator = links[field]
        if not isinstance(locator, dict) or set(locator) != {"hvg", "dataset", "role"}:
            raise RevisionProtocolError("Generic comparator validity evidence locator is invalid")
        hvg = locator["hvg"]
        dataset = locator["dataset"]
        expected_role = (
            "gene_order" if field == "gene_order_sha256" else GENERIC_VALIDITY_LINK_ROLES[field]
        )
        if (
            isinstance(hvg, bool)
            or hvg not in HVG_SCALES
            or dataset not in DATASETS
            or locator["role"] != expected_role
        ):
            raise RevisionProtocolError("Generic comparator validity evidence locator is invalid")
        bundle = bundles[(hvg, dataset)]
        if field == "split_manifest_sha256":
            observed = bundle["split"]["manifest_sha256"]
        elif field == "target_mapping_manifest_sha256":
            observed = bundle["target_mapping"]["manifest_sha256"]
        else:
            observed = bundle["file_hashes"][expected_role]
        if member.get(field) != observed:
            raise RevisionProtocolError(
                f"Performance evidence does not match validity member field {field}"
            )
        locations.add((hvg, dataset))
        verified.append(
            {
                "validity_field": field,
                "hvg": hvg,
                "dataset": dataset,
                "performance_role": expected_role,
                "verified_sha256": observed,
            }
        )
    if len(locations) != 1:
        raise RevisionProtocolError("Generic validity links must identify one coherent raw bundle")
    linked_hvg, linked_dataset = next(iter(locations))
    if member.get("n_genes") is not None and member.get("n_genes") != linked_hvg:
        raise RevisionProtocolError("Generic validity member gene count differs from linked panel")
    linked_bundle = bundles[(linked_hvg, linked_dataset)]
    verified.append(
        {
            "validity_field": "gene_order_via_target_mapping_manifest_sha256_and_n_genes",
            "hvg": linked_hvg,
            "dataset": linked_dataset,
            "performance_role": "gene_order",
            "verified_sha256": canonical_sha256(linked_bundle["gene_order"]),
        }
    )
    return verified


def _summarize_adapter(
    records: Sequence[Mapping[str, Any]],
    *,
    comparator: str,
    bootstrap_replicates: int,
    bootstrap_random_seed: int,
    file_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates < 100
    ):
        raise RevisionProtocolError("External performance bootstrap count is invalid")
    frame = pd.DataFrame.from_records(records)
    expected_rows = len(DATASETS) * len(HVG_SCALES) * 50
    if (
        len(frame) != expected_rows
        or frame.duplicated(["dataset", "hvg", "condition"]).any()
        or not np.isfinite(frame["adapter_minus_revision_string_go"].to_numpy(dtype=float)).all()
    ):
        raise RevisionProtocolError("Recomputed condition-level performance table is invalid")
    rng = np.random.default_rng(bootstrap_random_seed + COMPARATORS.index(comparator))
    contrasts = []
    p_values = []
    for hvg in HVG_SCALES:
        local = frame[frame["hvg"] == hvg]
        dataset_estimates = (
            local.groupby("dataset", sort=True)["adapter_minus_revision_string_go"]
            .mean()
            .astype(float)
            .to_dict()
        )
        draws = np.empty(bootstrap_replicates, dtype=float)
        for index in range(bootstrap_replicates):
            means = []
            for dataset in DATASETS:
                values = local.loc[
                    local["dataset"] == dataset, "adapter_minus_revision_string_go"
                ].to_numpy(dtype=float)
                means.append(float(np.mean(values[rng.integers(0, len(values), len(values))])))
            draws[index] = float(np.mean(means))
        low, high = np.quantile(draws, [0.025, 0.975])
        p_value = min(1.0, 2.0 * min(float(np.mean(draws <= 0)), float(np.mean(draws >= 0))))
        p_values.append(p_value)
        contrasts.append(
            {
                "hvg": hvg,
                "contrast": f"{comparator}_minus_{REFERENCE_MODEL_ID}",
                "estimand": ESTIMAND,
                "reference_model_id": REFERENCE_MODEL_ID,
                "estimate": float(np.mean(list(dataset_estimates.values()))),
                "uncertainty_interval_95_low": float(low),
                "uncertainty_interval_95_high": float(high),
                "interval_label": "95% conditional bootstrap uncertainty interval",
                "two_sided_p": p_value,
                "dataset_estimates": [
                    {"dataset": dataset, "estimate": dataset_estimates[dataset]}
                    for dataset in DATASETS
                ],
            }
        )
    q_values = benjamini_hochberg(p_values)
    for row, q_value in zip(contrasts, q_values, strict=True):
        row["bh_q_three_scale_family"] = float(q_value)
    sorted_records = sorted(
        (dict(record) for record in records),
        key=lambda row: (int(row["hvg"]), str(row["dataset"]), str(row["condition"])),
    )
    sorted_bindings = sorted(
        (dict(binding) for binding in file_bindings),
        key=lambda row: (
            int(row["hvg"]),
            str(row["dataset"]),
            str(row["role"]),
        ),
    )
    return {
        "status": "RELEASED",
        "estimand": ESTIMAND,
        "reference_model_id": REFERENCE_MODEL_ID,
        "metric_definition": METRIC_DEFINITION,
        "contrasts": contrasts,
        "family_denominator": 3,
        "adjustment": "benjamini_hochberg_within_adapter",
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_random_seed": bootstrap_random_seed + COMPARATORS.index(comparator),
        "condition_level_metrics": sorted_records,
        "source_file_bindings": sorted_bindings,
        "source_file_bindings_sha256": canonical_sha256(sorted_bindings),
        "source_records_sha256": canonical_sha256(sorted_records),
    }


def _validate_split(payload: Any) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "split_id",
        "training_conditions",
        "validation_conditions",
        "test_conditions",
        "target_disjoint",
        "manifest_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise RevisionProtocolError("External performance split schema is invalid")
    unsigned = dict(payload)
    declared_hash = unsigned.pop("manifest_sha256")
    partitions = []
    for field in ("training_conditions", "validation_conditions", "test_conditions"):
        values = _ordered_identifiers(payload[field], label=field)
        partitions.append(set(values))
    if (
        declared_hash != canonical_sha256(unsigned)
        or payload["schema_version"] != "1.0"
        or not isinstance(payload["split_id"], str)
        or not payload["split_id"].strip()
        or payload["target_disjoint"] is not True
        or any(
            partitions[left] & partitions[right]
            for left in range(3)
            for right in range(left + 1, 3)
        )
    ):
        raise RevisionProtocolError("External performance split binding is invalid")
    return dict(payload)


def _validate_target_mapping(
    payload: Any, *, split: Mapping[str, Any], gene_order: Sequence[str]
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "condition_targets",
        "gene_order_sha256",
        "manifest_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise RevisionProtocolError("External performance target-mapping schema is invalid")
    unsigned = dict(payload)
    declared_hash = unsigned.pop("manifest_sha256")
    mapping = payload.get("condition_targets")
    expected_conditions = set().union(
        *(
            set(split[field])
            for field in ("training_conditions", "validation_conditions", "test_conditions")
        )
    )
    if (
        declared_hash != canonical_sha256(unsigned)
        or payload["schema_version"] != "1.0"
        or payload["gene_order_sha256"] != canonical_sha256(tuple(gene_order))
        or not isinstance(mapping, dict)
        or set(mapping) != expected_conditions
    ):
        raise RevisionProtocolError("External performance target-mapping binding is invalid")
    normalized: dict[str, list[str]] = {}
    for condition, targets in mapping.items():
        if (
            not isinstance(targets, list)
            or not targets
            or len(set(targets)) != len(targets)
            or any(not isinstance(target, str) or not target.strip() for target in targets)
        ):
            raise RevisionProtocolError("External performance condition target is invalid")
        normalized[condition] = list(targets)
    partition_targets = {
        field: set().union(*(set(normalized[condition]) for condition in split[field]))
        for field in ("training_conditions", "validation_conditions", "test_conditions")
    }
    names = tuple(partition_targets)
    if any(
        partition_targets[names[left]] & partition_targets[names[right]]
        for left in range(len(names))
        for right in range(left + 1, len(names))
    ):
        raise RevisionProtocolError("External performance target-disjoint split is contradicted")
    return {
        **dict(payload),
        "manifest_sha256": declared_hash,
        "condition_targets": normalized,
    }


def _validate_validity_registry(registry: Mapping[str, Any]) -> None:
    payload = dict(registry)
    declared_hash = payload.pop("registry_hash", None)
    decisions = registry.get("member_decisions")
    members = registry.get("members")
    if (
        registry.get("registry_id") != "EXTERNAL-COMPARATOR-VALIDATION"
        or registry.get("status") != "RELEASED"
        or declared_hash != canonical_sha256(payload)
        or not isinstance(decisions, Mapping)
        or set(decisions) != set(COMPARATORS)
        or any(decision not in {"PASS", "EXCLUDED"} for decision in decisions.values())
        or not isinstance(members, Mapping)
        or set(members) != set(COMPARATORS)
        or any(
            not isinstance(members[member], Mapping)
            or members[member].get("decision") != decisions[member]
            for member in COMPARATORS
        )
    ):
        raise RevisionProtocolError("External validity registry is not released and coherent")


def _bound_file(root: Path, binding: Any, *, label: str) -> tuple[Path, dict[str, str]]:
    if not isinstance(binding, dict) or set(binding) != {"source_id", "sha256"}:
        raise RevisionProtocolError(f"External performance {label} binding is invalid")
    path = _resolve(root, binding["source_id"])
    expected_hash = binding["sha256"]
    if (
        path is None
        or not path.is_file()
        or not isinstance(expected_hash, str)
        or not SHA256_RE.fullmatch(expected_hash)
        or file_sha256(path) != expected_hash
    ):
        raise RevisionProtocolError(f"External performance {label} hash binding is invalid")
    return path, {"source_id": binding["source_id"], "sha256": expected_hash}


def _read_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RevisionProtocolError(f"External performance {label} JSON is invalid") from error


def _numeric_matrix(path: Path, *, label: str) -> np.ndarray:
    try:
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            loaded.close()
            raise RevisionProtocolError(f"External performance {label} must be a .npy array")
        values = np.asarray(loaded, dtype=np.float64)
    except (OSError, ValueError, TypeError) as error:
        raise RevisionProtocolError(f"External performance {label} are invalid") from error
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise RevisionProtocolError(f"External performance {label} must be a finite matrix")
    return values


def _ordered_identifiers(payload: Any, *, label: str) -> tuple[str, ...]:
    if (
        not isinstance(payload, list)
        or not payload
        or not all(isinstance(value, str) and value and value == value.strip() for value in payload)
        or len(set(payload)) != len(payload)
    ):
        raise RevisionProtocolError(f"External performance {label} are invalid")
    return tuple(payload)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    denominator = float(
        np.sqrt(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered))
    )
    if not math.isfinite(denominator) or denominator <= 0:
        raise RevisionProtocolError("Per-condition Pearson is undefined for a degenerate vector")
    value = float(np.dot(left_centered, right_centered) / denominator)
    if not math.isfinite(value) or value < -1.000000000001 or value > 1.000000000001:
        raise RevisionProtocolError("Per-condition Pearson recomputation is invalid")
    return float(np.clip(value, -1.0, 1.0))


def _resolve(root: Path, source_id: Any) -> Path | None:
    if not isinstance(source_id, str) or not source_id:
        return None
    source = Path(source_id)
    if source.is_absolute():
        return None
    candidate = (root / source).resolve()
    return candidate if candidate.is_relative_to(root) else None


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is prohibited: {value}")


def _withheld(
    validity_registry: Mapping[str, Any], reason_code: str, detail: str = ""
) -> dict[str, Any]:
    registry: dict[str, Any] = {
        "schema_version": "2.0",
        "registry_id": "EXTERNAL-COMPARATOR-SCALE-PERFORMANCE",
        "status": "WITHHELD",
        "validity_registry_hash": validity_registry.get("registry_hash"),
        "estimand": ESTIMAND,
        "reference_model_id": REFERENCE_MODEL_ID,
        "metric_definition": METRIC_DEFINITION,
        "reason_codes": [reason_code],
        "detail": detail,
    }
    registry["registry_hash"] = canonical_sha256(registry)
    return registry

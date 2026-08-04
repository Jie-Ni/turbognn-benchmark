"""Explicit condition-to-target mappings and target-preserving gene panels."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

import jsonschema

from .hashing import normalize_json_nfc, sha256_json_nfc
from .panels import canonical_condition_label


@dataclass(frozen=True)
class TargetMapSpec:
    """Source-verified target genes for every evaluated condition in one dataset."""

    dataset: str
    evidence: str
    condition_targets: Mapping[str, tuple[str, ...]]
    raw_to_canonical_id: Mapping[str, str] = field(default_factory=dict)
    canonical_objects: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    canonicalization_evidence: str = ""
    design_field_schema: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    allowed_perturbation_types: tuple[str, ...] = ()
    raw_normalization_evidence: Mapping[str, str] = field(default_factory=dict)
    raw_record_hashes: Mapping[str, str] = field(default_factory=dict)

    @property
    def mapping_hash(self) -> str:
        """Hash the complete dataset-specific mapping and evidence string."""
        return sha256_json_nfc(
            {
                "dataset": self.dataset,
                "evidence": self.evidence,
                "condition_targets": {
                    condition: list(targets)
                    for condition, targets in sorted(self.condition_targets.items())
                },
                "raw_to_canonical_id": dict(sorted(self.raw_to_canonical_id.items())),
                "canonical_objects": {
                    identifier: value
                    for identifier, value in sorted(self.canonical_objects.items())
                },
                "canonicalization_evidence": self.canonicalization_evidence,
                "design_field_schema": {
                    name: dict(spec) for name, spec in sorted(self.design_field_schema.items())
                },
                "allowed_perturbation_types": list(self.allowed_perturbation_types),
                "raw_normalization_evidence": dict(sorted(self.raw_normalization_evidence.items())),
                "raw_record_hashes": dict(sorted(self.raw_record_hashes.items())),
            }
        )


def _require_nfc_unpadded_string(value: object, field_name: str) -> str:
    """Validate an identifier/design string without silently changing its bytes."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    if canonical_condition_label(value) != value:
        raise ValueError(f"{field_name} must already be Unicode NFC")
    if value != value.strip():
        raise ValueError(f"{field_name} has forbidden leading or trailing whitespace")
    return value


def _canonical_decimal(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Canonical decimal values must be strings")
    text = value
    try:
        number = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(f"Invalid canonical decimal {text!r}") from error
    if not number.is_finite():
        raise ValueError("Canonical quantities must be finite")
    canonical = format(number.normalize(), "f")
    if canonical == "-0":
        canonical = "0"
    if text != canonical:
        raise ValueError(f"Quantity {text!r} is not the canonical decimal {canonical!r}")
    return canonical


def _validated_design_value(
    field_name: str, value: object, field_spec: Mapping[str, object]
) -> object:
    value_type = str(field_spec["type"])
    if value_type == "nfc_string":
        return _require_nfc_unpadded_string(value, f"Design field {field_name!r}")
    if value_type == "canonical_quantity":
        if not isinstance(value, Mapping) or set(value) != {"value", "unit"}:
            raise ValueError(f"Design field {field_name!r} must contain value/unit")
        unit = _require_nfc_unpadded_string(value["unit"], f"Design field {field_name!r} unit")
        if unit not in field_spec["controlled_units"]:
            raise ValueError(
                f"Design field {field_name!r} unit {unit!r} is outside its controlled allowlist"
            )
        return {"value": _canonical_decimal(value["value"]), "unit": unit}
    if value_type == "integer" and isinstance(value, int) and not isinstance(value, bool):
        return value
    if value_type == "boolean" and isinstance(value, bool):
        return value
    raise ValueError(f"Design field {field_name!r} violates frozen type {value_type!r}")


def _parse_dataset_target_map(dataset: str, value: object) -> TargetMapSpec:
    if not isinstance(value, Mapping):
        raise ValueError(f"Target mapping for {dataset!r} must be a JSON object")
    evidence = value.get("evidence")
    canonicalization_evidence = value.get("canonicalization_evidence")
    design_schema = value.get("design_field_schema")
    raw_allowed_types = value.get("allowed_perturbation_types")
    raw_conditions = value.get("raw_conditions")
    evidence = _require_nfc_unpadded_string(evidence, f"Target mapping evidence for {dataset!r}")
    if "TODO" in evidence.upper():
        raise ValueError(f"Target mapping for {dataset!r} contains placeholder evidence")
    if "TODO" in str(canonicalization_evidence).upper():
        raise ValueError(f"Target mapping for {dataset!r} lacks canonicalization evidence")
    canonicalization_evidence = _require_nfc_unpadded_string(
        canonicalization_evidence,
        f"Canonicalization evidence for {dataset!r}",
    )
    if not isinstance(design_schema, Mapping):
        raise ValueError(f"Target mapping for {dataset!r} lacks design_field_schema")
    if not isinstance(raw_allowed_types, list) or not raw_allowed_types:
        raise ValueError(f"Target mapping for {dataset!r} lacks allowed perturbation types")
    allowed_perturbation_types = tuple(
        _require_nfc_unpadded_string(value, "Allowed perturbation type")
        for value in raw_allowed_types
    )
    if allowed_perturbation_types != tuple(sorted(set(allowed_perturbation_types))):
        raise ValueError("Allowed perturbation types must be sorted and unique")
    allowed_types = {"nfc_string", "canonical_quantity", "integer", "boolean"}
    schema: dict[str, Mapping[str, object]] = {}
    for field_name, raw_field_spec in design_schema.items():
        canonical_name = _require_nfc_unpadded_string(field_name, "Design-field name")
        if not isinstance(raw_field_spec, Mapping) or "type" not in raw_field_spec:
            raise ValueError(f"Design field {canonical_name!r} lacks a typed schema object")
        canonical_type = _require_nfc_unpadded_string(
            raw_field_spec["type"], f"Type for design field {canonical_name!r}"
        )
        if canonical_type not in allowed_types:
            raise ValueError(f"Unsupported design-field type {canonical_type!r}")
        if canonical_type == "canonical_quantity":
            if set(raw_field_spec) != {"type", "controlled_units", "conversion_evidence"}:
                raise ValueError(
                    f"Quantity field {canonical_name!r} must freeze units and conversion evidence"
                )
            raw_units = raw_field_spec["controlled_units"]
            if not isinstance(raw_units, list) or not raw_units:
                raise ValueError(f"Quantity field {canonical_name!r} lacks controlled units")
            units = [
                _require_nfc_unpadded_string(unit, f"Controlled unit for {canonical_name!r}")
                for unit in raw_units
            ]
            if units != sorted(set(units)):
                raise ValueError(
                    f"Controlled units for {canonical_name!r} must be sorted and unique"
                )
            conversion_evidence = _require_nfc_unpadded_string(
                raw_field_spec["conversion_evidence"],
                f"Conversion evidence for {canonical_name!r}",
            )
            if "TODO" in conversion_evidence.upper():
                raise ValueError(f"Quantity field {canonical_name!r} has placeholder evidence")
            normalized_field_spec: Mapping[str, object] = {
                "type": canonical_type,
                "controlled_units": units,
                "conversion_evidence": conversion_evidence,
            }
        elif canonical_type == "nfc_string":
            if (
                set(raw_field_spec) != {"type", "source_value_policy"}
                or raw_field_spec.get("source_value_policy") != "retain_nfc_unmodified"
            ):
                raise ValueError(
                    f"String field {canonical_name!r} must retain the NFC source value"
                )
            normalized_field_spec = {
                "type": canonical_type,
                "source_value_policy": "retain_nfc_unmodified",
            }
        else:
            if set(raw_field_spec) != {"type"}:
                raise ValueError(f"Field {canonical_name!r} has unsupported schema metadata")
            normalized_field_spec = {"type": canonical_type}
        if canonical_name in schema:
            raise ValueError("Design-field names collide after NFC normalization")
        schema[canonical_name] = normalized_field_spec
    if not isinstance(raw_conditions, Mapping) or not raw_conditions:
        raise ValueError(f"Target mapping for {dataset!r} lacks raw_conditions")

    raw_to_canonical: dict[str, str] = {}
    canonical_objects: dict[str, Mapping[str, object]] = {}
    canonical_targets: dict[str, tuple[str, ...]] = {}
    raw_normalization_evidence: dict[str, str] = {}
    raw_record_hashes: dict[str, str] = {}
    for raw_condition, raw_record in raw_conditions.items():
        condition = _require_nfc_unpadded_string(
            raw_condition, f"Raw condition label in {dataset!r}"
        )
        if condition in raw_to_canonical:
            raise ValueError("Raw condition labels collide after NFC normalization")
        if not condition or "REPLACE" in condition.upper() or "TODO" in condition.upper():
            raise ValueError(f"Target mapping for {dataset!r} contains a placeholder condition")
        if not isinstance(raw_record, Mapping) or set(raw_record) != {
            "canonical_object",
            "target_genes",
            "normalization_evidence",
        }:
            raise ValueError(
                f"Raw condition {condition!r} must contain canonical_object, target_genes, "
                "and normalization_evidence"
            )
        canonical_object = raw_record["canonical_object"]
        if not isinstance(canonical_object, Mapping) or set(canonical_object) != {
            "target_ids",
            "target_identifier_namespace",
            "target_identifier_release",
            "target_mapping_table_sha256",
            "perturbation_type",
            "design_fields",
        }:
            raise ValueError(f"Raw condition {condition!r} has an invalid canonical object")
        raw_target_ids = canonical_object["target_ids"]
        if not isinstance(raw_target_ids, list) or not raw_target_ids:
            raise ValueError(f"Raw condition {condition!r} lacks stable target IDs")
        target_ids = [
            _require_nfc_unpadded_string(target, f"Stable target ID for {condition!r}")
            for target in raw_target_ids
        ]
        if target_ids != sorted(set(target_ids)):
            raise ValueError(
                f"Raw condition {condition!r} target IDs must be sorted, unique, and NFC"
            )
        target_identifier_namespace = _require_nfc_unpadded_string(
            canonical_object["target_identifier_namespace"],
            f"Target identifier namespace for {condition!r}",
        )
        target_identifier_release = _require_nfc_unpadded_string(
            canonical_object["target_identifier_release"],
            f"Target identifier release for {condition!r}",
        )
        target_mapping_table_sha256 = _require_nfc_unpadded_string(
            canonical_object["target_mapping_table_sha256"],
            f"Target mapping-table SHA256 for {condition!r}",
        )
        if len(target_mapping_table_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in target_mapping_table_sha256
        ):
            raise ValueError(
                f"Raw condition {condition!r} target_mapping_table_sha256 must be lowercase hex"
            )
        perturbation_type = _require_nfc_unpadded_string(
            canonical_object["perturbation_type"],
            f"Perturbation type for {condition!r}",
        )
        design_fields = canonical_object["design_fields"]
        if not isinstance(design_fields, Mapping) or set(design_fields) != set(schema):
            raise ValueError(f"Raw condition {condition!r} design fields differ from schema")
        validated_design = {
            field_name: _validated_design_value(field_name, design_fields[field_name], field_spec)
            for field_name, field_spec in sorted(schema.items())
        }
        normalized_object = normalize_json_nfc(
            {
                "target_ids": target_ids,
                "target_identifier_namespace": target_identifier_namespace,
                "target_identifier_release": target_identifier_release,
                "target_mapping_table_sha256": target_mapping_table_sha256,
                "perturbation_type": perturbation_type,
                "design_fields": validated_design,
            }
        )
        canonical_id = sha256_json_nfc(normalized_object)
        raw_genes = raw_record["target_genes"]
        if not isinstance(raw_genes, list) or not raw_genes:
            raise ValueError(f"Raw condition {condition!r} lacks target_genes")
        if len(raw_genes) != len(target_ids):
            raise ValueError(
                f"Raw condition {condition!r} must map every stable target ID to one matrix gene"
            )
        genes = tuple(
            _require_nfc_unpadded_string(
                gene, f"Target gene label for condition {condition!r} in {dataset!r}"
            )
            for gene in raw_genes
        )
        if any(not gene or "REPLACE" in gene.upper() for gene in genes):
            raise ValueError(
                f"Target mapping for condition {condition!r} in {dataset!r} contains a placeholder"
            )
        if len(set(genes)) != len(genes):
            raise ValueError(
                f"Target mapping for condition {condition!r} in {dataset!r} "
                "contains duplicate genes"
            )
        stable_id_to_matrix_gene = tuple(zip(target_ids, genes, strict=True))
        normalization = _require_nfc_unpadded_string(
            raw_record["normalization_evidence"],
            f"Normalization evidence for raw condition {condition!r}",
        )
        if "TODO" in normalization.upper():
            raise ValueError(f"Raw condition {condition!r} lacks normalization evidence")
        if canonical_id in canonical_objects:
            if canonical_objects[canonical_id] != normalized_object:
                raise ValueError("Canonical perturbation SHA-256 collision")
            if canonical_targets[canonical_id] != genes:
                raise ValueError("Aliases in one canonical group map to different target genes")
        canonical_objects[canonical_id] = normalized_object
        canonical_targets[canonical_id] = genes
        raw_to_canonical[condition] = canonical_id
        raw_normalization_evidence[condition] = normalization
        raw_record_hashes[condition] = sha256_json_nfc(
            {
                "raw_condition": condition,
                "canonical_perturbation_id": canonical_id,
                "target_genes": list(genes),
                "stable_id_to_matrix_gene": dict(stable_id_to_matrix_gene),
                "normalization_evidence": normalization,
            }
        )
    return TargetMapSpec(
        dataset=dataset,
        evidence=evidence,
        condition_targets=canonical_targets,
        raw_to_canonical_id=raw_to_canonical,
        canonical_objects=canonical_objects,
        canonicalization_evidence=canonicalization_evidence,
        design_field_schema=schema,
        allowed_perturbation_types=allowed_perturbation_types,
        raw_normalization_evidence=raw_normalization_evidence,
        raw_record_hashes=raw_record_hashes,
    )


def load_target_maps(path: Path) -> dict[str, TargetMapSpec]:
    """Load strict JSON target maps with no condition-string inference fallback."""
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("Target map must be a non-empty JSON object")
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "target_map.schema.json"
    with schema_path.open(encoding="utf-8") as handle:
        schema = json.load(handle)
    try:
        jsonschema.validate(raw, schema)
    except jsonschema.ValidationError as error:
        raise ValueError(f"Target map fails TDS-02 schema: {error.message}") from error
    return {
        str(dataset): _parse_dataset_target_map(str(dataset), value)
        for dataset, value in raw.items()
    }


def resolve_condition_targets(
    dataset: str,
    conditions: Iterable[str],
    available_genes: Iterable[str],
    target_maps: Mapping[str, TargetMapSpec],
) -> tuple[dict[str, tuple[str, ...]], str]:
    """Validate complete mapping coverage and exact target membership in the raw gene universe."""
    if dataset not in target_maps:
        raise ValueError(
            f"Dataset {dataset!r} has no explicit target mapping; "
            "label parsing fallback is forbidden"
        )
    spec = target_maps[dataset]
    requested = tuple(str(condition) for condition in conditions)
    missing_conditions = sorted(set(requested) - set(spec.condition_targets))
    if missing_conditions:
        raise ValueError(
            f"Target mapping for {dataset!r} lacks {len(missing_conditions)} evaluated conditions: "
            f"{missing_conditions[:10]}"
        )
    gene_universe = {str(gene) for gene in available_genes}
    resolved = {condition: spec.condition_targets[condition] for condition in requested}
    missing_genes = sorted(
        {gene for targets in resolved.values() for gene in targets if gene not in gene_universe}
    )
    if missing_genes:
        raise ValueError(
            f"Target mapping for {dataset!r} references genes absent from the raw matrix: "
            f"{missing_genes[:10]}"
        )
    return resolved, spec.mapping_hash


def target_preserving_gene_panel(
    *,
    all_genes: Sequence[str],
    ranked_control_hvgs: Sequence[str],
    condition_targets: Mapping[str, Sequence[str]],
    panel_size: int,
) -> tuple[str, ...]:
    """Fill a fixed-size panel with all targets plus control-only ranked HVGs."""
    if panel_size < 1:
        raise ValueError("panel_size must be positive")
    gene_order = tuple(str(gene) for gene in all_genes)
    if len(gene_order) != len(set(gene_order)):
        raise ValueError("Raw gene names must be unique")
    universe = set(gene_order)
    target_genes = {str(gene) for targets in condition_targets.values() for gene in targets}
    missing_targets = sorted(target_genes - universe)
    if missing_targets:
        raise ValueError(f"Target genes are absent from the raw matrix: {missing_targets[:10]}")
    if len(target_genes) > panel_size:
        raise ValueError(
            f"{len(target_genes)} distinct target genes exceed fixed panel size {panel_size}"
        )
    ranked = tuple(str(gene) for gene in ranked_control_hvgs)
    if len(ranked) != len(set(ranked)):
        raise ValueError("Control-only HVG ranking contains duplicates")
    unknown_ranked = sorted(set(ranked) - universe)
    if unknown_ranked:
        raise ValueError(f"Control-only HVG ranking contains unknown genes: {unknown_ranked[:10]}")

    selected = set(target_genes)
    for gene in ranked:
        if len(selected) >= panel_size:
            break
        selected.add(gene)
    if len(selected) != panel_size:
        raise ValueError(
            f"Could only construct {len(selected)} genes for requested panel size {panel_size}"
        )
    panel = tuple(gene for gene in gene_order if gene in selected)
    if not target_genes <= set(panel):
        raise RuntimeError("Target-preserving panel construction dropped a mapped target")
    return panel


def validate_condition_targets_in_panel(
    condition_targets: Mapping[str, Sequence[str]],
    gene_panel: Sequence[str],
) -> None:
    """Fail if any evaluated condition cannot create its complete perturbation mask."""
    panel = set(gene_panel)
    missing = {
        condition: sorted(set(targets) - panel)
        for condition, targets in condition_targets.items()
        if not set(targets) <= panel
    }
    if missing:
        raise ValueError(f"Evaluated conditions have targets outside the gene panel: {missing}")

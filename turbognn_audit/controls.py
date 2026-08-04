"""Explicit, dataset-specific control-label contracts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .panels import canonical_condition_label


@dataclass(frozen=True)
class ControlSpec:
    """Verified perturbation-column and control-label selection for one dataset."""

    perturbation_column: str
    control_label: str
    evidence: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], dataset: str) -> ControlSpec:
        """Parse and validate one control specification."""
        required = ("perturbation_column", "control_label", "evidence")
        if set(value) != set(required):
            raise ValueError(
                f"Control mapping for {dataset!r} must contain exactly {sorted(required)}"
            )
        missing = [field for field in required if not isinstance(value.get(field), str)]
        if missing:
            raise ValueError(f"Control mapping for {dataset!r} lacks string fields: {missing}")
        raw_control_label = value["control_label"]
        canonical_control_label = canonical_condition_label(raw_control_label)
        if canonical_control_label != raw_control_label:
            raise ValueError(f"Control mapping for {dataset!r} has a non-canonical label")
        spec = cls(
            perturbation_column=value["perturbation_column"],
            control_label=canonical_control_label,
            evidence=value["evidence"],
        )
        if not spec.perturbation_column or not spec.control_label or not spec.evidence:
            raise ValueError(f"Control mapping for {dataset!r} contains blank values")
        for name, text in (
            ("perturbation_column", spec.perturbation_column),
            ("control_label", spec.control_label),
            ("evidence", spec.evidence),
        ):
            if canonical_condition_label(text) != text or text != text.strip():
                raise ValueError(
                    f"Control mapping for {dataset!r} field {name!r} must be NFC and unpadded"
                )
        if "REPLACE" in spec.control_label.upper() or "TODO" in spec.evidence.upper():
            raise ValueError(f"Control mapping for {dataset!r} is still a placeholder")
        return spec


def load_control_map(path: Path) -> dict[str, ControlSpec]:
    """Load a JSON control map; no implicit defaults are permitted."""
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Control map must be a non-empty JSON object")
    parsed: dict[str, ControlSpec] = {}
    for dataset, value in raw.items():
        if (
            not isinstance(dataset, str)
            or not dataset
            or canonical_condition_label(dataset) != dataset
            or dataset != dataset.strip()
        ):
            raise ValueError("Control-map dataset identifiers must be non-empty, NFC, and unpadded")
        if not isinstance(value, dict):
            raise ValueError(f"Control mapping for {dataset!r} must be a JSON object")
        parsed[dataset] = ControlSpec.from_mapping(value, dataset)
    return parsed


def resolve_control(
    dataset: str,
    available_columns: Iterable[str],
    observed_labels: Iterable[object],
    control_map: Mapping[str, ControlSpec],
) -> ControlSpec:
    """Resolve and verify a control without guessing from label frequencies."""
    if dataset not in control_map:
        raise ValueError(
            f"Dataset {dataset!r} has no explicit control mapping; "
            "modal-label fallback is forbidden"
        )
    spec = control_map[dataset]
    columns = {str(column) for column in available_columns}
    if spec.perturbation_column not in columns:
        raise ValueError(
            f"Configured perturbation column {spec.perturbation_column!r} is absent from "
            f"dataset {dataset!r}; available columns: {sorted(columns)}"
        )
    labels = {str(label) for label in observed_labels}
    if spec.control_label not in labels:
        raise ValueError(
            f"Configured control label {spec.control_label!r} is absent from dataset {dataset!r}"
        )
    return spec

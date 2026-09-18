"""Strict control-label and condition-alias resolution.

The submitted pipeline guessed a control label from a short candidate list and then
fell back to the most frequent condition. This module deliberately has no frequency-
based fallback: the dataset protocol must name every accepted raw control label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .errors import AmbiguousAliasError, MissingControlError, RevisionProtocolError


@dataclass(frozen=True)
class ControlDefinition:
    """Predeclared raw labels that identify control cells in one dataset."""

    condition_column: str
    canonical_label: str
    aliases: tuple[str, ...] = ()
    case_sensitive: bool = False
    strip_whitespace: bool = True

    def accepted_labels(self) -> tuple[str, ...]:
        """Return unique accepted labels in declaration order."""

        labels = (self.canonical_label, *self.aliases)
        if not self.condition_column:
            raise RevisionProtocolError("condition_column must be explicit and non-empty")
        normalized = [self.normalize(label) for label in labels]
        if len(set(normalized)) != len(normalized):
            raise AmbiguousAliasError("Control aliases collapse to duplicate normalized labels")
        return labels

    def normalize(self, value: object) -> str:
        """Normalize a raw label exactly as declared by this definition."""

        if value is None or (isinstance(value, float) and np.isnan(value)):
            raise RevisionProtocolError("Missing condition labels are not permitted")
        label = str(value)
        if self.strip_whitespace:
            label = label.strip()
        if not self.case_sensitive:
            label = label.casefold()
        return label


@dataclass(frozen=True)
class ControlResolution:
    """Auditable result of matching labels against a control definition."""

    mask: np.ndarray
    matched_raw_labels: tuple[str, ...]
    accepted_raw_labels: tuple[str, ...]
    n_control_cells: int
    n_total_cells: int


def resolve_control_labels(
    labels: Sequence[object], definition: ControlDefinition
) -> ControlResolution:
    """Resolve controls using only explicitly listed labels.

    The function raises if no declared control is observed. It never guesses from
    prevalence, so a common perturbation cannot silently become the control group.
    """

    accepted_raw = definition.accepted_labels()
    accepted_normalized = {definition.normalize(label) for label in accepted_raw}
    raw_labels = np.asarray(labels, dtype=object)
    normalized = np.asarray([definition.normalize(value) for value in raw_labels], dtype=object)
    mask = np.isin(normalized, list(accepted_normalized))
    if not bool(mask.any()):
        observed = tuple(sorted({str(value) for value in raw_labels}))
        preview = ", ".join(repr(value) for value in observed[:12])
        raise MissingControlError(
            "None of the predeclared control labels were observed. "
            f"Accepted={accepted_raw!r}; observed sample=({preview})"
        )
    matched = tuple(sorted({str(value) for value in raw_labels[mask]}))
    return ControlResolution(
        mask=mask.astype(bool, copy=False),
        matched_raw_labels=matched,
        accepted_raw_labels=accepted_raw,
        n_control_cells=int(mask.sum()),
        n_total_cells=int(len(mask)),
    )


def canonicalize_conditions(
    labels: Sequence[object],
    definition: ControlDefinition,
    aliases: Mapping[str, str] | None = None,
) -> np.ndarray:
    """Return canonical condition labels using an explicit raw-to-canonical map.

    Undeclared perturbation labels are retained verbatim. Control labels are mapped
    to ``canonical_label``. An alias cannot redefine any declared control label.
    """

    resolution = resolve_control_labels(labels, definition)
    alias_map = dict(aliases or {})
    normalized_controls = {definition.normalize(label) for label in definition.accepted_labels()}
    normalized_aliases: dict[str, str] = {}
    for raw, canonical in alias_map.items():
        key = definition.normalize(raw)
        if key in normalized_controls and canonical != definition.canonical_label:
            raise AmbiguousAliasError(
                f"Alias {raw!r} attempts to redefine a declared control as {canonical!r}"
            )
        if key in normalized_aliases and normalized_aliases[key] != canonical:
            raise AmbiguousAliasError(f"Alias {raw!r} has incompatible canonical targets")
        normalized_aliases[key] = str(canonical)

    output: list[str] = []
    for index, value in enumerate(labels):
        if resolution.mask[index]:
            output.append(definition.canonical_label)
            continue
        normalized = definition.normalize(value)
        output.append(normalized_aliases.get(normalized, str(value).strip()))
    return np.asarray(output, dtype=object)

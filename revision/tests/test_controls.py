from __future__ import annotations

import numpy as np
import pytest

from cbac_revision.controls import (
    ControlDefinition,
    canonicalize_conditions,
    resolve_control_labels,
)
from cbac_revision.errors import AmbiguousAliasError, MissingControlError


def test_explicit_aliases_are_resolved_and_reported() -> None:
    definition = ControlDefinition(
        condition_column="gene",
        canonical_label="control",
        aliases=("ctrl", "non-targeting"),
    )
    resolution = resolve_control_labels(["CTRL", "TP53", "non-targeting"], definition)

    assert resolution.mask.tolist() == [True, False, True]
    assert resolution.n_control_cells == 2
    assert resolution.matched_raw_labels == ("CTRL", "non-targeting")


def test_missing_control_never_falls_back_to_majority_class() -> None:
    definition = ControlDefinition("gene", "control", aliases=("ctrl",))
    labels = ["TP53", "TP53", "TP53", "MYC"]

    with pytest.raises(MissingControlError):
        resolve_control_labels(labels, definition)


def test_condition_alias_cannot_redefine_control() -> None:
    definition = ControlDefinition("gene", "control", aliases=("ctrl",))

    with pytest.raises(AmbiguousAliasError):
        canonicalize_conditions(
            ["ctrl", "P53"], definition, aliases={"ctrl": "TP53", "P53": "TP53"}
        )


def test_condition_aliases_are_explicit_and_deterministic() -> None:
    definition = ControlDefinition("gene", "control", aliases=("ctrl",))
    observed = canonicalize_conditions(["ctrl", "P53", "MYC"], definition, aliases={"P53": "TP53"})

    np.testing.assert_array_equal(observed, ["control", "TP53", "MYC"])

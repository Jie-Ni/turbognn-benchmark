from __future__ import annotations

import pytest
import numpy as np

from cbac_revision.runner import _canonicalize_unordered_target_conditions, _condition_split
from cbac_revision.targets import TargetEncoder, UnencodableTargetError


def test_unencodable_target_has_stable_reason_code_and_no_zero_mask() -> None:
    encoding = TargetEncoder(["TP53", "MYC"]).encode("KRAS")

    assert encoding.success is False
    assert encoding.reason_code == "TARGET_NOT_IN_SELECTED_GENE_SPACE"
    with pytest.raises(UnencodableTargetError, match="TARGET_NOT_IN_SELECTED_GENE_SPACE"):
        encoding.mask(2)


def test_multi_target_condition_sets_every_declared_target() -> None:
    encoding = TargetEncoder(["TP53", "MYC", "KRAS"]).encode("TP53+KRAS")

    assert encoding.success is True
    assert encoding.canonical_targets == ("TP53", "KRAS")
    assert encoding.mask(3).tolist() == [True, False, True]


def test_norman_underscore_combinations_are_unordered_and_target_equivalent() -> None:
    encoder = TargetEncoder(
        ["CDKN1C", "CDKN1B", "TP53"],
        separators_pattern=r"[_+,;|]",
        gene_space_name="dataset_gene_universe",
    )
    labels, encodings, raw_members = _canonicalize_unordered_target_conditions(
        np.asarray(["control", "CDKN1C_CDKN1B", "CDKN1B_CDKN1C", "TP53"], dtype=object),
        "control",
        encoder,
    )

    assert labels.tolist() == ["control", "CDKN1B+CDKN1C", "CDKN1B+CDKN1C", "TP53"]
    assert set(encodings["CDKN1B+CDKN1C"].canonical_targets) == {"CDKN1B", "CDKN1C"}
    assert raw_members["CDKN1B+CDKN1C"] == ("CDKN1B_CDKN1C", "CDKN1C_CDKN1B")


def test_adamson_guide_suffix_is_removed_without_generic_underscore_splitting() -> None:
    encoder = TargetEncoder(
        ["CARS", "DNAJC19", "EIF2AK3"],
        separators_pattern=r"[+,;|]",
        condition_regex_pattern=r"^(?P<target>.+)_(?:pDS|pBA)[0-9]+$",
        condition_regex_group="target",
        gene_space_name="dataset_gene_universe",
    )

    cars = encoder.encode("CARS_pDS460")
    dnajc19 = encoder.encode("DNAJC19_pDS026")
    eif2ak3 = encoder.encode("EIF2AK3_pBA572")
    construct = encoder.encode("Gal4-4(mod)_pBA582")
    assert cars.success and cars.canonical_targets == ("CARS",)
    assert dnajc19.success and dnajc19.canonical_targets == ("DNAJC19",)
    assert eif2ak3.success and eif2ak3.canonical_targets == ("EIF2AK3",)
    assert not construct.success
    assert construct.reason_code == "TARGET_NOT_IN_DATASET_GENE_UNIVERSE"


def test_adamson_same_target_guides_remain_distinct_held_out_conditions() -> None:
    encoder = TargetEncoder(
        ["IER3IP1", "SEC61A1"],
        separators_pattern=r"[+,;|]",
        condition_regex_pattern=r"^(?P<target>.+)_(?:pDS|pBA)[0-9]+$",
        condition_regex_group="target",
        gene_space_name="dataset_gene_universe",
    )
    labels, encodings, _ = _canonicalize_unordered_target_conditions(
        np.asarray(["control", "IER3IP1_pDS002", "IER3IP1_pDS110"], dtype=object),
        "control",
        encoder,
        pool_target_equivalent_labels=False,
    )

    assert labels.tolist() == ["control", "IER3IP1_pDS002", "IER3IP1_pDS110"]
    assert encodings["IER3IP1_pDS002"].canonical_targets == ("IER3IP1",)
    assert encodings["IER3IP1_pDS110"].canonical_targets == ("IER3IP1",)


def test_norman_inert_ctrl_component_is_not_treated_as_a_gene_target() -> None:
    encoder = TargetEncoder(
        ["TP53", "MYC"],
        separators_pattern=r"[_+,;|]",
        ignored_target_tokens=("ctrl",),
    )

    encoding = encoder.encode("TP53_ctrl")
    assert encoding.success
    assert encoding.canonical_targets == ("TP53",)


def test_related_target_conditions_are_excluded_before_train_validation_split() -> None:
    encoder = TargetEncoder(["G0", "G1", "G2", "G3", "G4"])
    conditions = ["G0+G1", "G0+G2", "G3", "G4"]
    encodings = {condition: encoder.encode(condition) for condition in conditions}
    train, validation = _condition_split(
        conditions,
        "G0+G1",
        {
            "validation_fraction": 0.25,
            "validation_selection_seed": 20260806,
            "related_target_exclusion": "exclude_any_shared_target_with_held_out",
        },
        encodings,
    )

    assert "G0+G2" not in train + validation
    assert set(train + validation) == {"G3", "G4"}

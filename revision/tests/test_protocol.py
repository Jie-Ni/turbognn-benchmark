from __future__ import annotations

import copy
from pathlib import Path

import pytest

from cbac_revision.errors import RevisionProtocolError
from cbac_revision.protocol import REQUIRED_ARTIFACT_FIELDS, load_protocol, validate_protocol

PROTOCOL_PATH = Path(__file__).parents[1] / "protocol.yaml"


def test_frozen_protocol_carries_all_design_invariants() -> None:
    protocol = load_protocol(PROTOCOL_PATH)

    assert protocol["preprocessing"]["fit_scope"] == "control_only"
    assert protocol["preprocessing"]["forced_target_retention_scope"] == "selected_panel_only"
    assert protocol["graph_supports"]["fixed_architecture_across_modes"] is True
    assert protocol["statistics"]["aggregation_and_resampling_unit"] == "held_out_condition"
    assert REQUIRED_ARTIFACT_FIELDS <= set(protocol["artifacts"]["required_fields"])
    assert protocol["planned_fit_counts"]["mandatory_total_fits"] == 10_800
    assert protocol["planned_fit_counts"]["worst_case_total_fits"] == 12_000
    assert len(protocol["graph_supports"]["topology_null"]["replicate_seeds"]) == 10
    assert protocol["graph_supports"]["topology_null"]["minimum_swapped_edge_fraction"] == 0.80
    assert protocol["statistics"]["bootstrap_random_seed"] == 20260806
    assert protocol["evaluation"]["minimum_cells_per_condition"] == 20
    assert protocol["datasets"]["adamson"]["condition_column"] == "perturbation"
    for dataset in protocol["datasets"].values():
        assert dataset["condition_column"]
        assert dataset["control"]["canonical_label"]
        assert "fallback" not in dataset["control"]


def test_claim_gate_binds_absolute_skill_and_condition_ranking_consequences() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    requirements = protocol["statistics"]["primary_claim_gate"]
    assert "BASELINE_ABSOLUTE_SKILL_ELIGIBLE" in requirements["sparse_support_wording_requires"]
    assert (
        "CONDITION_RANKING_CONSEQUENCE_ELIGIBLE" in requirements["biological_edge_wording_requires"]
    )
    forged = copy.deepcopy(protocol)
    forged["statistics"]["primary_claim_gate"]["biological_edge_wording_requires"].remove(
        "BASELINE_ABSOLUTE_SKILL_ELIGIBLE"
    )
    with pytest.raises(RevisionProtocolError, match="claim gate"):
        validate_protocol(forged)


def test_trigger_rejects_optional_metadata_as_an_authoritative_component() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    forged = copy.deepcopy(protocol)
    forged["condition_panel"]["conditional_sensitivity_trigger"][
        "pathway_total_variation_distance_gt"
    ] = 0.1
    with pytest.raises(RevisionProtocolError, match="trigger thresholds"):
        validate_protocol(forged)


@pytest.mark.parametrize("missing", ["mixed_support", "external_performance"])
def test_hypothesis_registry_requires_mixed_and_external_performance_families(
    missing: str,
) -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    forged = copy.deepcopy(protocol)
    del forged["hypothesis_families"][missing]
    with pytest.raises(RevisionProtocolError, match="all nine families"):
        validate_protocol(forged)


def test_external_performance_bh_is_within_adapter_and_has_three_scales() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    family = protocol["hypothesis_families"]["external_performance"]
    assert family["denominator"] == "3_per_passing_adapter"
    assert len(family["scale_contrasts"]) == 3

    for field, value in (
        ("scale_contrasts", ["200_hvg_overall", "500_hvg_overall"]),
        ("adjustment", "benjamini_hochberg_across_all_adapters"),
        ("cross_adapter_pooling", "allowed"),
    ):
        forged = copy.deepcopy(protocol)
        forged["hypothesis_families"]["external_performance"][field] = value
        with pytest.raises(RevisionProtocolError, match="External performance"):
            validate_protocol(forged)


def test_external_invalid_evidence_is_withheld_not_excluded() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    family = protocol["external_comparator_validity_family"]
    assert family["invalid_or_incomplete_evidence_decision"] == "WITHHELD"
    assert family["completed_scientific_gate_failure_decision"] == "EXCLUDED"

    forged = copy.deepcopy(protocol)
    forged["external_comparator_validity_family"][
        "invalid_or_incomplete_evidence_decision"
    ] = "EXCLUDED"
    with pytest.raises(RevisionProtocolError, match="fail-closed"):
        validate_protocol(forged)


def test_revision_stage_provenance_requires_august_7_and_legacy_outcomes_disclosure() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    record = next(
        row
        for row in protocol["hyperparameter_provenance"]["records"]
        if row["parameter"] == "hidden_normalization"
    )
    assert record["selection_date"] == "2026-08-07"
    assert record["legacy_outcomes_seen"] is True

    for field, value in (
        ("selection_date", "2026-08-06"),
        ("legacy_outcomes_seen", False),
    ):
        forged = copy.deepcopy(protocol)
        forged_record = next(
            row
            for row in forged["hyperparameter_provenance"]["records"]
            if row["parameter"] == "hidden_normalization"
        )
        forged_record[field] = value
        with pytest.raises(RevisionProtocolError, match="provenance"):
            validate_protocol(forged)


def test_hyperparameter_ledger_rejects_value_type_duplicate_and_missing_rows() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    mutations = []

    wrong_type = copy.deepcopy(protocol)
    next(
        row
        for row in wrong_type["hyperparameter_provenance"]["records"]
        if row["parameter"] == "ridge_alpha"
    )["value"] = "1.0"
    mutations.append(wrong_type)

    duplicate = copy.deepcopy(protocol)
    duplicate["hyperparameter_provenance"]["records"][-1] = copy.deepcopy(
        duplicate["hyperparameter_provenance"]["records"][-2]
    )
    mutations.append(duplicate)

    missing = copy.deepcopy(protocol)
    missing["hyperparameter_provenance"]["records"].pop()
    mutations.append(missing)

    for forged in mutations:
        with pytest.raises(RevisionProtocolError, match="[Hh]yperparameter"):
            validate_protocol(forged)

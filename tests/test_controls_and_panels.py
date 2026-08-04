from __future__ import annotations

import json

import pytest

from turbognn_audit.controls import ControlSpec, load_control_map, resolve_control
from turbognn_audit.panels import (
    build_condition_eligibility_panel,
    build_condition_panel,
    limit_condition_panel,
)


def test_control_resolution_requires_explicit_mapping() -> None:
    with pytest.raises(ValueError, match="modal-label fallback is forbidden"):
        resolve_control("adamson", ["condition"], ["most_common", "other"], {})


def test_control_resolution_validates_configured_label() -> None:
    mapping = {
        "toy": ControlSpec(
            perturbation_column="condition",
            control_label="negative_control",
            evidence="Toy fixture metadata",
        )
    }
    assert (
        resolve_control("toy", ["condition"], ["negative_control", "KO_A"], mapping)
        == mapping["toy"]
    )
    with pytest.raises(ValueError, match="is absent"):
        resolve_control("toy", ["condition"], ["KO_A", "KO_B"], mapping)


def test_control_map_rejects_placeholders(tmp_path) -> None:
    path = tmp_path / "controls.json"
    path.write_text(
        json.dumps(
            {
                "adamson": {
                    "perturbation_column": "condition",
                    "control_label": "REPLACE_WITH_VERIFIED_LABEL",
                    "evidence": "TODO",
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="placeholder"):
        load_control_map(path)


def test_condition_panel_is_order_invariant_and_outcome_agnostic() -> None:
    first = build_condition_panel(
        "toy",
        {"control": 10, "KO_B": 3, "KO_A": 4, "KO_small": 1},
        "control",
    )
    second = build_condition_panel(
        "toy",
        {"KO_small": 1, "KO_A": 4, "control": 10, "KO_B": 3},
        "control",
    )
    assert first.conditions == ("KO_A", "KO_B")
    assert first.panel_hash == second.panel_hash


def test_condition_panel_hash_changes_with_eligibility_rule() -> None:
    counts = {"control": 10, "KO_A": 4, "KO_B": 2}
    panel_two = build_condition_panel("toy", counts, "control", minimum_cells=2)
    panel_three = build_condition_panel("toy", counts, "control", minimum_cells=3)
    assert panel_two.panel_hash != panel_three.panel_hash


def test_limited_panel_hashes_exact_deterministic_prefix() -> None:
    panel = build_condition_panel(
        "toy",
        {"control": 10, "KO_C": 2, "KO_A": 2, "KO_B": 2},
        "control",
    )
    limited = limit_condition_panel(panel, 2)
    assert len(limited.conditions) == 2
    assert set(limited.conditions) <= set(panel.conditions)
    assert limited.panel_hash != panel.panel_hash
    assert limit_condition_panel(panel, 2).panel_hash == limited.panel_hash


def test_panel_hash_binds_requested_selection_limit_even_when_candidates_are_fewer() -> None:
    panel = build_condition_panel(
        "toy",
        {"control": 10, "KO_A": 2, "KO_B": 2},
        "control",
    )
    limited_five = limit_condition_panel(panel, 5)
    limited_ten = limit_condition_panel(panel, 10)
    assert limited_five.conditions == limited_ten.conditions
    assert limited_five.panel_hash != limited_ten.panel_hash
    assert limited_five.selection_limit == 5


def test_all_eligibility_gates_run_before_panel_ranking() -> None:
    kwargs = {
        "dataset": "toy",
        "pre_qc_label_counts": {
            "control": 100,
            "unmapped_early": 30,
            "mapped_A": 20,
            "mapped_B": 21,
            "too_small": 19,
        },
        "post_qc_label_counts": {
            "control": 100,
            "unmapped_early": 30,
            "mapped_A": 20,
            "mapped_B": 21,
            "too_small": 19,
        },
        "control_label": "control",
        "minimum_cells": 20,
        "condition_targets": {
            "mapped_A": ("A",),
            "mapped_B": ("B",),
            "too_small": ("C",),
        },
        "available_genes": ("A", "B", "C"),
        "cell_qc_policy": "source_filtered_matrix_no_additional_cell_filter",
        "perturbation_type_policy": "explicit_mapped_single_or_multi_target",
    }
    first = build_condition_eligibility_panel(**kwargs)
    reversed_kwargs = {
        **kwargs,
        "pre_qc_label_counts": dict(reversed(kwargs["pre_qc_label_counts"].items())),
        "post_qc_label_counts": dict(reversed(kwargs["post_qc_label_counts"].items())),
    }
    second = build_condition_eligibility_panel(**reversed_kwargs)
    assert first.conditions == ("mapped_A", "mapped_B")
    assert first.panel_hash == second.panel_hash
    assert first.eligibility_ledger_hash == second.eligibility_ledger_hash
    states = {row.condition: (row.terminal_state, row.reason) for row in first.eligibility_ledger}
    assert states["unmapped_early"] == ("ineligible", "explicit_target_mapping_missing")
    assert states["too_small"] == (
        "ineligible",
        "post_qc_cell_count_below_frozen_minimum_20",
    )
    assert len(states) == len(kwargs["pre_qc_label_counts"])

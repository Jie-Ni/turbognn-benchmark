from __future__ import annotations

import pandas as pd

from cbac_revision.panels import build_condition_panels


def test_metadata_stratified_panels_are_deterministic_and_non_overlapping() -> None:
    conditions = [f"G{index:03d}" for index in range(120)]
    labels = [label for condition in conditions for label in (condition, condition)]
    metadata = pd.DataFrame(
        {
            "perturbation_class": [
                "class_a" if index % 2 == 0 else "class_b" for index in range(120) for _ in range(2)
            ],
            "pathway_class": [f"pathway_{index % 3}" for index in range(120) for _ in range(2)],
        }
    )

    first = build_condition_panels(
        labels,
        "control",
        metadata,
        panel_size=50,
        descriptive_metadata_columns=("perturbation_class", "pathway_class"),
        trigger_total_variation_threshold=0.10,
        trigger_cell_count_threshold=0.25,
    )
    second = build_condition_panels(
        labels,
        "control",
        metadata,
        panel_size=50,
        descriptive_metadata_columns=("perturbation_class", "pathway_class"),
        trigger_total_variation_threshold=0.10,
        trigger_cell_count_threshold=0.25,
    )

    assert len(first.primary) == 50
    assert len(first.sensitivity) == 50
    assert set(first.primary).isdisjoint(first.sensitivity)
    assert first.primary == second.primary
    assert first.sensitivity == second.sensitivity
    overlap = first.representativeness[
        first.representativeness["statistic"] == "overlap_condition_count"
    ]
    assert (overlap["value"] == 0).all()
    assert set(first.representativeness.columns) == {
        "dataset",
        "panel",
        "variable",
        "level",
        "statistic",
        "value",
        "status",
    }
    numeric_statistics = set(
        first.representativeness.loc[
            (first.representativeness["panel"] == "primary")
            & (first.representativeness["variable"] == "condition_cell_count")
            & (first.representativeness["level"] == "__OVERALL__"),
            "statistic",
        ]
    )
    assert {
        "n",
        "mean",
        "sd",
        "median",
        "q1",
        "q3",
        "iqr",
        "universe_n",
        "universe_mean",
        "universe_sd",
        "universe_median",
        "universe_q1",
        "universe_q3",
        "universe_iqr",
    }.issubset(numeric_statistics)
    assert "sensitivity_triggered" in first.sensitivity_trigger


def test_perturbation_order_stratum_is_derived_for_combinations() -> None:
    labels = ["TP53", "TP53", "TP53+MYC", "TP53+MYC", "KRAS", "KRAS", "EGFR", "EGFR"]
    panels = build_condition_panels(
        labels,
        "control",
        pd.DataFrame(index=range(len(labels))),
        panel_size=2,
        trigger_total_variation_threshold=0.10,
        trigger_cell_count_threshold=0.25,
    )
    order = dict(
        zip(panels.condition_table["condition"], panels.condition_table["perturbation_order"])
    )

    assert order["TP53"] == "single"
    assert order["TP53+MYC"] == "combination"


def test_descriptive_pathway_and_class_fields_do_not_change_panel_selection() -> None:
    labels: list[str] = []
    classes: list[str] = []
    for index in range(40):
        condition = f"G{index}"
        n_cells = index + 1
        labels.extend([condition] * n_cells)
        classes.extend(["rare" if index == 0 else "common"] * n_cells)
    with_descriptives = build_condition_panels(
        labels,
        "control",
        pd.DataFrame({"perturbation_class": classes}),
        panel_size=10,
        descriptive_metadata_columns=("perturbation_class",),
        trigger_total_variation_threshold=0.10,
        trigger_cell_count_threshold=0.25,
    )
    without_descriptives = build_condition_panels(
        labels,
        "control",
        pd.DataFrame(index=range(len(labels))),
        panel_size=10,
        trigger_total_variation_threshold=0.10,
        trigger_cell_count_threshold=0.25,
    )

    assert with_descriptives.primary == without_descriptives.primary
    assert with_descriptives.sensitivity == without_descriptives.sensitivity


def test_optional_pathway_imbalance_is_measured_but_remains_descriptive() -> None:
    conditions = [f"G{index:03d}" for index in range(40)]
    labels = [condition for condition in conditions for _ in range(2)]
    baseline = build_condition_panels(
        labels,
        "control",
        pd.DataFrame(index=range(len(labels))),
        panel_size=10,
        trigger_total_variation_threshold=0.10,
        trigger_cell_count_threshold=0.25,
    )
    primary = set(baseline.primary)
    pathways = [
        "selected_pathway" if condition in primary else "background_pathway"
        for condition in conditions
        for _ in range(2)
    ]

    audited = build_condition_panels(
        labels,
        "control",
        pd.DataFrame({"pathway_class": pathways}),
        panel_size=10,
        descriptive_metadata_columns=("pathway_class", "perturbation_class"),
        trigger_total_variation_threshold=0.10,
        trigger_cell_count_threshold=0.25,
        dataset_name="toy",
    )
    pathway_tv = audited.representativeness[
        (audited.representativeness["panel"] == "primary")
        & (audited.representativeness["variable"] == "pathway_class")
        & (audited.representativeness["statistic"] == "variable_total_variation_distance")
    ]
    unavailable = audited.representativeness[
        (audited.representativeness["variable"] == "perturbation_class")
        & (audited.representativeness["status"] == "NOT_AVAILABLE")
    ]

    assert float(pathway_tv.iloc[0]["value"]) > 0.10
    assert audited.sensitivity_trigger["sensitivity_triggered"] is False
    assert audited.sensitivity_trigger["trigger_maximum_total_variation_variable"] == (
        "combined_selection_stratum"
    )
    assert (
        "pathway_class"
        not in audited.sensitivity_trigger["trigger_variable_total_variation_distances"]
    )
    assert len(unavailable) == 2

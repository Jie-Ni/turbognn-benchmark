from __future__ import annotations

import pandas as pd
import pytest
import yaml
from pathlib import Path

from cbac_revision.errors import IncompleteSeedSetError, UnpairedConditionError
from cbac_revision.statistics import (
    benjamini_hochberg,
    bootstrap_settings_from_protocol,
    paired_condition_contrasts,
    seed_average_by_condition,
    summarize_paired_contrast,
    summarize_stratified_paired_contrast,
)


def test_bootstrap_settings_are_loaded_from_protocol_not_hidden_defaults() -> None:
    protocol = yaml.safe_load(
        (Path(__file__).parents[1] / "protocol.yaml").read_text(encoding="utf-8")
    )

    assert bootstrap_settings_from_protocol(protocol) == {
        "n_bootstrap": 20_000,
        "random_seed": 20260806,
    }


def _frame() -> pd.DataFrame:
    rows = []
    arm_offsets = {"combined": 0.10, "dense": 0.00}
    condition_offsets = {"TP53": 0.20, "MYC": 0.40}
    for arm, arm_offset in arm_offsets.items():
        for condition, condition_offset in condition_offsets.items():
            for seed_offset, seed in enumerate((42, 43, 44)):
                rows.append(
                    {
                        "dataset": "adamson",
                        "hvg": 200,
                        "arm": arm,
                        "condition": condition,
                        "seed": seed,
                        "pearson_r": condition_offset + arm_offset + seed_offset * 0.01,
                    }
                )
    return pd.DataFrame(rows)


def test_seed_averaging_returns_conditions_not_seed_rows() -> None:
    averaged = seed_average_by_condition(_frame(), ["pearson_r"])

    assert len(averaged) == 4
    assert set(averaged["n_seeds"]) == {3}
    assert set(averaged["seed_ids"]) == {"42|43|44"}


def test_paired_contrast_uses_two_distinct_held_out_conditions() -> None:
    paired = paired_condition_contrasts(
        _frame(), "combined", "dense", "pearson_r", expected_seeds=(42, 43, 44)
    )
    summary = summarize_paired_contrast(
        paired, "combined", "dense", "pearson_r", n_bootstrap=500, random_seed=7
    )

    assert len(paired) == 2
    assert paired["delta_pearson_r"].round(12).tolist() == [0.1, 0.1]
    assert summary.n_conditions == 2
    assert summary.mean_difference == pytest.approx(0.1)


def test_incomplete_seed_set_is_not_silently_used() -> None:
    incomplete = _frame().iloc[:-1].copy()

    with pytest.raises(IncompleteSeedSetError):
        seed_average_by_condition(incomplete, ["pearson_r"])


def test_unpaired_condition_is_not_silently_dropped() -> None:
    unpaired = _frame()
    mask = ~((unpaired["arm"] == "dense") & (unpaired["condition"] == "MYC"))

    with pytest.raises(UnpairedConditionError):
        paired_condition_contrasts(unpaired[mask], "combined", "dense", "pearson_r")


def test_equal_weight_dataset_bootstrap_resamples_conditions_within_stratum() -> None:
    second_dataset = _frame().assign(dataset="norman")
    second_dataset["pearson_r"] = (
        second_dataset["pearson_r"] + (second_dataset["arm"] == "combined").astype(float) * 0.10
    )
    frame = pd.concat([_frame(), second_dataset], ignore_index=True)
    paired = paired_condition_contrasts(frame, "combined", "dense", "pearson_r")
    summary = summarize_stratified_paired_contrast(
        paired, "combined", "dense", "pearson_r", n_bootstrap=500, random_seed=7
    )

    assert summary.n_datasets == 2
    assert summary.n_conditions == 4
    assert summary.equal_weight_dataset_mean_difference == pytest.approx(0.15)


def test_benjamini_hochberg_is_monotone_in_ranked_order() -> None:
    adjusted = benjamini_hochberg([0.04, 0.001, 0.03, 0.5])

    assert adjusted[1] == pytest.approx(0.004)
    assert adjusted[2] <= adjusted[0] <= adjusted[3]

from __future__ import annotations

import pandas as pd
import pytest

from turbognn_audit.inference import (
    centered_residual_q_test,
    dataset_sign_flip_holm_family,
    holm_adjust,
    sign_flip_equal_dataset_test,
)

DATASETS = ("d1", "d2", "d3", "d4")


def _effects(values: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"dataset": dataset, "condition": f"c{index}", "delta_z": value}
            for dataset, dataset_values in values.items()
            for index, value in enumerate(dataset_values)
        ]
    )


def test_sign_flip_uses_plus_one_and_inclusive_ties_with_pcg64() -> None:
    frame = _effects({dataset: [0.0, 0.0] for dataset in DATASETS})
    result = sign_flip_equal_dataset_test(
        frame,
        datasets=DATASETS,
        draws=50,
        seed=20_260_804,
    )
    assert result.extreme_draw_n == 50
    assert result.p_value == 1.0
    assert result.rng == "PCG64"


def test_sign_flip_is_invariant_to_input_row_permutation() -> None:
    frame = _effects({dataset: [0.2, -0.1, 0.3] for dataset in DATASETS})
    first = sign_flip_equal_dataset_test(
        frame.sample(frac=1.0, random_state=1), datasets=DATASETS, draws=100, seed=8
    )
    second = sign_flip_equal_dataset_test(
        frame.sample(frac=1.0, random_state=2), datasets=DATASETS, draws=100, seed=8
    )
    assert first == second


def test_dataset_sign_flip_family_is_deterministic_and_holm_adjusted() -> None:
    frame = _effects(
        {
            "d1": [1.0, 1.0, 1.0],
            "d2": [0.5, 0.5, 0.5],
            "d3": [0.1, -0.1, 0.1],
            "d4": [0.0, 0.0, 0.0],
        }
    )
    first = dataset_sign_flip_holm_family(frame, datasets=DATASETS, draws=100, seed=7)
    second = dataset_sign_flip_holm_family(frame, datasets=DATASETS, draws=100, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert (first["holm_p"] >= first["raw_p"]).all()
    assert first["dataset"].tolist() == list(DATASETS)


def test_centered_residual_q_bootstrap_is_deterministic() -> None:
    frame = _effects(
        {
            "d1": [1.0, 1.1, 0.9],
            "d2": [0.5, 0.6, 0.4],
            "d3": [0.0, 0.1, -0.1],
            "d4": [-0.5, -0.4, -0.6],
        }
    )
    first = centered_residual_q_test(frame, datasets=DATASETS, draws=100, seed=20_260_805)
    second = centered_residual_q_test(frame, datasets=DATASETS, draws=100, seed=20_260_805)
    assert first == second
    assert first.statistic > 0
    assert 0 < first.p_value <= 1


def test_holm_adjustment_rejects_invalid_values() -> None:
    assert holm_adjust({"a": 0.01, "b": 0.04}) == pytest.approx({"a": 0.02, "b": 0.04})
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm_adjust({"bad": 2.0})

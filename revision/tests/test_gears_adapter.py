from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from run_gears_positive_control import _baseline_results, _parse_epoch_history


def test_epoch_history_requires_every_official_epoch() -> None:
    train, validation = _parse_epoch_history(
        [
            "Start Training...",
            "Epoch 1: Train Overall MSE: 1.2500 Validation Overall MSE: 1.5000. ",
            "Epoch 2: Train Overall MSE: 0.7500 Validation Overall MSE: 0.9000. ",
        ],
        2,
    )

    assert train == [1.25, 0.75]
    assert validation == [1.5, 0.9]
    with pytest.raises(RuntimeError, match="history is incomplete"):
        _parse_epoch_history(["Epoch 1: Train Overall MSE: 1.0 Validation Overall MSE: 1.1."], 2)


def test_control_mean_baseline_preserves_official_de_row_alignment() -> None:
    results = {
        "pert_cat": np.asarray(["A", "B"]),
        "truth": np.asarray([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]),
        "truth_de": np.asarray([[1.0, 3.0], [2.5, 3.5]]),
    }
    loader = [SimpleNamespace(de_idx=[np.asarray([0, 2]), np.asarray([1, 2])])]

    baseline = _baseline_results(results, loader, np.asarray([0.1, 0.2, 0.3]))

    np.testing.assert_allclose(baseline["pred"], [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    np.testing.assert_allclose(baseline["pred_de"], [[0.1, 0.3], [0.2, 0.3]])

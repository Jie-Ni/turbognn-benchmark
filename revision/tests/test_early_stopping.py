from __future__ import annotations

import math

import pytest

from cbac_revision.artifacts import replay_early_stopping
from cbac_revision.errors import ArtifactValidationError


def test_replay_keeps_selected_epoch_when_raw_minimum_is_below_by_less_than_delta() -> None:
    observation = replay_early_stopping(
        (1.0, 0.95), minimum_delta=0.1, patience=5, epochs_requested=2
    )

    assert observation["best_epoch_zero_based"] == 0
    assert observation["best_validation_loss"] == 1.0
    assert observation["trace"][1]["qualified_improvement"] is False


def test_replay_treats_exact_minimum_delta_threshold_as_non_improvement() -> None:
    observation = replay_early_stopping(
        (1.0, 0.9), minimum_delta=0.1, patience=5, epochs_requested=2
    )

    assert observation["best_epoch_zero_based"] == 0
    assert observation["trace"][1]["improvement_threshold_exclusive"] == pytest.approx(0.9)
    assert observation["trace"][1]["qualified_improvement"] is False


def test_replay_selects_strictly_qualifying_improvement() -> None:
    observation = replay_early_stopping(
        (1.0, 0.899), minimum_delta=0.1, patience=5, epochs_requested=2
    )

    assert observation["best_epoch_zero_based"] == 1
    assert observation["best_validation_loss"] == pytest.approx(0.899)
    assert observation["trace"][1]["qualified_improvement"] is True


def test_replay_treats_ties_as_non_improvements() -> None:
    observation = replay_early_stopping(
        (1.0, 1.0), minimum_delta=0.0, patience=2, epochs_requested=2
    )

    assert observation["best_epoch_zero_based"] == 0
    assert observation["trace"][1]["epochs_without_improvement_after_epoch"] == 1


def test_replay_records_exact_patience_termination_epoch_and_trace() -> None:
    observation = replay_early_stopping(
        (1.0, 0.95, 0.94), minimum_delta=0.1, patience=2, epochs_requested=10
    )

    assert observation["stop_reason"] == "PATIENCE_EXHAUSTED"
    assert observation["stopped_epoch_zero_based"] == 2
    assert [row["epochs_without_improvement_after_epoch"] for row in observation["trace"]] == [
        0,
        1,
        2,
    ]
    assert observation["trace"][-1]["stop_triggered_after_epoch"] is True


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_replay_rejects_nonfinite_validation_history(value: float) -> None:
    with pytest.raises(ArtifactValidationError, match="non-finite"):
        replay_early_stopping((1.0, value), minimum_delta=0.1, patience=2, epochs_requested=2)


def test_replay_rejects_history_continuing_after_patience_termination() -> None:
    with pytest.raises(ArtifactValidationError, match="after the patience stop"):
        replay_early_stopping((1.0, 1.0, 0.5), minimum_delta=0.0, patience=1, epochs_requested=3)


def test_replay_rejects_unexplained_short_history() -> None:
    with pytest.raises(ArtifactValidationError, match="ended before"):
        replay_early_stopping((1.0, 0.5), minimum_delta=0.0, patience=10, epochs_requested=3)

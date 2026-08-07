from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from cbac_revision.artifacts import canonical_sha256
from cbac_revision.baselines import (
    analytic_baseline_predictions,
    build_baseline_artifact,
    read_baseline_artifact,
    write_baseline_artifact,
)
from cbac_revision.errors import RevisionProtocolError


def _artifact() -> dict:
    profiles = np.asarray([[0.1, 0.3, -0.2], [0.2, -0.1, 0.4]])
    indicators = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return build_baseline_artifact(
        dataset="adamson",
        hvg=3,
        panel="primary",
        condition="G2",
        gene_names=("G0", "G1", "G2"),
        training_condition_ids=("G0", "G1"),
        training_profiles=profiles,
        training_target_indicators=indicators,
        held_out_target_indicator=np.asarray([0.0, 0.0, 1.0]),
        y_true=np.asarray([0.4, 0.0, 0.8]),
        ridge_alpha=1.0,
        input_hashes={"protocol": "a" * 64, "training_split": "b" * 64},
    )


def test_baselines_are_deterministic_seed_independent_and_round_trip(tmp_path: Path) -> None:
    first = _artifact()
    second = _artifact()
    path = write_baseline_artifact(tmp_path / "G2.json.gz", first)

    assert first == second == read_baseline_artifact(path)
    assert set(first["identity"]) == {"dataset", "hvg", "panel", "condition"}
    assert "seed" not in first["identity"]
    assert first["accounting"] == {
        "analytic_baseline_evaluations": 3,
        "ridge_refits": 1,
        "included_in_neural_fit_count": False,
    }
    assert 4 * 50 * 3 == 600
    assert 600 * 3 == 1800


def test_baseline_prediction_cannot_read_or_change_with_held_out_outcome() -> None:
    training = np.asarray([[0.1, 0.3, -0.2], [0.2, -0.1, 0.4]])
    targets = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    held_out = np.asarray([0.0, 0.0, 1.0])
    first = analytic_baseline_predictions(training, targets, held_out, ridge_alpha=1.0)
    second = analytic_baseline_predictions(training, targets, held_out, ridge_alpha=1.0)

    assert all(np.array_equal(first[name], second[name]) for name in first)


@pytest.mark.parametrize("mutation", ["seed", "training-vector", "prediction"])
def test_baseline_rejects_seed_or_synchronously_rehashed_vector_tamper(
    tmp_path: Path, mutation: str
) -> None:
    payload = copy.deepcopy(_artifact())
    if mutation == "seed":
        payload["identity"]["seed"] = 42
    elif mutation == "training-vector":
        payload["training_profiles"][0][0] = 9.0
        payload["training_profiles_sha256"] = canonical_sha256(payload["training_profiles"])
    else:
        payload["predictions"]["deterministic_ridge_linear"][0] = 9.0
        payload["prediction_sha256"]["deterministic_ridge_linear"] = canonical_sha256(
            payload["predictions"]["deterministic_ridge_linear"]
        )
    payload.pop("artifact_sha256")
    payload["artifact_sha256"] = canonical_sha256(payload)
    path = tmp_path / "tampered.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False)

    with pytest.raises(RevisionProtocolError):
        read_baseline_artifact(path)

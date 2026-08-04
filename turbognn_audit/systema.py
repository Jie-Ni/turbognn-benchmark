"""Pinned Systema centroid-accuracy metric and frozen TDS-36 hypothesis families."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .hashing import nfc_text, sha256_file, sha256_json, utf8_sort
from .inference import holm_adjust

SYSTEMA_COMMIT = "aaf5b5353993b48b78543f2f93b3e18ca65df515"
SYSTEMA_PATH = "evaluation/centroid_accuracy.py"
SYSTEMA_BLOB_SHA1 = "d38a1ba6fe5e03441725479a668a9d99140587f4"
SYSTEMA_RAW_SHA256 = "dd82add60a9bcaf12d2705c58659e3393bff222d30dc0db3902dc9b0edc9e987"
SYSTEMA_RAW_BYTES = 2_894
SYSTEMA_SIGN_FLIP_DRAWS = 100_000
SYSTEMA_SIGN_FLIP_SEED = 20_260_836


@dataclass(frozen=True)
class SystemaSourcePassport:
    """Verified identity of the only allowed centroid-accuracy implementation."""

    commit: str
    repository_path: str
    blob_sha1: str
    raw_sha256: str
    byte_count: int


def verify_systema_source(path: Path) -> SystemaSourcePassport:
    """Verify exact upstream LF bytes and their Git blob identity before execution."""
    raw = path.read_bytes()
    raw_sha256 = sha256_file(path)
    blob_sha1 = hashlib.sha1(
        f"blob {len(raw)}\0".encode("ascii") + raw,
        usedforsecurity=False,
    ).hexdigest()
    mismatches: list[str] = []
    if len(raw) != SYSTEMA_RAW_BYTES:
        mismatches.append(f"bytes={len(raw)} expected={SYSTEMA_RAW_BYTES}")
    if raw_sha256 != SYSTEMA_RAW_SHA256:
        mismatches.append(f"sha256={raw_sha256} expected={SYSTEMA_RAW_SHA256}")
    if blob_sha1 != SYSTEMA_BLOB_SHA1:
        mismatches.append(f"blob_sha1={blob_sha1} expected={SYSTEMA_BLOB_SHA1}")
    if mismatches:
        raise ValueError("Pinned Systema source mismatch: " + "; ".join(mismatches))
    return SystemaSourcePassport(
        commit=SYSTEMA_COMMIT,
        repository_path=SYSTEMA_PATH,
        blob_sha1=blob_sha1,
        raw_sha256=raw_sha256,
        byte_count=len(raw),
    )


def _canonical_centroid_inputs(
    predictions: pd.DataFrame,
    truth_centroids: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(predictions.index, pd.MultiIndex) or predictions.index.nlevels != 2:
        raise ValueError("predictions must use a (condition, method) MultiIndex")
    canonical_predictions = predictions.copy()
    canonical_truth = truth_centroids.copy()
    prediction_columns = [nfc_text(str(column)) for column in canonical_predictions.columns]
    truth_columns = [nfc_text(str(column)) for column in canonical_truth.columns]
    if len(set(prediction_columns)) != len(prediction_columns) or len(set(truth_columns)) != len(
        truth_columns
    ):
        raise ValueError("Gene columns collide after canonical normalization")
    canonical_predictions.columns = prediction_columns
    canonical_truth.columns = truth_columns
    if prediction_columns != truth_columns:
        raise ValueError("Predictions and truth centroids must share exact ordered genes")
    canonical_truth.index = pd.Index(
        [nfc_text(str(value)) for value in canonical_truth.index],
        name=canonical_truth.index.name,
    )
    canonical_predictions.index = pd.MultiIndex.from_tuples(
        [
            (nfc_text(str(condition)), nfc_text(str(method)))
            for condition, method in canonical_predictions.index
        ],
        names=canonical_predictions.index.names,
    )
    if canonical_predictions.index.has_duplicates or canonical_truth.index.has_duplicates:
        raise ValueError("Prediction and truth-centroid identities must be canonically unique")
    if len(canonical_truth) < 2:
        raise ValueError("Centroid accuracy requires at least two truth conditions")
    try:
        prediction_values = canonical_predictions.to_numpy(dtype=float)
        truth_values = canonical_truth.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Centroid inputs must be numeric") from error
    if not np.isfinite(prediction_values).all() or not np.isfinite(truth_values).all():
        raise ValueError("Centroid inputs must be finite")
    truth_support = set(canonical_truth.index.astype(str))
    methods = canonical_predictions.index.get_level_values(1).unique()
    if not len(methods):
        raise ValueError("At least one prediction method is required")
    for method in methods:
        mask = canonical_predictions.index.get_level_values(1) == method
        method_support = set(canonical_predictions.index.get_level_values(0)[mask].astype(str))
        if method_support != truth_support or int(mask.sum()) != len(truth_support):
            raise ValueError(
                f"Method {method!r} must predict every frozen truth condition exactly once"
            )
    canonical_predictions.iloc[:, :] = prediction_values
    canonical_truth.iloc[:, :] = truth_values
    return canonical_predictions, canonical_truth


def centroid_accuracies_exact(
    predictions: pd.DataFrame,
    truth_centroids: pd.DataFrame,
) -> pd.DataFrame:
    """Reference the frozen strict-distance definition for deterministic toy parity."""
    predictions, truth_centroids = _canonical_centroid_inputs(predictions, truth_centroids)
    conditions = predictions.index.get_level_values(0)
    missing = sorted(set(conditions) - set(truth_centroids.index))
    if missing:
        raise ValueError(f"Predicted conditions lack truth centroids: {missing}")
    distances = cdist(predictions.to_numpy(), truth_centroids.to_numpy(), metric="euclidean")
    truth_positions = {condition: index for index, condition in enumerate(truth_centroids.index)}
    correct = np.asarray(
        [distances[row, truth_positions[condition]] for row, condition in enumerate(conditions)]
    )
    denominator = truth_centroids.shape[0] - 1
    scores: dict[str, pd.Series] = {}
    methods = predictions.index.get_level_values(1).unique()
    for method in methods:
        mask = predictions.index.get_level_values(1) == method
        method_conditions = predictions.index.get_level_values(0)[mask]
        method_distances = distances[mask]
        method_correct = correct[mask]
        values = (method_distances > method_correct[:, None]).sum(axis=1) / denominator
        scores[str(method)] = pd.Series(values, index=method_conditions).sort_index()
    return pd.DataFrame(scores)


def calculate_pinned_centroid_accuracies(
    predictions: pd.DataFrame,
    truth_centroids: pd.DataFrame,
    source_path: Path,
) -> tuple[pd.DataFrame, SystemaSourcePassport]:
    """Execute only the hash-pinned upstream function and enforce toy-equivalent semantics."""
    passport = verify_systema_source(source_path)
    predictions, truth_centroids = _canonical_centroid_inputs(predictions, truth_centroids)
    namespace: dict[str, Any] = {}
    source = source_path.read_bytes().decode("utf-8")
    exec(compile(source, SYSTEMA_PATH, "exec"), namespace)
    function = namespace.get("calculate_centroid_accuracies")
    if not callable(function):
        raise RuntimeError("Pinned Systema source lacks calculate_centroid_accuracies")
    upstream = function(predictions, truth_centroids)
    if not isinstance(upstream, pd.DataFrame):
        raise RuntimeError("Pinned Systema function returned a non-DataFrame value")
    reference = centroid_accuracies_exact(predictions, truth_centroids)
    pd.testing.assert_frame_equal(upstream, reference, check_exact=True)
    return upstream, passport


def _contrast_arrays(
    frame: pd.DataFrame,
    datasets: Sequence[str],
) -> dict[str, np.ndarray]:
    required = {"dataset", "condition", "centroid_accuracy_difference"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Centroid contrast lacks columns: {sorted(missing)}")
    if frame.duplicated(["dataset", "condition"]).any():
        raise ValueError("Centroid contrast has duplicate dataset-condition rows")
    canonical = frame.copy()
    canonical["dataset"] = canonical["dataset"].map(lambda value: nfc_text(str(value)))
    canonical["condition"] = canonical["condition"].map(lambda value: nfc_text(str(value)))
    normalized_datasets = tuple(nfc_text(dataset) for dataset in datasets)
    if len(set(normalized_datasets)) != len(normalized_datasets):
        raise ValueError("Frozen dataset strata are duplicated after NFC normalization")
    if canonical.duplicated(["dataset", "condition"]).any():
        raise ValueError("Centroid contrast has duplicate canonical dataset-condition rows")
    if set(canonical["dataset"].astype(str)) != set(normalized_datasets):
        raise ValueError("Centroid contrast does not contain the frozen dataset strata")
    arrays: dict[str, np.ndarray] = {}
    for dataset in normalized_datasets:
        rows = canonical.loc[canonical["dataset"] == dataset].set_index("condition")
        order = utf8_sort(rows.index.astype(str).tolist())
        values = rows.loc[order, "centroid_accuracy_difference"].to_numpy(dtype=float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"Dataset {dataset!r} has empty or non-finite centroid contrasts")
        arrays[dataset] = values
    return arrays


def centroid_sign_flip_families(
    contrasts: Mapping[str, pd.DataFrame],
    *,
    datasets: Sequence[str],
    retained_external_models: Sequence[str] = (),
    draws: int = SYSTEMA_SIGN_FLIP_DRAWS,
    seed: int = SYSTEMA_SIGN_FLIP_SEED,
) -> pd.DataFrame:
    """Consume one PCG64 stream for only union, STRING, and GO graph-minus-self tests."""
    if draws < 1:
        raise ValueError("draws must be positive")
    if retained_external_models:
        raise ValueError("External models are not members of the TDS-36 sign-flip family")
    order = ("string_go_union", "string_ppi", "gene_ontology")
    if set(contrasts) != set(order):
        raise ValueError(
            f"Centroid contrast family mismatch: expected {list(order)}, got {sorted(contrasts)}"
        )
    arrays = {label: _contrast_arrays(contrasts[label], datasets) for label in order}
    support = {
        label: tuple(
            sorted(
                [
                    (nfc_text(str(row.dataset)), nfc_text(str(row.condition)))
                    for row in contrasts[label].itertuples()
                ],
                key=lambda value: (value[0].encode("utf-8"), value[1].encode("utf-8")),
            )
        )
        for label in order
    }
    generator = np.random.Generator(np.random.PCG64(seed))
    rows: list[dict[str, Any]] = []
    raw_p: dict[str, float] = {}
    for label in order:
        observed = float(np.mean([values.mean() for values in arrays[label].values()]))
        extreme = 0
        for _ in range(draws):
            statistic = float(
                np.mean(
                    [
                        np.mean(values * generator.choice((-1.0, 1.0), size=len(values)))
                        for values in arrays[label].values()
                    ]
                )
            )
            extreme += int(abs(statistic) >= abs(observed))
        raw_p[label] = (extreme + 1) / (draws + 1)
        rows.append(
            {
                "contrast": label,
                "mean_centroid_accuracy_difference": observed,
                "raw_p": raw_p[label],
                "extreme_draw_n": extreme,
                "draw_n": draws,
                "rng": "PCG64",
                "seed": seed,
                "pair_support_n": len(support[label]),
                "pair_support_hash": sha256_json(support[label]),
            }
        )
    g2_adjusted = holm_adjust({label: raw_p[label] for label in ("string_ppi", "gene_ontology")})
    for row in rows:
        label = str(row["contrast"])
        if label == "string_go_union":
            row["family"] = "G1_unadjusted_primary_sensitivity"
            row["adjusted_p"] = row["raw_p"]
        elif label in g2_adjusted:
            row["family"] = "G2_STRING_GO_Holm"
            row["adjusted_p"] = g2_adjusted[label]
        else:
            raise RuntimeError(f"Unexpected TDS-36 contrast {label!r}")
    return pd.DataFrame(rows)


def run_toy_parity(source_path: Path) -> dict[str, Any]:
    """Run the required deterministic upstream parity fixture."""
    predictions = pd.DataFrame(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [0.0, 1.0],
        ],
        index=pd.MultiIndex.from_tuples(
            [
                ("A", "perfect"),
                ("A", "tie"),
                ("B", "perfect"),
                ("B", "tie"),
                ("C", "perfect"),
                ("C", "tie"),
            ],
            names=["condition", "method"],
        ),
        columns=[1, 2],
    )
    truth = pd.DataFrame(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
        index=["A", "B", "C"],
        columns=[1, 2],
    )
    scores, passport = calculate_pinned_centroid_accuracies(predictions, truth, source_path)
    expected = pd.DataFrame(
        {"perfect": [1.0, 1.0, 1.0], "tie": [0.5, 0.5, 0.5]},
        index=pd.Index(["A", "B", "C"]),
    )
    pd.testing.assert_frame_equal(scores, expected, check_exact=True)
    return {"passport": asdict(passport), "scores": scores.to_dict()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--toy-parity", action="store_true", required=True)
    args = parser.parse_args()
    run_toy_parity(args.source)
    print("Pinned Systema source and deterministic toy parity verified.")


if __name__ == "__main__":
    main()

"""Fail-closed implementations of frozen TDS-14/15/16/18/22 rules."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd

from .hashing import nfc_text, utf8_sort
from .inference import holm_adjust

CANONICAL_DATASETS = ("norman", "adamson", "replogle_k562", "replogle_rpe1")
CANONICAL_SCALES = (200, 500, 1000)
REQUIRED_SEEDS = (42, 43, 44)
RANKING_GRAPHS = (
    "self_loop_gat",
    "string_ppi",
    "gene_ontology",
    "string_go_union",
)
G2_CONTRAST_ORDER = ("string_ppi", "gene_ontology")


class FrozenRuleError(ValueError):
    """Raised when an input cannot satisfy a frozen rule without a silent fallback."""


@dataclass(frozen=True)
class RankingGate:
    """TDS-14 graph-pair and seed-stability threshold decision."""

    benchmark_median: float
    minimum_cell_median: float
    seed_overall_median: float
    minimum_seed_cell_graph_median: float
    graph_pair_passed: bool
    seed_stability_passed: bool
    passed: bool


@dataclass(frozen=True)
class RankingIdentityResult:
    """Detailed TDS-14 ranking overlaps and descriptive permutation reference."""

    graph_pair_overlaps: pd.DataFrame
    cell_graph_pair_summary: pd.DataFrame
    seed_pair_overlaps: pd.DataFrame
    cell_graph_seed_summary: pd.DataFrame
    top_conditions: pd.DataFrame
    permutation_benchmark_medians: tuple[float, ...]
    permutation_upper_tail_p: float
    permutation_draw_n: int
    permutation_seed: int
    graph_pair_order: tuple[tuple[str, str], ...]
    ignored_graphs: tuple[str, ...]
    gate: RankingGate


@dataclass(frozen=True)
class OfficialSanityResult:
    """One TDS-15 reference-task comparison."""

    eligible: bool
    passed: bool
    metric_kind: str
    reproduced: float
    reference: float
    absolute_error: float
    tolerance: float | None
    tolerance_source: str
    reason: str


@dataclass(frozen=True)
class OutputValidityAudit:
    """TDS-16 fold ledger and dataset-scale cell decisions."""

    fold_ledger: pd.DataFrame
    cell_summary: pd.DataFrame
    passed: bool
    minimum_valid_fraction: float
    prediction_std_threshold: float


@dataclass(frozen=True)
class ResolutionEstimate:
    """TDS-32 matched-seed resolution for one frozen pairwise contrast."""

    point: float
    upper_95: float
    bootstrap_distribution: tuple[float, ...]
    required_seeds: tuple[int, ...]
    bootstrap_draw_n: int


@dataclass(frozen=True)
class PairEvidence:
    """Raw-scale interval and pair-specific resolution used by a claim gate."""

    delta_r_star: float
    ci_lower: float
    ci_upper: float
    resolution: ResolutionEstimate


@dataclass(frozen=True)
class DirectionalGate:
    """Shared TDS-18/TDS-22 interval, multiplicity, and magnitude gate."""

    direction: str
    ci_excludes_zero_in_direction: bool
    multiplicity_passed: bool
    magnitude_passed: bool
    operative_margin: float
    passed: bool


@dataclass(frozen=True)
class DirectionalFamilyResult:
    """One sequential-PCG64 sign-flip family with Holm-adjusted decisions."""

    contrasts: pd.DataFrame
    contrast_order: tuple[str, ...]
    draw_n: int
    rng: str
    seed: int


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise FrozenRuleError(f"{label} must be a string")
    normalized = nfc_text(value)
    if not normalized:
        raise FrozenRuleError(f"{label} must not be empty")
    return normalized


def _unique_identifiers(values: Sequence[str], label: str) -> tuple[str, ...]:
    normalized = tuple(_identifier(value, label) for value in values)
    if not normalized:
        raise FrozenRuleError(f"{label} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise FrozenRuleError(f"{label} contains duplicates after NFC normalization")
    return normalized


def _ordered_datasets(datasets: Sequence[str]) -> tuple[str, ...]:
    normalized = _unique_identifiers(datasets, "dataset")
    if set(normalized) == set(CANONICAL_DATASETS):
        return CANONICAL_DATASETS
    return tuple(utf8_sort(list(normalized)))


def _integer(value: object, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise FrozenRuleError(f"{label} must be an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise FrozenRuleError(f"{label} must be an integer") from exc
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise FrozenRuleError(f"{label} must be an integer")
    return int(numeric)


def _ordered_scales(scales: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(_integer(scale, "scale") for scale in scales)
    if not normalized or len(set(normalized)) != len(normalized):
        raise FrozenRuleError("scales must be nonempty and unique")
    return tuple(sorted(normalized))


def _ordered_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(_integer(seed, "seed") for seed in seeds)
    if not normalized or len(set(normalized)) != len(normalized):
        raise FrozenRuleError("seeds must be nonempty and unique")
    return tuple(sorted(normalized))


def _linear_median(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise FrozenRuleError("A median input is empty or nonfinite")
    return float(np.quantile(array, 0.5, method="linear"))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        raise FrozenRuleError("Jaccard is undefined for two empty sets")
    return len(left & right) / len(union)


def _top_conditions(scores: Mapping[str, float], top_k: int) -> tuple[str, ...]:
    if len(scores) < top_k:
        raise FrozenRuleError("The condition universe is smaller than top_k")
    if any(not np.isfinite(value) for value in scores.values()):
        raise FrozenRuleError("Ranking scores must be finite")
    ranked = sorted(scores, key=lambda item: (-scores[item], item.encode("utf-8")))
    return tuple(ranked[:top_k])


def _normalize_cell_universes(
    universes: Mapping[tuple[str, int], Sequence[str]],
    *,
    datasets: tuple[str, ...],
    scales: tuple[int, ...],
    minimum_conditions: int,
    top_k: int,
) -> dict[tuple[str, int], tuple[str, ...]]:
    if minimum_conditions < top_k or top_k < 1:
        raise FrozenRuleError("minimum_conditions must be at least top_k, and top_k positive")
    normalized: dict[tuple[str, int], tuple[str, ...]] = {}
    for raw_key, raw_conditions in universes.items():
        if not isinstance(raw_key, tuple) or len(raw_key) != 2:
            raise FrozenRuleError("Ranking universe keys must be (dataset, scale) tuples")
        key = (_identifier(raw_key[0], "dataset"), _integer(raw_key[1], "scale"))
        if key in normalized:
            raise FrozenRuleError("Ranking universe keys collide after normalization")
        conditions = _unique_identifiers(raw_conditions, "condition")
        if len(conditions) < minimum_conditions:
            raise FrozenRuleError(
                f"Ranking universe {key!r} has {len(conditions)} conditions; "
                f"requires {minimum_conditions}"
            )
        normalized[key] = tuple(utf8_sort(list(conditions)))
    expected = {(dataset, scale) for dataset in datasets for scale in scales}
    if set(normalized) != expected:
        raise FrozenRuleError("Ranking universes do not exactly cover the frozen cells")
    return normalized


def ranking_identity_gate(
    benchmark_median: float,
    cell_medians: Sequence[float],
    seed_overall_median: float,
    seed_cell_graph_medians: Sequence[float],
    *,
    benchmark_floor: float = 0.80,
    cell_floor: float = 0.70,
    seed_overall_floor: float = 0.80,
    seed_cell_graph_floor: float = 0.70,
) -> RankingGate:
    """Apply the inclusive TDS-14 graph-pair and seed-stability floors."""
    scalars = (benchmark_median, seed_overall_median)
    vectors = (tuple(cell_medians), tuple(seed_cell_graph_medians))
    thresholds = (benchmark_floor, cell_floor, seed_overall_floor, seed_cell_graph_floor)
    if any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise FrozenRuleError("Ranking thresholds must lie in [0, 1]")
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in scalars):
        raise FrozenRuleError("Ranking medians must be finite and lie in [0, 1]")
    if any(not values for values in vectors):
        raise FrozenRuleError("Cell-level ranking medians must not be empty")
    if any(
        not np.isfinite(value) or not 0.0 <= value <= 1.0 for values in vectors for value in values
    ):
        raise FrozenRuleError("Cell-level ranking medians must lie in [0, 1]")
    minimum_cell = min(vectors[0])
    minimum_seed_cell = min(vectors[1])
    graph_pair_passed = benchmark_median >= benchmark_floor and minimum_cell >= cell_floor
    seed_passed = (
        seed_overall_median >= seed_overall_floor and minimum_seed_cell >= seed_cell_graph_floor
    )
    return RankingGate(
        benchmark_median=float(benchmark_median),
        minimum_cell_median=float(minimum_cell),
        seed_overall_median=float(seed_overall_median),
        minimum_seed_cell_graph_median=float(minimum_seed_cell),
        graph_pair_passed=graph_pair_passed,
        seed_stability_passed=seed_passed,
        passed=graph_pair_passed and seed_passed,
    )


def analyze_ranking_identity(
    frame: pd.DataFrame,
    *,
    condition_universes: Mapping[tuple[str, int], Sequence[str]],
    datasets: Sequence[str] = CANONICAL_DATASETS,
    scales: Sequence[int] = CANONICAL_SCALES,
    required_seeds: Sequence[int] = REQUIRED_SEEDS,
    minimum_conditions: int = 45,
    top_k: int = 10,
    draws: int = 10_000,
    seed: int = 20_260_814,
) -> RankingIdentityResult:
    """Evaluate the complete TDS-14 ranking, stability, and permutation specification."""
    required_columns = {"dataset", "scale", "condition", "graph_type", "seed", "pearson_r"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise FrozenRuleError(f"Ranking frame lacks columns: {sorted(missing)}")
    if draws < 1:
        raise FrozenRuleError("draws must be positive")
    dataset_order = _ordered_datasets(datasets)
    scale_order = _ordered_scales(scales)
    seed_order = _ordered_seeds(required_seeds)
    if seed_order != REQUIRED_SEEDS:
        raise FrozenRuleError("TDS-14 requires the exact seed set {42, 43, 44}")
    graph_order = tuple(utf8_sort(list(RANKING_GRAPHS)))
    graph_pairs = tuple(combinations(graph_order, 2))
    universes = _normalize_cell_universes(
        condition_universes,
        datasets=dataset_order,
        scales=scale_order,
        minimum_conditions=minimum_conditions,
        top_k=top_k,
    )

    canonical = frame.loc[:, list(required_columns)].copy()
    canonical["dataset"] = canonical["dataset"].map(lambda value: _identifier(value, "dataset"))
    canonical["condition"] = canonical["condition"].map(
        lambda value: _identifier(value, "condition")
    )
    canonical["graph_type"] = canonical["graph_type"].map(
        lambda value: _identifier(value, "graph_type")
    )
    canonical["scale"] = canonical["scale"].map(lambda value: _integer(value, "scale"))
    canonical["seed"] = canonical["seed"].map(lambda value: _integer(value, "seed"))
    canonical["pearson_r"] = pd.to_numeric(canonical["pearson_r"], errors="coerce")
    if canonical.duplicated(["dataset", "scale", "condition", "graph_type", "seed"]).any():
        raise FrozenRuleError("Ranking frame has duplicate rows after NFC normalization")
    if not np.isfinite(canonical["pearson_r"].to_numpy(dtype=float)).all():
        raise FrozenRuleError("Ranking correlations must be finite")
    if not canonical["pearson_r"].between(-1.0, 1.0, inclusive="both").all():
        raise FrozenRuleError("Ranking correlations must lie in [-1, 1]")
    if set(canonical["dataset"]) != set(dataset_order):
        raise FrozenRuleError("Ranking frame dataset family is incomplete or contains extras")
    if set(canonical["scale"]) != set(scale_order):
        raise FrozenRuleError("Ranking frame scale family is incomplete or contains extras")

    ignored_graphs = tuple(utf8_sort(list(set(canonical["graph_type"]) - set(RANKING_GRAPHS))))
    claim_frame = canonical.loc[canonical["graph_type"].isin(RANKING_GRAPHS)].copy()
    if set(claim_frame["graph_type"]) != set(RANKING_GRAPHS):
        raise FrozenRuleError("The four-arm TDS-14 graph family is incomplete")

    values: dict[tuple[str, int, str, str, int], float] = {}
    for row in claim_frame.itertuples(index=False):
        values[(row.dataset, row.scale, row.graph_type, row.condition, row.seed)] = float(
            row.pearson_r
        )
    expected_keys = {
        (dataset, scale, graph, condition, training_seed)
        for dataset in dataset_order
        for scale in scale_order
        for graph in RANKING_GRAPHS
        for condition in universes[(dataset, scale)]
        for training_seed in seed_order
    }
    present_relevant = {key for key in values if key[3] in set(universes[(key[0], key[1])])}
    if present_relevant != expected_keys:
        missing_n = len(expected_keys - present_relevant)
        extra_n = len(present_relevant - expected_keys)
        raise FrozenRuleError(
            f"Ranking common-universe matrix is incomplete: missing={missing_n}, extra={extra_n}"
        )

    aggregate_scores: dict[tuple[str, int, str], dict[str, float]] = {}
    seed_scores: dict[tuple[str, int, str, int], dict[str, float]] = {}
    top_rows: list[dict[str, object]] = []
    for dataset in dataset_order:
        for scale in scale_order:
            conditions = universes[(dataset, scale)]
            for graph in graph_order:
                graph_scores: dict[str, float] = {}
                for condition in conditions:
                    correlations = np.asarray(
                        [values[(dataset, scale, graph, condition, item)] for item in seed_order],
                        dtype=float,
                    )
                    clipped = np.clip(correlations, -1.0 + 1e-7, 1.0 - 1e-7)
                    graph_scores[condition] = float(np.tanh(np.arctanh(clipped).mean()))
                aggregate_scores[(dataset, scale, graph)] = graph_scores
                aggregate_top = _top_conditions(graph_scores, top_k)
                for rank, condition in enumerate(aggregate_top, start=1):
                    top_rows.append(
                        {
                            "dataset": dataset,
                            "scale": scale,
                            "graph_type": graph,
                            "ranking": "seed_aggregated",
                            "seed": pd.NA,
                            "rank": rank,
                            "condition": condition,
                            "score": graph_scores[condition],
                        }
                    )
                for training_seed in seed_order:
                    per_seed = {
                        condition: values[(dataset, scale, graph, condition, training_seed)]
                        for condition in conditions
                    }
                    seed_scores[(dataset, scale, graph, training_seed)] = per_seed
                    for rank, condition in enumerate(_top_conditions(per_seed, top_k), start=1):
                        top_rows.append(
                            {
                                "dataset": dataset,
                                "scale": scale,
                                "graph_type": graph,
                                "ranking": "single_seed",
                                "seed": training_seed,
                                "rank": rank,
                                "condition": condition,
                                "score": per_seed[condition],
                            }
                        )

    pair_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    for dataset in dataset_order:
        for scale in scale_order:
            cell_values: list[float] = []
            for graph_a, graph_b in graph_pairs:
                top_a = set(_top_conditions(aggregate_scores[(dataset, scale, graph_a)], top_k))
                top_b = set(_top_conditions(aggregate_scores[(dataset, scale, graph_b)], top_k))
                overlap = _jaccard(top_a, top_b)
                cell_values.append(overlap)
                pair_rows.append(
                    {
                        "dataset": dataset,
                        "scale": scale,
                        "graph_a": graph_a,
                        "graph_b": graph_b,
                        "jaccard": overlap,
                    }
                )
            cell_rows.append(
                {
                    "dataset": dataset,
                    "scale": scale,
                    "pair_n": len(cell_values),
                    "median_jaccard": _linear_median(cell_values),
                }
            )

    seed_pair_rows: list[dict[str, object]] = []
    seed_cell_rows: list[dict[str, object]] = []
    seed_pairs = tuple(combinations(seed_order, 2))
    for dataset in dataset_order:
        for scale in scale_order:
            for graph in graph_order:
                overlaps: list[float] = []
                for seed_a, seed_b in seed_pairs:
                    top_a = set(
                        _top_conditions(seed_scores[(dataset, scale, graph, seed_a)], top_k)
                    )
                    top_b = set(
                        _top_conditions(seed_scores[(dataset, scale, graph, seed_b)], top_k)
                    )
                    overlap = _jaccard(top_a, top_b)
                    overlaps.append(overlap)
                    seed_pair_rows.append(
                        {
                            "dataset": dataset,
                            "scale": scale,
                            "graph_type": graph,
                            "seed_a": seed_a,
                            "seed_b": seed_b,
                            "jaccard": overlap,
                        }
                    )
                seed_cell_rows.append(
                    {
                        "dataset": dataset,
                        "scale": scale,
                        "graph_type": graph,
                        "pair_n": len(overlaps),
                        "median_jaccard": _linear_median(overlaps),
                    }
                )

    pair_frame = pd.DataFrame(pair_rows)
    cell_frame = pd.DataFrame(cell_rows)
    seed_pair_frame = pd.DataFrame(seed_pair_rows)
    seed_cell_frame = pd.DataFrame(seed_cell_rows)
    benchmark_median = _linear_median(pair_frame["jaccard"].tolist())
    seed_overall_median = _linear_median(seed_pair_frame["jaccard"].tolist())
    gate = ranking_identity_gate(
        benchmark_median,
        cell_frame["median_jaccard"].tolist(),
        seed_overall_median,
        seed_cell_frame["median_jaccard"].tolist(),
    )

    generator = np.random.Generator(np.random.PCG64(seed))
    permutation_statistics: list[float] = []
    for _ in range(draws):
        null_overlaps: list[float] = []
        for dataset in dataset_order:
            for scale in scale_order:
                conditions = universes[(dataset, scale)]
                for graph_a, graph_b in graph_pairs:
                    top_a = set(_top_conditions(aggregate_scores[(dataset, scale, graph_a)], top_k))
                    second = aggregate_scores[(dataset, scale, graph_b)]
                    permuted_values = generator.permutation(
                        np.asarray([second[condition] for condition in conditions], dtype=float)
                    )
                    permuted = dict(zip(conditions, permuted_values, strict=True))
                    null_overlaps.append(_jaccard(top_a, set(_top_conditions(permuted, top_k))))
        permutation_statistics.append(_linear_median(null_overlaps))
    extreme = sum(value >= benchmark_median for value in permutation_statistics)

    return RankingIdentityResult(
        graph_pair_overlaps=pair_frame,
        cell_graph_pair_summary=cell_frame,
        seed_pair_overlaps=seed_pair_frame,
        cell_graph_seed_summary=seed_cell_frame,
        top_conditions=pd.DataFrame(top_rows),
        permutation_benchmark_medians=tuple(permutation_statistics),
        permutation_upper_tail_p=(extreme + 1) / (draws + 1),
        permutation_draw_n=draws,
        permutation_seed=seed,
        graph_pair_order=graph_pairs,
        ignored_graphs=ignored_graphs,
        gate=gate,
    )


def evaluate_official_sanity(
    reproduced: float,
    reference: float,
    *,
    metric_kind: Literal["correlation_like", "loss", "other"],
    official_tolerance: float | None = None,
) -> OfficialSanityResult:
    """Apply the exact TDS-15 absolute-error rule without an implicit fallback."""
    reproduced_value = float(reproduced)
    reference_value = float(reference)
    if not np.isfinite(reproduced_value) or not np.isfinite(reference_value):
        raise FrozenRuleError("Official sanity values must be finite")
    if metric_kind not in {"correlation_like", "loss", "other"}:
        raise FrozenRuleError("metric_kind must be correlation_like, loss, or other")
    if official_tolerance is not None:
        tolerance = float(official_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise FrozenRuleError("official_tolerance must be finite and nonnegative")
        source = "official"
        eligible = True
        reason = "official tolerance supplied"
    elif metric_kind == "correlation_like":
        tolerance = max(0.1 * abs(reference_value), 0.02)
        source = "tds15_correlation_default"
        eligible = True
        reason = "TDS-15 correlation-like fallback tolerance"
    elif metric_kind == "loss" and reference_value > 0.0:
        tolerance = 0.1 * reference_value
        source = "tds15_positive_loss_default"
        eligible = True
        reason = "TDS-15 strictly positive loss fallback tolerance"
    else:
        tolerance = None
        source = "none"
        eligible = False
        reason = "an official tolerance is required for this metric/reference"
    absolute_error = abs(reproduced_value - reference_value)
    within_tolerance = tolerance is not None and (
        absolute_error <= tolerance
        or math.isclose(absolute_error, tolerance, rel_tol=1e-12, abs_tol=1e-15)
    )
    passed = bool(eligible and within_tolerance)
    return OfficialSanityResult(
        eligible=eligible,
        passed=passed,
        metric_kind=metric_kind,
        reproduced=reproduced_value,
        reference=reference_value,
        absolute_error=absolute_error,
        tolerance=tolerance,
        tolerance_source=source,
        reason=reason,
    )


def _normalize_expected_folds(
    expected_folds: Mapping[tuple[str, str, int], Sequence[str]],
) -> dict[tuple[str, str, int], tuple[str, ...]]:
    normalized: dict[tuple[str, str, int], tuple[str, ...]] = {}
    for raw_key, raw_folds in expected_folds.items():
        if not isinstance(raw_key, tuple) or len(raw_key) != 3:
            raise FrozenRuleError("Expected-fold keys must be (model, dataset, scale) tuples")
        key = (
            _identifier(raw_key[0], "model"),
            _identifier(raw_key[1], "dataset"),
            _integer(raw_key[2], "scale"),
        )
        if key in normalized:
            raise FrozenRuleError("Expected-fold cells collide after normalization")
        fold_ids = _unique_identifiers(raw_folds, "fold_id")
        normalized[key] = tuple(utf8_sort(list(fold_ids)))
    if not normalized:
        raise FrozenRuleError("At least one expected output-validity cell is required")
    return normalized


def _vector(value: object, label: str) -> tuple[np.ndarray | None, str | None]:
    if value is None:
        return None, f"missing_{label}"
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None, f"invalid_{label}"
    if array.ndim != 1 or not len(array):
        return None, f"invalid_{label}_shape"
    if not np.isfinite(array).all():
        return None, f"nonfinite_{label}"
    return array, None


def audit_output_validity(
    frame: pd.DataFrame,
    *,
    expected_folds: Mapping[tuple[str, str, int], Sequence[str]],
    minimum_valid_fraction: float = 0.95,
    prediction_std_threshold: float = 1e-8,
    terminal_statuses: Sequence[str] = ("succeeded", "failed", "skipped"),
) -> OutputValidityAudit:
    """Audit the pre-execution TDS-16 denominator and every terminal fold outcome."""
    required = {
        "model",
        "dataset",
        "scale",
        "fold_id",
        "mapping_passed",
        "status",
        "y_true",
        "y_pred",
        "training_side_mean",
    }
    missing = required - set(frame.columns)
    if missing:
        raise FrozenRuleError(f"Output-validity frame lacks columns: {sorted(missing)}")
    if not 0.0 <= minimum_valid_fraction <= 1.0:
        raise FrozenRuleError("minimum_valid_fraction must lie in [0, 1]")
    if not np.isfinite(prediction_std_threshold) or prediction_std_threshold < 0.0:
        raise FrozenRuleError("prediction_std_threshold must be finite and nonnegative")
    expected = _normalize_expected_folds(expected_folds)
    terminal = set(_unique_identifiers(terminal_statuses, "terminal_status"))
    if "succeeded" not in terminal:
        raise FrozenRuleError("terminal_statuses must include 'succeeded'")

    canonical = frame.loc[:, list(required)].copy()
    for column in ("model", "dataset", "fold_id"):
        canonical[column] = canonical[column].map(
            lambda value, name=column: _identifier(value, name)
        )
    canonical["scale"] = canonical["scale"].map(lambda value: _integer(value, "scale"))
    if canonical.duplicated(["model", "dataset", "scale", "fold_id"]).any():
        raise FrozenRuleError("Output-validity rows are duplicated after NFC normalization")
    observed_keys = set(
        canonical[["model", "dataset", "scale", "fold_id"]].itertuples(index=False, name=None)
    )
    expected_keys = {
        (*cell, fold_id) for cell, fold_ids in expected.items() for fold_id in fold_ids
    }
    if observed_keys != expected_keys:
        missing_n = len(expected_keys - observed_keys)
        extra_n = len(observed_keys - expected_keys)
        raise FrozenRuleError(
            "Output-validity rows do not exactly match the prespecified manifest folds: "
            f"missing={missing_n}, extra={extra_n}"
        )

    row_lookup = {
        (row.model, row.dataset, row.scale, row.fold_id): row
        for row in canonical.itertuples(index=False)
    }
    ordered_cells = sorted(
        expected,
        key=lambda key: (key[0].encode("utf-8"), key[1].encode("utf-8"), key[2]),
    )
    ledger_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    for model, dataset, scale in ordered_cells:
        denominator_n = 0
        valid_n = 0
        terminal_n = 0
        for fold_id in expected[(model, dataset, scale)]:
            row = row_lookup[(model, dataset, scale, fold_id)]
            if not isinstance(row.mapping_passed, (bool, np.bool_)):
                raise FrozenRuleError("mapping_passed values must be booleans")
            mapping_passed = bool(row.mapping_passed)
            status = (
                _identifier(row.status, "status")
                if isinstance(row.status, str) and row.status
                else None
            )
            prediction_std = np.nan
            valid = False
            if not mapping_passed:
                reason = "mapping_gate_excluded"
            else:
                denominator_n += 1
                if status not in terminal:
                    reason = "nonterminal_status"
                else:
                    terminal_n += 1
                    if status != "succeeded":
                        reason = f"terminal_{status}"
                    else:
                        prediction, reason = _vector(row.y_pred, "prediction")
                        truth: np.ndarray | None = None
                        mean_vector: np.ndarray | None = None
                        if reason is None:
                            truth, reason = _vector(row.y_true, "truth")
                        if reason is None:
                            mean_vector, reason = _vector(
                                row.training_side_mean, "training_side_mean"
                            )
                        if reason is None:
                            assert prediction is not None
                            assert truth is not None
                            assert mean_vector is not None
                            if len(truth) != len(prediction) or len(mean_vector) != len(prediction):
                                reason = "vector_length_mismatch"
                            else:
                                prediction_std = float(np.std(prediction, ddof=0))
                                if not prediction_std > prediction_std_threshold:
                                    reason = "degenerate_prediction"
                                else:
                                    reason = "valid"
                                    valid = True
                                    valid_n += 1
            ledger_rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "scale": scale,
                    "fold_id": fold_id,
                    "mapping_passed": mapping_passed,
                    "in_denominator": mapping_passed,
                    "status": status,
                    "prediction_std": prediction_std,
                    "valid": valid,
                    "reason": reason,
                }
            )
        valid_fraction = valid_n / denominator_n if denominator_n else 0.0
        passed = (
            denominator_n > 0
            and terminal_n == denominator_n
            and valid_fraction >= minimum_valid_fraction
        )
        cell_rows.append(
            {
                "model": model,
                "dataset": dataset,
                "scale": scale,
                "planned_fold_n": len(expected[(model, dataset, scale)]),
                "denominator_n": denominator_n,
                "terminal_n": terminal_n,
                "valid_n": valid_n,
                "valid_fraction": valid_fraction,
                "passed": passed,
            }
        )
    summary = pd.DataFrame(cell_rows)
    return OutputValidityAudit(
        fold_ledger=pd.DataFrame(ledger_rows),
        cell_summary=summary,
        passed=bool(summary["passed"].all()),
        minimum_valid_fraction=minimum_valid_fraction,
        prediction_std_threshold=prediction_std_threshold,
    )


def pair_specific_resolution(
    seed_effects: Mapping[int, float],
    bootstrap_seed_effects: Mapping[int, Sequence[float]],
    *,
    required_seeds: Sequence[int] = REQUIRED_SEEDS,
    expected_bootstrap_draws: int | None = 10_000,
) -> ResolutionEstimate:
    """Apply the TDS-32 matched-seed pair algorithm to one pairwise contrast."""
    seed_order = _ordered_seeds(required_seeds)
    if seed_order != REQUIRED_SEEDS:
        raise FrozenRuleError("Pair-specific resolution requires seeds {42, 43, 44}")
    normalized_effects: dict[int, float] = {}
    for raw_seed, raw_value in seed_effects.items():
        training_seed = _integer(raw_seed, "seed")
        if training_seed in normalized_effects:
            raise FrozenRuleError("Seed effects collide after integer normalization")
        try:
            normalized_effects[training_seed] = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise FrozenRuleError("Seed effects must be numeric") from exc
    if set(normalized_effects) != set(seed_order):
        raise FrozenRuleError("Seed effects do not exactly match the required seed set")
    if not np.isfinite(list(normalized_effects.values())).all():
        raise FrozenRuleError("Seed effects must be finite")
    normalized_bootstrap: dict[int, np.ndarray] = {}
    for raw_seed, raw_values in bootstrap_seed_effects.items():
        training_seed = _integer(raw_seed, "seed")
        if training_seed in normalized_bootstrap:
            raise FrozenRuleError("Bootstrap seed effects collide after integer normalization")
        try:
            values = np.asarray(raw_values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise FrozenRuleError("Bootstrap seed effects must be numeric arrays") from exc
        if values.ndim != 1:
            raise FrozenRuleError("Bootstrap seed effects must be one-dimensional arrays")
        normalized_bootstrap[training_seed] = values
    if set(normalized_bootstrap) != set(seed_order):
        raise FrozenRuleError("Bootstrap seed effects do not exactly match the required seed set")
    lengths = {len(values) for values in normalized_bootstrap.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
        raise FrozenRuleError("Bootstrap seed-effect arrays must have one equal positive length")
    draw_n = next(iter(lengths))
    if expected_bootstrap_draws is not None and draw_n != expected_bootstrap_draws:
        raise FrozenRuleError(
            f"Resolution bootstrap has {draw_n} draws; expected {expected_bootstrap_draws}"
        )
    if any(
        values.ndim != 1 or not np.isfinite(values).all()
        for values in normalized_bootstrap.values()
    ):
        raise FrozenRuleError("Bootstrap seed effects must be finite one-dimensional arrays")
    seed_pairs = tuple(combinations(seed_order, 2))

    def resolution_statistic(values: Mapping[int, float]) -> float:
        pair_values = [
            abs(values[left] - values[right]) / np.sqrt(2.0) for left, right in seed_pairs
        ]
        return float(np.quantile(pair_values, 0.95, method="linear"))

    point = resolution_statistic(normalized_effects)
    bootstrap = tuple(
        resolution_statistic({seed: float(normalized_bootstrap[seed][draw]) for seed in seed_order})
        for draw in range(draw_n)
    )
    upper = float(np.quantile(np.asarray(bootstrap), 0.95, method="linear"))
    return ResolutionEstimate(
        point=point,
        upper_95=upper,
        bootstrap_distribution=bootstrap,
        required_seeds=seed_order,
        bootstrap_draw_n=draw_n,
    )


def directional_claim_gate(
    evidence: PairEvidence,
    adjusted_p: float,
    *,
    practical_margin: float = 0.02,
    alpha: float = 0.05,
) -> DirectionalGate:
    """Apply the shared strict TDS-18/TDS-22 directional claim gate."""
    values = (
        evidence.delta_r_star,
        evidence.ci_lower,
        evidence.ci_upper,
        evidence.resolution.point,
        evidence.resolution.upper_95,
        adjusted_p,
        practical_margin,
        alpha,
    )
    if not np.isfinite(values).all():
        raise FrozenRuleError("Directional-gate inputs must be finite")
    if evidence.ci_lower > evidence.ci_upper:
        raise FrozenRuleError("Confidence-interval bounds are reversed")
    if evidence.resolution.point < 0.0 or evidence.resolution.upper_95 < 0.0:
        raise FrozenRuleError("Resolution values must be nonnegative")
    if evidence.resolution.required_seeds != REQUIRED_SEEDS:
        raise FrozenRuleError("Directional evidence requires seeds {42, 43, 44}")
    distribution = evidence.resolution.bootstrap_distribution
    if (
        evidence.resolution.bootstrap_draw_n < 1
        or len(distribution) != evidence.resolution.bootstrap_draw_n
        or not np.isfinite(distribution).all()
        or any(item < 0.0 for item in distribution)
    ):
        raise FrozenRuleError("Directional evidence has an invalid resolution distribution")
    if not 0.0 <= adjusted_p <= 1.0 or not 0.0 < alpha < 1.0:
        raise FrozenRuleError("P value and alpha are outside their admissible ranges")
    if practical_margin < 0.0:
        raise FrozenRuleError("practical_margin must be nonnegative")
    direction = (
        "positive"
        if evidence.delta_r_star > 0.0
        else "negative" if evidence.delta_r_star < 0.0 else "zero"
    )
    ci_positive = evidence.ci_lower > 0.0
    ci_negative = evidence.ci_upper < 0.0
    ci_in_direction = (direction == "positive" and ci_positive) or (
        direction == "negative" and ci_negative
    )
    operative_margin = max(evidence.resolution.upper_95, practical_margin)
    multiplicity_passed = adjusted_p < alpha
    magnitude_passed = abs(evidence.delta_r_star) > operative_margin
    return DirectionalGate(
        direction=direction,
        ci_excludes_zero_in_direction=ci_in_direction,
        multiplicity_passed=multiplicity_passed,
        magnitude_passed=magnitude_passed,
        operative_margin=operative_margin,
        passed=ci_in_direction and multiplicity_passed and magnitude_passed,
    )


def _normalize_family_universes(
    universes: Mapping[tuple[str, str], Sequence[str]],
    *,
    contrasts: tuple[str, ...],
    datasets: tuple[str, ...],
    minimum_conditions: int,
) -> dict[tuple[str, str], tuple[str, ...]]:
    if minimum_conditions < 1:
        raise FrozenRuleError("minimum_conditions must be positive")
    normalized: dict[tuple[str, str], tuple[str, ...]] = {}
    for raw_key, raw_conditions in universes.items():
        if not isinstance(raw_key, tuple) or len(raw_key) != 2:
            raise FrozenRuleError("Family universe keys must be (contrast, dataset) tuples")
        key = (
            _identifier(raw_key[0], "contrast"),
            _identifier(raw_key[1], "dataset"),
        )
        if key in normalized:
            raise FrozenRuleError("Family universe keys collide after NFC normalization")
        conditions = _unique_identifiers(raw_conditions, "condition")
        if len(conditions) < minimum_conditions:
            raise FrozenRuleError(
                f"Family universe {key!r} has {len(conditions)} conditions; "
                f"requires {minimum_conditions}"
            )
        normalized[key] = tuple(utf8_sort(list(conditions)))
    expected = {(contrast, dataset) for contrast in contrasts for dataset in datasets}
    if set(normalized) != expected:
        raise FrozenRuleError("Condition universes do not exactly cover the frozen family")
    return normalized


def _normalize_pair_evidence(
    evidence: Mapping[str, PairEvidence],
    contrasts: tuple[str, ...],
    expected_resolution_bootstrap_draws: int,
) -> dict[str, PairEvidence]:
    normalized: dict[str, PairEvidence] = {}
    for raw_contrast, value in evidence.items():
        contrast = _identifier(raw_contrast, "contrast")
        if contrast in normalized:
            raise FrozenRuleError("Evidence identifiers collide after NFC normalization")
        if not isinstance(value, PairEvidence):
            raise FrozenRuleError("Every family member requires PairEvidence")
        resolution = value.resolution
        if resolution.required_seeds != REQUIRED_SEEDS:
            raise FrozenRuleError("Pair evidence does not use required seeds {42, 43, 44}")
        if resolution.bootstrap_draw_n != expected_resolution_bootstrap_draws:
            raise FrozenRuleError(
                "Pair evidence has the wrong resolution-bootstrap draw count: "
                f"observed={resolution.bootstrap_draw_n}, "
                f"expected={expected_resolution_bootstrap_draws}"
            )
        if len(resolution.bootstrap_distribution) != resolution.bootstrap_draw_n:
            raise FrozenRuleError("Pair evidence has an inconsistent resolution distribution")
        resolution_values = (
            resolution.point,
            resolution.upper_95,
            *resolution.bootstrap_distribution,
        )
        if not np.isfinite(resolution_values).all() or any(
            item < 0.0 for item in resolution_values
        ):
            raise FrozenRuleError("Pair-specific resolution values must be finite and nonnegative")
        normalized[contrast] = value
    if set(normalized) != set(contrasts):
        raise FrozenRuleError("Pair evidence does not exactly match the frozen family")
    return normalized


def _evaluate_directional_family(
    frame: pd.DataFrame,
    *,
    contrast_order: tuple[str, ...],
    condition_universes: Mapping[tuple[str, str], Sequence[str]],
    evidence: Mapping[str, PairEvidence],
    positive_favors: Mapping[str, str],
    negative_favors: Mapping[str, str],
    datasets: Sequence[str],
    minimum_conditions: int,
    draws: int,
    seed: int,
    expected_resolution_bootstrap_draws: int,
) -> DirectionalFamilyResult:
    required = {"contrast", "dataset", "condition", "scale", "seed", "delta_z"}
    missing = required - set(frame.columns)
    if missing:
        raise FrozenRuleError(f"Directional-family frame lacks columns: {sorted(missing)}")
    if draws < 1:
        raise FrozenRuleError("draws must be positive")
    contrasts = _unique_identifiers(contrast_order, "contrast")
    dataset_order = _ordered_datasets(datasets)
    universes = _normalize_family_universes(
        condition_universes,
        contrasts=contrasts,
        datasets=dataset_order,
        minimum_conditions=minimum_conditions,
    )
    if expected_resolution_bootstrap_draws < 1:
        raise FrozenRuleError("expected_resolution_bootstrap_draws must be positive")
    pair_evidence = _normalize_pair_evidence(
        evidence, contrasts, expected_resolution_bootstrap_draws
    )
    normalized_positive = {
        _identifier(key, "contrast"): _identifier(value, "favored_arm")
        for key, value in positive_favors.items()
    }
    normalized_negative = {
        _identifier(key, "contrast"): _identifier(value, "favored_arm")
        for key, value in negative_favors.items()
    }
    if set(normalized_positive) != set(contrasts) or set(normalized_negative) != set(contrasts):
        raise FrozenRuleError("Favored-arm labels do not exactly match the frozen family")

    canonical = frame.loc[:, list(required)].copy()
    for column in ("contrast", "dataset", "condition"):
        canonical[column] = canonical[column].map(
            lambda value, name=column: _identifier(value, name)
        )
    canonical["scale"] = canonical["scale"].map(lambda value: _integer(value, "scale"))
    canonical["seed"] = canonical["seed"].map(lambda value: _integer(value, "seed"))
    canonical["delta_z"] = pd.to_numeric(canonical["delta_z"], errors="coerce")
    if canonical.duplicated(["contrast", "dataset", "condition", "scale", "seed"]).any():
        raise FrozenRuleError("Directional-family rows are duplicated after NFC normalization")
    if not np.isfinite(canonical["delta_z"].to_numpy(dtype=float)).all():
        raise FrozenRuleError("Directional-family condition effects must be finite")
    if set(canonical["contrast"]) != set(contrasts):
        raise FrozenRuleError("Directional-family membership is incomplete or contains extras")
    if set(canonical["dataset"]) != set(dataset_order):
        raise FrozenRuleError("Directional-family dataset strata are incomplete or contain extras")
    if set(canonical["scale"]) != set(CANONICAL_SCALES):
        raise FrozenRuleError("Directional-family rows require scales {200, 500, 1000}")
    if set(canonical["seed"]) != set(REQUIRED_SEEDS):
        raise FrozenRuleError("Directional-family rows require seeds {42, 43, 44}")

    values = {
        (row.contrast, row.dataset, row.condition, row.scale, row.seed): float(row.delta_z)
        for row in canonical.itertuples(index=False)
    }
    expected_keys = {
        (contrast, dataset, condition, scale, training_seed)
        for contrast in contrasts
        for dataset in dataset_order
        for condition in universes[(contrast, dataset)]
        for scale in CANONICAL_SCALES
        for training_seed in REQUIRED_SEEDS
    }
    if set(values) != expected_keys:
        raise FrozenRuleError(
            "Directional-family effects do not exactly match frozen support: "
            f"missing={len(expected_keys - set(values))}, extra={len(set(values) - expected_keys)}"
        )

    arrays: dict[tuple[str, str], np.ndarray] = {}
    for contrast in contrasts:
        for dataset in dataset_order:
            condition_effects = []
            for condition in universes[(contrast, dataset)]:
                matched = [
                    values[(contrast, dataset, condition, scale, training_seed)]
                    for scale in CANONICAL_SCALES
                    for training_seed in REQUIRED_SEEDS
                ]
                condition_effects.append(float(np.mean(matched)))
            arrays[(contrast, dataset)] = np.asarray(condition_effects, dtype=float)
    generator = np.random.Generator(np.random.PCG64(seed))
    raw_p: dict[str, float] = {}
    statistics: dict[str, float] = {}
    extreme_counts: dict[str, int] = {}
    for contrast in contrasts:
        observed = float(np.mean([arrays[(contrast, dataset)].mean() for dataset in dataset_order]))
        extreme = 0
        for _ in range(draws):
            dataset_statistics = []
            for dataset in dataset_order:
                condition_values = arrays[(contrast, dataset)]
                signs = generator.choice((-1.0, 1.0), size=len(condition_values))
                dataset_statistics.append(float(np.mean(condition_values * signs)))
            null_statistic = float(np.mean(dataset_statistics))
            extreme += int(abs(null_statistic) >= abs(observed))
        statistics[contrast] = observed
        extreme_counts[contrast] = extreme
        raw_p[contrast] = (extreme + 1) / (draws + 1)
    adjusted = holm_adjust(raw_p)

    rows: list[dict[str, object]] = []
    for contrast in contrasts:
        item = pair_evidence[contrast]
        gate = directional_claim_gate(item, adjusted[contrast])
        favored = (
            normalized_positive[contrast]
            if gate.direction == "positive"
            else normalized_negative[contrast] if gate.direction == "negative" else "none"
        )
        rows.append(
            {
                "contrast": contrast,
                "mean_delta_z": statistics[contrast],
                "delta_r_star": item.delta_r_star,
                "ci_lower": item.ci_lower,
                "ci_upper": item.ci_upper,
                "resolution_point": item.resolution.point,
                "resolution_upper_95": item.resolution.upper_95,
                "operative_margin": gate.operative_margin,
                "raw_p": raw_p[contrast],
                "holm_p": adjusted[contrast],
                "extreme_draw_n": extreme_counts[contrast],
                "draw_n": draws,
                "rng": "PCG64",
                "seed": seed,
                "direction": gate.direction,
                "favored_arm": favored,
                "ci_excludes_zero_in_direction": gate.ci_excludes_zero_in_direction,
                "multiplicity_passed": gate.multiplicity_passed,
                "magnitude_passed": gate.magnitude_passed,
                "claim_passed": gate.passed,
            }
        )
    return DirectionalFamilyResult(
        contrasts=pd.DataFrame(rows),
        contrast_order=contrasts,
        draw_n=draws,
        rng="PCG64",
        seed=seed,
    )


def evaluate_external_family(
    frame: pd.DataFrame,
    *,
    retained_models: Sequence[str],
    condition_universes: Mapping[tuple[str, str], Sequence[str]],
    evidence: Mapping[str, PairEvidence],
    datasets: Sequence[str] = CANONICAL_DATASETS,
    minimum_conditions: int = 40,
    draws: int = 100_000,
    seed: int = 20_260_818,
    expected_resolution_bootstrap_draws: int = 10_000,
) -> DirectionalFamilyResult:
    """Evaluate the complete TDS-18 external-minus-union Holm family."""
    normalized = _unique_identifiers(retained_models, "retained_model")
    order = tuple(utf8_sort(list(normalized)))
    return _evaluate_directional_family(
        frame,
        contrast_order=order,
        condition_universes=condition_universes,
        evidence=evidence,
        positive_favors={model: model for model in order},
        negative_favors={model: "string_go_union" for model in order},
        datasets=datasets,
        minimum_conditions=minimum_conditions,
        draws=draws,
        seed=seed,
        expected_resolution_bootstrap_draws=expected_resolution_bootstrap_draws,
    )


def evaluate_g2_family(
    frame: pd.DataFrame,
    *,
    condition_universes: Mapping[tuple[str, str], Sequence[str]],
    evidence: Mapping[str, PairEvidence],
    datasets: Sequence[str] = CANONICAL_DATASETS,
    minimum_conditions: int = 45,
    draws: int = 100_000,
    seed: int = 20_260_822,
    expected_resolution_bootstrap_draws: int = 10_000,
) -> DirectionalFamilyResult:
    """Evaluate the frozen STRING-then-GO TDS-22 graph-minus-self Holm family."""
    return _evaluate_directional_family(
        frame,
        contrast_order=G2_CONTRAST_ORDER,
        condition_universes=condition_universes,
        evidence=evidence,
        positive_favors={graph: graph for graph in G2_CONTRAST_ORDER},
        negative_favors={graph: "self_loop_gat" for graph in G2_CONTRAST_ORDER},
        datasets=datasets,
        minimum_conditions=minimum_conditions,
        draws=draws,
        seed=seed,
        expected_resolution_bootstrap_draws=expected_resolution_bootstrap_draws,
    )

"""Condition-level inference with seeds treated as repeated measurements."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .artifacts import canonical_sha256, metric_row_from_artifact
from .errors import IncompleteSeedSetError, RevisionProtocolError, UnpairedConditionError

DEFAULT_CONDITION_KEYS = ("dataset", "hvg", "arm", "condition")
RESULT_LOCK_IDS = (
    "PRIMARY-DELTA-R",
    "PRIMARY-UNCERTAINTY-INTERVAL",
    "PRIMARY-DECISION",
    "PROPAGATION-CONTROL",
    "SCALE-INTERACTION",
    "TOPOLOGY-NULL",
    "REPRESENTATIVENESS-TRIGGER",
    "CONDITIONAL-PANEL",
    "FRACTION-IMPROVED",
    "EMPIRICAL-SEED-RESOLUTION",
    "SECONDARY-METRICS",
    "EXTERNAL-COMPARATOR-VALIDATION",
    "MEASURED-COMPUTE",
)


@dataclass(frozen=True)
class ContrastSummary:
    """Paired condition-level contrast and cluster-bootstrap interval."""

    left_arm: str
    right_arm: str
    metric: str
    n_conditions: int
    mean_difference: float
    median_difference: float
    paired_cohens_d: float
    uncertainty_interval_95_low: float
    uncertainty_interval_95_high: float
    fraction_positive: float


@dataclass(frozen=True)
class StratifiedContrastSummary:
    """Equal-weight dataset summary with conditions resampled within datasets."""

    left_arm: str
    right_arm: str
    metric: str
    n_datasets: int
    n_conditions: int
    equal_weight_dataset_mean_difference: float
    uncertainty_interval_95_low: float
    uncertainty_interval_95_high: float
    probability_difference_positive: float


@dataclass
class AnalysisReleaseResult:
    """One registry-lock result plus its auditable detail tables and failures."""

    registry: dict[str, Any]
    detail_tables: dict[str, pd.DataFrame]
    failures: pd.DataFrame

    @property
    def released(self) -> bool:
        return self.registry.get("status") == "RELEASED"


def metric_frame_from_artifacts(paths: Iterable[Path]) -> pd.DataFrame:
    """Load lossless artifacts and recompute their fold metrics."""

    rows = [metric_row_from_artifact(path) for path in paths]
    return pd.DataFrame(rows)


def seed_average_by_condition(
    frame: pd.DataFrame,
    metric_columns: Sequence[str],
    *,
    condition_keys: Sequence[str] = DEFAULT_CONDITION_KEYS,
    seed_column: str = "seed",
    expected_seeds: Sequence[int] = (42, 43, 44),
) -> pd.DataFrame:
    """Average seeds within each condition after enforcing the complete seed set."""

    required = [*condition_keys, seed_column, *metric_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RevisionProtocolError(f"Missing statistical columns: {missing}")
    duplicate_keys = [*condition_keys, seed_column]
    duplicates = frame.duplicated(duplicate_keys, keep=False)
    if duplicates.any():
        raise RevisionProtocolError(
            f"Found {int(duplicates.sum())} rows with duplicate condition-seed keys"
        )
    metric_values = frame[list(metric_columns)].to_numpy(dtype=float)
    if not np.isfinite(metric_values).all():
        raise RevisionProtocolError("Metric columns contain missing or non-finite values")

    expected = tuple(sorted(int(seed) for seed in expected_seeds))
    if not expected:
        raise RevisionProtocolError("expected_seeds cannot be empty")
    incomplete: list[str] = []
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(list(condition_keys), sort=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        observed = tuple(sorted(int(seed) for seed in group[seed_column]))
        if observed != expected:
            incomplete.append(f"{key_tuple!r}: observed={observed}")
            continue
        row = dict(zip(condition_keys, key_tuple, strict=True))
        row.update({metric: float(group[metric].mean()) for metric in metric_columns})
        row["n_seeds"] = len(observed)
        row["seed_ids"] = "|".join(str(seed) for seed in observed)
        rows.append(row)
    if incomplete:
        preview = "; ".join(incomplete[:10])
        raise IncompleteSeedSetError(
            f"{len(incomplete)} condition groups lack the predeclared seed set {expected}: {preview}"
        )
    return pd.DataFrame(rows)


def paired_condition_contrasts(
    frame: pd.DataFrame,
    left_arm: str,
    right_arm: str,
    metric: str,
    *,
    expected_seeds: Sequence[int] = (42, 43, 44),
    pair_keys: Sequence[str] = ("dataset", "hvg", "condition"),
) -> pd.DataFrame:
    """Pair seed-averaged arms by condition and return ``left - right`` differences."""

    averaged = seed_average_by_condition(
        frame,
        [metric],
        condition_keys=(*pair_keys[:-1], "arm", pair_keys[-1]),
        expected_seeds=expected_seeds,
    )
    left = averaged[averaged["arm"] == left_arm][[*pair_keys, metric, "n_seeds"]].copy()
    right = averaged[averaged["arm"] == right_arm][[*pair_keys, metric, "n_seeds"]].copy()
    left_keys = {tuple(row) for row in left[list(pair_keys)].itertuples(index=False, name=None)}
    right_keys = {tuple(row) for row in right[list(pair_keys)].itertuples(index=False, name=None)}
    if not left_keys or not right_keys:
        raise UnpairedConditionError(
            f"Both compared arms must be present; left={left_arm!r}, right={right_arm!r}"
        )
    if left_keys != right_keys:
        left_only = sorted(left_keys - right_keys)[:10]
        right_only = sorted(right_keys - left_keys)[:10]
        raise UnpairedConditionError(
            "Compared arms must have identical condition keys; "
            f"left_only={left_only}, right_only={right_only}"
        )
    merged = left.merge(
        right,
        on=list(pair_keys),
        how="inner",
        validate="one_to_one",
        suffixes=("_left", "_right"),
    )
    merged["left_arm"] = left_arm
    merged["right_arm"] = right_arm
    merged[f"delta_{metric}"] = merged[f"{metric}_left"] - merged[f"{metric}_right"]
    return merged


def summarize_paired_contrast(
    contrasts: pd.DataFrame,
    left_arm: str,
    right_arm: str,
    metric: str,
    *,
    random_seed: int,
    n_bootstrap: int = 20_000,
) -> ContrastSummary:
    """Summarize a contrast by resampling distinct held-out-condition rows."""

    difference_column = f"delta_{metric}"
    if difference_column not in contrasts:
        raise RevisionProtocolError(f"Missing {difference_column!r} in contrast table")
    values = contrasts[difference_column].to_numpy(dtype=float)
    if len(values) < 2:
        raise RevisionProtocolError("At least two distinct held-out conditions are required")
    if not np.isfinite(values).all():
        raise RevisionProtocolError("Contrast values must be finite")
    if n_bootstrap <= 0:
        raise RevisionProtocolError("n_bootstrap must be positive")
    rng = np.random.default_rng(random_seed)
    draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    standard_deviation = float(np.std(values, ddof=1))
    effect_size = float(np.mean(values) / standard_deviation) if standard_deviation > 0 else 0.0
    return ContrastSummary(
        left_arm=left_arm,
        right_arm=right_arm,
        metric=metric,
        n_conditions=len(values),
        mean_difference=float(np.mean(values)),
        median_difference=float(np.median(values)),
        paired_cohens_d=effect_size,
        uncertainty_interval_95_low=float(np.quantile(draws, 0.025)),
        uncertainty_interval_95_high=float(np.quantile(draws, 0.975)),
        fraction_positive=float(np.mean(values > 0)),
    )


def summarize_stratified_paired_contrast(
    contrasts: pd.DataFrame,
    left_arm: str,
    right_arm: str,
    metric: str,
    *,
    random_seed: int,
    dataset_column: str = "dataset",
    n_bootstrap: int = 20_000,
) -> StratifiedContrastSummary:
    """Bootstrap conditions within dataset and weight dataset means equally."""

    difference_column = f"delta_{metric}"
    required = {dataset_column, difference_column}
    if not required <= set(contrasts.columns):
        raise RevisionProtocolError(
            f"Contrast table is missing {sorted(required - set(contrasts))}"
        )
    strata = {
        str(dataset): group[difference_column].to_numpy(dtype=float)
        for dataset, group in contrasts.groupby(dataset_column, sort=True)
    }
    if not strata or any(len(values) < 2 for values in strata.values()):
        raise RevisionProtocolError("Each dataset stratum requires at least two conditions")
    if any(not np.isfinite(values).all() for values in strata.values()):
        raise RevisionProtocolError("Contrast values must be finite")
    if n_bootstrap <= 0:
        raise RevisionProtocolError("n_bootstrap must be positive")

    observed = float(np.mean([values.mean() for values in strata.values()]))
    rng = np.random.default_rng(random_seed)
    draws = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        draws[index] = np.mean(
            [
                rng.choice(values, size=len(values), replace=True).mean()
                for values in strata.values()
            ]
        )
    return StratifiedContrastSummary(
        left_arm=left_arm,
        right_arm=right_arm,
        metric=metric,
        n_datasets=len(strata),
        n_conditions=int(sum(len(values) for values in strata.values())),
        equal_weight_dataset_mean_difference=observed,
        uncertainty_interval_95_low=float(np.quantile(draws, 0.025)),
        uncertainty_interval_95_high=float(np.quantile(draws, 0.975)),
        probability_difference_positive=float(np.mean(draws > 0)),
    )


def bootstrap_settings_from_protocol(protocol: Mapping[str, Any]) -> dict[str, int]:
    """Return explicit bootstrap kwargs from the frozen protocol, with no code default."""

    statistics = protocol.get("statistics", {})
    try:
        n_bootstrap = int(statistics["bootstrap_replicates"])
        random_seed = int(statistics["bootstrap_random_seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise RevisionProtocolError(
            "Protocol must declare integer bootstrap_replicates and bootstrap_random_seed"
        ) from error
    if n_bootstrap <= 0:
        raise RevisionProtocolError("bootstrap_replicates must be positive")
    return {"n_bootstrap": n_bootstrap, "random_seed": random_seed}


def _primary_condition_bootstrap(
    table: pd.DataFrame,
    *,
    datasets: Sequence[str],
    cell_line_map: Mapping[str, str],
    random_seed: int,
    n_bootstrap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Resample conditions within dataset for both declared weighting estimands."""

    arrays: dict[str, np.ndarray] = {}
    for dataset in datasets:
        values = table.loc[table["dataset"] == dataset, "delta"].to_numpy(dtype=float)
        if len(values) < 2 or not np.isfinite(values).all():
            raise RevisionProtocolError(
                f"Dataset {dataset!r} needs at least two finite condition effects"
            )
        arrays[str(dataset)] = values
    if set(arrays) != set(cell_line_map):
        raise RevisionProtocolError("Cell-line map must cover exactly the benchmark datasets")
    grouped_datasets: dict[str, list[str]] = {}
    for dataset in datasets:
        grouped_datasets.setdefault(str(cell_line_map[dataset]), []).append(str(dataset))
    if set(grouped_datasets) != {"K562", "RPE1"}:
        raise RevisionProtocolError("Equal-cell-line sensitivity requires K562 and RPE1")

    rng = np.random.default_rng(random_seed)
    equal_dataset = np.empty(n_bootstrap, dtype=float)
    equal_cell_line = np.empty(n_bootstrap, dtype=float)
    fraction_improved = np.empty(n_bootstrap, dtype=float)
    digest = hashlib.sha256()
    for replicate in range(n_bootstrap):
        dataset_means: dict[str, float] = {}
        dataset_fractions: list[float] = []
        for dataset in datasets:
            values = arrays[str(dataset)]
            indices = rng.integers(0, len(values), size=len(values))
            digest.update(str(dataset).encode("utf-8"))
            digest.update(indices.astype(np.int64).tobytes())
            sample = values[indices]
            dataset_means[str(dataset)] = float(sample.mean())
            dataset_fractions.append(float(np.mean(sample > 0)))
        digest.update(int(replicate).to_bytes(8, "little", signed=False))
        equal_dataset[replicate] = float(np.mean(list(dataset_means.values())))
        equal_cell_line[replicate] = float(
            np.mean(
                [
                    np.mean([dataset_means[dataset] for dataset in cell_datasets])
                    for cell_datasets in grouped_datasets.values()
                ]
            )
        )
        fraction_improved[replicate] = float(np.mean(dataset_fractions))
    return equal_dataset, equal_cell_line, fraction_improved, digest.hexdigest()


def _dataset_condition_intervals(
    condition_table: pd.DataFrame,
    *,
    datasets: Sequence[str],
    n_bootstrap: int,
    random_seed: int,
) -> tuple[pd.DataFrame, str]:
    """Return dataset-local mean and fraction intervals from condition resampling."""

    rows = []
    digest = hashlib.sha256()
    for dataset_index, dataset in enumerate(datasets):
        values = condition_table.loc[condition_table["dataset"] == dataset, "delta"].to_numpy(
            dtype=float
        )
        if len(values) < 2:
            raise RevisionProtocolError(
                f"Dataset {dataset!r} requires at least two condition effects"
            )
        rng = np.random.default_rng(random_seed + 10_000 + dataset_index)
        indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
        digest.update(str(dataset).encode("utf-8"))
        digest.update(indices.astype(np.int64).tobytes())
        samples = values[indices]
        mean_low, mean_high = np.quantile(samples.mean(axis=1), [0.025, 0.975])
        fraction_low, fraction_high = np.quantile(np.mean(samples > 0, axis=1), [0.025, 0.975])
        rows.append(
            {
                "dataset": str(dataset),
                "uncertainty_interval_95_low": float(mean_low),
                "uncertainty_interval_95_high": float(mean_high),
                "fraction_improved_uncertainty_interval_95_low": float(fraction_low),
                "fraction_improved_uncertainty_interval_95_high": float(fraction_high),
            }
        )
    return pd.DataFrame(rows), digest.hexdigest()


def _leave_cluster_out_table(
    table: pd.DataFrame,
    *,
    cluster_column: str,
    value_column: str,
    datasets: Sequence[str],
) -> pd.DataFrame:
    """Compute the equal-dataset estimate after each dataset-local cluster omission."""

    dataset_full = {
        str(dataset): float(
            table.loc[table["dataset"] == dataset, value_column].to_numpy(dtype=float).mean()
        )
        for dataset in datasets
    }
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        local = table[table["dataset"] == dataset]
        for cluster in sorted(local[cluster_column].astype(str).unique()):
            retained = local[local[cluster_column].astype(str) != cluster][value_column].to_numpy(
                dtype=float
            )
            if not len(retained):
                raise RevisionProtocolError(
                    f"Cannot leave out the only {cluster_column} in dataset {dataset!r}"
                )
            estimates = dict(dataset_full)
            estimates[str(dataset)] = float(retained.mean())
            rows.append(
                {
                    "status": "PASS",
                    "reason": "APPLICABLE",
                    "omitted_dataset": str(dataset),
                    f"omitted_{cluster_column}": cluster,
                    "estimate": float(np.mean(list(estimates.values()))),
                }
            )
    return pd.DataFrame(rows)


def _pathway_sensitivity_tables(
    filtered: pd.DataFrame,
    condition_table: pd.DataFrame,
    *,
    datasets: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return leave-pathway-out evidence or a reason-coded explicit NA table."""

    unavailable = pd.DataFrame(
        [
            {
                "status": "NOT_AVAILABLE",
                "reason": "PATHWAY_ANNOTATION_NOT_RETAINED_IN_FOLD_ARTIFACTS",
                "omitted_dataset": "NOT_APPLICABLE",
                "omitted_pathway_class": "NOT_APPLICABLE",
                "estimate": np.nan,
            }
        ]
    )
    if "pathway_class" not in filtered.columns:
        return pd.DataFrame(columns=["dataset", "condition", "pathway_class"]), unavailable
    mapping = filtered[["dataset", "condition", "pathway_class"]].drop_duplicates()
    if (
        mapping.duplicated(["dataset", "condition"], keep=False).any()
        or mapping["pathway_class"].isna().any()
        or mapping["pathway_class"].astype(str).str.strip().eq("").any()
    ):
        unavailable.loc[0, "reason"] = "PATHWAY_ANNOTATION_INCOMPLETE_OR_NONUNIQUE"
        return mapping, unavailable
    pathway_table = (
        condition_table.merge(mapping, on=["dataset", "condition"], validate="one_to_one")
        .groupby(["dataset", "pathway_class"], sort=True)["delta"]
        .mean()
        .reset_index()
    )
    if any(len(group) < 2 for _, group in pathway_table.groupby("dataset", sort=True)):
        unavailable.loc[0, "reason"] = "FEWER_THAN_TWO_PATHWAYS_IN_AT_LEAST_ONE_DATASET"
        return mapping, unavailable
    return mapping, _leave_cluster_out_table(
        pathway_table,
        cluster_column="pathway_class",
        value_column="delta",
        datasets=datasets,
    )


def _shared_control_cell_bootstrap_feasibility(frame: pd.DataFrame) -> dict[str, str]:
    """Report whether cell-level shared-control resampling can be reconstructed."""

    required = {"shared_control_cell_id", "shared_control_profile_value"}
    if required <= set(frame.columns):
        values = frame[list(required)]
        if not values.isna().any().any() and len(values):
            return {
                "status": "FEASIBLE",
                "reason": "CELL_LEVEL_SHARED_CONTROL_IDENTIFIERS_AND_VALUES_RETAINED",
            }
    return {
        "status": "NOT_FEASIBLE",
        "reason": "CELL_LEVEL_SHARED_CONTROL_IDENTIFIERS_OR_VALUES_NOT_RETAINED",
    }


def primary_hierarchical_release(
    frame: pd.DataFrame,
    protocol: Mapping[str, Any],
    *,
    metric: str = "pearson_r",
) -> AnalysisReleaseResult:
    """Release the fixed STRING-GO contrast with conditional uncertainty summaries."""

    analysis_id = "PRIMARY"
    filtered, failures = _strict_analysis_block(
        frame,
        protocol,
        analysis_id=analysis_id,
        panel="primary",
        hvgs=(200,),
        arms=("string_go", "dense"),
        metric=metric,
    )
    if failures:
        return _withheld_result(analysis_id, failures)
    pivot = filtered.pivot(index=["dataset", "condition", "seed"], columns="arm", values=metric)
    pivot["delta"] = pivot["string_go"] - pivot["dense"]
    seed_delta_table = pivot["delta"].rename("seed_paired_delta").reset_index()
    target_map = filtered[["dataset", "condition", "canonical_target_set"]].drop_duplicates()
    if target_map.duplicated(["dataset", "condition"], keep=False).any():
        return _withheld_result(
            analysis_id,
            [_failure(analysis_id, "CONDITION_TARGET_MAPPING_NOT_UNIQUE")],
        )
    condition_table = (
        seed_delta_table.groupby(["dataset", "condition"], sort=True)["seed_paired_delta"]
        .agg(delta="mean", paired_seed_delta_sd="std")
        .reset_index()
        .merge(target_map, on=["dataset", "condition"], validate="one_to_one")
    )
    dataset_table = (
        condition_table.groupby("dataset", sort=True)["delta"]
        .agg(
            estimate="mean",
            n_conditions="size",
            fraction_conditions_improved=lambda values: float(np.mean(values > 0)),
        )
        .reset_index()
    )
    estimate = float(dataset_table["estimate"].mean())
    settings = bootstrap_settings_from_protocol(protocol)
    cell_line_map = {
        str(dataset): str(definition["cell_line"])
        for dataset, definition in protocol["datasets"].items()
    }
    primary_draws, equal_cell_line_draws, fraction_draws, index_hash = _primary_condition_bootstrap(
        condition_table,
        datasets=tuple(sorted(protocol["datasets"])),
        cell_line_map=cell_line_map,
        **settings,
    )
    interval_low, interval_high = np.quantile(primary_draws, [0.025, 0.975])
    equal_cell_line_low, equal_cell_line_high = np.quantile(equal_cell_line_draws, [0.025, 0.975])
    fraction_low, fraction_high = np.quantile(fraction_draws, [0.025, 0.975])
    dataset_intervals, dataset_interval_index_hash = _dataset_condition_intervals(
        condition_table,
        datasets=tuple(sorted(protocol["datasets"])),
        **settings,
    )
    dataset_table = dataset_table.merge(
        dataset_intervals, on="dataset", how="left", validate="one_to_one"
    )
    decision = (
        primary_decision_rule(float(interval_low), float(interval_high))
        if metric == "pearson_r"
        else "DESCRIPTIVE_SENSITIVITY_NO_PRIMARY_DECISION"
    )
    target_table = (
        condition_table.groupby(["dataset", "canonical_target_set"], sort=True)
        .agg(
            delta=("delta", "mean"),
            guide_condition_count=("condition", "size"),
            guide_condition_ids=("condition", lambda values: "|".join(sorted(values))),
        )
        .reset_index()
    )
    if any(len(group) < 2 for _, group in target_table.groupby("dataset", sort=True)):
        return _withheld_result(
            analysis_id,
            [_failure(analysis_id, "INSUFFICIENT_CANONICAL_TARGETS_FOR_CLUSTER_RESAMPLING")],
        )
    target_draws, target_index_hash = _condition_within_dataset_bootstrap(
        target_table,
        value_columns=("delta",),
        datasets=tuple(sorted(protocol["datasets"])),
        **settings,
    )
    target_low, target_high = np.quantile(target_draws[:, 0], [0.025, 0.975])
    leave_target_out = _leave_cluster_out_table(
        target_table,
        cluster_column="canonical_target_set",
        value_column="delta",
        datasets=tuple(sorted(protocol["datasets"])),
    )
    pathway_table, leave_pathway_out = _pathway_sensitivity_tables(
        filtered,
        condition_table,
        datasets=tuple(sorted(protocol["datasets"])),
    )
    shared_control_cell_bootstrap = {
        "status": "WITHHELD_PENDING_CELL_RESAMPLING",
        "reason": "HASH_BOUND_CONTROL_CELL_EVIDENCE_MUST_BE_RESAMPLED_DURING_POSTPROCESS",
    }
    equal_cell_line_table = (
        dataset_table.assign(cell_line=lambda table: table["dataset"].map(cell_line_map))
        .groupby("cell_line", sort=True)["estimate"]
        .agg(estimate="mean", n_benchmark_datasets="size")
        .reset_index()
    )
    equal_cell_line_estimate = float(equal_cell_line_table["estimate"].mean())
    seed_specific_dataset = (
        seed_delta_table.groupby(["dataset", "seed"], sort=True)["seed_paired_delta"]
        .mean()
        .rename("mean_paired_delta")
        .reset_index()
    )
    seed_specific_overall = (
        seed_specific_dataset.groupby("seed", sort=True)["mean_paired_delta"]
        .mean()
        .rename("equal_dataset_mean_paired_delta")
        .reset_index()
    )
    condition_seed_resolution = (
        condition_table.groupby("dataset", sort=True)["paired_seed_delta_sd"]
        .agg(
            median_condition_paired_seed_sd="median",
            p95_condition_paired_seed_sd=lambda values: float(np.quantile(values, 0.95)),
        )
        .reset_index()
    )
    dataset_seed_resolution = (
        seed_specific_dataset.groupby("dataset", sort=True)["mean_paired_delta"]
        .std(ddof=1)
        .rename("sd_across_seed_specific_dataset_means")
        .reset_index()
    )
    registry = {
        "analysis_id": analysis_id,
        "status": "RELEASED",
        "metric": metric,
        "estimand": (
            "equal_weight_dataset_mean_of_guide_condition_level_seed_averaged_"
            "string_go_minus_dense"
        ),
        "estimand_scope": "fixed_protocol_not_per_arm_optimum",
        "resampling_unit": "guide_condition; not a unique gene or target set",
        "target_level_generalization": "PROHIBITED_FROM_PRIMARY_ESTIMAND",
        "estimate": estimate,
        "uncertainty_interval_95_low": float(interval_low),
        "uncertainty_interval_95_high": float(interval_high),
        "interval_label": "95% conditional bootstrap uncertainty interval",
        "nominal_coverage_claim": False,
        "decision": decision,
        "decision_reference": "zero_only",
        "archived_optimisation_repeatability_reference": 0.010,
        "archived_reference_has_success_authority": False,
        "equal_dataset_mean_fraction_conditions_improved": float(
            dataset_table["fraction_conditions_improved"].mean()
        ),
        "fraction_improved_uncertainty_interval_95_low": float(fraction_low),
        "fraction_improved_uncertainty_interval_95_high": float(fraction_high),
        "dataset_estimates": dataset_table.to_dict(orient="records"),
        "dataset_composition": {
            "benchmark_datasets": 4,
            "K562_datasets": 3,
            "statement": "three_of_four_benchmark_datasets_are_K562",
        },
        "equal_cell_line_sensitivity": {
            "estimate": equal_cell_line_estimate,
            "uncertainty_interval_95_low": float(equal_cell_line_low),
            "uncertainty_interval_95_high": float(equal_cell_line_high),
            "interval_label": "95% conditional bootstrap uncertainty interval",
            "weighting": "K562_datasets_averaged_then_K562_and_RPE1_equal_weight",
            "cell_line_estimates": equal_cell_line_table.to_dict(orient="records"),
        },
        "target_cluster_resampling": {
            "status": "PASS",
            "estimate": float(target_table.groupby("dataset")["delta"].mean().mean()),
            "uncertainty_interval_95_low": float(target_low),
            "uncertainty_interval_95_high": float(target_high),
            "bootstrap_index_hash": target_index_hash,
            "canonical_target_sets": int(len(target_table)),
        },
        "leave_target_out": {
            "status": "PASS",
            "minimum_estimate": float(leave_target_out["estimate"].min()),
            "maximum_estimate": float(leave_target_out["estimate"].max()),
            "omissions": int(len(leave_target_out)),
        },
        "leave_pathway_out": {
            "status": str(leave_pathway_out.iloc[0]["status"]),
            "reason": str(leave_pathway_out.iloc[0]["reason"]),
        },
        "shared_control_cell_bootstrap": shared_control_cell_bootstrap,
        "n_conditions_expected": _expected_condition_total(protocol),
        "n_conditions_actual": int(len(condition_table)),
        "seed_ids": list(_expected_seeds(protocol)),
        "seed_completeness": "PASS",
        "bootstrap_replicates": settings["n_bootstrap"],
        "bootstrap_random_seed": settings["random_seed"],
        "bootstrap_index_hash": index_hash,
        "dataset_interval_bootstrap_index_hash": dataset_interval_index_hash,
        "input_results_manifest_hash": _results_manifest_hash(filtered, metric),
        "empirical_seed_resolution": {
            "condition_paired_seed_sd_by_dataset": condition_seed_resolution.to_dict(
                orient="records"
            ),
            "overall_median_condition_paired_seed_sd": float(
                condition_table["paired_seed_delta_sd"].median()
            ),
            "overall_p95_condition_paired_seed_sd": float(
                np.quantile(condition_table["paired_seed_delta_sd"], 0.95)
            ),
            "dataset_seed_specific_mean_sd": dataset_seed_resolution.to_dict(orient="records"),
            "overall_sd_across_seed_specific_equal_dataset_means": float(
                seed_specific_overall["equal_dataset_mean_paired_delta"].std(ddof=1)
            ),
        },
    }
    return AnalysisReleaseResult(
        registry=registry,
        detail_tables={
            "condition_contrasts": condition_table,
            "seed_paired_deltas": seed_delta_table,
            "seed_specific_dataset_means": seed_specific_dataset,
            "seed_specific_overall_means": seed_specific_overall,
            "dataset_estimates": dataset_table,
            "equal_cell_line_estimates": equal_cell_line_table,
            "target_cluster_estimates": target_table,
            "leave_target_out": leave_target_out,
            "pathway_mapping": pathway_table,
            "leave_pathway_out": leave_pathway_out,
        },
        failures=_failure_frame([]),
    )


def scale_interaction_release(
    frame: pd.DataFrame,
    protocol: Mapping[str, Any],
    *,
    metric: str = "pearson_r",
) -> AnalysisReleaseResult:
    """Release native-space and common-200 STRING-GO scale interactions."""

    analysis_id = "SCALE-INTERACTION"
    filtered, failures = _strict_analysis_block(
        frame,
        protocol,
        analysis_id=analysis_id,
        panel="primary",
        hvgs=(200, 500, 1000),
        arms=("string_go", "dense"),
        metric=metric,
    )
    if failures:
        return _withheld_result(analysis_id, failures)
    common_required = {
        "common_200_pearson_r",
        "common_evaluation_hvg",
        "common_gene_order_hash",
        "nested_gene_panel_manifest_hash",
    }
    missing_common = sorted(common_required - set(filtered.columns))
    if missing_common:
        return _withheld_result(
            analysis_id,
            [
                _failure(
                    analysis_id, "COMMON_200_RESULT_COLUMNS_MISSING", detail=str(missing_common)
                )
            ],
        )
    common_values = pd.to_numeric(filtered["common_200_pearson_r"], errors="coerce")
    if common_values.isna().any() or not np.isfinite(common_values.to_numpy(dtype=float)).all():
        return _withheld_result(
            analysis_id,
            [_failure(analysis_id, "COMMON_200_METRIC_NONFINITE")],
        )
    common_hvg = pd.to_numeric(filtered["common_evaluation_hvg"], errors="coerce")
    if common_hvg.isna().any() or not (common_hvg == 200).all():
        return _withheld_result(
            analysis_id,
            [_failure(analysis_id, "COMMON_EVALUATION_HVG_NOT_200")],
        )
    for dataset, group in filtered.groupby("dataset", sort=True):
        if (
            group["common_gene_order_hash"].nunique(dropna=False) != 1
            or group["nested_gene_panel_manifest_hash"].nunique(dropna=False) != 1
        ):
            failures.append(
                _failure(
                    analysis_id,
                    "NESTED_GENE_PANEL_BINDING_MISMATCH",
                    dataset=str(dataset),
                )
            )
    base = filtered[filtered["hvg"] == 200]
    if not np.allclose(
        base[metric].to_numpy(dtype=float),
        base["common_200_pearson_r"].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
    ):
        failures.append(_failure(analysis_id, "COMMON_200_BASE_METRIC_MISMATCH"))
    if failures:
        return _withheld_result(analysis_id, failures)

    contrast_columns = ("500_minus_200_hvg", "1000_minus_200_hvg")
    interaction_rows: list[pd.DataFrame] = []
    registries: list[dict[str, Any]] = []
    bootstrap_hashes: dict[str, str] = {}
    null_bootstrap_hashes: dict[str, str] = {}
    settings = bootstrap_settings_from_protocol(protocol)
    for evaluation_space, value_column in (
        ("native_hvg", metric),
        ("common_200_gene", "common_200_pearson_r"),
    ):
        pivot = filtered.pivot(
            index=["dataset", "condition", "seed"],
            columns=["hvg", "arm"],
            values=value_column,
        )
        local_rows: list[pd.DataFrame] = []
        for hvg in (500, 1000):
            values = (
                pivot[(hvg, "string_go")]
                - pivot[(hvg, "dense")]
                - pivot[(200, "string_go")]
                + pivot[(200, "dense")]
            )
            averaged = values.groupby(level=["dataset", "condition"]).mean().rename("interaction")
            local_rows.append(
                averaged.reset_index().assign(
                    contrast=f"{hvg}_minus_200_hvg",
                    evaluation_space=evaluation_space,
                )
            )
        local_long = pd.concat(local_rows, ignore_index=True)
        interaction_rows.append(local_long)
        wide = local_long.pivot(
            index=["dataset", "condition"], columns="contrast", values="interaction"
        ).reset_index()
        draws, index_hash = _condition_within_dataset_bootstrap(
            wide,
            value_columns=contrast_columns,
            datasets=tuple(sorted(protocol["datasets"])),
            **settings,
        )
        centered = wide.copy()
        for contrast in contrast_columns:
            centered[contrast] = centered[contrast] - centered.groupby("dataset")[
                contrast
            ].transform("mean")
        null_draws, null_index_hash = _condition_within_dataset_bootstrap(
            centered,
            value_columns=contrast_columns,
            datasets=tuple(sorted(protocol["datasets"])),
            **settings,
        )
        bootstrap_hashes[evaluation_space] = index_hash
        null_bootstrap_hashes[evaluation_space] = null_index_hash
        observed = np.asarray(
            [
                float(wide.groupby("dataset")[contrast].mean().mean())
                for contrast in contrast_columns
            ]
        )
        p_values = np.asarray(
            [_null_bootstrap_p(null_draws[:, index], observed[index]) for index in range(2)]
        )
        q_values = benjamini_hochberg(p_values)
        for index, contrast in enumerate(contrast_columns):
            interval_low, interval_high = np.quantile(draws[:, index], [0.025, 0.975])
            dataset_rows = (
                local_long[local_long["contrast"] == contrast]
                .groupby("dataset", sort=True)["interaction"]
                .agg(estimate="mean", n_conditions="size")
                .reset_index()
            )
            registries.append(
                {
                    "evaluation_space": evaluation_space,
                    "contrast": contrast,
                    "formula": ("mean_seed[(STRING_GO_h-dense_h)-(STRING_GO_200-dense_200)]"),
                    "estimate": float(observed[index]),
                    "uncertainty_interval_95_low": float(interval_low),
                    "uncertainty_interval_95_high": float(interval_high),
                    "interval_label": "95% conditional bootstrap uncertainty interval",
                    "two_sided_bootstrap_p": float(p_values[index]),
                    "bh_q_two_contrast_family": float(q_values[index]),
                    "dataset_estimates": dataset_rows.to_dict(orient="records"),
                }
            )
    long_table = pd.concat(interaction_rows, ignore_index=True)
    dataset_table = (
        long_table.groupby(["evaluation_space", "dataset", "contrast"], sort=True)["interaction"]
        .agg(estimate="mean", n_conditions="size")
        .reset_index()
    )
    registry = {
        "analysis_id": analysis_id,
        "status": "RELEASED",
        "metric": metric,
        "estimand": "paired_arm_by_scale_interaction_with_equal_dataset_weight",
        "contrasts": registries,
        "evaluation_spaces": ["native_hvg", "common_200_gene"],
        "multiplicity_family": list(contrast_columns),
        "multiplicity_method": "benjamini_hochberg_separately_within_evaluation_space",
        "n_conditions_expected": _expected_condition_total(protocol),
        "n_conditions_actual_per_evaluation_space": _expected_condition_total(protocol),
        "seed_ids": list(_expected_seeds(protocol)),
        "seed_completeness": "PASS",
        "invariant_condition_seed_support": "PASS",
        "bootstrap_replicates": settings["n_bootstrap"],
        "bootstrap_random_seed": settings["random_seed"],
        "bootstrap_index_hashes": bootstrap_hashes,
        "null_bootstrap_index_hashes": null_bootstrap_hashes,
        "input_results_manifest_hashes": {
            "native_hvg": _results_manifest_hash(filtered, metric),
            "common_200_gene": _results_manifest_hash(filtered, "common_200_pearson_r"),
        },
        "nested_gene_panel_binding_status": "PASS",
    }
    return AnalysisReleaseResult(
        registry=registry,
        detail_tables={
            "condition_interactions": long_table,
            "dataset_estimates": dataset_table,
        },
        failures=_failure_frame([]),
    )


def propagation_control_release(
    frame: pd.DataFrame,
    protocol: Mapping[str, Any],
    *,
    metric: str = "pearson_r",
) -> AnalysisReleaseResult:
    """Release STRING-GO and dense contrasts against one matched self-loop control."""

    analysis_id = "PROPAGATION-CONTROL"
    filtered, failures = _strict_analysis_block(
        frame,
        protocol,
        analysis_id=analysis_id,
        panel="primary",
        hvgs=(200,),
        arms=("string_go", "dense", "self_loop"),
        metric=metric,
    )
    if failures:
        return _withheld_result(analysis_id, failures)
    pivot = filtered.pivot(index=["dataset", "condition", "seed"], columns="arm", values=metric)
    seed_contrasts = pd.DataFrame(
        {
            "string_go_minus_self_loop": pivot["string_go"] - pivot["self_loop"],
            "dense_minus_self_loop": pivot["dense"] - pivot["self_loop"],
        }
    )
    condition_table = seed_contrasts.groupby(level=["dataset", "condition"]).mean().reset_index()
    contrast_columns = ("string_go_minus_self_loop", "dense_minus_self_loop")
    dataset_table = (
        condition_table.melt(
            id_vars=["dataset", "condition"],
            value_vars=list(contrast_columns),
            var_name="contrast",
            value_name="delta",
        )
        .groupby(["dataset", "contrast"], sort=True)["delta"]
        .agg(estimate="mean", n_conditions="size")
        .reset_index()
    )
    settings = bootstrap_settings_from_protocol(protocol)
    draws, index_hash = _condition_within_dataset_bootstrap(
        condition_table,
        value_columns=contrast_columns,
        datasets=tuple(sorted(protocol["datasets"])),
        **settings,
    )
    centered = condition_table.copy()
    for contrast in contrast_columns:
        centered[contrast] = centered[contrast] - centered.groupby("dataset")[contrast].transform(
            "mean"
        )
    null_draws, null_index_hash = _condition_within_dataset_bootstrap(
        centered,
        value_columns=contrast_columns,
        datasets=tuple(sorted(protocol["datasets"])),
        **settings,
    )
    observed = np.asarray(
        [
            float(condition_table.groupby("dataset")[contrast].mean().mean())
            for contrast in contrast_columns
        ]
    )
    p_values = np.asarray(
        [_null_bootstrap_p(null_draws[:, index], observed[index]) for index in range(2)]
    )
    q_values = benjamini_hochberg(p_values)
    contrasts = []
    for index, contrast in enumerate(contrast_columns):
        ci_low, ci_high = np.quantile(draws[:, index], [0.025, 0.975])
        contrasts.append(
            {
                "contrast": contrast,
                "formula": (
                    "mean_seed[string_go-self_loop]"
                    if contrast == "string_go_minus_self_loop"
                    else "mean_seed[dense-self_loop]"
                ),
                "estimate": float(observed[index]),
                "uncertainty_interval_95_low": float(ci_low),
                "uncertainty_interval_95_high": float(ci_high),
                "interval_label": "95% conditional bootstrap uncertainty interval",
                "two_sided_bootstrap_p": float(p_values[index]),
                "bh_q_two_contrast_family": float(q_values[index]),
                "dataset_estimates": dataset_table[dataset_table["contrast"] == contrast].to_dict(
                    orient="records"
                ),
            }
        )
    registry = {
        "analysis_id": analysis_id,
        "status": "RELEASED",
        "metric": metric,
        "estimand": (
            "equal_weight_dataset_means_of_string_go_and_dense_against_the_same_"
            "matched_self_loop_control"
        ),
        "contrasts": contrasts,
        "multiplicity_family": list(contrast_columns),
        "multiplicity_method": "benjamini_hochberg",
        "support_scope": "primary_panel_200_hvg_complete_three_arm_matched_support",
        "common_self_loop_control": (
            "one identical dataset-condition-seed self_loop observation is subtracted "
            "from both string_go and dense"
        ),
        "n_conditions_expected": _expected_condition_total(protocol),
        "n_conditions_actual": int(len(condition_table)),
        "seed_ids": list(_expected_seeds(protocol)),
        "seed_completeness": "PASS",
        "bootstrap_replicates": settings["n_bootstrap"],
        "bootstrap_random_seed": settings["random_seed"],
        "bootstrap_index_hash": index_hash,
        "null_bootstrap_index_hash": null_index_hash,
        "input_results_manifest_hash": _results_manifest_hash(filtered, metric),
    }
    return AnalysisReleaseResult(
        registry=registry,
        detail_tables={
            "condition_contrasts": condition_table,
            "dataset_estimates": dataset_table,
        },
        failures=_failure_frame([]),
    )


def mandatory_fit_coverage_release(
    frame: pd.DataFrame,
    protocol: Mapping[str, Any],
    preflight_summaries: Sequence[Mapping[str, Any]] | None = None,
    *,
    protocol_file_hash: str | None = None,
    expected_code_hash: str | None = None,
) -> AnalysisReleaseResult:
    """Require every unique key in the frozen 10,800-fit mandatory design."""

    analysis_id = "MANDATORY-FIT-COVERAGE"
    required_columns = {
        "dataset",
        "hvg",
        "panel",
        "arm",
        "condition",
        "seed",
        "analysis_block",
        "epochs_requested",
        "artifact_hash",
        "initialization_hash",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        return _withheld_result(
            analysis_id,
            [_failure(analysis_id, "MISSING_RESULT_COLUMNS", detail=str(missing_columns))],
        )
    combinations: set[tuple[int, str]] = {
        (200, str(arm)) for arm in protocol["analysis_blocks"]["topology_primary"]["default_arms"]
    }
    combinations.update(
        (200, str(arm))
        for arm in protocol["analysis_blocks"]["mixed_support_sensitivity"]["default_arms"]
    )
    for hvg, arms in protocol["analysis_blocks"]["scale_extension"]["default_arms_by_hvg"].items():
        combinations.update((int(hvg), str(arm)) for arm in arms)
    datasets = tuple(sorted(protocol["datasets"]))
    seeds = _expected_seeds(protocol)
    expected_conditions = int(protocol["planned_fit_counts"]["conditions_per_panel"])
    failures: list[dict[str, Any]] = []
    expected_keys: set[tuple[Any, ...]] = set()
    for dataset in datasets:
        reference = frame[
            (frame["dataset"] == dataset)
            & (frame["panel"] == "primary")
            & (frame["hvg"] == 200)
            & (frame["arm"] == "string_go")
        ]
        conditions = sorted(set(reference["condition"].astype(str)))
        if len(conditions) != expected_conditions:
            failures.append(
                _failure(
                    analysis_id,
                    "CONDITION_COUNT_MISMATCH",
                    dataset=dataset,
                    detail=f"expected={expected_conditions}; observed={len(conditions)}",
                )
            )
        expected_keys.update(
            (dataset, hvg, arm, condition, seed)
            for hvg, arm in combinations
            for condition in conditions
            for seed in seeds
        )
    mandatory = frame[
        (frame["panel"] == "primary")
        & frame.apply(lambda row: (int(row["hvg"]), str(row["arm"])) in combinations, axis=1)
        & frame["dataset"].isin(datasets)
    ]
    binding_failures, binding_hash = _preflight_binding_failures(
        mandatory,
        protocol,
        preflight_summaries,
        analysis_id=analysis_id,
        panel="primary",
        protocol_file_hash=protocol_file_hash,
        expected_code_hash=expected_code_hash,
    )
    failures.extend(binding_failures)
    key_columns = ["dataset", "hvg", "arm", "condition", "seed"]
    duplicates = mandatory.duplicated(key_columns, keep=False)
    if duplicates.any():
        failures.append(
            _failure(
                analysis_id,
                "DUPLICATE_MANDATORY_FIT_KEY",
                detail=f"duplicate_rows={int(duplicates.sum())}",
            )
        )
    observed_keys = {
        (str(dataset), int(hvg), str(arm), str(condition), int(seed))
        for dataset, hvg, arm, condition, seed in mandatory[key_columns].itertuples(
            index=False, name=None
        )
    }
    missing_keys = expected_keys - observed_keys
    extra_keys = observed_keys - expected_keys
    if missing_keys:
        failures.append(
            _failure(
                analysis_id,
                "MISSING_MANDATORY_FIT_KEYS",
                detail=f"count={len(missing_keys)}; preview={sorted(missing_keys)[:5]}",
            )
        )
    if extra_keys:
        failures.append(
            _failure(
                analysis_id,
                "UNEXPECTED_MANDATORY_FIT_KEYS",
                detail=f"count={len(extra_keys)}; preview={sorted(extra_keys)[:5]}",
            )
        )
    expected_total = int(protocol["planned_fit_counts"]["mandatory_total_fits"])
    if len(expected_keys) != expected_total or len(observed_keys) != expected_total:
        failures.append(
            _failure(
                analysis_id,
                "MANDATORY_FIT_COUNT_MISMATCH",
                detail=(
                    f"protocol={expected_total}; expected_keys={len(expected_keys)}; "
                    f"observed_unique_keys={len(observed_keys)}"
                ),
            )
        )
    if failures:
        return _withheld_result(analysis_id, failures)
    initialization_rows: list[dict[str, Any]] = []
    for (dataset, hvg, condition, seed), group in mandatory.groupby(
        ["dataset", "hvg", "condition", "seed"], sort=True
    ):
        expected_arms = sorted(arm for scale, arm in combinations if scale == int(hvg))
        observed_arms = sorted(group["arm"].astype(str).unique())
        initial_hashes = sorted(group["initialization_hash"].astype(str).unique())
        status = "PASS" if observed_arms == expected_arms and len(initial_hashes) == 1 else "FAIL"
        initialization_rows.append(
            {
                "dataset": str(dataset),
                "hvg": int(hvg),
                "condition": str(condition),
                "seed": int(seed),
                "expected_arms": "|".join(expected_arms),
                "observed_arms": "|".join(observed_arms),
                "initial_state_sha256": initial_hashes[0] if len(initial_hashes) == 1 else "",
                "status": status,
            }
        )
    initialization_audit = pd.DataFrame(initialization_rows)
    if initialization_audit.empty or not initialization_audit["status"].eq("PASS").all():
        return _withheld_result(
            analysis_id,
            [_failure(analysis_id, "PAIRED_INITIAL_STATE_HASH_MISMATCH")],
        )
    registry = {
        "analysis_id": analysis_id,
        "status": "RELEASED",
        "expected_unique_fit_keys": expected_total,
        "observed_unique_fit_keys": len(observed_keys),
        "arm_scale_cells_per_dataset": len(combinations),
        "artifact_manifest_hash": canonical_sha256(
            sorted(mandatory["artifact_hash"].astype(str).tolist())
        ),
        "preflight_binding_manifest_hash": binding_hash,
        "initialization_interpretation": "identically_initialized_separately_trained",
        "paired_initial_state_hash_status": "PASS",
        "paired_initial_state_audit_hash": canonical_sha256(
            initialization_audit.to_dict(orient="records")
        ),
    }
    return AnalysisReleaseResult(
        registry=registry,
        detail_tables={"paired_initial_state_hash_audit": initialization_audit},
        failures=_failure_frame([]),
    )


def _validate_topology_diagnostics(
    frame: pd.DataFrame,
    graph_arms: Sequence[str],
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Independently verify every retained STRING-GO topology-null diagnostic record."""

    analysis_id = "TOPOLOGY-NULL"
    diagnostic_keys = (
        "arm",
        "mode",
        "n_nodes",
        "n_undirected_nonself_edges",
        "degree_sequence_sha256",
        "n_connected_components",
        "component_partition_sha256",
        "n_isolates",
        "support_sha256",
        "source_arm",
        "source_support_sha256",
        "source_edge_sha256",
        "source_n_nodes",
        "source_n_undirected_nonself_edges",
        "source_degree_sequence_sha256",
        "source_n_connected_components",
        "source_component_partition_sha256",
        "source_n_isolates",
        "swapped_edge_fraction",
        "rewire_seed",
        "cross_dataset_graph_index_pairing",
        "diagnostics_sha256",
    )
    columns = tuple(f"graph_diagnostic_{key}" for key in diagnostic_keys)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        return pd.DataFrame(), [
            _failure(analysis_id, "TOPOLOGY_GRAPH_DIAGNOSTICS_MISSING", detail=str(missing))
        ]
    failures: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    expected_seeds = tuple(
        int(value) for value in protocol["graph_supports"]["topology_null"]["replicate_seeds"]
    )
    minimum_fraction = float(
        protocol["graph_supports"]["topology_null"]["minimum_swapped_edge_fraction"]
    )
    for dataset in sorted(protocol["datasets"]):
        dataset_rows = frame[frame["dataset"] == dataset]
        normalized: dict[str, dict[str, Any]] = {}
        for arm in ("string_go", *graph_arms):
            arm_rows = dataset_rows[dataset_rows["arm"] == arm]
            if arm_rows.empty:
                failures.append(
                    _failure(
                        analysis_id,
                        "TOPOLOGY_DIAGNOSTIC_ARM_MISSING",
                        dataset=dataset,
                        arm=arm,
                    )
                )
                continue
            if any(arm_rows[column].nunique(dropna=False) != 1 for column in columns):
                failures.append(
                    _failure(
                        analysis_id,
                        "TOPOLOGY_DIAGNOSTIC_NOT_CONSTANT_ACROSS_FOLDS",
                        dataset=dataset,
                        arm=arm,
                    )
                )
                continue
            row = arm_rows.iloc[0]
            try:
                record: dict[str, Any] = {
                    "arm": str(row["graph_diagnostic_arm"]),
                    "mode": str(row["graph_diagnostic_mode"]),
                    "n_nodes": int(row["graph_diagnostic_n_nodes"]),
                    "n_undirected_nonself_edges": int(
                        row["graph_diagnostic_n_undirected_nonself_edges"]
                    ),
                    "degree_sequence_sha256": str(row["graph_diagnostic_degree_sequence_sha256"]),
                    "n_connected_components": int(row["graph_diagnostic_n_connected_components"]),
                    "component_partition_sha256": str(
                        row["graph_diagnostic_component_partition_sha256"]
                    ),
                    "n_isolates": int(row["graph_diagnostic_n_isolates"]),
                    "support_sha256": str(row["graph_diagnostic_support_sha256"]),
                    "source_arm": str(row["graph_diagnostic_source_arm"]),
                    "source_support_sha256": str(row["graph_diagnostic_source_support_sha256"]),
                    "source_edge_sha256": str(row["graph_diagnostic_source_edge_sha256"]),
                    "source_n_nodes": int(row["graph_diagnostic_source_n_nodes"]),
                    "source_n_undirected_nonself_edges": int(
                        row["graph_diagnostic_source_n_undirected_nonself_edges"]
                    ),
                    "source_degree_sequence_sha256": str(
                        row["graph_diagnostic_source_degree_sequence_sha256"]
                    ),
                    "source_n_connected_components": int(
                        row["graph_diagnostic_source_n_connected_components"]
                    ),
                    "source_component_partition_sha256": str(
                        row["graph_diagnostic_source_component_partition_sha256"]
                    ),
                    "source_n_isolates": int(row["graph_diagnostic_source_n_isolates"]),
                    "swapped_edge_fraction": (
                        None
                        if pd.isna(row["graph_diagnostic_swapped_edge_fraction"])
                        else float(row["graph_diagnostic_swapped_edge_fraction"])
                    ),
                    "rewire_seed": (
                        None
                        if pd.isna(row["graph_diagnostic_rewire_seed"])
                        else int(row["graph_diagnostic_rewire_seed"])
                    ),
                    "cross_dataset_graph_index_pairing": str(
                        row["graph_diagnostic_cross_dataset_graph_index_pairing"]
                    ),
                }
                declared_hash = str(row["graph_diagnostic_diagnostics_sha256"])
            except (TypeError, ValueError) as error:
                failures.append(
                    _failure(
                        analysis_id,
                        "TOPOLOGY_DIAGNOSTIC_TYPE_INVALID",
                        dataset=dataset,
                        arm=arm,
                        detail=str(error),
                    )
                )
                continue
            if (
                record["arm"] != arm
                or record["support_sha256"] != str(arm_rows.iloc[0]["graph_support_hash"])
                or declared_hash != canonical_sha256(record)
            ):
                failures.append(
                    _failure(
                        analysis_id,
                        "TOPOLOGY_DIAGNOSTIC_HASH_OR_ARTIFACT_BINDING_INVALID",
                        dataset=dataset,
                        arm=arm,
                    )
                )
            record["diagnostics_sha256"] = declared_hash
            normalized[arm] = record
            records.append({"dataset": dataset, **record})
        base = normalized.get("string_go")
        if base is None:
            continue
        if (
            base["mode"] != "curated"
            or base["source_arm"] != "string_go"
            or base["source_support_sha256"] != base["support_sha256"]
            or base["source_n_nodes"] != base["n_nodes"]
            or base["source_n_undirected_nonself_edges"] != base["n_undirected_nonself_edges"]
            or base["source_degree_sequence_sha256"] != base["degree_sequence_sha256"]
            or base["source_n_connected_components"] != base["n_connected_components"]
            or base["source_component_partition_sha256"] != base["component_partition_sha256"]
            or base["source_n_isolates"] != base["n_isolates"]
        ):
            failures.append(
                _failure(analysis_id, "STRING_GO_BASE_DIAGNOSTIC_INVALID", dataset=dataset)
            )
        rewire_hashes: list[str] = []
        for index, arm in enumerate(graph_arms):
            record = normalized.get(arm)
            if record is None:
                continue
            rewire_hashes.append(record["support_sha256"])
            if (
                record["mode"] != "degree_preserving_rewire"
                or record["source_arm"] != "string_go"
                or record["source_support_sha256"] != base["support_sha256"]
                or record["source_edge_sha256"] != base["source_edge_sha256"]
                or record["n_nodes"] != base["n_nodes"]
                or record["n_undirected_nonself_edges"] != base["n_undirected_nonself_edges"]
                or record["degree_sequence_sha256"] != base["degree_sequence_sha256"]
                or record["n_connected_components"] != base["n_connected_components"]
                or record["component_partition_sha256"] != base["component_partition_sha256"]
                or record["n_isolates"] != base["n_isolates"]
                or record["source_n_nodes"] != base["n_nodes"]
                or record["source_n_undirected_nonself_edges"] != base["n_undirected_nonself_edges"]
                or record["source_degree_sequence_sha256"] != base["degree_sequence_sha256"]
                or record["source_n_connected_components"] != base["n_connected_components"]
                or record["source_component_partition_sha256"] != base["component_partition_sha256"]
                or record["source_n_isolates"] != base["n_isolates"]
                or record["swapped_edge_fraction"] is None
                or record["swapped_edge_fraction"] < minimum_fraction
                or record["rewire_seed"] != expected_seeds[index]
                or record["cross_dataset_graph_index_pairing"] != "PROHIBITED_LOCAL_INSTANCE_LABEL"
            ):
                failures.append(
                    _failure(
                        analysis_id,
                        "TOPOLOGY_PRESERVATION_DIAGNOSTIC_FAILED",
                        dataset=dataset,
                        arm=arm,
                    )
                )
        if (
            len(rewire_hashes) != 10
            or len(set(rewire_hashes)) != 10
            or base["support_sha256"] in set(rewire_hashes)
        ):
            failures.append(
                _failure(analysis_id, "TOPOLOGY_SUPPORT_HASH_UNIQUENESS_FAILED", dataset=dataset)
            )
    table = pd.DataFrame(records).sort_values(["dataset", "arm"]).reset_index(drop=True)
    return table, failures


def topology_hierarchical_release(
    frame: pd.DataFrame,
    protocol: Mapping[str, Any],
    *,
    metric: str = "pearson_r",
) -> AnalysisReleaseResult:
    """Release the STRING-GO-minus-rewire ensemble after topology-diagnostic validation."""

    analysis_id = "TOPOLOGY-NULL"
    graph_arms = tuple(f"string_go_rewire_{index:02d}" for index in range(1, 11))
    filtered, failures = _strict_analysis_block(
        frame,
        protocol,
        analysis_id=analysis_id,
        panel="primary",
        hvgs=(200,),
        arms=("string_go", *graph_arms),
        metric=metric,
    )
    if failures:
        return _withheld_result(analysis_id, failures)
    diagnostic_table, diagnostic_failures = _validate_topology_diagnostics(
        filtered, graph_arms, protocol
    )
    if diagnostic_failures:
        return _withheld_result(analysis_id, diagnostic_failures)
    pivot = filtered.pivot(index=["dataset", "condition", "seed"], columns="arm", values=metric)
    per_graph: list[pd.DataFrame] = []
    for graph_index, arm in enumerate(graph_arms, start=1):
        delta = (pivot["string_go"] - pivot[arm]).groupby(level=["dataset", "condition"]).mean()
        per_graph.append(
            delta.rename("delta").reset_index().assign(graph_id=graph_index, rewire_arm=arm)
        )
    long_table = pd.concat(per_graph, ignore_index=True)
    matrices: dict[str, np.ndarray] = {}
    centered_matrices: dict[str, np.ndarray] = {}
    for dataset in sorted(protocol["datasets"]):
        subset = long_table[long_table["dataset"] == dataset]
        matrix = subset.pivot(index="condition", columns="graph_id", values="delta").sort_index()
        matrices[dataset] = matrix.to_numpy(dtype=float)
        centered_matrices[dataset] = matrices[dataset] - matrices[dataset].mean()
    dataset_table = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "estimate": float(matrix.mean()),
                "n_conditions": int(matrix.shape[0]),
                "n_graph_instances": int(matrix.shape[1]),
                "mean_across_local_graph_instances": float(matrix.mean()),
                "sd_across_local_graph_instance_means": float(np.std(matrix.mean(axis=0), ddof=1)),
            }
            for dataset, matrix in matrices.items()
        ]
    )
    estimate = float(dataset_table["estimate"].mean())
    settings = bootstrap_settings_from_protocol(protocol)
    rng = np.random.default_rng(settings["random_seed"])
    digest = hashlib.sha256()
    draws = np.empty(settings["n_bootstrap"], dtype=float)
    null_draws = np.empty(settings["n_bootstrap"], dtype=float)
    for replicate in range(settings["n_bootstrap"]):
        dataset_draws: list[float] = []
        dataset_null_draws: list[float] = []
        for dataset in sorted(matrices):
            matrix = matrices[dataset]
            condition_indices = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
            graph_indices = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
            digest.update(dataset.encode("utf-8"))
            digest.update(condition_indices.astype(np.int64).tobytes())
            digest.update(graph_indices.astype(np.int64).tobytes())
            dataset_draws.append(float(matrix[np.ix_(condition_indices, graph_indices)].mean()))
            dataset_null_draws.append(
                float(centered_matrices[dataset][np.ix_(condition_indices, graph_indices)].mean())
            )
        draws[replicate] = float(np.mean(dataset_draws))
        null_draws[replicate] = float(np.mean(dataset_null_draws))
    graph_table = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "local_graph_index": graph_index,
                "rewire_arm": graph_arms[graph_index - 1],
                "local_graph_condition_mean_contrast": float(matrix[:, graph_index - 1].mean()),
                "cross_dataset_pairing_status": "NOT_PAIRED_ACROSS_DATASETS",
            }
            for dataset, matrix in matrices.items()
            for graph_index in range(1, matrix.shape[1] + 1)
        ]
    )
    local_graph_estimates = graph_table["local_graph_condition_mean_contrast"].to_numpy(dtype=float)
    ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
    if float(ci_low) > 0:
        directional_support = "TOPOLOGY_DIRECTIONALLY_SUPPORTIVE"
    elif float(ci_high) < 0:
        directional_support = "TOPOLOGY_DIRECTIONALLY_OPPOSED"
    else:
        directional_support = "TOPOLOGY_DIRECTION_INCONCLUSIVE"
    registry = {
        "analysis_id": analysis_id,
        "status": "RELEASED",
        "metric": metric,
        "estimand": (
            "equal_dataset_mean_condition_and_graph_instance_average_" "string_go_minus_rewire"
        ),
        "estimate": estimate,
        "uncertainty_interval_95_low": float(ci_low),
        "uncertainty_interval_95_high": float(ci_high),
        "interval_label": "95% conditional bootstrap uncertainty interval",
        "nominal_coverage_claim": False,
        "directional_support": directional_support,
        "diagnostics_status": "TOPOLOGY_DIAGNOSTICS_PASS",
        "graph_diagnostic_manifest_hash": canonical_sha256(
            diagnostic_table.fillna("NOT_APPLICABLE").to_dict(orient="records")
        ),
        "n_graph_instances_expected": 10,
        "n_graph_instances_actual": int(len(graph_arms)),
        "graph_instance_coverage": "PASS",
        "two_sided_centered_null_bootstrap_p": _null_bootstrap_p(null_draws, estimate),
        "local_graph_instances_total": int(len(local_graph_estimates)),
        "local_graph_instance_mean": float(np.mean(local_graph_estimates)),
        "local_graph_instance_sd_descriptive": float(np.std(local_graph_estimates, ddof=1)),
        "graph_index_cross_dataset_pairing": "PROHIBITED_ARBITRARY_LABELS",
        "dataset_estimates": dataset_table.to_dict(orient="records"),
        "n_conditions_expected": _expected_condition_total(protocol),
        "n_conditions_actual": int(sum(matrix.shape[0] for matrix in matrices.values())),
        "seed_ids": list(_expected_seeds(protocol)),
        "seed_completeness": "PASS",
        "bootstrap_scheme": (
            "conditions_and_local_graph_instances_resampled_independently_within_each_dataset"
        ),
        "bootstrap_replicates": settings["n_bootstrap"],
        "bootstrap_random_seed": settings["random_seed"],
        "bootstrap_index_hash": digest.hexdigest(),
        "null_bootstrap_centering": "within_dataset_over_all_conditions_and_local_graphs",
        "input_results_manifest_hash": _results_manifest_hash(filtered, metric),
    }
    return AnalysisReleaseResult(
        registry=registry,
        detail_tables={
            "condition_graph_contrasts": long_table,
            "dataset_estimates": dataset_table,
            "graph_instance_estimates": graph_table,
            "graph_diagnostics": diagnostic_table,
        },
        failures=_failure_frame([]),
    )


def conditional_nonoverlap_release(
    frame: pd.DataFrame,
    protocol: Mapping[str, Any],
    global_trigger_manifest: Mapping[str, Any],
    preflight_summaries: Sequence[Mapping[str, Any]] | None = None,
    *,
    metric: str = "pearson_r",
    protocol_file_hash: str | None = None,
    expected_code_hash: str | None = None,
) -> AnalysisReleaseResult:
    """Release the separate four-dataset sensitivity effect only under the global OR gate."""

    analysis_id = "CONDITIONAL-NONOVERLAP"
    if global_trigger_manifest.get("status") != "PASS":
        return _withheld_result(
            analysis_id,
            [_failure(analysis_id, "GLOBAL_TRIGGER_MANIFEST_NOT_PASS")],
        )
    if not global_trigger_manifest.get("global_triggered"):
        unexpected = frame[
            (frame.get("panel", pd.Series(index=frame.index, dtype=object)) == "sensitivity")
            & (
                frame.get("analysis_block", pd.Series(index=frame.index, dtype=object))
                == "conditional_nonoverlap"
            )
        ]
        if len(unexpected):
            return _withheld_result(
                analysis_id,
                [
                    _failure(
                        analysis_id,
                        "UNAUTHORIZED_CONDITIONAL_ARTIFACTS_WHEN_NOT_TRIGGERED",
                        detail=f"rows={len(unexpected)}",
                    )
                ],
            )
        return AnalysisReleaseResult(
            registry={
                "analysis_id": analysis_id,
                "status": "RELEASED",
                "execution_status": "NOT_TRIGGERED_NOT_REQUIRED",
                "global_trigger_manifest_hash": global_trigger_manifest.get("manifest_hash"),
                "model_fits_required": 0,
            },
            detail_tables={},
            failures=_failure_frame([]),
        )
    filtered, failures = _strict_analysis_block(
        frame,
        protocol,
        analysis_id=analysis_id,
        panel="sensitivity",
        hvgs=(200,),
        arms=("string_go", "dense"),
        metric=metric,
    )
    expected_manifest_hash = global_trigger_manifest.get("manifest_hash")
    if "global_trigger_manifest_hash" not in filtered:
        failures.append(_failure(analysis_id, "GLOBAL_TRIGGER_ARTIFACT_LINK_MISSING"))
    elif set(filtered["global_trigger_manifest_hash"].dropna()) != {expected_manifest_hash}:
        failures.append(_failure(analysis_id, "GLOBAL_TRIGGER_ARTIFACT_LINK_MISMATCH"))
    binding_failures, binding_hash = _preflight_binding_failures(
        filtered,
        protocol,
        preflight_summaries,
        analysis_id=analysis_id,
        panel="sensitivity",
        protocol_file_hash=protocol_file_hash,
        expected_code_hash=expected_code_hash,
    )
    failures.extend(binding_failures)
    global_rows = {
        str(row.get("dataset")): row for row in global_trigger_manifest.get("dataset_triggers", [])
    }
    for dataset in sorted(protocol["datasets"]):
        dataset_rows = filtered[filtered["dataset"] == dataset]
        expected_panel_hash = global_rows.get(dataset, {}).get("condition_panel_manifest_hash")
        if set(dataset_rows["condition_panel_manifest_hash"].dropna()) != {expected_panel_hash}:
            failures.append(
                _failure(
                    analysis_id,
                    "GLOBAL_TRIGGER_PANEL_BINDING_MISMATCH",
                    dataset=dataset,
                )
            )
    if failures:
        return _withheld_result(analysis_id, failures)
    pivot = filtered.pivot(index=["dataset", "condition", "seed"], columns="arm", values=metric)
    condition_table = (
        (pivot["string_go"] - pivot["dense"])
        .groupby(level=["dataset", "condition"])
        .mean()
        .rename("delta")
        .reset_index()
    )
    dataset_table = (
        condition_table.groupby("dataset", sort=True)["delta"]
        .agg(estimate="mean", n_conditions="size")
        .reset_index()
    )
    settings = bootstrap_settings_from_protocol(protocol)
    draws, index_hash = _condition_within_dataset_bootstrap(
        condition_table,
        value_columns=("delta",),
        datasets=tuple(sorted(protocol["datasets"])),
        **settings,
    )
    centered = condition_table.copy()
    centered["delta"] = centered["delta"] - centered.groupby("dataset")["delta"].transform("mean")
    null_draws, null_index_hash = _condition_within_dataset_bootstrap(
        centered,
        value_columns=("delta",),
        datasets=tuple(sorted(protocol["datasets"])),
        **settings,
    )
    ci_low, ci_high = np.quantile(draws[:, 0], [0.025, 0.975])
    expected_fits = int(protocol["planned_fit_counts"]["conditional_non_overlap_fits_if_triggered"])
    registry = {
        "analysis_id": analysis_id,
        "status": "RELEASED",
        "execution_status": "TRIGGERED_AND_COMPLETE",
        "estimand": "separate_equal_dataset_sensitivity_string_go_minus_dense",
        "estimate": float(dataset_table["estimate"].mean()),
        "uncertainty_interval_95_low": float(ci_low),
        "uncertainty_interval_95_high": float(ci_high),
        "interval_label": "95% conditional bootstrap uncertainty interval",
        "dataset_estimates": dataset_table.to_dict(orient="records"),
        "global_trigger_manifest_hash": expected_manifest_hash,
        "model_fits_expected": expected_fits,
        "model_fits_actual": int(len(filtered)),
        "bootstrap_replicates": settings["n_bootstrap"],
        "bootstrap_random_seed": settings["random_seed"],
        "bootstrap_index_hash": index_hash,
        "two_sided_centered_null_bootstrap_p": _null_bootstrap_p(
            null_draws[:, 0], float(dataset_table["estimate"].mean())
        ),
        "null_bootstrap_index_hash": null_index_hash,
        "input_results_manifest_hash": _results_manifest_hash(filtered, metric),
        "preflight_binding_manifest_hash": binding_hash,
    }
    if registry["model_fits_actual"] != expected_fits:
        return _withheld_result(
            analysis_id,
            [
                _failure(
                    analysis_id,
                    "CONDITIONAL_FIT_COUNT_MISMATCH",
                    detail=f"expected={expected_fits}; observed={len(filtered)}",
                )
            ],
        )
    return AnalysisReleaseResult(
        registry=registry,
        detail_tables={
            "condition_contrasts": condition_table,
            "dataset_estimates": dataset_table,
        },
        failures=_failure_frame([]),
    )


def mixed_support_sensitivity_release(
    frame: pd.DataFrame,
    protocol: Mapping[str, Any],
    *,
    metric: str = "pearson_r",
) -> AnalysisReleaseResult:
    """Release the prespecified 200-HVG mixed STRING-GO-coexpression sensitivity."""

    analysis_id = "MIXED-SUPPORT-SENSITIVITY"
    filtered, failures = _strict_analysis_block(
        frame,
        protocol,
        analysis_id=analysis_id,
        panel="primary",
        hvgs=(200,),
        arms=("combined", "dense"),
        metric=metric,
    )
    if failures:
        return _withheld_result(analysis_id, failures)
    pivot = filtered.pivot(index=["dataset", "condition", "seed"], columns="arm", values=metric)
    seed_table = (pivot["combined"] - pivot["dense"]).rename("seed_paired_delta").reset_index()
    condition_table = (
        seed_table.groupby(["dataset", "condition"], sort=True)["seed_paired_delta"]
        .agg(delta="mean", paired_seed_delta_sd="std")
        .reset_index()
    )
    dataset_table = (
        condition_table.groupby("dataset", sort=True)["delta"]
        .agg(estimate="mean", n_conditions="size")
        .reset_index()
    )
    settings = bootstrap_settings_from_protocol(protocol)
    draws, index_hash = _condition_within_dataset_bootstrap(
        condition_table,
        value_columns=("delta",),
        datasets=tuple(sorted(protocol["datasets"])),
        **settings,
    )
    centered = condition_table.copy()
    centered["delta"] = centered["delta"] - centered.groupby("dataset")["delta"].transform("mean")
    null_draws, null_index_hash = _condition_within_dataset_bootstrap(
        centered,
        value_columns=("delta",),
        datasets=tuple(sorted(protocol["datasets"])),
        **settings,
    )
    interval_low, interval_high = np.quantile(draws[:, 0], [0.025, 0.975])
    expected_new_fits = _expected_condition_total(protocol) * len(_expected_seeds(protocol))
    actual_new_fits = int((filtered["arm"] == "combined").sum())
    if actual_new_fits != expected_new_fits:
        return _withheld_result(
            analysis_id,
            [
                _failure(
                    analysis_id,
                    "MIXED_SUPPORT_NEW_FIT_COUNT_MISMATCH",
                    detail=f"expected={expected_new_fits}; observed={actual_new_fits}",
                )
            ],
        )
    registry = {
        "analysis_id": analysis_id,
        "status": "RELEASED",
        "role": "prespecified_mixed_support_sensitivity_not_primary_topology_evidence",
        "metric": metric,
        "contrast": "combined_minus_dense",
        "estimand": (
            "equal_benchmark_dataset_mean_mixed_string_go_control_coexpression_minus_dense"
        ),
        "estimate": float(dataset_table["estimate"].mean()),
        "uncertainty_interval_95_low": float(interval_low),
        "uncertainty_interval_95_high": float(interval_high),
        "interval_label": "95% conditional bootstrap uncertainty interval",
        "dataset_estimates": dataset_table.to_dict(orient="records"),
        "mixed_support_new_fits_expected": expected_new_fits,
        "mixed_support_new_fits_actual": actual_new_fits,
        "dense_reference_evaluations_reused": int((filtered["arm"] == "dense").sum()),
        "total_method_evaluations": int(len(filtered)),
        "bootstrap_replicates": settings["n_bootstrap"],
        "bootstrap_random_seed": settings["random_seed"],
        "bootstrap_index_hash": index_hash,
        "two_sided_centered_null_bootstrap_p": _null_bootstrap_p(
            null_draws[:, 0], float(dataset_table["estimate"].mean())
        ),
        "null_bootstrap_index_hash": null_index_hash,
        "multiplicity_family": ["combined_minus_dense"],
        "multiplicity_adjustment": "not_applicable_single_sensitivity_test",
        "input_results_manifest_hash": _results_manifest_hash(filtered, metric),
    }
    return AnalysisReleaseResult(
        registry=registry,
        detail_tables={
            "condition_contrasts": condition_table,
            "dataset_estimates": dataset_table,
        },
        failures=_failure_frame([]),
    )


def primary_decision_rule(interval_low: float, interval_high: float) -> str:
    """Classify direction solely from the unrounded uncertainty interval relative to zero."""

    if not np.isfinite([interval_low, interval_high]).all():
        raise RevisionProtocolError("Decision inputs must be finite")
    if interval_low > interval_high:
        raise RevisionProtocolError("Decision interval is invalid")
    if interval_low > 0:
        return "DIRECTIONAL_POSITIVE"
    if interval_high < 0:
        return "DIRECTIONAL_NEGATIVE"
    return "INCONCLUSIVE"


def build_decision_release_registry(
    primary: AnalysisReleaseResult,
    scale: AnalysisReleaseResult,
    topology: AnalysisReleaseResult,
    propagation: AnalysisReleaseResult,
    coverage: AnalysisReleaseResult,
    conditional: AnalysisReleaseResult | None = None,
) -> dict[str, Any]:
    """Release one decision registry only if all three mandatory locks pass."""

    analyses = (primary, scale, topology, propagation, coverage)
    if conditional is not None:
        analyses = (*analyses, conditional)
    withheld = [result.registry["analysis_id"] for result in analyses if not result.released]
    primary_decision = (
        primary.registry.get("decision")
        if primary.released
        else "WITHHELD_INCOMPLETE_PRIMARY_ANALYSIS"
    )
    if withheld:
        return {
            "package_release_status": "WITHHELD",
            "withheld_analyses": withheld,
            "primary_decision": primary_decision,
        }
    registry = {
        "package_release_status": "RELEASED",
        "primary_decision": primary_decision,
        "mandatory_analysis_ids": [result.registry["analysis_id"] for result in analyses],
        "analysis_registry_hashes": {
            result.registry["analysis_id"]: canonical_sha256(result.registry) for result in analyses
        },
    }
    registry["decision_registry_hash"] = canonical_sha256(registry)
    return registry


def build_thirteen_result_lock_registry(
    *,
    primary: AnalysisReleaseResult,
    propagation: AnalysisReleaseResult,
    scale: AnalysisReleaseResult,
    topology: AnalysisReleaseResult,
    coverage: AnalysisReleaseResult,
    conditional: AnalysisReleaseResult,
    secondary: AnalysisReleaseResult,
    mixed_support: AnalysisReleaseResult,
    global_trigger_manifest: Mapping[str, Any],
    external_comparator_registry: Mapping[str, Any],
    measured_compute_registry: Mapping[str, Any],
    claim_consequence_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Build exactly the 13 frozen result locks and cascade all upstream gates."""

    coverage_pass = coverage.released

    def analysis_lock(
        lock_id: str,
        result: AnalysisReleaseResult,
        payload: Mapping[str, Any] | None = None,
        *,
        require_coverage: bool = True,
    ) -> dict[str, Any]:
        if (require_coverage and not coverage_pass) or not result.released:
            reasons = (
                ["MANDATORY_FIT_COVERAGE_WITHHELD"]
                if require_coverage and not coverage_pass
                else []
            )
            reasons.extend(result.failures.get("reason_code", pd.Series(dtype=str)).astype(str))
            return _finalize_lock(
                {
                    "registry_id": lock_id,
                    "status": "WITHHELD",
                    "reason_codes": sorted(set(reasons)) or ["UPSTREAM_ANALYSIS_WITHHELD"],
                }
            )
        return _finalize_lock(
            {
                "registry_id": lock_id,
                "status": "RELEASED",
                "analysis_registry_hash": result.registry.get(
                    "analysis_registry_hash", canonical_sha256(result.registry)
                ),
                "source_table_manifest": result.registry.get("detail_table_manifest", []),
                "source_table_record_hashes": {
                    name: canonical_sha256(table.fillna("NOT_APPLICABLE").to_dict(orient="records"))
                    for name, table in sorted(result.detail_tables.items())
                },
                **dict(payload if payload is not None else result.registry),
            }
        )

    primary_registry = primary.registry
    primary_direction = primary_registry.get("decision")
    topology_direction = topology.registry.get("directional_support")
    topology_diagnostics = topology.registry.get("diagnostics_status")
    consequence_payload = dict(claim_consequence_registry)
    consequence_hash = consequence_payload.pop("registry_hash", None)
    consequence_valid = (
        claim_consequence_registry.get("registry_id") == "CLAIM-CONSEQUENCE-GATE"
        and claim_consequence_registry.get("status") == "RELEASED"
        and consequence_hash == canonical_sha256(consequence_payload)
        and isinstance(claim_consequence_registry.get("baseline_absolute_skill_eligible"), bool)
        and isinstance(claim_consequence_registry.get("ranking_consequence_eligible"), bool)
    )
    consequence_supportive = (
        consequence_valid
        and bool(claim_consequence_registry["baseline_absolute_skill_eligible"])
        and bool(claim_consequence_registry["ranking_consequence_eligible"])
    )
    if (
        primary_direction == "DIRECTIONAL_POSITIVE"
        and topology_direction == "TOPOLOGY_DIRECTIONALLY_SUPPORTIVE"
        and topology_diagnostics == "TOPOLOGY_DIAGNOSTICS_PASS"
        and consequence_supportive
    ):
        biological_claim_scope = "BIOLOGICAL_EDGE_WORDING_ALLOWED"
    elif primary_direction == "DIRECTIONAL_POSITIVE" and consequence_supportive:
        biological_claim_scope = "SPARSE_SUPPORT_WORDING_ALLOWED"
    elif primary_direction == "DIRECTIONAL_POSITIVE":
        biological_claim_scope = "POSITIVE_EFFECT_WITHOUT_INTERPRETABLE_BENEFIT"
    elif primary_direction == "DIRECTIONAL_NEGATIVE":
        biological_claim_scope = "NEGATIVE_DIRECTIONAL_EFFECT"
    else:
        biological_claim_scope = "BENEFIT_NOT_DEMONSTRATED"
    locks: dict[str, dict[str, Any]] = {}
    locks["PRIMARY-DELTA-R"] = analysis_lock(
        "PRIMARY-DELTA-R",
        primary,
        {
            "metric": primary_registry.get("metric"),
            "estimand": primary_registry.get("estimand"),
            "estimate": primary_registry.get("estimate"),
            "dataset_estimates": primary_registry.get("dataset_estimates"),
            "equal_cell_line_sensitivity": primary_registry.get("equal_cell_line_sensitivity"),
            "dataset_composition": primary_registry.get("dataset_composition"),
            "resampling_unit": primary_registry.get("resampling_unit"),
        },
    )
    locks["PRIMARY-UNCERTAINTY-INTERVAL"] = analysis_lock(
        "PRIMARY-UNCERTAINTY-INTERVAL",
        primary,
        {
            "uncertainty_interval_95_low": primary_registry.get("uncertainty_interval_95_low"),
            "uncertainty_interval_95_high": primary_registry.get("uncertainty_interval_95_high"),
            "interval_label": "95% conditional bootstrap uncertainty interval",
            "nominal_coverage_claim": False,
            "bootstrap_replicates": primary_registry.get("bootstrap_replicates"),
            "bootstrap_random_seed": primary_registry.get("bootstrap_random_seed"),
            "bootstrap_index_hash": primary_registry.get("bootstrap_index_hash"),
        },
    )
    if primary_registry.get("shared_control_cell_bootstrap", {}).get("status") != "MEASURED":
        locks["PRIMARY-UNCERTAINTY-INTERVAL"] = _finalize_lock(
            {
                "registry_id": "PRIMARY-UNCERTAINTY-INTERVAL",
                "status": "WITHHELD",
                "reason_codes": ["SHARED_CONTROL_CELL_BOOTSTRAP_NOT_MEASURED"],
            }
        )
    locks["PRIMARY-DECISION"] = analysis_lock(
        "PRIMARY-DECISION",
        primary,
        {
            "decision": primary_registry.get("decision"),
            "directional_decision_input": (
                "unrounded_conditional_uncertainty_interval_vs_zero_only"
            ),
            "claim_scope_inputs": (
                "direction_plus_absolute_skill_plus_condition_ranking_plus_topology"
            ),
            "archived_optimisation_repeatability_reference": 0.010,
            "archived_reference_has_success_authority": False,
            "topology_directional_support": topology_direction,
            "topology_diagnostics_status": topology_diagnostics,
            "topology_registry_hash": canonical_sha256(topology.registry),
            "biological_claim_scope": biological_claim_scope,
            "claim_consequence_gate": dict(claim_consequence_registry),
            "claim_consequence_registry_hash": consequence_hash,
            "empirical_seed_resolution_used_as_threshold": False,
        },
    )
    if locks["PRIMARY-DECISION"].get("status") == "RELEASED" and not consequence_valid:
        locks["PRIMARY-DECISION"] = _finalize_lock(
            {
                "registry_id": "PRIMARY-DECISION",
                "status": "WITHHELD",
                "reason_codes": ["CLAIM_CONSEQUENCE_GATE_INVALID"],
            }
        )
    locks["PROPAGATION-CONTROL"] = analysis_lock("PROPAGATION-CONTROL", propagation)
    locks["SCALE-INTERACTION"] = analysis_lock("SCALE-INTERACTION", scale)
    locks["TOPOLOGY-NULL"] = analysis_lock("TOPOLOGY-NULL", topology)
    trigger_status = global_trigger_manifest.get("status") == "PASS"
    locks["REPRESENTATIVENESS-TRIGGER"] = _finalize_lock(
        {
            "registry_id": "REPRESENTATIVENESS-TRIGGER",
            "status": "RELEASED" if trigger_status else "WITHHELD",
            "execution_state": (
                "TRIGGERED" if global_trigger_manifest.get("global_triggered") else "NOT_TRIGGERED"
            ),
            "global_triggered": global_trigger_manifest.get("global_triggered"),
            "dataset_triggers": global_trigger_manifest.get("dataset_triggers"),
            "thresholds": global_trigger_manifest.get("thresholds"),
            "global_trigger_manifest_hash": global_trigger_manifest.get("manifest_hash"),
        }
    )
    locks["CONDITIONAL-PANEL"] = analysis_lock(
        "CONDITIONAL-PANEL", conditional, require_coverage=False
    )
    fraction_table = primary.detail_tables.get("condition_contrasts", pd.DataFrame()).copy()
    fraction_coordinates: list[dict[str, Any]] = []
    if {"dataset", "condition", "delta"} <= set(fraction_table.columns):
        fraction_table = fraction_table.sort_values(
            ["dataset", "condition"], kind="mergesort"
        ).reset_index(drop=True)
        for dataset, group in fraction_table.groupby("dataset", sort=True):
            improved = int((group["delta"] > 0).sum())
            fraction_coordinates.append(
                {
                    "dataset": str(dataset),
                    "numerator_improved": improved,
                    "denominator_guide_conditions": int(len(group)),
                    "fraction_improved": float(improved / len(group)),
                    "source_table_id": "condition_contrasts",
                    "source_row_first_1_based": int(group.index.min() + 2),
                    "source_row_last_1_based": int(group.index.max() + 2),
                }
            )
    locks["FRACTION-IMPROVED"] = analysis_lock(
        "FRACTION-IMPROVED",
        primary,
        {
            "equal_dataset_mean_fraction_conditions_improved": primary_registry.get(
                "equal_dataset_mean_fraction_conditions_improved"
            ),
            "uncertainty_interval_95_low": primary_registry.get(
                "fraction_improved_uncertainty_interval_95_low"
            ),
            "uncertainty_interval_95_high": primary_registry.get(
                "fraction_improved_uncertainty_interval_95_high"
            ),
            "interval_label": "95% conditional bootstrap uncertainty interval",
            "unit": "guide_condition",
            "dataset_numerators_denominators_and_source_coordinates": fraction_coordinates,
            "condition_contrasts_records_hash": canonical_sha256(
                fraction_table.fillna("NOT_APPLICABLE").to_dict(orient="records")
            ),
            "primary_input_results_manifest_hash": primary_registry.get(
                "input_results_manifest_hash"
            ),
        },
    )
    if primary.released and not fraction_coordinates:
        locks["FRACTION-IMPROVED"] = _finalize_lock(
            {
                "registry_id": "FRACTION-IMPROVED",
                "status": "WITHHELD",
                "reason_codes": ["CONDITION_CONTRAST_SOURCE_TABLE_MISSING"],
            }
        )
    locks["EMPIRICAL-SEED-RESOLUTION"] = analysis_lock(
        "EMPIRICAL-SEED-RESOLUTION",
        primary,
        {
            "empirical_seed_resolution": primary_registry.get("empirical_seed_resolution"),
            "role": "parallel_descriptive_resolution_audit_not_a_decision_threshold",
        },
    )
    if mixed_support.released:
        secondary_payload = dict(secondary.registry)
        secondary_payload["mixed_support_sensitivity"] = dict(mixed_support.registry)
        secondary_payload["mixed_support_source_table_record_hashes"] = {
            name: canonical_sha256(table.fillna("NOT_APPLICABLE").to_dict(orient="records"))
            for name, table in sorted(mixed_support.detail_tables.items())
        }
        locks["SECONDARY-METRICS"] = analysis_lock(
            "SECONDARY-METRICS", secondary, secondary_payload
        )
    else:
        locks["SECONDARY-METRICS"] = _finalize_lock(
            {
                "registry_id": "SECONDARY-METRICS",
                "status": "WITHHELD",
                "reason_codes": ["MIXED_SUPPORT_SENSITIVITY_WITHHELD"],
            }
        )
    locks["EXTERNAL-COMPARATOR-VALIDATION"] = _external_lock(
        "EXTERNAL-COMPARATOR-VALIDATION", external_comparator_registry
    )
    locks["MEASURED-COMPUTE"] = _external_lock("MEASURED-COMPUTE", measured_compute_registry)
    if tuple(locks) != RESULT_LOCK_IDS:
        raise RevisionProtocolError("The final result-lock registry must contain exactly 13 IDs")
    package_released = coverage_pass and all(
        lock["status"] == "RELEASED" for lock in locks.values()
    )
    registry: dict[str, Any] = {
        "schema_version": "1.0",
        "package_release_status": "RELEASED" if package_released else "WITHHELD",
        "result_lock_count": len(locks),
        "result_lock_ids": list(RESULT_LOCK_IDS),
        "mandatory_fit_coverage_gate": coverage.registry,
        "result_locks": locks,
        "result_lock_hashes": {lock_id: lock["registry_hash"] for lock_id, lock in locks.items()},
        "withheld_result_locks": [
            lock_id for lock_id, lock in locks.items() if lock["status"] != "RELEASED"
        ],
    }
    registry["decision_registry_hash"] = canonical_sha256(registry)
    return registry


def _external_lock(lock_id: str, registry: Mapping[str, Any]) -> dict[str, Any]:
    registry_payload = dict(registry)
    registry_hash = registry_payload.pop("registry_hash", None)
    hash_valid = _is_sha256(registry_hash) and registry_hash == canonical_sha256(registry_payload)
    if (
        registry.get("registry_id") != lock_id
        or registry.get("status") != "RELEASED"
        or not hash_valid
    ):
        return _finalize_lock(
            {
                "registry_id": lock_id,
                "status": "WITHHELD",
                "reason_codes": ["EXTERNAL_REGISTRY_NOT_RELEASED"],
                "external_registry": dict(registry),
            }
        )
    return _finalize_lock(
        {
            "registry_id": lock_id,
            "status": "RELEASED",
            "external_registry": dict(registry),
        }
    )


def _finalize_lock(payload: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(payload)
    output["registry_hash"] = canonical_sha256(output)
    return output


def condition_ranking_overlap(
    frame: pd.DataFrame,
    protocol: Mapping[str, Any],
    *,
    metric: str = "pearson_r",
    top_k: int = 10,
) -> pd.DataFrame:
    """Compare deterministic top-condition rankings for STRING-GO and dense per dataset."""

    filtered, failures = _strict_analysis_block(
        frame,
        protocol,
        analysis_id="CONDITION-RANKING",
        panel="primary",
        hvgs=(200,),
        arms=("string_go", "dense"),
        metric=metric,
    )
    if failures:
        raise RevisionProtocolError(_failure_frame(failures).to_json(orient="records"))
    averaged = (
        filtered.groupby(["dataset", "condition", "arm"], sort=True)[metric].mean().reset_index()
    )
    rows: list[dict[str, Any]] = []
    from scipy.stats import spearmanr

    for dataset, group in averaged.groupby("dataset", sort=True):
        rankings: dict[str, list[str]] = {}
        for arm in ("string_go", "dense"):
            arm_rows = group[group["arm"] == arm]
            rankings[arm] = [
                condition
                for condition, _ in sorted(
                    zip(arm_rows["condition"], arm_rows[metric], strict=True),
                    key=lambda item: (-float(item[1]), str(item[0])),
                )[:top_k]
            ]
        left = set(rankings["string_go"])
        right = set(rankings["dense"])
        shared = sorted(left & right)
        if len(shared) >= 2:
            left_rank = {condition: index for index, condition in enumerate(rankings["string_go"])}
            right_rank = {condition: index for index, condition in enumerate(rankings["dense"])}
            rank_spearman = float(
                spearmanr(
                    [left_rank[condition] for condition in shared],
                    [right_rank[condition] for condition in shared],
                ).statistic
            )
            rank_status = "MEASURED" if np.isfinite(rank_spearman) else "UNDEFINED_TIES"
            if not np.isfinite(rank_spearman):
                rank_spearman = None
        else:
            rank_spearman = None
            rank_status = "UNDEFINED_FEWER_THAN_TWO_SHARED_CONDITIONS"
        rows.append(
            {
                "dataset": dataset,
                "top_k": top_k,
                "string_go_ranking": "|".join(rankings["string_go"]),
                "dense_ranking": "|".join(rankings["dense"]),
                "shared_conditions": "|".join(shared),
                "jaccard": float(len(shared) / len(left | right)),
                "shared_rank_spearman": rank_spearman,
                "shared_rank_spearman_status": rank_status,
            }
        )
    return pd.DataFrame(rows)


def secondary_metric_release(
    frame: pd.DataFrame, protocol: Mapping[str, Any]
) -> AnalysisReleaseResult:
    """Release benefit-oriented secondary metrics without reusing the primary decision rule."""

    analysis_id = "SECONDARY-METRICS"
    metric_directions = {
        "fisher_z_pearson": ("string_go", "dense"),
        "spearman_r": ("string_go", "dense"),
        "mse": ("dense", "string_go"),
        "top20_absolute_delta_jaccard": ("string_go", "dense"),
    }
    failures: list[dict[str, Any]] = []
    condition_tables: list[pd.DataFrame] = []
    filtered_by_metric: dict[str, pd.DataFrame] = {}
    for metric, (benefit_arm, reference_arm) in metric_directions.items():
        filtered, metric_failures = _strict_analysis_block(
            frame,
            protocol,
            analysis_id=analysis_id,
            panel="primary",
            hvgs=(200,),
            arms=("string_go", "dense"),
            metric=metric,
        )
        failures.extend(metric_failures)
        if metric_failures:
            continue
        filtered_by_metric[metric] = filtered
        pivot = filtered.pivot(index=["dataset", "condition", "seed"], columns="arm", values=metric)
        condition = (
            (pivot[benefit_arm] - pivot[reference_arm])
            .groupby(level=["dataset", "condition"])
            .mean()
            .rename("benefit_delta")
            .reset_index()
            .assign(metric=metric, benefit_arm=benefit_arm, reference_arm=reference_arm)
        )
        condition_tables.append(condition)
    if failures:
        return _withheld_result(analysis_id, failures)
    long_table = pd.concat(condition_tables, ignore_index=True)
    wide = long_table.pivot(
        index=["dataset", "condition"], columns="metric", values="benefit_delta"
    ).reset_index()
    metrics = tuple(metric_directions)
    settings = bootstrap_settings_from_protocol(protocol)
    draws, index_hash = _condition_within_dataset_bootstrap(
        wide,
        value_columns=metrics,
        datasets=tuple(sorted(protocol["datasets"])),
        **settings,
    )
    centered = wide.copy()
    for metric in metrics:
        centered[metric] = centered[metric] - centered.groupby("dataset")[metric].transform("mean")
    null_draws, null_index_hash = _condition_within_dataset_bootstrap(
        centered,
        value_columns=metrics,
        datasets=tuple(sorted(protocol["datasets"])),
        **settings,
    )
    estimates = np.asarray(
        [float(wide.groupby("dataset")[metric].mean().mean()) for metric in metrics]
    )
    p_values = np.asarray(
        [_null_bootstrap_p(null_draws[:, index], estimates[index]) for index in range(len(metrics))]
    )
    q_values = benjamini_hochberg(p_values)
    metric_registry = []
    for index, metric in enumerate(metrics):
        ci_low, ci_high = np.quantile(draws[:, index], [0.025, 0.975])
        metric_registry.append(
            {
                "metric": metric,
                "benefit_direction": (
                    f"{metric_directions[metric][0]}_minus_{metric_directions[metric][1]}"
                ),
                "estimate": float(estimates[index]),
                "uncertainty_interval_95_low": float(ci_low),
                "uncertainty_interval_95_high": float(ci_high),
                "interval_label": "95% conditional bootstrap uncertainty interval",
                "centered_null_bootstrap_p": float(p_values[index]),
                "bh_q_four_metric_family": float(q_values[index]),
                "dataset_estimates": [
                    {
                        "dataset": str(dataset),
                        "estimate": float(values.mean()),
                        "n_guide_conditions": int(len(values)),
                    }
                    for dataset, values in wide.groupby("dataset", sort=True)[metric]
                ],
                "decision": "DESCRIPTIVE_SECONDARY_NO_PRIMARY_DECISION",
            }
        )
    target_sensitivity, target_failures = _target_level_primary_sensitivity(frame, protocol)
    if target_failures:
        return _withheld_result(analysis_id, target_failures)
    registry = {
        "analysis_id": analysis_id,
        "status": "RELEASED",
        "metrics": metric_registry,
        "multiplicity_family": list(metrics),
        "multiplicity_method": "benjamini_hochberg",
        "bootstrap_replicates": settings["n_bootstrap"],
        "bootstrap_random_seed": settings["random_seed"],
        "bootstrap_index_hash": index_hash,
        "null_bootstrap_index_hash": null_index_hash,
        "input_results_manifest_hashes": {
            metric: _results_manifest_hash(filtered_by_metric[metric], metric) for metric in metrics
        },
        "target_level_primary_sensitivity": target_sensitivity["registry"],
    }
    return AnalysisReleaseResult(
        registry=registry,
        detail_tables={
            "condition_benefit_deltas": long_table,
            "target_level_primary_sensitivity": target_sensitivity["target_table"],
            "target_level_dataset_estimates": target_sensitivity["dataset_table"],
        },
        failures=_failure_frame([]),
    )


def _target_level_primary_sensitivity(
    frame: pd.DataFrame, protocol: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Aggregate guide conditions within canonical targets before target-level resampling."""

    analysis_id = "SECONDARY-METRICS"
    filtered, failures = _strict_analysis_block(
        frame,
        protocol,
        analysis_id=analysis_id,
        panel="primary",
        hvgs=(200,),
        arms=("string_go", "dense"),
        metric="pearson_r",
    )
    if failures:
        return {}, failures
    target_values = filtered["canonical_target_set"].astype(str)
    if target_values.str.strip().eq("").any() or target_values.eq("None").any():
        return {}, [_failure(analysis_id, "CANONICAL_TARGET_SET_MISSING")]
    pivot = filtered.pivot(
        index=["dataset", "condition", "canonical_target_set", "seed"],
        columns="arm",
        values="pearson_r",
    )
    guide_table = (
        (pivot["string_go"] - pivot["dense"])
        .groupby(level=["dataset", "condition", "canonical_target_set"])
        .mean()
        .rename("guide_condition_delta")
        .reset_index()
    )
    target_table = (
        guide_table.groupby(["dataset", "canonical_target_set"], sort=True)
        .agg(
            delta=("guide_condition_delta", "mean"),
            guide_condition_count=("condition", "size"),
            guide_condition_ids=("condition", lambda values: "|".join(sorted(values))),
        )
        .reset_index()
    )
    if any(len(group) < 2 for _, group in target_table.groupby("dataset", sort=True)):
        return {}, [_failure(analysis_id, "INSUFFICIENT_CANONICAL_TARGETS_FOR_SENSITIVITY")]
    dataset_table = (
        target_table.groupby("dataset", sort=True)["delta"]
        .agg(estimate="mean", n_canonical_target_sets="size")
        .reset_index()
    )
    settings = bootstrap_settings_from_protocol(protocol)
    draws, index_hash = _condition_within_dataset_bootstrap(
        target_table,
        value_columns=("delta",),
        datasets=tuple(sorted(protocol["datasets"])),
        **settings,
    )
    ci_low, ci_high = np.quantile(draws[:, 0], [0.025, 0.975])
    registry = {
        "status": "RELEASED",
        "estimand": (
            "equal_weight_dataset_mean_of_canonical_target_set_mean_guide_condition_"
            "seed_averaged_string_go_minus_dense"
        ),
        "estimate": float(dataset_table["estimate"].mean()),
        "uncertainty_interval_95_low": float(ci_low),
        "uncertainty_interval_95_high": float(ci_high),
        "interval_label": "95% conditional bootstrap uncertainty interval",
        "resampling_unit": "canonical_target_set_within_dataset",
        "guide_aggregation": "equal_weight_guide_conditions_within_canonical_target_set",
        "decision": "DESCRIPTIVE_SENSITIVITY_NO_PRIMARY_DECISION",
        "bootstrap_replicates": settings["n_bootstrap"],
        "bootstrap_random_seed": settings["random_seed"],
        "bootstrap_index_hash": index_hash,
        "dataset_estimates": dataset_table.to_dict(orient="records"),
        "canonical_target_sets_represented": int(len(target_table)),
        "guide_conditions_represented": int(len(guide_table)),
        "guide_multiplicity_distribution": {
            "minimum": int(target_table["guide_condition_count"].min()),
            "median": float(target_table["guide_condition_count"].median()),
            "maximum": int(target_table["guide_condition_count"].max()),
            "counts": [
                {
                    "guide_condition_count": int(count),
                    "canonical_target_sets": int(frequency),
                }
                for count, frequency in target_table["guide_condition_count"]
                .value_counts()
                .sort_index()
                .items()
            ],
        },
        "complete_seed_support": "PASS",
        "seed_ids": list(_expected_seeds(protocol)),
    }
    return {
        "registry": registry,
        "target_table": target_table,
        "dataset_table": dataset_table,
    }, []


def descriptive_paired_tests(condition_contrasts: pd.DataFrame) -> dict[str, Any]:
    """Return explicitly descriptive Wilcoxon and exact sign tests."""

    if "delta" not in condition_contrasts:
        raise RevisionProtocolError("condition_contrasts requires a delta column")
    values = condition_contrasts["delta"].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise RevisionProtocolError("Descriptive paired-test values must be finite and non-empty")
    from scipy.stats import binomtest, wilcoxon

    nonzero = values[values != 0]
    if len(nonzero):
        wilcoxon_result = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
        wilcoxon_statistic = float(wilcoxon_result.statistic)
        wilcoxon_p = float(wilcoxon_result.pvalue)
        sign_p = float(binomtest(int(np.sum(nonzero > 0)), len(nonzero), 0.5).pvalue)
        status = "DESCRIPTIVE_ONLY"
    else:
        wilcoxon_statistic = None
        wilcoxon_p = None
        sign_p = None
        status = "UNDEFINED_ALL_ZERO_DIFFERENCES"
    return {
        "status": status,
        "n_total": int(len(values)),
        "n_nonzero": int(len(nonzero)),
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_two_sided_p": wilcoxon_p,
        "exact_sign_two_sided_p": sign_p,
    }


def _preflight_binding_failures(
    frame: pd.DataFrame,
    protocol: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]] | None,
    *,
    analysis_id: str,
    panel: str,
    protocol_file_hash: str | None,
    expected_code_hash: str | None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Bind every released row to an executable preflight and exact panel contract."""

    failures: list[dict[str, Any]] = []
    if not summaries:
        return [_failure(analysis_id, "PREFLIGHT_RELEASE_CONTRACTS_MISSING")], None
    if not _is_sha256(protocol_file_hash) or not _is_sha256(expected_code_hash):
        return [_failure(analysis_id, "CURRENT_RELEASE_CODE_OR_PROTOCOL_HASH_MISSING")], None
    expected_datasets = set(str(value) for value in protocol["datasets"])
    expected_panel_size = int(protocol["planned_fit_counts"]["conditions_per_panel"])
    contracts: list[dict[str, Any]] = []
    panel_hashes_by_dataset: dict[str, set[str]] = {}
    conditions_by_dataset: dict[str, tuple[str, ...]] = {}
    target_maps_by_dataset: dict[str, dict[str, str]] = {}
    for summary in summaries:
        if not isinstance(summary, Mapping):
            failures.append(_failure(analysis_id, "PREFLIGHT_SUMMARY_INVALID"))
            continue
        payload = dict(summary)
        summary_hash = payload.pop("summary_hash", None)
        dataset = str(summary.get("dataset", ""))
        if not _is_sha256(summary_hash) or summary_hash != canonical_sha256(payload):
            failures.append(
                _failure(
                    analysis_id,
                    "PREFLIGHT_SUMMARY_HASH_MISMATCH",
                    dataset=dataset or None,
                )
            )
            continue
        if dataset not in expected_datasets:
            failures.append(
                _failure(
                    analysis_id,
                    "PREFLIGHT_SUMMARY_DATASET_MISMATCH",
                    dataset=dataset or None,
                )
            )
            continue
        if summary.get("can_execute") is not True or summary.get("blockers") != []:
            failures.append(
                _failure(analysis_id, "PREFLIGHT_SUMMARY_NOT_EXECUTABLE", dataset=dataset)
            )
        if (
            summary.get("input_hashes", {}).get("protocol") != protocol_file_hash
            or summary.get("code_hash") != expected_code_hash
        ):
            failures.append(
                _failure(
                    analysis_id,
                    "PREFLIGHT_CURRENT_CODE_OR_PROTOCOL_MISMATCH",
                    dataset=dataset,
                )
            )
        if summary.get("panel") != panel:
            continue
        manifest = summary.get("condition_panel_manifest")
        if not isinstance(manifest, Mapping):
            failures.append(_failure(analysis_id, "PANEL_MANIFEST_MISSING", dataset=dataset))
            continue
        manifest_payload = dict(manifest)
        manifest_hash = manifest_payload.pop("manifest_hash", None)
        if not _is_sha256(manifest_hash) or manifest_hash != canonical_sha256(manifest_payload):
            failures.append(_failure(analysis_id, "PANEL_MANIFEST_HASH_MISMATCH", dataset=dataset))
            continue
        if (
            manifest.get("dataset") != dataset
            or manifest.get("protocol_id") != protocol["protocol_id"]
            or manifest.get("protocol_file_sha256") != protocol_file_hash
            or manifest.get("dataset_passport_sha256") == "NOT_APPLICABLE_FIXTURE"
            or manifest.get("control_binding", {}).get("binding_source")
            != "hash_verified_dataset_passport"
            or manifest.get("non_overlap_status") != "PASS"
            or manifest.get("primary_sensitivity_overlap") != []
        ):
            failures.append(
                _failure(analysis_id, "PANEL_MANIFEST_CONTRACT_MISMATCH", dataset=dataset)
            )
        panel_section = manifest.get(panel, {})
        manifest_conditions = tuple(
            str(value) for value in panel_section.get("ordered_canonical_condition_ids", [])
        )
        entries = panel_section.get("entries", [])
        manifest_target_map = {
            str(entry.get("condition")): "+".join(
                sorted(
                    (str(value) for value in entry.get("canonical_targets", [])),
                    key=str.casefold,
                )
            )
            for entry in entries
            if isinstance(entry, Mapping)
        }
        if (
            len(manifest_conditions) != expected_panel_size
            or len(set(manifest_conditions)) != expected_panel_size
            or tuple(str(value) for value in summary.get("selected_conditions", []))
            != manifest_conditions
            or int(summary.get("n_selected_conditions", -1)) != expected_panel_size
            or int(summary.get("requested_panel_size", -1)) != expected_panel_size
            or set(manifest_target_map) != set(manifest_conditions)
            or any(not value for value in manifest_target_map.values())
        ):
            failures.append(
                _failure(analysis_id, "PANEL_MANIFEST_CONDITION_SET_MISMATCH", dataset=dataset)
            )
        input_hashes = summary.get("input_hashes", {})
        binding_fields = {
            "condition_panel_manifest_hash": "condition_panel_manifest",
            "target_encoding_audit_hash": "target_encoding_audit",
            "dataset_passport_hash": "dataset_passport",
            "environment_manifest_hash": "environment_manifest",
            "environment_lock_hash": "environment_lock",
            "git_provenance_hash": "git_provenance",
            "shared_control_cell_evidence_hash": "shared_control_cell_evidence",
        }
        if (
            summary.get("condition_panel_manifest_hash") != manifest_hash
            or input_hashes.get("condition_panel_manifest") != manifest_hash
        ):
            failures.append(
                _failure(analysis_id, "PANEL_MANIFEST_BINDING_MISMATCH", dataset=dataset)
            )
        if any(not _is_sha256(input_hashes.get(name)) for name in binding_fields.values()):
            failures.append(
                _failure(analysis_id, "PREFLIGHT_INPUT_BINDING_MISSING", dataset=dataset)
            )
        environment_manifest = summary.get("environment_manifest")
        if isinstance(environment_manifest, Mapping):
            environment_payload = dict(environment_manifest)
            environment_hash = environment_payload.pop("manifest_hash", None)
            if (
                environment_hash != canonical_sha256(environment_payload)
                or environment_hash != input_hashes.get("environment_manifest")
                or environment_manifest.get("environment_lock_sha256")
                != input_hashes.get("environment_lock")
                or environment_manifest.get("fixture_status") != "PRODUCTION"
            ):
                failures.append(
                    _failure(
                        analysis_id,
                        "ENVIRONMENT_MANIFEST_BINDING_MISMATCH",
                        dataset=dataset,
                    )
                )
        else:
            failures.append(_failure(analysis_id, "ENVIRONMENT_MANIFEST_MISSING", dataset=dataset))
        git_provenance = summary.get("git_provenance")
        if isinstance(git_provenance, Mapping):
            git_payload = dict(git_provenance)
            git_hash = git_payload.pop("provenance_hash", None)
            if (
                git_hash != canonical_sha256(git_payload)
                or git_hash != input_hashes.get("git_provenance")
                or not re.fullmatch(r"[0-9a-f]{40}", str(git_provenance.get("git_commit", "")))
                or git_provenance.get("git_worktree_status") not in {"CLEAN", "DIRTY"}
                or not _is_sha256(git_provenance.get("git_status_hash"))
            ):
                failures.append(
                    _failure(analysis_id, "GIT_PROVENANCE_BINDING_MISMATCH", dataset=dataset)
                )
        else:
            failures.append(_failure(analysis_id, "GIT_PROVENANCE_MISSING", dataset=dataset))
        panel_hashes_by_dataset.setdefault(dataset, set()).add(str(manifest_hash))
        previous_conditions = conditions_by_dataset.setdefault(dataset, manifest_conditions)
        if previous_conditions != manifest_conditions:
            failures.append(
                _failure(analysis_id, "PANEL_MANIFEST_CHANGED_ACROSS_RUNS", dataset=dataset)
            )
        previous_target_map = target_maps_by_dataset.setdefault(dataset, manifest_target_map)
        if previous_target_map != manifest_target_map:
            failures.append(
                _failure(
                    analysis_id,
                    "PANEL_TARGET_MAPPING_CHANGED_ACROSS_RUNS",
                    dataset=dataset,
                )
            )
        contracts.append(
            {
                "dataset": dataset,
                "hvg": int(summary.get("hvg", -1)),
                "panel": panel,
                "analysis_block": str(summary.get("analysis_block", "")),
                "requested_arms": set(str(value) for value in summary.get("requested_arms", [])),
                "summary_hash": summary_hash,
                "condition_panel_manifest_hash": manifest_hash,
                "target_encoding_audit_hash": input_hashes.get("target_encoding_audit"),
                "dataset_passport_hash": input_hashes.get("dataset_passport"),
                "protocol_file_hash": input_hashes.get("protocol"),
                "environment_manifest_hash": input_hashes.get("environment_manifest"),
                "environment_lock_hash": input_hashes.get("environment_lock"),
                "git_provenance_hash": input_hashes.get("git_provenance"),
                "shared_control_cell_evidence_hash": input_hashes.get(
                    "shared_control_cell_evidence"
                ),
                "git_commit": (
                    git_provenance.get("git_commit")
                    if isinstance(git_provenance, Mapping)
                    else None
                ),
                "git_worktree_status": (
                    git_provenance.get("git_worktree_status")
                    if isinstance(git_provenance, Mapping)
                    else None
                ),
                "git_status_hash": (
                    git_provenance.get("git_status_hash")
                    if isinstance(git_provenance, Mapping)
                    else None
                ),
                "code_hash": summary.get("code_hash"),
            }
        )
    for dataset in expected_datasets:
        if len(panel_hashes_by_dataset.get(dataset, set())) != 1:
            failures.append(
                _failure(
                    analysis_id,
                    "PANEL_MANIFEST_DATASET_COVERAGE_MISMATCH",
                    dataset=dataset,
                )
            )
            continue
        observed_conditions = set(frame.loc[frame["dataset"] == dataset, "condition"].astype(str))
        expected_conditions = set(conditions_by_dataset.get(dataset, ()))
        if observed_conditions != expected_conditions:
            failures.append(
                _failure(
                    analysis_id,
                    "ARTIFACT_CONDITION_SET_DIFFERS_FROM_PANEL_MANIFEST",
                    dataset=dataset,
                    detail=(
                        f"missing={sorted(expected_conditions - observed_conditions)[:5]}; "
                        f"extra={sorted(observed_conditions - expected_conditions)[:5]}"
                    ),
                )
            )
        target_rows = frame.loc[
            frame["dataset"] == dataset, ["condition", "canonical_target_set"]
        ].drop_duplicates()
        expected_target_map = target_maps_by_dataset.get(dataset, {})
        for row in target_rows.itertuples(index=False):
            if expected_target_map.get(str(row.condition)) != str(row.canonical_target_set):
                failures.append(
                    _failure(
                        analysis_id,
                        "ARTIFACT_TARGET_SET_DIFFERS_FROM_PANEL_MANIFEST",
                        dataset=dataset,
                        condition=str(row.condition),
                    )
                )
    binding_columns = {
        "condition_panel_manifest_hash",
        "target_encoding_audit_hash",
        "preflight_summary_hash",
        "dataset_passport_hash",
        "canonical_target_set",
        "dataset_input_hash",
        "protocol_file_hash",
        "environment_manifest_hash",
        "environment_lock_hash",
        "git_commit",
        "git_worktree_status",
        "git_status_hash",
        "git_provenance_hash",
        "analysis_block",
        "shared_control_cell_evidence_hash",
    }
    if not binding_columns <= set(frame.columns):
        failures.append(
            _failure(
                analysis_id,
                "PREFLIGHT_ARTIFACT_BINDING_COLUMNS_MISSING",
                detail=str(sorted(binding_columns - set(frame.columns))),
            )
        )
        return failures, canonical_sha256(sorted(c["summary_hash"] for c in contracts))
    for row in (
        frame[
            [
                "dataset",
                "hvg",
                "panel",
                "analysis_block",
                "arm",
                "condition_panel_manifest_hash",
                "target_encoding_audit_hash",
                "preflight_summary_hash",
                "dataset_passport_hash",
                "protocol_file_hash",
                "environment_manifest_hash",
                "environment_lock_hash",
                "git_commit",
                "git_worktree_status",
                "git_status_hash",
                "git_provenance_hash",
                "shared_control_cell_evidence_hash",
                "code_hash",
            ]
        ]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        candidates = [
            contract
            for contract in contracts
            if contract["dataset"] == str(row.dataset)
            and contract["hvg"] == int(row.hvg)
            and contract["panel"] == str(row.panel)
            and contract["analysis_block"] == str(row.analysis_block)
            and str(row.arm) in contract["requested_arms"]
            and contract["summary_hash"] == row.preflight_summary_hash
        ]
        if not candidates:
            failures.append(
                _failure(
                    analysis_id,
                    "ARTIFACT_NOT_BOUND_TO_SUBMITTED_PREFLIGHT",
                    dataset=str(row.dataset),
                    hvg=int(row.hvg),
                    arm=str(row.arm),
                )
            )
            continue
        if not any(
            row.condition_panel_manifest_hash == candidate["condition_panel_manifest_hash"]
            and row.target_encoding_audit_hash == candidate["target_encoding_audit_hash"]
            and row.dataset_passport_hash == candidate["dataset_passport_hash"]
            and row.protocol_file_hash == candidate["protocol_file_hash"]
            and row.environment_manifest_hash == candidate["environment_manifest_hash"]
            and row.environment_lock_hash == candidate["environment_lock_hash"]
            and row.git_provenance_hash == candidate["git_provenance_hash"]
            and row.shared_control_cell_evidence_hash
            == candidate["shared_control_cell_evidence_hash"]
            and row.git_commit == candidate["git_commit"]
            and row.git_worktree_status == candidate["git_worktree_status"]
            and row.git_status_hash == candidate["git_status_hash"]
            and row.code_hash == candidate["code_hash"]
            for candidate in candidates
        ):
            failures.append(
                _failure(
                    analysis_id,
                    "ARTIFACT_PREFLIGHT_INPUT_BINDING_MISMATCH",
                    dataset=str(row.dataset),
                    hvg=int(row.hvg),
                    arm=str(row.arm),
                )
            )
    if set(frame["protocol_file_hash"].dropna()) != {protocol_file_hash}:
        failures.append(_failure(analysis_id, "ARTIFACT_CURRENT_PROTOCOL_HASH_MISMATCH"))
    if set(frame["code_hash"].dropna()) != {expected_code_hash}:
        failures.append(_failure(analysis_id, "ARTIFACT_CURRENT_CODE_HASH_MISMATCH"))
    if frame["environment_lock_hash"].nunique(dropna=False) != 1:
        failures.append(_failure(analysis_id, "ENVIRONMENT_LOCK_CHANGED_ACROSS_ARTIFACTS"))
    binding_hash = canonical_sha256(sorted(contract["summary_hash"] for contract in contracts))
    return failures, binding_hash


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _strict_analysis_block(
    frame: pd.DataFrame,
    protocol: Mapping[str, Any],
    *,
    analysis_id: str,
    panel: str,
    hvgs: Sequence[int],
    arms: Sequence[str],
    metric: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    provenance_columns = {
        "artifact_hash",
        "architecture_hash",
        "initialization_hash",
        "preprocessing_state_hash",
        "input_hashes_hash",
        "matched_input_hashes_hash",
        "split_hash",
        "graph_support_hash",
        "code_hash",
        "y_true_hash",
        "gene_names_hash",
        "vector_space",
        "condition_panel_manifest_hash",
        "target_encoding_audit_hash",
        "preflight_summary_hash",
        "dataset_passport_hash",
        "canonical_target_set",
        "dataset_input_hash",
        "protocol_file_hash",
        "environment_manifest_hash",
        "environment_lock_hash",
        "git_commit",
        "git_worktree_status",
        "git_status_hash",
        "git_provenance_hash",
        "shared_control_cell_evidence_hash",
    }
    required_columns = {
        "dataset",
        "hvg",
        "panel",
        "arm",
        "condition",
        "seed",
        "analysis_block",
        "epochs_requested",
        metric,
        *provenance_columns,
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        return frame.iloc[0:0].copy(), [
            _failure(analysis_id, "MISSING_RESULT_COLUMNS", detail=str(missing_columns))
        ]
    datasets = tuple(sorted(protocol["datasets"]))
    expected_seeds = _expected_seeds(protocol)
    expected_conditions = int(protocol["planned_fit_counts"]["conditions_per_panel"])
    filtered = frame[
        frame["dataset"].isin(datasets)
        & (frame["panel"] == panel)
        & frame["hvg"].isin(hvgs)
        & frame["arm"].isin(arms)
    ].copy()
    failures: list[dict[str, Any]] = []
    if filtered.empty:
        failures.append(_failure(analysis_id, "EMPTY_ANALYSIS_BLOCK"))
        return filtered, failures
    values = pd.to_numeric(filtered[metric], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        failures.append(_failure(analysis_id, "NONFINITE_METRIC_VALUE"))
    duplicate_keys = ["dataset", "hvg", "panel", "arm", "condition", "seed"]
    duplicates = filtered.duplicated(duplicate_keys, keep=False)
    if duplicates.any():
        failures.append(
            _failure(
                analysis_id,
                "DUPLICATE_RESULT_KEY",
                detail=f"duplicate_rows={int(duplicates.sum())}",
            )
        )
    for dataset in datasets:
        dataset_rows = filtered[filtered["dataset"] == dataset]
        conditions = sorted(set(dataset_rows["condition"].astype(str)))
        if len(conditions) != expected_conditions:
            failures.append(
                _failure(
                    analysis_id,
                    "CONDITION_COUNT_MISMATCH",
                    dataset=dataset,
                    detail=f"expected={expected_conditions}; observed={len(conditions)}",
                )
            )
        for condition in conditions:
            for hvg in hvgs:
                for arm in arms:
                    rows = dataset_rows[
                        (dataset_rows["condition"].astype(str) == condition)
                        & (dataset_rows["hvg"] == hvg)
                        & (dataset_rows["arm"] == arm)
                    ]
                    observed_seeds = tuple(sorted(int(seed) for seed in rows["seed"]))
                    if observed_seeds != expected_seeds:
                        failures.append(
                            _failure(
                                analysis_id,
                                "INCOMPLETE_SEED_OR_SUPPORT_CELL",
                                dataset=dataset,
                                condition=condition,
                                hvg=hvg,
                                arm=arm,
                                detail=f"expected={expected_seeds}; observed={observed_seeds}",
                            )
                        )
    failures.extend(_matched_provenance_failures(filtered, analysis_id, hvgs, arms))
    requested_epochs = pd.to_numeric(filtered["epochs_requested"], errors="coerce")
    frozen_epochs = int(protocol["training"]["maximum_epochs"])
    if requested_epochs.isna().any() or not (requested_epochs == frozen_epochs).all():
        failures.append(
            _failure(
                analysis_id,
                "TRAINING_EPOCH_CONTRACT_MISMATCH",
                detail=f"required_epochs_requested={frozen_epochs}",
            )
        )
    failures.extend(_analysis_block_label_failures(filtered, analysis_id))
    return filtered, failures


def _analysis_block_label_failures(frame: pd.DataFrame, analysis_id: str) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in frame[["hvg", "arm", "analysis_block"]].drop_duplicates().itertuples(index=False):
        if analysis_id == "SCALE-INTERACTION":
            expected = (
                "topology_primary"
                if int(row.hvg) == 200 and str(row.arm) in {"string_go", "dense"}
                else "scale_extension"
            )
        elif analysis_id == "CONDITIONAL-NONOVERLAP":
            expected = "conditional_nonoverlap"
        elif str(row.arm) == "combined":
            expected = "mixed_support_sensitivity"
        else:
            expected = "topology_primary"
        if row.analysis_block != expected:
            failures.append(
                _failure(
                    analysis_id,
                    "ANALYSIS_BLOCK_LABEL_MISMATCH",
                    hvg=int(row.hvg),
                    arm=str(row.arm),
                    detail=f"expected={expected}; observed={row.analysis_block}",
                )
            )
    return failures


def _matched_provenance_failures(
    frame: pd.DataFrame, analysis_id: str, hvgs: Sequence[int], arms: Sequence[str]
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if frame["artifact_hash"].duplicated().any():
        failures.append(_failure(analysis_id, "DUPLICATE_ARTIFACT_HASH"))
    constant_within_dataset_hvg = (
        "architecture_hash",
        "preprocessing_state_hash",
        "matched_input_hashes_hash",
        "code_hash",
        "dataset_input_hash",
        "protocol_file_hash",
        "environment_manifest_hash",
        "environment_lock_hash",
        "git_commit",
        "git_worktree_status",
        "git_status_hash",
        "git_provenance_hash",
    )
    for (dataset, hvg), group in frame.groupby(["dataset", "hvg"], sort=True):
        for column in constant_within_dataset_hvg:
            if group[column].nunique(dropna=False) != 1:
                failures.append(
                    _failure(
                        analysis_id,
                        "MATCHED_ARM_PROVENANCE_MISMATCH",
                        dataset=str(dataset),
                        hvg=int(hvg),
                        detail=f"{column} differs across matched arms/folds",
                    )
                )
        for arm, arm_rows in group.groupby("arm", sort=True):
            if arm_rows["graph_support_hash"].nunique(dropna=False) != 1:
                failures.append(
                    _failure(
                        analysis_id,
                        "GRAPH_SUPPORT_HASH_NOT_CONSTANT_ACROSS_FOLDS",
                        dataset=str(dataset),
                        hvg=int(hvg),
                        arm=str(arm),
                    )
                )
        arm_graph_hashes = group.groupby("arm")["graph_support_hash"].first()
        if len(arm_graph_hashes) == len(arms) and arm_graph_hashes.nunique() != len(arms):
            failures.append(
                _failure(
                    analysis_id,
                    "COMPARATOR_GRAPH_SUPPORT_HASH_COLLISION",
                    dataset=str(dataset),
                    hvg=int(hvg),
                )
            )
    for (dataset, hvg, condition, seed), group in frame.groupby(
        ["dataset", "hvg", "condition", "seed"], sort=True
    ):
        observed_arms = set(group["arm"].astype(str))
        if not set(arms) <= observed_arms:
            continue
        for column in (
            "initialization_hash",
            "split_hash",
            "y_true_hash",
            "gene_names_hash",
            "vector_space",
            "canonical_target_set",
        ):
            if group[column].nunique(dropna=False) != 1:
                failures.append(
                    _failure(
                        analysis_id,
                        "MATCHED_ARM_PROVENANCE_MISMATCH",
                        dataset=str(dataset),
                        condition=str(condition),
                        hvg=int(hvg),
                        detail=f"{column} differs at seed={seed}",
                    )
                )
    if analysis_id == "TOPOLOGY-NULL":
        topology_required = {
            "topology_null_ensemble_gate_status",
            "topology_null_ensemble_audit_hash",
            "swapped_edge_fraction",
        }
        missing = topology_required - set(frame.columns)
        if missing:
            failures.append(
                _failure(
                    analysis_id,
                    "TOPOLOGY_PREFLIGHT_LINK_MISSING",
                    detail=str(sorted(missing)),
                )
            )
            return failures
        graph_arms = [arm for arm in arms if arm.startswith("string_go_rewire_")]
        for dataset, group in frame.groupby("dataset", sort=True):
            if set(group["topology_null_ensemble_gate_status"].dropna()) != {"PASS"}:
                failures.append(
                    _failure(
                        analysis_id,
                        "TOPOLOGY_PREFLIGHT_GATE_NOT_PASS",
                        dataset=str(dataset),
                    )
                )
            if group["topology_null_ensemble_audit_hash"].nunique(dropna=False) != 1:
                failures.append(
                    _failure(
                        analysis_id,
                        "TOPOLOGY_PREFLIGHT_AUDIT_HASH_MISMATCH",
                        dataset=str(dataset),
                    )
                )
            support_hashes = {
                arm: group.loc[group["arm"] == arm, "graph_support_hash"].iloc[0]
                for arm in ("string_go", *graph_arms)
                if len(group.loc[group["arm"] == arm])
            }
            rewire_hashes = [support_hashes[arm] for arm in graph_arms if arm in support_hashes]
            if len(rewire_hashes) != 10 or len(set(rewire_hashes)) != 10:
                failures.append(
                    _failure(
                        analysis_id,
                        "TOPOLOGY_REWIRE_HASH_UNIQUENESS_FAILED",
                        dataset=str(dataset),
                    )
                )
            if support_hashes.get("string_go") in set(rewire_hashes):
                failures.append(
                    _failure(
                        analysis_id,
                        "TOPOLOGY_REWIRE_EQUALS_CURATED",
                        dataset=str(dataset),
                    )
                )
            rewire_rows = group[group["arm"].isin(graph_arms)]
            fractions = pd.to_numeric(rewire_rows["swapped_edge_fraction"], errors="coerce")
            if fractions.isna().any() or (fractions < 0.80).any():
                failures.append(
                    _failure(
                        analysis_id,
                        "TOPOLOGY_REWIRE_COVERAGE_GATE_FAILED",
                        dataset=str(dataset),
                    )
                )
    return failures


def _condition_within_dataset_bootstrap(
    table: pd.DataFrame,
    *,
    value_columns: Sequence[str],
    datasets: Sequence[str],
    n_bootstrap: int,
    random_seed: int,
) -> tuple[np.ndarray, str]:
    arrays = {
        dataset: table.loc[table["dataset"] == dataset, list(value_columns)].to_numpy(dtype=float)
        for dataset in datasets
    }
    if any(not len(values) for values in arrays.values()):
        raise RevisionProtocolError("Every dataset requires conditions for hierarchical bootstrap")
    rng = np.random.default_rng(random_seed)
    digest = hashlib.sha256()
    draws = np.empty((n_bootstrap, len(value_columns)), dtype=float)
    for replicate in range(n_bootstrap):
        dataset_means = []
        for dataset, values in arrays.items():
            indices = rng.integers(0, len(values), size=len(values))
            digest.update(dataset.encode("utf-8"))
            digest.update(indices.astype(np.int64).tobytes())
            dataset_means.append(values[indices].mean(axis=0))
        draws[replicate] = np.mean(dataset_means, axis=0)
    return draws, digest.hexdigest()


def _null_bootstrap_p(null_draws: np.ndarray, observed: float) -> float:
    values = np.asarray(null_draws, dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(observed):
        raise RevisionProtocolError("Null-bootstrap values and observed estimate must be finite")
    exceedances = int(np.sum(np.abs(values) >= abs(observed)))
    return float((exceedances + 1) / (len(values) + 1))


def _expected_seeds(protocol: Mapping[str, Any]) -> tuple[int, ...]:
    return tuple(sorted(int(seed) for seed in protocol["statistics"]["expected_seeds"]))


def _expected_condition_total(protocol: Mapping[str, Any]) -> int:
    return int(protocol["planned_fit_counts"]["conditions_per_panel"]) * len(protocol["datasets"])


def _results_manifest_hash(frame: pd.DataFrame, metric: str) -> str:
    columns = [
        "dataset",
        "hvg",
        "panel",
        "arm",
        "condition",
        "seed",
        "analysis_block",
        "epochs_requested",
        metric,
        "artifact_hash",
        "architecture_hash",
        "initialization_hash",
        "preprocessing_state_hash",
        "input_hashes_hash",
        "matched_input_hashes_hash",
        "split_hash",
        "graph_support_hash",
        "code_hash",
        "y_true_hash",
        "gene_names_hash",
        "vector_space",
        "condition_panel_manifest_hash",
        "target_encoding_audit_hash",
        "preflight_summary_hash",
        "dataset_passport_hash",
        "canonical_target_set",
        "dataset_input_hash",
        "protocol_file_hash",
        "environment_manifest_hash",
        "environment_lock_hash",
        "git_commit",
        "git_worktree_status",
        "git_status_hash",
        "git_provenance_hash",
    ]
    records = frame[columns].sort_values(columns[:-1], kind="mergesort").to_dict(orient="records")
    return canonical_sha256(records)


def _failure(
    analysis_id: str,
    reason_code: str,
    *,
    dataset: str | None = None,
    condition: str | None = None,
    hvg: int | None = None,
    arm: str | None = None,
    detail: str = "",
) -> dict[str, Any]:
    return {
        "analysis_id": analysis_id,
        "reason_code": reason_code,
        "dataset": dataset,
        "condition": condition,
        "hvg": hvg,
        "arm": arm,
        "detail": detail,
    }


def _failure_frame(failures: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    columns = ["analysis_id", "reason_code", "dataset", "condition", "hvg", "arm", "detail"]
    return pd.DataFrame(failures, columns=columns)


def _withheld_result(
    analysis_id: str, failures: Sequence[Mapping[str, Any]]
) -> AnalysisReleaseResult:
    frame = _failure_frame(failures)
    return AnalysisReleaseResult(
        registry={
            "analysis_id": analysis_id,
            "status": "WITHHELD",
            "failure_count": int(len(frame)),
            "failure_ledger_hash": canonical_sha256(frame.fillna("").to_dict(orient="records")),
        },
        detail_tables={},
        failures=frame,
    )


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Return monotone Benjamini-Hochberg adjusted p-values in original order."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not len(values):
        raise RevisionProtocolError("p_values must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise RevisionProtocolError("p_values must be finite values in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted

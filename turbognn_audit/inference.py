"""Frozen randomization and heterogeneity sensitivity tests for TDS-09/TDS-36."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .hashing import nfc_text, utf8_sort


@dataclass(frozen=True)
class MonteCarloTest:
    """One plus-one Monte Carlo result with inclusive tie counting."""

    statistic: float
    p_value: float
    extreme_draw_n: int
    draw_n: int
    rng: str
    seed: int


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm-adjusted p values while preserving caller labels."""
    if not p_values:
        return {}
    if any(not 0.0 <= value <= 1.0 for value in p_values.values()):
        raise ValueError("p values must lie in [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    family_size = len(ordered)
    for rank, (label, value) in enumerate(ordered):
        running = max(running, min(1.0, (family_size - rank) * value))
        adjusted[label] = running
    return {label: adjusted[label] for label in p_values}


def _condition_arrays(
    frame: pd.DataFrame,
    datasets: Sequence[str],
    value_column: str,
) -> dict[str, np.ndarray]:
    required = {"dataset", "condition", value_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Condition table lacks columns: {sorted(missing)}")
    if frame.duplicated(["dataset", "condition"]).any():
        raise ValueError("Condition table contains duplicate dataset-condition rows")
    canonical = frame.copy()
    canonical["dataset"] = canonical["dataset"].map(lambda value: nfc_text(str(value)))
    canonical["condition"] = canonical["condition"].map(lambda value: nfc_text(str(value)))
    normalized_datasets = tuple(nfc_text(dataset) for dataset in datasets)
    if len(set(normalized_datasets)) != len(normalized_datasets):
        raise ValueError("Frozen dataset strata are duplicated after NFC normalization")
    if canonical.duplicated(["dataset", "condition"]).any():
        raise ValueError("Condition table contains duplicate canonical dataset-condition rows")
    observed = set(canonical["dataset"].astype(str))
    if observed != set(normalized_datasets):
        raise ValueError(
            f"Dataset strata mismatch: expected {sorted(normalized_datasets)}, "
            f"observed {sorted(observed)}"
        )
    arrays: dict[str, np.ndarray] = {}
    for dataset in normalized_datasets:
        dataset_rows = canonical.loc[canonical["dataset"] == dataset].set_index("condition")
        order = utf8_sort(dataset_rows.index.astype(str).tolist())
        values = dataset_rows.loc[order, value_column].to_numpy(dtype=float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"Dataset {dataset!r} has empty or non-finite condition effects")
        arrays[dataset] = values
    return arrays


def equal_dataset_statistic(arrays: Mapping[str, np.ndarray]) -> float:
    """Average condition effects within dataset and dataset means with equal weight."""
    if not arrays:
        raise ValueError("At least one dataset stratum is required")
    return float(np.mean([np.asarray(values, dtype=float).mean() for values in arrays.values()]))


def sign_flip_equal_dataset_test(
    frame: pd.DataFrame,
    *,
    datasets: Sequence[str],
    value_column: str = "delta_z",
    draws: int = 10_000,
    seed: int = 20_260_804,
) -> MonteCarloTest:
    """Sign-flip complete conditions and count two-sided inclusive ties with plus one."""
    if draws < 1:
        raise ValueError("draws must be positive")
    arrays = _condition_arrays(frame, datasets, value_column)
    observed = equal_dataset_statistic(arrays)
    generator = np.random.Generator(np.random.PCG64(seed))
    extreme = 0
    for _ in range(draws):
        permuted = {
            dataset: values * generator.choice((-1.0, 1.0), size=len(values))
            for dataset, values in arrays.items()
        }
        statistic = equal_dataset_statistic(permuted)
        extreme += int(abs(statistic) >= abs(observed))
    return MonteCarloTest(
        statistic=observed,
        p_value=(extreme + 1) / (draws + 1),
        extreme_draw_n=extreme,
        draw_n=draws,
        rng="PCG64",
        seed=seed,
    )


def dataset_sign_flip_holm_family(
    frame: pd.DataFrame,
    *,
    datasets: Sequence[str],
    value_column: str = "delta_z",
    draws: int = 100_000,
    seed: int = 20_260_804,
) -> pd.DataFrame:
    """Test four dataset effects in frozen order and Holm-adjust the raw p values."""
    arrays = _condition_arrays(frame, datasets, value_column)
    generator = np.random.Generator(np.random.PCG64(seed))
    rows: list[dict[str, float | int | str]] = []
    raw: dict[str, float] = {}
    for dataset in datasets:
        values = arrays[dataset]
        observed = float(values.mean())
        extreme = 0
        for _ in range(draws):
            signs = generator.choice((-1.0, 1.0), size=len(values))
            extreme += int(abs(float(np.mean(values * signs))) >= abs(observed))
        p_value = (extreme + 1) / (draws + 1)
        raw[dataset] = p_value
        rows.append(
            {
                "dataset": dataset,
                "mean_delta_z": observed,
                "raw_p": p_value,
                "extreme_draw_n": extreme,
                "draw_n": draws,
                "rng": "PCG64",
                "seed": seed,
            }
        )
    adjusted = holm_adjust(raw)
    for row in rows:
        row["holm_p"] = adjusted[str(row["dataset"])]
    return pd.DataFrame(rows)


def _q_statistic(arrays: Mapping[str, np.ndarray]) -> float:
    means = {dataset: float(values.mean()) for dataset, values in arrays.items()}
    grand = float(np.concatenate(list(arrays.values())).mean())
    return float(sum(len(arrays[dataset]) * (means[dataset] - grand) ** 2 for dataset in arrays))


def centered_residual_q_test(
    frame: pd.DataFrame,
    *,
    datasets: Sequence[str],
    value_column: str = "delta_z",
    draws: int = 10_000,
    seed: int = 20_260_805,
) -> MonteCarloTest:
    """Bootstrap within-dataset centered residuals under a common-effect null."""
    if draws < 1:
        raise ValueError("draws must be positive")
    arrays = _condition_arrays(frame, datasets, value_column)
    observed = _q_statistic(arrays)
    common_mean = float(np.concatenate(list(arrays.values())).mean())
    residuals = {dataset: values - values.mean() for dataset, values in arrays.items()}
    generator = np.random.Generator(np.random.PCG64(seed))
    extreme = 0
    for _ in range(draws):
        bootstrap = {
            dataset: common_mean + generator.choice(values, size=len(values), replace=True)
            for dataset, values in residuals.items()
        }
        statistic = _q_statistic(bootstrap)
        extreme += int(statistic >= observed)
    return MonteCarloTest(
        statistic=observed,
        p_value=(extreme + 1) / (draws + 1),
        extreme_draw_n=extreme,
        draw_n=draws,
        rng="PCG64",
        seed=seed,
    )

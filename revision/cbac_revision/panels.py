"""Deterministic metadata-stratified primary and non-overlap condition panels."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .artifacts import canonical_sha256
from .errors import RevisionProtocolError


@dataclass(frozen=True)
class ConditionPanels:
    """Primary and sensitivity panels plus condition-level metadata."""

    primary: tuple[str, ...]
    sensitivity: tuple[str, ...]
    legacy_first50: tuple[str, ...]
    condition_table: pd.DataFrame
    representativeness: pd.DataFrame
    sensitivity_trigger: Mapping[str, Any]


def build_condition_panels(
    canonical_labels: Sequence[str],
    control_label: str,
    cell_metadata: pd.DataFrame,
    *,
    panel_size: int = 50,
    descriptive_metadata_columns: Sequence[str] = (),
    selection_seed: int = 20260806,
    trigger_total_variation_threshold: float,
    trigger_cell_count_threshold: float,
    dataset_name: str = "UNSPECIFIED_FIXTURE",
    target_mapping_records: Sequence[Mapping[str, Any]] = (),
    legacy_first50_conditions: Sequence[str] = (),
) -> ConditionPanels:
    """Select new panels after excluding the hash-bound frozen legacy first-50 panel."""

    if panel_size <= 0:
        raise RevisionProtocolError("panel_size must be positive")
    if (
        not np.isfinite(trigger_total_variation_threshold)
        or not np.isfinite(trigger_cell_count_threshold)
        or trigger_total_variation_threshold < 0
        or trigger_cell_count_threshold < 0
    ):
        raise RevisionProtocolError(
            "Sensitivity-trigger thresholds must be finite and non-negative"
        )
    labels = np.asarray(canonical_labels, dtype=object)
    if len(labels) != len(cell_metadata):
        raise RevisionProtocolError("cell_metadata and condition labels must have equal lengths")
    perturbation_mask = labels != control_label
    conditions = sorted(set(str(value) for value in labels[perturbation_mask]))
    if not conditions:
        raise RevisionProtocolError("No non-control conditions are available for panel selection")

    rows: list[dict[str, object]] = []
    for condition in conditions:
        mask = labels == condition
        row: dict[str, object] = {"condition": condition, "n_cells": int(mask.sum())}
        for column in descriptive_metadata_columns:
            if column not in cell_metadata:
                row[column] = "__NOT_AVAILABLE__"
                continue
            source = cell_metadata.loc[mask, column].dropna()
            values = source.unique()
            if len(values) > 1:
                raise RevisionProtocolError(
                    f"Metadata column {column!r} is not condition-constant for {condition!r}"
                )
            if not len(values):
                row[column] = "__MISSING__"
            elif pd.api.types.is_numeric_dtype(cell_metadata[column]):
                row[column] = float(values[0])
            else:
                row[column] = str(values[0])
        if "perturbation_order" in cell_metadata:
            order_values = (
                cell_metadata.loc[mask, "perturbation_order"].dropna().astype(str).unique()
            )
            if len(order_values) > 1:
                raise RevisionProtocolError(
                    f"perturbation_order is not condition-constant for {condition!r}"
                )
            row["perturbation_order"] = order_values[0] if len(order_values) else "__MISSING__"
        else:
            target_count = len([part for part in re.split(r"[+,;|]", condition) if part.strip()])
            row["perturbation_order"] = "combination" if target_count > 1 else "single"
        rows.append(row)
    table = pd.DataFrame(rows)
    quartile_count = min(4, len(table))
    ranked = table["n_cells"].rank(method="first")
    table["condition_cell_count_quartile"] = pd.qcut(
        ranked, q=quartile_count, labels=[f"Q{index + 1}" for index in range(quartile_count)]
    ).astype(str)
    strata_columns = [
        "condition_cell_count_quartile",
        "perturbation_order",
    ]
    table["panel_stratum"] = table[strata_columns].astype(str).agg("|".join, axis=1)

    legacy = tuple(str(condition) for condition in legacy_first50_conditions)
    if len(set(legacy)) != len(legacy):
        raise RevisionProtocolError("legacy_first50_conditions must be unique")
    legacy_set = set(legacy)
    primary_size = min(panel_size, int((~table["condition"].isin(legacy_set)).sum()))
    primary = _stratified_select(table, primary_size, selection_seed, excluded=legacy_set)
    sensitivity_exclusions = legacy_set | set(primary)
    sensitivity_size = min(
        panel_size, int((~table["condition"].isin(sensitivity_exclusions)).sum())
    )
    sensitivity = _stratified_select(
        table,
        sensitivity_size,
        selection_seed + 1,
        excluded=sensitivity_exclusions,
    )
    if (
        set(primary) & set(sensitivity)
        or set(primary) & legacy_set
        or set(sensitivity) & legacy_set
    ):
        raise RevisionProtocolError("Legacy, primary, and sensitivity panels must be disjoint")

    table["legacy_first50_panel"] = table["condition"].isin(legacy_set)
    table["primary_panel"] = table["condition"].isin(primary)
    table["sensitivity_panel"] = table["condition"].isin(sensitivity)
    representativeness = _representativeness(
        table,
        primary,
        sensitivity,
        dataset_name=dataset_name,
        descriptive_metadata_columns=descriptive_metadata_columns,
        target_mapping_records=target_mapping_records,
    )
    representativeness_hash = canonical_sha256(
        representativeness.fillna("NOT_APPLICABLE").to_dict(orient="records")
    )
    sensitivity_trigger = _sensitivity_trigger(
        table,
        representativeness,
        primary,
        total_variation_threshold=trigger_total_variation_threshold,
        cell_count_threshold=trigger_cell_count_threshold,
    )
    sensitivity_trigger["representativeness_table_hash"] = representativeness_hash
    return ConditionPanels(
        primary=tuple(primary),
        sensitivity=tuple(sensitivity),
        legacy_first50=legacy,
        condition_table=table.sort_values("condition").reset_index(drop=True),
        representativeness=representativeness,
        sensitivity_trigger=sensitivity_trigger,
    )


def _stratified_select(table: pd.DataFrame, size: int, seed: int, excluded: set[str]) -> list[str]:
    if size == 0:
        return []
    available = table[~table["condition"].isin(excluded)].copy()
    if size > len(available):
        raise RevisionProtocolError("Panel size exceeds the remaining condition universe")
    universe_counts = table["panel_stratum"].value_counts().sort_index()
    available_counts = (
        available["panel_stratum"].value_counts().reindex(universe_counts.index, fill_value=0)
    )
    raw_quotas = size * universe_counts / universe_counts.sum()
    nonempty_strata = [
        stratum for stratum in universe_counts.index if available_counts[stratum] > 0
    ]
    if size >= len(nonempty_strata):
        quotas = pd.Series(0, index=universe_counts.index, dtype=int)
        quotas.loc[nonempty_strata] = 1
    else:
        quotas = pd.Series(0, index=universe_counts.index, dtype=int)
    remaining = size - int(quotas.sum())
    while remaining:
        candidates = [
            stratum
            for stratum in universe_counts.index
            if quotas[stratum] < available_counts[stratum]
        ]
        if not candidates:
            raise RevisionProtocolError("Unable to allocate the requested stratified panel")
        chosen = min(
            candidates,
            key=lambda stratum: (
                -(raw_quotas[stratum] - quotas[stratum]),
                _stable_hash(seed, stratum),
            ),
        )
        quotas[chosen] += 1
        remaining -= 1

    selected: list[str] = []
    for stratum, quota in quotas.items():
        candidates = available.loc[available["panel_stratum"] == stratum, "condition"].tolist()
        candidates.sort(key=lambda condition: _stable_hash(seed, condition))
        selected.extend(candidates[: int(quota)])
    return sorted(selected)


def _representativeness(
    table: pd.DataFrame,
    primary: Sequence[str],
    sensitivity: Sequence[str],
    *,
    dataset_name: str,
    descriptive_metadata_columns: Sequence[str],
    target_mapping_records: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def append(
        panel: str,
        variable: str,
        level: str,
        statistic: str,
        value: float | int | None,
        *,
        status: str = "MEASURED",
    ) -> None:
        rows.append(
            {
                "dataset": dataset_name,
                "panel": panel,
                "variable": variable,
                "level": level,
                "statistic": statistic,
                "value": value,
                "status": status,
            }
        )

    working = table.copy()
    working["combined_selection_stratum"] = working["panel_stratum"].astype(str)
    categorical = [
        "condition_cell_count_quartile",
        "perturbation_order",
        "combined_selection_stratum",
    ]
    numeric = ["condition_cell_count", "log1p_condition_cell_count"]
    working["condition_cell_count"] = working["n_cells"].astype(float)
    working["log1p_condition_cell_count"] = np.log1p(working["n_cells"].astype(float))
    for column in descriptive_metadata_columns:
        if column not in working or (working[column] == "__NOT_AVAILABLE__").all():
            for panel_name in ("primary", "sensitivity"):
                append(
                    panel_name,
                    column,
                    "__NOT_AVAILABLE__",
                    "availability",
                    None,
                    status="NOT_AVAILABLE",
                )
        elif pd.api.types.is_numeric_dtype(working[column]):
            numeric.append(column)
        else:
            categorical.append(column)

    for panel_name, panel in (("primary", primary), ("sensitivity", sensitivity)):
        selected = working[working["condition"].isin(panel)]
        for variable in categorical:
            universe_counts = working[variable].astype(str).value_counts().sort_index()
            panel_counts = (
                selected[variable]
                .astype(str)
                .value_counts()
                .reindex(universe_counts.index, fill_value=0)
            )
            universe_proportions = universe_counts / universe_counts.sum()
            panel_proportions = panel_counts / panel_counts.sum()
            tv = float(0.5 * np.abs(universe_proportions - panel_proportions).sum())
            for level in universe_counts.index:
                values = {
                    "universe_count": int(universe_counts[level]),
                    "panel_count": int(panel_counts[level]),
                    "universe_proportion": float(universe_proportions[level]),
                    "panel_proportion": float(panel_proportions[level]),
                    "absolute_proportion_difference": float(
                        abs(universe_proportions[level] - panel_proportions[level])
                    ),
                }
                for statistic, value in values.items():
                    append(panel_name, variable, str(level), statistic, value)
            append(
                panel_name,
                variable,
                "__ALL_LEVELS__",
                "variable_total_variation_distance",
                tv,
            )
        for variable in numeric:
            universe_values = working[variable].astype(float)
            panel_values = selected[variable].astype(float)
            universe_sd = float(universe_values.std(ddof=1)) if len(universe_values) > 1 else 0.0
            q1, q3 = panel_values.quantile([0.25, 0.75])
            universe_q1, universe_q3 = universe_values.quantile([0.25, 0.75])
            statistics = {
                "n": int(len(panel_values)),
                "mean": float(panel_values.mean()),
                "sd": float(panel_values.std(ddof=1)) if len(panel_values) > 1 else 0.0,
                "median": float(panel_values.median()),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(q3 - q1),
                "standardized_mean_difference": (
                    float((panel_values.mean() - universe_values.mean()) / universe_sd)
                    if universe_sd > 0
                    else 0.0
                ),
                "universe_n": int(len(universe_values)),
                "universe_mean": float(universe_values.mean()),
                "universe_sd": universe_sd,
                "universe_median": float(universe_values.median()),
                "universe_q1": float(universe_q1),
                "universe_q3": float(universe_q3),
                "universe_iqr": float(universe_q3 - universe_q1),
            }
            for statistic, value in statistics.items():
                append(panel_name, variable, "__OVERALL__", statistic, value)
        append(
            panel_name,
            "panel_coverage",
            "eligible_condition_universe",
            "condition_coverage",
            float(len(selected) / len(working)),
        )
        append(
            panel_name,
            "panel_coverage",
            "eligible_condition_universe",
            "cell_coverage",
            float(selected["n_cells"].sum() / working["n_cells"].sum()),
        )
        append(
            panel_name,
            "panel_overlap",
            "other_panel",
            "overlap_condition_count",
            int(len(set(primary) & set(sensitivity))),
        )
        _append_target_mapping_rows(append, panel_name, target_mapping_records)
    return (
        pd.DataFrame(rows)
        .sort_values(["dataset", "panel", "variable", "level", "statistic"])
        .reset_index(drop=True)
    )


def _append_target_mapping_rows(
    append: Any,
    panel_name: str,
    records: Sequence[Mapping[str, Any]],
) -> None:
    if not records:
        append(
            panel_name,
            "target_mapping",
            "__NOT_AVAILABLE__",
            "availability",
            None,
            status="NOT_AVAILABLE",
        )
        return
    eligible = [record for record in records if bool(record.get("cell_count_eligible"))]
    eligible_mapped = sum(bool(record.get("mapped")) for record in eligible)
    raw_total = sum(int(record.get("raw_non_control_label_count", 0)) for record in records)
    raw_mapped = sum(
        int(record.get("raw_non_control_label_count", 0))
        for record in records
        if bool(record.get("mapped")) and bool(record.get("cell_count_eligible"))
    )
    for statistic, value in {
        "condition_count": len(eligible),
        "mapped_condition_count": eligible_mapped,
        "mapping_coverage": float(eligible_mapped / len(eligible)) if eligible else 0.0,
    }.items():
        append(panel_name, "target_mapping", "eligible_universe", statistic, value)
    for statistic, value in {
        "raw_non_control_label_count": raw_total,
        "mapped_raw_label_count": raw_mapped,
        "excluded_raw_label_count": raw_total - raw_mapped,
    }.items():
        append(panel_name, "target_mapping", "raw_non_control", statistic, value)
    reason_counts: dict[str, int] = {}
    for record in records:
        if bool(record.get("mapped")) and bool(record.get("cell_count_eligible")):
            continue
        reason = str(record.get("exclusion_reason_code") or "UNSPECIFIED_EXCLUSION")
        reason_counts[reason] = reason_counts.get(reason, 0) + int(
            record.get("raw_non_control_label_count", 0)
        )
    if not reason_counts:
        append(panel_name, "target_mapping_exclusion_reason", "NONE", "excluded_raw_label_count", 0)
    for reason, count in sorted(reason_counts.items()):
        append(
            panel_name,
            "target_mapping_exclusion_reason",
            reason,
            "excluded_raw_label_count",
            count,
        )


def _stable_hash(seed: int, value: object) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _sensitivity_trigger(
    table: pd.DataFrame,
    representativeness: pd.DataFrame,
    primary: Sequence[str],
    *,
    total_variation_threshold: float,
    cell_count_threshold: float,
) -> dict[str, Any]:
    primary_rows = representativeness[representativeness["panel"] == "primary"]
    tv_rows = primary_rows[
        (primary_rows["statistic"] == "variable_total_variation_distance")
        & (primary_rows["variable"] == "combined_selection_stratum")
    ]
    tv_by_variable = {
        str(row.variable): float(row.value) for row in tv_rows.itertuples(index=False)
    }
    total_variation = max(tv_by_variable.values(), default=0.0)
    maximum_variable = "combined_selection_stratum" if tv_by_variable else "NOT_AVAILABLE"
    cell_row = primary_rows[
        (primary_rows["variable"] == "log1p_condition_cell_count")
        & (primary_rows["statistic"] == "standardized_mean_difference")
    ]
    if len(cell_row) == 1:
        absolute_standardized_difference = abs(float(cell_row.iloc[0]["value"]))
        cell_count_status = "MEASURED"
    else:
        absolute_standardized_difference = 0.0
        cell_count_status = "NOT_AVAILABLE"
    tv_trigger = total_variation > total_variation_threshold
    cell_count_trigger = absolute_standardized_difference > cell_count_threshold
    return {
        "sensitivity_triggered": bool(tv_trigger or cell_count_trigger),
        "trigger_total_variation_distance": total_variation,
        "trigger_maximum_total_variation_variable": maximum_variable,
        "trigger_variable_total_variation_distances": tv_by_variable,
        "trigger_tv_threshold": total_variation_threshold,
        "trigger_tv_exceeded": bool(tv_trigger),
        "trigger_abs_standardized_log1p_cell_count_difference": absolute_standardized_difference,
        "trigger_cell_count_threshold": cell_count_threshold,
        "trigger_cell_count_exceeded": bool(cell_count_trigger),
        "trigger_cell_count_status": cell_count_status,
    }

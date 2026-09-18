"""Pre-outcome precision simulation using only archived variance structure."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .artifacts import canonical_sha256, file_sha256
from .errors import RevisionProtocolError

SHA256_RE = re.compile(r"[0-9a-f]{64}")
DATASETS = ("adamson", "norman", "replogle_k562", "replogle_rpe1")
FROZEN_ARCHIVE_ID = "submitted_legacy_archive_pre_revision"
FROZEN_SELECTION_DATE = "2026-08-07"
CONSERVATIVE_SCENARIOS = (
    {
        "scenario_id": "archive_proxy",
        "variance_multiplier": 1.0,
        "target_cluster_icc_floor": 0.0,
    },
    {
        "scenario_id": "variance_inflated_125",
        "variance_multiplier": 1.25,
        "target_cluster_icc_floor": 0.0,
    },
    {
        "scenario_id": "cluster_floor_025",
        "variance_multiplier": 1.0,
        "target_cluster_icc_floor": 0.25,
    },
    {
        "scenario_id": "joint_variance_125_cluster_floor_025",
        "variance_multiplier": 1.25,
        "target_cluster_icc_floor": 0.25,
    },
)
CONDITION_COLUMNS = ("dataset", "condition", "archived_delta")
TARGET_COLUMNS = ("dataset", "condition", "target")


def build_precision_registry(
    specification: Mapping[str, Any],
    *,
    condition_table_path: Path,
    target_map_path: Path,
    expected_condition_table_sha256: str,
    expected_target_map_sha256: str,
) -> dict[str, Any]:
    """Derive and simulate precision from caller-pinned legacy source rows."""

    expected_fields = {
        "schema_version",
        "stage",
        "source_archive_id",
        "source_selection_date",
        "planned_conditions_per_dataset",
        "uncertainty_half_width_target",
        "simulation_replicates",
        "bootstrap_replicates_per_simulation",
        "simulation_random_seed",
        "conservative_scenarios",
    }
    if set(specification) != expected_fields:
        raise RevisionProtocolError("Precision specification must have exact pre-outcome fields")
    if specification.get("schema_version") != "1.0" or specification.get("stage") != "PRE_OUTCOME":
        raise RevisionProtocolError("Precision specification stage must be PRE_OUTCOME")
    if (
        specification.get("source_archive_id") != FROZEN_ARCHIVE_ID
        or specification.get("source_selection_date") != FROZEN_SELECTION_DATE
    ):
        raise RevisionProtocolError("Precision archive identity/date differs from frozen design")
    _validate_pinned_source(
        condition_table_path,
        expected_condition_table_sha256,
        label="archive condition table",
    )
    _validate_pinned_source(
        target_map_path,
        expected_target_map_sha256,
        label="canonical target map",
    )
    planned = _strict_integer(specification.get("planned_conditions_per_dataset"), "planned")
    replicates = _strict_integer(specification.get("simulation_replicates"), "replicates")
    bootstrap_replicates = _strict_integer(
        specification.get("bootstrap_replicates_per_simulation"),
        "bootstrap_replicates_per_simulation",
    )
    random_seed = _strict_integer(specification.get("simulation_random_seed"), "random_seed")
    target = _strict_number(
        specification.get("uncertainty_half_width_target"), "uncertainty_half_width_target"
    )
    if (
        planned != 50
        or replicates < 100
        or bootstrap_replicates < 500
        or random_seed != 20260806
        or target != 0.010
        or not _exact_typed_equal(
            specification.get("conservative_scenarios"), list(CONSERVATIVE_SCENARIOS)
        )
    ):
        raise RevisionProtocolError("Precision design constants are not frozen")
    condition_rows = _read_condition_table(condition_table_path)
    target_rows = _read_target_map(target_map_path)
    by_dataset = _derive_variance_components(condition_rows, target_rows)

    scenario_results = []
    for scenario_index, scenario in enumerate(CONSERVATIVE_SCENARIOS):
        variance_multiplier = float(scenario["variance_multiplier"])
        icc_floor = float(scenario["target_cluster_icc_floor"])
        adjusted_components = {
            dataset: {
                **by_dataset[dataset],
                "condition_variance": (
                    by_dataset[dataset]["condition_variance"] * variance_multiplier
                ),
                "target_cluster_icc": max(by_dataset[dataset]["target_cluster_icc"], icc_floor),
            }
            for dataset in DATASETS
        }
        dataset_variances: dict[str, float] = {}
        for dataset in DATASETS:
            component = adjusted_components[dataset]
            design_effect = (
                1.0 + (component["mean_guides_per_target"] - 1.0) * component["target_cluster_icc"]
            )
            dataset_variances[dataset] = component["condition_variance"] * design_effect / planned
        standard_error = math.sqrt(sum(dataset_variances.values()) / len(DATASETS) ** 2)
        analytic_half_width = 1.959963984540054 * standard_error
        simulated_widths = _simulate_interval_widths(
            adjusted_components,
            planned=planned,
            replicates=replicates,
            bootstrap_replicates=bootstrap_replicates,
            random_seed=random_seed + scenario_index,
        )
        scenario_results.append(
            {
                **scenario,
                "simulation_random_seed": random_seed + scenario_index,
                "effective_components": [
                    {
                        "dataset": dataset,
                        "condition_variance": adjusted_components[dataset]["condition_variance"],
                        "target_cluster_icc": adjusted_components[dataset]["target_cluster_icc"],
                    }
                    for dataset in DATASETS
                ],
                "dataset_mean_sampling_variances": dataset_variances,
                "expected_standard_error": standard_error,
                "analytic_uncertainty_half_width_95": analytic_half_width,
                "simulated_interval_width_q90": float(np.quantile(simulated_widths, 0.90)),
                "simulation_interval_widths_sha256": canonical_sha256(simulated_widths.tolist()),
            }
        )
    worst = max(
        scenario_results,
        key=lambda row: (float(row["simulated_interval_width_q90"]), row["scenario_id"]),
    )
    simulated_width_q90 = float(worst["simulated_interval_width_q90"])
    registry: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "PRE_OUTCOME_PRECISION_REGISTERED",
        "design_input": dict(specification),
        "trusted_source_bindings": {
            "condition_table_source_id": condition_table_path.name,
            "condition_table_file_sha256": expected_condition_table_sha256,
            "condition_table_records_sha256": canonical_sha256(
                condition_rows.to_dict(orient="records")
            ),
            "target_map_source_id": target_map_path.name,
            "target_map_file_sha256": expected_target_map_sha256,
            "target_map_records_sha256": canonical_sha256(target_rows.to_dict(orient="records")),
        },
        "variance_component_derivation": {
            "condition_variance": "sample_variance_ddof_1_of_archived_condition_deltas",
            "target_cluster_icc": (
                "one_way_random_effects_anova_icc_with_unbalanced_n0_truncated_to_0_0.999999"
            ),
            "mean_guides_per_target": "arithmetic_mean_condition_count_per_target",
            "condition_target_join": "exact_one_to_one_on_dataset_and_condition",
            "components": [{"dataset": dataset, **by_dataset[dataset]} for dataset in DATASETS],
        },
        "zero_centered": True,
        "new_outcome_access": "PROHIBITED",
        "target_role": "PRECISION_NOT_UTILITY_OR_SUCCESS",
        "equal_benchmark_dataset_weight": True,
        "source_limitations": (
            "legacy_architecture_and_mask_defective_contrasts_are_an_archive_proxy_"
            "and_do_not_guarantee_matched_run_interval_width"
        ),
        "conservative_scenario_contract": list(CONSERVATIVE_SCENARIOS),
        "scenario_results": scenario_results,
        "worst_case_scenario_id": worst["scenario_id"],
        "worst_case_selection_rule": "maximum_unrounded_simulated_interval_width_q90",
        "simulation_scheme": (
            "zero_centered_target_cluster_random_effect_generation_then_"
            "target_cluster_resampling_within_dataset_equal_dataset_mean"
        ),
        "simulated_interval_width_q90": simulated_width_q90,
        "precision_interval_width_threshold": 2.0 * target,
        "precision_gate": "PASS" if simulated_width_q90 <= 2.0 * target else "NOT_MET",
        "precision_target_met_in_design_simulation": simulated_width_q90 <= 2.0 * target,
        "simulation_interval_widths_sha256": worst["simulation_interval_widths_sha256"],
    }
    registry["registry_sha256"] = canonical_sha256(registry)
    return registry


def _simulate_interval_widths(
    components: Mapping[str, Mapping[str, Any]],
    *,
    planned: int,
    replicates: int,
    bootstrap_replicates: int,
    random_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    simulated_widths = np.empty(replicates, dtype=np.float64)
    for simulation_index in range(replicates):
        dataset_bootstrap_means: list[np.ndarray] = []
        for dataset in DATASETS:
            component = components[dataset]
            mean_guides = float(component["mean_guides_per_target"])
            n_targets = max(2, int(math.ceil(planned / mean_guides)))
            target_indices = np.minimum(
                np.floor(np.arange(planned) / mean_guides).astype(int), n_targets - 1
            )
            condition_variance = float(component["condition_variance"])
            target_cluster_icc = float(component["target_cluster_icc"])
            cluster_sd = math.sqrt(condition_variance * target_cluster_icc)
            residual_sd = math.sqrt(condition_variance * (1.0 - target_cluster_icc))
            effects = rng.normal(0.0, cluster_sd, size=n_targets)[target_indices] + rng.normal(
                0.0, residual_sd, size=planned
            )
            cluster_sums = np.bincount(target_indices, weights=effects, minlength=n_targets)
            cluster_counts = np.bincount(target_indices, minlength=n_targets)
            sampled_targets = rng.integers(0, n_targets, size=(bootstrap_replicates, n_targets))
            dataset_bootstrap_means.append(
                cluster_sums[sampled_targets].sum(axis=1)
                / cluster_counts[sampled_targets].sum(axis=1)
            )
        equal_dataset_draws = np.mean(np.stack(dataset_bootstrap_means), axis=0)
        lower, upper = np.quantile(equal_dataset_draws, [0.025, 0.975])
        simulated_widths[simulation_index] = float(upper - lower)
    return simulated_widths


def read_precision_registry(
    path: Path,
    *,
    condition_table_path: Path,
    target_map_path: Path,
    expected_condition_table_sha256: str,
    expected_target_map_sha256: str,
) -> dict[str, Any]:
    """Read and independently recompute a pre-outcome precision registry."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RevisionProtocolError(f"Invalid precision registry: {error}") from error
    if not isinstance(payload, dict):
        raise RevisionProtocolError("Precision registry must be a mapping")
    declared_hash = payload.get("registry_sha256")
    unsigned = dict(payload)
    unsigned.pop("registry_sha256", None)
    if declared_hash != canonical_sha256(unsigned):
        raise RevisionProtocolError("Precision registry self hash is invalid")
    design_input = payload.get("design_input")
    if not isinstance(design_input, dict):
        raise RevisionProtocolError("Precision registry design_input is missing")
    expected = build_precision_registry(
        design_input,
        condition_table_path=condition_table_path,
        target_map_path=target_map_path,
        expected_condition_table_sha256=expected_condition_table_sha256,
        expected_target_map_sha256=expected_target_map_sha256,
    )
    if payload != expected:
        raise RevisionProtocolError("Precision registry does not match independent recomputation")
    return payload


def write_precision_registry(
    path: Path,
    registry: Mapping[str, Any],
    *,
    condition_table_path: Path,
    target_map_path: Path,
    expected_condition_table_sha256: str,
    expected_target_map_sha256: str,
) -> Path:
    """Atomically write a validated precision registry."""

    validated = build_precision_registry(
        registry["design_input"],
        condition_table_path=condition_table_path,
        target_map_path=target_map_path,
        expected_condition_table_sha256=expected_condition_table_sha256,
        expected_target_map_sha256=expected_target_map_sha256,
    )
    if dict(registry) != validated:
        raise RevisionProtocolError("Precision registry failed round-trip validation")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(validated, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)
    return path


def _validate_pinned_source(path: Path, expected_sha256: str, *, label: str) -> None:
    if not isinstance(expected_sha256, str) or not SHA256_RE.fullmatch(expected_sha256):
        raise RevisionProtocolError(f"Precision {label} caller-pinned SHA-256 is invalid")
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise RevisionProtocolError(f"Precision {label} differs from caller-pinned source")


def _read_condition_table(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise RevisionProtocolError(
            f"Precision archive condition table is invalid: {error}"
        ) from error
    if tuple(frame.columns) != CONDITION_COLUMNS or frame.empty or frame.isna().any().any():
        raise RevisionProtocolError("Precision archive condition table schema is invalid")
    if (
        not frame["dataset"].map(lambda value: isinstance(value, str) and value in DATASETS).all()
        or not frame["condition"]
        .map(lambda value: isinstance(value, str) and value and value == value.strip())
        .all()
        or frame.duplicated(["dataset", "condition"]).any()
        or set(frame["dataset"]) != set(DATASETS)
    ):
        raise RevisionProtocolError("Precision archive condition identities are invalid")
    numeric = pd.to_numeric(frame["archived_delta"], errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise RevisionProtocolError("Precision archived deltas must be finite numeric values")
    output = frame.copy()
    output["archived_delta"] = numeric.astype(float)
    return output.sort_values(["dataset", "condition"], kind="mergesort").reset_index(drop=True)


def _read_target_map(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, dtype=str)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise RevisionProtocolError(
            f"Precision canonical target map is invalid: {error}"
        ) from error
    if tuple(frame.columns) != TARGET_COLUMNS or frame.empty or frame.isna().any().any():
        raise RevisionProtocolError("Precision canonical target-map schema is invalid")
    if (
        not frame["dataset"].isin(DATASETS).all()
        or set(frame["dataset"]) != set(DATASETS)
        or frame.duplicated(["dataset", "condition"]).any()
        or not frame["condition"].map(lambda value: bool(value) and value == value.strip()).all()
        or not frame["target"].map(lambda value: bool(value) and value == value.strip()).all()
    ):
        raise RevisionProtocolError("Precision canonical target-map identities are invalid")
    return frame.sort_values(["dataset", "condition"], kind="mergesort").reset_index(drop=True)


def _derive_variance_components(
    condition_rows: pd.DataFrame, target_rows: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    left_keys = set(map(tuple, condition_rows[["dataset", "condition"]].to_numpy().tolist()))
    right_keys = set(map(tuple, target_rows[["dataset", "condition"]].to_numpy().tolist()))
    if left_keys != right_keys:
        missing = sorted(left_keys - right_keys)
        extra = sorted(right_keys - left_keys)
        raise RevisionProtocolError(
            "Precision condition/target-map sets differ: "
            f"missing_targets={missing[:3]}; extra_targets={extra[:3]}"
        )
    merged = condition_rows.merge(
        target_rows,
        on=["dataset", "condition"],
        how="inner",
        validate="one_to_one",
    )
    output: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        group = merged.loc[merged["dataset"] == dataset].copy()
        values = group["archived_delta"].to_numpy(dtype=np.float64)
        if len(values) < 3:
            raise RevisionProtocolError(
                f"Precision dataset {dataset} requires at least three archived conditions"
            )
        condition_variance = float(np.var(values, ddof=1))
        if not math.isfinite(condition_variance) or condition_variance <= 0:
            raise RevisionProtocolError(
                f"Precision dataset {dataset} has non-positive archived condition variance"
            )
        target_counts = group.groupby("target", sort=True).size().astype(float)
        if len(target_counts) < 2:
            raise RevisionProtocolError(
                f"Precision dataset {dataset} requires at least two target clusters"
            )
        mean_guides = float(target_counts.mean())
        raw_icc: float | None
        if bool((target_counts == 1).all()):
            raw_icc = None
            icc = 0.0
            icc_status = "STRUCTURALLY_IRRELEVANT_ONE_GUIDE_PER_TARGET"
        else:
            grand_mean = float(np.mean(values))
            target_means = group.groupby("target", sort=True)["archived_delta"].mean()
            between_ss = float(
                sum(
                    target_counts[target] * (float(target_means[target]) - grand_mean) ** 2
                    for target in target_counts.index
                )
            )
            within_ss = float(
                sum(
                    (float(row.archived_delta) - float(target_means[row.target])) ** 2
                    for row in group.itertuples(index=False)
                )
            )
            n_total = len(group)
            n_targets = len(target_counts)
            within_df = n_total - n_targets
            if within_df < 1:
                raise RevisionProtocolError(
                    f"Precision dataset {dataset} has no within-target ICC degrees of freedom"
                )
            between_ms = between_ss / (n_targets - 1)
            within_ms = within_ss / within_df
            n0 = (n_total - float(np.square(target_counts).sum()) / n_total) / (n_targets - 1)
            denominator = between_ms + (n0 - 1.0) * within_ms
            raw_icc = float((between_ms - within_ms) / denominator) if denominator > 0 else 0.0
            icc = float(min(0.999999, max(0.0, raw_icc)))
            icc_status = "ANOVA_ESTIMATED_TRUNCATED_TO_FROZEN_RANGE"
        output[dataset] = {
            "condition_variance": condition_variance,
            "target_cluster_icc": icc,
            "target_cluster_icc_raw": raw_icc,
            "target_cluster_icc_status": icc_status,
            "mean_guides_per_target": mean_guides,
            "n_archived_conditions": int(len(group)),
            "n_target_clusters": int(len(target_counts)),
            "dataset_condition_records_sha256": canonical_sha256(
                group[["condition", "target", "archived_delta"]]
                .sort_values(["condition"], kind="mergesort")
                .to_dict(orient="records")
            ),
        }
    return output


def _strict_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RevisionProtocolError(f"Precision field {field} must be a JSON number")
    output = float(value)
    if not math.isfinite(output):
        raise RevisionProtocolError(f"Precision field {field} must be finite")
    return output


def _strict_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RevisionProtocolError(f"Precision field {field} must be a JSON integer")
    return value


def _exact_typed_equal(observed: Any, expected: Any) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            _exact_typed_equal(observed[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _exact_typed_equal(left, right) for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant {value!r} is prohibited")


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the precision-registry CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--archive-condition-table", type=Path, required=True)
    parser.add_argument("--archive-condition-table-sha256", required=True)
    parser.add_argument("--canonical-target-map", type=Path, required=True)
    parser.add_argument("--canonical-target-map-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Create a pre-outcome registry without accepting any new outcome vector."""

    parsed = build_argument_parser().parse_args(argv)
    specification = json.loads(
        parsed.specification.read_text(encoding="utf-8"), parse_constant=_reject_constant
    )
    registry = build_precision_registry(
        specification,
        condition_table_path=parsed.archive_condition_table,
        target_map_path=parsed.canonical_target_map,
        expected_condition_table_sha256=parsed.archive_condition_table_sha256,
        expected_target_map_sha256=parsed.canonical_target_map_sha256,
    )
    write_precision_registry(
        parsed.output,
        registry,
        condition_table_path=parsed.archive_condition_table,
        target_map_path=parsed.canonical_target_map,
        expected_condition_table_sha256=parsed.archive_condition_table_sha256,
        expected_target_map_sha256=parsed.canonical_target_map_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

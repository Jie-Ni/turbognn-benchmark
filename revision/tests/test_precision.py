from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from cbac_revision.artifacts import canonical_sha256, file_sha256
from cbac_revision.errors import RevisionProtocolError
from cbac_revision.precision import (
    CONSERVATIVE_SCENARIOS,
    DATASETS,
    build_precision_registry,
    read_precision_registry,
)


def _sources(root: Path) -> tuple[Path, Path]:
    condition_rows = []
    target_rows = []
    values = (1.0, -1.0, 1.1, -1.1, 0.8, -0.8, 0.9, -0.9)
    for dataset_index, dataset in enumerate(DATASETS):
        for index, value in enumerate(values):
            condition = f"{dataset}_condition_{index}"
            condition_rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "archived_delta": value * (1.0 + dataset_index * 0.05),
                }
            )
            target_rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "target": f"target_{index // 2}",
                }
            )
    condition_path = root / "archive_conditions.csv"
    target_path = root / "target_map.csv"
    pd.DataFrame(condition_rows).to_csv(condition_path, index=False)
    pd.DataFrame(target_rows).to_csv(target_path, index=False)
    return condition_path, target_path


def _specification() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "stage": "PRE_OUTCOME",
        "source_archive_id": "submitted_legacy_archive_pre_revision",
        "source_selection_date": "2026-08-07",
        "planned_conditions_per_dataset": 50,
        "uncertainty_half_width_target": 0.010,
        "simulation_replicates": 100,
        "bootstrap_replicates_per_simulation": 500,
        "simulation_random_seed": 20260806,
        "conservative_scenarios": list(CONSERVATIVE_SCENARIOS),
    }


def _build(root: Path):
    condition_path, target_path = _sources(root)
    condition_hash = file_sha256(condition_path)
    target_hash = file_sha256(target_path)
    registry = build_precision_registry(
        _specification(),
        condition_table_path=condition_path,
        target_map_path=target_path,
        expected_condition_table_sha256=condition_hash,
        expected_target_map_sha256=target_hash,
    )
    return registry, condition_path, target_path, condition_hash, target_hash


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False), encoding="utf-8")


def test_precision_uses_source_recomputed_components_and_worst_scenario(tmp_path: Path) -> None:
    registry, *_ = _build(tmp_path)
    components = registry["variance_component_derivation"]["components"]
    assert all(row["target_cluster_icc_raw"] < 0 for row in components)
    assert all(row["target_cluster_icc"] == 0 for row in components)
    assert registry["conservative_scenario_contract"] == list(CONSERVATIVE_SCENARIOS)
    widths = {
        row["scenario_id"]: row["simulated_interval_width_q90"]
        for row in registry["scenario_results"]
    }
    assert registry["simulated_interval_width_q90"] == max(widths.values())
    assert registry["worst_case_scenario_id"] in {
        scenario for scenario, width in widths.items() if width == max(widths.values())
    }
    joint = next(
        row
        for row in registry["scenario_results"]
        if row["scenario_id"] == "joint_variance_125_cluster_floor_025"
    )
    assert all(row["target_cluster_icc"] == 0.25 for row in joint["effective_components"])
    assert all(
        row["condition_variance"] > source["condition_variance"]
        for row, source in zip(joint["effective_components"], components, strict=True)
    )


def test_registry_component_forgery_with_valid_outer_hash_is_rejected(tmp_path: Path) -> None:
    registry, condition_path, target_path, condition_hash, target_hash = _build(tmp_path)
    registry["variance_component_derivation"]["components"][0]["condition_variance"] = 1e-12
    registry["registry_sha256"] = canonical_sha256(
        {key: value for key, value in registry.items() if key != "registry_sha256"}
    )
    registry_path = tmp_path / "forged.json"
    _write(registry_path, registry)
    with pytest.raises(RevisionProtocolError, match="independent recomputation"):
        read_precision_registry(
            registry_path,
            condition_table_path=condition_path,
            target_map_path=target_path,
            expected_condition_table_sha256=condition_hash,
            expected_target_map_sha256=target_hash,
        )


def test_source_row_and_target_mapping_tamper_fail_caller_pinned_hash(tmp_path: Path) -> None:
    _, condition_path, target_path, condition_hash, target_hash = _build(tmp_path)
    condition_path.write_text(
        condition_path.read_text(encoding="utf-8").replace("1.0", "0.000001", 1),
        encoding="utf-8",
    )
    with pytest.raises(RevisionProtocolError, match="caller-pinned source"):
        build_precision_registry(
            _specification(),
            condition_table_path=condition_path,
            target_map_path=target_path,
            expected_condition_table_sha256=condition_hash,
            expected_target_map_sha256=target_hash,
        )

    condition_path, target_path = _sources(tmp_path)
    condition_hash = file_sha256(condition_path)
    target_hash = file_sha256(target_path)
    target_path.write_text(
        target_path.read_text(encoding="utf-8").replace("target_0", "forged_cluster", 1),
        encoding="utf-8",
    )
    with pytest.raises(RevisionProtocolError, match="caller-pinned source"):
        build_precision_registry(
            _specification(),
            condition_table_path=condition_path,
            target_map_path=target_path,
            expected_condition_table_sha256=condition_hash,
            expected_target_map_sha256=target_hash,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_selection_date", "2026-08-06"),
        ("source_archive_id", "new-outcomes"),
        ("planned_conditions_per_dataset", 49),
        ("planned_conditions_per_dataset", "50"),
        ("uncertainty_half_width_target", "0.010"),
        ("simulation_random_seed", 7),
    ],
)
def test_design_identity_date_selection_and_numeric_types_are_frozen(
    tmp_path: Path, field: str, value: object
) -> None:
    condition_path, target_path = _sources(tmp_path)
    specification = _specification()
    specification[field] = value
    with pytest.raises(RevisionProtocolError):
        build_precision_registry(
            specification,
            condition_table_path=condition_path,
            target_map_path=target_path,
            expected_condition_table_sha256=file_sha256(condition_path),
            expected_target_map_sha256=file_sha256(target_path),
        )


def test_scenario_grid_is_exact_and_type_strict(tmp_path: Path) -> None:
    condition_path, target_path = _sources(tmp_path)
    specification = _specification()
    forged = copy.deepcopy(specification["conservative_scenarios"])
    forged[-1]["variance_multiplier"] = "1.25"
    specification["conservative_scenarios"] = forged
    with pytest.raises(RevisionProtocolError, match="constants are not frozen"):
        build_precision_registry(
            specification,
            condition_table_path=condition_path,
            target_map_path=target_path,
            expected_condition_table_sha256=file_sha256(condition_path),
            expected_target_map_sha256=file_sha256(target_path),
        )


def test_target_map_requires_exact_condition_join(tmp_path: Path) -> None:
    condition_path, target_path = _sources(tmp_path)
    frame = pd.read_csv(target_path)
    frame.loc[0, "condition"] = "wrong-condition"
    frame.to_csv(target_path, index=False)
    with pytest.raises(RevisionProtocolError, match="sets differ"):
        build_precision_registry(
            _specification(),
            condition_table_path=condition_path,
            target_map_path=target_path,
            expected_condition_table_sha256=file_sha256(condition_path),
            expected_target_map_sha256=file_sha256(target_path),
        )

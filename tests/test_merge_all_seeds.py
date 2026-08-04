from __future__ import annotations

import json
from pathlib import Path

import pytest

from merge_all_seeds import (
    MergeValidationError,
    merge_directory,
    merge_sources,
    parse_split_file,
)
from turbognn_audit.hashing import sha256_file, sha256_json
from turbognn_audit.manifest import RunManifest, RunRecord


def _fold(
    condition: str,
    *,
    seed: int,
    fold_index: int,
    planned: list[str],
    pearson: float = 0.2,
    preprocessing_hash: str = "4" * 64,
) -> dict[str, object]:
    split_hash = sha256_json([planned, condition])
    fold_seed = int(sha256_json([seed, condition, split_hash])[:8], 16)
    identity = {
        "dataset": "toy",
        "input_hash": "1" * 64,
        "environment_lock_hash": "2" * 64,
        "gene_panel_hash": "5" * 64,
        "condition_panel_hash": "6" * 64,
        "preprocessing_hash": preprocessing_hash,
        "dataset_passport_registry_hash": "7" * 64,
        "dataset_passport_hash": "8" * 64,
        "matrix_contract_hash": "9" * 64,
        "control_selection_hash": "0" * 64,
        "target_mapping_hash": "b" * 64,
        "condition_eligibility_ledger_hash": "c" * 64,
        "cell_qc_policy_hash": "d" * 64,
        "response_definition_hash": "e" * 64,
        "graph_type": "self_loop_gat",
        "graph_instance": "self_loop_gat__instance_0",
        "graph_hash": "3" * 64,
        "graph_source_config_hash": "f" * 64,
        "graph_ensemble_config_hash": "1" * 64,
        "graph_contract_hash": "2" * 64,
        "model_name": "TurboGNN",
        "model_config_hash": "3" * 64,
        "input_contract_policy_hash": "4" * 64,
        "input_definition_hash": "5" * 64,
        "configuration_universe_hash": "6" * 64,
        "schedule_manifest_hash": "7" * 64,
        "schedule_content_hash": "8" * 64,
        "schedule_configuration_hash": "9" * 64,
        "split_hash": split_hash,
        "code_commit": "a" * 40,
        "seed": seed,
        "fold_seed": fold_seed,
        "condition": condition,
    }
    return {
        "schema_version": "1.0.0",
        "run_key": sha256_json(identity),
        "condition": condition,
        "fold_index": fold_index,
        "metrics": {
            "pearson_r": pearson,
            "spearman_rho": 0.1,
            "gene_top20_absolute_delta_jaccard": 0.3,
            "mse": 0.4,
        },
        "metric_states": {
            "pearson_r": "valid",
            "spearman_rho": "valid",
            "gene_top20_absolute_delta_jaccard": "valid",
            "mse": "valid",
        },
        "metadata": {
            "dataset": "toy",
            "input_hash": "1" * 64,
            "environment_lock_hash": "2" * 64,
            "gene_panel_hash": "5" * 64,
            "condition_panel_hash": "6" * 64,
            "preprocessing_hash": preprocessing_hash,
            "dataset_passport_registry_hash": "7" * 64,
            "dataset_passport_hash": "8" * 64,
            "matrix_contract_hash": "9" * 64,
            "control_selection_hash": "0" * 64,
            "target_mapping_hash": "b" * 64,
            "condition_eligibility_ledger_hash": "c" * 64,
            "cell_qc_policy_hash": "d" * 64,
            "response_definition_hash": "e" * 64,
            "graph_source_config_hash": "f" * 64,
            "graph_ensemble_config_hash": "1" * 64,
            "graph_contract_hash": "2" * 64,
            "model_name": "TurboGNN",
            "model_config_hash": "3" * 64,
            "input_contract_policy_hash": "4" * 64,
            "input_definition_hash": "5" * 64,
            "configuration_universe_hash": "6" * 64,
            "schedule_manifest_hash": "7" * 64,
            "schedule_content_hash": "8" * 64,
            "schedule_configuration_hash": "9" * 64,
            "code_commit": "a" * 40,
            "split_hash": split_hash,
            "seed": seed,
            "fold_seed": fold_seed,
            "evaluated_conditions": planned,
            "graph": {
                "graph_type": "self_loop_gat",
                "graph_instance": "self_loop_gat__instance_0",
                "edge_hash": "3" * 64,
            },
        },
    }


def _write_chunk(
    path: Path,
    seed: int,
    fold_indices: list[int],
    planned: list[str],
    *,
    preprocessing_hash: str = "4" * 64,
) -> None:
    payload = {
        "dataset": "toy",
        "graph_type": "self_loop_gat",
        "graph_instance": "self_loop_gat__instance_0",
        "seeds": {
            f"seed_{seed}": {
                "folds": [
                    _fold(
                        planned[index],
                        seed=seed,
                        fold_index=index,
                        planned=planned,
                        preprocessing_hash=preprocessing_hash,
                    )
                    for index in fold_indices
                ],
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    records = []
    for fold in payload["seeds"][f"seed_{seed}"]["folds"]:
        metadata = fold["metadata"]
        graph = metadata["graph"]
        records.append(
            RunRecord(
                dataset="toy",
                input_hash=metadata["input_hash"],
                environment_lock_hash=metadata["environment_lock_hash"],
                gene_panel_hash=metadata["gene_panel_hash"],
                condition_panel_hash=metadata["condition_panel_hash"],
                preprocessing_hash=metadata["preprocessing_hash"],
                dataset_passport_registry_hash=metadata["dataset_passport_registry_hash"],
                dataset_passport_hash=metadata["dataset_passport_hash"],
                matrix_contract_hash=metadata["matrix_contract_hash"],
                control_selection_hash=metadata["control_selection_hash"],
                target_mapping_hash=metadata["target_mapping_hash"],
                condition_eligibility_ledger_hash=metadata["condition_eligibility_ledger_hash"],
                cell_qc_policy_hash=metadata["cell_qc_policy_hash"],
                response_definition_hash=metadata["response_definition_hash"],
                graph_type=graph["graph_type"],
                graph_instance=graph["graph_instance"],
                graph_hash=graph["edge_hash"],
                graph_source_config_hash=metadata["graph_source_config_hash"],
                graph_ensemble_config_hash=metadata["graph_ensemble_config_hash"],
                graph_contract_hash=metadata["graph_contract_hash"],
                model_name=metadata["model_name"],
                model_config_hash=metadata["model_config_hash"],
                input_contract_policy_hash=metadata["input_contract_policy_hash"],
                input_definition_hash=metadata["input_definition_hash"],
                configuration_universe_hash=metadata["configuration_universe_hash"],
                schedule_manifest_hash=metadata["schedule_manifest_hash"],
                schedule_content_hash=metadata["schedule_content_hash"],
                schedule_configuration_hash=metadata["schedule_configuration_hash"],
                split_hash=metadata["split_hash"],
                code_commit=metadata["code_commit"],
                seed=seed,
                fold_seed=metadata["fold_seed"],
                condition=fold["condition"],
                status="succeeded",
                result_path=path.name,
                result_hash=sha256_file(path),
            )
        )
    RunManifest.build(records).write(path.with_name(path.stem + ".manifest.json"))


def test_seed_coded_split_is_not_mistaken_for_aggregate(tmp_path) -> None:
    planned = ["KO_A", "KO_B"]
    chunk = tmp_path / "toy__self_loop_gat__s42f0-2.json"
    _write_chunk(chunk, 42, [0, 1], planned)
    parsed = parse_split_file(chunk)
    assert parsed is not None
    assert parsed.seed == 42
    assert parsed.base == "toy__self_loop_gat"
    outputs = merge_directory(tmp_path)
    payload = json.loads(outputs[0].read_text(encoding="utf-8"))
    assert payload["overall"]["n_total_folds"] == 2
    assert payload["seeds"]["seed_42"]["n_folds"] == 2
    assert payload["graph_instance"] == "self_loop_gat__instance_0"
    assert payload["immutable_config_hash"]


def test_three_seed_runner_shaped_files_merge_to_three_seeds(tmp_path) -> None:
    planned = ["KO_A", "KO_B"]
    for seed in (42, 43, 44):
        _write_chunk(
            tmp_path / f"toy__self_loop_gat__s{seed}f0-2.json",
            seed,
            [0, 1],
            planned,
        )
    output = merge_directory(tmp_path)[0]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert set(payload["seeds"]) == {"seed_42", "seed_43", "seed_44"}
    assert payload["overall"]["n_seeds"] == 3
    assert payload["overall"]["n_total_folds"] == 6


def test_duplicate_condition_across_chunks_fails_closed(tmp_path) -> None:
    planned = ["KO_A", "KO_B", "KO_C"]
    first = tmp_path / "toy__self_loop_gat__s42f0-2.json"
    second = tmp_path / "toy__self_loop_gat__s42f1-3.json"
    _write_chunk(first, 42, [0, 1], planned)
    _write_chunk(second, 42, [1, 2], planned)
    parsed = [parse_split_file(first), parse_split_file(second)]
    with pytest.raises(MergeValidationError, match="Duplicate run_key|Duplicate fold key"):
        merge_sources([value for value in parsed if value is not None])


def test_existing_aggregate_is_never_silently_overwritten(tmp_path) -> None:
    planned = ["KO_A"]
    chunk = tmp_path / "toy__self_loop_gat__s42f0-1.json"
    _write_chunk(chunk, 42, [0], planned)
    aggregate = tmp_path / "toy__self_loop_gat.json"
    aggregate.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        merge_directory(tmp_path)


def test_seed_in_filename_must_match_payload(tmp_path) -> None:
    planned = ["KO_A"]
    chunk = tmp_path / "toy__self_loop_gat__s42f0-1.json"
    _write_chunk(chunk, 43, [0], planned)
    parsed = parse_split_file(chunk)
    assert parsed is not None
    with pytest.raises(MergeValidationError, match="disagrees with payload seeds"):
        merge_sources([parsed])


def test_fold_index_must_lie_inside_filename_interval(tmp_path) -> None:
    planned = ["KO_A", "KO_B"]
    chunk = tmp_path / "toy__self_loop_gat__s42f0-1.json"
    _write_chunk(chunk, 42, [1], planned)
    parsed = parse_split_file(chunk)
    assert parsed is not None
    with pytest.raises(MergeValidationError, match="outside filename interval"):
        merge_sources([parsed])


def test_immutable_config_cannot_change_between_disjoint_chunks(tmp_path) -> None:
    planned = ["KO_A", "KO_B"]
    first = tmp_path / "toy__self_loop_gat__s42f0-1.json"
    second = tmp_path / "toy__self_loop_gat__s42f1-2.json"
    _write_chunk(first, 42, [0], planned)
    _write_chunk(second, 42, [1], planned, preprocessing_hash="a" * 64)
    parsed = [parse_split_file(first), parse_split_file(second)]
    with pytest.raises(MergeValidationError, match="Immutable configuration differs"):
        merge_sources([value for value in parsed if value is not None])

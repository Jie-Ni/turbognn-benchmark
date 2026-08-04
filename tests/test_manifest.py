from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from turbognn_audit.manifest import ManifestValidationError, RunManifest, RunRecord
from turbognn_audit.results import build_fold_result


def _record(**overrides) -> RunRecord:
    values = {
        "dataset": "toy",
        "input_hash": "b" * 64,
        "environment_lock_hash": "c" * 64,
        "gene_panel_hash": "d" * 64,
        "condition_panel_hash": "e" * 64,
        "preprocessing_hash": "f" * 64,
        "dataset_passport_registry_hash": "1" * 64,
        "dataset_passport_hash": "2" * 64,
        "matrix_contract_hash": "3" * 64,
        "control_selection_hash": "4" * 64,
        "target_mapping_hash": "5" * 64,
        "condition_eligibility_ledger_hash": "6" * 64,
        "cell_qc_policy_hash": "7" * 64,
        "response_definition_hash": "8" * 64,
        "graph_type": "self_loop_gat",
        "graph_instance": "self_loop_gat__instance_0",
        "graph_hash": "9" * 64,
        "graph_source_config_hash": "0" * 64,
        "graph_ensemble_config_hash": "a" * 64,
        "graph_contract_hash": "b" * 64,
        "model_name": "TurboGNN",
        "model_config_hash": "c" * 64,
        "input_contract_policy_hash": "d" * 64,
        "input_definition_hash": "e" * 64,
        "configuration_universe_hash": "f" * 64,
        "schedule_manifest_hash": "1" * 64,
        "schedule_content_hash": "2" * 64,
        "schedule_configuration_hash": "3" * 64,
        "split_hash": "4" * 64,
        "code_commit": "a" * 40,
        "seed": 42,
        "fold_seed": 1234,
        "condition": "KO_A",
    }
    values.update(overrides)
    return RunRecord(**values)


def test_manifest_rejects_duplicate_complete_run_key() -> None:
    record = _record()
    with pytest.raises(ManifestValidationError, match="Duplicate run key"):
        RunManifest.build([record, record])


def test_manifest_state_counts_are_conserved() -> None:
    manifest = RunManifest.build(
        [
            _record(status="succeeded", result_path="fold_a.json", result_hash="d" * 64),
            _record(condition="KO_B", status="failed", failure_reason="OOM"),
            _record(condition="KO_C", status="skipped", skip_reason="ineligible before model"),
        ]
    )
    counts = manifest.status_counts()
    assert sum(counts.values()) == 3
    assert counts["succeeded"] == counts["failed"] == counts["skipped"] == 1
    manifest.validate(require_terminal=True)
    schema = json.loads(
        (Path(__file__).parents[1] / "schemas" / "run_manifest.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(manifest.as_dict(), schema)


def test_manifest_roundtrip_rejects_tampered_conservation(tmp_path) -> None:
    manifest = RunManifest.build([_record()])
    path = tmp_path / "manifest.json"
    manifest.write(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["planned_total"] = 2
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="planned_total"):
        RunManifest.read(path)


def test_manifest_and_fold_result_share_one_run_key_contract() -> None:
    record = _record()
    fold = build_fold_result(
        run_identity=record.identity,
        condition=record.condition,
        fold_index=0,
        gene_order=["A"],
        y_true=[1.0],
        y_pred=[0.9],
        training_mean=[0.8],
        control_profile=[0.0],
        training_loss_by_epoch=[1.0],
        metrics={"mse": 0.01},
        metadata={},
    )
    assert fold.run_key == record.run_key

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import cbac_revision.runner as runner_module
from cbac_revision.artifacts import canonical_sha256, file_sha256, read_fold_artifact
from cbac_revision.compute_reconcile import reconcile_interrupted_attempt
from cbac_revision.data import write_npz_fixture
from cbac_revision.errors import RevisionProtocolError
from cbac_revision.runner import (
    RunnerSettings,
    _audit_topology_null_ensemble,
    _direct_runtime_dependencies,
    _installed_distribution_versions,
    _load_existing_compute_attempts,
    _start_compute_attempt,
    execute_matched_benchmark,
    main,
    planned_conditional_nonoverlap_matrix,
    planned_mandatory_design_matrix,
    preflight_matched_benchmark,
    write_complete_environment_lock,
)
from cbac_revision.graph_supports import GraphMode, build_graph_support
from cbac_revision.precision import CONSERVATIVE_SCENARIOS, build_precision_registry


def _toy_protocol(tmp_path: Path) -> Path:
    source = Path(__file__).parents[1] / "protocol.yaml"
    protocol = yaml.safe_load(source.read_text(encoding="utf-8"))
    protocol["model"].update({"hidden_dim": 8, "attention_heads": 1, "layers": 1, "dropout": 0.0})
    protocol["training"].update({"maximum_epochs": 2, "validation_fraction": 0.25})
    protocol["training"]["early_stopping"].update({"patience": 1, "minimum_delta": 0.0})
    protocol["graph_composition"]["coexpression"].update({"absolute_pearson_threshold": 0.99})
    protocol["graph_supports"]["topology_null"]["rewire_multiplier"] = 1.0
    ledger_updates = {
        "hidden_dim": 8,
        "attention_heads": 1,
        "layers": 1,
        "dropout": 0.0,
        "maximum_epochs": 2,
        "validation_fraction": 0.25,
        "early_stopping_patience": 1,
        "early_stopping_minimum_delta": 0.0,
        "coexpression_abs_r_threshold": 0.99,
        "topology_rewire_multiplier": 1.0,
    }
    for record in protocol["hyperparameter_provenance"]["records"]:
        if record["parameter"] in ledger_updates:
            record["value"] = ledger_updates[record["parameter"]]
    path = tmp_path / "toy_protocol.yaml"
    path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    return path


def _toy_fixture(tmp_path: Path, *, short_last_condition: bool = False) -> Path:
    rng = np.random.default_rng(11)
    genes = [f"G{index}" for index in range(8)]
    controls = rng.poisson(lam=np.arange(2, 10), size=(16, 8)).astype(float) + 1.0
    expression = [*controls]
    labels = ["control"] * len(controls)
    classes = ["control"] * len(controls)
    pathways = ["control"] * len(controls)
    baseline = np.rint(controls.mean(axis=0))
    for index in range(6):
        replicates = 19 if short_last_condition and index == 5 else 20
        for replicate in range(replicates):
            profile = baseline.copy()
            profile[index] += 4.0 + replicate
            profile[(index + 2) % 8] += 2.0
            expression.append(profile)
            labels.append(f"G{index}")
            classes.append("class_a" if index % 2 == 0 else "class_b")
            pathways.append(f"pathway_{index % 3}")
    ppi = [(genes[index], genes[(index + 1) % 8]) for index in range(8)]
    go = [(genes[index], genes[(index + 3) % 8]) for index in range(8)]
    return write_npz_fixture(
        tmp_path / "toy.npz",
        np.asarray(expression),
        genes,
        labels,
        cell_metadata={"perturbation_class": classes, "pathway_class": pathways},
        embedded_graphs={"string_ppi": ppi, "gene_ontology": go},
    )


def _environment_lock(tmp_path: Path) -> Path:
    return write_complete_environment_lock(
        tmp_path / "requirements.lock", require_direct_dependencies=False
    )


def _precision_settings(tmp_path: Path) -> dict[str, object]:
    condition_path = tmp_path / "precision_archive_conditions.csv"
    target_path = tmp_path / "precision_target_map.csv"
    registry_path = tmp_path / "precision_design_registry.json"
    condition_rows = []
    target_rows = []
    for dataset in ("adamson", "norman", "replogle_k562", "replogle_rpe1"):
        for index, value in enumerate((-0.02, -0.01, 0.01, 0.02)):
            condition = f"{dataset}_archive_{index}"
            condition_rows.append(
                {"dataset": dataset, "condition": condition, "archived_delta": value}
            )
            target_rows.append(
                {"dataset": dataset, "condition": condition, "target": f"T{index // 2}"}
            )
    pd.DataFrame(condition_rows).to_csv(condition_path, index=False)
    pd.DataFrame(target_rows).to_csv(target_path, index=False)
    condition_hash = file_sha256(condition_path)
    target_hash = file_sha256(target_path)
    registry = build_precision_registry(
        {
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
        },
        condition_table_path=condition_path,
        target_map_path=target_path,
        expected_condition_table_sha256=condition_hash,
        expected_target_map_sha256=target_hash,
    )
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "precision_registry_path": registry_path,
        "precision_condition_table_path": condition_path,
        "precision_condition_table_sha256": condition_hash,
        "precision_target_map_path": target_path,
        "precision_target_map_sha256": target_hash,
    }


def test_default_cli_is_dry_run_and_builds_all_10_rewires(tmp_path: Path) -> None:
    protocol = _toy_protocol(tmp_path)
    fixture = _toy_fixture(tmp_path)
    output = tmp_path / "dry_run"

    arms = [
        "dense",
        "self_loop",
        "string_go",
        *[f"string_go_rewire_{i:02d}" for i in range(1, 11)],
    ]
    arguments = [
        "--protocol",
        str(protocol),
        "--dataset",
        "adamson",
        "--data",
        str(fixture),
        "--output",
        str(output),
        "--hvg",
        "8",
        "--panel-size",
        "3",
        "--seed",
        "42",
        "--fixture-mode",
    ]
    for arm in arms:
        arguments.extend(["--arm", arm])
    status = main(arguments)

    assert status == 0
    assert (output / "preflight" / "preflight_summary.json").exists()
    assert not (output / "artifacts").exists()
    manifest = pd.read_csv(output / "preflight" / "graph_manifest.csv")
    arms = set(manifest["arm"])
    assert len({arm for arm in arms if arm.startswith("string_go_rewire_")}) == 10
    assert {
        "mean_clustering_coefficient",
        "degree_assortativity_status",
        "modularity_status",
        "mapping_edge_coverage",
        "component_partition_hash",
    } <= set(manifest.columns)
    assert not pd.read_csv(output / "preflight" / "graph_edge_overlap.csv").empty
    ensemble_audit = json.loads(
        (output / "preflight" / "topology_null_ensemble_audit.json").read_text(encoding="utf-8")
    )
    assert ensemble_audit["status"] == "TEST_FIXTURE_ONLY_NOT_ENFORCED"


def test_cpu_toy_execution_writes_lossless_artifacts_with_shared_initialization(
    tmp_path: Path,
) -> None:
    protocol = _toy_protocol(tmp_path)
    fixture = _toy_fixture(tmp_path)
    output = tmp_path / "execute"
    settings = RunnerSettings(
        protocol_path=protocol,
        dataset_name="adamson",
        data_path=fixture,
        output_dir=output,
        hvg=8,
        panel_size=3,
        seeds=(42,),
        arms=("dense", "self_loop", "string_go", "string_go_rewire_01"),
        epochs_override=2,
        device="cpu",
        fixture_mode=True,
    )

    bundle = preflight_matched_benchmark(settings)
    artifacts = execute_matched_benchmark(bundle)
    payloads = [read_fold_artifact(path) for path in artifacts]

    assert bundle.can_execute
    assert len(artifacts) == 12
    assert len({payload["initialization_hash"] for payload in payloads}) == 1
    assert {payload["identity"]["arm"] for payload in payloads} == set(settings.arms)
    assert all(len(payload["train_loss"]) == 2 for payload in payloads)
    assert all(len(payload["validation_loss"]) == 2 for payload in payloads)
    assert all(payload["learning_rate"][1] < payload["learning_rate"][0] for payload in payloads)
    assert all(len(payload["y_true"]) == 8 for payload in payloads)
    assert all(len(payload["y_pred"]) == 8 for payload in payloads)
    assert all(
        payload["vector_space"] == "control_fitted_standardized_delta_expression"
        for payload in payloads
    )
    assert all(
        payload["runtime"]["peak_memory_status"] == "NOT_APPLICABLE_CPU" for payload in payloads
    )
    assert all(payload["checkpoint_hash"] for payload in payloads)
    first_condition = payloads[0]["identity"]["condition"]
    expected_true = bundle.condition_profiles[first_condition] - bundle.control_profile
    np.testing.assert_allclose(payloads[0]["y_true"], expected_true, atol=1e-6)
    assert int(bundle.hvg_manifest["forced_selected_panel_target"].sum()) == 3


def test_forced_hvg_targets_are_scoped_to_the_panel_being_executed(tmp_path: Path) -> None:
    protocol = _toy_protocol(tmp_path)
    fixture = _toy_fixture(tmp_path)
    primary = preflight_matched_benchmark(
        RunnerSettings(
            protocol_path=protocol,
            dataset_name="adamson",
            data_path=fixture,
            output_dir=tmp_path / "primary_scope",
            hvg=3,
            panel_size=3,
            seeds=(42,),
            arms=("dense",),
            fixture_mode=True,
        )
    )
    sensitivity = preflight_matched_benchmark(
        RunnerSettings(
            protocol_path=protocol,
            dataset_name="adamson",
            data_path=fixture,
            output_dir=tmp_path / "sensitivity_scope",
            hvg=3,
            analysis_block="topology_primary",
            panel_name="sensitivity",
            panel_size=3,
            seeds=(42,),
            arms=("dense",),
            fixture_mode=True,
        )
    )

    primary_forced = set(
        primary.hvg_manifest.loc[primary.hvg_manifest["forced_selected_panel_target"], "gene"]
    )
    sensitivity_forced = set(
        sensitivity.hvg_manifest.loc[
            sensitivity.hvg_manifest["forced_selected_panel_target"], "gene"
        ]
    )
    assert primary_forced == set(primary.selected_conditions)
    assert sensitivity_forced == set(sensitivity.selected_conditions)
    assert primary_forced.isdisjoint(sensitivity_forced)


def test_authoritative_mandatory_design_has_10800_fits_and_13_primary_arms(
    tmp_path: Path,
) -> None:
    protocol = yaml.safe_load(_toy_protocol(tmp_path).read_text(encoding="utf-8"))
    matrix = planned_mandatory_design_matrix(protocol)

    assert int(matrix["model_fits"].sum()) == 10_800
    primary = matrix[matrix["analysis_block"] == "topology_primary"]
    assert primary.groupby("dataset")["arm"].nunique().eq(13).all()
    assert len(primary) * 150 == 7_800
    scale = matrix[matrix["analysis_block"] == "scale_extension"]
    assert int(scale["model_fits"].sum()) == 2_400
    mixed = matrix[matrix["analysis_block"] == "mixed_support_sensitivity"]
    assert int(mixed["model_fits"].sum()) == 600
    assert scale.loc[scale["hvg"] == 200, "dense_200_reused"].all()
    conditional = planned_conditional_nonoverlap_matrix(protocol)
    assert int(conditional["model_fits"].sum()) == 1_200


def test_topology_null_gate_blocks_duplicate_hashes_and_low_rewire_coverage(
    tmp_path: Path,
) -> None:
    protocol = yaml.safe_load(_toy_protocol(tmp_path).read_text(encoding="utf-8"))
    cycle = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]])
    support = build_graph_support(
        GraphMode.DEGREE_PRESERVING_REWIRE,
        6,
        curated_edges=cycle,
        rewire_seed=1001,
        rewire_multiplier=2.0,
    )
    high_coverage = replace(support, swapped_edge_fraction=0.9)
    duplicates = {f"string_go_rewire_{index:02d}": high_coverage for index in range(1, 11)}
    _, duplicate_blockers = _audit_topology_null_ensemble(
        duplicates, protocol, required=True, enforce=True
    )
    _, coverage_blockers = _audit_topology_null_ensemble(
        {"string_go_rewire_01": replace(support, swapped_edge_fraction=0.79)},
        protocol,
        required=False,
        enforce=True,
    )

    assert any(
        blocker["reason_code"] == "DUPLICATE_TOPOLOGY_NULL_SUPPORT_HASH"
        for blocker in duplicate_blockers
    )
    assert any(
        blocker["reason_code"] == "REWIRE_SWAPPED_EDGE_FRACTION_BELOW_MINIMUM"
        for blocker in coverage_blockers
    )


def test_condition_with_fewer_than_20_cells_is_reason_coded_ineligible(
    tmp_path: Path,
) -> None:
    bundle = preflight_matched_benchmark(
        RunnerSettings(
            protocol_path=_toy_protocol(tmp_path),
            dataset_name="adamson",
            data_path=_toy_fixture(tmp_path, short_last_condition=True),
            output_dir=tmp_path / "short_condition",
            hvg=8,
            panel_size=2,
            seeds=(42,),
            arms=("dense",),
            fixture_mode=True,
        )
    )
    row = bundle.target_audit[bundle.target_audit["condition"] == "G5"].iloc[0]

    assert row["status"] == "INELIGIBLE"
    assert row["reason_code"] == "INSUFFICIENT_CONDITION_CELLS"


def test_primary_is_not_blocked_by_short_conditional_sensitivity_panel(
    tmp_path: Path,
) -> None:
    bundle = preflight_matched_benchmark(
        RunnerSettings(
            protocol_path=_toy_protocol(tmp_path),
            dataset_name="adamson",
            data_path=_toy_fixture(tmp_path, short_last_condition=True),
            output_dir=tmp_path / "conditional_shortfall",
            hvg=8,
            panel_size=3,
            seeds=(42,),
            arms=("dense",),
            fixture_mode=True,
        )
    )

    assert len(bundle.panels.primary) == 3
    assert len(bundle.panels.sensitivity) == 2
    assert bundle.can_execute
    assert bundle.summary()["conditional_sensitivity_feasible"] is False


def test_forced_selected_panel_target_union_cannot_silently_exceed_hvg(tmp_path: Path) -> None:
    output = tmp_path / "overflow"

    with pytest.raises(RevisionProtocolError, match="HVG_FORCED_RETENTION_OVERFLOW"):
        preflight_matched_benchmark(
            RunnerSettings(
                protocol_path=_toy_protocol(tmp_path),
                dataset_name="adamson",
                data_path=_toy_fixture(tmp_path),
                output_dir=output,
                hvg=2,
                panel_size=3,
                seeds=(42,),
                arms=("dense",),
                fixture_mode=True,
            )
        )
    assert (output / "preflight" / "hvg_forced_retention_overflow.csv").exists()


def test_production_go_provenance_requires_versioned_hashed_inputs(tmp_path: Path) -> None:
    bundle = preflight_matched_benchmark(
        RunnerSettings(
            protocol_path=_toy_protocol(tmp_path),
            dataset_name="adamson",
            data_path=_toy_fixture(tmp_path),
            output_dir=tmp_path / "missing_go_provenance",
            hvg=8,
            panel_size=3,
            seeds=(42,),
            arms=("string_go",),
            environment_lock_path=_environment_lock(tmp_path),
            **_precision_settings(tmp_path),
        )
    )

    assert any(
        blocker["reason_code"] == "GO_PROVENANCE_INPUT_MISSING" for blocker in bundle.blockers
    )


def test_arbitrary_edge_csv_without_derivation_sidecars_is_blocked(tmp_path: Path) -> None:
    ppi = tmp_path / "unbound_string.csv"
    go_edges = tmp_path / "unbound_go.csv"
    gaf = tmp_path / "goa_human.gaf"
    obo = tmp_path / "go-basic.obo"
    metadata = tmp_path / "go_release.json"
    ppi.write_text("source,target,combined_score\nG0,G1,401\n", encoding="utf-8")
    go_edges.write_text("source,target\nG0,G2\n", encoding="utf-8")
    gaf.write_text("!gaf-version: 2.2\n", encoding="utf-8")
    obo.write_text("format-version: 1.2\n", encoding="utf-8")
    metadata.write_text(
        json.dumps(
            {
                "release": "declared-but-unbound",
                "gaf_sha256": file_sha256(gaf),
                "obo_sha256": file_sha256(obo),
            }
        ),
        encoding="utf-8",
    )
    bundle = preflight_matched_benchmark(
        RunnerSettings(
            protocol_path=_toy_protocol(tmp_path),
            dataset_name="adamson",
            data_path=_toy_fixture(tmp_path),
            output_dir=tmp_path / "unbound_edges",
            hvg=8,
            analysis_block="scale_extension",
            panel_size=3,
            seeds=(42,),
            arms=("string_go",),
            environment_lock_path=_environment_lock(tmp_path),
            **_precision_settings(tmp_path),
            graph_paths={
                "string_ppi": ppi,
                "gene_ontology": go_edges,
                "go_gaf": gaf,
                "go_obo": obo,
                "go_release_metadata": metadata,
            },
        )
    )
    reason_codes = {blocker["reason_code"] for blocker in bundle.blockers}

    assert "STRING_DERIVATION_INPUT_MISSING" in reason_codes
    assert "GO_PROVENANCE_INPUT_MISSING" in reason_codes
    assert not bundle.can_execute


def test_production_go_provenance_verifies_release_and_checksums(tmp_path: Path) -> None:
    ppi = tmp_path / "string.csv"
    string_raw = tmp_path / "string_raw.txt"
    string_identifier_map = tmp_path / "string_identifier_map.tsv"
    string_builder = tmp_path / "build_string.py"
    string_manifest = tmp_path / "string_derivation.json"
    go_edges = tmp_path / "go_edges.csv"
    gaf = tmp_path / "goa_human.gaf"
    obo = tmp_path / "go-basic.obo"
    go_builder = tmp_path / "build_go.py"
    go_manifest = tmp_path / "go_derivation.json"
    metadata = tmp_path / "go_release.json"
    ppi.write_text(
        "source,target,combined_score\nG0,G1,401\nG1,G2,700\nG2,G3,999\n",
        encoding="utf-8",
    )
    go_edges.write_text("source,target\nG0,G3\nG1,G4\nG2,G5\n", encoding="utf-8")
    string_raw.write_text("toy STRING v12.0 source fixture\n", encoding="utf-8")
    string_identifier_map.write_text(
        "string_id\tgene_symbol\n9606.ENSP0\tG0\n",
        encoding="utf-8",
    )
    string_builder.write_text("# frozen toy STRING builder\n", encoding="utf-8")
    gaf.write_text("!gaf-version: 2.2\ntoy\n", encoding="utf-8")
    obo.write_text("format-version: 1.2\n[Term]\n", encoding="utf-8")
    go_builder.write_text("# frozen toy GO builder\n", encoding="utf-8")
    metadata.write_text(
        json.dumps(
            {
                "release": "toy-release-for-test",
                "gaf_sha256": file_sha256(gaf),
                "obo_sha256": file_sha256(obo),
            }
        ),
        encoding="utf-8",
    )
    string_manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source_name": "STRING",
                "source_release": "STRING_v12.0",
                "source_url": "https://string-db.org/",
                "edge_file_sha256": file_sha256(ppi),
                "builder": {
                    "name": "toy-string-builder",
                    "version": "1.0",
                    "code_sha256": file_sha256(string_builder),
                },
                "input_files": {
                    "string_raw": file_sha256(string_raw),
                    "identifier_map": file_sha256(string_identifier_map),
                },
                "identifier_mapping": {"rule": "exact_symbol", "coverage": 1.0},
                "policy": {
                    "species_taxon": 9606,
                    "score_field": "combined_score",
                    "threshold_operator": ">",
                    "threshold_value": 400,
                },
            }
        ),
        encoding="utf-8",
    )
    go_manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source_name": "Gene Ontology",
                "source_release": "toy-release-for-test",
                "source_url": "https://geneontology.org/",
                "edge_file_sha256": file_sha256(go_edges),
                "builder": {
                    "name": "toy-go-builder",
                    "version": "1.0",
                    "code_sha256": file_sha256(go_builder),
                },
                "input_files": {
                    "go_gaf": file_sha256(gaf),
                    "go_obo": file_sha256(obo),
                },
                "identifier_mapping": {"rule": "exact_symbol", "coverage": 1.0},
                "policy": {
                    "namespace": "biological_process",
                    "edge_rule": "shared_annotation",
                    "qualifier_policy": "exclude_NOT",
                    "evidence_code_policy": "declared_all_codes_for_toy_test",
                    "ancestor_propagation_policy": "none",
                    "depth_policy": "direct_annotations_only",
                },
            }
        ),
        encoding="utf-8",
    )
    bundle = preflight_matched_benchmark(
        RunnerSettings(
            protocol_path=_toy_protocol(tmp_path),
            dataset_name="adamson",
            data_path=_toy_fixture(tmp_path),
            output_dir=tmp_path / "verified_go_provenance",
            hvg=8,
            analysis_block="scale_extension",
            panel_size=3,
            seeds=(42,),
            arms=("string_go",),
            environment_lock_path=_environment_lock(tmp_path),
            **_precision_settings(tmp_path),
            graph_paths={
                "string_ppi": ppi,
                "string_raw": string_raw,
                "string_identifier_map": string_identifier_map,
                "string_builder": string_builder,
                "string_derivation_manifest": string_manifest,
                "gene_ontology": go_edges,
                "go_gaf": gaf,
                "go_obo": obo,
                "go_builder": go_builder,
                "go_derivation_manifest": go_manifest,
                "go_release_metadata": metadata,
            },
        )
    )

    assert bundle.can_execute
    assert set(bundle.supports) == {"string_go"}
    assert set(bundle.graph_provenance["status"]) == {
        "VERIFIED_LOCAL_INPUT",
        "VERIFIED_DERIVED_EDGE_INPUT",
        "VERIFIED_DERIVATION_CONTRACT",
    }


def test_production_environment_lock_must_match_active_versions(tmp_path: Path) -> None:
    lock = _environment_lock(tmp_path)
    text = lock.read_text(encoding="utf-8")
    lock.write_text(text.replace("numpy==", "numpy==0.invalid#", 1), encoding="utf-8")

    with pytest.raises(RevisionProtocolError, match="ENVIRONMENT_LOCK_VERSION_MISMATCH"):
        preflight_matched_benchmark(
            RunnerSettings(
                protocol_path=_toy_protocol(tmp_path),
                dataset_name="adamson",
                data_path=_toy_fixture(tmp_path),
                output_dir=tmp_path / "bad_environment",
                hvg=8,
                panel_size=3,
                seeds=(42,),
                arms=("dense",),
                environment_lock_path=lock,
                **_precision_settings(tmp_path),
            )
        )


def test_complete_environment_lock_covers_h5py_and_every_active_distribution(
    tmp_path: Path,
) -> None:
    lock = _environment_lock(tmp_path)
    package_lines = [
        line
        for line in lock.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]

    assert "h5py" in _direct_runtime_dependencies()
    assert len(package_lines) == len(_installed_distribution_versions())
    assert any(line.startswith("h5py==") for line in package_lines) is (
        "h5py" in _installed_distribution_versions()
    )


def test_environment_lock_rejects_omitted_h5py_and_added_compiled_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulated_versions = _installed_distribution_versions()
    simulated_versions.update({"anndata": "0.test", "h5py": "0.test"})
    monkeypatch.setattr(
        runner_module, "_installed_distribution_versions", lambda: simulated_versions
    )
    lock = write_complete_environment_lock(tmp_path / "requirements.lock")
    original = lock.read_text(encoding="utf-8")
    lock.write_text(
        "\n".join(line for line in original.splitlines() if not line.startswith("h5py==")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RevisionProtocolError, match="ENVIRONMENT_LOCK_PACKAGE_SET_MISMATCH"):
        preflight_matched_benchmark(
            RunnerSettings(
                protocol_path=_toy_protocol(tmp_path),
                dataset_name="adamson",
                data_path=_toy_fixture(tmp_path),
                output_dir=tmp_path / "missing_h5py",
                hvg=8,
                panel_size=3,
                seeds=(42,),
                arms=("dense",),
                environment_lock_path=lock,
                **_precision_settings(tmp_path),
            )
        )
    lock.write_text(original + "invented-compiled-extension==1.0\n", encoding="utf-8")
    with pytest.raises(RevisionProtocolError, match="ENVIRONMENT_LOCK_PACKAGE_SET_MISMATCH"):
        preflight_matched_benchmark(
            RunnerSettings(
                protocol_path=_toy_protocol(tmp_path),
                dataset_name="adamson",
                data_path=_toy_fixture(tmp_path),
                output_dir=tmp_path / "extra_compiled",
                hvg=8,
                panel_size=3,
                seeds=(42,),
                arms=("dense",),
                environment_lock_path=lock,
                **_precision_settings(tmp_path),
            )
        )


def test_unfinalized_started_attempt_blocks_silent_resume(tmp_path: Path) -> None:
    bundle = preflight_matched_benchmark(
        RunnerSettings(
            protocol_path=_toy_protocol(tmp_path),
            dataset_name="adamson",
            data_path=_toy_fixture(tmp_path),
            output_dir=tmp_path / "interrupted",
            hvg=8,
            panel_size=3,
            seeds=(42,),
            arms=("dense",),
            fixture_mode=True,
        )
    )
    hardware = {
        "device": "cpu",
        "host": "test-host",
        "accelerator_model": "NOT_APPLICABLE_CPU_EXECUTION",
        "accelerator_count": 0,
        "cpu_model": "test-cpu",
        "logical_cpu_count": 4,
        "host_ram_bytes": 8_000_000_000,
    }
    attempts: list[dict] = []
    _start_compute_attempt(
        bundle,
        attempts,
        "dense",
        bundle.selected_conditions[0],
        42,
        hardware,
        1,
        datetime.now(timezone.utc),
    )

    with pytest.raises(RevisionProtocolError, match="COMPUTE_UNRECONCILED_INTERRUPTED_ATTEMPT"):
        execute_matched_benchmark(bundle)


def test_interrupted_attempt_reconciles_then_retries_to_one_final_success(tmp_path: Path) -> None:
    bundle = preflight_matched_benchmark(
        RunnerSettings(
            protocol_path=_toy_protocol(tmp_path),
            dataset_name="adamson",
            data_path=_toy_fixture(tmp_path),
            output_dir=tmp_path / "reconciled_retry",
            hvg=8,
            panel_size=3,
            seeds=(42,),
            arms=("dense",),
            epochs_override=2,
            fixture_mode=True,
        )
    )
    hardware = {
        "device": "cpu",
        "host": "test-host",
        "accelerator_model": "NOT_APPLICABLE_CPU_EXECUTION",
        "accelerator_count": 0,
        "cpu_model": "test-cpu",
        "logical_cpu_count": 4,
        "host_ram_bytes": 8_000_000_000,
    }
    attempts: list[dict] = []
    started = datetime.now(timezone.utc).replace(microsecond=0)
    condition = bundle.selected_conditions[0]
    _start_compute_attempt(
        bundle,
        attempts,
        "dense",
        condition,
        42,
        hardware,
        1,
        started,
    )
    started_text = started.isoformat().replace("+00:00", "Z")
    finished_text = (started + pd.Timedelta(seconds=17)).isoformat().replace("+00:00", "Z")
    source = bundle.settings.output_dir / "scheduler_record.json"
    source_payload = {
        "scheduler_job_id": attempts[0]["scheduler_job_id"],
        "scheduler_state": "PREEMPTED",
        "started_at_utc": started_text,
        "finished_at_utc": finished_text,
        "elapsed_allocation_seconds": 17.0,
        "device_count": 0,
        "exit_or_preemption_reason": "SCHEDULER_PREEMPTION",
    }
    source.write_text(json.dumps(source_payload), encoding="utf-8")

    reconcile_interrupted_attempt(
        bundle.settings.output_dir / "measured_compute_registry.json",
        run_id=bundle.run_id,
        dataset="adamson",
        hvg=8,
        panel="primary",
        arm="dense",
        condition=condition,
        seed=42,
        attempt_index=1,
        scheduler_job_id=attempts[0]["scheduler_job_id"],
        scheduler_state="PREEMPTED",
        started_at_utc=started_text,
        finished_at_utc=finished_text,
        elapsed_allocation_seconds=17.0,
        device_count=0,
        exit_or_preemption_reason="SCHEDULER_PREEMPTION",
        source_record_path=source,
        source_record_sha256=file_sha256(source),
    )
    execute_matched_benchmark(bundle)
    rows = _load_existing_compute_attempts(bundle)
    matched = [
        row
        for row in rows
        if row["arm"] == "dense" and row["condition"] == condition and row["seed"] == 42
    ]

    assert [row["attempt_index"] for row in matched] == [1, 2]
    assert matched[0]["execution_status"] == "FAILED_RECONCILED_PREEMPTED"
    assert matched[0]["scheduler_allocation_seconds"] == 17.0
    assert matched[0]["unattributed_interrupted_allocation_seconds"] == 17.0
    assert [row["execution_status"] for row in matched].count("SUCCESS") == 1
    assert matched[-1]["execution_status"] == "SUCCESS"


def test_resume_revalidates_retained_checkpoint_before_skipping_success(tmp_path: Path) -> None:
    bundle = preflight_matched_benchmark(
        RunnerSettings(
            protocol_path=_toy_protocol(tmp_path),
            dataset_name="adamson",
            data_path=_toy_fixture(tmp_path),
            output_dir=tmp_path / "checkpoint_revalidation",
            hvg=8,
            panel_size=3,
            seeds=(42,),
            arms=("dense",),
            epochs_override=2,
            fixture_mode=True,
        )
    )
    artifacts = execute_matched_benchmark(bundle)
    target_artifact = artifacts[0]
    payload = read_fold_artifact(target_artifact)
    checkpoint = (target_artifact.parent / payload["checkpoint_path"]).resolve()
    checkpoint.write_bytes(b"changed retained checkpoint")
    identity = payload["identity"]

    execute_matched_benchmark(bundle)
    rows = _load_existing_compute_attempts(bundle)
    matched = [
        row
        for row in rows
        if row["arm"] == identity["arm"]
        and row["condition"] == identity["condition"]
        and row["seed"] == identity["seed"]
    ]

    assert [row["attempt_index"] for row in matched] == [1, 2]
    assert matched[0]["execution_status"] == "FAILED_RETAINED_ARTIFACT_INVALID"
    assert matched[0]["artifact_validation_status"] == "INVALIDATED_BEFORE_RESUME_SKIP"
    assert matched[1]["execution_status"] == "SUCCESS"


def test_newer_orphan_ledger_blocks_resume(tmp_path: Path) -> None:
    bundle = preflight_matched_benchmark(
        RunnerSettings(
            protocol_path=_toy_protocol(tmp_path),
            dataset_name="adamson",
            data_path=_toy_fixture(tmp_path),
            output_dir=tmp_path / "orphan",
            hvg=8,
            panel_size=3,
            seeds=(42,),
            arms=("dense",),
            fixture_mode=True,
        )
    )
    hardware = {
        "device": "cpu",
        "host": "test-host",
        "accelerator_model": "NOT_APPLICABLE_CPU_EXECUTION",
        "accelerator_count": 0,
        "cpu_model": "test-cpu",
        "logical_cpu_count": 4,
        "host_ram_bytes": 8_000_000_000,
    }
    attempts: list[dict] = []
    _start_compute_attempt(
        bundle,
        attempts,
        "dense",
        bundle.selected_conditions[0],
        42,
        hardware,
        1,
        datetime.now(timezone.utc),
    )
    orphan_rows = [*attempts, {**attempts[0], "condition": bundle.selected_conditions[1]}]
    orphan_hash = canonical_sha256(orphan_rows)
    orphan = bundle.settings.output_dir / f"compute_attempt_ledger.{orphan_hash}.json"
    orphan.write_text(json.dumps(orphan_rows), encoding="utf-8")

    with pytest.raises(RevisionProtocolError, match="COMPUTE_ORPHAN_LEDGER_DETECTED"):
        _load_existing_compute_attempts(bundle)

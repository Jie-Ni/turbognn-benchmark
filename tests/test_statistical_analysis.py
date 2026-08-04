from __future__ import annotations

import json
import math
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

import statistical_analysis as statistical_analysis_module
from statistical_analysis import (
    CANONICAL_DATASETS,
    FISHER_CLIP_EPSILON,
    FROZEN_SCALES,
    LEGACY_COMPARATOR,
    PRIMARY_COMPARATOR,
    PRIMARY_TREATMENT,
    REQUIRED_SEEDS,
    StatisticalValidationError,
    aggregate_seeds_within_condition,
    analyze_fold_data,
    analyze_primary_estimand,
    canonical_graph_label,
    comparison_specs,
    finalize_tds09_decision,
    load_fold_data,
    pair_condition_values,
    validate_raw_fold_data,
)
from turbognn_audit.hashing import canonical_json, sha256_file, sha256_json
from turbognn_audit.inference import MonteCarloTest
from turbognn_audit.results import build_fold_result


def _raw_row(
    graph_type: str,
    condition: str,
    seed: int,
    value: float,
    *,
    dataset: str = "toy",
    hvg: int = 200,
    planned: tuple[str, ...] = ("A_only", "B", "C", "D_only"),
    graph_instance: str | None = None,
    metric_state: str = "valid",
) -> dict[str, object]:
    split = sha256_json([dataset, condition])
    fold_seed = int(sha256_json([seed, condition, split])[:8], 16)
    instance = graph_instance or f"{graph_type}__instance_0"
    identity = {
        "dataset": dataset,
        "hvg": hvg,
        "graph_type": graph_type,
        "graph_instance": instance,
        "seed": seed,
        "condition": condition,
        "split": split,
    }
    gene_order = [f"gene_{hvg}"]
    return {
        "dataset": dataset,
        "hvg": hvg,
        "input_hash": "1" * 64,
        "dataset_passport_hash": sha256_json([dataset, "passport"]),
        "dataset_passport_registry_hash": "6" * 64,
        "matrix_contract_hash": sha256_json([dataset, "matrix_contract"]),
        "environment_lock_hash": "2" * 64,
        "gene_panel_hash": f"genes_{hvg}",
        "condition_panel_hash": f"conditions_{dataset}",
        "preprocessing_hash": f"preprocessing_{dataset}_{hvg}",
        "control_selection_hash": f"controls_{dataset}",
        "condition_eligibility_ledger_hash": sha256_json([dataset, "eligibility"]),
        "cell_qc_policy_hash": sha256_json([dataset, "cell_qc"]),
        "response_definition_hash": sha256_json([dataset, "response_definition"]),
        "target_mapping_hash": sha256_json([dataset, "target_mapping"]),
        "gene_order_hash": sha256_json(gene_order),
        "gene_order_json": canonical_json(gene_order),
        "y_true_hash": sha256_json([dataset, hvg, condition, "truth"]),
        "training_mean_hash": sha256_json([dataset, hvg, condition, "training_mean"]),
        "control_profile_hash": sha256_json([dataset, hvg, "control"]),
        "planned_condition_list_hash": sha256_json(list(planned)),
        "planned_conditions_json": canonical_json(list(planned)),
        "graph_type": graph_type,
        "graph_type_raw": graph_type,
        "graph_instance": instance,
        "graph_hash": sha256_json([dataset, hvg, instance]),
        "model_name": "TurboGNN",
        "model_config_hash": "model_config",
        "initialization_hash": sha256_json([dataset, hvg, seed, "initialization"]),
        "target_mask_hash": sha256_json([dataset, hvg, condition, "target_mask"]),
        "initial_perturbation_context_hash": sha256_json(
            [dataset, hvg, seed, condition, "initial_context"]
        ),
        "initial_raw_feature_hash": sha256_json(
            [dataset, hvg, seed, condition, "initial_raw_features"]
        ),
        "initial_raw_feature_artifact": "feature_artifacts/synthetic.npy",
        "initial_raw_feature_artifact_sha256": "3" * 64,
        "baseline_feature_hash": sha256_json([dataset, hvg, "baseline"]),
        "input_contract_policy_hash": "5" * 64,
        "input_definition_hash": sha256_json([dataset, hvg, "input_definition"]),
        "input_instance_hash": sha256_json([dataset, hvg, seed, condition, "input_instance"]),
        "input_contract_hash": sha256_json([dataset, hvg, seed, condition, "input_instance"]),
        "code_commit": "a" * 40,
        "split_hash": split,
        "seed": seed,
        "fold_seed": fold_seed,
        "condition": condition,
        "fold_index": planned.index(condition),
        "run_key": sha256_json(identity),
        "value": value if metric_state == "valid" else None,
        "metric_state": metric_state,
        "metric": "pearson_r",
        "source_file": f"{graph_type}.json",
    }


def _unbalanced_raw_table() -> pd.DataFrame:
    rows = [
        _raw_row(PRIMARY_TREATMENT, "A_only", 42, 0.9),
        _raw_row(PRIMARY_TREATMENT, "B", 42, 0.1),
        _raw_row(PRIMARY_TREATMENT, "B", 43, 0.3),
        _raw_row(PRIMARY_TREATMENT, "C", 42, 0.5),
        _raw_row(PRIMARY_TREATMENT, "C", 43, 0.7),
        _raw_row(PRIMARY_COMPARATOR, "B", 42, 0.05),
        _raw_row(PRIMARY_COMPARATOR, "B", 43, 0.15),
        _raw_row(PRIMARY_COMPARATOR, "C", 42, 0.4),
        _raw_row(PRIMARY_COMPARATOR, "C", 43, 0.4),
        _raw_row(PRIMARY_COMPARATOR, "D_only", 42, -0.9),
    ]
    return pd.DataFrame(rows).sample(frac=1.0, random_state=7).reset_index(drop=True)


def test_seed_aggregation_and_panel_based_coverage() -> None:
    condition_level, paired, summary = analyze_fold_data(
        _unbalanced_raw_table(),
        bootstrap_replicates=100,
        bootstrap_seed=1,
    )
    union_b = condition_level.loc[
        (condition_level["graph_type"] == PRIMARY_TREATMENT) & (condition_level["condition"] == "B")
    ].iloc[0]
    assert union_b["value"] == pytest.approx(0.2)
    assert union_b["seed_set"] == "[42,43]"
    primary_pairs = paired.loc[
        (paired["treatment"] == PRIMARY_TREATMENT) & (paired["comparator"] == PRIMARY_COMPARATOR)
    ]
    assert primary_pairs["condition"].tolist() == ["B", "C"]
    result = summary.loc[summary["comparison_class"] == "singular_primary"].iloc[0]
    assert result["planned_condition_n"] == 4
    assert result["unique_condition_n"] == 2
    assert result["coverage"] == 0.5
    assert result["treatment_coverage"] == 0.75
    assert result["comparator_coverage"] == 0.75
    assert result["coverage_basis"] == "frozen_evaluated_conditions_panel"


def test_three_training_seeds_form_one_condition_scale_arm_record() -> None:
    frame = pd.DataFrame(
        [
            _raw_row(PRIMARY_TREATMENT, "B", seed, value)
            for seed, value in zip(REQUIRED_SEEDS, (0.1, 0.2, 0.3), strict=True)
        ]
    )
    aggregated = aggregate_seeds_within_condition(frame)
    assert len(aggregated) == 1
    assert aggregated.iloc[0]["seed_run_n"] == 3
    assert aggregated.iloc[0]["seed_set"] == "[42,43,44]"


def test_pairing_rejects_mismatched_seed_sets() -> None:
    frame = _unbalanced_raw_table()
    frame = frame.loc[
        ~(
            (frame["graph_type"] == PRIMARY_COMPARATOR)
            & (frame["condition"] == "B")
            & (frame["seed"] == 43)
        )
    ]
    condition_level = aggregate_seeds_within_condition(frame)
    spec = next(
        spec
        for spec in comparison_specs(frame["graph_type"])
        if spec.treatment == PRIMARY_TREATMENT
    )
    with pytest.raises(StatisticalValidationError, match="seed sets differ"):
        pair_condition_values(condition_level, spec)


def test_graph_instances_are_retained_instead_of_averaged() -> None:
    rows: list[dict[str, object]] = []
    for instance, value in (("string_ppi__seed_1", 0.2), ("string_ppi__seed_2", 0.4)):
        rows.append(
            _raw_row(
                "string_ppi",
                "B",
                42,
                value,
                graph_instance=instance,
            )
        )
    rows.append(_raw_row(PRIMARY_COMPARATOR, "B", 42, 0.1))
    condition_level = aggregate_seeds_within_condition(pd.DataFrame(rows))
    spec = next(spec for spec in comparison_specs(["string_ppi", PRIMARY_COMPARATOR]))
    paired = pair_condition_values(condition_level, spec)
    assert set(paired["treatment_graph_instance"]) == {
        "string_ppi__seed_1",
        "string_ppi__seed_2",
    }
    assert len(paired) == 2


def test_invalid_metric_state_is_not_aggregated_as_zero() -> None:
    rows = [
        _raw_row(PRIMARY_TREATMENT, "B", 42, 0.0, metric_state="constant_prediction"),
        _raw_row(PRIMARY_COMPARATOR, "B", 42, 0.2),
    ]
    condition_level = aggregate_seeds_within_condition(pd.DataFrame(rows))
    assert not (
        (condition_level["graph_type"] == PRIMARY_TREATMENT) & (condition_level["condition"] == "B")
    ).any()


def test_duplicate_run_key_is_rejected() -> None:
    frame = _unbalanced_raw_table()
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(StatisticalValidationError, match="Duplicate run_key"):
        validate_raw_fold_data(duplicated)


def test_comparison_family_classification_is_frozen() -> None:
    assert canonical_graph_label("no_graph") == LEGACY_COMPARATOR
    specs = comparison_specs(
        [
            PRIMARY_TREATMENT,
            "string_ppi",
            "gene_ontology",
            "coexpression",
            "barabasi_albert",
            PRIMARY_COMPARATOR,
            "no_graph",
        ]
    )
    classes = {(spec.treatment, spec.comparator): spec.comparison_class for spec in specs}
    assert classes[(PRIMARY_TREATMENT, PRIMARY_COMPARATOR)] == "singular_primary"
    assert classes[("string_ppi", PRIMARY_COMPARATOR)] == "holm_secondary_graph_family"
    assert classes[("gene_ontology", PRIMARY_COMPARATOR)] == "holm_secondary_graph_family"
    assert classes[("coexpression", PRIMARY_COMPARATOR)] == "sensitivity_only"
    assert classes[("barabasi_albert", PRIMARY_COMPARATOR)] == "sensitivity_only"
    assert classes[(PRIMARY_TREATMENT, LEGACY_COMPARATOR)] == "legacy_architecture_secondary"


def _runner_fold(feature_hash: str, feature_file_hash: str) -> dict[str, object]:
    planned = ["KO_A"]
    split = sha256_json(["toy", "KO_A"])
    graph_hash = sha256_json(["self_loop_gat", 200])
    gene_order = ["A", "B", "C"]
    vectors = {
        "y_true": [0.0, 1.0, 2.0],
        "y_pred": [0.0, 1.0, 2.0],
        "training_mean": [0.5, 0.5, 0.5],
        "control_profile": [0.0, 0.0, 0.0],
    }
    identity = {
        "dataset": "toy",
        "input_hash": "1" * 64,
        "environment_lock_hash": "2" * 64,
        "gene_panel_hash": "3" * 64,
        "condition_panel_hash": "4" * 64,
        "preprocessing_hash": "5" * 64,
        "dataset_passport_registry_hash": "6" * 64,
        "dataset_passport_hash": "7" * 64,
        "matrix_contract_hash": "8" * 64,
        "control_selection_hash": "9" * 64,
        "target_mapping_hash": "b" * 64,
        "condition_eligibility_ledger_hash": "c" * 64,
        "cell_qc_policy_hash": "d" * 64,
        "response_definition_hash": "e" * 64,
        "graph_type": "self_loop_gat",
        "graph_instance": "self_loop_gat__instance_0",
        "graph_hash": graph_hash,
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
        "split_hash": split,
        "code_commit": "a" * 40,
        "seed": 42,
        "fold_seed": 123,
        "condition": "KO_A",
    }
    delta_true = np.asarray(vectors["y_true"]) - np.asarray(vectors["control_profile"])
    delta_pred = np.asarray(vectors["y_pred"]) - np.asarray(vectors["control_profile"])
    metadata = {
        **identity,
        "execution_stage": "full",
        "evaluated_conditions": planned,
        "graph": {
            "graph_type": identity["graph_type"],
            "graph_instance": identity["graph_instance"],
            "edge_hash": identity["graph_hash"],
        },
        "initialization_hash": "4" * 64,
        "target_mask": [True, False, False],
        "target_mask_hash": sha256_json([True, False, False]),
        "initial_perturbation_context_hash": "5" * 64,
        "initial_raw_feature_hash": feature_hash,
        "initial_raw_feature_artifact": "feature_artifacts/synthetic.npy",
        "initial_raw_feature_artifact_sha256": feature_file_hash,
        "baseline_feature_hash": "6" * 64,
        "input_instance_hash": "7" * 64,
        "input_contract_hash": "7" * 64,
        "training_condition_list_hash": "8" * 64,
        "fold_order_seed_policy_hash": "9" * 64,
        "delta_true_sd_ddof0": float(np.std(delta_true, ddof=0)),
        "delta_pred_sd_ddof0": float(np.std(delta_pred, ddof=0)),
        "delta_true_l2_norm": float(np.linalg.norm(delta_true)),
        "delta_pred_l2_norm": float(np.linalg.norm(delta_pred)),
    }
    return build_fold_result(
        run_identity=identity,
        condition="KO_A",
        fold_index=0,
        gene_order=gene_order,
        y_true=vectors["y_true"],
        y_pred=vectors["y_pred"],
        training_mean=vectors["training_mean"],
        control_profile=vectors["control_profile"],
        training_loss_by_epoch=[1.0],
        metrics={"pearson_r": 1.0},
        metric_states={"pearson_r": "valid"},
        metadata=metadata,
    ).as_dict()


def test_loader_requires_revision_metadata_and_metric_state(tmp_path, monkeypatch) -> None:
    hvg_dir = tmp_path / "hvg200"
    hvg_dir.mkdir()
    artifact_dir = hvg_dir / "feature_artifacts"
    artifact_dir.mkdir()
    features = np.zeros((3, 68), dtype=np.float32)
    artifact = artifact_dir / "synthetic.npy"
    np.save(artifact, features, allow_pickle=False)
    feature_hash = sha256_json({"dtype": "float32", "shape": [3, 68], "values": features.tolist()})
    feature_file_hash = sha256_file(artifact)
    monkeypatch.setattr(
        statistical_analysis_module,
        "_validate_aggregate_lineage",
        lambda _path, _data: None,
    )
    aggregate = {
        "dataset": "toy",
        "graph_type": "self_loop_gat",
        "seeds": {"seed_42": {"folds": [_runner_fold(feature_hash, feature_file_hash)]}},
    }
    (hvg_dir / "toy__self_loop_gat.json").write_text(json.dumps(aggregate), encoding="utf-8")
    loaded = load_fold_data(tmp_path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["value"] == pytest.approx(1.0)
    broken = _runner_fold(feature_hash, feature_file_hash)
    del broken["metadata"]["input_hash"]
    aggregate["seeds"]["seed_42"]["folds"] = [broken]
    (hvg_dir / "toy__self_loop_gat.json").write_text(json.dumps(aggregate), encoding="utf-8")
    with pytest.raises(StatisticalValidationError, match="input_hash"):
        load_fold_data(tmp_path)


def _canonical_fixture(drop_one_scale_seed: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    planned = ("KO_A", "KO_B")
    for dataset_index, dataset in enumerate(CANONICAL_DATASETS):
        for condition_index, condition in enumerate(planned):
            for hvg in FROZEN_SCALES:
                for seed_index, seed in enumerate(REQUIRED_SEEDS):
                    comparator = 0.05 + 0.01 * dataset_index + 0.005 * condition_index
                    comparator += hvg / 100_000 + seed_index / 1_000
                    treatment = comparator + 0.04
                    rows.append(
                        _raw_row(
                            PRIMARY_COMPARATOR,
                            condition,
                            seed,
                            comparator,
                            dataset=dataset,
                            hvg=hvg,
                            planned=planned,
                        )
                    )
                    if not (
                        drop_one_scale_seed
                        and dataset == CANONICAL_DATASETS[0]
                        and condition == "KO_A"
                        and hvg == 1000
                        and seed == 44
                    ):
                        rows.append(
                            _raw_row(
                                PRIMARY_TREATMENT,
                                condition,
                                seed,
                                treatment,
                                dataset=dataset,
                                hvg=hvg,
                                planned=planned,
                            )
                        )
    return pd.DataFrame(rows)


def test_primary_estimator_uses_arm_specific_fisher_means_and_fixed_hierarchy() -> None:
    frame = _canonical_fixture()
    result = analyze_primary_estimand(
        frame,
        bootstrap_replicates=100,
        bootstrap_seed=7,
    )
    source = result.seed_pairs
    dataset_arm_z: list[tuple[float, float]] = []
    for dataset in CANONICAL_DATASETS:
        subset = source.loc[source["dataset"] == dataset]
        treatment_condition_z = (
            subset.groupby(["condition", "hvg"])["treatment_z"].mean().groupby("condition").mean()
        )
        comparator_condition_z = (
            subset.groupby(["condition", "hvg"])["comparator_z"].mean().groupby("condition").mean()
        )
        dataset_arm_z.append(
            (float(treatment_condition_z.mean()), float(comparator_condition_z.mean()))
        )
    treatment_mu_z = float(np.mean([value[0] for value in dataset_arm_z]))
    comparator_mu_z = float(np.mean([value[1] for value in dataset_arm_z]))
    expected_delta_r = np.tanh(treatment_mu_z) - np.tanh(comparator_mu_z)
    summary = result.primary_summary.iloc[0]
    assert summary["delta_z"] == pytest.approx(treatment_mu_z - comparator_mu_z)
    assert summary["raw_delta_r_estimand"] == pytest.approx(expected_delta_r)
    assert summary["raw_delta_r_estimand"] != pytest.approx(np.tanh(summary["delta_z"]))
    assert summary["bootstrap_replicates"] == 100
    assert summary["raw_delta_r_ci_half_width"] == pytest.approx(
        max(
            summary["raw_delta_r_estimand"] - summary["raw_delta_r_ci_low"],
            summary["raw_delta_r_ci_high"] - summary["raw_delta_r_estimand"],
        )
    )
    assert len(result.bootstrap_distribution) == 100
    assert set(result.condition_scale["hvg"]) == set(FROZEN_SCALES)


def test_primary_complete_case_excludes_condition_missing_one_required_seed_scale() -> None:
    result = analyze_primary_estimand(
        _canonical_fixture(drop_one_scale_seed=True),
        bootstrap_replicates=20,
        bootstrap_seed=3,
    )
    exclusion = result.support_ledger.loc[
        (result.support_ledger["dataset"] == CANONICAL_DATASETS[0])
        & (result.support_ledger["condition"] == "KO_A")
    ].iloc[0]
    assert not bool(exclusion["primary_complete_case"])
    assert "hvg1000:treatment_valid_seeds=42,43" in exclusion["exclusion_reason"]
    assert not (
        (result.condition_estimates["dataset"] == CANONICAL_DATASETS[0])
        & (result.condition_estimates["condition"] == "KO_A")
    ).any()


def test_resolution_uses_seed_specific_hierarchical_effects_not_pooled_cell_deltas() -> None:
    result = analyze_primary_estimand(
        _canonical_fixture(),
        bootstrap_replicates=10,
        bootstrap_seed=4,
    )
    source = result.seed_pairs
    seed_effects: dict[int, float] = {}
    for seed in REQUIRED_SEEDS:
        dataset_arms: list[tuple[float, float]] = []
        for dataset in CANONICAL_DATASETS:
            subset = source.loc[(source["dataset"] == dataset) & (source["seed"] == seed)]
            treatment = subset.groupby(["condition", "hvg"])["treatment_z"].mean()
            comparator = subset.groupby(["condition", "hvg"])["comparator_z"].mean()
            treatment_condition = treatment.groupby("condition").mean()
            comparator_condition = comparator.groupby("condition").mean()
            dataset_arms.append(
                (float(treatment_condition.mean()), float(comparator_condition.mean()))
            )
        seed_effects[seed] = float(
            np.tanh(np.mean([value[0] for value in dataset_arms]))
            - np.tanh(np.mean([value[1] for value in dataset_arms]))
        )
    expected = np.quantile(
        [
            abs(seed_effects[seed_a] - seed_effects[seed_b]) / math.sqrt(2.0)
            for seed_a, seed_b in combinations(REQUIRED_SEEDS, 2)
        ],
        0.95,
        method="linear",
    )
    observed = result.resolution_summary.loc[
        result.resolution_summary["scope"] == "global_equal_dataset",
        "delta_res_point",
    ].iloc[0]
    assert observed == pytest.approx(expected)
    pooled_cell_algorithm = 0.0
    assert observed != pytest.approx(pooled_cell_algorithm, abs=1e-12)
    expected_columns = {
        "delta_res_statistic",
        *(f"delta_res_dataset__{dataset}" for dataset in CANONICAL_DATASETS),
    }
    assert expected_columns <= set(result.bootstrap_distribution.columns)


def test_g3_global_effect_averages_arm_z_across_datasets_before_inverse_transform() -> None:
    frame = _canonical_fixture()
    values = {
        200: {
            CANONICAL_DATASETS[0]: (0.90, 0.80),
            CANONICAL_DATASETS[1]: (0.20, -0.20),
            CANONICAL_DATASETS[2]: (-0.50, -0.80),
            CANONICAL_DATASETS[3]: (0.60, 0.00),
        },
        1000: {
            CANONICAL_DATASETS[0]: (0.30, -0.70),
            CANONICAL_DATASETS[1]: (0.80, 0.70),
            CANONICAL_DATASETS[2]: (-0.10, -0.30),
            CANONICAL_DATASETS[3]: (0.10, 0.00),
        },
    }
    for hvg, dataset_values in values.items():
        for dataset, (treatment, comparator) in dataset_values.items():
            frame.loc[
                (frame["dataset"] == dataset)
                & (frame["hvg"] == hvg)
                & (frame["graph_type"] == PRIMARY_TREATMENT),
                "value",
            ] = treatment
            frame.loc[
                (frame["dataset"] == dataset)
                & (frame["hvg"] == hvg)
                & (frame["graph_type"] == PRIMARY_COMPARATOR),
                "value",
            ] = comparator
    result = analyze_primary_estimand(frame, bootstrap_replicates=20, bootstrap_seed=9)
    global_row = result.g3_summary.loc[result.g3_summary["dataset"] == "ALL"].iloc[0]
    expected_effects: dict[int, float] = {}
    for hvg in (200, 1000):
        scale = result.condition_scale.loc[result.condition_scale["hvg"] == hvg]
        treatment_z = scale.groupby("dataset")["treatment_mu_z"].mean().mean()
        comparator_z = scale.groupby("dataset")["comparator_mu_z"].mean().mean()
        expected_effects[hvg] = float(np.tanh(treatment_z) - np.tanh(comparator_z))
    assert global_row["effect_hvg200"] == pytest.approx(expected_effects[200])
    assert global_row["effect_hvg1000"] == pytest.approx(expected_effects[1000])
    assert global_row["interaction_1000_minus_200"] == pytest.approx(
        expected_effects[1000] - expected_effects[200]
    )
    mean_dataset_interaction = result.g3_summary.loc[
        result.g3_summary["scope"] == "dataset", "interaction_1000_minus_200"
    ].mean()
    assert global_row["interaction_1000_minus_200"] != pytest.approx(mean_dataset_interaction)
    assert global_row["bootstrap_seed"] == 20_260_811


def test_tds09_final_branch_requires_heterogeneity_and_all_local_gates() -> None:
    summary = pd.DataFrame(
        [
            {
                "preliminary_branch": "TDS09_or_unresolved_requires_frozen_heterogeneity_family",
                "minimum_45_per_dataset_gate": True,
                "global_precision_gate": True,
                "all_dataset_precision_gate": True,
            }
        ]
    )
    estimates = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "raw_delta_r_estimand": 0.08 if index == 0 else 0.0,
                "raw_delta_r_ci_low": 0.04 if index == 0 else -0.01,
                "raw_delta_r_ci_high": 0.12 if index == 0 else 0.01,
                "operative_margin": 0.02,
                "minimum_45_condition_gate": True,
                "precision_gate": True,
            }
            for index, dataset in enumerate(CANONICAL_DATASETS)
        ]
    )
    tests = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "raw_p": 0.001 if index == 0 else 0.5,
                "holm_p": 0.004 if index == 0 else 1.0,
            }
            for index, dataset in enumerate(CANONICAL_DATASETS)
        ]
    )
    heterogeneity = MonteCarloTest(1.0, 0.01, 9, 1_000, "PCG64", 20_260_805)
    final_summary, final_datasets, decision = finalize_tds09_decision(
        summary, estimates, tests, heterogeneity
    )
    assert final_summary.iloc[0]["final_branch"] == "TDS09_dataset_specific"
    assert decision["claimed_datasets"] == [CANONICAL_DATASETS[0]]
    assert final_datasets["tds09_local_gate"].tolist() == [True, False, False, False]

    no_heterogeneity = MonteCarloTest(1.0, 0.05, 49, 1_000, "PCG64", 20_260_805)
    unresolved, _, failed = finalize_tds09_decision(summary, estimates, tests, no_heterogeneity)
    assert unresolved.iloc[0]["final_branch"] == "unresolved_or_imprecise"
    assert "heterogeneity_p_not_below_0.05" in failed["branch_failures"]


def test_fisher_clipping_is_only_used_for_z() -> None:
    frame = _canonical_fixture()
    mask = (frame["graph_type"] == PRIMARY_TREATMENT) & (frame["condition"] == "KO_A")
    frame.loc[mask, "value"] = 1.0
    result = analyze_primary_estimand(frame, bootstrap_replicates=10, bootstrap_seed=1)
    assert np.isfinite(result.seed_pairs["treatment_z"]).all()
    expected = np.arctanh(1.0 - FISHER_CLIP_EPSILON)
    assert result.seed_pairs.loc[result.seed_pairs["condition"] == "KO_A", "treatment_z"].iloc[
        0
    ] == pytest.approx(expected)


def test_primary_randomization_is_invariant_to_input_row_permutation() -> None:
    frame = _canonical_fixture()
    first = analyze_primary_estimand(
        frame.sample(frac=1.0, random_state=10),
        bootstrap_replicates=30,
        bootstrap_seed=20_260_804,
    )
    second = analyze_primary_estimand(
        frame.sample(frac=1.0, random_state=20),
        bootstrap_replicates=30,
        bootstrap_seed=20_260_804,
    )
    pd.testing.assert_frame_equal(first.primary_summary, second.primary_summary)
    pd.testing.assert_frame_equal(first.dataset_estimates, second.dataset_estimates)
    pd.testing.assert_frame_equal(
        first.bootstrap_distribution,
        second.bootstrap_distribution,
    )


def test_primary_rejects_cross_scale_build_or_dataset_passport_drift() -> None:
    frame = _canonical_fixture()
    drift = frame.copy()
    drift.loc[
        (drift["dataset"] == CANONICAL_DATASETS[0]) & (drift["hvg"] == 1000),
        "input_hash",
    ] = (
        "9" * 64
    )
    with pytest.raises(StatisticalValidationError, match="cross-scale input_hash"):
        analyze_primary_estimand(drift, bootstrap_replicates=5)
    drift = frame.copy()
    drift.loc[drift["hvg"] == 500, "code_commit"] = "b" * 40
    with pytest.raises(StatisticalValidationError, match="mixes code_commit"):
        analyze_primary_estimand(drift, bootstrap_replicates=5)

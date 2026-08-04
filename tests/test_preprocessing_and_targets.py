from __future__ import annotations

import json

import numpy as np
import pytest

from run_benchmark import (
    compute_fold_metrics,
    derive_fold_seed,
    make_mask,
    split_hash,
    training_condition_order,
)
from turbognn_audit.hashing import sha256_json
from turbognn_audit.panels import (
    build_canonical_condition_eligibility_panel,
    canonical_rank_hash,
    limit_condition_panel,
)
from turbognn_audit.preprocessing import (
    PreprocessingSpec,
    assert_held_out_mutation_invariant,
    fit_control_scaler,
    load_preprocessing_spec,
    preprocessing_fit_hash,
    rank_genes_by_control_variance,
    select_coexpression_fit_rows,
)
from turbognn_audit.targets import (
    TargetMapSpec,
    load_target_maps,
    resolve_condition_targets,
    target_preserving_gene_panel,
    validate_condition_targets_in_panel,
)


def _preprocessing_spec(scope: str = "control_only") -> PreprocessingSpec:
    return PreprocessingSpec(
        input_expression_state="log1p_normalized",
        input_expression_evidence="Toy fixture generated from a declared log1p matrix",
        normalization_method="identity_preverified_log1p",
        normalize_total_target_sum=10_000.0,
        log1p_pseudocount=1.0,
        log1p_base=None,
        hvg_fit_scope="control_only",
        scaler_fit_scope="control_only",
        coexpression_fit_scope=scope,
        hvg_method="scanpy_highly_variable_genes",
        hvg_flavor="seurat",
        scaler_ddof=0,
        scaler_epsilon=1e-8,
        scaler_constant_scale=1.0,
        coexpression_method="pearson",
        coexpression_threshold=0.3,
        coexpression_threshold_rule="strict_abs_gt",
        coexpression_constant_nan_policy="exclude_edge",
        coexpression_symmetry_policy="simple_undirected_then_bidirectional",
        coexpression_self_loop_policy="one_per_gene",
        cell_qc_policy="source_filtered_matrix_no_additional_cell_filter",
        perturbation_type_policy="explicit_mapped_single_or_multi_target",
        batch_policy="no_batch_correction",
        multi_target_policy="retain_all_explicit_mapped_targets",
        minimum_control_cells=2,
        minimum_perturbed_cells=20,
    )


def test_target_map_requires_complete_explicit_condition_coverage() -> None:
    spec = TargetMapSpec(
        dataset="toy",
        evidence="Toy metadata",
        condition_targets={"KO_A": ("A",)},
    )
    with pytest.raises(ValueError, match="lacks 1 evaluated conditions"):
        resolve_condition_targets(
            "toy",
            ["KO_A", "KO_B"],
            ["A", "B", "C"],
            {"toy": spec},
        )
    with pytest.raises(ValueError, match="absent from the raw matrix"):
        resolve_condition_targets("toy", ["KO_A"], ["B", "C"], {"toy": spec})


def test_target_map_loader_rejects_placeholders(tmp_path) -> None:
    path = tmp_path / "targets.json"
    path.write_text(
        json.dumps(
            {
                "toy": {
                    "evidence": "TODO",
                    "canonicalization_evidence": "Verified alias curation",
                    "allowed_perturbation_types": ["CRISPRi"],
                    "design_field_schema": {
                        "cell_context": {
                            "type": "nfc_string",
                            "source_value_policy": "retain_nfc_unmodified",
                        }
                    },
                    "raw_conditions": {
                        "KO_A": {
                            "canonical_object": {
                                "target_ids": ["ENSG_A"],
                                "target_identifier_namespace": "Ensembl",
                                "target_identifier_release": "test-release",
                                "target_mapping_table_sha256": "1" * 64,
                                "perturbation_type": "CRISPRi",
                                "design_fields": {"cell_context": "toy"},
                            },
                            "target_genes": ["A"],
                            "normalization_evidence": "Verified source label",
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="placeholder evidence"):
        load_target_maps(path)


def test_aliases_form_one_pooled_canonical_lopo_unit_and_exact_rank_hash(tmp_path) -> None:
    canonical_object = {
        "target_ids": ["ENSG_A", "ENSG_B"],
        "target_identifier_namespace": "Ensembl",
        "target_identifier_release": "test-release",
        "target_mapping_table_sha256": "1" * 64,
        "perturbation_type": "CRISPRi",
        "design_fields": {"cell_context": "toy"},
    }
    raw_record = {
        "canonical_object": canonical_object,
        "target_genes": ["A", "B"],
        "normalization_evidence": "Verified target-order normalization",
    }
    payload = {
        "toy": {
            "evidence": "Verified toy source metadata",
            "canonicalization_evidence": "Verified alias curation",
            "allowed_perturbation_types": ["CRISPRi"],
            "design_field_schema": {
                "cell_context": {
                    "type": "nfc_string",
                    "source_value_policy": "retain_nfc_unmodified",
                }
            },
            "raw_conditions": {"A+B": raw_record, "B+A": raw_record},
        }
    }
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    spec = load_target_maps(path)["toy"]
    assert len(set(spec.raw_to_canonical_id.values())) == 1
    canonical_id = next(iter(spec.condition_targets))
    panel = build_canonical_condition_eligibility_panel(
        dataset="toy",
        pre_qc_label_counts={"control": 100, "A+B": 12, "B+A": 9},
        post_qc_label_counts={"control": 100, "A+B": 12, "B+A": 9},
        control_label="control",
        minimum_cells=20,
        raw_to_canonical_id=spec.raw_to_canonical_id,
        condition_targets=spec.condition_targets,
        canonical_objects=spec.canonical_objects,
        raw_normalization_evidence=spec.raw_normalization_evidence,
        raw_record_hashes=spec.raw_record_hashes,
        allowed_perturbation_types=spec.allowed_perturbation_types,
        available_genes=("A", "B", "C"),
        cell_qc_policy="source_filtered_matrix_no_additional_cell_filter",
        perturbation_type_policy="explicit_mapped_single_or_multi_target",
    )
    assert panel.conditions == (canonical_id,)
    group = next(row for row in panel.eligibility_ledger if row.canonical_perturbation_id)
    assert group.raw_members == ("A+B", "B+A")
    assert group.post_qc_cell_count == 21
    assert group.rank_hash == canonical_rank_hash("toy", canonical_id)
    assert limit_condition_panel(panel, 1).conditions == (canonical_id,)


def test_target_preserving_panel_is_fixed_size_and_complete() -> None:
    panel = target_preserving_gene_panel(
        all_genes=["A", "B", "C", "D", "E"],
        ranked_control_hvgs=["E", "D", "C", "B", "A"],
        condition_targets={"KO_A": ("A",), "KO_B_C": ("B", "C")},
        panel_size=4,
    )
    assert panel == ("A", "B", "C", "E")
    validate_condition_targets_in_panel({"KO_A": ("A",), "KO_B_C": ("B", "C")}, panel)
    with pytest.raises(ValueError, match="exceed fixed panel size"):
        target_preserving_gene_panel(
            all_genes=["A", "B", "C"],
            ranked_control_hvgs=["A", "B", "C"],
            condition_targets={"triple": ("A", "B", "C")},
            panel_size=2,
        )


def test_target_mask_has_no_string_parsing_or_missing_target_fallback() -> None:
    genes = np.asarray(["A", "B", "C"])
    mask = make_mask(genes, ("A", "C"))
    assert mask.tolist() == [True, False, True]
    with pytest.raises(ValueError, match="missing or non-unique"):
        make_mask(genes, ("NOT_IN_PANEL",))
    with pytest.raises(ValueError, match="at least one target"):
        make_mask(genes, ())


def test_control_only_fit_is_invariant_to_held_out_outcome_mutation() -> None:
    expression = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 4.0, 8.0],
            [3.0, 5.0, 7.0, 9.0],
            [4.0, 6.0, 8.0, 10.0],
            [5.0, 7.0, 9.0, 11.0],
        ]
    )
    control_mask = np.asarray([True, True, False, False, False])
    held_out_mask = np.asarray([False, False, False, False, True])
    target_map = {"KO_A": ("A",), "KO_B": ("B",)}
    ranked_before = rank_genes_by_control_variance(
        expression,
        ["A", "B", "C", "D"],
        control_mask,
    )
    gene_panel = target_preserving_gene_panel(
        all_genes=["A", "B", "C", "D"],
        ranked_control_hvgs=ranked_before,
        condition_targets=target_map,
        panel_size=3,
    )
    spec = _preprocessing_spec()

    scaler_before = fit_control_scaler(
        expression[:, [0, 1, 3]],
        control_mask,
        minimum_control_cells=2,
        ddof=0,
        epsilon=1e-8,
        constant_scale=1.0,
    )
    normalized_before = scaler_before.transform(expression[:, [0, 1, 3]])
    _, coexpression_hash_before = select_coexpression_fit_rows(
        normalized_before,
        scope="control_only",
        control_mask=control_mask,
        held_out_mask=held_out_mask,
    )
    fit_hash_before = preprocessing_fit_hash(
        gene_panel=gene_panel,
        scaler=scaler_before,
        coexpression_fit_hash=coexpression_hash_before,
        target_mapping_hash="targets",
        control_selection_hash="controls",
        condition_eligibility_ledger_hash="eligibility",
        spec=spec,
    )

    mutated = expression.copy()
    mutated[held_out_mask] += 1_000_000.0
    ranked_after = rank_genes_by_control_variance(
        mutated,
        ["A", "B", "C", "D"],
        control_mask,
    )
    mutated_gene_panel = target_preserving_gene_panel(
        all_genes=["A", "B", "C", "D"],
        ranked_control_hvgs=ranked_after,
        condition_targets=target_map,
        panel_size=3,
    )
    scaler_after = fit_control_scaler(
        mutated[:, [0, 1, 3]],
        control_mask,
        minimum_control_cells=2,
        ddof=0,
        epsilon=1e-8,
        constant_scale=1.0,
    )
    normalized_after = scaler_after.transform(mutated[:, [0, 1, 3]])
    _, coexpression_hash_after = select_coexpression_fit_rows(
        normalized_after,
        scope="control_only",
        control_mask=control_mask,
        held_out_mask=held_out_mask,
    )
    fit_hash_after = preprocessing_fit_hash(
        gene_panel=mutated_gene_panel,
        scaler=scaler_after,
        coexpression_fit_hash=coexpression_hash_after,
        target_mapping_hash="targets",
        control_selection_hash="controls",
        condition_eligibility_ledger_hash="eligibility",
        spec=spec,
    )
    assert scaler_before.scaler_hash == scaler_after.scaler_hash
    assert ranked_before == ranked_after
    assert gene_panel == mutated_gene_panel
    assert coexpression_hash_before == coexpression_hash_after
    assert_held_out_mutation_invariant(fit_hash_before, fit_hash_after)


def test_training_only_coexpression_proves_held_out_exclusion() -> None:
    expression = np.arange(30, dtype=float).reshape(6, 5)
    controls = np.asarray([True, True, False, False, False, False])
    training = np.asarray([True, True, True, True, False, False])
    held_out = np.asarray([False, False, False, False, True, True])
    _, before_hash = select_coexpression_fit_rows(
        expression,
        scope="training_only",
        control_mask=controls,
        training_mask=training,
        held_out_mask=held_out,
    )
    mutated = expression.copy()
    mutated[held_out] *= -10_000
    _, after_hash = select_coexpression_fit_rows(
        mutated,
        scope="training_only",
        control_mask=controls,
        training_mask=training,
        held_out_mask=held_out,
    )
    assert before_hash == after_hash
    with pytest.raises(ValueError, match="overlap held-out"):
        select_coexpression_fit_rows(
            expression,
            scope="training_only",
            control_mask=controls,
            training_mask=np.asarray([True, True, True, True, True, False]),
            held_out_mask=held_out,
        )


def test_preprocessing_config_requires_verified_input_state(tmp_path) -> None:
    path = tmp_path / "preprocessing.json"
    raw = {
        "schema_version": "1.0.0",
        "input_expression_state": "log1p_normalized",
        "input_expression_evidence": "Toy fixture generated from a declared log1p matrix",
        "normalization_method": "identity_preverified_log1p",
        "normalize_total_target_sum": 10_000.0,
        "log1p_pseudocount": 1.0,
        "log1p_base": None,
        "hvg_fit_scope": "control_only",
        "scaler_fit_scope": "control_only",
        "coexpression_fit_scope": "control_only",
        "hvg_method": "scanpy_highly_variable_genes",
        "hvg_flavor": "seurat",
        "scaler_ddof": 0,
        "scaler_epsilon": 1e-8,
        "scaler_constant_scale": 1.0,
        "coexpression_method": "pearson",
        "coexpression_threshold": 0.3,
        "coexpression_threshold_rule": "strict_abs_gt",
        "coexpression_constant_nan_policy": "exclude_edge",
        "coexpression_symmetry_policy": "simple_undirected_then_bidirectional",
        "coexpression_self_loop_policy": "one_per_gene",
        "cell_qc_policy": "source_filtered_matrix_no_additional_cell_filter",
        "perturbation_type_policy": "explicit_mapped_single_or_multi_target",
        "batch_policy": "no_batch_correction",
        "multi_target_policy": "retain_all_explicit_mapped_targets",
        "minimum_control_cells": 2,
        "minimum_perturbed_cells": 20,
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert load_preprocessing_spec(path) == _preprocessing_spec()
    raw["input_expression_evidence"] = "TODO"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="source-verified"):
        load_preprocessing_spec(path)
    raw["input_expression_evidence"] = "Verified toy evidence"
    del raw["minimum_perturbed_cells"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="minimum_perturbed_cells"):
        load_preprocessing_spec(path)


def test_mutation_invariant_reports_leakage() -> None:
    with pytest.raises(ValueError, match="outcome leakage"):
        assert_held_out_mutation_invariant("before", "after")


def test_constant_prediction_has_explicit_invalid_metric_state() -> None:
    metrics, states = compute_fold_metrics(
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([1.0, 1.0, 1.0]),
        "KO_A",
    )
    assert metrics["pearson_r"] is None
    assert states["pearson_r"] == "constant_prediction"
    assert metrics["gene_top20_absolute_delta_jaccard"] is None
    assert states["gene_top20_absolute_delta_jaccard"] == ("fewer_than_20_or_malformed_gene_vector")


def test_fold_rng_is_invariant_to_full_or_split_chunk_position() -> None:
    conditions = ["KO_A", "KO_B", "KO_C"]
    aliases = {condition: (f"raw_{condition}",) for condition in conditions}
    cells = {condition: (f"cell_{condition}",) for condition in conditions}
    current_split = split_hash("panel", conditions, "KO_B", aliases, cells)
    from_full_run = derive_fold_seed(42, "KO_B", current_split)
    from_split_run = derive_fold_seed(42, "KO_B", current_split)
    assert from_full_run == from_split_run
    training = ["KO_A", "KO_C"]
    assert training_condition_order(training, from_full_run, 0) == training_condition_order(
        training, from_split_run, 0
    )
    assert training_condition_order(training, from_full_run, 1) == training_condition_order(
        training, from_split_run, 1
    )


def test_tds42_fold_order_seed_and_epoch_json_are_exact() -> None:
    conditions = ["KO_A", "KO_B", "KO_C"]
    aliases = {condition: (f"raw_{condition}",) for condition in conditions}
    cells = {condition: (f"cell_{condition}",) for condition in conditions}
    split = split_hash("panel", conditions, "KO_B", aliases, cells)
    digest = sha256_json(
        {
            "policy": "sha256_training_seed_condition_split_v1",
            "training_seed": 42,
            "held_out_condition": "KO_B",
            "split_hash": split,
        }
    )
    fold_order_seed = derive_fold_seed(42, "KO_B", split)
    assert fold_order_seed == int(digest[:8], 16)
    conditions = ["KO_C", "KO_A"]
    expected = sorted(
        conditions,
        key=lambda condition: sha256_json(
            {
                "policy": "sha256_fold_seed_epoch_nfc_condition_v1",
                "fold_order_seed": fold_order_seed,
                "epoch": 3,
                "condition": condition,
            }
        ),
    )
    assert training_condition_order(conditions, fold_order_seed, 3) == expected

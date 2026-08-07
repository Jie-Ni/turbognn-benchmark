"""Load and validate the frozen major-revision protocol."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import yaml

from .controls import ControlDefinition
from .errors import RevisionProtocolError
from .graph_supports import GraphMode

REQUIRED_ARTIFACT_FIELDS = {
    "gene_names",
    "y_true",
    "y_pred",
    "train_loss",
    "validation_loss",
    "learning_rate",
    "vector_space",
    "config",
    "config_hash",
    "input_hashes",
    "graph_support_hash",
    "preprocessing_state_hash",
    "architecture_hash",
    "initialization_hash",
    "code_hash",
    "checkpoint_path",
    "checkpoint_hash",
    "runtime",
}
REQUIRED_INPUT_HASHES = {
    "dataset",
    "protocol",
    "dataset_passport",
    "condition_panel_manifest",
    "target_encoding_audit",
    "preflight_summary",
    "environment_manifest",
    "environment_lock",
    "git_provenance",
    "nested_gene_panel_manifest",
    "shared_control_cell_evidence",
    "precision_design_registry",
    "precision_archive_condition_table",
    "precision_target_map",
    "hyperparameter_provenance",
}
RESULT_LOCK_IDS = [
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
]


def load_protocol(path: Path) -> dict[str, Any]:
    """Load YAML and reject protocol fields that permit silent fallback."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RevisionProtocolError("Protocol root must be a mapping")
    validate_protocol(payload)
    return payload


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    """Validate the design invariants required by the reviewer-facing revision."""

    if str(protocol.get("schema_version")) != "1.0":
        raise RevisionProtocolError("schema_version must be 1.0")
    datasets = protocol.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise RevisionProtocolError("datasets must be a non-empty mapping")
    for dataset, definition in datasets.items():
        if not isinstance(definition, dict):
            raise RevisionProtocolError(f"Dataset {dataset!r} definition must be a mapping")
        control = definition.get("control")
        if not isinstance(control, dict):
            raise RevisionProtocolError(f"Dataset {dataset!r} requires a control mapping")
        parsed = ControlDefinition(
            condition_column=str(definition.get("condition_column", "")),
            canonical_label=str(control.get("canonical_label", "")),
            aliases=tuple(str(value) for value in control.get("aliases", [])),
            case_sensitive=bool(control.get("case_sensitive", False)),
        )
        parsed.accepted_labels()
        if "fallback" in control:
            raise RevisionProtocolError(f"Dataset {dataset!r} control cannot declare a fallback")
        if definition.get("dataset_passport") != "required_for_production_h5ad":
            raise RevisionProtocolError(
                f"Dataset {dataset!r} must require a production H5AD passport"
            )
        if (
            definition.get("production_condition_column_source")
            != "hash_verified_dataset_passport_control_evidence"
        ):
            raise RevisionProtocolError(
                f"Dataset {dataset!r} must bind its production condition column to evidence"
            )
    if {name: definition.get("cell_line") for name, definition in datasets.items()} != {
        "adamson": "K562",
        "norman": "K562",
        "replogle_k562": "K562",
        "replogle_rpe1": "RPE1",
    }:
        raise RevisionProtocolError("Dataset-to-cell-line mapping must expose the 3/4 K562 design")

    preprocessing = protocol.get("preprocessing", {})
    if preprocessing.get("fit_scope") != "control_only":
        raise RevisionProtocolError("preprocessing.fit_scope must be control_only")
    if preprocessing.get("input_scale") not in {"counts", "log1p"}:
        raise RevisionProtocolError("preprocessing.input_scale must be explicit")
    if preprocessing.get("hvg_method") != "control_variance":
        raise RevisionProtocolError("preprocessing.hvg_method must be control_variance")
    if preprocessing.get("forced_target_retention_scope") != "selected_panel_only":
        raise RevisionProtocolError(
            "preprocessing.forced_target_retention_scope must be selected_panel_only"
        )
    if (
        preprocessing.get("nested_gene_panels_required") is not True
        or preprocessing.get("nested_gene_panel_order") != "200_subset_500_subset_1000"
        or preprocessing.get("scale_evaluation_spaces") != ["native_hvg", "common_200_gene"]
    ):
        raise RevisionProtocolError("Nested 200/500/1000 gene panels are required")

    supports = protocol.get("graph_supports", {}).get("modes", [])
    required_modes = {mode.value for mode in GraphMode}
    if set(supports) != required_modes:
        raise RevisionProtocolError(
            f"graph_supports.modes must contain exactly {sorted(required_modes)}"
        )
    graph_supports = protocol.get("graph_supports", {})
    if graph_supports.get("implicit_gat_self_loops") is not False:
        raise RevisionProtocolError("Implicit GAT self-loops must be disabled for every arm")
    if (
        graph_supports.get("primary_sparse_support_arm") != "string_go"
        or graph_supports.get("mixed_support_sensitivity_arm") != "combined"
        or graph_supports.get("curated_arms") != ["string_go"]
    ):
        raise RevisionProtocolError(
            "STRING-GO must be primary and combined must be mixed sensitivity"
        )
    if graph_supports.get("legacy_internal_identifier_meanings") != {
        "curated": "generic_supplied_sparse_support_not_expert_curation",
        "string_ppi": (
            "legacy_internal_identifier_for_STRING_combined_score_functional_"
            "association_not_physical_PPI"
        ),
    }:
        raise RevisionProtocolError(
            "Legacy graph-mode identifiers require exact terminology bounds"
        )
    topology_null = graph_supports.get("topology_null", {})
    if len(topology_null.get("replicate_seeds", [])) != 10:
        raise RevisionProtocolError("Exactly 10 topology-null rewire seeds are required")
    if topology_null.get("degree_sequence_preserved_exactly") is not True:
        raise RevisionProtocolError("Topology rewires must preserve the exact degree sequence")
    if topology_null.get("connected_component_count_preserved_exactly") is not True:
        raise RevisionProtocolError("Topology rewires must preserve connected-component count")
    if topology_null.get("connected_component_membership_preserved_exactly") is not True:
        raise RevisionProtocolError("Topology rewires must preserve exact component membership")
    if topology_null.get("unique_support_hashes_required") is not True:
        raise RevisionProtocolError("Topology-null support hashes must be unique")
    if topology_null.get("minimum_swapped_edge_fraction") != 0.80:
        raise RevisionProtocolError("Topology-null minimum swapped-edge fraction must be 0.80")
    if (
        topology_null.get("source_arm") != "string_go"
        or topology_null.get("arm_prefix") != "string_go_rewire_"
    ):
        raise RevisionProtocolError("Topology nulls must rewire STRING-GO")

    graph_composition = protocol.get("graph_composition", {})
    string_ppi = graph_composition.get("string_ppi", {})
    if (
        string_ppi.get("source") != "STRING_v12.0"
        or string_ppi.get("species_taxon") != 9606
        or string_ppi.get("combined_score_rule") != "exclusive_gt_400"
        or string_ppi.get("derivation_manifest") != "required_and_hash_verified"
        or string_ppi.get("required_local_inputs") != ["string_raw", "string_identifier_map"]
        or string_ppi.get("identifier_mapping_policy") != "explicit_hash_verified_exact_map"
        or string_ppi.get("evidence_scope") != "functional_association_not_physical_ppi"
    ):
        raise RevisionProtocolError("STRING provenance must be v12.0, human, score >400")
    gene_ontology = graph_composition.get("gene_ontology", {})
    if gene_ontology.get("release") != "populated_by_preflight_required":
        raise RevisionProtocolError("GO release must be populated and hashed by preflight")
    if gene_ontology.get("derivation_manifest") != "required_and_hash_verified":
        raise RevisionProtocolError("GO derivation manifest must be required and hash verified")
    if set(gene_ontology.get("required_derivation_policy_fields", [])) != {
        "namespace",
        "qualifier_policy",
        "evidence_code_policy",
        "ancestor_propagation_policy",
        "depth_policy",
    }:
        raise RevisionProtocolError("GO derivation-policy fields are not completely frozen")
    combined = graph_composition.get("combined", {})
    if combined.get("members") != ["string_ppi", "gene_ontology", "coexpression"]:
        raise RevisionProtocolError("Combined graph must be the exact three-source union")
    if combined.get("interpretation") != "mixed_string_go_control_coexpression_support_sensitivity":
        raise RevisionProtocolError("Combined support must be labelled mixed, not curated")
    if graph_composition.get("coexpression", {}).get("fit_scope") != "control_only":
        raise RevisionProtocolError("Coexpression must be fitted on controls only")
    if (
        graph_composition.get("coexpression", {}).get("threshold_rule")
        != "exclusive_gt_0.3_retain_all_edges"
    ):
        raise RevisionProtocolError("Coexpression must retain every edge with |r| > 0.3")

    panel = protocol.get("condition_panel", {})
    if panel.get("primary_size") != 50 or panel.get("sensitivity_size") != 50:
        raise RevisionProtocolError("Primary and sensitivity panels must each target 50 conditions")
    if panel.get("non_overlap_required") is not True:
        raise RevisionProtocolError("Sensitivity panel must be non-overlapping")
    if panel.get("legacy_first50_exclusion") != {
        "source": "hash_verified_dataset_passport",
        "primary_overlap_allowed": False,
        "primary_shortfall_policy": "PREFLIGHT_BLOCK",
        "conditional_overlap_allowed_with_legacy_or_primary": False,
        "triggered_conditional_shortfall_policy": "WITHHELD",
    }:
        raise RevisionProtocolError("Legacy-first50 panel exclusion contract is not frozen")
    trigger = panel.get("conditional_sensitivity_trigger", {})
    if (
        set(trigger)
        != {
            "combined_selection_stratum_total_variation_distance_gt",
            "absolute_standardized_log1p_condition_cell_count_difference_gt",
            "execution_arms_if_triggered",
        }
        or trigger.get("combined_selection_stratum_total_variation_distance_gt") != 0.10
        or trigger.get("absolute_standardized_log1p_condition_cell_count_difference_gt") != 0.25
        or trigger.get("execution_arms_if_triggered") != ["string_go", "dense"]
    ):
        raise RevisionProtocolError("Representativeness trigger thresholds are not frozen")

    target = protocol.get("target_encoding", {})
    if target.get("zero_target_indicator_allowed") is not False:
        raise RevisionProtocolError("A zero target indicator must be a reason-coded failure")
    if target.get("unordered_multi_target_canonicalization") is not True:
        raise RevisionProtocolError(
            "Multi-target conditions must use unordered-set canonicalization"
        )
    norman = datasets.get("norman", {})
    if norman.get("condition_column") != "perturbation":
        raise RevisionProtocolError(
            "Norman exact archived H5AD condition column must be perturbation"
        )
    if "_" not in str(norman.get("target_separator_pattern", "")):
        raise RevisionProtocolError("Norman target_separator_pattern must include underscore")
    if norman.get("unordered_multi_target_canonicalization") is not True:
        raise RevisionProtocolError(
            "Norman combinations must be canonicalized as unordered targets"
        )
    if "ctrl" not in norman.get("ignored_target_tokens", []):
        raise RevisionProtocolError("Norman inert ctrl components must be explicitly ignored")
    adamson = datasets.get("adamson", {})
    adamson_control = adamson.get("control", {})
    adamson_evidence = adamson_control.get("evidence", {})
    if (
        adamson.get("condition_column") != "perturbation"
        or adamson.get("condition_target_regex_pattern") != "^(?P<target>.+)_(?:pDS|pBA)[0-9]+$"
        or adamson.get("condition_target_regex_group") != "target"
        or adamson.get("target_equivalent_label_policy") != "preserve_guide_level_conditions"
        or not {"62(mod)_pBA581", "63(mod)_pBA580"} <= set(adamson_control.get("aliases", []))
        or adamson_evidence.get("commit") != "41f70980b481b9c854772cd8b8c5a4753c1c3eac"
        or adamson_evidence.get("modal_or_frequency_fallback_allowed") is not False
    ):
        raise RevisionProtocolError("Adamson public metadata and guide-level parser are not frozen")
    expected_exclusions = {
        "Gal4-4(mod)_pBA582": "NON_GENE_CONSTRUCT_ENDPOINT_EXCLUDED",
        "*": "UNRESOLVED_ARCHIVE_LABEL_ENDPOINT_EXCLUDED",
        "__MISSING_CONDITION__": "MISSING_ARCHIVE_CONDITION_ENDPOINT_EXCLUDED",
    }
    if adamson.get("excluded_endpoint_conditions") != expected_exclusions:
        raise RevisionProtocolError("Adamson unresolved/non-gene endpoints must be excluded")

    model = protocol.get("model", {})
    if model.get("input_features") != [
        "control_only_log_normalized_mean_expression",
        "binary_perturbation_indicator",
    ]:
        raise RevisionProtocolError("Model input must contain expression and indicator channels")
    if model.get("input_channels") != 2:
        raise RevisionProtocolError("MatchedGAT input_channels must be 2")
    identity_embedding = model.get("gene_identity_embedding", {})
    if (
        identity_embedding.get("enabled") is not True
        or int(identity_embedding.get("dimension", 0)) <= 0
        or identity_embedding.get("selected_gene_order_bound_by_preprocessing_hash") is not True
        or identity_embedding.get("initialization_shared_across_support_arms") is not True
    ):
        raise RevisionProtocolError("Gene-identity embedding contract is incomplete")
    if (
        model.get("hidden_normalization") != "per_node_layer_norm"
        or model.get("self_loop_independence_contract")
        != {
            "graph_edge_message_passing": "absent_between_distinct_nodes",
            "other_node_input_invariance_in_eval_fixed_state": "required_and_tested",
        }
        or model.get("initialization_interpretation")
        != "identically_initialized_separately_trained"
        or model.get("paired_initial_state_hash_validation") != "required_across_all_support_arms"
    ):
        raise RevisionProtocolError(
            "LayerNorm/self-loop/initialization model contract is incomplete"
        )

    training = protocol.get("training", {})
    if training.get("loss") != "mean_squared_error":
        raise RevisionProtocolError("Training loss must be mean_squared_error")
    if training.get("optimizer", {}).get("name") != "AdamW":
        raise RevisionProtocolError("Optimizer must be explicitly frozen as AdamW")
    if training.get("scheduler", {}).get("name") != "CosineAnnealingLR":
        raise RevisionProtocolError("Scheduler must be explicitly frozen as CosineAnnealingLR")
    if training.get("early_stopping", {}).get("restore_best_weights") is not True:
        raise RevisionProtocolError("Early stopping must restore best weights")
    early_stopping = training.get("early_stopping", {})
    patience = early_stopping.get("patience")
    minimum_delta = early_stopping.get("minimum_delta")
    if isinstance(patience, bool) or not isinstance(patience, int) or patience <= 0:
        raise RevisionProtocolError("Early-stopping patience must be a positive integer")
    if isinstance(minimum_delta, bool) or not isinstance(minimum_delta, (int, float)):
        raise RevisionProtocolError("Early-stopping minimum_delta must be numeric")
    if not math.isfinite(float(minimum_delta)) or float(minimum_delta) < 0:
        raise RevisionProtocolError("Early-stopping minimum_delta must be finite and non-negative")
    early_stopping_replay_contract = {
        "improvement_comparison": "strict_less_than_best_minus_minimum_delta",
        "tie_policy": "not_an_improvement",
        "first_finite_epoch_selected": True,
        "stop_check": "after_scheduler_step_each_epoch",
        "stop_condition": ("consecutive_nonqualifying_epochs_greater_than_or_equal_to_patience"),
        "artifact_replay_required": True,
        "trace_bound_to_artifact_and_checkpoint": True,
    }
    if any(
        early_stopping.get(key) != value for key, value in early_stopping_replay_contract.items()
    ):
        raise RevisionProtocolError("Early-stopping replay contract is incomplete")
    if training.get("validation_selection_seed") != 20260806:
        raise RevisionProtocolError("training.validation_selection_seed must be 20260806")
    if training.get("related_target_exclusion") != "exclude_any_shared_target_with_held_out":
        raise RevisionProtocolError("Related-target exclusion must remove every shared target")
    batching_contract = {
        "optimizer_step_unit": "one_condition_level_mean_profile",
        "condition_weighting": "equal_one_update_per_training_condition_per_epoch",
        "cell_count_weighting": "none_after_condition_mean_aggregation",
        "training_condition_order": "fixed_lexicographic_no_shuffle",
        "hidden_normalization_axis": "feature_channels_within_each_node_independently",
        "validation_reduction": "equal_mean_across_validation_conditions",
    }
    if any(training.get(key) != value for key, value in batching_contract.items()):
        raise RevisionProtocolError("Condition-level batching and weighting contract is incomplete")
    if training.get("separately_trained_after_identical_initialization") is not True:
        raise RevisionProtocolError("Matched arms must be separately trained after identical init")

    statistics = protocol.get("statistics", {})
    if statistics.get("aggregation_and_resampling_unit") != "held_out_condition":
        raise RevisionProtocolError(
            "statistics.aggregation_and_resampling_unit must be held_out_condition"
        )
    if statistics.get("seed_handling") != "average_within_condition_before_inference":
        raise RevisionProtocolError("Seeds must be averaged within condition before inference")
    if not statistics.get("expected_seeds"):
        raise RevisionProtocolError("statistics.expected_seeds cannot be empty")
    if (
        statistics.get("primary_estimand")
        != "fixed_protocol_equal_benchmark_dataset_mean_string_go_minus_dense_pearson_r"
        or statistics.get("primary_contrast") != "string_go_minus_dense"
        or statistics.get("fixed_protocol_not_per_arm_optimum") is not True
        or statistics.get("archived_optimisation_repeatability_reference") != 0.010
        or statistics.get("archived_reference_role")
        != "descriptive_precision_context_without_success_authority"
        or statistics.get("primary_interval") != "95pct_conditional_bootstrap_uncertainty_interval"
        or statistics.get("nominal_coverage_claim") != "prohibited"
    ):
        raise RevisionProtocolError(
            "The primary fixed-protocol estimand and zero-reference interval are not frozen"
        )
    if statistics.get("bootstrap_random_seed") != 20260806:
        raise RevisionProtocolError("statistics.bootstrap_random_seed must be 20260806")
    if statistics.get("primary_multiplicity", {}).get(
        "correction"
    ) != "not_applicable_single_primary_contrast" or statistics.get("primary_multiplicity", {}).get(
        "family"
    ) != [
        "string_go_minus_dense"
    ]:
        raise RevisionProtocolError("Primary multiplicity correction must be frozen")
    scale_contrast = statistics.get("scale_contrast", {})
    if (
        scale_contrast.get("contrasts") != ["500_minus_200_hvg", "1000_minus_200_hvg"]
        or scale_contrast.get("evaluation_spaces") != ["native_hvg", "common_200_gene"]
        or scale_contrast.get("common_200_gene_order_hash_required") is not True
    ):
        raise RevisionProtocolError("Exactly two predeclared scale contrasts are required")
    if (
        statistics.get("cross_dataset_summary")
        != "equal_benchmark_dataset_mean_with_condition_resampling_within_dataset"
        or statistics.get("dataset_composition_statement")
        != "three_of_four_benchmark_datasets_are_K562"
        or statistics.get("equal_cell_line_sensitivity")
        != {
            "enabled": True,
            "cell_line_groups": {
                "K562": ["adamson", "norman", "replogle_k562"],
                "RPE1": ["replogle_rpe1"],
            },
            "weighting": "equal_cell_line_after_equal_dataset_mean_within_cell_line",
        }
    ):
        raise RevisionProtocolError("Equal-dataset and equal-cell-line estimands are incomplete")
    if statistics.get("dependence_sensitivities") != {
        "target_cluster_resampling": "required",
        "leave_target_out": "required",
        "leave_pathway_out": "required_or_explicit_NOT_AVAILABLE",
        "shared_control_cell_bootstrap": (
            "measured_cell_resampling_interval_from_hash_bound_control_profiles_required"
        ),
    }:
        raise RevisionProtocolError("Dependence-sensitivity requirements are incomplete")
    if statistics.get("primary_claim_gate") != {
        "directional_decisions": [
            "DIRECTIONAL_POSITIVE",
            "DIRECTIONAL_NEGATIVE",
            "INCONCLUSIVE",
        ],
        "directional_decision_source": "unrounded_primary_interval_relative_to_zero_only",
        "sparse_support_wording_requires": [
            "DIRECTIONAL_POSITIVE",
            "CLAIM_CONSEQUENCE_GATE_PASS",
            "BASELINE_ABSOLUTE_SKILL_ELIGIBLE",
            "CONDITION_RANKING_CONSEQUENCE_ELIGIBLE",
        ],
        "biological_edge_wording_requires": [
            "DIRECTIONAL_POSITIVE",
            "CLAIM_CONSEQUENCE_GATE_PASS",
            "BASELINE_ABSOLUTE_SKILL_ELIGIBLE",
            "CONDITION_RANKING_CONSEQUENCE_ELIGIBLE",
            "TOPOLOGY_DIRECTIONALLY_SUPPORTIVE",
            "TOPOLOGY_DIAGNOSTICS_PASS",
        ],
        "positive_without_consequence_wording": ("positive_effect_without_interpretable_benefit"),
        "positive_with_consequence_without_topology_wording": ("sparse_support_wording_allowed"),
        "negative_wording": "negative_directional_effect",
        "inconclusive_wording": "benefit_not_demonstrated",
    }:
        raise RevisionProtocolError("Cross-lock biological-edge claim gate is incomplete")
    target_sensitivity = statistics.get("target_level_sensitivity", {})
    if (
        statistics.get("primary_analysis_unit") != "guide_condition_identifier"
        or statistics.get("primary_unit_interpretation")
        != "guide_conditions_are_not_unique_genes_or_target_sets"
        or target_sensitivity.get("status") != "explicitly_secondary_sensitivity"
        or target_sensitivity.get("guide_aggregation")
        != "equal_weight_guide_conditions_within_canonical_target_set"
        or target_sensitivity.get("resampling_unit") != "canonical_target_set_within_dataset"
        or target_sensitivity.get("bootstrap_replicates") != statistics.get("bootstrap_replicates")
        or target_sensitivity.get("bootstrap_random_seed")
        != statistics.get("bootstrap_random_seed")
    ):
        raise RevisionProtocolError(
            "Guide-condition primary and target-level sensitivity estimands are not frozen"
        )

    fields = set(protocol.get("artifacts", {}).get("required_fields", []))
    missing_fields = REQUIRED_ARTIFACT_FIELDS - fields
    if missing_fields:
        raise RevisionProtocolError(
            f"artifacts.required_fields is missing {sorted(missing_fields)}"
        )
    input_hashes = set(protocol.get("artifacts", {}).get("required_input_hashes", []))
    if input_hashes != REQUIRED_INPUT_HASHES:
        raise RevisionProtocolError("artifacts.required_input_hashes is incomplete")
    checkpoint_validation = protocol.get("artifacts", {}).get("checkpoint_validation", {})
    required_checkpoint_bindings = {
        "identity",
        "full_config_hash",
        "input_hashes_hash",
        "architecture",
        "graph_support_hash",
        "preprocessing_state_hash",
        "code_hash",
        "git_commit",
        "git_worktree_status",
        "git_status_hash",
        "vector_space",
        "gene_names_hash",
        "initialization_hash",
        "best_epoch",
        "early_stopping_observation",
        "deterministic_backend",
    }
    if (
        checkpoint_validation.get("safe_decode_required") is not True
        or set(checkpoint_validation.get("cross_bindings", [])) != required_checkpoint_bindings
    ):
        raise RevisionProtocolError("Checkpoint-to-artifact cross-bindings are incomplete")
    environment_lock = protocol.get("environment_lock", {})
    if (
        environment_lock.get("format") != "cbac-complete-pip-freeze-v2"
        or environment_lock.get("required_runtime_headers")
        != [
            "python-version",
            "cuda-runtime-version",
            "cudnn-version",
            "nvidia-driver-version",
            "pytorch-version",
            "pytorch-geometric-version",
            "complete-distribution-set-sha256",
            "direct-runtime-dependencies-sha256",
            "pyg-extension-versions-sha256",
        ]
        or environment_lock.get("package_policy")
        != "exact_complete_active_distribution_set_and_versions"
        or environment_lock.get("direct_dependency_source")
        != "pyproject_runtime_model_and_h5ad_dependencies"
        or environment_lock.get("production_h5ad_requires_h5py") is not True
        or environment_lock.get("placeholder_policy") != "hard_failure"
    ):
        raise RevisionProtocolError("Exact environment-lock contract is not frozen")
    git_provenance = protocol.get("git_provenance", {})
    if (
        git_provenance.get("base_commit") != "exact_40_lowercase_hex"
        or git_provenance.get("worktree_status") != ["CLEAN", "DIRTY"]
        or git_provenance.get("code_tree_hash_required") is not True
        or git_provenance.get("bound_to_each_fold_artifact") is not True
    ):
        raise RevisionProtocolError("Git provenance contract is incomplete")

    release = protocol.get("release_registry", {})
    source_values_rendering = release.get("source_values_rendering", {})
    if (
        release.get("exact_result_lock_count") != 13
        or release.get("exact_result_lock_ids") != RESULT_LOCK_IDS
        or release.get("mandatory_fit_coverage_cascades_to_numeric_locks") is not True
        or release.get("not_triggered_is_valid_for")
        != ["REPRESENTATIVENESS-TRIGGER", "CONDITIONAL-PANEL"]
        or release.get("empirical_seed_resolution_role")
        != "descriptive_parallel_audit_not_primary_threshold"
        or source_values_rendering
        != {
            "modes": ["author-review", "released"],
            "exact_lock_macro_count": 13,
            "released_mode_requires_all_locks_released": True,
            "self_hashed_round_trip_manifest_required": True,
            "manual_transcription_policy": "prohibited",
            "numeric_json_type_policy": "number_only_bool_and_numeric_strings_rejected",
            "nonfinite_numeric_policy": "rejected",
            "display_rounding": "decimal_round_half_up",
            "primary_point_inside_ci_required": True,
            "primary_decision_recomputed_from_unrounded_ci": True,
            "primary_decision_reference": "zero_only",
            "archived_optimisation_repeatability_reference_has_success_authority": False,
            "cross_lock_biological_claim_gate_required": True,
        }
    ):
        raise RevisionProtocolError("The exact 13-lock release registry is not frozen")
    release_assets = release.get("release_assets", {})
    if (
        release_assets.get("modes") != ["author-review", "released"]
        or release_assets.get("released_mode_missing_asset_policy") != "WITHHELD"
        or release_assets.get("semantic_round_trip_required") is not True
        or release_assets.get("manifest_self_hash_required") is not True
        or release_assets.get("formats") != ["csv", "tex", "svg", "pdf"]
        or release_assets.get("required_asset_groups")
        != [
            "primary_summary_and_forest",
            "paired_condition_distribution",
            "fraction_improved_interval",
            "target_sensitivity",
            "topology_instances_and_diagnostics",
            "scale_native_and_common200",
            "propagation",
            "representativeness_and_conditional",
            "secondary_ranking_and_multiplicity",
            "baseline_absolute_skill",
            "compute_and_failure_denominators",
            "external_diagnostics_and_exclusions",
        ]
    ):
        raise RevisionProtocolError("Release-asset contract is incomplete")
    trust_boundary = release.get("main_release_trust_boundary", {})
    if trust_boundary != {
        "anchor_id": "CBAC-MAIN-RELEASE-TRUST-V1",
        "mode": "caller_pinned_detached_file_sha256",
        "detached_from_evidence_bundle_required": True,
        "caller_pinned_sha256_required": True,
        "synchronized_bundle_rewrite_policy": "WITHHELD",
        "author_review_anchor_policy": "absent_no_placeholder_anchor",
        "trusted_categories": [
            "dataset_passports",
            "dataset_files",
            "graph_files",
            "string_go_source_and_policy",
            "split_condition_target_order",
            "protocol",
            "entrypoint_environment",
            "raw_predictions_traces",
            "baseline_inputs",
            "precision_simulation_inputs",
            "topology_diagnostics",
            "release_asset_manifest",
        ],
        "caller_control_plane_boundary": "caller_hash_must_be_immutable_to_bundle_attacker",
    }:
        raise RevisionProtocolError("Main release trust boundary is incomplete")
    measured = protocol.get("measured_compute", {})
    if (
        measured.get("attempt_ledger") != "append_only_across_invocations_by_exact_fit_key"
        or measured.get("completed_fit_retry_policy")
        != "skip_existing_success_and_retain_prior_failures"
        or measured.get("attempt_start_persistence")
        != "STARTED_UNFINALIZED_written_atomically_before_fit"
        or measured.get("unresolved_started_attempt_policy")
        != "block_resume_until_operator_resolution"
        or measured.get("orphan_ledger_policy") != "detect_newer_content_addressed_ledger_and_block"
        or measured.get("successful_attempt_policy")
        != "exactly_one_success_and_success_is_final_attempt_per_fit_key"
    ):
        raise RevisionProtocolError("Measured-compute attempt accounting is not frozen")
    external = protocol.get("external_positive_control", {})
    if (
        external.get("system") != "GEARS"
        or external.get("official_repository") != "https://github.com/snap-stanford/GEARS"
        or external.get("pinned_commit") != "f374e43e197b295016d80395d7a54ddb81cc6769"
        or external.get("official_metric_source") != "gears/inference.py"
        or external.get("official_metric_function") != "gears.inference.compute_metrics"
        or external.get("manifest_and_hash_verified_reference_required") is not True
        or external.get("missing_or_placeholder_policy") != "WITHHELD"
        or external.get("official_metric_recomputation_required") is not True
        or external.get("semantic_diagnostic_gates")
        != {
            "prediction_nondegeneracy": ("strictly_positive_prediction_standard_deviation"),
            "baseline_superiority": ("strictly_positive_directional_improvement_for_every_metric"),
            "training_loss_improvement": ("first_epoch_minus_final_epoch_strictly_positive"),
            "validation_loss_improvement": ("first_epoch_minus_final_epoch_strictly_positive"),
            "reader_null_pass_policy": "prohibited",
        }
        or external.get("raw_evidence_required")
        != [
            "prediction_vectors",
            "truth_vectors",
            "de_prediction_vectors",
            "de_truth_vectors",
            "reported_metrics",
            "baseline_metrics",
            "training_loss",
            "validation_loss",
            "gene_order",
            "perturbable_gene_order",
            "condition_target_indices",
            "condition_target_indicators",
            "identifier_coverage",
            "target_mapping_audit",
        ]
        or external.get("phase_roles")
        != {
            "official_reference": "official_reference",
            "candidate": "heldout_adaptation",
        }
        or external.get("expected_value_provenance_cross_bindings")
        != [
            "phase_role",
            "expected_metrics",
            "repository_url",
            "commit",
            "dataset_sha256",
            "split",
            "entrypoint_sha256",
            "argv_sha256",
            "frozen_before_execution",
        ]
        or external.get("identical_workflow_policy")
        != "explicit_pinned_reuse_and_distinct_provenance_required"
        or external.get("trust_boundary")
        != {
            "mode": "caller_pinned_sha256_over_detached_anchor_and_external_sources",
            "detached_anchor_must_be_outside_evidence_bundle": True,
            "caller_pinned_anchor_sha256_required": True,
            "trusted_external_sources": [
                "environment_lock",
                "execution_plan",
                "dataset",
                "official_metric_source",
                "phase_entrypoints",
                "phase_expected_value_provenance",
                "phase_execution_audits",
                "phase_raw_evidence",
                "phase_initialization_state",
                "phase_checkpoint_state",
                "phase_graph_state",
                "phase_outputs",
            ],
            "independent_recomputation_required": [
                "environment_lock_structure",
                "execution_plan_self_hash",
                "dataset_content_hash",
                "phase_roles",
                "expected_value_provenance",
                "gene_order",
                "perturbable_gene_order",
                "condition_target_indices",
                "condition_target_indicators",
                "graph_semantic_bindings",
            ],
            "synchronized_bundle_rewrite_policy": "WITHHELD",
            "caller_control_plane_boundary": (
                "caller_pinned_sha256_must_be_immutable_to_bundle_attacker"
            ),
        }
        or external.get("eligibility_branches")
        != {
            "valid_and_within_tolerance": "ELIGIBLE_POSITIVE_CONTROL_PASSED",
            "valid_but_outside_tolerance": "EXCLUDED_FAILED_POSITIVE_CONTROL",
            "invalid_or_incomplete_evidence": "WITHHELD_INVALID_OR_INCOMPLETE_EVIDENCE",
        }
    ):
        raise RevisionProtocolError("GEARS validation must remain fail-closed")

    external_family = protocol.get("external_comparator_validity_family", {})
    if (
        external_family.get("headline_lock_id") != "EXTERNAL-COMPARATOR-VALIDATION"
        or external_family.get("members") != ["GEARS", "scGPT", "Geneformer"]
        or external_family.get("allowed_member_decisions") != ["PASS", "EXCLUDED", "WITHHELD"]
        or external_family.get("family_release_requires_no_withheld_member") is not True
        or external_family.get("parse_valid_scalar_is_scientific_validity") is not False
        or external_family.get("generic_required_bindings")
        != [
            "official_repository",
            "pinned_commit",
            "dataset",
            "split",
            "target_mapping",
            "prediction_vectors",
            "truth_vectors",
            "vector_row_conditions",
            "training_loss",
            "validation_loss",
            "gene_order",
            "baseline_vectors",
            "baseline_provenance",
            "baseline_training_vectors",
            "baseline_training_row_conditions",
            "independent_expected_value_provenance",
        ]
        or external_family.get("missing_binding_decision") != "WITHHELD"
        or external_family.get("invalid_or_incomplete_evidence_decision") != "WITHHELD"
        or external_family.get("completed_scientific_gate_failure_decision") != "EXCLUDED"
        or external_family.get("claim_deletion_exclusion")
        != {
            "explicit_self_hashed_reason_coded_manifest_required": True,
            "claim_deleted_required": True,
            "naked_missing_manifest_decision": "WITHHELD",
            "valid_claim_deletion_decision": "EXCLUDED",
        }
        or external_family.get("ranking_policy") != "include_only_PASS_members"
        or external_family.get("GEARS_validator") != "detached_anchor_positive_control"
        or external_family.get("scGPT_validator") != "hash_bound_full_vector_evidence"
        or external_family.get("Geneformer_validator") != "hash_bound_full_vector_evidence"
    ):
        raise RevisionProtocolError("External-comparator family must be fail-closed")

    _validate_hypothesis_families(protocol.get("hypothesis_families"))
    _validate_analytic_baselines(protocol.get("analytic_baselines"))
    _validate_precision_design(protocol.get("precision_design"))
    _validate_hyperparameter_provenance(protocol)

    fit_counts = protocol.get("planned_fit_counts", {})
    if fit_counts.get("topology_primary_200_hvg_arms", {}).get("degree_preserving_rewire") != 10:
        raise RevisionProtocolError("Planned fit matrix must contain exactly 10 rewire arms")
    if (
        fit_counts.get("topology_primary_arms_total") != 13
        or fit_counts.get("topology_primary_200_hvg_fits") != 7_800
        or fit_counts.get("mixed_support_sensitivity_arm_scale_cells") != {"combined_200": 1}
        or fit_counts.get("mixed_support_sensitivity_fits") != 600
        or fit_counts.get("scale_extension_arm_scale_cells")
        != {
            "string_go_500": 1,
            "dense_500": 1,
            "string_go_1000": 1,
            "dense_1000": 1,
        }
        or fit_counts.get("scale_extension_fits") != 2_400
        or fit_counts.get("analytic_baseline_evaluations") != 1_800
        or fit_counts.get("analytic_ridge_refits") != 600
        or fit_counts.get("analytic_baselines_in_neural_fit_total") is not False
        or fit_counts.get("mandatory_total_fits") != 10_800
        or fit_counts.get("worst_case_total_fits") != 12_000
    ):
        raise RevisionProtocolError("Planned fit counts are inconsistent with the frozen design")
    analysis_blocks = protocol.get("analysis_blocks", {})
    expected_topology_arms = [
        "dense",
        "self_loop",
        "string_go",
        *[f"string_go_rewire_{index:02d}" for index in range(1, 11)],
    ]
    if (
        analysis_blocks.get("topology_primary")
        != {"hvg_scales": [200], "default_arms": expected_topology_arms}
        or analysis_blocks.get("mixed_support_sensitivity")
        != {"hvg_scales": [200], "default_arms": ["combined"]}
        or analysis_blocks.get("scale_extension")
        != {
            "default_arms_by_hvg": {
                500: ["string_go", "dense"],
                1000: ["string_go", "dense"],
            },
            "string_go_and_dense_200_reused_from": "topology_primary",
        }
        or analysis_blocks.get("conditional_nonoverlap")
        != {
            "hvg_scales": [200],
            "default_arms": ["string_go", "dense"],
            "execute_only_if_representativeness_triggered": True,
        }
    ):
        raise RevisionProtocolError("Analysis-block arm/scale matrix is not frozen")
    if protocol.get("evaluation", {}).get("minimum_cells_per_condition") != 20:
        raise RevisionProtocolError("minimum_cells_per_condition must be 20")
    if (
        protocol.get("evaluation", {}).get("top20_tie_break")
        != "descending_absolute_delta_then_ascending_gene_identifier"
    ):
        raise RevisionProtocolError("Top-20 gene overlap requires a deterministic tie break")


def _validate_hypothesis_families(raw: Any) -> None:
    """Reject incomplete or relabelled multiplicity families."""

    if not isinstance(raw, dict):
        raise RevisionProtocolError("hypothesis_families must be a mapping")
    expected = {
        "primary": ("PRIMARY_STRING_GO_VS_DENSE_200", 1),
        "mixed_support": ("MIXED_COMBINED_VS_DENSE_200_SENSITIVITY", 1),
        "topology": ("TOPOLOGY_STRING_GO_VS_10_REWIRES", "10_graph_instances_one_ensemble_test"),
        "propagation": ("PROPAGATION_STRING_GO_AND_DENSE_VS_SELF_LOOP", 2),
        "scale": ("SCALE_500_AND_1000_VS_200", "2_per_evaluation_space"),
        "conditional": ("CONDITIONAL_INDEPENDENT_PANEL_STRING_GO_VS_DENSE", "1_if_triggered"),
        "secondary": ("SECONDARY_METRICS_AND_RANKING", "4_per_declared_metric_family"),
        "external_validity": ("EXTERNAL_COMPARATOR_VALIDITY", 3),
        "external_performance": (
            "EXTERNAL_COMPARATOR_SCALE_PERFORMANCE",
            "3_per_passing_adapter",
        ),
    }
    if set(raw) != set(expected):
        raise RevisionProtocolError("Hypothesis-family registry must list all nine families")
    for key, (label, denominator) in expected.items():
        value = raw.get(key)
        if not isinstance(value, dict) or value.get("label") != label:
            raise RevisionProtocolError(f"Hypothesis family {key!r} has an invalid label")
        if value.get("denominator") != denominator:
            raise RevisionProtocolError(f"Hypothesis family {key!r} has an invalid denominator")
        if not isinstance(value.get("adjustment"), str) or not value["adjustment"]:
            raise RevisionProtocolError(f"Hypothesis family {key!r} requires an adjustment")
    if raw["scale"].get("evaluation_spaces") != ["native_hvg", "common_200_gene"]:
        raise RevisionProtocolError("Scale family must bind both evaluation spaces")
    for family in ("external_validity", "external_performance"):
        if raw[family].get("members") != ["GEARS", "scGPT", "Geneformer"]:
            raise RevisionProtocolError("External family members are incomplete")
    performance = raw["external_performance"]
    if (
        performance.get("scale_contrasts")
        != ["200_hvg_overall", "500_hvg_overall", "1000_hvg_overall"]
        or performance.get("adjustment") != "benjamini_hochberg_within_each_passing_adapter"
        or performance.get("cross_adapter_pooling") != "prohibited"
        or performance.get("estimand") != "adapter_minus_revision_string_go_pearson_r"
        or performance.get("reference_model_id") != "revision_string_go"
        or performance.get("identical_support_required")
        != [
            "dataset",
            "hvg",
            "primary_panel",
            "condition_order",
            "gene_order",
            "split",
            "target_mapping",
        ]
        or performance.get("prohibited_references")
        != ["legacy_combined", "dense", "anonymous_matched_reference"]
    ):
        raise RevisionProtocolError("External performance multiplicity family is incomplete")


def _validate_analytic_baselines(raw: Any) -> None:
    """Validate frozen, leakage-resistant analytic baselines."""

    expected = {
        "methods": [
            "zero_control_delta",
            "training_condition_mean",
            "deterministic_ridge_linear",
        ],
        "source": "retained_training_and_truth_vectors",
        "split": "identical_leave_one_perturbation_out",
        "ridge_alpha": 1.0,
        "ridge_feature_space": "training_only_binary_target_indicators",
        "test_condition_used_for_fit": False,
        "shared_artifact_key": ["dataset", "hvg", "panel", "held_out_condition"],
        "neural_seed_in_artifact_identity": False,
        "shared_across_neural_seeds_and_analysis_blocks": True,
        "reporting": "absolute_skill_and_improvement_over_each_baseline",
        "accounting": "analytic_refits_excluded_from_neural_fit_count",
    }
    if raw != expected:
        raise RevisionProtocolError("Analytic-baseline contract is not frozen")


def _validate_precision_design(raw: Any) -> None:
    """Validate that precision simulation is pre-outcome and non-authoritative."""

    expected = {
        "status": "pre_outcome_required",
        "source": "legacy_archive_variance_and_dependence_structure_only",
        "frozen_archive_id": "submitted_legacy_archive_pre_revision",
        "frozen_source_selection_date": date(2026, 8, 7),
        "trusted_source_bindings": [
            "caller_pinned_archive_condition_table_sha256",
            "caller_pinned_canonical_target_map_sha256",
        ],
        "condition_variance_formula": ("sample_variance_ddof_1_of_archived_condition_deltas"),
        "target_cluster_icc_formula": "one_way_random_effects_anova_unbalanced_n0",
        "mean_guides_per_target_formula": "arithmetic_mean_condition_count_per_target",
        "source_join": "exact_one_to_one_dataset_condition",
        "zero_center_before_simulation": True,
        "new_outcome_access": "prohibited",
        "planned_conditions_per_dataset": 50,
        "uncertainty_half_width_target": 0.010,
        "target_role": "precision_not_utility_or_success",
        "simulation_summary": ("q90_of_simulated_95pct_conditional_bootstrap_interval_width"),
        "interval_width_gate": 0.020,
        "simulation_random_seed": 20260806,
        "minimum_simulation_replicates": 100,
        "minimum_bootstrap_replicates_per_simulation": 500,
        "cluster_aware_simulation_required": True,
        "conservative_scenario_grid": [
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
        ],
        "worst_scenario_q90_controls_gate": True,
        "legacy_archive_is_proxy_not_matched_run_width_guarantee": True,
        "source_component_recomputation_required": True,
        "registry_self_hash_required": True,
    }
    if raw != expected:
        raise RevisionProtocolError("Pre-outcome precision-design registry is incomplete")


def _validate_hyperparameter_provenance(protocol: Mapping[str, Any]) -> None:
    """Validate the machine-readable frozen-hyperparameter ledger."""

    raw = protocol.get("hyperparameter_provenance")
    if not isinstance(raw, dict):
        raise RevisionProtocolError("hyperparameter_provenance must be a mapping")
    required = [
        "parameter",
        "value",
        "source",
        "selection_date",
        "inherited_from_submission",
        "legacy_outcomes_seen",
        "tuning_policy",
    ]
    if (
        raw.get("estimand") != "fixed_frozen_protocol_not_per_arm_optimum"
        or raw.get("required_fields") != required
    ):
        raise RevisionProtocolError("Hyperparameter-provenance schema is incomplete")
    expected_values: dict[str, Any] = {
        "hidden_dim": protocol["model"]["hidden_dim"],
        "attention_heads": protocol["model"]["attention_heads"],
        "layers": protocol["model"]["layers"],
        "dropout": protocol["model"]["dropout"],
        "learning_rate": protocol["training"]["optimizer"]["learning_rate"],
        "weight_decay": protocol["training"]["optimizer"]["weight_decay"],
        "maximum_epochs": protocol["training"]["maximum_epochs"],
        "early_stopping_patience": protocol["training"]["early_stopping"]["patience"],
        "coexpression_abs_r_threshold": protocol["graph_composition"]["coexpression"][
            "absolute_pearson_threshold"
        ],
        "string_combined_score_threshold": 400,
        "hidden_normalization": protocol["model"]["hidden_normalization"],
        "gene_identity_dimension": protocol["model"]["gene_identity_embedding"]["dimension"],
        "validation_fraction": protocol["training"]["validation_fraction"],
        "validation_selection_seed": protocol["training"]["validation_selection_seed"],
        "early_stopping_minimum_delta": protocol["training"]["early_stopping"]["minimum_delta"],
        "scheduler_name": protocol["training"]["scheduler"]["name"],
        "scheduler_eta_min": protocol["training"]["scheduler"]["eta_min"],
        "ridge_alpha": protocol["analytic_baselines"]["ridge_alpha"],
        "primary_panel_size": protocol["condition_panel"]["primary_size"],
        "panel_selection_seed": protocol["condition_panel"]["selection_seed"],
        "primary_bootstrap_replicates": protocol["statistics"]["bootstrap_replicates"],
        "primary_bootstrap_random_seed": protocol["statistics"]["bootstrap_random_seed"],
        "precision_interval_width_q90_threshold": protocol["precision_design"][
            "interval_width_gate"
        ],
        "precision_simulation_random_seed": protocol["precision_design"]["simulation_random_seed"],
        "precision_variance_multipliers": [1.0, 1.25],
        "precision_target_cluster_icc_floors": [0.0, 0.25],
        "topology_rewire_seeds": protocol["graph_supports"]["topology_null"]["replicate_seeds"],
        "topology_rewire_multiplier": protocol["graph_supports"]["topology_null"][
            "rewire_multiplier"
        ],
        "topology_minimum_swapped_edge_fraction": protocol["graph_supports"]["topology_null"][
            "minimum_swapped_edge_fraction"
        ],
        "string_release": protocol["graph_composition"]["string_ppi"]["source"],
        "go_edge_policy": "biological_process_shared_annotation_hash_verified_derivation",
    }
    records = raw.get("records")
    if not isinstance(records, list) or len(records) != len(expected_values):
        raise RevisionProtocolError("Hyperparameter-provenance records are incomplete")
    parameters: list[str] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != set(required):
            raise RevisionProtocolError("Each hyperparameter record must have exact fields")
        if isinstance(record.get("inherited_from_submission"), bool) is False:
            raise RevisionProtocolError("inherited_from_submission must be boolean")
        if isinstance(record.get("legacy_outcomes_seen"), bool) is False:
            raise RevisionProtocolError("legacy_outcomes_seen must be boolean")
        if not all(
            isinstance(record.get(key), str) and record[key].strip()
            for key in ["parameter", "source", "selection_date", "tuning_policy"]
        ):
            raise RevisionProtocolError("Hyperparameter provenance text cannot be blank")
        parameter = str(record["parameter"])
        parameters.append(parameter)
        if parameter not in expected_values:
            raise RevisionProtocolError(f"Unexpected hyperparameter provenance row: {parameter}")
        expected_value = expected_values[parameter]
        if (
            type(record.get("value")) is not type(expected_value)
            or record.get("value") != expected_value
        ):
            raise RevisionProtocolError(f"Hyperparameter provenance value drift for {parameter}")
        inherited = record["inherited_from_submission"]
        if not inherited and record["legacy_outcomes_seen"] is not True:
            raise RevisionProtocolError(
                f"Revision-stage provenance must disclose legacy outcomes for {parameter}"
            )
        expected_date = "pre_revision_submission" if inherited else "2026-08-07"
        if record["selection_date"] != expected_date:
            raise RevisionProtocolError(
                f"Hyperparameter provenance date is invalid for {parameter}"
            )
    if len(parameters) != len(set(parameters)):
        raise RevisionProtocolError("Hyperparameter names must be unique")
    if set(parameters) != set(expected_values):
        raise RevisionProtocolError("Hyperparameter-provenance parameter set is incomplete")

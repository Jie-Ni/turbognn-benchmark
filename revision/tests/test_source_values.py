from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cbac_revision.artifacts import canonical_sha256
from cbac_revision.errors import RevisionProtocolError
from cbac_revision.protocol import RESULT_LOCK_IDS
from cbac_revision.source_values import (
    LOCK_MACROS,
    PRIMARY_DECISION_LABELS,
    render_source_values,
    verify_rendered_source_values,
)


def _finalize(payload: dict[str, Any]) -> dict[str, Any]:
    output = dict(payload)
    output["registry_hash"] = canonical_sha256(output)
    return output


def _contrast(name: str, estimate: float) -> dict[str, Any]:
    return {
        "contrast": name,
        "estimate": estimate,
        "uncertainty_interval_95_low": estimate - 0.01,
        "uncertainty_interval_95_high": estimate + 0.01,
        "interval_label": "95% conditional bootstrap uncertainty interval",
        "bh_q_two_contrast_family": 0.04,
    }


def _external_family() -> dict[str, Any]:
    decisions = {"GEARS": "PASS", "scGPT": "EXCLUDED", "Geneformer": "EXCLUDED"}
    validity = _finalize(
        {
            "schema_version": "1.0",
            "registry_id": "EXTERNAL-COMPARATOR-VALIDATION",
            "status": "RELEASED",
            "member_decisions": decisions,
        }
    )
    contrasts = []
    for hvg, estimate in ((200, 0.01), (500, 0.02), (1000, 0.03)):
        contrasts.append(
            {
                "hvg": hvg,
                "estimate": estimate,
                "uncertainty_interval_95_low": estimate - 0.005,
                "uncertainty_interval_95_high": estimate + 0.005,
                "interval_label": "95% conditional bootstrap uncertainty interval",
                "bh_q_three_scale_family": 0.03,
            }
        )
    performance = _finalize(
        {
            "schema_version": "1.0",
            "registry_id": "EXTERNAL-COMPARATOR-SCALE-PERFORMANCE",
            "status": "RELEASED",
            "validity_registry_hash": validity["registry_hash"],
            "adapter_results": {
                "GEARS": {
                    "status": "RELEASED",
                    "contrasts": contrasts,
                    "family_denominator": 3,
                    "adjustment": "benjamini_hochberg_within_adapter",
                },
                "scGPT": {
                    "status": "NOT_APPLICABLE_EXCLUDED",
                    "reason_code": "CLAIM_DELETED",
                },
                "Geneformer": {
                    "status": "NOT_APPLICABLE_EXCLUDED",
                    "reason_code": "CLAIM_DELETED",
                },
            },
        }
    )
    return _finalize(
        {
            "schema_version": "1.0",
            "registry_id": "EXTERNAL-COMPARATOR-VALIDATION",
            "status": "RELEASED",
            "member_decisions": decisions,
            "validity_registry": validity,
            "validity_registry_hash": validity["registry_hash"],
            "performance_registry": performance,
            "performance_registry_hash": performance["registry_hash"],
            "reason_codes": [],
        }
    )


def _external_compute() -> dict[str, Any]:
    return _finalize(
        {
            "schema_version": "1.0",
            "registry_id": "MEASURED-COMPUTE",
            "status": "RELEASED",
            "total_attempts": 10820,
            "failed_attempts": 20,
            "device_hours": 123.456,
            "successful_fit_phase_seconds": {
                "training_wall_seconds": {
                    "n": 10800,
                    "median": 12.345,
                    "q1": 10.0,
                    "q3": 15.0,
                    "iqr": 5.0,
                },
                "inference_wall_seconds": {
                    "n": 10800,
                    "median": 1.234,
                    "q1": 1.0,
                    "q3": 1.5,
                    "iqr": 0.5,
                },
            },
            "peak_device_memory_bytes": {"maximum": 8 * 1024**3},
        }
    )


def _claim_gate(*, baseline_eligible: bool = True, ranking_eligible: bool = True) -> dict[str, Any]:
    return _finalize(
        {
            "schema_version": "1.0",
            "registry_id": "CLAIM-CONSEQUENCE-GATE",
            "status": "RELEASED",
            "baseline_absolute_skill_eligible": baseline_eligible,
            "ranking_consequence_eligible": ranking_eligible,
        }
    )


def _released_registry() -> dict[str, Any]:
    claim_gate = _claim_gate()
    secondary_metrics = []
    for index, metric in enumerate(
        (
            "fisher_z_pearson",
            "spearman_r",
            "mse",
            "top20_absolute_delta_jaccard",
        ),
        start=1,
    ):
        secondary_metrics.append(
            {
                "metric": metric,
                "estimate": 0.001 * index,
                "uncertainty_interval_95_low": -0.001,
                "uncertainty_interval_95_high": 0.005,
                "interval_label": "95% conditional bootstrap uncertainty interval",
                "bh_q_four_metric_family": 0.08,
            }
        )
    locks = {
        "PRIMARY-DELTA-R": _finalize(
            {"registry_id": "PRIMARY-DELTA-R", "status": "RELEASED", "estimate": 0.012345}
        ),
        "PRIMARY-UNCERTAINTY-INTERVAL": _finalize(
            {
                "registry_id": "PRIMARY-UNCERTAINTY-INTERVAL",
                "status": "RELEASED",
                "uncertainty_interval_95_low": 0.0012,
                "uncertainty_interval_95_high": 0.0234,
                "interval_label": "95% conditional bootstrap uncertainty interval",
                "nominal_coverage_claim": False,
            }
        ),
        "PRIMARY-DECISION": _finalize(
            {
                "registry_id": "PRIMARY-DECISION",
                "status": "RELEASED",
                "decision": "DIRECTIONAL_POSITIVE",
                "directional_decision_input": (
                    "unrounded_conditional_uncertainty_interval_vs_zero_only"
                ),
                "claim_scope_inputs": (
                    "direction_plus_absolute_skill_plus_condition_ranking_plus_topology"
                ),
                "archived_optimisation_repeatability_reference": 0.010,
                "archived_reference_has_success_authority": False,
                "topology_directional_support": "TOPOLOGY_DIRECTIONALLY_SUPPORTIVE",
                "topology_diagnostics_status": "TOPOLOGY_DIAGNOSTICS_PASS",
                "topology_registry_hash": "9" * 64,
                "biological_claim_scope": "BIOLOGICAL_EDGE_WORDING_ALLOWED",
                "claim_consequence_gate": claim_gate,
                "claim_consequence_registry_hash": claim_gate["registry_hash"],
            }
        ),
        "PROPAGATION-CONTROL": _finalize(
            {
                "registry_id": "PROPAGATION-CONTROL",
                "status": "RELEASED",
                "contrasts": [
                    _contrast("string_go_minus_self_loop", 0.03),
                    _contrast("dense_minus_self_loop", 0.02),
                ],
            }
        ),
        "SCALE-INTERACTION": _finalize(
            {
                "registry_id": "SCALE-INTERACTION",
                "status": "RELEASED",
                "contrasts": [
                    _contrast("500_minus_200_hvg", -0.004),
                    _contrast("1000_minus_200_hvg", 0.006),
                ],
            }
        ),
        "TOPOLOGY-NULL": _finalize(
            {
                "registry_id": "TOPOLOGY-NULL",
                "status": "RELEASED",
                "analysis_registry_hash": "9" * 64,
                "estimate": 0.007,
                "uncertainty_interval_95_low": 0.001,
                "uncertainty_interval_95_high": 0.015,
                "interval_label": "95% conditional bootstrap uncertainty interval",
                "nominal_coverage_claim": False,
                "directional_support": "TOPOLOGY_DIRECTIONALLY_SUPPORTIVE",
                "diagnostics_status": "TOPOLOGY_DIAGNOSTICS_PASS",
                "local_graph_instance_sd_descriptive": 0.003,
                "two_sided_centered_null_bootstrap_p": 0.12,
            }
        ),
        "REPRESENTATIVENESS-TRIGGER": _finalize(
            {
                "registry_id": "REPRESENTATIVENESS-TRIGGER",
                "status": "RELEASED",
                "execution_state": "NOT_TRIGGERED",
                "global_triggered": False,
            }
        ),
        "CONDITIONAL-PANEL": _finalize(
            {
                "registry_id": "CONDITIONAL-PANEL",
                "status": "RELEASED",
                "execution_status": "NOT_TRIGGERED_NOT_REQUIRED",
            }
        ),
        "FRACTION-IMPROVED": _finalize(
            {
                "registry_id": "FRACTION-IMPROVED",
                "status": "RELEASED",
                "equal_dataset_mean_fraction_conditions_improved": 0.5321,
            }
        ),
        "EMPIRICAL-SEED-RESOLUTION": _finalize(
            {
                "registry_id": "EMPIRICAL-SEED-RESOLUTION",
                "status": "RELEASED",
                "empirical_seed_resolution": {
                    "overall_median_condition_paired_seed_sd": 0.004321,
                    "overall_p95_condition_paired_seed_sd": 0.009876,
                },
            }
        ),
        "SECONDARY-METRICS": _finalize(
            {
                "registry_id": "SECONDARY-METRICS",
                "status": "RELEASED",
                "metrics": secondary_metrics,
                "target_level_primary_sensitivity": {
                    "status": "RELEASED",
                    "estimate": 0.011,
                    "uncertainty_interval_95_low": -0.002,
                    "uncertainty_interval_95_high": 0.022,
                    "interval_label": "95% conditional bootstrap uncertainty interval",
                },
            }
        ),
        "EXTERNAL-COMPARATOR-VALIDATION": _finalize(
            {
                "registry_id": "EXTERNAL-COMPARATOR-VALIDATION",
                "status": "RELEASED",
                "external_registry": _external_family(),
            }
        ),
        "MEASURED-COMPUTE": _finalize(
            {
                "registry_id": "MEASURED-COMPUTE",
                "status": "RELEASED",
                "external_registry": _external_compute(),
            }
        ),
    }
    assert list(locks) == RESULT_LOCK_IDS
    registry: dict[str, Any] = {
        "schema_version": "1.0",
        "package_release_status": "RELEASED",
        "result_lock_count": 13,
        "result_lock_ids": list(RESULT_LOCK_IDS),
        "mandatory_fit_coverage_gate": {"status": "RELEASED"},
        "result_locks": locks,
        "result_lock_hashes": {
            lock_id: locks[lock_id]["registry_hash"] for lock_id in RESULT_LOCK_IDS
        },
        "withheld_result_locks": [],
        "secondary_registry": {"status": "RELEASED"},
        "artifact_count": 10800,
        "artifact_content_manifest_hash": "1" * 64,
        "protocol_source_id": "protocol.yaml",
        "protocol_file_sha256": "2" * 64,
        "release_code_tree_sha256": "3" * 64,
        "global_trigger_manifest_hash": "4" * 64,
        "preflight_summary_manifest_hash": "5" * 64,
        "release_mode": "released",
        "release_asset_manifest_source_id": "release_asset_manifest.json",
        "release_asset_manifest_file_sha256": "6" * 64,
        "release_asset_manifest_self_hash": "7" * 64,
        "release_asset_manifest_mode": "released",
        "main_release_trust_gate": _finalize(
            {
                "registry_id": "MAIN-RELEASE-TRUST-ANCHOR",
                "status": "RELEASED",
            }
        ),
        "main_release_trust_anchor_file_sha256": "8" * 64,
        "cryptographic_boundary": (
            "release_is_valid_only_while_the_caller_pinned_detached_anchor_sha256_is_"
            "immutable_to_the_evidence_bundle_attacker"
        ),
        "release_output_manifest": [],
        "release_output_manifest_hash": canonical_sha256([]),
    }
    _rehash(registry)
    return registry


def _rehash(registry: dict[str, Any]) -> None:
    registry.pop("release_package_hash", None)
    registry.pop("decision_registry_hash", None)
    registry["decision_registry_hash"] = canonical_sha256(registry)
    registry["release_package_hash"] = canonical_sha256(registry)


def _refresh_all_hashes(registry: dict[str, Any]) -> None:
    for lock_id in RESULT_LOCK_IDS:
        lock = registry["result_locks"][lock_id]
        external = lock.get("external_registry")
        if isinstance(external, dict):
            external.pop("registry_hash", None)
            external["registry_hash"] = canonical_sha256(external)
        lock.pop("registry_hash", None)
        lock["registry_hash"] = canonical_sha256(lock)
        registry["result_lock_hashes"][lock_id] = lock["registry_hash"]
    _rehash(registry)


def _set_nested(payload: Any, path: tuple[str | int, ...], value: Any) -> None:
    current = payload
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def _render_test_registry(tmp_path: Path, registry: dict[str, Any]) -> str:
    registry_path = tmp_path / "decision_release_registry.json"
    template_path = tmp_path / "source_values.template.tex"
    output_path = tmp_path / "source_values.tex"
    _write_json(registry_path, registry)
    _template(template_path)
    render_source_values(registry_path, template_path, output_path, mode="released")
    return output_path.read_text(encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _template(path: Path) -> None:
    lines = ["% fixed author-review values"]
    for lock_id in RESULT_LOCK_IDS:
        lines.append(rf"\newcommand{{\{LOCK_MACROS[lock_id]}}}{{UNRENDERED}}")
    lines.append(r"\newcommand{\ResultLockCount}{13}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_released_registry_round_trips_to_exactly_thirteen_macros(tmp_path: Path) -> None:
    registry_path = tmp_path / "decision_release_registry.json"
    template_path = tmp_path / "source_values.template.tex"
    output_path = tmp_path / "source_values.tex"
    manifest_path = tmp_path / "source_values_render_manifest.json"
    _write_json(registry_path, _released_registry())
    _template(template_path)

    manifest = render_source_values(
        registry_path,
        template_path,
        output_path,
        mode="released",
        manifest_path=manifest_path,
    )
    verified = verify_rendered_source_values(registry_path, output_path, manifest_path)
    text = output_path.read_text(encoding="utf-8")

    assert manifest == verified
    assert text.count("\\newcommand") == 14
    assert r"\newcommand{\PrimaryEstimate}{+0.012}" in text
    assert r"\newcommand{\PrimaryUncertaintyInterval}{[+0.001, +0.023]}" in text
    assert (
        r"\newcommand{\PrimaryDecision}{directional positive; biological-edge wording allowed}"
        in text
    )
    assert r"\newcommand{\FractionImprovedResult}{53.2\%}" in text


def test_author_review_mode_renders_only_canonical_lock_placeholders(tmp_path: Path) -> None:
    registry_path = tmp_path / "decision_release_registry.json"
    template_path = tmp_path / "source_values.template.tex"
    output_path = tmp_path / "source_values.tex"
    manifest_path = tmp_path / "source_values_render_manifest.json"
    _write_json(registry_path, _released_registry())
    _template(template_path)

    render_source_values(
        registry_path,
        template_path,
        output_path,
        mode="author-review",
        manifest_path=manifest_path,
    )
    verify_rendered_source_values(registry_path, output_path, manifest_path)
    text = output_path.read_text(encoding="utf-8")

    for lock_id, macro in LOCK_MACROS.items():
        assert rf"\newcommand{{\{macro}}}{{\texttt{{[{lock_id}]}}}}" in text


@pytest.mark.parametrize("mutation", ["missing", "extra", "tampered-lock"])
def test_renderer_rejects_missing_extra_and_hash_tampered_locks(
    tmp_path: Path, mutation: str
) -> None:
    registry = _released_registry()
    if mutation == "missing":
        registry["result_locks"].pop("PRIMARY-DELTA-R")
    elif mutation == "extra":
        registry["result_locks"]["EXTRA"] = _finalize(
            {"registry_id": "EXTRA", "status": "RELEASED"}
        )
    else:
        registry["result_locks"]["PRIMARY-DELTA-R"]["estimate"] = 9.0
    _rehash(registry)
    registry_path = tmp_path / "decision_release_registry.json"
    template_path = tmp_path / "source_values.template.tex"
    _write_json(registry_path, registry)
    _template(template_path)

    with pytest.raises(RevisionProtocolError):
        render_source_values(
            registry_path,
            template_path,
            tmp_path / "source_values.tex",
            mode="released",
        )


def test_renderer_rejects_invalid_decision_even_after_all_hashes_are_refreshed(
    tmp_path: Path,
) -> None:
    registry = _released_registry()
    lock = registry["result_locks"]["PRIMARY-DECISION"]
    lock["decision"] = "BENEFICIAL"
    lock.pop("registry_hash")
    lock["registry_hash"] = canonical_sha256(lock)
    registry["result_lock_hashes"]["PRIMARY-DECISION"] = lock["registry_hash"]
    _rehash(registry)
    registry_path = tmp_path / "decision_release_registry.json"
    template_path = tmp_path / "source_values.template.tex"
    _write_json(registry_path, registry)
    _template(template_path)

    with pytest.raises(RevisionProtocolError, match="SOURCE_VALUES_PRIMARY_DECISION_INVALID"):
        render_source_values(
            registry_path,
            template_path,
            tmp_path / "source_values.tex",
            mode="released",
        )


def test_renderer_recomputes_primary_decision_and_rejects_valid_but_false_token(
    tmp_path: Path,
) -> None:
    registry = _released_registry()
    registry["result_locks"]["PRIMARY-DECISION"]["decision"] = "DIRECTIONAL_NEGATIVE"
    _refresh_all_hashes(registry)

    with pytest.raises(RevisionProtocolError, match="SOURCE_VALUES_PRIMARY_DECISION_MISMATCH"):
        _render_test_registry(tmp_path, registry)


@pytest.mark.parametrize("eligibility_field", ["baseline", "ranking"])
def test_biological_edge_wording_requires_hash_bound_consequence_eligibility(
    tmp_path: Path, eligibility_field: str
) -> None:
    registry = _released_registry()
    decision = registry["result_locks"]["PRIMARY-DECISION"]
    gate = decision["claim_consequence_gate"]
    field = (
        "baseline_absolute_skill_eligible"
        if eligibility_field == "baseline"
        else "ranking_consequence_eligible"
    )
    gate[field] = False
    gate.pop("registry_hash")
    gate["registry_hash"] = canonical_sha256(gate)
    decision["claim_consequence_registry_hash"] = gate["registry_hash"]
    _refresh_all_hashes(registry)

    with pytest.raises(RevisionProtocolError, match="SOURCE_VALUES_BIOLOGICAL_CLAIM_GATE_INVALID"):
        _render_test_registry(tmp_path, registry)

    decision["biological_claim_scope"] = "POSITIVE_EFFECT_WITHOUT_INTERPRETABLE_BENEFIT"
    _refresh_all_hashes(registry)
    rendered = _render_test_registry(tmp_path, registry)
    assert "directional positive" in rendered
    assert "interpretable benefit wording withheld" in rendered


def test_coordinated_consequence_gate_rehash_cannot_preserve_false_biological_claim(
    tmp_path: Path,
) -> None:
    registry = _released_registry()
    decision = registry["result_locks"]["PRIMARY-DECISION"]
    gate = decision["claim_consequence_gate"]
    gate["baseline_absolute_skill_eligible"] = False
    gate["ranking_consequence_eligible"] = False
    gate.pop("registry_hash")
    gate["registry_hash"] = canonical_sha256(gate)
    decision["claim_consequence_registry_hash"] = gate["registry_hash"]
    _refresh_all_hashes(registry)

    with pytest.raises(RevisionProtocolError, match="SOURCE_VALUES_BIOLOGICAL_CLAIM_GATE_INVALID"):
        _render_test_registry(tmp_path, registry)


def test_positive_with_consequences_but_without_topology_allows_only_sparse_wording(
    tmp_path: Path,
) -> None:
    registry = _released_registry()
    topology = registry["result_locks"]["TOPOLOGY-NULL"]
    topology["directional_support"] = "TOPOLOGY_NOT_DIRECTIONALLY_SUPPORTIVE"
    decision = registry["result_locks"]["PRIMARY-DECISION"]
    decision["topology_directional_support"] = "TOPOLOGY_NOT_DIRECTIONALLY_SUPPORTIVE"
    decision["biological_claim_scope"] = "SPARSE_SUPPORT_WORDING_ALLOWED"
    _refresh_all_hashes(registry)

    rendered = _render_test_registry(tmp_path, registry)

    assert "sparse-support wording allowed" in rendered


def test_primary_decision_input_role_tamper_is_rejected_after_full_rehash(
    tmp_path: Path,
) -> None:
    registry = _released_registry()
    registry["result_locks"]["PRIMARY-DECISION"][
        "directional_decision_input"
    ] = "interval_plus_baseline"
    _refresh_all_hashes(registry)
    with pytest.raises(RevisionProtocolError, match="PRIMARY_DECISION_INPUTS_INVALID"):
        _render_test_registry(tmp_path, registry)


def test_renderer_rejects_archived_reference_role_tamper_after_full_rehash(
    tmp_path: Path,
) -> None:
    registry = _released_registry()
    registry["result_locks"]["PRIMARY-DECISION"][
        "archived_optimisation_repeatability_reference"
    ] = 0.020
    _refresh_all_hashes(registry)

    with pytest.raises(
        RevisionProtocolError, match="SOURCE_VALUES_ARCHIVED_REFERENCE_ROLE_INVALID"
    ):
        _render_test_registry(tmp_path, registry)


def test_renderer_rejects_primary_estimate_outside_interval_after_full_rehash(
    tmp_path: Path,
) -> None:
    registry = _released_registry()
    registry["result_locks"]["PRIMARY-DELTA-R"]["estimate"] = 0.03
    _refresh_all_hashes(registry)

    with pytest.raises(
        RevisionProtocolError,
        match="SOURCE_VALUES_PRIMARY_ESTIMATE_OUTSIDE_INTERVAL",
    ):
        _render_test_registry(tmp_path, registry)


@pytest.mark.parametrize(
    ("low", "high", "estimate", "decision"),
    [
        (
            0.0100000001,
            0.020,
            0.015,
            "DIRECTIONAL_POSITIVE",
        ),
        (
            0.010,
            0.020,
            0.015,
            "DIRECTIONAL_POSITIVE",
        ),
        (
            -0.020,
            -0.0000000001,
            -0.010,
            "DIRECTIONAL_NEGATIVE",
        ),
        (
            -0.020,
            0.0,
            -0.010,
            "INCONCLUSIVE",
        ),
        (
            0.0,
            0.009,
            0.001,
            "INCONCLUSIVE",
        ),
        (-0.010, 0.010, 0.0, "INCONCLUSIVE"),
        (-0.010, 0.020, 0.0, "INCONCLUSIVE"),
        (
            0.010,
            0.010,
            0.010,
            "DIRECTIONAL_POSITIVE",
        ),
    ],
    ids=(
        "low-strictly-above-delta",
        "low-equals-delta",
        "high-strictly-below-zero",
        "high-equals-zero",
        "low-equals-zero-upper-below-delta",
        "high-equals-delta",
        "interval-spans-zero-and-delta",
        "point-interval-at-delta",
    ),
)
def test_primary_decision_boundaries_use_unrounded_ordered_rule(
    tmp_path: Path,
    low: float,
    high: float,
    estimate: float,
    decision: str,
) -> None:
    registry = _released_registry()
    registry["result_locks"]["PRIMARY-DELTA-R"]["estimate"] = estimate
    registry["result_locks"]["PRIMARY-UNCERTAINTY-INTERVAL"]["uncertainty_interval_95_low"] = low
    registry["result_locks"]["PRIMARY-UNCERTAINTY-INTERVAL"]["uncertainty_interval_95_high"] = high
    registry["result_locks"]["PRIMARY-DECISION"]["decision"] = decision
    registry["result_locks"]["PRIMARY-DECISION"]["biological_claim_scope"] = {
        "DIRECTIONAL_POSITIVE": "BIOLOGICAL_EDGE_WORDING_ALLOWED",
        "DIRECTIONAL_NEGATIVE": "NEGATIVE_DIRECTIONAL_EFFECT",
        "INCONCLUSIVE": "BENEFIT_NOT_DEMONSTRATED",
    }[decision]
    _refresh_all_hashes(registry)

    text = _render_test_registry(tmp_path, registry)

    assert PRIMARY_DECISION_LABELS[decision] in text


@pytest.mark.parametrize(
    "path",
    [
        ("result_locks", "PRIMARY-DELTA-R", "estimate"),
        (
            "result_locks",
            "PRIMARY-UNCERTAINTY-INTERVAL",
            "uncertainty_interval_95_low",
        ),
        (
            "result_locks",
            "PRIMARY-DECISION",
            "archived_optimisation_repeatability_reference",
        ),
        ("result_locks", "PROPAGATION-CONTROL", "contrasts", 0, "estimate"),
        (
            "result_locks",
            "PROPAGATION-CONTROL",
            "contrasts",
            0,
            "bh_q_two_contrast_family",
        ),
        (
            "result_locks",
            "SCALE-INTERACTION",
            "contrasts",
            1,
            "uncertainty_interval_95_high",
        ),
        ("result_locks", "TOPOLOGY-NULL", "local_graph_instance_sd_descriptive"),
        ("result_locks", "TOPOLOGY-NULL", "two_sided_centered_null_bootstrap_p"),
        (
            "result_locks",
            "FRACTION-IMPROVED",
            "equal_dataset_mean_fraction_conditions_improved",
        ),
        (
            "result_locks",
            "EMPIRICAL-SEED-RESOLUTION",
            "empirical_seed_resolution",
            "overall_p95_condition_paired_seed_sd",
        ),
        ("result_locks", "SECONDARY-METRICS", "metrics", 0, "estimate"),
        (
            "result_locks",
            "SECONDARY-METRICS",
            "metrics",
            1,
            "bh_q_four_metric_family",
        ),
        (
            "result_locks",
            "SECONDARY-METRICS",
            "target_level_primary_sensitivity",
            "uncertainty_interval_95_low",
        ),
        (
            "result_locks",
            "MEASURED-COMPUTE",
            "external_registry",
            "successful_fit_phase_seconds",
            "training_wall_seconds",
            "n",
        ),
        (
            "result_locks",
            "MEASURED-COMPUTE",
            "external_registry",
            "successful_fit_phase_seconds",
            "inference_wall_seconds",
            "median",
        ),
        (
            "result_locks",
            "MEASURED-COMPUTE",
            "external_registry",
            "peak_device_memory_bytes",
            "maximum",
        ),
        (
            "result_locks",
            "MEASURED-COMPUTE",
            "external_registry",
            "failed_attempts",
        ),
        (
            "result_locks",
            "MEASURED-COMPUTE",
            "external_registry",
            "device_hours",
        ),
    ],
)
def test_renderer_rejects_numeric_strings_across_result_locks_after_full_rehash(
    tmp_path: Path,
    path: tuple[str | int, ...],
) -> None:
    registry = _released_registry()
    current: Any = registry
    for key in path:
        current = current[key]
    _set_nested(registry, path, str(current))
    _refresh_all_hashes(registry)

    with pytest.raises(RevisionProtocolError):
        _render_test_registry(tmp_path, registry)


def test_renderer_rejects_numeric_string_in_triggered_conditional_lock(
    tmp_path: Path,
) -> None:
    registry = _released_registry()
    conditional = registry["result_locks"]["CONDITIONAL-PANEL"]
    conditional.update(
        {
            "execution_status": "TRIGGERED_AND_COMPLETE",
            "estimate": "0.004",
            "uncertainty_interval_95_low": -0.002,
            "uncertainty_interval_95_high": 0.010,
            "interval_label": "95% conditional bootstrap uncertainty interval",
        }
    )
    _refresh_all_hashes(registry)

    with pytest.raises(RevisionProtocolError, match="SOURCE_VALUES_NUMERIC_FIELD_INVALID"):
        _render_test_registry(tmp_path, registry)


@pytest.mark.parametrize(
    "path",
    [
        ("result_locks", "PRIMARY-DELTA-R", "estimate"),
        (
            "result_locks",
            "PROPAGATION-CONTROL",
            "contrasts",
            0,
            "bh_q_two_contrast_family",
        ),
        (
            "result_locks",
            "MEASURED-COMPUTE",
            "external_registry",
            "successful_fit_phase_seconds",
            "training_wall_seconds",
            "n",
        ),
    ],
)
def test_renderer_rejects_boolean_numeric_impostors_after_full_rehash(
    tmp_path: Path,
    path: tuple[str | int, ...],
) -> None:
    registry = _released_registry()
    _set_nested(registry, path, True)
    _refresh_all_hashes(registry)

    with pytest.raises(RevisionProtocolError):
        _render_test_registry(tmp_path, registry)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_renderer_rejects_nonfinite_json_numbers_before_hash_validation(
    tmp_path: Path,
    literal: str,
) -> None:
    registry_path = tmp_path / "decision_release_registry.json"
    template_path = tmp_path / "source_values.template.tex"
    text = json.dumps(_released_registry(), indent=2, sort_keys=True, allow_nan=False)
    text = text.replace('"estimate": 0.012345', f'"estimate": {literal}', 1)
    registry_path.write_text(text + "\n", encoding="utf-8")
    _template(template_path)

    with pytest.raises(RevisionProtocolError):
        render_source_values(
            registry_path,
            template_path,
            tmp_path / "source_values.tex",
            mode="released",
        )


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        (
            (
                "result_locks",
                "PROPAGATION-CONTROL",
                "contrasts",
                0,
                "bh_q_two_contrast_family",
            ),
            1.001,
            "SOURCE_VALUES_PROBABILITY_INVALID",
        ),
        (
            ("result_locks", "TOPOLOGY-NULL", "local_graph_instance_sd_descriptive"),
            -0.001,
            "SOURCE_VALUES_NUMERIC_FIELD_INVALID",
        ),
        (
            (
                "result_locks",
                "MEASURED-COMPUTE",
                "external_registry",
                "failed_attempts",
            ),
            10821,
            "SOURCE_VALUES_COMPUTE_COUNT_MISMATCH",
        ),
        (
            (
                "result_locks",
                "MEASURED-COMPUTE",
                "external_registry",
                "successful_fit_phase_seconds",
                "training_wall_seconds",
                "n",
            ),
            10800.5,
            "SOURCE_VALUES_INTEGER_FIELD_INVALID",
        ),
        (
            (
                "result_locks",
                "MEASURED-COMPUTE",
                "external_registry",
                "successful_fit_phase_seconds",
                "training_wall_seconds",
                "iqr",
            ),
            4.999,
            "SOURCE_VALUES_COMPUTE_DISTRIBUTION_INVALID",
        ),
    ],
)
def test_renderer_enforces_numeric_ranges_and_exact_integer_or_distribution_semantics(
    tmp_path: Path,
    path: tuple[str | int, ...],
    value: float,
    error: str,
) -> None:
    registry = _released_registry()
    _set_nested(registry, path, value)
    _refresh_all_hashes(registry)

    with pytest.raises(RevisionProtocolError, match=error):
        _render_test_registry(tmp_path, registry)


def test_renderer_uses_explicit_half_up_display_rounding(tmp_path: Path) -> None:
    registry = _released_registry()
    registry["result_locks"]["PRIMARY-DELTA-R"]["estimate"] = 0.0125
    registry["result_locks"]["FRACTION-IMPROVED"][
        "equal_dataset_mean_fraction_conditions_improved"
    ] = 0.5325
    _refresh_all_hashes(registry)

    text = _render_test_registry(tmp_path, registry)

    assert r"\newcommand{\PrimaryEstimate}{+0.013}" in text
    assert r"\newcommand{\FractionImprovedResult}{53.3\%}" in text


def test_round_trip_verification_rejects_manual_macro_edit(tmp_path: Path) -> None:
    registry_path = tmp_path / "decision_release_registry.json"
    template_path = tmp_path / "source_values.template.tex"
    output_path = tmp_path / "source_values.tex"
    manifest_path = tmp_path / "source_values_render_manifest.json"
    _write_json(registry_path, _released_registry())
    _template(template_path)
    render_source_values(
        registry_path,
        template_path,
        output_path,
        mode="released",
        manifest_path=manifest_path,
    )
    output_path.write_text(
        output_path.read_text(encoding="utf-8").replace("+0.012", "+9.999"),
        encoding="utf-8",
    )

    with pytest.raises(RevisionProtocolError, match="SOURCE_VALUES_RENDER_MANIFEST_MISMATCH"):
        verify_rendered_source_values(registry_path, output_path, manifest_path)

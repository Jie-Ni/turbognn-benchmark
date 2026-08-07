"""Render the 13 self-hashed result locks into TeX without manual transcription."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .artifacts import canonical_sha256, file_sha256
from .errors import RevisionProtocolError
from .protocol import RESULT_LOCK_IDS

LOCK_MACROS = {
    "PRIMARY-DELTA-R": "PrimaryEstimate",
    "PRIMARY-UNCERTAINTY-INTERVAL": "PrimaryUncertaintyInterval",
    "PRIMARY-DECISION": "PrimaryDecision",
    "PROPAGATION-CONTROL": "PropagationControlResult",
    "SCALE-INTERACTION": "ScaleInteraction",
    "TOPOLOGY-NULL": "TopologyNullResult",
    "REPRESENTATIVENESS-TRIGGER": "RepresentativenessTrigger",
    "CONDITIONAL-PANEL": "ConditionalPanelResult",
    "FRACTION-IMPROVED": "FractionImprovedResult",
    "EMPIRICAL-SEED-RESOLUTION": "EmpiricalSeedResolution",
    "SECONDARY-METRICS": "SecondaryMetricsResult",
    "EXTERNAL-COMPARATOR-VALIDATION": "ExternalComparatorValidationResult",
    "MEASURED-COMPUTE": "MeasuredComputeResult",
}
PRIMARY_DECISION_LABELS = {
    "DIRECTIONAL_POSITIVE": "directional positive",
    "DIRECTIONAL_NEGATIVE": "directional negative",
    "INCONCLUSIVE": "inconclusive",
}
BIOLOGICAL_CLAIM_SCOPES = {
    "BIOLOGICAL_EDGE_WORDING_ALLOWED",
    "SPARSE_SUPPORT_WORDING_ALLOWED",
    "POSITIVE_EFFECT_WITHOUT_INTERPRETABLE_BENEFIT",
    "NEGATIVE_DIRECTIONAL_EFFECT",
    "BENEFIT_NOT_DEMONSTRATED",
}
REGISTRY_FIELDS = {
    "schema_version",
    "package_release_status",
    "result_lock_count",
    "result_lock_ids",
    "mandatory_fit_coverage_gate",
    "result_locks",
    "result_lock_hashes",
    "withheld_result_locks",
    "secondary_registry",
    "artifact_count",
    "artifact_content_manifest_hash",
    "protocol_source_id",
    "protocol_file_sha256",
    "release_code_tree_sha256",
    "global_trigger_manifest_hash",
    "preflight_summary_manifest_hash",
    "release_mode",
    "release_asset_manifest_source_id",
    "release_asset_manifest_file_sha256",
    "release_asset_manifest_self_hash",
    "release_asset_manifest_mode",
    "main_release_trust_gate",
    "main_release_trust_anchor_file_sha256",
    "cryptographic_boundary",
    "release_output_manifest",
    "release_output_manifest_hash",
    "decision_registry_hash",
    "release_package_hash",
}
RENDER_MANIFEST_FIELDS = {
    "schema_version",
    "mode",
    "source_registry_id",
    "source_registry_sha256",
    "decision_registry_hash",
    "input_template_sha256",
    "rendered_source_values_sha256",
    "lock_macro_count",
    "lock_macro_ids",
    "lock_macro_values_hash",
    "render_manifest_hash",
}
MARKER_RE = re.compile(r"^% AUTO-GENERATED RESULT LOCKS \|.*$", re.MULTILINE)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def render_source_values(
    registry_path: Path,
    template_path: Path,
    output_path: Path,
    *,
    mode: str,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Validate a release registry and atomically render its 13 TeX lock macros."""

    registry = _read_registry(registry_path)
    values = _macro_values(registry, mode)
    template = template_path.read_text(encoding="utf-8")
    rendered = _render_text(
        template,
        values,
        mode=mode,
        registry_sha256=file_sha256(registry_path),
    )
    _atomic_write_text(output_path, rendered)
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "mode": mode,
        "source_registry_id": registry_path.name,
        "source_registry_sha256": file_sha256(registry_path),
        "decision_registry_hash": registry["decision_registry_hash"],
        "input_template_sha256": file_sha256(template_path),
        "rendered_source_values_sha256": file_sha256(output_path),
        "lock_macro_count": len(values),
        "lock_macro_ids": list(RESULT_LOCK_IDS),
        "lock_macro_values_hash": canonical_sha256(values),
    }
    manifest["render_manifest_hash"] = canonical_sha256(manifest)
    if manifest_path is not None:
        _atomic_write_json(manifest_path, manifest)
    return manifest


def verify_rendered_source_values(
    registry_path: Path,
    rendered_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Round-trip the rendered file against its registry and self-hashed render manifest."""

    registry = _read_registry(registry_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RevisionProtocolError("[SOURCE_VALUES_RENDER_MANIFEST_INVALID]") from error
    if not isinstance(manifest, dict) or set(manifest) != RENDER_MANIFEST_FIELDS:
        raise RevisionProtocolError("[SOURCE_VALUES_RENDER_MANIFEST_INVALID]")
    payload = dict(manifest)
    declared_hash = payload.pop("render_manifest_hash", None)
    mode = manifest.get("mode")
    values = _macro_values(registry, str(mode))
    valid = (
        manifest.get("schema_version") == "1.0"
        and declared_hash == canonical_sha256(payload)
        and manifest.get("source_registry_id") == registry_path.name
        and manifest.get("source_registry_sha256") == file_sha256(registry_path)
        and manifest.get("decision_registry_hash") == registry["decision_registry_hash"]
        and manifest.get("rendered_source_values_sha256") == file_sha256(rendered_path)
        and manifest.get("lock_macro_count") == 13
        and manifest.get("lock_macro_ids") == list(RESULT_LOCK_IDS)
        and manifest.get("lock_macro_values_hash") == canonical_sha256(values)
    )
    if not valid:
        raise RevisionProtocolError("[SOURCE_VALUES_RENDER_MANIFEST_MISMATCH]")
    text = rendered_path.read_text(encoding="utf-8")
    rerendered = _render_text(
        text,
        values,
        mode=str(mode),
        registry_sha256=file_sha256(registry_path),
    )
    if rerendered != text:
        raise RevisionProtocolError("[SOURCE_VALUES_ROUND_TRIP_MISMATCH]")
    return manifest


def _read_registry(path: Path) -> dict[str, Any]:
    try:
        registry = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RevisionProtocolError("[SOURCE_VALUES_REGISTRY_INVALID]") from error
    _reject_nonfinite_json_numbers(registry)
    if not isinstance(registry, dict) or set(registry) != REGISTRY_FIELDS:
        raise RevisionProtocolError("[SOURCE_VALUES_REGISTRY_SCHEMA_MISMATCH]")
    release_payload = dict(registry)
    declared_release_hash = release_payload.pop("release_package_hash", None)
    decision_payload = dict(release_payload)
    declared_decision_hash = decision_payload.pop("decision_registry_hash", None)
    if (
        registry.get("schema_version") != "1.0"
        or declared_release_hash != canonical_sha256(release_payload)
        or declared_decision_hash != canonical_sha256(decision_payload)
        or registry.get("result_lock_count") != 13
        or registry.get("result_lock_ids") != list(RESULT_LOCK_IDS)
    ):
        raise RevisionProtocolError("[SOURCE_VALUES_REGISTRY_HASH_OR_COUNT_MISMATCH]")
    locks = registry.get("result_locks")
    hashes = registry.get("result_lock_hashes")
    if (
        not isinstance(locks, Mapping)
        or set(locks) != set(RESULT_LOCK_IDS)
        or not isinstance(hashes, Mapping)
        or set(hashes) != set(RESULT_LOCK_IDS)
    ):
        raise RevisionProtocolError("[SOURCE_VALUES_LOCK_ORDER_MISMATCH]")
    withheld = []
    for lock_id in RESULT_LOCK_IDS:
        lock = locks[lock_id]
        if not isinstance(lock, Mapping) or lock.get("registry_id") != lock_id:
            raise RevisionProtocolError(f"[SOURCE_VALUES_LOCK_SCHEMA_MISMATCH] {lock_id}")
        payload = dict(lock)
        lock_hash = payload.pop("registry_hash", None)
        if lock_hash != canonical_sha256(payload) or hashes[lock_id] != lock_hash:
            raise RevisionProtocolError(f"[SOURCE_VALUES_LOCK_HASH_MISMATCH] {lock_id}")
        if lock.get("status") not in {"RELEASED", "WITHHELD"}:
            raise RevisionProtocolError(f"[SOURCE_VALUES_LOCK_STATUS_INVALID] {lock_id}")
        if lock.get("status") != "RELEASED":
            withheld.append(lock_id)
    if registry.get("withheld_result_locks") != withheld:
        raise RevisionProtocolError("[SOURCE_VALUES_WITHHELD_LOCK_LIST_MISMATCH]")
    trust_gate = registry.get("main_release_trust_gate")
    trust_payload = dict(trust_gate) if isinstance(trust_gate, Mapping) else {}
    trust_hash = trust_payload.pop("registry_hash", None)
    if (
        trust_hash != canonical_sha256(trust_payload)
        or not _is_sha256(registry.get("release_asset_manifest_file_sha256"))
        or not _is_sha256(registry.get("release_asset_manifest_self_hash"))
        or registry.get("cryptographic_boundary")
        != (
            "release_is_valid_only_while_the_caller_pinned_detached_anchor_sha256_is_"
            "immutable_to_the_evidence_bundle_attacker"
        )
    ):
        raise RevisionProtocolError("[SOURCE_VALUES_RELEASE_TRUST_BINDING_INVALID]")
    trust_released = (
        isinstance(trust_gate, Mapping)
        and trust_gate.get("registry_id") == "MAIN-RELEASE-TRUST-ANCHOR"
        and trust_gate.get("status") == "RELEASED"
        and registry.get("release_mode") == "released"
        and registry.get("release_asset_manifest_mode") == "released"
        and _is_sha256(registry.get("main_release_trust_anchor_file_sha256"))
    )
    released = not withheld and trust_released
    if registry.get("package_release_status") != ("RELEASED" if released else "WITHHELD"):
        raise RevisionProtocolError("[SOURCE_VALUES_PACKAGE_STATUS_MISMATCH]")
    return registry


def _macro_values(registry: Mapping[str, Any], mode: str) -> dict[str, str]:
    if mode not in {"author-review", "released"}:
        raise RevisionProtocolError("[SOURCE_VALUES_MODE_INVALID]")
    if mode == "author-review":
        return {lock_id: rf"\texttt{{[{lock_id}]}}" for lock_id in RESULT_LOCK_IDS}
    if registry.get("package_release_status") != "RELEASED" or registry.get(
        "withheld_result_locks"
    ):
        raise RevisionProtocolError("[SOURCE_VALUES_RELEASED_MODE_REQUIRES_RELEASED_PACKAGE]")
    locks = registry["result_locks"]
    _validate_primary_semantics(locks)
    renderers: dict[str, Callable[[Mapping[str, Any]], str]] = {
        "PRIMARY-DELTA-R": _render_primary_estimate,
        "PRIMARY-UNCERTAINTY-INTERVAL": _render_primary_interval,
        "PRIMARY-DECISION": _render_primary_decision,
        "PROPAGATION-CONTROL": _render_propagation,
        "SCALE-INTERACTION": _render_scale,
        "TOPOLOGY-NULL": _render_topology,
        "REPRESENTATIVENESS-TRIGGER": _render_trigger,
        "CONDITIONAL-PANEL": _render_conditional,
        "FRACTION-IMPROVED": _render_fraction,
        "EMPIRICAL-SEED-RESOLUTION": _render_seed_resolution,
        "SECONDARY-METRICS": _render_secondary,
        "EXTERNAL-COMPARATOR-VALIDATION": _render_external_comparators,
        "MEASURED-COMPUTE": _render_compute,
    }
    return {lock_id: renderers[lock_id](locks[lock_id]) for lock_id in RESULT_LOCK_IDS}


def _render_text(
    template: str,
    values: Mapping[str, str],
    *,
    mode: str,
    registry_sha256: str,
) -> str:
    output = template
    marker = (
        f"% AUTO-GENERATED RESULT LOCKS | mode={mode} | " f"registry_file_sha256={registry_sha256}"
    )
    if MARKER_RE.search(output):
        output = MARKER_RE.sub(marker, output, count=1)
    else:
        output = marker + "\n" + output
    for lock_id, macro in LOCK_MACROS.items():
        pattern = re.compile(rf"^\\newcommand\{{\\{re.escape(macro)}\}}\{{.*\}}$", re.MULTILINE)
        matches = pattern.findall(output)
        if len(matches) != 1:
            raise RevisionProtocolError(f"[SOURCE_VALUES_LOCK_MACRO_COUNT_INVALID] {macro}")
        replacement = rf"\newcommand{{\{macro}}}{{{values[lock_id]}}}"
        output = pattern.sub(lambda _: replacement, output, count=1)
    count_pattern = re.compile(r"^\\newcommand\{\\ResultLockCount\}\{.*\}$", re.MULTILINE)
    if len(count_pattern.findall(output)) != 1:
        raise RevisionProtocolError("[SOURCE_VALUES_LOCK_COUNT_MACRO_INVALID]")
    output = count_pattern.sub(lambda _: r"\newcommand{\ResultLockCount}{13}", output, count=1)
    return output if output.endswith("\n") else output + "\n"


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON numeric constant: {value}")


def _reject_nonfinite_json_numbers(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RevisionProtocolError("[SOURCE_VALUES_NONFINITE_JSON_NUMBER]")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_nonfinite_json_numbers(item)
    elif isinstance(value, list):
        for item in value:
            _reject_nonfinite_json_numbers(item)


def _number(
    lock: Mapping[str, Any],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = lock.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RevisionProtocolError(f"[SOURCE_VALUES_NUMERIC_FIELD_INVALID] {key}")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise RevisionProtocolError(f"[SOURCE_VALUES_NUMERIC_FIELD_INVALID] {key}") from error
    if (
        not math.isfinite(number)
        or (minimum is not None and number < minimum)
        or (maximum is not None and number > maximum)
    ):
        raise RevisionProtocolError(f"[SOURCE_VALUES_NUMERIC_FIELD_INVALID] {key}")
    return number


def _integer(
    lock: Mapping[str, Any],
    key: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = lock.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RevisionProtocolError(f"[SOURCE_VALUES_INTEGER_FIELD_INVALID] {key}")
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise RevisionProtocolError(f"[SOURCE_VALUES_INTEGER_FIELD_INVALID] {key}")
    return value


def _interval(
    lock: Mapping[str, Any],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float, float]:
    low = _number(lock, "uncertainty_interval_95_low", minimum=minimum, maximum=maximum)
    high = _number(lock, "uncertainty_interval_95_high", minimum=minimum, maximum=maximum)
    if low > high:
        raise RevisionProtocolError("[SOURCE_VALUES_INTERVAL_INVALID]")
    if lock.get("interval_label") != "95% conditional bootstrap uncertainty interval":
        raise RevisionProtocolError("[SOURCE_VALUES_INTERVAL_LABEL_INVALID]")
    if "nominal_coverage_claim" in lock and lock.get("nominal_coverage_claim") is not False:
        raise RevisionProtocolError("[SOURCE_VALUES_NOMINAL_COVERAGE_CLAIM_PROHIBITED]")
    return low, high


def _signed(value: float, digits: int = 3) -> str:
    return _fixed(value, digits, signed=True)


def _fixed(value: float, digits: int, *, signed: bool = False) -> str:
    quantum = Decimal(1).scaleb(-digits)
    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    sign = "+" if signed else ""
    return format(rounded, f"{sign}.{digits}f")


def _format_interval(low: float, high: float, digits: int = 3) -> str:
    return f"[{_signed(low, digits)}, {_signed(high, digits)}]"


def _probability(lock: Mapping[str, Any], key: str) -> float:
    value = _number(lock, key, minimum=0.0)
    if value > 1.0:
        raise RevisionProtocolError(f"[SOURCE_VALUES_PROBABILITY_INVALID] {key}")
    return value


def _render_primary_estimate(lock: Mapping[str, Any]) -> str:
    return _signed(_number(lock, "estimate", minimum=-2.0, maximum=2.0))


def _render_primary_interval(lock: Mapping[str, Any]) -> str:
    return _format_interval(*_interval(lock, minimum=-2.0, maximum=2.0))


def _render_primary_decision(lock: Mapping[str, Any]) -> str:
    token = lock.get("decision")
    if token not in PRIMARY_DECISION_LABELS:
        raise RevisionProtocolError("[SOURCE_VALUES_PRIMARY_DECISION_INVALID]")
    scope = lock.get("biological_claim_scope")
    scope_labels = {
        "BIOLOGICAL_EDGE_WORDING_ALLOWED": "biological-edge wording allowed",
        "SPARSE_SUPPORT_WORDING_ALLOWED": "sparse-support wording allowed",
        "POSITIVE_EFFECT_WITHOUT_INTERPRETABLE_BENEFIT": ("interpretable benefit wording withheld"),
        "NEGATIVE_DIRECTIONAL_EFFECT": "negative directional effect",
        "BENEFIT_NOT_DEMONSTRATED": "benefit not demonstrated",
    }
    if scope not in scope_labels:
        raise RevisionProtocolError("[SOURCE_VALUES_BIOLOGICAL_CLAIM_GATE_INVALID]")
    return f"{PRIMARY_DECISION_LABELS[str(token)]}; {scope_labels[str(scope)]}"


def _validate_primary_semantics(locks: Mapping[str, Any]) -> None:
    estimate_lock = locks["PRIMARY-DELTA-R"]
    interval_lock = locks["PRIMARY-UNCERTAINTY-INTERVAL"]
    decision_lock = locks["PRIMARY-DECISION"]
    estimate = _number(estimate_lock, "estimate", minimum=-2.0, maximum=2.0)
    low, high = _interval(interval_lock, minimum=-2.0, maximum=2.0)
    if estimate < low or estimate > high:
        raise RevisionProtocolError("[SOURCE_VALUES_PRIMARY_ESTIMATE_OUTSIDE_INTERVAL]")
    archived_reference = _number(decision_lock, "archived_optimisation_repeatability_reference")
    if (
        archived_reference != 0.010
        or decision_lock.get("archived_reference_has_success_authority") is not False
    ):
        raise RevisionProtocolError("[SOURCE_VALUES_ARCHIVED_REFERENCE_ROLE_INVALID]")
    if (
        decision_lock.get("directional_decision_input")
        != "unrounded_conditional_uncertainty_interval_vs_zero_only"
        or decision_lock.get("claim_scope_inputs")
        != "direction_plus_absolute_skill_plus_condition_ranking_plus_topology"
    ):
        raise RevisionProtocolError("[SOURCE_VALUES_PRIMARY_DECISION_INPUTS_INVALID]")
    expected = _primary_decision_from_interval(low, high)
    observed = decision_lock.get("decision")
    if observed not in PRIMARY_DECISION_LABELS:
        raise RevisionProtocolError("[SOURCE_VALUES_PRIMARY_DECISION_INVALID]")
    if observed != expected:
        raise RevisionProtocolError(
            "[SOURCE_VALUES_PRIMARY_DECISION_MISMATCH] " f"expected={expected};observed={observed}"
        )
    topology = locks["TOPOLOGY-NULL"]
    topology_direction = topology.get("directional_support")
    topology_diagnostics = topology.get("diagnostics_status")
    if (
        decision_lock.get("topology_directional_support") != topology_direction
        or decision_lock.get("topology_diagnostics_status") != topology_diagnostics
        or decision_lock.get("topology_registry_hash") != topology.get("analysis_registry_hash")
    ):
        raise RevisionProtocolError("[SOURCE_VALUES_CROSS_LOCK_TOPOLOGY_BINDING_INVALID]")
    consequence = decision_lock.get("claim_consequence_gate")
    if not isinstance(consequence, Mapping):
        raise RevisionProtocolError("[SOURCE_VALUES_CLAIM_CONSEQUENCE_GATE_INVALID]")
    consequence_payload = dict(consequence)
    consequence_hash = consequence_payload.pop("registry_hash", None)
    if (
        consequence.get("registry_id") != "CLAIM-CONSEQUENCE-GATE"
        or consequence.get("status") != "RELEASED"
        or consequence_hash != canonical_sha256(consequence_payload)
        or decision_lock.get("claim_consequence_registry_hash") != consequence_hash
        or not isinstance(consequence.get("baseline_absolute_skill_eligible"), bool)
        or not isinstance(consequence.get("ranking_consequence_eligible"), bool)
    ):
        raise RevisionProtocolError("[SOURCE_VALUES_CLAIM_CONSEQUENCE_GATE_INVALID]")
    consequence_supportive = bool(consequence["baseline_absolute_skill_eligible"]) and bool(
        consequence["ranking_consequence_eligible"]
    )
    if observed == "DIRECTIONAL_POSITIVE" and (
        topology_direction == "TOPOLOGY_DIRECTIONALLY_SUPPORTIVE"
        and topology_diagnostics == "TOPOLOGY_DIAGNOSTICS_PASS"
        and consequence_supportive
    ):
        expected_claim_scope = "BIOLOGICAL_EDGE_WORDING_ALLOWED"
    elif observed == "DIRECTIONAL_POSITIVE" and consequence_supportive:
        expected_claim_scope = "SPARSE_SUPPORT_WORDING_ALLOWED"
    elif observed == "DIRECTIONAL_POSITIVE":
        expected_claim_scope = "POSITIVE_EFFECT_WITHOUT_INTERPRETABLE_BENEFIT"
    elif observed == "DIRECTIONAL_NEGATIVE":
        expected_claim_scope = "NEGATIVE_DIRECTIONAL_EFFECT"
    else:
        expected_claim_scope = "BENEFIT_NOT_DEMONSTRATED"
    claim_scope = decision_lock.get("biological_claim_scope")
    if claim_scope not in BIOLOGICAL_CLAIM_SCOPES or claim_scope != expected_claim_scope:
        raise RevisionProtocolError("[SOURCE_VALUES_BIOLOGICAL_CLAIM_GATE_INVALID]")


def _primary_decision_from_interval(low: float, high: float) -> str:
    if low > 0.0:
        return "DIRECTIONAL_POSITIVE"
    if high < 0.0:
        return "DIRECTIONAL_NEGATIVE"
    return "INCONCLUSIVE"


def _render_contrasts(lock: Mapping[str, Any], expected: Sequence[str], q_field: str) -> str:
    contrasts = lock.get("contrasts")
    if (
        not isinstance(contrasts, list)
        or not all(isinstance(row, Mapping) for row in contrasts)
        or [row.get("contrast") for row in contrasts] != list(expected)
    ):
        raise RevisionProtocolError("[SOURCE_VALUES_CONTRAST_FAMILY_INVALID]")
    rendered = []
    for row in contrasts:
        low, high = _interval(row)
        label = str(row["contrast"]).replace("_", "-")
        rendered.append(
            f"{label} {_signed(_number(row, 'estimate'))} "
            f"{_format_interval(low, high)}; BH q={_fixed(_probability(row, q_field), 3)}"
        )
    return "; ".join(rendered)


def _render_propagation(lock: Mapping[str, Any]) -> str:
    return _render_contrasts(
        lock,
        ("string_go_minus_self_loop", "dense_minus_self_loop"),
        "bh_q_two_contrast_family",
    )


def _render_scale(lock: Mapping[str, Any]) -> str:
    return _render_contrasts(
        lock,
        ("500_minus_200_hvg", "1000_minus_200_hvg"),
        "bh_q_two_contrast_family",
    )


def _render_topology(lock: Mapping[str, Any]) -> str:
    low, high = _interval(lock)
    return (
        f"STRING-GO-minus-rewire {_signed(_number(lock, 'estimate'))} "
        f"{_format_interval(low, high)}; graph-instance SD "
        f"{_fixed(_number(lock, 'local_graph_instance_sd_descriptive', minimum=0.0), 3)}; "
        f"p={_fixed(_probability(lock, 'two_sided_centered_null_bootstrap_p'), 3)}"
    )


def _render_trigger(lock: Mapping[str, Any]) -> str:
    triggered = lock.get("global_triggered")
    state = lock.get("execution_state")
    if not isinstance(triggered, bool) or state != ("TRIGGERED" if triggered else "NOT_TRIGGERED"):
        raise RevisionProtocolError("[SOURCE_VALUES_TRIGGER_STATE_INVALID]")
    return str(state).replace("_", r"\_")


def _render_conditional(lock: Mapping[str, Any]) -> str:
    status = lock.get("execution_status")
    if status == "NOT_TRIGGERED_NOT_REQUIRED":
        return r"NOT\_TRIGGERED (not required)"
    if status != "TRIGGERED_AND_COMPLETE":
        raise RevisionProtocolError("[SOURCE_VALUES_CONDITIONAL_STATE_INVALID]")
    low, high = _interval(lock)
    return f"{_signed(_number(lock, 'estimate'))} {_format_interval(low, high)}"


def _render_fraction(lock: Mapping[str, Any]) -> str:
    value = _probability(lock, "equal_dataset_mean_fraction_conditions_improved")
    return f"{_fixed(100.0 * value, 1)}\\%"


def _render_seed_resolution(lock: Mapping[str, Any]) -> str:
    resolution = lock.get("empirical_seed_resolution")
    if not isinstance(resolution, Mapping):
        raise RevisionProtocolError("[SOURCE_VALUES_SEED_RESOLUTION_INVALID]")
    median = _number(resolution, "overall_median_condition_paired_seed_sd", minimum=0.0)
    p95 = _number(resolution, "overall_p95_condition_paired_seed_sd", minimum=0.0)
    if median > p95:
        raise RevisionProtocolError("[SOURCE_VALUES_SEED_RESOLUTION_INVALID]")
    return f"median SD {_fixed(median, 4)}; 95th percentile SD {_fixed(p95, 4)}"


def _render_secondary(lock: Mapping[str, Any]) -> str:
    expected = (
        "fisher_z_pearson",
        "spearman_r",
        "mse",
        "top20_absolute_delta_jaccard",
    )
    metrics = lock.get("metrics")
    if (
        not isinstance(metrics, list)
        or not all(isinstance(row, Mapping) for row in metrics)
        or [row.get("metric") for row in metrics] != list(expected)
    ):
        raise RevisionProtocolError("[SOURCE_VALUES_SECONDARY_METRICS_INVALID]")
    values = []
    for row in metrics:
        low, high = _interval(row)
        values.append(
            f"{str(row['metric']).replace('_', '-')} {_signed(_number(row, 'estimate'))} "
            f"{_format_interval(low, high)}; BH q="
            f"{_fixed(_probability(row, 'bh_q_four_metric_family'), 3)}"
        )
    target = lock.get("target_level_primary_sensitivity")
    if not isinstance(target, Mapping) or target.get("status") != "RELEASED":
        raise RevisionProtocolError("[SOURCE_VALUES_TARGET_SENSITIVITY_INVALID]")
    target_low, target_high = _interval(target)
    values.append(
        f"target-level {_signed(_number(target, 'estimate'))} "
        f"{_format_interval(target_low, target_high)}"
    )
    return "; ".join(values)


def _external_registry(lock: Mapping[str, Any], registry_id: str) -> Mapping[str, Any]:
    external = lock.get("external_registry")
    if not isinstance(external, Mapping):
        raise RevisionProtocolError(f"[SOURCE_VALUES_EXTERNAL_REGISTRY_INVALID] {registry_id}")
    payload = dict(external)
    declared_hash = payload.pop("registry_hash", None)
    if (
        external.get("registry_id") != registry_id
        or external.get("status") != "RELEASED"
        or declared_hash != canonical_sha256(payload)
    ):
        raise RevisionProtocolError(f"[SOURCE_VALUES_EXTERNAL_REGISTRY_INVALID] {registry_id}")
    return external


def _render_external_comparators(lock: Mapping[str, Any]) -> str:
    registry = _external_registry(lock, "EXTERNAL-COMPARATOR-VALIDATION")
    decisions = registry.get("member_decisions")
    if (
        not isinstance(decisions, Mapping)
        or set(decisions) != {"GEARS", "scGPT", "Geneformer"}
        or any(value not in {"PASS", "EXCLUDED"} for value in decisions.values())
    ):
        raise RevisionProtocolError("[SOURCE_VALUES_EXTERNAL_FAMILY_DECISION_INVALID]")
    performance = registry.get("performance_registry")
    if not isinstance(performance, Mapping):
        raise RevisionProtocolError("[SOURCE_VALUES_EXTERNAL_PERFORMANCE_REGISTRY_INVALID]")
    performance_payload = dict(performance)
    performance_hash = performance_payload.pop("registry_hash", None)
    if (
        performance.get("registry_id") != "EXTERNAL-COMPARATOR-SCALE-PERFORMANCE"
        or performance.get("status") != "RELEASED"
        or performance_hash != canonical_sha256(performance_payload)
        or registry.get("performance_registry_hash") != performance_hash
        or performance.get("validity_registry_hash") != registry.get("validity_registry_hash")
    ):
        raise RevisionProtocolError("[SOURCE_VALUES_EXTERNAL_PERFORMANCE_REGISTRY_INVALID]")
    rendered = []
    adapter_results = performance.get("adapter_results")
    if not isinstance(adapter_results, Mapping) or set(adapter_results) != set(decisions):
        raise RevisionProtocolError("[SOURCE_VALUES_EXTERNAL_PERFORMANCE_REGISTRY_INVALID]")
    for member in ("GEARS", "scGPT", "Geneformer"):
        decision = str(decisions[member])
        if decision == "EXCLUDED":
            if adapter_results[member].get("status") != "NOT_APPLICABLE_EXCLUDED":
                raise RevisionProtocolError("[SOURCE_VALUES_EXTERNAL_PERFORMANCE_REGISTRY_INVALID]")
            rendered.append(f"{member}: excluded")
            continue
        rows = adapter_results[member].get("contrasts")
        if (
            not isinstance(rows, list)
            or [row.get("hvg") for row in rows] != [200, 500, 1000]
            or adapter_results[member].get("family_denominator") != 3
            or adapter_results[member].get("adjustment") != "benjamini_hochberg_within_adapter"
        ):
            raise RevisionProtocolError("[SOURCE_VALUES_EXTERNAL_PERFORMANCE_REGISTRY_INVALID]")
        scale_values = []
        for row in rows:
            _interval(row)
            scale_values.append(
                f"{_integer(row, 'hvg', minimum=200, maximum=1000)}-HVG "
                f"{_signed(_number(row, 'estimate'))} "
                f"(BH q={_fixed(_probability(row, 'bh_q_three_scale_family'), 3)})"
            )
        rendered.append(f"{member}: pass, " + ", ".join(scale_values))
    return "; ".join(rendered)


def _distribution(registry: Mapping[str, Any], field: str) -> tuple[int, float, float]:
    phases = registry.get("successful_fit_phase_seconds")
    summary = phases.get(field) if isinstance(phases, Mapping) else None
    if not isinstance(summary, Mapping):
        raise RevisionProtocolError(f"[SOURCE_VALUES_COMPUTE_DISTRIBUTION_INVALID] {field}")
    count = _integer(summary, "n", minimum=1)
    median = _number(summary, "median", minimum=0.0)
    q1 = _number(summary, "q1", minimum=0.0)
    q3 = _number(summary, "q3", minimum=0.0)
    iqr = _number(summary, "iqr", minimum=0.0)
    if q1 > median or median > q3 or not math.isclose(iqr, q3 - q1, rel_tol=1e-12, abs_tol=1e-12):
        raise RevisionProtocolError(f"[SOURCE_VALUES_COMPUTE_DISTRIBUTION_INVALID] {field}")
    return count, median, iqr


def _render_compute(lock: Mapping[str, Any]) -> str:
    registry = _external_registry(lock, "MEASURED-COMPUTE")
    train_n, train_median, train_iqr = _distribution(registry, "training_wall_seconds")
    inference_n, inference_median, inference_iqr = _distribution(registry, "inference_wall_seconds")
    peak = registry.get("peak_device_memory_bytes")
    if not isinstance(peak, Mapping):
        raise RevisionProtocolError("[SOURCE_VALUES_COMPUTE_PEAK_MEMORY_INVALID]")
    peak_gib = _integer(peak, "maximum", minimum=0) / (1024**3)
    failures = _integer(registry, "failed_attempts", minimum=0)
    attempts = _integer(registry, "total_attempts", minimum=1)
    if failures > attempts or train_n != inference_n or train_n != attempts - failures:
        raise RevisionProtocolError("[SOURCE_VALUES_COMPUTE_COUNT_MISMATCH]")
    return (
        f"train median/IQR {_fixed(train_median, 2)}/{_fixed(train_iqr, 2)} s; "
        f"inference {_fixed(inference_median, 2)}/{_fixed(inference_iqr, 2)} s; "
        f"peak {_fixed(peak_gib, 2)} GiB; device-hours "
        f"{_fixed(_number(registry, 'device_hours', minimum=0.0), 2)}; "
        f"failures {failures}/{attempts}"
    )


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--mode", choices=("author-review", "released"), required=True)
    parser.add_argument("--verify", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_argument_parser().parse_args(arguments)
    if parsed.verify:
        manifest = verify_rendered_source_values(
            parsed.registry, parsed.output, parsed.manifest_output
        )
    else:
        manifest = render_source_values(
            parsed.registry,
            parsed.template,
            parsed.output,
            mode=parsed.mode,
            manifest_path=parsed.manifest_output,
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

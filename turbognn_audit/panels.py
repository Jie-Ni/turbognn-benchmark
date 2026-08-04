"""Outcome-agnostic, deterministic held-out condition panels."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

from .hashing import sha256_json, sha256_json_nfc


@dataclass(frozen=True)
class ConditionEligibility:
    """One outcome-blind terminal state, including canonical alias-group provenance."""

    condition: str
    pre_qc_cell_count: int
    post_qc_cell_count: int
    terminal_state: str
    reason: str
    mapped_targets: tuple[str, ...]
    canonical_perturbation_id: str | None = None
    raw_members: tuple[str, ...] = ()
    raw_member_pre_qc_counts: tuple[tuple[str, int], ...] = ()
    raw_member_post_qc_counts: tuple[tuple[str, int], ...] = ()
    raw_member_states: tuple[tuple[str, str], ...] = ()
    raw_member_reasons: tuple[tuple[str, str], ...] = ()
    raw_normalization_evidence: tuple[tuple[str, str], ...] = ()
    raw_record_hashes: tuple[tuple[str, str], ...] = ()
    alias_member_hash: str | None = None
    group_hash: str | None = None
    rank_hash: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible ledger row."""
        value = asdict(self)
        value["mapped_targets"] = list(self.mapped_targets)
        value["raw_members"] = list(self.raw_members)
        for field_name in (
            "raw_member_pre_qc_counts",
            "raw_member_post_qc_counts",
            "raw_member_states",
            "raw_member_reasons",
            "raw_normalization_evidence",
            "raw_record_hashes",
        ):
            value[field_name] = dict(value[field_name])
        return value


@dataclass(frozen=True)
class ConditionPanel:
    """Ordered perturbation conditions and their immutable provenance hash."""

    dataset: str
    control_label: str
    minimum_cells: int
    conditions: tuple[str, ...]
    panel_hash: str
    selection_limit: int | None = None
    eligibility_ledger: tuple[ConditionEligibility, ...] = ()
    eligibility_ledger_hash: str | None = None
    cell_qc_policy_hash: str | None = None
    rank_hashes: tuple[tuple[str, str], ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "dataset": self.dataset,
            "control_label": self.control_label,
            "minimum_cells": self.minimum_cells,
            "conditions": list(self.conditions),
            "selection_limit": self.selection_limit,
            "eligibility_ledger": [row.as_dict() for row in self.eligibility_ledger],
            "eligibility_ledger_hash": self.eligibility_ledger_hash,
            "cell_qc_policy_hash": self.cell_qc_policy_hash,
            "rank_hashes": dict(self.rank_hashes),
            "panel_hash": self.panel_hash,
        }


def canonical_condition_label(value: object) -> str:
    """Return the exact canonical form required for condition identities."""
    return unicodedata.normalize("NFC", str(value))


def _validated_raw_counts(values: Mapping[object, int], name: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for raw_label, raw_count in values.items():
        if not isinstance(raw_label, str):
            raise ValueError(f"{name} labels must be strings")
        label = canonical_condition_label(raw_label)
        if label != raw_label:
            raise ValueError(f"{name} contains a non-NFC label: {raw_label!r}")
        if not label or label != label.strip():
            raise ValueError(f"{name} contains a blank or padded identifier: {label!r}")
        count = int(raw_count)
        if isinstance(raw_count, bool) or count != raw_count or count < 0:
            raise ValueError(f"{name} contains an invalid count for {label!r}")
        if label in result:
            raise ValueError(f"{name} labels collide after NFC normalization: {label!r}")
        result[label] = count
    return result


def canonical_rank_hash(dataset: str, canonical_perturbation_id: str) -> str:
    """Return the exact TDS-02 dataset-scoped rank hash."""
    dataset_nfc = canonical_condition_label(dataset)
    perturbation_id = canonical_condition_label(canonical_perturbation_id)
    if not dataset_nfc or dataset_nfc != str(dataset) or dataset_nfc != dataset_nfc.strip():
        raise ValueError("Dataset identifier must be non-empty, NFC, and unpadded")
    if len(perturbation_id) != 64 or any(
        character not in "0123456789abcdef" for character in perturbation_id
    ):
        raise ValueError("canonical_perturbation_id must be a lowercase SHA-256 digest")
    return sha256_json_nfc(
        {
            "policy": "sha256_dataset_canonical_condition_v1",
            "dataset": dataset_nfc,
            "canonical_perturbation_id": perturbation_id,
        }
    )


def build_condition_panel(
    dataset: str,
    label_counts: Mapping[object, int],
    control_label: str,
    minimum_cells: int = 2,
    allowed_conditions: Iterable[object] | None = None,
) -> ConditionPanel:
    """Build a stable panel using labels and cell counts, never model outcomes."""
    if minimum_cells < 1:
        raise ValueError("minimum_cells must be positive")
    canonical_control = canonical_condition_label(control_label)
    allowed = None
    if allowed_conditions is not None:
        allowed = {canonical_condition_label(value) for value in allowed_conditions}

    canonical_counts: dict[str, int] = {}
    for raw_label, raw_count in label_counts.items():
        label = canonical_condition_label(raw_label)
        count = int(raw_count)
        if label in canonical_counts:
            raise ValueError(f"Labels collide after Unicode normalization: {label!r}")
        canonical_counts[label] = count

    if canonical_control not in canonical_counts:
        raise ValueError(f"Control label {canonical_control!r} is absent from label counts")

    conditions = tuple(
        sorted(
            label
            for label, count in canonical_counts.items()
            if label != canonical_control
            and count >= minimum_cells
            and (allowed is None or label in allowed)
        )
    )
    if not conditions:
        raise ValueError(f"No eligible conditions remain for dataset {dataset!r}")

    payload = {
        "dataset": dataset,
        "control_label": canonical_control,
        "minimum_cells": minimum_cells,
        "conditions": list(conditions),
    }
    return ConditionPanel(
        dataset=dataset,
        control_label=canonical_control,
        minimum_cells=minimum_cells,
        conditions=conditions,
        panel_hash=sha256_json(payload),
        selection_limit=None,
    )


def build_condition_eligibility_panel(
    *,
    dataset: str,
    pre_qc_label_counts: Mapping[object, int],
    post_qc_label_counts: Mapping[object, int],
    control_label: str,
    minimum_cells: int,
    condition_targets: Mapping[str, Sequence[str]],
    available_genes: Iterable[str],
    cell_qc_policy: str,
    perturbation_type_policy: str,
) -> ConditionPanel:
    """Apply every frozen outcome-independent gate before deterministic panel ranking."""
    if minimum_cells != 20:
        raise ValueError("Canonical eligibility requires minimum_cells=20")
    if cell_qc_policy != "source_filtered_matrix_no_additional_cell_filter":
        raise ValueError("Unsupported condition-blind cell_qc_policy")
    if perturbation_type_policy != "explicit_mapped_single_or_multi_target":
        raise ValueError("Unsupported perturbation_type_policy")
    canonical_control = canonical_condition_label(control_label)

    def canonical_counts(values: Mapping[object, int], name: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for raw_label, raw_count in values.items():
            label = canonical_condition_label(raw_label)
            count = int(raw_count)
            if label in result:
                raise ValueError(f"{name} labels collide after normalization: {label!r}")
            if count < 0:
                raise ValueError(f"{name} contains a negative count for {label!r}")
            result[label] = count
        return result

    pre_counts = canonical_counts(pre_qc_label_counts, "pre_qc_label_counts")
    post_counts = canonical_counts(post_qc_label_counts, "post_qc_label_counts")
    if set(post_counts) - set(pre_counts):
        raise ValueError("Cell QC introduced condition labels absent before QC")
    if canonical_control not in pre_counts or post_counts.get(canonical_control, 0) < 1:
        raise ValueError("Configured control is absent before or after cell QC")
    genes = {str(gene) for gene in available_genes}
    canonical_targets = {
        canonical_condition_label(condition): tuple(str(gene) for gene in targets)
        for condition, targets in condition_targets.items()
    }
    ledger: list[ConditionEligibility] = []
    for condition in sorted(pre_counts, key=lambda value: value.encode("utf-8")):
        pre_count = pre_counts[condition]
        post_count = post_counts.get(condition, 0)
        targets = canonical_targets.get(condition, ())
        if condition == canonical_control:
            state, reason = "ineligible", "control_reference_not_candidate"
        elif not targets:
            state, reason = "ineligible", "explicit_target_mapping_missing"
        elif len(set(targets)) != len(targets) or any(not target for target in targets):
            state, reason = "ineligible", "unsupported_or_invalid_perturbation_mapping"
        elif any(target not in genes for target in targets):
            state, reason = "ineligible", "mapped_target_absent_from_raw_gene_universe"
        elif post_count < minimum_cells:
            state, reason = "ineligible", "post_qc_cell_count_below_frozen_minimum_20"
        else:
            state, reason = "eligible", "passed_all_frozen_outcome_independent_gates"
        ledger.append(
            ConditionEligibility(
                condition=condition,
                pre_qc_cell_count=pre_count,
                post_qc_cell_count=post_count,
                terminal_state=state,
                reason=reason,
                mapped_targets=targets,
            )
        )
    conditions = tuple(row.condition for row in ledger if row.terminal_state == "eligible")
    if not conditions:
        raise ValueError(f"No eligible conditions remain for dataset {dataset!r}")
    ledger_payload = [row.as_dict() for row in ledger]
    ledger_hash = sha256_json(ledger_payload)
    qc_hash = sha256_json(
        {
            "cell_qc_policy": cell_qc_policy,
            "minimum_perturbed_cells": minimum_cells,
            "pre_qc_counts": pre_counts,
            "post_qc_counts": post_counts,
        }
    )
    payload = {
        "dataset": dataset,
        "control_label": canonical_control,
        "minimum_cells": minimum_cells,
        "conditions": list(conditions),
        "eligibility_ledger_hash": ledger_hash,
        "cell_qc_policy_hash": qc_hash,
        "perturbation_type_policy": perturbation_type_policy,
    }
    return ConditionPanel(
        dataset=dataset,
        control_label=canonical_control,
        minimum_cells=minimum_cells,
        conditions=conditions,
        panel_hash=sha256_json(payload),
        eligibility_ledger=tuple(ledger),
        eligibility_ledger_hash=ledger_hash,
        cell_qc_policy_hash=qc_hash,
    )


def build_canonical_condition_eligibility_panel(
    *,
    dataset: str,
    pre_qc_label_counts: Mapping[object, int],
    post_qc_label_counts: Mapping[object, int],
    control_label: str,
    minimum_cells: int,
    raw_to_canonical_id: Mapping[str, str],
    condition_targets: Mapping[str, Sequence[str]],
    canonical_objects: Mapping[str, Mapping[str, object]],
    raw_normalization_evidence: Mapping[str, str],
    raw_record_hashes: Mapping[str, str],
    allowed_perturbation_types: Sequence[str],
    available_genes: Iterable[str],
    cell_qc_policy: str,
    perturbation_type_policy: str,
) -> ConditionPanel:
    """Pool raw aliases into the frozen TDS-02 canonical LOPO unit before ranking."""
    if minimum_cells != 20:
        raise ValueError("Canonical eligibility requires minimum_cells=20")
    if cell_qc_policy != "source_filtered_matrix_no_additional_cell_filter":
        raise ValueError("Unsupported condition-blind cell_qc_policy")
    if perturbation_type_policy != "explicit_mapped_single_or_multi_target":
        raise ValueError("Unsupported perturbation_type_policy")
    canonical_dataset = canonical_condition_label(dataset)
    canonical_control = canonical_condition_label(control_label)
    if (
        not canonical_dataset
        or canonical_dataset != dataset
        or canonical_dataset != canonical_dataset.strip()
    ):
        raise ValueError("Dataset identifier must be non-empty, NFC, and unpadded")
    if (
        not canonical_control
        or canonical_control != control_label
        or canonical_control != canonical_control.strip()
    ):
        raise ValueError("Control label must be non-empty, NFC, and unpadded")
    pre_counts = _validated_raw_counts(pre_qc_label_counts, "pre_qc_label_counts")
    post_counts = _validated_raw_counts(post_qc_label_counts, "post_qc_label_counts")
    if set(post_counts) - set(pre_counts):
        raise ValueError("Cell QC introduced condition labels absent before QC")
    if canonical_control not in pre_counts or post_counts.get(canonical_control, 0) < 1:
        raise ValueError("Configured control is absent before or after cell QC")
    if canonical_control in raw_to_canonical_id:
        raise ValueError("The configured control must not map to a perturbation object")

    canonical_raw_map: dict[str, str] = {}
    for raw_label, canonical_id in raw_to_canonical_id.items():
        if not isinstance(raw_label, str) or canonical_condition_label(raw_label) != raw_label:
            raise ValueError("Raw alias labels must already be Unicode NFC strings")
        if not raw_label or raw_label != raw_label.strip():
            raise ValueError("Raw alias labels must be non-empty and unpadded")
        if raw_label in canonical_raw_map:
            raise ValueError("Raw alias labels collide after NFC normalization")
        canonical_id_text = canonical_condition_label(canonical_id)
        if canonical_id_text != canonical_id or (
            len(canonical_id_text) != 64
            or any(character not in "0123456789abcdef" for character in canonical_id_text)
        ):
            raise ValueError("Canonical perturbation IDs must be lowercase SHA-256 digests")
        canonical_raw_map[raw_label] = canonical_id_text
    if set(raw_normalization_evidence) != set(canonical_raw_map):
        raise ValueError("Every raw alias must have exactly one normalization-evidence record")
    if set(raw_record_hashes) != set(canonical_raw_map):
        raise ValueError("Every raw alias must have exactly one raw-record hash")
    for raw_label in canonical_raw_map:
        evidence = raw_normalization_evidence[raw_label]
        record_hash = raw_record_hashes[raw_label]
        if (
            not isinstance(evidence, str)
            or not evidence
            or canonical_condition_label(evidence) != evidence
            or evidence != evidence.strip()
        ):
            raise ValueError(f"Raw alias {raw_label!r} has invalid normalization evidence")
        if (
            not isinstance(record_hash, str)
            or len(record_hash) != 64
            or any(character not in "0123456789abcdef" for character in record_hash)
        ):
            raise ValueError(f"Raw alias {raw_label!r} has invalid raw-record hash")

    groups: dict[str, list[str]] = {}
    for raw_label, canonical_id in canonical_raw_map.items():
        groups.setdefault(canonical_id, []).append(raw_label)
    if set(groups) != set(condition_targets) or set(groups) != set(canonical_objects):
        raise ValueError(
            "Canonical objects, target mappings, and raw alias groups must have identical IDs"
        )
    genes = {str(gene) for gene in available_genes}
    allowed_types = tuple(str(value) for value in allowed_perturbation_types)
    if not allowed_types or allowed_types != tuple(sorted(set(allowed_types))):
        raise ValueError("Allowed perturbation types must be a sorted, non-empty set")
    ledger: list[ConditionEligibility] = []

    control_pre = pre_counts[canonical_control]
    control_post = post_counts.get(canonical_control, 0)
    ledger.append(
        ConditionEligibility(
            condition=canonical_control,
            pre_qc_cell_count=control_pre,
            post_qc_cell_count=control_post,
            terminal_state="ineligible",
            reason="control_reference_not_candidate",
            mapped_targets=(),
            raw_members=(canonical_control,),
            raw_member_pre_qc_counts=((canonical_control, control_pre),),
            raw_member_post_qc_counts=((canonical_control, control_post),),
            raw_member_states=((canonical_control, "ineligible"),),
            raw_member_reasons=((canonical_control, "control_reference_not_candidate"),),
            alias_member_hash=sha256_json_nfc([canonical_control]),
        )
    )
    observed_unmapped = sorted(
        set(pre_counts) - set(canonical_raw_map) - {canonical_control},
        key=lambda value: value.encode("utf-8"),
    )
    for raw_label in observed_unmapped:
        raw_pre = pre_counts[raw_label]
        raw_post = post_counts.get(raw_label, 0)
        ledger.append(
            ConditionEligibility(
                condition=raw_label,
                pre_qc_cell_count=raw_pre,
                post_qc_cell_count=raw_post,
                terminal_state="ineligible",
                reason="explicit_target_mapping_missing",
                mapped_targets=(),
                raw_members=(raw_label,),
                raw_member_pre_qc_counts=((raw_label, raw_pre),),
                raw_member_post_qc_counts=((raw_label, raw_post),),
                raw_member_states=((raw_label, "ineligible"),),
                raw_member_reasons=((raw_label, "explicit_target_mapping_missing"),),
                alias_member_hash=sha256_json_nfc([raw_label]),
            )
        )

    seen_rank_hashes: dict[str, str] = {}
    for canonical_id in sorted(groups, key=lambda value: bytes.fromhex(value)):
        members = tuple(sorted(groups[canonical_id], key=lambda value: value.encode("utf-8")))
        targets = tuple(str(gene) for gene in condition_targets[canonical_id])
        canonical_object = canonical_objects[canonical_id]
        if sha256_json_nfc(canonical_object) != canonical_id:
            raise ValueError("Canonical perturbation ID does not hash to its frozen object")
        pre_member_counts = tuple((member, pre_counts.get(member, 0)) for member in members)
        post_member_counts = tuple((member, post_counts.get(member, 0)) for member in members)
        pooled_pre = sum(count for _, count in pre_member_counts)
        pooled_post = sum(count for _, count in post_member_counts)
        perturbation_type = canonical_object.get("perturbation_type")
        if perturbation_type not in allowed_types:
            state, reason = "ineligible", "unsupported_perturbation_type"
        elif (
            not targets
            or len(set(targets)) != len(targets)
            or any(not target for target in targets)
        ):
            state, reason = "ineligible", "unsupported_or_invalid_perturbation_mapping"
        elif any(target not in genes for target in targets):
            state, reason = "ineligible", "mapped_target_absent_from_raw_gene_universe"
        elif pooled_post < minimum_cells:
            state, reason = "ineligible", "pooled_post_qc_cell_count_below_frozen_minimum_20"
        else:
            state, reason = "eligible", "passed_all_frozen_outcome_independent_gates"
        member_states = tuple(
            (
                member,
                (
                    "not_observed_in_frozen_artifact"
                    if member not in pre_counts
                    else (
                        "eligible_as_member_of_canonical_group"
                        if state == "eligible"
                        else "ineligible_with_canonical_group"
                    )
                ),
            )
            for member in members
        )
        member_reasons = tuple(
            (
                member,
                (
                    "mapped_alias_absent_from_artifact"
                    if member not in pre_counts
                    else (
                        "pooled_with_complete_canonical_alias_group"
                        if state == "eligible"
                        else reason
                    )
                ),
            )
            for member in members
        )
        normalization_evidence = tuple(
            (member, raw_normalization_evidence[member]) for member in members
        )
        record_hashes = tuple((member, raw_record_hashes[member]) for member in members)
        alias_member_hash = sha256_json_nfc(
            {
                "canonical_perturbation_id": canonical_id,
                "raw_members": list(members),
            }
        )
        rank_hash = canonical_rank_hash(canonical_dataset, canonical_id)
        if rank_hash in seen_rank_hashes and seen_rank_hashes[rank_hash] != canonical_id:
            raise ValueError("Fatal canonical rank-hash collision")
        seen_rank_hashes[rank_hash] = canonical_id
        group_payload = {
            "canonical_perturbation_id": canonical_id,
            "canonical_object": canonical_object,
            "mapped_targets": list(targets),
            "raw_members": list(members),
            "raw_member_pre_qc_counts": dict(pre_member_counts),
            "raw_member_post_qc_counts": dict(post_member_counts),
            "raw_member_states": dict(member_states),
            "raw_member_reasons": dict(member_reasons),
            "raw_normalization_evidence": dict(normalization_evidence),
            "raw_record_hashes": dict(record_hashes),
            "pooled_pre_qc_cell_count": pooled_pre,
            "pooled_post_qc_cell_count": pooled_post,
            "terminal_state": state,
            "reason": reason,
            "alias_member_hash": alias_member_hash,
            "rank_hash": rank_hash,
        }
        ledger.append(
            ConditionEligibility(
                condition=canonical_id,
                canonical_perturbation_id=canonical_id,
                pre_qc_cell_count=pooled_pre,
                post_qc_cell_count=pooled_post,
                terminal_state=state,
                reason=reason,
                mapped_targets=targets,
                raw_members=members,
                raw_member_pre_qc_counts=pre_member_counts,
                raw_member_post_qc_counts=post_member_counts,
                raw_member_states=member_states,
                raw_member_reasons=member_reasons,
                raw_normalization_evidence=normalization_evidence,
                raw_record_hashes=record_hashes,
                alias_member_hash=alias_member_hash,
                group_hash=sha256_json_nfc(group_payload),
                rank_hash=rank_hash,
            )
        )
    conditions = tuple(
        row.canonical_perturbation_id
        for row in ledger
        if row.terminal_state == "eligible" and row.canonical_perturbation_id is not None
    )
    if not conditions:
        raise ValueError(f"No eligible canonical perturbation groups remain for {dataset!r}")
    ledger_payload = [row.as_dict() for row in ledger]
    ledger_hash = sha256_json_nfc(ledger_payload)
    grouped_counts = {
        row.canonical_perturbation_id: {
            "pre_qc": row.pre_qc_cell_count,
            "post_qc": row.post_qc_cell_count,
        }
        for row in ledger
        if row.canonical_perturbation_id is not None
    }
    qc_hash = sha256_json_nfc(
        {
            "cell_qc_policy": cell_qc_policy,
            "alias_pooling_policy": "sum_all_raw_alias_cells_before_minimum_support_gate",
            "minimum_perturbed_cells": minimum_cells,
            "raw_pre_qc_counts": pre_counts,
            "raw_post_qc_counts": post_counts,
            "canonical_group_counts": grouped_counts,
        }
    )
    rank_hashes = tuple(
        (str(row.canonical_perturbation_id), str(row.rank_hash))
        for row in ledger
        if row.canonical_perturbation_id is not None
    )
    payload = {
        "dataset": canonical_dataset,
        "control_label": canonical_control,
        "minimum_cells": minimum_cells,
        "conditions": list(conditions),
        "eligibility_ledger_hash": ledger_hash,
        "cell_qc_policy_hash": qc_hash,
        "perturbation_type_policy": perturbation_type_policy,
        "rank_hashes": dict(rank_hashes),
    }
    return ConditionPanel(
        dataset=canonical_dataset,
        control_label=canonical_control,
        minimum_cells=minimum_cells,
        conditions=conditions,
        panel_hash=sha256_json_nfc(payload),
        eligibility_ledger=tuple(ledger),
        eligibility_ledger_hash=ledger_hash,
        cell_qc_policy_hash=qc_hash,
        rank_hashes=rank_hashes,
    )


def limit_condition_panel(panel: ConditionPanel, maximum_conditions: int) -> ConditionPanel:
    """Select by deterministic dataset-scoped hash order and hash the exact panel."""
    if maximum_conditions < 1:
        raise ValueError("maximum_conditions must be positive")
    declared_rank_hashes = dict(panel.rank_hashes)
    canonical_ids = all(
        len(condition) == 64 and all(character in "0123456789abcdef" for character in condition)
        for condition in panel.conditions
    )
    computed_rank_hashes = (
        {condition: canonical_rank_hash(panel.dataset, condition) for condition in panel.conditions}
        if canonical_ids
        else {
            condition: sha256_json({"dataset": panel.dataset, "condition": condition})
            for condition in panel.conditions
        }
    )
    if declared_rank_hashes and declared_rank_hashes != computed_rank_hashes:
        raise ValueError("Condition panel contains a stale or incomplete canonical rank-hash map")
    rank_hashes = declared_rank_hashes or computed_rank_hashes
    if len(set(rank_hashes.values())) != len(rank_hashes):
        raise ValueError("Fatal canonical rank-hash collision")
    hash_ordered = sorted(
        panel.conditions, key=lambda condition: bytes.fromhex(rank_hashes[condition])
    )
    conditions = tuple(hash_ordered[:maximum_conditions])
    if not conditions:
        raise ValueError("Condition panel is empty")
    payload = {
        "dataset": panel.dataset,
        "control_label": panel.control_label,
        "minimum_cells": panel.minimum_cells,
        "conditions": list(conditions),
        "selection_limit": maximum_conditions,
        "eligibility_ledger_hash": panel.eligibility_ledger_hash,
        "cell_qc_policy_hash": panel.cell_qc_policy_hash,
        "rank_hashes": {condition: rank_hashes[condition] for condition in conditions},
    }
    return ConditionPanel(
        dataset=panel.dataset,
        control_label=panel.control_label,
        minimum_cells=panel.minimum_cells,
        conditions=conditions,
        panel_hash=sha256_json_nfc(payload),
        selection_limit=maximum_conditions,
        eligibility_ledger=panel.eligibility_ledger,
        eligibility_ledger_hash=panel.eligibility_ledger_hash,
        cell_qc_policy_hash=panel.cell_qc_policy_hash,
        rank_hashes=tuple((condition, rank_hashes[condition]) for condition in conditions),
    )

"""Strict run-manifest schema and conservation checks."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import jsonschema

from .hashing import sha256_json
from .io import write_json_atomic

ALLOWED_STATUSES = frozenset({"planned", "queued", "running", "succeeded", "failed", "skipped"})
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "skipped"})
RUN_IDENTITY_FIELDS = (
    "dataset",
    "input_hash",
    "environment_lock_hash",
    "gene_panel_hash",
    "condition_panel_hash",
    "preprocessing_hash",
    "dataset_passport_registry_hash",
    "dataset_passport_hash",
    "matrix_contract_hash",
    "control_selection_hash",
    "target_mapping_hash",
    "condition_eligibility_ledger_hash",
    "cell_qc_policy_hash",
    "response_definition_hash",
    "graph_type",
    "graph_instance",
    "graph_hash",
    "graph_source_config_hash",
    "graph_ensemble_config_hash",
    "graph_contract_hash",
    "model_name",
    "model_config_hash",
    "input_contract_policy_hash",
    "input_definition_hash",
    "configuration_universe_hash",
    "schedule_manifest_hash",
    "schedule_content_hash",
    "schedule_configuration_hash",
    "split_hash",
    "code_commit",
    "seed",
    "fold_seed",
    "condition",
)


class ManifestValidationError(ValueError):
    """Raised when a manifest violates identity or state-conservation rules."""


@dataclass(frozen=True)
class RunRecord:
    """One model/graph/seed/held-out-condition execution unit."""

    dataset: str
    input_hash: str
    environment_lock_hash: str
    gene_panel_hash: str
    condition_panel_hash: str
    preprocessing_hash: str
    dataset_passport_registry_hash: str
    dataset_passport_hash: str
    matrix_contract_hash: str
    control_selection_hash: str
    target_mapping_hash: str
    condition_eligibility_ledger_hash: str
    cell_qc_policy_hash: str
    response_definition_hash: str
    graph_type: str
    graph_instance: str
    graph_hash: str
    graph_source_config_hash: str
    graph_ensemble_config_hash: str
    graph_contract_hash: str
    model_name: str
    model_config_hash: str
    input_contract_policy_hash: str
    input_definition_hash: str
    configuration_universe_hash: str
    schedule_manifest_hash: str
    schedule_content_hash: str
    schedule_configuration_hash: str
    split_hash: str
    code_commit: str
    seed: int
    fold_seed: int
    condition: str
    status: str = "planned"
    result_path: str | None = None
    result_hash: str | None = None
    failure_reason: str | None = None
    skip_reason: str | None = None

    @property
    def key(
        self,
    ) -> tuple[tuple[str, object], ...]:
        """Return the complete, immutable run identity."""
        return tuple(self.identity.items())

    @property
    def identity(self) -> dict[str, object]:
        """Return the canonical identity mapping shared with fold-result records."""
        return {name: getattr(self, name) for name in RUN_IDENTITY_FIELDS}

    @property
    def run_key(self) -> str:
        """Return the stable SHA-256 identity used by lossless fold outputs."""
        return sha256_json(self.identity)

    def validate(self) -> None:
        """Validate one record independently of the surrounding manifest."""
        text_fields = (
            self.dataset,
            self.input_hash,
            self.environment_lock_hash,
            self.gene_panel_hash,
            self.condition_panel_hash,
            self.preprocessing_hash,
            self.dataset_passport_registry_hash,
            self.dataset_passport_hash,
            self.matrix_contract_hash,
            self.control_selection_hash,
            self.target_mapping_hash,
            self.condition_eligibility_ledger_hash,
            self.cell_qc_policy_hash,
            self.response_definition_hash,
            self.graph_type,
            self.graph_instance,
            self.graph_hash,
            self.graph_source_config_hash,
            self.graph_ensemble_config_hash,
            self.graph_contract_hash,
            self.model_name,
            self.model_config_hash,
            self.input_contract_policy_hash,
            self.input_definition_hash,
            self.configuration_universe_hash,
            self.schedule_manifest_hash,
            self.schedule_content_hash,
            self.schedule_configuration_hash,
            self.split_hash,
            self.code_commit,
            self.condition,
        )
        if any(not value.strip() for value in text_fields):
            raise ManifestValidationError(f"Blank identity field in run record: {self.key}")
        for name, value in (
            ("input_hash", self.input_hash),
            ("environment_lock_hash", self.environment_lock_hash),
            ("gene_panel_hash", self.gene_panel_hash),
            ("condition_panel_hash", self.condition_panel_hash),
            ("preprocessing_hash", self.preprocessing_hash),
            ("dataset_passport_registry_hash", self.dataset_passport_registry_hash),
            ("dataset_passport_hash", self.dataset_passport_hash),
            ("matrix_contract_hash", self.matrix_contract_hash),
            ("control_selection_hash", self.control_selection_hash),
            ("target_mapping_hash", self.target_mapping_hash),
            ("condition_eligibility_ledger_hash", self.condition_eligibility_ledger_hash),
            ("cell_qc_policy_hash", self.cell_qc_policy_hash),
            ("response_definition_hash", self.response_definition_hash),
            ("graph_hash", self.graph_hash),
            ("graph_source_config_hash", self.graph_source_config_hash),
            ("graph_ensemble_config_hash", self.graph_ensemble_config_hash),
            ("graph_contract_hash", self.graph_contract_hash),
            ("model_config_hash", self.model_config_hash),
            ("input_contract_policy_hash", self.input_contract_policy_hash),
            ("input_definition_hash", self.input_definition_hash),
            ("configuration_universe_hash", self.configuration_universe_hash),
            ("schedule_manifest_hash", self.schedule_manifest_hash),
            ("schedule_content_hash", self.schedule_content_hash),
            ("schedule_configuration_hash", self.schedule_configuration_hash),
            ("split_hash", self.split_hash),
        ):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ManifestValidationError(
                    f"{name} must be a lowercase 64-character SHA-256 digest: {value!r}"
                )
        if re.fullmatch(r"[0-9a-f]{40}", self.code_commit) is None:
            raise ManifestValidationError(
                f"code_commit must be a lowercase 40-character Git SHA: {self.code_commit!r}"
            )
        if self.status not in ALLOWED_STATUSES:
            raise ManifestValidationError(f"Unsupported run status: {self.status!r}")
        if self.status == "succeeded":
            if not self.result_path:
                raise ManifestValidationError(f"Succeeded run lacks result_path: {self.key}")
            if self.result_hash is None or re.fullmatch(r"[0-9a-f]{64}", self.result_hash) is None:
                raise ManifestValidationError(
                    f"Succeeded run lacks a valid result_hash: {self.key}"
                )
        if self.status == "failed" and not self.failure_reason:
            raise ManifestValidationError(f"Failed run lacks failure_reason: {self.key}")
        if self.status != "failed" and self.failure_reason:
            raise ManifestValidationError(f"Non-failed run has failure_reason: {self.key}")
        if self.status == "skipped" and not self.skip_reason:
            raise ManifestValidationError(f"Skipped run lacks skip_reason: {self.key}")
        if self.status != "skipped" and self.skip_reason:
            raise ManifestValidationError(f"Non-skipped run has skip_reason: {self.key}")
        if self.status != "succeeded" and (self.result_path or self.result_hash):
            raise ManifestValidationError(f"Non-succeeded run references a result: {self.key}")


@dataclass(frozen=True)
class RunManifest:
    """Immutable collection of uniquely identified execution units."""

    records: tuple[RunRecord, ...]
    schema_version: str = "1.0.0"

    @classmethod
    def build(cls, records: Iterable[RunRecord]) -> RunManifest:
        """Build and strictly validate a manifest."""
        manifest = cls(records=tuple(records))
        manifest.validate()
        return manifest

    def validate(self, require_terminal: bool = False) -> None:
        """Reject duplicate identities and invalid or non-conserved states."""
        if self.schema_version != "1.0.0":
            raise ManifestValidationError(
                f"Unsupported manifest schema_version: {self.schema_version!r}"
            )
        if not self.records:
            raise ManifestValidationError("Manifest contains no planned execution units")
        seen: set[tuple[tuple[str, object], ...]] = set()
        for record in self.records:
            record.validate()
            if record.key in seen:
                raise ManifestValidationError(f"Duplicate run key: {record.key}")
            seen.add(record.key)
        if require_terminal:
            nonterminal = [
                record.key for record in self.records if record.status not in TERMINAL_STATUSES
            ]
            if nonterminal:
                raise ManifestValidationError(
                    f"Manifest is not terminal; {len(nonterminal)} execution units remain"
                )

    @property
    def manifest_hash(self) -> str:
        """Hash identity and state for exact release provenance."""
        return sha256_json([asdict(record) for record in self.records])

    def status_counts(self) -> dict[str, int]:
        """Return counts whose sum must equal the manifest's planned total."""
        counts = Counter(record.status for record in self.records)
        conserved = sum(counts.values())
        if conserved != len(self.records):
            raise ManifestValidationError(
                f"State conservation failed: {conserved} statuses for {len(self.records)} records"
            )
        return {status: counts.get(status, 0) for status in sorted(ALLOWED_STATUSES)}

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible manifest with conservation summary."""
        return {
            "schema_version": self.schema_version,
            "planned_total": len(self.records),
            "status_counts": self.status_counts(),
            "manifest_hash": self.manifest_hash,
            "records": [asdict(record) for record in self.records],
        }

    def write(self, path: Path, *, overwrite: bool = False) -> None:
        """Write canonical JSON without clobbering an unrelated run manifest."""
        self.validate()
        if overwrite:
            previous = self.read(path)
            previous_identities = [record.identity for record in previous.records]
            current_identities = [record.identity for record in self.records]
            if previous_identities != current_identities:
                raise ManifestValidationError(
                    f"Refusing to overwrite manifest with different run identities: {path}"
                )
        payload = self.as_dict()
        schema_path = Path(__file__).resolve().parents[1] / "schemas" / "run_manifest.schema.json"
        import json

        with schema_path.open(encoding="utf-8") as handle:
            schema = json.load(handle)
        try:
            jsonschema.validate(payload, schema)
        except jsonschema.ValidationError as error:
            raise ManifestValidationError(
                f"Run manifest fails its JSON schema: {error.message}"
            ) from error
        write_json_atomic(path, payload, overwrite=overwrite)

    @classmethod
    def read(cls, path: Path) -> RunManifest:
        """Load and validate a manifest, including declared conservation fields."""
        import json

        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        schema_path = Path(__file__).resolve().parents[1] / "schemas" / "run_manifest.schema.json"
        with schema_path.open(encoding="utf-8") as handle:
            schema = json.load(handle)
        try:
            jsonschema.validate(raw, schema)
        except jsonschema.ValidationError as error:
            raise ManifestValidationError(
                f"Run manifest fails its JSON schema: {error.message}"
            ) from error
        if not isinstance(raw, Mapping) or not isinstance(raw.get("records"), list):
            raise ManifestValidationError("Manifest JSON lacks a records array")
        records = tuple(RunRecord(**record) for record in raw["records"])
        manifest = cls(records=records, schema_version=str(raw.get("schema_version", "")))
        manifest.validate()
        if raw.get("planned_total") != len(records):
            raise ManifestValidationError("Declared planned_total does not match records")
        if raw.get("status_counts") != manifest.status_counts():
            raise ManifestValidationError("Declared status_counts do not match records")
        if raw.get("manifest_hash") != manifest.manifest_hash:
            raise ManifestValidationError("Declared manifest_hash does not match records")
        return manifest

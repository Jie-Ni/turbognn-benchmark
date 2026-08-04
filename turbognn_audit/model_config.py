"""Required checksummed TDS-42 model and optimization contract."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema

from .hashing import canonical_json, sha256_file, sha256_json


@dataclass(frozen=True)
class ModelSpec:
    """Validated frozen model contract and both byte/canonical hashes."""

    values: Mapping[str, Any]
    file_hash: str
    canonical_hash: str

    @property
    def feature_encoder(self) -> Mapping[str, Any]:
        return self.values["feature_encoder"]

    @property
    def gat(self) -> Mapping[str, Any]:
        return self.values["gat"]

    @property
    def transformer(self) -> Mapping[str, Any]:
        return self.values["transformer"]

    @property
    def training(self) -> Mapping[str, Any]:
        return self.values["training"]

    @property
    def runtime(self) -> Mapping[str, Any]:
        return self.values["runtime"]


def _frozen_contract() -> Mapping[str, Any]:
    path = Path(__file__).resolve().parents[1] / "config" / "model_tds42.json"
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise RuntimeError("Bundled TDS-42 model contract is malformed")
    return value


def load_model_spec(path: Path) -> ModelSpec:
    """Load an exact TDS-42 contract; missing, extra, or changed choices fail closed."""
    with path.open(encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, Mapping):
        raise ValueError("Model config must be a JSON object")
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "model_tds42.schema.json"
    with schema_path.open(encoding="utf-8") as handle:
        schema = json.load(handle)
    try:
        jsonschema.validate(values, schema)
    except jsonschema.ValidationError as error:
        raise ValueError(f"Model config fails TDS-42 schema: {error.message}") from error
    frozen = _frozen_contract()
    if canonical_json(values) != canonical_json(frozen):
        raise ValueError("Model config differs from the frozen TDS-42 contract")
    return ModelSpec(
        values=dict(values),
        file_hash=sha256_file(path),
        canonical_hash=sha256_json(values),
    )

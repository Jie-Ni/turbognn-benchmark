#!/usr/bin/env python
"""Download and verify frozen H5AD passports without a size-only fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turbognn_audit.hashing import sha256_json


@dataclass(frozen=True)
class DatasetPassport:
    """Immutable source and H5AD identity required before canonical use."""

    dataset: str
    relative_path: str
    url: str
    sha256: str
    size_bytes: int
    n_obs: int
    n_vars: int
    source_accession: str
    source_version: str
    matrix_layer: str
    matrix_dtype: str
    required_obs_fields: tuple[str, ...]
    required_var_fields: tuple[str, ...]
    perturbation_column: str
    control_field: str
    control_value: str
    target_mapping_fields: tuple[str, ...]
    input_expression_state: str
    minimum_perturbed_cells: int
    cell_qc_policy: str
    condition_eligibility_ledger_sha256: str

    @property
    def passport_hash(self) -> str:
        """Hash the complete executable registry record."""
        return sha256_json(
            {
                **self.__dict__,
                "required_obs_fields": list(self.required_obs_fields),
                "required_var_fields": list(self.required_var_fields),
                "target_mapping_fields": list(self.target_mapping_fields),
            }
        )


def load_dataset_passports(path: Path) -> tuple[DatasetPassport, ...]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping) or raw.get("schema_version") != "1.0.0":
        raise ValueError("Dataset passport must be a schema_version 1.0.0 object")
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "data_sources.schema.json"
    with schema_path.open(encoding="utf-8") as handle:
        schema = json.load(handle)
    try:
        jsonschema.validate(raw, schema)
    except jsonschema.ValidationError as error:
        raise ValueError(f"Dataset passport fails its JSON schema: {error.message}") from error
    values = raw.get("datasets")
    if not isinstance(values, list) or not values:
        raise ValueError("Dataset passport requires a non-empty datasets array")
    passports: list[DatasetPassport] = []
    required = {
        "dataset",
        "relative_path",
        "url",
        "sha256",
        "size_bytes",
        "n_obs",
        "n_vars",
        "source_accession",
        "source_version",
        "matrix_layer",
        "matrix_dtype",
        "required_obs_fields",
        "required_var_fields",
        "perturbation_column",
        "control_field",
        "control_value",
        "target_mapping_fields",
        "input_expression_state",
        "minimum_perturbed_cells",
        "cell_qc_policy",
        "condition_eligibility_ledger_sha256",
    }
    for index, value in enumerate(values):
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(
                f"Dataset passport row {index} must contain exactly {sorted(required)}"
            )
        for field in ("required_obs_fields", "required_var_fields", "target_mapping_fields"):
            if not isinstance(value[field], list):
                raise ValueError(f"Dataset passport row {index} field {field!r} must be a list")
        passport = DatasetPassport(
            dataset=str(value["dataset"]),
            relative_path=str(value["relative_path"]),
            url=str(value["url"]),
            sha256=str(value["sha256"]).lower(),
            size_bytes=int(value["size_bytes"]),
            n_obs=int(value["n_obs"]),
            n_vars=int(value["n_vars"]),
            source_accession=str(value["source_accession"]),
            source_version=str(value["source_version"]),
            matrix_layer=str(value["matrix_layer"]),
            matrix_dtype=str(value["matrix_dtype"]),
            required_obs_fields=tuple(str(item) for item in value["required_obs_fields"]),
            required_var_fields=tuple(str(item) for item in value["required_var_fields"]),
            perturbation_column=str(value["perturbation_column"]),
            control_field=str(value["control_field"]),
            control_value=str(value["control_value"]),
            target_mapping_fields=tuple(str(item) for item in value["target_mapping_fields"]),
            input_expression_state=str(value["input_expression_state"]),
            minimum_perturbed_cells=int(value["minimum_perturbed_cells"]),
            cell_qc_policy=str(value["cell_qc_policy"]),
            condition_eligibility_ledger_sha256=str(
                value["condition_eligibility_ledger_sha256"]
            ).lower(),
        )
        fields = (
            passport.dataset,
            passport.relative_path,
            passport.url,
            passport.perturbation_column,
            passport.source_accession,
            passport.source_version,
            passport.matrix_layer,
            passport.matrix_dtype,
            passport.control_field,
            passport.control_value,
            passport.input_expression_state,
            passport.cell_qc_policy,
        )
        if any(
            not field.strip() or "REPLACE" in field.upper() or "TODO" in field.upper()
            for field in fields
        ):
            raise ValueError(f"Dataset passport row {index} contains a placeholder")
        if re.fullmatch(r"[0-9a-f]{64}", passport.sha256) is None or len(set(passport.sha256)) == 1:
            raise ValueError(f"Dataset passport row {index} has a placeholder/invalid SHA-256")
        if min(passport.size_bytes, passport.n_obs, passport.n_vars) < 1:
            raise ValueError(f"Dataset passport row {index} has a non-positive size or shape")
        relative = Path(passport.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Dataset passport row {index} has an unsafe relative_path")
        sequence_fields = (
            passport.required_obs_fields,
            passport.required_var_fields,
            passport.target_mapping_fields,
        )
        if any(
            not values
            or len(set(values)) != len(values)
            or any(
                not field.strip() or "REPLACE" in field.upper() or "TODO" in field.upper()
                for field in values
            )
            for values in sequence_fields
        ):
            raise ValueError(f"Dataset passport row {index} has invalid required-field lists")
        if passport.perturbation_column not in passport.required_obs_fields:
            raise ValueError("perturbation_column must occur in required_obs_fields")
        if passport.control_field not in passport.required_obs_fields:
            raise ValueError("control_field must occur in required_obs_fields")
        if not set(passport.target_mapping_fields) <= set(passport.required_obs_fields):
            raise ValueError("target_mapping_fields must be included in required_obs_fields")
        if passport.matrix_layer != "X" and not passport.matrix_layer.startswith("layers/"):
            raise ValueError("matrix_layer must be 'X' or 'layers/<name>'")
        if passport.input_expression_state not in {"raw_counts", "log1p_normalized"}:
            raise ValueError("input_expression_state is invalid")
        if passport.minimum_perturbed_cells != 20:
            raise ValueError("minimum_perturbed_cells must be exactly 20")
        if passport.cell_qc_policy != "source_filtered_matrix_no_additional_cell_filter":
            raise ValueError("Dataset passport has a noncanonical cell_qc_policy")
        ledger_hash = passport.condition_eligibility_ledger_sha256
        if re.fullmatch(r"[0-9a-f]{64}", ledger_hash) is None or len(set(ledger_hash)) == 1:
            raise ValueError("Dataset passport has an invalid eligibility-ledger SHA-256")
        passports.append(passport)
    if len({passport.dataset for passport in passports}) != len(passports):
        raise ValueError("Dataset passport contains duplicate dataset identifiers")
    return tuple(passports)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset_file(path: Path, passport: DatasetPassport) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset is missing: {path}")
    observed_size = path.stat().st_size
    if observed_size != passport.size_bytes:
        raise ValueError(
            f"Dataset {passport.dataset!r} size mismatch: {observed_size} != {passport.size_bytes}"
        )
    observed_hash = _sha256(path)
    if observed_hash != passport.sha256:
        raise ValueError(
            f"Dataset {passport.dataset!r} SHA-256 mismatch: {observed_hash} != {passport.sha256}"
        )
    try:
        import scanpy as sc
    except ModuleNotFoundError as error:
        raise RuntimeError("scanpy is required for fail-closed H5AD validation") from error
    adata: Any | None = None
    try:
        adata = sc.read_h5ad(path, backed="r")
        if adata.n_obs != passport.n_obs or adata.n_vars != passport.n_vars:
            raise ValueError(
                f"Dataset {passport.dataset!r} shape mismatch: "
                f"{adata.n_obs}x{adata.n_vars} != {passport.n_obs}x{passport.n_vars}"
            )
        if passport.perturbation_column not in adata.obs.columns:
            raise ValueError(
                f"Dataset {passport.dataset!r} lacks frozen perturbation column "
                f"{passport.perturbation_column!r}"
            )
        missing_obs = sorted(set(passport.required_obs_fields) - set(adata.obs.columns))
        missing_var = sorted(set(passport.required_var_fields) - set(adata.var.columns))
        if missing_obs or missing_var:
            raise ValueError(
                f"Dataset {passport.dataset!r} schema mismatch: "
                f"missing obs={missing_obs}, var={missing_var}"
            )
        if passport.matrix_layer == "X":
            matrix = adata.X
        else:
            layer_name = passport.matrix_layer.removeprefix("layers/")
            if layer_name not in adata.layers:
                raise ValueError(f"Dataset {passport.dataset!r} lacks frozen layer {layer_name!r}")
            matrix = adata.layers[layer_name]
        dtype = np.dtype(matrix.dtype)
        if not np.issubdtype(dtype, np.number):
            raise ValueError(f"Dataset {passport.dataset!r} matrix is not numeric: {dtype}")
        if dtype.name != passport.matrix_dtype:
            raise ValueError(
                f"Dataset {passport.dataset!r} dtype mismatch: "
                f"{dtype.name!r} != {passport.matrix_dtype!r}"
            )
        controls = adata.obs[passport.control_field].astype(str)
        if not bool((controls == passport.control_value).any()):
            raise ValueError(
                f"Dataset {passport.dataset!r} lacks frozen control value "
                f"{passport.control_value!r} in {passport.control_field!r}"
            )
    finally:
        if adata is not None and getattr(adata, "file", None) is not None:
            adata.file.close()


def _download_atomic(passport: DatasetPassport, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        raise FileExistsError(f"Stale partial download requires explicit reconciliation: {partial}")
    try:
        request = urllib.request.Request(
            passport.url,
            headers={"User-Agent": "TurboGNN-frozen-dataset-downloader/1.0"},
        )
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("xb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        validate_dataset_file(partial, passport)
        os.replace(partial, destination)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise


def synchronize_datasets(manifest: Path, data_root: Path) -> None:
    """Validate existing datasets or atomically download and validate exact replacements."""
    for passport in load_dataset_passports(manifest):
        destination = data_root / passport.relative_path
        if destination.exists():
            validate_dataset_file(destination, passport)
        else:
            _download_atomic(passport, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "TurboGNN" / "data",
    )
    args = parser.parse_args()
    synchronize_datasets(args.manifest, args.data_root)
    print("All frozen datasets passed byte, shape, and perturbation-column validation.")


if __name__ == "__main__":
    main()

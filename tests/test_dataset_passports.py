from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from run_benchmark import select_passport_matrix
from slurm.download_data import (
    DatasetPassport,
    load_dataset_passports,
    validate_dataset_file,
)


def _passport_payload() -> dict[str, object]:
    return {
        "dataset": "toy",
        "relative_path": "processed/toy.h5ad",
        "url": "https://example.org/releases/v1/toy.h5ad",
        "sha256": "0123456789abcdef" * 4,
        "size_bytes": 100,
        "n_obs": 2,
        "n_vars": 2,
        "source_accession": "TOY:1",
        "source_version": "v1",
        "matrix_layer": "X",
        "matrix_dtype": "float32",
        "required_obs_fields": ["condition", "target"],
        "required_var_fields": ["symbol"],
        "perturbation_column": "condition",
        "control_field": "condition",
        "control_value": "control",
        "target_mapping_fields": ["target"],
        "input_expression_state": "log1p_normalized",
        "minimum_perturbed_cells": 20,
        "cell_qc_policy": "source_filtered_matrix_no_additional_cell_filter",
        "condition_eligibility_ledger_sha256": "1234567890abcdef" * 4,
    }


def test_dataset_passport_rejects_missing_schema_field_and_placeholder(tmp_path) -> None:
    payload = _passport_payload()
    path = tmp_path / "passport.json"
    path.write_text(
        json.dumps({"schema_version": "1.0.0", "datasets": [payload]}),
        encoding="utf-8",
    )
    assert load_dataset_passports(path)[0].matrix_layer == "X"
    del payload["matrix_dtype"]
    path.write_text(
        json.dumps({"schema_version": "1.0.0", "datasets": [payload]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="matrix_dtype"):
        load_dataset_passports(path)
    payload = _passport_payload()
    payload["control_value"] = "REPLACE_CONTROL"
    path.write_text(
        json.dumps({"schema_version": "1.0.0", "datasets": [payload]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="placeholder"):
        load_dataset_passports(path)


def test_dataset_file_validation_checks_layer_dtype_fields_and_control(tmp_path) -> None:
    ad = pytest.importorskip("anndata")
    path = tmp_path / "toy.h5ad"
    adata = ad.AnnData(
        X=np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        obs=pd.DataFrame(
            {"condition": ["control", "KO_A"], "target": ["none", "A"]},
            index=["c1", "c2"],
        ),
        var=pd.DataFrame({"symbol": ["A", "B"]}, index=["A", "B"]),
    )
    adata.write_h5ad(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    passport = DatasetPassport(
        dataset="toy",
        relative_path="toy.h5ad",
        url="https://example.org/releases/v1/toy.h5ad",
        sha256=digest,
        size_bytes=path.stat().st_size,
        n_obs=2,
        n_vars=2,
        source_accession="TOY:1",
        source_version="v1",
        matrix_layer="X",
        matrix_dtype="float32",
        required_obs_fields=("condition", "target"),
        required_var_fields=("symbol",),
        perturbation_column="condition",
        control_field="condition",
        control_value="control",
        target_mapping_fields=("target",),
        input_expression_state="log1p_normalized",
        minimum_perturbed_cells=20,
        cell_qc_policy="source_filtered_matrix_no_additional_cell_filter",
        condition_eligibility_ledger_sha256="1234567890abcdef" * 4,
    )
    validate_dataset_file(path, passport)
    with pytest.raises(ValueError, match="dtype mismatch"):
        validate_dataset_file(
            path,
            DatasetPassport(**{**passport.__dict__, "matrix_dtype": "float64"}),
        )
    with pytest.raises(ValueError, match="lacks frozen control"):
        validate_dataset_file(
            path,
            DatasetPassport(**{**passport.__dict__, "control_value": "missing"}),
        )


def test_runner_binds_frozen_non_x_layer_before_preprocessing() -> None:
    ad = pytest.importorskip("anndata")
    adata = ad.AnnData(X=np.zeros((2, 2), dtype=np.float32))
    adata.layers["counts"] = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    payload = _passport_payload()
    payload.update({"matrix_layer": "layers/counts", "sha256": "0123456789abcdef" * 4})
    passport = DatasetPassport(
        **{
            **payload,
            "required_obs_fields": tuple(payload["required_obs_fields"]),
            "required_var_fields": tuple(payload["required_var_fields"]),
            "target_mapping_fields": tuple(payload["target_mapping_fields"]),
        }
    )
    selected = select_passport_matrix(adata, passport)
    np.testing.assert_array_equal(selected.X, adata.layers["counts"])
    assert not np.array_equal(selected.X, np.zeros((2, 2), dtype=np.float32))

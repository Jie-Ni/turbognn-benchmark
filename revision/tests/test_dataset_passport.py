from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cbac_revision.artifacts import canonical_sha256, file_sha256
from cbac_revision.data import BenchmarkDataset, load_dataset_passport
from cbac_revision.errors import RevisionProtocolError
from cbac_revision.runner import PreflightBlockedError, validate_dataset_passport_binding
from cbac_revision.targets import TargetEncoding


def _payload(root: Path) -> tuple[dict[str, object], Path]:
    data_path = root / "dataset.h5ad"
    data_path.write_bytes(b"fixture-data-binding")
    (root / "control.txt").write_text("ctrl is the archived control\n", encoding="utf-8")
    (root / "legacy.txt").write_text("frozen legacy panel\n", encoding="utf-8")
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "dataset": "norman",
        "accession": "GSE133344",
        "data_file_sha256": file_sha256(data_path),
        "expression_source": {
            "container": "X",
            "declared_scale": "counts",
            "layer_key": None,
            "integer_tolerance": 1e-6,
        },
        "matrix_schema": {
            "n_cells": 5,
            "n_genes": 2,
            "condition_column": "perturbation",
            "gene_identifier_space": "var_names:HUGO_gene_symbol",
            "gene_order_sha256": canonical_sha256(["A", "B"]),
        },
        "condition_mapping": {"ctrl": "control", "A": "A", "bad": "bad"},
        "target_mapping": {"A": ["A"]},
        "condition_cell_counts": {"ctrl": 2, "A": 2, "bad": 1},
        "canonical_condition_cell_counts": {"control": 2, "A": 2, "bad": 1},
        "control_evidence": {
            "condition_column": "perturbation",
            "raw_labels": ["ctrl"],
            "raw_label_counts": {"ctrl": 2},
            "evidence_source_id": "archived-source",
            "evidence_locator": "table-1",
            "evidence_file": "control.txt",
            "evidence_file_sha256": file_sha256(root / "control.txt"),
        },
        "attrition": {
            "raw_condition_count": 3,
            "canonical_condition_count": 3,
            "excluded_raw_conditions": {"bad": "FROZEN_ENDPOINT_EXCLUDED"},
            "excluded_raw_condition_count": 1,
            "retained_target_mapped_condition_count": 1,
            "raw_cell_count": 5,
            "control_cell_count": 2,
            "excluded_cell_count": 1,
            "retained_target_mapped_cell_count": 2,
        },
        "legacy_first50_panel": {
            "ordered_condition_ids": [f"legacy_{index:02d}" for index in range(50)],
            "evidence_source_id": "submitted-analysis-archive",
            "evidence_locator": "ordered-first-50",
            "evidence_file": "legacy.txt",
            "evidence_file_sha256": file_sha256(root / "legacy.txt"),
        },
    }
    return payload, data_path


def _write(root: Path, payload: dict[str, object]) -> Path:
    path = root / "passport.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _load(root: Path, payload: dict[str, object], data_path: Path):
    return load_dataset_passport(
        _write(root, payload),
        dataset_name="norman",
        data_path=data_path,
        expected_scale="counts",
        expected_condition_column="perturbation",
    )


def _dataset(root: Path) -> BenchmarkDataset:
    return BenchmarkDataset(
        name="norman",
        expression=np.ones((5, 2), dtype=float),
        gene_names=("A", "B"),
        condition_labels=("ctrl", "ctrl", "A", "A", "bad"),
        cell_metadata=pd.DataFrame(index=range(5)),
        embedded_graphs={},
        source_path=root / "dataset.h5ad",
        source_hash=file_sha256(root / "dataset.h5ad"),
    )


def _encodings() -> dict[str, TargetEncoding]:
    return {
        "A": TargetEncoding("A", True, ("A",), (0,), None, None),
        "bad": TargetEncoding("bad", False, (), (), "FROZEN_ENDPOINT_EXCLUDED", "frozen exclusion"),
    }


def test_passport_positive_round_trip_and_loaded_dataset_binding(tmp_path: Path) -> None:
    payload, data_path = _payload(tmp_path)
    passport = _load(tmp_path, payload, data_path)
    validate_dataset_passport_binding(
        _dataset(tmp_path),
        passport,
        canonical_labels=("control", "control", "A", "A", "bad"),
        target_encodings=_encodings(),
        control_label="control",
    )


@pytest.mark.parametrize(
    "missing",
    [
        "accession",
        "matrix_schema",
        "condition_mapping",
        "target_mapping",
        "condition_cell_counts",
        "canonical_condition_cell_counts",
        "attrition",
    ],
)
def test_passport_missing_required_field_is_rejected(tmp_path: Path, missing: str) -> None:
    payload, data_path = _payload(tmp_path)
    del payload[missing]
    with pytest.raises(RevisionProtocolError, match="DATASET_PASSPORT_SCHEMA_MISMATCH"):
        _load(tmp_path, payload, data_path)


def test_synchronized_declared_mapping_forgery_is_rejected_against_rows(tmp_path: Path) -> None:
    payload, data_path = _payload(tmp_path)
    payload["condition_mapping"] = {"ctrl": "control", "A": "B", "bad": "bad"}
    payload["target_mapping"] = {"B": ["A"]}
    payload["canonical_condition_cell_counts"] = {"control": 2, "B": 2, "bad": 1}
    passport = _load(tmp_path, payload, data_path)
    with pytest.raises(PreflightBlockedError, match="CONDITION_MAPPING_MISMATCH"):
        validate_dataset_passport_binding(
            _dataset(tmp_path),
            passport,
            canonical_labels=("control", "control", "A", "A", "bad"),
            target_encodings=_encodings(),
            control_label="control",
        )


def test_gene_order_and_target_mapping_tamper_are_rejected_against_data(tmp_path: Path) -> None:
    payload, data_path = _payload(tmp_path)
    payload["matrix_schema"]["gene_order_sha256"] = canonical_sha256(["B", "A"])
    passport = _load(tmp_path, payload, data_path)
    with pytest.raises(PreflightBlockedError, match="GENE_ORDER_MISMATCH"):
        validate_dataset_passport_binding(
            _dataset(tmp_path),
            passport,
            canonical_labels=("control", "control", "A", "A", "bad"),
            target_encodings=_encodings(),
            control_label="control",
        )

    payload, data_path = _payload(tmp_path)
    payload["target_mapping"] = {"A": ["B"]}
    passport = _load(tmp_path, payload, data_path)
    with pytest.raises(PreflightBlockedError, match="TARGET_MAPPING_MISMATCH"):
        validate_dataset_passport_binding(
            _dataset(tmp_path),
            passport,
            canonical_labels=("control", "control", "A", "A", "bad"),
            target_encodings=_encodings(),
            control_label="control",
        )


def test_attrition_internal_and_actual_row_inconsistency_are_rejected(tmp_path: Path) -> None:
    payload, data_path = _payload(tmp_path)
    payload["attrition"]["excluded_cell_count"] = 2
    with pytest.raises(RevisionProtocolError, match="ATTRITION_TARGET_MISMATCH"):
        _load(tmp_path, payload, data_path)

    payload, data_path = _payload(tmp_path)
    payload["attrition"]["excluded_raw_conditions"] = {"bad": "SYNCHRONIZED_BUT_FALSE_REASON"}
    passport = _load(tmp_path, payload, data_path)
    with pytest.raises(PreflightBlockedError, match="ATTRITION_REASON_MISMATCH"):
        validate_dataset_passport_binding(
            _dataset(tmp_path),
            passport,
            canonical_labels=("control", "control", "A", "A", "bad"),
            target_encodings=_encodings(),
            control_label="control",
        )


def test_control_count_and_adamson_exact_label_contracts_are_rejected(tmp_path: Path) -> None:
    payload, data_path = _payload(tmp_path)
    payload["control_evidence"]["raw_label_counts"] = {"ctrl": 1}
    with pytest.raises(RevisionProtocolError, match="CONTROL_EVIDENCE_INVALID"):
        _load(tmp_path, payload, data_path)

    payload, data_path = _payload(tmp_path)
    payload["dataset"] = "adamson"
    with pytest.raises(RevisionProtocolError, match="ADAMSON_CONTROL_EVIDENCE_LABEL_SET"):
        load_dataset_passport(
            _write(tmp_path, payload),
            dataset_name="adamson",
            data_path=data_path,
            expected_scale="counts",
            expected_condition_column="perturbation",
        )


def test_unverified_control_synonym_cannot_be_hidden_in_mapping(tmp_path: Path) -> None:
    payload, data_path = _payload(tmp_path)
    passport = _load(tmp_path, payload, data_path)
    forged = copy.deepcopy(_dataset(tmp_path))
    object.__setattr__(
        forged,
        "condition_labels",
        ("non-targeting", "ctrl", "A", "A", "bad"),
    )
    with pytest.raises(PreflightBlockedError, match="RAW_CONDITION_COUNTS_MISMATCH"):
        validate_dataset_passport_binding(
            forged,
            passport,
            canonical_labels=("control", "control", "A", "A", "bad"),
            target_encodings=_encodings(),
            control_label="control",
        )

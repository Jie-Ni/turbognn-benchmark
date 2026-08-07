"""Dataset loaders for auditable NPZ fixtures and optional H5AD inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .artifacts import SHA256_RE, canonical_sha256, file_sha256
from .errors import RevisionProtocolError


@dataclass(frozen=True)
class BenchmarkDataset:
    """In-memory cell-by-gene data plus cell metadata and embedded graph sources."""

    name: str
    expression: np.ndarray | sparse.spmatrix
    gene_names: tuple[str, ...]
    condition_labels: tuple[str, ...]
    cell_metadata: pd.DataFrame
    embedded_graphs: Mapping[str, tuple[tuple[str, str], ...]]
    source_path: Path
    source_hash: str

    def validate(self) -> None:
        """Validate dimensions and identifiers before any preflight calculation."""

        if self.expression.ndim != 2:
            raise RevisionProtocolError("Dataset expression must be cell-by-gene")
        if self.expression.shape != (len(self.condition_labels), len(self.gene_names)):
            raise RevisionProtocolError("Expression dimensions do not match labels and genes")
        if len(self.cell_metadata) != len(self.condition_labels):
            raise RevisionProtocolError("Cell metadata row count does not match expression")
        if len(set(self.gene_names)) != len(self.gene_names):
            raise RevisionProtocolError("Dataset gene names must be unique")
        finite_values = (
            self.expression.data if sparse.issparse(self.expression) else self.expression
        )
        if not np.isfinite(finite_values).all():
            raise RevisionProtocolError("Dataset expression contains non-finite values")


@dataclass(frozen=True)
class ExpressionSource:
    """Explicit H5AD expression container; no X/layer/raw fallback is permitted."""

    container: str
    declared_scale: str
    layer_key: str | None = None
    integer_tolerance: float = 1e-6

    def validate(self) -> None:
        if self.container not in {"X", "layer", "raw"}:
            raise RevisionProtocolError("expression_source.container must be X, layer, or raw")
        if self.container == "layer" and not self.layer_key:
            raise RevisionProtocolError("A layer expression source requires layer_key")
        if self.container != "layer" and self.layer_key is not None:
            raise RevisionProtocolError("layer_key is valid only for a layer expression source")
        if self.declared_scale not in {"counts", "log1p"}:
            raise RevisionProtocolError("expression source scale must be counts or log1p")
        if self.integer_tolerance < 0:
            raise RevisionProtocolError("integer_tolerance cannot be negative")


@dataclass(frozen=True)
class DatasetPassport:
    """Hash-verified binding between data, expression container, and control evidence."""

    dataset_name: str
    accession: str
    data_file_sha256: str
    expression_source: ExpressionSource
    matrix_schema: Mapping[str, Any]
    condition_mapping: Mapping[str, str]
    target_mapping: Mapping[str, tuple[str, ...]]
    condition_cell_counts: Mapping[str, int]
    canonical_condition_cell_counts: Mapping[str, int]
    attrition: Mapping[str, Any]
    control_evidence: "ControlEvidence"
    legacy_first50_panel: "LegacyFirst50Panel"
    source_path: Path
    source_hash: str


@dataclass(frozen=True)
class ControlEvidence:
    """Explicit raw control labels backed by a locally hash-verified source."""

    condition_column: str
    raw_labels: tuple[str, ...]
    raw_label_counts: Mapping[str, int]
    evidence_source_id: str
    evidence_locator: str
    evidence_file_source_id: str
    evidence_file_sha256: str


@dataclass(frozen=True)
class LegacyFirst50Panel:
    """Frozen legacy first-50 conditions excluded from every new evaluation panel."""

    ordered_condition_ids: tuple[str, ...]
    evidence_source_id: str
    evidence_locator: str
    evidence_file_source_id: str
    evidence_file_sha256: str


def load_benchmark_dataset(
    path: Path,
    dataset_name: str,
    condition_column: str,
    metadata_columns: Sequence[str] = (),
    expression_source: ExpressionSource | None = None,
) -> BenchmarkDataset:
    """Load an NPZ fixture or H5AD dataset without scale auto-detection."""

    suffix = path.suffix.casefold()
    if suffix == ".npz":
        dataset = _load_npz(path, dataset_name, metadata_columns)
    elif suffix == ".h5ad":
        if expression_source is None:
            raise RevisionProtocolError(
                "[DATASET_PASSPORT_MISSING] H5AD input requires an explicit X/layer/raw source"
            )
        dataset = _load_h5ad(
            path, dataset_name, condition_column, metadata_columns, expression_source
        )
    else:
        raise RevisionProtocolError("Data input must be an .npz fixture or .h5ad file")
    dataset.validate()
    source = expression_source or ExpressionSource(container="X", declared_scale="counts")
    _validate_declared_expression_scale(dataset.expression, source)
    return dataset


def load_dataset_passport(
    path: Path,
    *,
    dataset_name: str,
    data_path: Path,
    expected_scale: str,
    expected_condition_column: str | None,
) -> DatasetPassport:
    """Load a production passport and verify data, scale, and control evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise RevisionProtocolError(f"[DATASET_PASSPORT_INVALID] {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise RevisionProtocolError("[DATASET_PASSPORT_INVALID] schema_version must be 1.0")
    required_root_fields = {
        "schema_version",
        "dataset",
        "accession",
        "data_file_sha256",
        "expression_source",
        "matrix_schema",
        "condition_mapping",
        "target_mapping",
        "condition_cell_counts",
        "canonical_condition_cell_counts",
        "control_evidence",
        "attrition",
        "legacy_first50_panel",
    }
    if set(payload) != required_root_fields:
        raise RevisionProtocolError("[DATASET_PASSPORT_SCHEMA_MISMATCH]")
    if payload.get("dataset") != dataset_name:
        raise RevisionProtocolError("[DATASET_PASSPORT_DATASET_MISMATCH]")
    accession = str(payload.get("accession", "")).strip()
    if not accession or "REPLACE" in accession.upper():
        raise RevisionProtocolError("[DATASET_PASSPORT_ACCESSION_INVALID]")
    source_payload = payload.get("expression_source")
    if not isinstance(source_payload, dict):
        raise RevisionProtocolError("[DATASET_PASSPORT_INVALID] expression_source is required")
    source = ExpressionSource(
        container=str(source_payload.get("container", "")),
        declared_scale=str(source_payload.get("declared_scale", "")),
        layer_key=source_payload.get("layer_key"),
        integer_tolerance=float(source_payload.get("integer_tolerance", 1e-6)),
    )
    source.validate()
    if source.declared_scale != expected_scale:
        raise RevisionProtocolError(
            "[DATASET_PASSPORT_SCALE_MISMATCH] "
            f"passport={source.declared_scale}; protocol={expected_scale}"
        )
    observed_data_hash = file_sha256(data_path)
    if payload.get("data_file_sha256") != observed_data_hash:
        raise RevisionProtocolError("[DATASET_PASSPORT_FILE_HASH_MISMATCH]")
    matrix_schema = payload.get("matrix_schema")
    if (
        not isinstance(matrix_schema, dict)
        or set(matrix_schema)
        != {
            "n_cells",
            "n_genes",
            "condition_column",
            "gene_identifier_space",
            "gene_order_sha256",
        }
        or any(
            isinstance(matrix_schema[field], bool)
            or not isinstance(matrix_schema[field], int)
            or matrix_schema[field] <= 0
            for field in ("n_cells", "n_genes")
        )
        or not all(
            isinstance(matrix_schema[field], str)
            and matrix_schema[field].strip()
            and "REPLACE" not in matrix_schema[field].upper()
            for field in ("condition_column", "gene_identifier_space")
        )
        or not SHA256_RE.fullmatch(str(matrix_schema.get("gene_order_sha256", "")))
    ):
        raise RevisionProtocolError("[DATASET_PASSPORT_MATRIX_SCHEMA_INVALID]")
    condition_mapping = payload.get("condition_mapping")
    condition_cell_counts = payload.get("condition_cell_counts")
    canonical_condition_cell_counts = payload.get("canonical_condition_cell_counts")
    if (
        not isinstance(condition_mapping, dict)
        or not condition_mapping
        or any(
            not isinstance(raw, str)
            or not raw.strip()
            or not isinstance(canonical, str)
            or not canonical.strip()
            for raw, canonical in condition_mapping.items()
        )
        or not isinstance(condition_cell_counts, dict)
        or set(condition_cell_counts) != set(condition_mapping)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in condition_cell_counts.values()
        )
        or sum(condition_cell_counts.values()) != matrix_schema["n_cells"]
        or not isinstance(canonical_condition_cell_counts, dict)
        or set(canonical_condition_cell_counts) != set(condition_mapping.values())
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in canonical_condition_cell_counts.values()
        )
        or sum(canonical_condition_cell_counts.values()) != matrix_schema["n_cells"]
        or any(
            canonical_condition_cell_counts[canonical]
            != sum(
                condition_cell_counts[raw]
                for raw, mapped in condition_mapping.items()
                if mapped == canonical
            )
            for canonical in canonical_condition_cell_counts
        )
    ):
        raise RevisionProtocolError("[DATASET_PASSPORT_CONDITION_CONTRACT_INVALID]")
    target_mapping_raw = payload.get("target_mapping")
    if not isinstance(target_mapping_raw, dict) or any(
        not isinstance(condition, str)
        or not condition
        or not isinstance(targets, list)
        or not targets
        or len(set(targets)) != len(targets)
        or any(not isinstance(target, str) or not target.strip() for target in targets)
        for condition, targets in target_mapping_raw.items()
    ):
        raise RevisionProtocolError("[DATASET_PASSPORT_TARGET_MAPPING_INVALID]")
    control_payload = payload.get("control_evidence")
    if not isinstance(control_payload, dict):
        raise RevisionProtocolError(
            "[CONTROL_EVIDENCE_MISSING] Production H5AD requires explicit control evidence"
        )
    raw_labels = tuple(str(value).strip() for value in control_payload.get("raw_labels", ()))
    if not raw_labels or any(not value or "REPLACE" in value.upper() for value in raw_labels):
        raise RevisionProtocolError("[CONTROL_EVIDENCE_INVALID] raw_labels must be explicit")
    if len({value.casefold() for value in raw_labels}) != len(raw_labels):
        raise RevisionProtocolError("[CONTROL_EVIDENCE_INVALID] raw_labels contain duplicates")
    raw_label_counts = control_payload.get("raw_label_counts")
    if (
        not isinstance(raw_label_counts, dict)
        or set(raw_label_counts) != set(raw_labels)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in raw_label_counts.values()
        )
        or any(raw_label_counts[label] != condition_cell_counts[label] for label in raw_labels)
    ):
        raise RevisionProtocolError(
            "[CONTROL_EVIDENCE_INVALID] raw_label_counts must bind every label exactly"
        )
    if dataset_name == "adamson" and set(raw_labels) != {
        "62(mod)_pBA581",
        "63(mod)_pBA580",
    }:
        raise RevisionProtocolError("[ADAMSON_CONTROL_EVIDENCE_LABEL_SET_MISMATCH]")
    condition_column = str(control_payload.get("condition_column", ""))
    if expected_condition_column is not None and condition_column != expected_condition_column:
        raise RevisionProtocolError(
            "[CONTROL_EVIDENCE_CONDITION_COLUMN_MISMATCH] "
            f"passport={condition_column!r}; protocol={expected_condition_column!r}"
        )
    evidence_source_id = str(control_payload.get("evidence_source_id", "")).strip()
    evidence_locator = str(control_payload.get("evidence_locator", "")).strip()
    evidence_file_source_id = str(control_payload.get("evidence_file", "")).strip()
    expected_evidence_hash = str(control_payload.get("evidence_file_sha256", ""))
    text_fields = (evidence_source_id, evidence_locator, evidence_file_source_id)
    if any(not value or "REPLACE" in value.upper() for value in text_fields):
        raise RevisionProtocolError(
            "[CONTROL_EVIDENCE_INVALID] source, locator, and evidence_file are required"
        )
    if not SHA256_RE.fullmatch(expected_evidence_hash):
        raise RevisionProtocolError("[CONTROL_EVIDENCE_INVALID] evidence hash must be SHA-256")
    evidence_file = Path(evidence_file_source_id)
    if evidence_file.is_absolute():
        raise RevisionProtocolError(
            "[CONTROL_EVIDENCE_INVALID] evidence_file must be passport-relative"
        )
    passport_root = path.resolve().parent
    resolved_evidence = (passport_root / evidence_file).resolve()
    if not resolved_evidence.is_relative_to(passport_root):
        raise RevisionProtocolError(
            "[CONTROL_EVIDENCE_INVALID] evidence_file escapes the passport directory"
        )
    if not resolved_evidence.is_file():
        raise RevisionProtocolError("[CONTROL_EVIDENCE_FILE_MISSING]")
    if file_sha256(resolved_evidence) != expected_evidence_hash:
        raise RevisionProtocolError("[CONTROL_EVIDENCE_FILE_HASH_MISMATCH]")
    evidence = ControlEvidence(
        condition_column=condition_column,
        raw_labels=raw_labels,
        raw_label_counts={str(key): int(value) for key, value in raw_label_counts.items()},
        evidence_source_id=evidence_source_id,
        evidence_locator=evidence_locator,
        evidence_file_source_id=evidence_file_source_id,
        evidence_file_sha256=expected_evidence_hash,
    )
    attrition = payload.get("attrition")
    attrition_fields = {
        "raw_condition_count",
        "canonical_condition_count",
        "excluded_raw_conditions",
        "excluded_raw_condition_count",
        "retained_target_mapped_condition_count",
        "raw_cell_count",
        "control_cell_count",
        "excluded_cell_count",
        "retained_target_mapped_cell_count",
    }
    if not isinstance(attrition, dict) or set(attrition) != attrition_fields:
        raise RevisionProtocolError("[DATASET_PASSPORT_ATTRITION_INVALID]")
    excluded = attrition.get("excluded_raw_conditions")
    if (
        not isinstance(excluded, dict)
        or not set(excluded) <= set(condition_mapping)
        or any(not isinstance(reason, str) or not reason.strip() for reason in excluded.values())
        or attrition.get("raw_condition_count") != len(condition_mapping)
        or attrition.get("canonical_condition_count") != len(set(condition_mapping.values()))
        or attrition.get("excluded_raw_condition_count") != len(excluded)
        or attrition.get("raw_cell_count") != matrix_schema["n_cells"]
    ):
        raise RevisionProtocolError("[DATASET_PASSPORT_ATTRITION_INVALID]")
    control_canonicals = {condition_mapping[label] for label in raw_labels}
    if len(control_canonicals) != 1:
        raise RevisionProtocolError("[DATASET_PASSPORT_CONTROL_MAPPING_INVALID]")
    excluded_canonicals = {condition_mapping[label] for label in excluded}
    retained_canonicals = set(condition_mapping.values()) - control_canonicals - excluded_canonicals
    if (
        set(target_mapping_raw) != retained_canonicals
        or attrition.get("retained_target_mapped_condition_count") != len(retained_canonicals)
        or attrition.get("control_cell_count")
        != sum(condition_cell_counts[label] for label in raw_labels)
        or attrition.get("excluded_cell_count")
        != sum(condition_cell_counts[label] for label in excluded)
        or attrition.get("retained_target_mapped_cell_count")
        != sum(canonical_condition_cell_counts[label] for label in retained_canonicals)
        or attrition["control_cell_count"]
        + attrition["excluded_cell_count"]
        + attrition["retained_target_mapped_cell_count"]
        != matrix_schema["n_cells"]
    ):
        raise RevisionProtocolError("[DATASET_PASSPORT_ATTRITION_TARGET_MISMATCH]")
    legacy_payload = payload.get("legacy_first50_panel")
    if not isinstance(legacy_payload, dict):
        raise RevisionProtocolError(
            "[LEGACY_FIRST50_EVIDENCE_MISSING] Production passport requires the frozen panel"
        )
    legacy_conditions = tuple(
        str(value).strip() for value in legacy_payload.get("ordered_condition_ids", ())
    )
    if (
        len(legacy_conditions) != 50
        or len(set(legacy_conditions)) != 50
        or any(not value or "REPLACE" in value.upper() for value in legacy_conditions)
    ):
        raise RevisionProtocolError(
            "[LEGACY_FIRST50_EVIDENCE_INVALID] Exactly 50 unique ordered IDs are required"
        )
    legacy_source_id = str(legacy_payload.get("evidence_source_id", "")).strip()
    legacy_locator = str(legacy_payload.get("evidence_locator", "")).strip()
    legacy_file_source_id = str(legacy_payload.get("evidence_file", "")).strip()
    legacy_hash = str(legacy_payload.get("evidence_file_sha256", ""))
    if any(
        not value or "REPLACE" in value.upper()
        for value in (legacy_source_id, legacy_locator, legacy_file_source_id)
    ) or not SHA256_RE.fullmatch(legacy_hash):
        raise RevisionProtocolError(
            "[LEGACY_FIRST50_EVIDENCE_INVALID] Source, locator, file, and SHA-256 are required"
        )
    legacy_file = Path(legacy_file_source_id)
    if legacy_file.is_absolute():
        raise RevisionProtocolError(
            "[LEGACY_FIRST50_EVIDENCE_INVALID] evidence_file must be passport-relative"
        )
    resolved_legacy = (passport_root / legacy_file).resolve()
    if not resolved_legacy.is_relative_to(passport_root):
        raise RevisionProtocolError(
            "[LEGACY_FIRST50_EVIDENCE_INVALID] evidence_file escapes the passport directory"
        )
    if not resolved_legacy.is_file():
        raise RevisionProtocolError("[LEGACY_FIRST50_EVIDENCE_FILE_MISSING]")
    if file_sha256(resolved_legacy) != legacy_hash:
        raise RevisionProtocolError("[LEGACY_FIRST50_EVIDENCE_FILE_HASH_MISMATCH]")
    legacy_panel = LegacyFirst50Panel(
        ordered_condition_ids=legacy_conditions,
        evidence_source_id=legacy_source_id,
        evidence_locator=legacy_locator,
        evidence_file_source_id=legacy_file_source_id,
        evidence_file_sha256=legacy_hash,
    )
    return DatasetPassport(
        dataset_name=dataset_name,
        accession=accession,
        data_file_sha256=observed_data_hash,
        expression_source=source,
        matrix_schema=dict(matrix_schema),
        condition_mapping={str(key): str(value) for key, value in condition_mapping.items()},
        target_mapping={
            str(key): tuple(str(value) for value in values)
            for key, values in target_mapping_raw.items()
        },
        condition_cell_counts={
            str(key): int(value) for key, value in condition_cell_counts.items()
        },
        canonical_condition_cell_counts={
            str(key): int(value) for key, value in canonical_condition_cell_counts.items()
        },
        attrition=dict(attrition),
        control_evidence=evidence,
        legacy_first50_panel=legacy_panel,
        source_path=path,
        source_hash=file_sha256(path),
    )


def write_npz_fixture(
    path: Path,
    expression: np.ndarray,
    gene_names: Sequence[str],
    condition_labels: Sequence[str],
    *,
    cell_metadata: Mapping[str, Sequence[object]] | None = None,
    embedded_graphs: Mapping[str, Sequence[tuple[str, str]]] | None = None,
) -> Path:
    """Write the documented small-fixture schema used by CPU integration tests."""

    payload: dict[str, np.ndarray] = {
        "X": np.asarray(expression),
        "gene_names": np.asarray(gene_names, dtype=str),
        "condition_labels": np.asarray(condition_labels, dtype=str),
    }
    for column, values in (cell_metadata or {}).items():
        payload[f"metadata__{column}"] = np.asarray(values)
    for graph, edges in (embedded_graphs or {}).items():
        payload[f"graph__{graph}"] = np.asarray(edges, dtype=str).reshape(-1, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def load_named_edges(path: Path) -> tuple[tuple[str, str], ...]:
    """Load a two-column named edge list from CSV, TSV, NPY, or NPZ."""

    suffix = path.suffix.casefold()
    if suffix in {".csv", ".tsv"}:
        separator = "\t" if suffix == ".tsv" else ","
        frame = pd.read_csv(path, sep=separator)
        if {"source", "target"} <= set(frame.columns):
            array = frame[["source", "target"]].to_numpy(dtype=str)
        elif frame.shape[1] == 2:
            array = frame.to_numpy(dtype=str)
        else:
            raise RevisionProtocolError(f"Graph file {path} requires source and target columns")
    elif suffix == ".npy":
        array = np.load(path, allow_pickle=False).astype(str)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if "edges" not in payload:
                raise RevisionProtocolError(f"Graph NPZ {path} requires an 'edges' array")
            array = payload["edges"].astype(str)
    else:
        raise RevisionProtocolError(f"Unsupported graph edge format: {path.suffix}")
    if array.ndim != 2 or array.shape[1] != 2:
        raise RevisionProtocolError(f"Graph edge list {path} must have shape (n_edges, 2)")
    return tuple((str(left), str(right)) for left, right in array)


def hash_named_edges(edges: Sequence[tuple[str, str]]) -> str:
    """Hash a named edge list independently of input row direction or order."""

    canonical = sorted(
        {tuple(sorted((str(left), str(right)))) for left, right in edges if left != right}
    )
    return canonical_sha256(canonical)


def _load_npz(path: Path, dataset_name: str, metadata_columns: Sequence[str]) -> BenchmarkDataset:
    with np.load(path, allow_pickle=False) as payload:
        required = {"X", "gene_names", "condition_labels"}
        missing = required - set(payload.files)
        if missing:
            raise RevisionProtocolError(f"NPZ fixture is missing {sorted(missing)}")
        expression = np.asarray(payload["X"], dtype=np.float64)
        genes = tuple(str(value) for value in payload["gene_names"].tolist())
        labels = tuple(str(value) for value in payload["condition_labels"].tolist())
        metadata: dict[str, np.ndarray] = {}
        available_metadata = {
            key.removeprefix("metadata__"): key
            for key in payload.files
            if key.startswith("metadata__")
        }
        for column in metadata_columns:
            if column in available_metadata:
                metadata[column] = np.asarray(payload[available_metadata[column]])
        for column, key in available_metadata.items():
            if column not in metadata:
                metadata[column] = np.asarray(payload[key])
        graphs = {
            key.removeprefix("graph__"): tuple(
                (str(left), str(right)) for left, right in np.asarray(payload[key]).reshape(-1, 2)
            )
            for key in payload.files
            if key.startswith("graph__")
        }
    return BenchmarkDataset(
        name=dataset_name,
        expression=expression,
        gene_names=genes,
        condition_labels=labels,
        cell_metadata=pd.DataFrame(metadata, index=np.arange(len(labels))),
        embedded_graphs=graphs,
        source_path=path,
        source_hash=file_sha256(path),
    )


def _load_h5ad(
    path: Path,
    dataset_name: str,
    condition_column: str,
    metadata_columns: Sequence[str],
    expression_source: ExpressionSource,
) -> BenchmarkDataset:
    try:
        import anndata
    except ImportError as error:
        raise RevisionProtocolError(
            "Reading H5AD requires the optional 'anndata' package"
        ) from error
    adata = anndata.read_h5ad(path)
    if condition_column not in adata.obs:
        raise RevisionProtocolError(
            f"Declared condition column {condition_column!r} is absent from {path}"
        )
    metadata = adata.obs[[column for column in metadata_columns if column in adata.obs]].copy()
    expression, gene_names = _select_h5ad_expression(adata, expression_source)
    return BenchmarkDataset(
        name=dataset_name,
        expression=expression,
        gene_names=gene_names,
        condition_labels=tuple(
            "__MISSING_CONDITION__" if pd.isna(value) else str(value)
            for value in adata.obs[condition_column]
        ),
        cell_metadata=metadata.reset_index(drop=True),
        embedded_graphs={},
        source_path=path,
        source_hash=file_sha256(path),
    )


def _select_h5ad_expression(
    adata: object, expression_source: ExpressionSource
) -> tuple[np.ndarray | sparse.spmatrix, tuple[str, ...]]:
    """Select exactly the passport-declared H5AD container."""

    expression_source.validate()
    if expression_source.container == "X":
        matrix = adata.X
        names = adata.var_names
    elif expression_source.container == "layer":
        if expression_source.layer_key not in adata.layers:
            raise RevisionProtocolError(
                f"[DECLARED_COUNTS_LAYER_MISSING] {expression_source.layer_key!r}"
            )
        matrix = adata.layers[expression_source.layer_key]
        names = adata.var_names
    else:
        if adata.raw is None:
            raise RevisionProtocolError("[DECLARED_RAW_MATRIX_MISSING]")
        matrix = adata.raw.X
        names = adata.raw.var_names
    output = (
        matrix.astype(np.float64).tocsr()
        if sparse.issparse(matrix)
        else np.asarray(matrix, dtype=np.float64)
    )
    _validate_declared_expression_scale(output, expression_source)
    return output, tuple(str(value) for value in names)


def _validate_declared_expression_scale(
    expression: np.ndarray | sparse.spmatrix, expression_source: ExpressionSource
) -> None:
    expression_scale_audit(expression, expression_source)


def expression_scale_audit(
    expression: np.ndarray | sparse.spmatrix, expression_source: ExpressionSource
) -> dict[str, float | int | str]:
    """Validate and summarize the full declared expression matrix."""

    expression_source.validate()
    matrix = (
        expression.astype(np.float64)
        if sparse.issparse(expression)
        else np.asarray(expression, dtype=np.float64)
    )
    explicit_values = matrix.data if sparse.issparse(matrix) else matrix
    if not np.isfinite(explicit_values).all():
        raise RevisionProtocolError("[EXPRESSION_NONFINITE]")
    if (explicit_values < 0).any():
        raise RevisionProtocolError("[EXPRESSION_NEGATIVE]")
    maximum_residual = 0.0
    violating_values = 0
    if expression_source.declared_scale == "counts":
        residuals = np.abs(explicit_values - np.rint(explicit_values))
        maximum_residual = float(np.max(residuals, initial=0.0))
        violating_values = int(np.sum(residuals > expression_source.integer_tolerance))
        if maximum_residual > expression_source.integer_tolerance:
            raise RevisionProtocolError(
                "[COUNT_MATRIX_NOT_INTEGER_LIKE] "
                f"maximum_integer_residual={maximum_residual:.6g}; "
                f"violating_values={violating_values}; "
                f"tolerance={expression_source.integer_tolerance:.6g}"
            )
    return {
        "status": "PASS",
        "container": expression_source.container,
        "layer_key": expression_source.layer_key or "NOT_APPLICABLE",
        "declared_scale": expression_source.declared_scale,
        "n_cells": int(matrix.shape[0]),
        "n_genes": int(matrix.shape[1]),
        "n_values": int(matrix.shape[0] * matrix.shape[1]),
        "storage": "sparse" if sparse.issparse(matrix) else "dense",
        "explicit_stored_values": int(matrix.nnz) if sparse.issparse(matrix) else int(matrix.size),
        "minimum_value": float(matrix.min()),
        "maximum_value": float(matrix.max()),
        "integer_tolerance": float(expression_source.integer_tolerance),
        "maximum_integer_residual": maximum_residual,
        "noninteger_value_count": violating_values,
    }

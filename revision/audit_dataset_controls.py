"""Build a source-backed control-label audit for the staged Adamson and Norman datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def categorical_counts(path: Path, column: str) -> tuple[dict[str, int], int, tuple[int, int]]:
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("h5py and numpy are required for this audit") from exc

    with h5py.File(path, "r") as handle:
        if column not in handle["obs"]:
            raise ValueError(f"Missing obs column {column!r} in {path.name}")
        node = handle["obs"][column]
        categories = [decode(value) for value in node["categories"][:]]
        codes = node["codes"][:]
        valid_codes = codes[codes >= 0]
        values = np.bincount(valid_codes, minlength=len(categories))
        counts = {category: int(values[index]) for index, category in enumerate(categories)}
        missing = int((codes < 0).sum())
        x_node = handle["X"]
        raw_shape = x_node.attrs["shape"] if "shape" in x_node.attrs else x_node.shape
        shape = tuple(int(value) for value in raw_shape)
    return counts, missing, shape


def notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adamson-h5ad", required=True, type=Path)
    parser.add_argument("--norman-h5ad", required=True, type=Path)
    parser.add_argument("--pfizer-notebook", required=True, type=Path)
    parser.add_argument("--pfizer-commit", required=True)
    parser.add_argument("--adamson-table-s1", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    adamson = args.adamson_h5ad.resolve()
    norman = args.norman_h5ad.resolve()
    notebook = args.pfizer_notebook.resolve()
    table_s1 = args.adamson_table_s1.resolve()

    adamson_counts, adamson_missing, adamson_shape = categorical_counts(adamson, "perturbation")
    norman_counts, norman_missing, norman_shape = categorical_counts(norman, "perturbation")
    source = notebook_source(notebook)
    required_notebook_checks = [
        "s.split('_')[0]",
        "replace('62(mod)', 'control')",
        "replace('63(mod)', 'control')",
    ]
    missing_checks = [check for check in required_notebook_checks if check not in source]
    if missing_checks:
        raise ValueError(f"Source-rebuild notebook is missing checks: {missing_checks}")

    control_rows = [
        {
            "dataset": "Adamson",
            "condition_column": "perturbation",
            "verified_control_labels": "62(mod)_pBA581|63(mod)_pBA580",
            "verified_control_cells": adamson_counts.get("62(mod)_pBA581", 0)
            + adamson_counts.get("63(mod)_pBA580", 0),
            "control_evidence": (
                "source-rebuild notebook strips the vector suffix and maps both 62(mod) and "
                f"63(mod) to control; repository commit {args.pfizer_commit}"
            ),
            "endpoint_exclusions": "Gal4-4(mod)_pBA582|*|missing perturbation annotation",
            "exclusion_reason": (
                "not verified as a human-gene perturbation endpoint; missing annotations are ineligible"
            ),
            "legacy_fallback_result": "63(mod)_pBA580 selected only because it was modal",
            "n_obs": adamson_shape[0],
            "n_vars": adamson_shape[1],
        },
        {
            "dataset": "Norman",
            "condition_column": "perturbation",
            "verified_control_labels": "control",
            "verified_control_cells": norman_counts.get("control", 0),
            "control_evidence": "explicit source-H5AD perturbation category",
            "endpoint_exclusions": "",
            "exclusion_reason": "",
            "legacy_fallback_result": "explicit control label recognised",
            "n_obs": norman_shape[0],
            "n_vars": norman_shape[1],
        },
    ]
    write_csv(
        output_dir / "dataset_control_contract.csv",
        control_rows,
        [
            "dataset",
            "condition_column",
            "verified_control_labels",
            "verified_control_cells",
            "control_evidence",
            "endpoint_exclusions",
            "exclusion_reason",
            "legacy_fallback_result",
            "n_obs",
            "n_vars",
        ],
    )

    count_rows = [
        {
            "dataset": "Adamson",
            "raw_label": "62(mod)_pBA581",
            "n_cells": adamson_counts.get("62(mod)_pBA581", 0),
            "status": "verified_control",
        },
        {
            "dataset": "Adamson",
            "raw_label": "63(mod)_pBA580",
            "n_cells": adamson_counts.get("63(mod)_pBA580", 0),
            "status": "verified_control",
        },
        {
            "dataset": "Adamson",
            "raw_label": "Gal4-4(mod)_pBA582",
            "n_cells": adamson_counts.get("Gal4-4(mod)_pBA582", 0),
            "status": "excluded_unresolved_control_like_label",
        },
        {
            "dataset": "Adamson",
            "raw_label": "*",
            "n_cells": adamson_counts.get("*", 0),
            "status": "excluded_unresolved_label",
        },
        {
            "dataset": "Adamson",
            "raw_label": "<missing>",
            "n_cells": adamson_missing,
            "status": "excluded_missing_annotation",
        },
        {
            "dataset": "Norman",
            "raw_label": "control",
            "n_cells": norman_counts.get("control", 0),
            "status": "verified_control",
        },
        {
            "dataset": "Norman",
            "raw_label": "<missing>",
            "n_cells": norman_missing,
            "status": "excluded_missing_annotation",
        },
    ]
    write_csv(
        output_dir / "dataset_control_label_counts.csv",
        count_rows,
        ["dataset", "raw_label", "n_cells", "status"],
    )

    source_rows = [
        {
            "source_id": "adamson_h5ad",
            "file_name": adamson.name,
            "sha256": digest(adamson, "sha256"),
            "md5": digest(adamson, "md5"),
            "source_url": "https://doi.org/10.5281/zenodo.10044268",
        },
        {
            "source_id": "norman_h5ad",
            "file_name": norman.name,
            "sha256": digest(norman, "sha256"),
            "md5": digest(norman, "md5"),
            "source_url": "https://doi.org/10.5281/zenodo.10044268",
        },
        {
            "source_id": "pfizer_source_rebuild_notebook",
            "file_name": notebook.name,
            "sha256": digest(notebook, "sha256"),
            "md5": "",
            "source_url": (
                "https://github.com/pfizer-opensource/perturb_seq/blob/"
                f"{args.pfizer_commit}/dataset_correction/{notebook.name}"
            ),
        },
        {
            "source_id": "adamson_table_s1",
            "file_name": table_s1.name,
            "sha256": digest(table_s1, "sha256"),
            "md5": digest(table_s1, "md5"),
            "source_url": (
                "https://ars.els-cdn.com/content/image/" "1-s2.0-S0092867416316609-mmc1.xlsx"
            ),
        },
    ]
    write_csv(
        output_dir / "dataset_control_source_manifest.csv",
        source_rows,
        ["source_id", "file_name", "sha256", "md5", "source_url"],
    )

    report = f"""# Dataset control-label audit

## Frozen production contract

The staged Adamson and Norman files use `obs.perturbation`; neither uses `obs.gene`. Norman has an
explicit `control` category ({norman_counts.get('control', 0):,} cells). The Adamson file has no
literal `control` category. A source-rebuild notebook at Pfizer's public repository commit
`{args.pfizer_commit}` strips the vector suffix and maps both `62(mod)` and `63(mod)` to `control`.
The exact staged labels are `62(mod)_pBA581` ({adamson_counts.get('62(mod)_pBA581', 0):,} cells) and
`63(mod)_pBA580` ({adamson_counts.get('63(mod)_pBA580', 0):,} cells); they are pooled only as the
verified control reference.

The Adamson labels `Gal4-4(mod)_pBA582`, `*`, and missing perturbation annotations are not admitted
as prediction endpoints. `Gal4-4(mod)_pBA582` is control-like/non-human but is not silently pooled
without an exact source mapping. This exclusion prevents the archived error in which a modal-label
fallback selected `63(mod)_pBA580` while another control-like label entered the 50-condition panel.

## Release gate

The production runner must match both the condition column and every control/exclusion label above,
record counts in the dataset passport, and stop on any mismatch. It may not choose the modal condition
or infer a control from string frequency. Different Adamson guide labels remain distinct held-out
conditions; a shared canonical target is used for target masking and related-target exclusion, not for
automatic guide pooling.
"""
    (output_dir / "dataset_control_audit_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

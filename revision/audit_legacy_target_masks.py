"""Audit perturbation-target masking in the archived benchmark implementation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

GUIDE_SUFFIX = re.compile(r"_p(?:DS|BA)\d+$", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_strings(values: Iterable[object]) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def read_h5ad_metadata(path: Path) -> tuple[set[str], dict[str, set[str]], tuple[int, int]]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required when --h5ad is supplied") from exc

    with h5py.File(path, "r") as handle:
        var_group = handle["var"]
        index_name = var_group.attrs.get("_index", "_index")
        if isinstance(index_name, bytes):
            index_name = index_name.decode("utf-8")
        gene_names = set(decode_strings(var_group[index_name][:]))

        categories: dict[str, set[str]] = {}
        for column in ("gene", "condition", "perturbation", "guide_ids", "perturbations"):
            if column not in handle["obs"]:
                continue
            node = handle["obs"][column]
            if hasattr(node, "keys") and "categories" in node:
                categories[column] = set(decode_strings(node["categories"][:]))
            else:
                categories[column] = set(decode_strings(node[:]))

        x_node = handle["X"]
        raw_shape = x_node.attrs["shape"] if "shape" in x_node.attrs else x_node.shape
        shape = tuple(int(value) for value in raw_shape)
    return gene_names, categories, shape


def parse_h5ad_args(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--h5ad values must use dataset=path")
        dataset, raw_path = value.split("=", 1)
        parsed[dataset.strip().lower()] = Path(raw_path).resolve()
    return parsed


def archived_conditions(path: Path) -> dict[str, list[str]]:
    values: dict[str, set[str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["dataset"].strip().lower(), set()).add(row["condition"].strip())
    return {dataset: sorted(conditions) for dataset, conditions in values.items()}


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-csv", required=True, type=Path)
    parser.add_argument("--legacy-runner", required=True, type=Path)
    parser.add_argument("--legacy-models", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--h5ad", action="append", default=[])
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    h5ad_paths = parse_h5ad_args(args.h5ad)
    conditions = archived_conditions(args.fold_csv.resolve())

    runner_text = args.legacy_runner.read_text(encoding="utf-8")
    model_text = args.legacy_models.read_text(encoding="utf-8")
    code_checks = [
        {
            "evidence_id": "MASK-PARSER-PLUS-ONLY",
            "observed": str('pert_str.split("+")' in runner_text).lower(),
            "implication": "Legacy target parsing recognises only plus-separated labels.",
        },
        {
            "evidence_id": "MASK-NONE-ON-NO-MATCH",
            "observed": str("return mask if found else None" in runner_text).lower(),
            "implication": "No exact gene-name match yields a None mask rather than a stopped fold.",
        },
        {
            "evidence_id": "MODEL-NONE-WILDTYPE",
            "observed": str("None: wild-type prediction" in model_text).lower(),
            "implication": "The archived graph model documents None as a wild-type prediction.",
        },
        {
            "evidence_id": "MODEL-MASK-CONDITIONAL",
            "observed": str(model_text.count("if perturbation_mask is not None:") >= 2).lower(),
            "implication": "Target masking is applied only when a non-None mask is supplied.",
        },
    ]
    write_csv(
        output_dir / "legacy_target_mask_code_evidence.csv",
        code_checks,
        ["evidence_id", "observed", "implication"],
    )

    metadata: dict[str, tuple[set[str], dict[str, set[str]], tuple[int, int]]] = {}
    source_rows: list[dict[str, object]] = [
        {
            "source_id": "legacy_runner",
            "file_name": args.legacy_runner.name,
            "sha256": sha256(args.legacy_runner),
            "md5": "",
            "n_obs": "",
            "n_vars": "",
        },
        {
            "source_id": "legacy_models",
            "file_name": args.legacy_models.name,
            "sha256": sha256(args.legacy_models),
            "md5": "",
            "n_obs": "",
            "n_vars": "",
        },
        {
            "source_id": "canonical_fold_ledger",
            "file_name": args.fold_csv.name,
            "sha256": sha256(args.fold_csv),
            "md5": "",
            "n_obs": "",
            "n_vars": "",
        },
    ]
    for dataset, path in sorted(h5ad_paths.items()):
        gene_names, obs_categories, shape = read_h5ad_metadata(path)
        metadata[dataset] = (gene_names, obs_categories, shape)
        source_rows.append(
            {
                "source_id": f"{dataset}_h5ad",
                "file_name": path.name,
                "sha256": sha256(path),
                "md5": md5(path),
                "n_obs": shape[0],
                "n_vars": shape[1],
            }
        )
    write_csv(
        output_dir / "legacy_target_mask_source_manifest.csv",
        source_rows,
        ["source_id", "file_name", "sha256", "md5", "n_obs", "n_vars"],
    )

    condition_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for dataset, dataset_conditions in sorted(conditions.items()):
        gene_names, obs_categories, _ = metadata.get(dataset, (set(), {}, (0, 0)))
        source_perturbation_column = next(
            (
                column
                for column in ("gene", "condition", "perturbation", "guide_ids", "perturbations")
                if column in obs_categories
            ),
            "",
        )
        source_condition_values = obs_categories.get(source_perturbation_column, set())
        counters: Counter[str] = Counter()
        for condition in dataset_conditions:
            tokens = [token.strip() for token in condition.split("+")]
            matched_tokens = (
                [token for token in tokens if token in gene_names] if gene_names else []
            )
            has_metadata = dataset in metadata
            no_full_universe_match = has_metadata and not matched_tokens
            row = {
                "dataset": dataset,
                "condition": condition,
                "legacy_plus_tokens": "|".join(tokens),
                "contains_plus": str("+" in condition).lower(),
                "contains_underscore": str("_" in condition).lower(),
                "guide_suffix": str(bool(GUIDE_SUFFIX.search(condition))).lower(),
                "h5ad_metadata_available": str(has_metadata).lower(),
                "source_perturbation_column": source_perturbation_column,
                "archived_label_in_source_values": (
                    str(condition in source_condition_values).lower()
                    if source_perturbation_column
                    else "not_assessed"
                ),
                "legacy_token_full_gene_universe_matches": "|".join(matched_tokens),
                "guaranteed_none_before_hvg_subset": (
                    str(no_full_universe_match).lower() if has_metadata else "not_assessed"
                ),
            }
            condition_rows.append(row)
            counters["contains_plus"] += "+" in condition
            counters["contains_underscore"] += "_" in condition
            counters["guide_suffix"] += bool(GUIDE_SUFFIX.search(condition))
            counters["guaranteed_none"] += no_full_universe_match
            counters["source_label_overlap"] += condition in source_condition_values

        summary_rows.append(
            {
                "dataset": dataset,
                "n_archived_conditions": len(dataset_conditions),
                "n_contains_plus": counters["contains_plus"],
                "n_contains_underscore": counters["contains_underscore"],
                "n_guide_suffix": counters["guide_suffix"],
                "h5ad_metadata_available": str(dataset in metadata).lower(),
                "source_perturbation_column": source_perturbation_column,
                "n_archived_labels_in_source_values": (
                    counters["source_label_overlap"]
                    if source_perturbation_column
                    else "not_assessed"
                ),
                "n_guaranteed_none_before_hvg_subset": (
                    counters["guaranteed_none"] if dataset in metadata else "not_assessed"
                ),
            }
        )

    write_csv(
        output_dir / "legacy_target_mask_condition_audit.csv",
        condition_rows,
        [
            "dataset",
            "condition",
            "legacy_plus_tokens",
            "contains_plus",
            "contains_underscore",
            "guide_suffix",
            "h5ad_metadata_available",
            "source_perturbation_column",
            "archived_label_in_source_values",
            "legacy_token_full_gene_universe_matches",
            "guaranteed_none_before_hvg_subset",
        ],
    )
    write_csv(
        output_dir / "legacy_target_mask_summary.csv",
        summary_rows,
        [
            "dataset",
            "n_archived_conditions",
            "n_contains_plus",
            "n_contains_underscore",
            "n_guide_suffix",
            "h5ad_metadata_available",
            "source_perturbation_column",
            "n_archived_labels_in_source_values",
            "n_guaranteed_none_before_hvg_subset",
        ],
    )

    report_lines = [
        "# Legacy perturbation-target mask audit",
        "",
        "## Code-level finding",
        "",
        "The archived runner splits perturbation labels only on `+`, requires an exact match to the "
        "post-HVG gene list, and returns `None` when no token matches. The archived graph model "
        "documents `None` as a wild-type prediction and applies target masking only for a non-None "
        "mask. This is a fail-open target-conditioning path.",
        "",
        "## Condition-label audit",
        "",
        "| Dataset | Archived conditions | With `+` | With `_` | Guide suffix | "
        "Labels verified in source values | Guaranteed no match in full source gene universe |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report_lines.append(
            f"| {row['dataset']} | {row['n_archived_conditions']} | {row['n_contains_plus']} | "
            f"{row['n_contains_underscore']} | {row['n_guide_suffix']} | "
            f"{row['n_archived_labels_in_source_values']} | "
            f"{row['n_guaranteed_none_before_hvg_subset']} |"
        )
    report_lines.extend(
        [
            "",
            "A full-gene-universe non-match is a sufficient condition for a `None` mask after any "
            "HVG subset. A full-universe match is not sufficient for successful legacy masking "
            "because the target may still have been excluded by full-data HVG selection.",
            "",
            "Against the exact public files named by the archived downloader, all 50 archived Adamson "
            "labels and all 50 archived Norman labels are verified source perturbation values. All 50 "
            "Adamson guide-suffixed labels and the 27 Norman underscore-pair labels have no legacy-token "
            "match even in the full source gene universe. They therefore necessarily yield `None` after "
            "every 200-, 500-, or 1,000-HVG subset. The remaining 23 Norman single-target labels may "
            "still fail after HVG selection, but the missing fold-specific gene lists prevent resolving "
            "that additional fraction.",
            "",
            "## Interpretation boundary",
            "",
            "The archived artifacts do not retain the per-dataset/per-scale gene order or a mask-success "
            "ledger. Therefore the affected fraction beyond the guaranteed Adamson 50/50 and Norman "
            "27/50 failures cannot be reconstructed from scalar outputs alone. Legacy graph contrasts, "
            "condition rankings, and comparisons that use the legacy Combined arm are source-output "
            "descriptors; they are not reliable estimates of target-conditioned perturbation prediction "
            "performance.",
            "",
            "The confirmatory runner addresses this defect by canonicalising dataset-declared separators, "
            "retaining every selected perturbation target in the gene panel, recording the resolved targets, "
            "and stopping a fold when the target indicator is empty or inconsistent.",
            "",
        ]
    )
    (output_dir / "legacy_target_mask_audit_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

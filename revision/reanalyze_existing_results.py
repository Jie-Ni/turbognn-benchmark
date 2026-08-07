#!/usr/bin/env python
"""Audit and reanalyse the saved CBAC benchmark outputs without running models.

Only seed-explicit chunk files are accepted. Legacy aggregate files are deliberately
excluded because they mix duplicated generic chunks and omit some seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CHUNK_RE = re.compile(
    r"^(?P<dataset>.+?)__(?P<graph>.+?)__s(?P<seed>\d+)f(?P<start>\d+)-(?P<end>\d+)\.json$"
)
DATASETS = ("adamson", "norman", "replogle_k562", "replogle_rpe1")
SCALES = (200, 500, 1000)
SEEDS = (42, 43, 44)
GRAPHS = ("string_ppi", "gene_ontology", "coexpression", "combined", "random", "no_graph")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_chunks(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    files: list[dict[str, object]] = []
    for path in sorted(results_dir.glob("hvg*/*.json")):
        match = CHUNK_RE.match(path.name)
        if not match:
            continue
        hvg = int(path.parent.name.removeprefix("hvg"))
        meta = match.groupdict()
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_id = (Path("results_merged_3seed") / path.relative_to(results_dir)).as_posix()
        seed = int(meta["seed"])
        seed_key = f"seed_{seed}"
        folds = payload.get("seeds", {}).get(seed_key, {}).get("folds", [])
        files.append(
            {
                "path": source_id,
                "sha256": sha256(path),
                "dataset": meta["dataset"],
                "hvg": hvg,
                "graph": meta["graph"],
                "seed": seed,
                "chunk_start": int(meta["start"]),
                "chunk_end": int(meta["end"]),
                "n_rows": len(folds),
            }
        )
        for fold in folds:
            rows.append(
                {
                    "dataset": meta["dataset"],
                    "hvg": hvg,
                    "graph": meta["graph"],
                    "seed": seed,
                    "condition": str(fold.get("condition", "")),
                    "pearson_r": float(fold["pearson_r"]),
                    "spearman_rho": float(fold["spearman_rho"]),
                    "mse": float(fold["mse"]),
                    "jaccard": float(fold["jaccard"]),
                    "source_file": source_id,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(files)


def deduplicate(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["dataset", "hvg", "graph", "seed", "condition"]
    metrics = ["pearson_r", "spearman_rho", "mse", "jaccard"]
    audit_rows: list[dict[str, object]] = []
    for key, group in raw.groupby(keys, dropna=False):
        spreads = {metric: float(group[metric].max() - group[metric].min()) for metric in metrics}
        audit_rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "copies": len(group),
                "max_metric_spread": max(spreads.values()),
                "identical": max(spreads.values()) <= 1e-12,
            }
        )
    audit = pd.DataFrame(audit_rows)
    conflicts = audit[(audit["copies"] > 1) & ~audit["identical"]]
    if not conflicts.empty:
        raise ValueError(f"Found {len(conflicts)} conflicting duplicate condition records")
    clean = raw.sort_values(keys + ["source_file"]).drop_duplicates(keys, keep="first")
    return clean.reset_index(drop=True), audit


def configuration_manifest(clean: pd.DataFrame) -> pd.DataFrame:
    observed = clean.groupby(["dataset", "hvg", "graph", "seed"], as_index=False).agg(
        n_conditions=("condition", "nunique"), n_rows=("condition", "size")
    )
    grid = pd.MultiIndex.from_product(
        [DATASETS, SCALES, GRAPHS, SEEDS], names=["dataset", "hvg", "graph", "seed"]
    ).to_frame(index=False)
    out = grid.merge(observed, how="left", on=["dataset", "hvg", "graph", "seed"])
    out[["n_conditions", "n_rows"]] = out[["n_conditions", "n_rows"]].fillna(0).astype(int)
    out["structurally_expected"] = ~((out["graph"] == "coexpression") & (out["hvg"] > 200))
    out["present"] = out["n_rows"] > 0
    return out


def condition_means(clean: pd.DataFrame) -> pd.DataFrame:
    return clean.groupby(["dataset", "hvg", "graph", "condition"], as_index=False).agg(
        pearson_r=("pearson_r", "mean"),
        pearson_seed_sd=("pearson_r", "std"),
        n_seeds=("seed", "nunique"),
        spearman_rho=("spearman_rho", "mean"),
        mse=("mse", "mean"),
        jaccard=("jaccard", "mean"),
    )


def bootstrap_mean_ci(values: np.ndarray, seed: int, n_boot: int = 20_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def paired_contrasts(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons = [
        ("combined", "no_graph"),
        ("string_ppi", "no_graph"),
        ("gene_ontology", "no_graph"),
        ("random", "no_graph"),
        ("combined", "random"),
        ("string_ppi", "random"),
        ("gene_ontology", "random"),
    ]
    detail_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for hvg in SCALES:
            subset = clean[(clean["dataset"] == dataset) & (clean["hvg"] == hvg)]
            for left, right in comparisons:
                a = subset[subset["graph"] == left][["condition", "seed", "pearson_r"]]
                b = subset[subset["graph"] == right][["condition", "seed", "pearson_r"]]
                paired_seed = a.merge(b, on=["condition", "seed"], suffixes=("_left", "_right"))
                if paired_seed.empty:
                    continue
                paired_seed["delta_r"] = (
                    paired_seed["pearson_r_left"] - paired_seed["pearson_r_right"]
                )
                paired = paired_seed.groupby("condition", as_index=False).agg(
                    pearson_r_left=("pearson_r_left", "mean"),
                    pearson_r_right=("pearson_r_right", "mean"),
                    delta_r=("delta_r", "mean"),
                    n_common_seeds=("seed", "nunique"),
                )
                paired["dataset"] = dataset
                paired["hvg"] = hvg
                paired["contrast"] = f"{left} - {right}"
                detail_frames.append(paired)
                values = paired["delta_r"].to_numpy()
                ci_low, ci_high = bootstrap_mean_ci(
                    values,
                    seed=10_000
                    + hvg
                    + DATASETS.index(dataset) * 100
                    + comparisons.index((left, right)),
                )
                t_stat, t_p = stats.ttest_1samp(values, 0.0)
                try:
                    w_stat, w_p = stats.wilcoxon(
                        values, zero_method="wilcox", alternative="two-sided"
                    )
                except ValueError:
                    w_stat, w_p = np.nan, np.nan
                sd = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "hvg": hvg,
                        "contrast": f"{left} - {right}",
                        "n_conditions": len(values),
                        "n_complete_3seed": int((paired["n_common_seeds"] == 3).sum()),
                        "min_common_seeds": int(paired["n_common_seeds"].min()),
                        "median_common_seeds": float(paired["n_common_seeds"].median()),
                        "mean_delta_r": float(np.mean(values)),
                        "median_delta_r": float(np.median(values)),
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "paired_cohens_d": (
                            float(np.mean(values) / sd) if sd and np.isfinite(sd) else np.nan
                        ),
                        "t_stat": float(t_stat),
                        "t_p": float(t_p),
                        "wilcoxon_stat": float(w_stat),
                        "wilcoxon_p": float(w_p),
                        "fraction_positive": float(np.mean(values > 0)),
                    }
                )
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        family = summary["contrast"].isin(
            ["combined - no_graph", "string_ppi - no_graph", "gene_ontology - no_graph"]
        )
        pvals = summary.loc[family, "wilcoxon_p"].to_numpy()
        order = np.argsort(pvals)
        adjusted = np.empty_like(pvals)
        running = 1.0
        for rank_from_end, index in enumerate(order[::-1], start=1):
            rank = len(pvals) - rank_from_end + 1
            running = min(running, pvals[index] * len(pvals) / rank)
            adjusted[index] = running
        summary["wilcoxon_bh_primary_family"] = np.nan
        summary.loc[family, "wilcoxon_bh_primary_family"] = adjusted
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return summary, details


def global_hierarchical_bootstrap(details: pd.DataFrame) -> pd.DataFrame:
    target = details[details["contrast"] == "combined - no_graph"].copy()
    rows: list[dict[str, object]] = []
    for hvg in SCALES:
        subset = target[target["hvg"] == hvg]
        strata = {
            dataset: group["delta_r"].to_numpy() for dataset, group in subset.groupby("dataset")
        }
        if not strata:
            continue
        rng = np.random.default_rng(31_000 + hvg)
        boot = np.empty(20_000)
        for index in range(len(boot)):
            dataset_means = [
                rng.choice(values, len(values), replace=True).mean() for values in strata.values()
            ]
            boot[index] = np.mean(dataset_means)
        observed = float(np.mean([values.mean() for values in strata.values()]))
        rows.append(
            {
                "hvg": hvg,
                "estimand": "equal-weight dataset-stratified mean condition-level delta_r",
                "n_datasets": len(strata),
                "n_conditions": int(sum(len(values) for values in strata.values())),
                "estimate": observed,
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
                "probability_delta_gt_0": float(np.mean(boot > 0)),
                "probability_delta_gt_0_01": float(np.mean(boot > 0.01)),
            }
        )
    return pd.DataFrame(rows)


def descriptive_summary(means: pd.DataFrame) -> pd.DataFrame:
    return means.groupby(["dataset", "hvg", "graph"], as_index=False).agg(
        mean_pearson=("pearson_r", "mean"),
        sd_across_conditions=("pearson_r", "std"),
        median_pearson=("pearson_r", "median"),
        n_conditions=("condition", "nunique"),
        median_seeds_per_condition=("n_seeds", "median"),
    )


def top_condition_identity(clean: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for hvg in SCALES:
            subset = clean[(clean["dataset"] == dataset) & (clean["hvg"] == hvg)]
            combined = subset[subset["graph"] == "combined"][["condition", "seed", "pearson_r"]]
            for graph in GRAPHS:
                if graph == "combined":
                    continue
                comparator = subset[subset["graph"] == graph][["condition", "seed", "pearson_r"]]
                shared = combined.merge(
                    comparator, on=["condition", "seed"], suffixes=("_combined", "_comparator")
                )
                if shared.empty:
                    continue
                ranked = shared.groupby("condition", as_index=False).agg(
                    combined=("pearson_r_combined", "mean"),
                    comparator=("pearson_r_comparator", "mean"),
                    n_common_seeds=("seed", "nunique"),
                )
                reference = ranked.nlargest(10, "combined")["condition"].tolist()
                ranking = ranked.nlargest(10, "comparator")["condition"].tolist()
                intersection = set(reference) & set(ranking)
                rows.append(
                    {
                        "dataset": dataset,
                        "hvg": hvg,
                        "comparison": f"combined vs {graph}",
                        "top10_overlap": len(intersection),
                        "top10_jaccard": len(intersection) / len(set(reference) | set(ranking)),
                        "min_common_seeds": int(ranked["n_common_seeds"].min()),
                        "shared_conditions": ";".join(sorted(intersection)),
                    }
                )
    return pd.DataFrame(rows)


def write_integrity_report(
    output_dir: Path,
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    file_manifest: pd.DataFrame,
    duplicate_audit: pd.DataFrame,
    config: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> None:
    duplicate_groups = duplicate_audit[duplicate_audit["copies"] > 1]
    incomplete = config[
        config["structurally_expected"] & ((~config["present"]) | (config["n_conditions"] < 50))
    ]
    primary = contrasts[contrasts["contrast"] == "combined - no_graph"]
    lines = [
        "# Existing-results integrity and reanalysis report",
        "",
        "This audit is read-only with respect to model outputs. It excludes every aggregate or seed-implicit JSON file and accepts only filenames that explicitly encode seed and fold range.",
        "",
        "## Ledger",
        "",
        f"- Accepted seed-explicit chunk files: {len(file_manifest):,}",
        f"- Parsed fold records before deduplication: {len(raw):,}",
        f"- Unique dataset/HVG/graph/seed/condition records: {len(clean):,}",
        f"- Duplicate condition groups among accepted chunks: {len(duplicate_groups):,}",
        f"- Conflicting duplicate groups: {int(((duplicate_groups['identical']) == False).sum()) if not duplicate_groups.empty else 0:,}",
        f"- Expected configuration-seed cells with fewer than 50 unique conditions: {len(incomplete):,}",
        "",
        "## Consequence for the submitted analysis",
        "",
        "The submitted aggregate lineage is not safe for inferential claims: seed-implicit chunks duplicate seed 42, some aggregate files omit seed 43, and positional truncation was used instead of condition-keyed pairing. The repaired tables use condition-keyed pairing and average available seeds within condition. This repairs accounting only; it does not make the saved values target-conditioned or inferentially valid.",
        "",
        "## Accounting-repaired Combined minus no-graph source-output differences",
        "",
        "| Dataset | HVGs | n condition keys | complete 3-seed pairs | mean source-output difference | 95% resampling interval |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in primary.iterrows():
        lines.append(
            f"| {row['dataset']} | {int(row['hvg'])} | {int(row['n_conditions'])} | "
            f"{int(row['n_complete_3seed'])} | {row['mean_delta_r']:+.4f} | "
            f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Hard interpretation boundary",
            "",
            "These statistics repair key-based pairing and run accounting only. They are source-output descriptors, not target-conditioned prediction estimates: the archived plus-only parser necessarily returned an empty target mask for Adamson 50/50 and Norman at least 27/50 labels, while the additional affected fraction is not reconstructable. The saved no-graph model is also a Transformer whereas the graph models are GATs, and preprocessing is transductive. A fail-closed, same-GAT rerun with prediction-level artifacts is required for the confirmatory claim.",
            "",
        ]
    )
    (output_dir / "existing_results_integrity_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw, files = load_chunks(args.results_dir)
    clean, duplicate_audit = deduplicate(raw)
    config = configuration_manifest(clean)
    means = condition_means(clean)
    contrasts, details = paired_contrasts(clean)
    global_bootstrap = global_hierarchical_bootstrap(details)
    descriptive = descriptive_summary(means)
    identity = top_condition_identity(clean)

    tables = {
        "accepted_file_manifest.csv": files,
        "duplicate_audit.csv": duplicate_audit,
        "canonical_fold_level.csv": clean,
        "configuration_manifest.csv": config,
        "condition_seed_averages.csv": means,
        "descriptive_summary.csv": descriptive,
        "paired_condition_contrasts.csv": contrasts,
        "paired_condition_details.csv": details,
        "hierarchical_bootstrap_combined_vs_no_graph.csv": global_bootstrap,
        "top_condition_identity.csv": identity,
    }
    for name, frame in tables.items():
        frame.to_csv(args.output_dir / name, index=False)

    write_integrity_report(args.output_dir, raw, clean, files, duplicate_audit, config, contrasts)
    print(json.dumps({name: len(frame) for name, frame in tables.items()}, indent=2))


if __name__ == "__main__":
    main()

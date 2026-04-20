#!/usr/bin/env python
"""Merge fold-split result files into unified per-(dataset, graph) results.

Scans for files like: norman__string_ppi__f0-10.json, norman__string_ppi__f10-20.json, ...
Merges their folds into: norman__string_ppi.json
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np


def merge_results_dir(results_dir: Path) -> None:
    """Merge all split results in one directory."""
    results_dir = Path(results_dir)
    # Find all split files: {dataset}__{graph}__f{start}-{end}.json
    split_files = sorted(results_dir.glob("*__f*.json"))
    if not split_files:
        print(f"  No split files in {results_dir}")
        return

    # Group by (dataset, graph_type)
    groups = defaultdict(list)
    for f in split_files:
        # Parse: dataset__graph_type__fX-Y.json
        name = f.stem  # e.g., "replogle_k562__string_ppi__f0-10"
        parts = name.split("__")
        if len(parts) >= 3 and parts[-1].startswith("f"):
            key = "__".join(parts[:-1])  # "replogle_k562__string_ppi"
            groups[key].append(f)

    for key, files in sorted(groups.items()):
        print(f"  Merging {key}: {len(files)} chunks")
        merged_seeds = {}

        for f in sorted(files):
            with open(f) as fh:
                data = json.load(fh)

            for seed_key, seed_data in data.get("seeds", {}).items():
                if seed_key not in merged_seeds:
                    merged_seeds[seed_key] = {"folds": [], "n_folds": 0}
                merged_seeds[seed_key]["folds"].extend(seed_data.get("folds", []))
                merged_seeds[seed_key]["n_folds"] += seed_data.get("n_folds", 0)

        # Recompute aggregates
        all_p, all_s, all_j, all_m = [], [], [], []
        final_seeds = {}
        for seed_key, sd in merged_seeds.items():
            folds = sd["folds"]
            ps = [r["pearson_r"] for r in folds]
            ss = [r["spearman_rho"] for r in folds]
            js = [r["jaccard"] for r in folds]
            ms = [r["mse"] for r in folds]
            all_p.extend(ps)
            all_s.extend(ss)
            all_j.extend(js)
            all_m.extend(ms)
            final_seeds[seed_key] = {
                "pearson_mean": float(np.mean(ps)),
                "pearson_std": float(np.std(ps)),
                "spearman_mean": float(np.mean(ss)),
                "jaccard_mean": float(np.mean(js)),
                "mse_mean": float(np.mean(ms)),
                "n_folds": len(folds),
                "folds": folds,
            }

        parts = key.split("__")
        payload = {
            "dataset": parts[0],
            "graph_type": parts[1] if len(parts) > 1 else "unknown",
            "overall": {
                "pearson_mean": float(np.mean(all_p)),
                "pearson_std": float(np.std(all_p)),
                "spearman_mean": float(np.mean(all_s)),
                "spearman_std": float(np.std(all_s)),
                "jaccard_mean": float(np.mean(all_j)),
                "jaccard_std": float(np.std(all_j)),
                "mse_mean": float(np.mean(all_m)),
                "mse_std": float(np.std(all_m)),
            },
            "seeds": final_seeds,
        }

        out_path = results_dir / f"{key}.json"
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        total_folds = sum(len(sd["folds"]) for sd in final_seeds.values())
        print(f"    -> {out_path.name} ({total_folds} total folds, Pearson={payload['overall']['pearson_mean']:.4f})")


if __name__ == "__main__":
    dirs = sys.argv[1:] if len(sys.argv) > 1 else [
        "results_benchmark/hvg200",
        "results_benchmark/hvg500",
        "results_benchmark/hvg1000",
    ]
    for d in dirs:
        p = Path(d)
        if p.exists():
            print(f"Processing {p}:")
            merge_results_dir(p)
        else:
            print(f"Skipping {p} (not found)")

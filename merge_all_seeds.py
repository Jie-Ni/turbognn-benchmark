#!/usr/bin/env python
"""
Final merge: combine old single-seed merged JSONs + new seed split files.

Strategy:
1. For each (dataset, graph, hvg): read the existing merged JSON (has seed_XX with 50 folds)
2. Read all new __fX-Y.json splits (each has a different seed_XX)
3. Combine all seeds into one unified JSON with 3 seeds x 50 folds = 150 obs
4. Recompute overall statistics
"""
import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict


def merge_directory(results_dir: Path) -> None:
    """Merge all data in one hvg directory."""
    # Step 1: Read existing merged JSONs (from round 1)
    existing = {}  # key -> {seed_key: {folds: [...], ...}}
    for jf in sorted(results_dir.glob("*.json")):
        if "__f" in jf.name or jf.name.startswith("results_") or jf.name == "summary.csv":
            continue
        with open(jf) as f:
            data = json.load(f)
        key = f"{data['dataset']}__{data['graph_type']}"
        existing[key] = {
            "dataset": data["dataset"],
            "graph_type": data["graph_type"],
            "seeds": data.get("seeds", {}),
        }

    # Step 2: Read all split files (both old __fX-Y and new __sNfX-Y formats)
    splits = defaultdict(lambda: defaultdict(list))  # key -> seed_key -> [folds]
    for jf in sorted(results_dir.glob("*__*f*-*.json")):
        name = jf.stem
        parts = name.split("__")
        # Formats: "dataset__graph__f0-10" or "dataset__graph__s42f0-10"
        tag = parts[-1]
        if not (tag.startswith("f") or (tag.startswith("s") and "f" in tag)):
            continue
        key = "__".join(parts[:-1])  # "dataset__graph"
        with open(jf) as f:
            data = json.load(f)
        for seed_key, seed_data in data.get("seeds", {}).items():
            splits[key][seed_key].extend(seed_data.get("folds", []))

    # Step 3: Merge
    all_keys = set(existing.keys()) | set(splits.keys())
    for key in sorted(all_keys):
        merged_seeds = {}

        # From existing merged JSON
        if key in existing:
            for seed_key, seed_data in existing[key]["seeds"].items():
                merged_seeds[seed_key] = seed_data.get("folds", [])

        # From split files (may add new seeds or extend existing ones)
        if key in splits:
            for seed_key, folds in splits[key].items():
                if seed_key in merged_seeds:
                    # Check for duplicates by condition name
                    existing_conds = {f["condition"] for f in merged_seeds[seed_key]}
                    for fold in folds:
                        if fold["condition"] not in existing_conds:
                            merged_seeds[seed_key].append(fold)
                            existing_conds.add(fold["condition"])
                else:
                    merged_seeds[seed_key] = folds

        if not merged_seeds:
            continue

        # Recompute aggregates
        all_p, all_s, all_j, all_m = [], [], [], []
        final_seeds = {}
        for seed_key in sorted(merged_seeds.keys()):
            folds = merged_seeds[seed_key]
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
        ds = parts[0] if len(parts) >= 1 else "unknown"
        gt = parts[1] if len(parts) >= 2 else "unknown"

        payload = {
            "dataset": ds,
            "graph_type": gt,
            "overall": {
                "pearson_mean": float(np.mean(all_p)),
                "pearson_std": float(np.std(all_p)),
                "spearman_mean": float(np.mean(all_s)),
                "spearman_std": float(np.std(all_s)),
                "jaccard_mean": float(np.mean(all_j)),
                "jaccard_std": float(np.std(all_j)),
                "mse_mean": float(np.mean(all_m)),
                "mse_std": float(np.std(all_m)),
                "n_total_folds": len(all_p),
                "n_seeds": len(final_seeds),
            },
            "seeds": final_seeds,
        }

        out_path = results_dir / f"{key}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

        n_seeds = len(final_seeds)
        n_folds = len(all_p)
        print(f"  {key}: {n_seeds} seeds, {n_folds} total folds, Pearson={payload['overall']['pearson_mean']:.4f}")


if __name__ == "__main__":
    dirs = sys.argv[1:] if len(sys.argv) > 1 else [
        "results_benchmark/hvg200",
        "results_benchmark/hvg500",
        "results_benchmark/hvg1000",
    ]
    for d in dirs:
        p = Path(d)
        if p.exists():
            print(f"\nMerging {p}:")
            merge_directory(p)
        else:
            print(f"Skipping {p}")

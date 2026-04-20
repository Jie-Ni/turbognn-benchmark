#!/usr/bin/env python
"""
Generate tasks for ALL missing seed-fold chunks.
Reads current merged JSONs to find which (dataset, graph, hvg, seed) combos
have <50 folds, then generates 10-fold split tasks for the missing ones.
"""
import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent.parent / "results_final"
FOLDS_PER_CHUNK = 10
TOTAL_FOLDS = 50

ALL_GRAPHS_200 = ["string_ppi", "gene_ontology", "coexpression", "combined", "random", "no_graph"]
KEY_GRAPHS = ["string_ppi", "gene_ontology", "combined", "random", "no_graph"]
DATASETS = ["adamson", "norman", "replogle_k562", "replogle_rpe1"]
SEEDS = [42, 43, 44]

# Build the full target set
target_configs = []
for ds in DATASETS:
    for hvg in [200]:
        for g in ALL_GRAPHS_200:
            for seed in SEEDS:
                target_configs.append((ds, g, hvg, seed))
    for hvg in [500, 1000]:
        for g in KEY_GRAPHS:
            for seed in SEEDS:
                target_configs.append((ds, g, hvg, seed))

print(f"Target: {len(target_configs)} seed-configs, each needing 50 folds")

# Check what we have
have = defaultdict(lambda: defaultdict(int))  # (ds, g, hvg) -> seed -> n_folds
for hvg in [200, 500, 1000]:
    d = RESULTS_DIR / f"hvg{hvg}"
    if not d.exists():
        continue
    for jf in sorted(d.glob("*.json")):
        if "__f" in jf.name or jf.name.startswith("results_") or jf.name == "summary.csv":
            continue
        with open(jf) as f:
            data = json.load(f)
        ds, gt = data["dataset"], data["graph_type"]
        for seed_key, seed_data in data.get("seeds", {}).items():
            seed_val = int(seed_key.replace("seed_", ""))
            n_folds = len(seed_data.get("folds", []))
            have[(ds, gt, hvg)][seed_val] = max(have[(ds, gt, hvg)][seed_val], n_folds)

# Find what's missing
missing_tasks = []
for ds, g, hvg, seed in target_configs:
    current_folds = have[(ds, g, hvg)].get(seed, 0)
    if current_folds >= 50:
        continue
    # Need all 5 chunks (fresh run, since old data may be corrupted by overwrites)
    for fs in range(0, TOTAL_FOLDS, FOLDS_PER_CHUNK):
        fe = min(fs + FOLDS_PER_CHUNK, TOTAL_FOLDS)
        missing_tasks.append(f"{ds} {g} {hvg} {seed} {fs} {fe}")

out_dir = Path(__file__).parent

# Split for two servers
mid = len(missing_tasks) // 2
with open(out_dir / "tasks_fix_inn.txt", "w") as f:
    f.write("\n".join(missing_tasks[:mid]) + "\n")
with open(out_dir / "tasks_fix_lnz.txt", "w") as f:
    f.write("\n".join(missing_tasks[mid:]) + "\n")
with open(out_dir / "tasks_fix_all.txt", "w") as f:
    f.write("\n".join(missing_tasks) + "\n")

n_inn = (mid + 3) // 4
n_lnz = (len(missing_tasks) - mid + 3) // 4

print(f"Missing seed-configs: {len(missing_tasks) // 5}")
print(f"Missing tasks (10-fold chunks): {len(missing_tasks)}")
print(f"  inn: {mid} tasks -> {n_inn} nodes")
print(f"  lnz: {len(missing_tasks) - mid} tasks -> {n_lnz} nodes")

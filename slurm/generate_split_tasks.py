#!/usr/bin/env python
"""Generate fold-split task files for Replogle datasets.

Each original task (50 folds) is split into 5 chunks of 10 folds.
Format: DATASET GRAPH_TYPE NUM_HVG SEED FOLD_START FOLD_END
"""
from pathlib import Path

REPLOGLE = ["replogle_k562", "replogle_rpe1"]
ALL_GRAPHS = ["string_ppi", "gene_ontology", "coexpression", "combined", "random", "no_graph"]
KEY_GRAPHS = ["string_ppi", "gene_ontology", "combined", "random", "no_graph"]
SEEDS = [42, 43, 44]
FOLDS_PER_CHUNK = 10
TOTAL_FOLDS = 50

out_dir = Path(__file__).parent
tasks = []

# Track A: HVG=200, all 6 graphs
for ds in REPLOGLE:
    for g in ALL_GRAPHS:
        for seed in SEEDS:
            for fs in range(0, TOTAL_FOLDS, FOLDS_PER_CHUNK):
                fe = min(fs + FOLDS_PER_CHUNK, TOTAL_FOLDS)
                tasks.append(f"{ds} {g} 200 {seed} {fs} {fe}")

# Track B: HVG=500/1000, 5 key graphs
for ds in REPLOGLE:
    for g in KEY_GRAPHS:
        for hvg in [500, 1000]:
            for seed in SEEDS:
                for fs in range(0, TOTAL_FOLDS, FOLDS_PER_CHUNK):
                    fe = min(fs + FOLDS_PER_CHUNK, TOTAL_FOLDS)
                    tasks.append(f"{ds} {g} {hvg} {seed} {fs} {fe}")

with open(out_dir / "tasks_replogle_split.txt", "w") as f:
    f.write("\n".join(tasks) + "\n")

nodes = (len(tasks) + 3) // 4
print(f"Total: {len(tasks)} sub-tasks -> tasks_replogle_split.txt")
print(f"Nodes needed: {nodes}")
print(f"Each sub-task: {FOLDS_PER_CHUNK} folds (vs 50 before) → ~5x faster per task")

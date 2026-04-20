#!/usr/bin/env python
"""Generate task files for SLURM job arrays.

Track A: Multi-seed (seeds 43, 44) for all 4 datasets x 6 graphs at HVG=200
  -> 4 x 6 x 2 = 48 tasks

Track B: HVG scaling (500, 1000) for 4 datasets x 4 key graphs x 3 seeds (42-44)
  -> 4 x 4 x 2 x 3 = 96 tasks

Output: tasks_track_a.txt, tasks_track_b.txt (one task per line)
Format: DATASET GRAPH_TYPE NUM_HVG SEED
"""
from pathlib import Path

DATASETS = ["norman", "adamson", "replogle_k562", "replogle_rpe1"]

ALL_GRAPHS = [
    "string_ppi", "gene_ontology", "coexpression",
    "combined", "random", "no_graph",
]

# Key graphs for HVG scaling ablation (must include random as structure-only control)
KEY_GRAPHS = ["string_ppi", "gene_ontology", "combined", "random", "no_graph"]

out_dir = Path(__file__).parent

# --- Track A: multi-seed at HVG=200 ---
# Run all 3 seeds (42, 43, 44) for full consistency with fresh data
track_a = []
for ds in DATASETS:
    for g in ALL_GRAPHS:
        for seed in [42, 43, 44]:
            track_a.append(f"{ds} {g} 200 {seed}")

with open(out_dir / "tasks_track_a.txt", "w") as f:
    f.write("\n".join(track_a) + "\n")
print(f"Track A: {len(track_a)} tasks -> tasks_track_a.txt")

# --- Track B: HVG scaling at 500 and 1000 ---
# All 3 seeds (42, 43, 44) for new HVG scales
track_b = []
for ds in DATASETS:
    for g in KEY_GRAPHS:
        for hvg in [500, 1000]:
            for seed in [42, 43, 44]:
                track_b.append(f"{ds} {g} {hvg} {seed}")

with open(out_dir / "tasks_track_b.txt", "w") as f:
    f.write("\n".join(track_b) + "\n")
print(f"Track B: {len(track_b)} tasks -> tasks_track_b.txt")

# --- Combined for convenience ---
all_tasks = track_a + track_b
with open(out_dir / "tasks_all.txt", "w") as f:
    f.write("\n".join(all_tasks) + "\n")
print(f"Total: {len(all_tasks)} tasks -> tasks_all.txt")

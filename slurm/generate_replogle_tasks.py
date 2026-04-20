#!/usr/bin/env python
"""Generate task files for Replogle-only experiments (long-running)."""
from pathlib import Path

REPLOGLE_DATASETS = ["replogle_k562", "replogle_rpe1"]

ALL_GRAPHS = [
    "string_ppi", "gene_ontology", "coexpression",
    "combined", "random", "no_graph",
]

KEY_GRAPHS = ["string_ppi", "gene_ontology", "combined", "random", "no_graph"]

out_dir = Path(__file__).parent

# Track A: Replogle at HVG=200, all graphs, 3 seeds
track_a_replogle = []
for ds in REPLOGLE_DATASETS:
    for g in ALL_GRAPHS:
        for seed in [42, 43, 44]:
            track_a_replogle.append(f"{ds} {g} 200 {seed}")

# Track B: Replogle at HVG=500/1000, key graphs, 3 seeds
track_b_replogle = []
for ds in REPLOGLE_DATASETS:
    for g in KEY_GRAPHS:
        for hvg in [500, 1000]:
            for seed in [42, 43, 44]:
                track_b_replogle.append(f"{ds} {g} {hvg} {seed}")

all_replogle = track_a_replogle + track_b_replogle
with open(out_dir / "tasks_replogle.txt", "w") as f:
    f.write("\n".join(all_replogle) + "\n")

print(f"Track A Replogle: {len(track_a_replogle)} tasks")
print(f"Track B Replogle: {len(track_b_replogle)} tasks")
print(f"Total Replogle: {len(all_replogle)} tasks -> tasks_replogle.txt")
print(f"Nodes needed: {(len(all_replogle) + 3) // 4}")

#!/usr/bin/env python
"""Generate fold-split tasks for all missing seed-configs."""
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results_final"
FOLDS_PER_CHUNK = 10
TOTAL_FOLDS = 50

# Find what's missing
missing = []
for hvg in [200, 500, 1000]:
    d = RESULTS_DIR / f"hvg{hvg}"
    for jf in sorted(d.glob("*.json")):
        if "__f" in jf.name or jf.name.startswith("results_") or jf.name == "summary.csv":
            continue
        with open(jf) as f:
            data = json.load(f)
        ds, gt = data["dataset"], data["graph_type"]
        seeds_present = set()
        for sk, sv in data.get("seeds", {}).items():
            seed_val = int(sk.replace("seed_", ""))
            nf = sv.get("n_folds", len(sv.get("folds", [])))
            if nf >= 50:
                seeds_present.add(seed_val)
        for need in [42, 43, 44]:
            if need not in seeds_present:
                missing.append((ds, gt, hvg, need))

# Generate split tasks
tasks = []
for ds, gt, hvg, seed in sorted(set(missing)):
    for fs in range(0, TOTAL_FOLDS, FOLDS_PER_CHUNK):
        fe = min(fs + FOLDS_PER_CHUNK, TOTAL_FOLDS)
        tasks.append(f"{ds} {gt} {hvg} {seed} {fs} {fe}")

out_dir = Path(__file__).parent

# Split roughly in half for two servers
mid = len(tasks) // 2
with open(out_dir / "tasks_missing_inn.txt", "w") as f:
    f.write("\n".join(tasks[:mid]) + "\n")
with open(out_dir / "tasks_missing_lnz.txt", "w") as f:
    f.write("\n".join(tasks[mid:]) + "\n")
with open(out_dir / "tasks_missing_all.txt", "w") as f:
    f.write("\n".join(tasks) + "\n")

print(f"Missing seed-configs: {len(set(missing))}")
print(f"Total split tasks: {len(tasks)}")
print(f"  inn: {mid} tasks -> {(mid+3)//4} nodes")
print(f"  lnz: {len(tasks)-mid} tasks -> {(len(tasks)-mid+3)//4} nodes")

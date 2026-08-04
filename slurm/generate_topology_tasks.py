#!/usr/bin/env python
"""Generate the frozen TDS-28 4x3x(5+5)x3 topology task matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turbognn_audit.graphs import load_ensemble_specs

DATASETS = ("norman", "adamson", "replogle_k562", "replogle_rpe1")
HVG_SCALES = (200, 500, 1000)
TRAINING_SEEDS = (42, 43, 44)


def generate_tasks(config: Path) -> list[str]:
    """Return 360 deterministic task rows with graph-instance seed in column five."""
    with config.open(encoding="utf-8") as handle:
        values = json.load(handle)
    specs = load_ensemble_specs(values["ensembles"])
    by_type = {spec.graph_type: spec for spec in specs}
    required = {"degree_preserving_rewired", "barabasi_albert"}
    if set(by_type) != required:
        raise ValueError(f"Topology config must contain exactly {sorted(required)}")
    if any(len(by_type[graph_type].instance_seeds) != 5 for graph_type in required):
        raise ValueError("TDS-28 requires exactly five frozen instances per topology type")
    rows = [
        f"{dataset} {graph_type} {hvg} {training_seed} {graph_seed}"
        for dataset in DATASETS
        for hvg in HVG_SCALES
        for graph_type in ("degree_preserving_rewired", "barabasi_albert")
        for graph_seed in by_type[graph_type].instance_seeds
        for training_seed in TRAINING_SEEDS
    ]
    if len(rows) != 360 or len(set(rows)) != 360:
        raise RuntimeError("Topology task conservation failed")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "graph_ensembles.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = generate_tasks(args.config)
    args.output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} frozen topology tasks to {args.output}")


if __name__ == "__main__":
    main()

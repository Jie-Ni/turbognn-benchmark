from __future__ import annotations

from pathlib import Path

from slurm.generate_topology_tasks import generate_tasks


def test_tds28_task_matrix_has_360_unique_fully_crossed_rows() -> None:
    config = Path(__file__).parents[1] / "config" / "graph_ensembles.json"
    rows = generate_tasks(config)
    assert len(rows) == 360
    assert len(set(rows)) == 360
    assert all(len(row.split()) == 5 for row in rows)
    assert {int(row.split()[4]) for row in rows} == {
        101,
        102,
        103,
        104,
        105,
        201,
        202,
        203,
        204,
        205,
    }

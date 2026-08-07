from __future__ import annotations

import numpy as np

import cbac_revision.graph_supports as graph_supports
from cbac_revision.graph_supports import GraphMode, build_graph_support


def test_dense_and_self_loop_supports_have_exact_expected_edges() -> None:
    dense = build_graph_support(GraphMode.DENSE, 4, include_self_loops=True)
    self_loop = build_graph_support(GraphMode.SELF_LOOP, 4, include_self_loops=True)

    assert dense.edge_index.shape == (2, 16)
    assert dense.undirected_degree_sequence == (3, 3, 3, 3)
    assert self_loop.edge_index.shape == (2, 4)
    assert self_loop.undirected_degree_sequence == (0, 0, 0, 0)


def test_dense_1000_and_self_loop_use_analytic_topology_diagnostics(monkeypatch) -> None:
    def fail_networkx(*args, **kwargs):
        raise AssertionError("analytic support diagnostics must not construct a NetworkX graph")

    monkeypatch.setattr(graph_supports.nx, "Graph", fail_networkx)
    dense = build_graph_support(GraphMode.DENSE, 1_000, include_self_loops=True)
    self_loop = build_graph_support(GraphMode.SELF_LOOP, 1_000, include_self_loops=True)

    assert dense.n_undirected_nonself_edges == 499_500
    assert dense.mean_clustering_coefficient == 1.0
    assert dense.modularity == 0.0
    assert dense.modularity_status == "ANALYTIC_COMPLETE_GRAPH_ONE_COMMUNITY"
    assert self_loop.n_undirected_nonself_edges == 0
    assert self_loop.mean_clustering_coefficient == 0.0
    assert self_loop.modularity is None
    assert self_loop.modularity_status == "UNDEFINED_NO_NONSELF_EDGES"


def test_curated_support_is_symmetrized_with_explicit_self_loops() -> None:
    curated = build_graph_support(
        GraphMode.CURATED,
        4,
        curated_edges=np.asarray([[0, 1], [1, 2], [2, 3]]),
        include_self_loops=True,
    )
    edges = set(map(tuple, curated.edge_index.T.tolist()))

    assert (0, 1) in edges and (1, 0) in edges
    assert {(node, node) for node in range(4)} <= edges
    assert curated.n_undirected_nonself_edges == 3


def test_rewire_preserves_exact_degree_sequence_but_changes_topology() -> None:
    cycle = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0]])
    curated = build_graph_support(GraphMode.CURATED, 8, curated_edges=cycle)
    rewired = build_graph_support(
        GraphMode.DEGREE_PRESERVING_REWIRE,
        8,
        curated_edges=cycle,
        rewire_seed=1001,
        rewire_multiplier=2.0,
    )

    assert rewired.undirected_degree_sequence == curated.undirected_degree_sequence
    assert rewired.n_undirected_nonself_edges == curated.n_undirected_nonself_edges
    assert rewired.n_connected_components == curated.n_connected_components
    assert rewired.n_isolates == curated.n_isolates
    assert rewired.component_partition_hash == curated.component_partition_hash
    assert rewired.swapped_edge_fraction is not None
    assert rewired.swapped_edge_fraction > 0
    assert rewired.sha256() != curated.sha256()


def test_rewire_preserves_exact_disconnected_component_membership() -> None:
    two_cycles = np.asarray(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 0],
            [6, 7],
            [7, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [11, 6],
        ]
    )
    curated = build_graph_support(GraphMode.CURATED, 13, curated_edges=two_cycles)
    rewired = build_graph_support(
        GraphMode.DEGREE_PRESERVING_REWIRE,
        13,
        curated_edges=two_cycles,
        rewire_seed=1002,
        rewire_multiplier=2.0,
    )

    assert curated.n_connected_components == 3
    assert rewired.component_partition_hash == curated.component_partition_hash
    assert rewired.n_isolates == 1

"""Matched graph supports for one fixed GAT architecture."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import networkx as nx
import numpy as np

from .errors import GraphSupportError


class GraphMode(str, Enum):
    """Predeclared adjacency supports; none changes the model architecture."""

    DENSE = "dense"
    SELF_LOOP = "self_loop"
    CURATED = "curated"
    DEGREE_PRESERVING_REWIRE = "degree_preserving_rewire"


@dataclass(frozen=True)
class GraphSupport:
    """A directed edge index plus an audit record of its undirected topology."""

    mode: GraphMode
    edge_index: np.ndarray
    n_nodes: int
    n_undirected_nonself_edges: int
    undirected_degree_sequence: tuple[int, ...]
    n_connected_components: int
    giant_component_size: int
    giant_component_fraction: float
    n_isolates: int
    component_partition_hash: str
    mean_clustering_coefficient: float
    degree_assortativity: float | None
    degree_assortativity_status: str
    modularity: float | None
    modularity_status: str
    swapped_edge_fraction: float | None
    include_self_loops: bool
    rewire_seed: int | None = None
    source_hash: str | None = None

    def sha256(self) -> str:
        """Hash the mode, dimensions, and canonical directed edge index."""

        digest = hashlib.sha256()
        digest.update(self.mode.value.encode("utf-8"))
        digest.update(str(self.n_nodes).encode("ascii"))
        digest.update(np.ascontiguousarray(self.edge_index, dtype=np.int64).tobytes())
        return digest.hexdigest()


def build_graph_support(
    mode: GraphMode | str,
    n_nodes: int,
    curated_edges: np.ndarray | Iterable[tuple[int, int]] | None = None,
    *,
    include_self_loops: bool = True,
    rewire_seed: int | None = None,
    rewire_multiplier: float = 10.0,
) -> GraphSupport:
    """Build one support while keeping the downstream GAT unchanged.

    Curated and rewired supports are treated as simple undirected graphs and emitted
    in both directions. Self-loops are controlled here because ``MatchedGAT`` disables
    the implicit self-loop behavior in every GAT layer.
    """

    graph_mode = GraphMode(mode)
    if n_nodes <= 0:
        raise GraphSupportError("n_nodes must be positive")
    if rewire_multiplier < 0:
        raise GraphSupportError("rewire_multiplier cannot be negative")

    source_edges: set[tuple[int, int]] = set()
    source_hash: str | None = None
    if curated_edges is not None:
        source_edges = _canonical_undirected_edges(curated_edges, n_nodes)
        source_hash = _hash_undirected_edges(source_edges, n_nodes)

    if graph_mode is GraphMode.DENSE:
        rows, columns = np.indices((n_nodes, n_nodes))
        edge_index = np.vstack([rows.ravel(), columns.ravel()]).astype(np.int64)
        if not include_self_loops:
            edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        n_undirected_edges = n_nodes * (n_nodes - 1) // 2
        degrees = (n_nodes - 1,) * n_nodes
        components = 1
        giant_component_size = n_nodes
        isolate_count = int(n_nodes == 1)
        partition_hash = _partition_hash((tuple(range(n_nodes)),))
        clustering = 1.0 if n_nodes >= 3 else 0.0
        assortativity = None
        assortativity_status = "UNDEFINED_ZERO_DEGREE_VARIANCE"
        modularity = 0.0
        modularity_status = "ANALYTIC_COMPLETE_GRAPH_ONE_COMMUNITY"
    elif graph_mode is GraphMode.SELF_LOOP:
        if not include_self_loops:
            raise GraphSupportError("self_loop mode requires include_self_loops=true")
        nodes = np.arange(n_nodes, dtype=np.int64)
        edge_index = np.vstack([nodes, nodes])
        n_undirected_edges = 0
        degrees = (0,) * n_nodes
        components = n_nodes
        giant_component_size = 1
        isolate_count = n_nodes
        partition_hash = _partition_hash(tuple((node,) for node in range(n_nodes)))
        clustering = 0.0
        assortativity = None
        assortativity_status = "UNDEFINED_ZERO_DEGREE_VARIANCE"
        modularity = None
        modularity_status = "UNDEFINED_NO_NONSELF_EDGES"
    elif graph_mode is GraphMode.CURATED:
        if curated_edges is None:
            raise GraphSupportError("curated mode requires curated_edges")
        undirected_edges = source_edges
        edge_index = _directed_edge_index(undirected_edges, n_nodes, include_self_loops)
    else:
        if curated_edges is None:
            raise GraphSupportError("degree_preserving_rewire mode requires curated_edges")
        if rewire_seed is None:
            raise GraphSupportError("degree_preserving_rewire requires an explicit rewire_seed")
        undirected_edges = _rewire_degree_preserving(
            source_edges,
            n_nodes,
            rewire_seed,
            int(np.ceil(len(source_edges) * rewire_multiplier)),
        )
        edge_index = _directed_edge_index(undirected_edges, n_nodes, include_self_loops)
    if graph_mode in {GraphMode.CURATED, GraphMode.DEGREE_PRESERVING_REWIRE}:
        n_undirected_edges = len(undirected_edges)
        degrees = _degree_sequence(undirected_edges, n_nodes)
        components, giant_component_size, isolate_count, partition_hash = _component_audit(
            undirected_edges, n_nodes
        )
        clustering, assortativity, assortativity_status, modularity, modularity_status = (
            _topology_statistics(undirected_edges, n_nodes)
        )
    swapped_edge_fraction: float | None = None
    if graph_mode is GraphMode.DEGREE_PRESERVING_REWIRE:
        original_degrees = _degree_sequence(source_edges, n_nodes)
        original_components, _, original_isolates, original_partition_hash = _component_audit(
            source_edges, n_nodes
        )
        if degrees != original_degrees:
            raise GraphSupportError("Rewiring changed the undirected degree sequence")
        if components != original_components:
            raise GraphSupportError("Rewiring changed the connected-component count")
        if isolate_count != original_isolates:
            raise GraphSupportError("Rewiring changed the isolate count")
        if partition_hash != original_partition_hash:
            raise GraphSupportError("Rewiring changed connected-component membership")
        if undirected_edges == source_edges and source_edges:
            raise GraphSupportError("Rewiring completed without changing the curated topology")
        swapped_edge_fraction = float(len(source_edges - undirected_edges) / len(source_edges))

    return GraphSupport(
        mode=graph_mode,
        edge_index=edge_index,
        n_nodes=n_nodes,
        n_undirected_nonself_edges=n_undirected_edges,
        undirected_degree_sequence=degrees,
        n_connected_components=components,
        giant_component_size=giant_component_size,
        giant_component_fraction=float(giant_component_size / n_nodes),
        n_isolates=isolate_count,
        component_partition_hash=partition_hash,
        mean_clustering_coefficient=clustering,
        degree_assortativity=assortativity,
        degree_assortativity_status=assortativity_status,
        modularity=modularity,
        modularity_status=modularity_status,
        swapped_edge_fraction=swapped_edge_fraction,
        include_self_loops=include_self_loops,
        rewire_seed=rewire_seed,
        source_hash=source_hash,
    )


def _canonical_undirected_edges(
    edges: np.ndarray | Iterable[tuple[int, int]], n_nodes: int
) -> set[tuple[int, int]]:
    array = np.asarray(list(edges) if not isinstance(edges, np.ndarray) else edges, dtype=np.int64)
    if array.ndim != 2:
        raise GraphSupportError("curated_edges must be a two-dimensional edge list")
    if array.shape[0] == 2 and array.shape[1] != 2:
        array = array.T
    if array.shape[1] != 2:
        raise GraphSupportError("curated_edges must have shape (n_edges, 2) or (2, n_edges)")
    if array.size and (array.min() < 0 or array.max() >= n_nodes):
        raise GraphSupportError("curated_edges contains an out-of-range node index")
    output: set[tuple[int, int]] = set()
    for left, right in array:
        if left == right:
            continue
        output.add((int(min(left, right)), int(max(left, right))))
    return output


def _directed_edge_index(
    edges: set[tuple[int, int]], n_nodes: int, include_self_loops: bool
) -> np.ndarray:
    directed = [(left, right) for left, right in edges]
    directed.extend((right, left) for left, right in edges)
    if include_self_loops:
        directed.extend((node, node) for node in range(n_nodes))
    directed.sort()
    if not directed:
        return np.empty((2, 0), dtype=np.int64)
    return np.asarray(directed, dtype=np.int64).T


def _degree_sequence(edges: set[tuple[int, int]], n_nodes: int) -> tuple[int, ...]:
    degrees = np.zeros(n_nodes, dtype=np.int64)
    for left, right in edges:
        degrees[left] += 1
        degrees[right] += 1
    return tuple(int(value) for value in degrees)


def _hash_undirected_edges(edges: set[tuple[int, int]], n_nodes: int) -> str:
    digest = hashlib.sha256(str(n_nodes).encode("ascii"))
    for left, right in sorted(edges):
        digest.update(f"{left},{right};".encode("ascii"))
    return digest.hexdigest()


def _rewire_degree_preserving(
    edges: set[tuple[int, int]], n_nodes: int, seed: int, n_swaps: int
) -> set[tuple[int, int]]:
    if not edges:
        raise GraphSupportError("Cannot rewire an empty curated graph")
    if n_swaps <= 0:
        raise GraphSupportError("A rewire arm must request at least one double-edge swap")
    original = set(edges)
    source_graph = nx.Graph()
    source_graph.add_nodes_from(range(n_nodes))
    source_graph.add_edges_from(original)
    rewired = set(original)
    total_edges = len(original)
    changed_any = False
    components = sorted(nx.connected_components(source_graph), key=lambda nodes: min(nodes))
    for component_index, component in enumerate(components):
        component_graph = source_graph.subgraph(component).copy()
        component_edges = component_graph.number_of_edges()
        if component_edges < 2 or len(component) < 4:
            continue
        component_swaps = max(1, int(round(n_swaps * component_edges / total_edges)))
        original_component_edges = {
            (int(min(left, right)), int(max(left, right)))
            for left, right in component_graph.edges()
        }
        component_rewired: set[tuple[int, int]] | None = None
        for offset in range(8):
            candidate = component_graph.copy()
            try:
                nx.double_edge_swap(
                    candidate,
                    nswap=component_swaps,
                    max_tries=max(1_000, component_swaps * 100),
                    seed=seed + component_index * 100 + offset,
                )
            except nx.NetworkXAlgorithmError:
                continue
            candidate_edges = {
                (int(min(left, right)), int(max(left, right))) for left, right in candidate.edges()
            }
            if candidate_edges != original_component_edges and nx.is_connected(candidate):
                component_rewired = candidate_edges
                break
        if component_rewired is None:
            continue
        rewired.difference_update(original_component_edges)
        rewired.update(component_rewired)
        changed_any = True
    if changed_any:
        return rewired
    raise GraphSupportError(
        "No connected component supported a topology-changing degree-preserving swap"
    )


def _component_audit(edges: set[tuple[int, int]], n_nodes: int) -> tuple[int, int, int, str]:
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(edges)
    components = [tuple(sorted(component)) for component in nx.connected_components(graph)]
    components.sort()
    component_sizes = [len(component) for component in components]
    isolate_count = sum(1 for degree in dict(graph.degree()).values() if degree == 0)
    partition_hash = _partition_hash(tuple(components))
    return len(component_sizes), max(component_sizes, default=0), isolate_count, partition_hash


def _partition_hash(components: tuple[tuple[int, ...], ...]) -> str:
    """Hash exact component membership without constructing a NetworkX graph."""

    return hashlib.sha256(repr(list(components)).encode("utf-8")).hexdigest()


def _topology_statistics(
    edges: set[tuple[int, int]], n_nodes: int
) -> tuple[float, float | None, str, float | None, str]:
    if len(edges) == n_nodes * (n_nodes - 1) // 2:
        return (
            1.0,
            None,
            "UNDEFINED_ZERO_DEGREE_VARIANCE",
            0.0,
            "ANALYTIC_COMPLETE_GRAPH",
        )
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(edges)
    clustering = float(nx.average_clustering(graph))
    degree_values = np.asarray([degree for _, degree in graph.degree()], dtype=float)
    if np.std(degree_values) == 0:
        assortativity = None
        assortativity_status = "UNDEFINED_ZERO_DEGREE_VARIANCE"
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assortativity_value = nx.degree_assortativity_coefficient(graph)
        if np.isfinite(assortativity_value):
            assortativity = float(assortativity_value)
            assortativity_status = "MEASURED"
        else:
            assortativity = None
            assortativity_status = "UNDEFINED_NUMERICAL"
    if graph.number_of_edges() == 0:
        modularity = None
        modularity_status = "UNDEFINED_NO_EDGES"
    else:
        communities = list(nx.community.greedy_modularity_communities(graph))
        modularity = float(nx.community.modularity(graph, communities))
        modularity_status = "MEASURED"
    return clustering, assortativity, assortativity_status, modularity, modularity_status

"""Graph controls, ensemble specifications, and exact provenance hashes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import networkx as nx
import numpy as np

from .hashing import sha256_json


@dataclass(frozen=True)
class GraphEnsembleSpec:
    """Deterministic graph-instance ensemble configuration."""

    graph_type: str
    instance_seeds: tuple[int, ...]
    swaps_per_edge: int | None = None
    connectivity_policy: str | None = None
    max_attempts_per_rewirable_edge: int | None = None
    minimum_rewirable_edge_fraction: float | None = None
    maximum_rewirable_original_overlap: float | None = None
    algorithm: str | None = None
    density_reference: str | None = None
    m_selection_policy: str | None = None
    node_mapping_policy: str | None = None
    initial_graph_policy: str | None = None
    self_loop_policy: str | None = None
    implementation_version_policy: str | None = None

    def validate(self) -> None:
        """Validate seeds and graph-specific configuration."""
        if self.graph_type not in {"barabasi_albert", "degree_preserving_rewired"}:
            raise ValueError(f"Unsupported graph ensemble type: {self.graph_type!r}")
        if not self.instance_seeds or len(set(self.instance_seeds)) != len(self.instance_seeds):
            raise ValueError("Graph-instance seeds must be non-empty and unique")
        expected_seeds = {
            "degree_preserving_rewired": (101, 102, 103, 104, 105),
            "barabasi_albert": (201, 202, 203, 204, 205),
        }[self.graph_type]
        if self.instance_seeds != expected_seeds:
            raise ValueError(
                f"{self.graph_type} seeds must be exactly the ordered tuple {expected_seeds}"
            )
        if self.graph_type == "degree_preserving_rewired":
            if self.swaps_per_edge is None or self.swaps_per_edge < 1:
                raise ValueError("Rewired ensembles require swaps_per_edge >= 1")
            if self.connectivity_policy != "preserve_original_component_node_sets_and_connectivity":
                raise ValueError(
                    "Rewired ensembles must freeze connectivity_policy as "
                    "'preserve_original_component_node_sets_and_connectivity'"
                )
            if self.max_attempts_per_rewirable_edge != 200:
                raise ValueError(
                    "Rewired ensembles require exactly 200 attempts per rewirable edge"
                )
            if self.minimum_rewirable_edge_fraction != 0.90:
                raise ValueError("Rewired ensembles require minimum rewirable coverage 0.90")
            if self.maximum_rewirable_original_overlap != 0.20:
                raise ValueError("Rewired ensembles require maximum rewirable overlap 0.20")
            if self.algorithm != "component_preserving_canonical_double_edge_swap_v2":
                raise ValueError("Rewired ensemble algorithm is not frozen")
        else:
            expected = {
                "algorithm": "networkx.barabasi_albert_graph",
                "density_reference": "string_go_union_simple_undirected_without_self_loops",
                "m_selection_policy": "argmin_(abs(m*(n-m)-E),m)_for_1_le_m_lt_n",
                "node_mapping_policy": "ordered_gene_panel_index_0_to_n_minus_1",
                "initial_graph_policy": "null_networkx_default_star_graph",
                "self_loop_policy": "restore_one_self_loop_per_gene_after_generation",
                "implementation_version_policy": "networkx_exact_version_from_environment_lock",
            }
            for field, value in expected.items():
                if getattr(self, field) != value:
                    raise ValueError(f"BA ensemble field {field!r} is not frozen")

    def instances(self) -> tuple[dict[str, int | str], ...]:
        """Expand the ensemble into auditable instance identifiers."""
        self.validate()
        return tuple(
            {
                "graph_type": self.graph_type,
                "instance_id": f"{self.graph_type}__seed_{seed}",
                "seed": seed,
            }
            for seed in self.instance_seeds
        )


@dataclass(frozen=True)
class GraphProvenance:
    """Provenance metadata bound to the exact canonical edge set."""

    graph_type: str
    graph_instance: str
    gene_panel_hash: str
    node_count: int
    directed_edge_count: int
    edge_hash: str
    source: str
    source_version: str
    parameters: Mapping[str, Any]
    parent_edge_hash: str | None = None
    seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible provenance."""
        return asdict(self)


@dataclass(frozen=True)
class RewiringAudit:
    """Exact counters from the canonical degree-preserving swap procedure."""

    algorithm_version: str
    connectivity_policy: str
    requested_swaps: int
    accepted_swaps: int
    proposal_count: int
    maximum_proposals: int
    total_non_self_edges: int
    rewirable_edges: int
    immutable_edges: int
    rewirable_edge_fraction: float
    immutable_component_node_sets: tuple[tuple[int, ...], ...]
    rejected_shared_endpoint: int
    rejected_self_loop: int
    rejected_duplicate: int
    rejected_component_disconnection: int
    rejection_rate: float
    total_original_edge_overlap: float
    rewirable_original_edge_overlap: float
    seed: int

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible audit counters."""
        return asdict(self)


def canonical_edges(edge_index: Any, node_count: int) -> tuple[tuple[int, int], ...]:
    """Return sorted, duplicate-free directed edges after range validation."""
    array = np.asarray(edge_index, dtype=np.int64)
    if array.ndim != 2 or array.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {array.shape}")
    edges = {(int(source), int(target)) for source, target in array.T}
    if any(
        source < 0 or target < 0 or source >= node_count or target >= node_count
        for source, target in edges
    ):
        raise ValueError("edge_index contains a node outside [0, node_count)")
    return tuple(sorted(edges))


def edge_hash(edge_index: Any, node_count: int) -> str:
    """Hash node count and canonical directed edges."""
    return sha256_json(
        {"node_count": node_count, "directed_edges": canonical_edges(edge_index, node_count)}
    )


def self_loop_edge_index(node_count: int) -> np.ndarray:
    """Create the same-backbone no-prior graph containing only self-loops."""
    if node_count < 1:
        raise ValueError("node_count must be positive")
    nodes = np.arange(node_count, dtype=np.int64)
    return np.stack([nodes, nodes])


def _symmetric_edge_index(graph: nx.Graph, node_count: int) -> np.ndarray:
    directed: set[tuple[int, int]] = {(node, node) for node in range(node_count)}
    for source, target in graph.edges():
        directed.add((int(source), int(target)))
        directed.add((int(target), int(source)))
    return np.asarray(sorted(directed), dtype=np.int64).T


def build_barabasi_albert_instance(node_count: int, m: int, seed: int) -> np.ndarray:
    """Build one deterministic BA instance with symmetric edges and self-loops."""
    if node_count < 2:
        raise ValueError("BA graph requires at least two nodes")
    if not 1 <= m < node_count:
        raise ValueError("BA attachment parameter m must satisfy 1 <= m < node_count")
    graph = nx.barabasi_albert_graph(node_count, m, seed=seed, initial_graph=None)
    return _symmetric_edge_index(graph, node_count)


def build_degree_preserving_rewired_instance(
    reference_edge_index: Any,
    node_count: int,
    seed: int,
    swaps_per_edge: int,
    max_attempts_per_rewirable_edge: int,
    minimum_rewirable_edge_fraction: float,
    maximum_rewirable_original_overlap: float,
    return_audit: bool = False,
) -> np.ndarray | tuple[np.ndarray, RewiringAudit]:
    """Rewire with counted canonical swaps while preserving every node degree exactly."""
    if swaps_per_edge < 1:
        raise ValueError("swaps_per_edge must be positive")
    if swaps_per_edge != 10:
        raise ValueError("Canonical rewiring requires exactly 10 swaps per rewirable edge")
    if max_attempts_per_rewirable_edge != 200:
        raise ValueError("Canonical rewiring requires exactly 200 proposals per rewirable edge")
    if minimum_rewirable_edge_fraction != 0.90:
        raise ValueError("Canonical rewiring requires rewirable-edge coverage 0.90")
    if maximum_rewirable_original_overlap != 0.20:
        raise ValueError("Canonical rewiring requires rewirable-edge overlap at most 0.20")
    reference_edges = canonical_edges(reference_edge_index, node_count)
    reference_loop_nodes = {source for source, target in reference_edges if source == target}
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    graph.add_edges_from((source, target) for source, target in reference_edges if source != target)
    before = dict(graph.degree())
    original_edges = {tuple(sorted((int(source), int(target)))) for source, target in graph.edges()}
    edge_count = len(original_edges)
    if edge_count < 2:
        raise ValueError("Reference graph has too few non-self edges to rewire")
    original_component_sets = tuple(
        tuple(sorted(int(node) for node in component))
        for component in sorted(
            nx.connected_components(graph), key=lambda value: tuple(sorted(value))
        )
    )
    component_states: list[dict[str, Any]] = []
    immutable_component_sets: list[tuple[int, ...]] = []
    immutable_edges: set[tuple[int, int]] = set()
    for nodes in original_component_sets:
        component_edges = sorted(
            tuple(sorted((int(source), int(target))))
            for source, target in graph.subgraph(nodes).edges()
        )
        has_nonincident_pair = any(
            len({*first, *second}) == 4
            for index, first in enumerate(component_edges)
            for second in component_edges[index + 1 :]
        )
        if len(nodes) < 4 or not has_nonincident_pair:
            immutable_component_sets.append(nodes)
            immutable_edges.update(component_edges)
        else:
            component_states.append(
                {"nodes": nodes, "edge_set": set(component_edges), "edge_list": component_edges}
            )
    rewirable_original_edges = original_edges - immutable_edges
    rewirable_edge_count = len(rewirable_original_edges)
    rewirable_fraction = rewirable_edge_count / edge_count
    if rewirable_fraction < minimum_rewirable_edge_fraction:
        raise ValueError(
            f"Only {rewirable_fraction:.6f} of edges are rewirable; "
            f"minimum is {minimum_rewirable_edge_fraction:.2f}"
        )
    requested_swaps = swaps_per_edge * rewirable_edge_count
    maximum_attempts = max_attempts_per_rewirable_edge * rewirable_edge_count
    generator = np.random.Generator(np.random.PCG64(seed))
    accepted = attempts = shared_endpoint = self_loop = duplicate = disconnected = 0
    weights = np.asarray([len(state["edge_list"]) for state in component_states], dtype=np.float64)
    weights /= weights.sum()
    while accepted < requested_swaps and attempts < maximum_attempts:
        attempts += 1
        state = component_states[int(generator.choice(len(component_states), p=weights))]
        edge_list = state["edge_list"]
        edge_set = state["edge_set"]
        first_index, second_index = generator.choice(len(edge_list), size=2, replace=False)
        first = edge_list[int(first_index)]
        second = edge_list[int(second_index)]
        a, b = first
        c, d = second
        if len({a, b, c, d}) < 4:
            shared_endpoint += 1
            continue
        if int(generator.integers(0, 2)) == 0:
            proposed = (tuple(sorted((a, c))), tuple(sorted((b, d))))
        else:
            proposed = (tuple(sorted((a, d))), tuple(sorted((b, c))))
        if any(source == target for source, target in proposed):
            self_loop += 1
            continue
        remaining = edge_set - {first, second}
        if proposed[0] == proposed[1] or any(edge in remaining for edge in proposed):
            duplicate += 1
            continue
        edge_set.remove(first)
        edge_set.remove(second)
        edge_set.update(proposed)
        candidate = nx.Graph()
        candidate.add_nodes_from(state["nodes"])
        candidate.add_edges_from(edge_set)
        if not nx.is_connected(candidate):
            edge_set.remove(proposed[0])
            edge_set.remove(proposed[1])
            edge_set.update((first, second))
            disconnected += 1
            continue
        edge_list[int(first_index)] = proposed[0]
        edge_list[int(second_index)] = proposed[1]
        edge_list.sort()
        accepted += 1
    if accepted != requested_swaps:
        raise RuntimeError(
            f"Canonical rewiring accepted {accepted}/{requested_swaps} swaps after "
            f"{attempts} attempts"
        )
    final_edges = set(immutable_edges)
    for state in component_states:
        final_edges.update(state["edge_set"])
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    graph.add_edges_from(final_edges)
    if dict(graph.degree()) != before:
        raise RuntimeError("Degree-preserving rewiring changed the degree sequence")
    final_component_sets = tuple(
        tuple(sorted(int(node) for node in component))
        for component in sorted(
            nx.connected_components(graph), key=lambda value: tuple(sorted(value))
        )
    )
    if final_component_sets != original_component_sets:
        raise RuntimeError("Degree-preserving rewiring changed connected-component node sets")
    rewirable_overlap = len(rewirable_original_edges & final_edges) / rewirable_edge_count
    total_overlap = len(original_edges & final_edges) / edge_count
    if rewirable_overlap > maximum_rewirable_original_overlap:
        raise RuntimeError(
            f"Rewirable original-edge overlap {rewirable_overlap:.6f} exceeds "
            f"{maximum_rewirable_original_overlap:.2f}"
        )
    if final_edges == original_edges:
        raise RuntimeError("Rewired edge set is identical to the curated union")
    audit = RewiringAudit(
        algorithm_version="component_preserving_canonical_double_edge_swap_v2",
        connectivity_policy="preserve_original_component_node_sets_and_connectivity",
        requested_swaps=requested_swaps,
        accepted_swaps=accepted,
        proposal_count=attempts,
        maximum_proposals=maximum_attempts,
        total_non_self_edges=edge_count,
        rewirable_edges=rewirable_edge_count,
        immutable_edges=len(immutable_edges),
        rewirable_edge_fraction=rewirable_fraction,
        immutable_component_node_sets=tuple(immutable_component_sets),
        rejected_shared_endpoint=shared_endpoint,
        rejected_self_loop=self_loop,
        rejected_duplicate=duplicate,
        rejected_component_disconnection=disconnected,
        rejection_rate=(attempts - accepted) / attempts,
        total_original_edge_overlap=total_overlap,
        rewirable_original_edge_overlap=rewirable_overlap,
        seed=seed,
    )
    directed = {(node, node) for node in reference_loop_nodes}
    for source, target in final_edges:
        directed.add((source, target))
        directed.add((target, source))
    edge_index = np.asarray(sorted(directed), dtype=np.int64).T
    return (edge_index, audit) if return_audit else edge_index


def graph_diagnostics(
    edge_index: Any,
    node_count: int,
    *,
    parent_edge_index: Any | None = None,
) -> dict[str, Any]:
    """Report frozen undirected graph properties and optional curated-parent overlap."""
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    graph.add_edges_from(
        (source, target)
        for source, target in canonical_edges(edge_index, node_count)
        if source != target
    )
    degrees = np.asarray([degree for _, degree in sorted(graph.degree())], dtype=float)
    with np.errstate(all="ignore"):
        assortativity = float(nx.degree_assortativity_coefficient(graph))
    diagnostics: dict[str, Any] = {
        "node_count": node_count,
        "undirected_edge_count": graph.number_of_edges(),
        "density": float(nx.density(graph)),
        "isolates": int(nx.number_of_isolates(graph)),
        "mean_degree": float(degrees.mean()),
        "degree_variance": float(degrees.var()),
        "average_clustering": float(nx.average_clustering(graph)),
        "connected_components": nx.number_connected_components(graph),
        "degree_assortativity": assortativity if np.isfinite(assortativity) else None,
        "degree_assortativity_state": "valid" if np.isfinite(assortativity) else "undefined",
    }
    if parent_edge_index is not None:
        parent = nx.Graph()
        parent.add_nodes_from(range(node_count))
        parent.add_edges_from(
            (source, target)
            for source, target in canonical_edges(parent_edge_index, node_count)
            if source != target
        )
        parent_edges = {tuple(sorted(edge)) for edge in parent.edges()}
        graph_edges = {tuple(sorted(edge)) for edge in graph.edges()}
        if not parent_edges:
            raise ValueError("Parent graph must contain at least one non-self edge")
        overlap = len(parent_edges & graph_edges) / len(parent_edges)
        parent_diagnostics = graph_diagnostics(parent_edge_index, node_count)
        diagnostics["curated_edge_overlap_fraction"] = overlap
        diagnostics["curated_edge_overlap_reduction"] = 1.0 - overlap
        diagnostics["parent_difference"] = {
            field: (
                diagnostics[field] - parent_diagnostics[field]
                if diagnostics[field] is not None and parent_diagnostics[field] is not None
                else None
            )
            for field in (
                "mean_degree",
                "degree_variance",
                "average_clustering",
                "connected_components",
                "degree_assortativity",
            )
        }
    return diagnostics


def graph_provenance(
    edge_index: Any,
    *,
    graph_type: str,
    graph_instance: str,
    gene_panel_hash: str,
    node_count: int,
    source: str,
    source_version: str,
    parameters: Mapping[str, Any],
    parent_edge_hash: str | None = None,
    seed: int | None = None,
) -> GraphProvenance:
    """Bind graph source metadata to an exact edge hash."""
    edges = canonical_edges(edge_index, node_count)
    return GraphProvenance(
        graph_type=graph_type,
        graph_instance=graph_instance,
        gene_panel_hash=gene_panel_hash,
        node_count=node_count,
        directed_edge_count=len(edges),
        edge_hash=sha256_json({"node_count": node_count, "directed_edges": edges}),
        source=source,
        source_version=source_version,
        parameters=dict(parameters),
        parent_edge_hash=parent_edge_hash,
        seed=seed,
    )


def load_ensemble_specs(values: Sequence[Mapping[str, Any]]) -> tuple[GraphEnsembleSpec, ...]:
    """Parse and validate graph-ensemble configuration mappings."""
    specs = tuple(
        GraphEnsembleSpec(
            graph_type=str(value["graph_type"]),
            instance_seeds=tuple(int(seed) for seed in value["instance_seeds"]),
            swaps_per_edge=(
                int(value["swaps_per_edge"]) if value.get("swaps_per_edge") is not None else None
            ),
            connectivity_policy=(
                str(value["connectivity_policy"])
                if value.get("connectivity_policy") is not None
                else None
            ),
            max_attempts_per_rewirable_edge=(
                int(value["max_attempts_per_rewirable_edge"])
                if value.get("max_attempts_per_rewirable_edge") is not None
                else None
            ),
            minimum_rewirable_edge_fraction=(
                float(value["minimum_rewirable_edge_fraction"])
                if value.get("minimum_rewirable_edge_fraction") is not None
                else None
            ),
            maximum_rewirable_original_overlap=(
                float(value["maximum_rewirable_original_overlap"])
                if value.get("maximum_rewirable_original_overlap") is not None
                else None
            ),
            algorithm=(str(value["algorithm"]) if value.get("algorithm") is not None else None),
            density_reference=(
                str(value["density_reference"])
                if value.get("density_reference") is not None
                else None
            ),
            m_selection_policy=(
                str(value["m_selection_policy"])
                if value.get("m_selection_policy") is not None
                else None
            ),
            node_mapping_policy=(
                str(value["node_mapping_policy"])
                if value.get("node_mapping_policy") is not None
                else None
            ),
            initial_graph_policy=(
                str(value["initial_graph_policy"])
                if value.get("initial_graph_policy") is not None
                else None
            ),
            self_loop_policy=(
                str(value["self_loop_policy"])
                if value.get("self_loop_policy") is not None
                else None
            ),
            implementation_version_policy=(
                str(value["implementation_version_policy"])
                if value.get("implementation_version_policy") is not None
                else None
            ),
        )
        for value in values
    )
    for spec in specs:
        spec.validate()
    return specs

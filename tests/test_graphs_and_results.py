from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import networkx as nx
import numpy as np
import pytest
import torch

from run_benchmark import model_parameter_hash
from turbognn_audit.graphs import (
    build_barabasi_albert_instance,
    build_degree_preserving_rewired_instance,
    edge_hash,
    graph_diagnostics,
    graph_provenance,
    self_loop_edge_index,
)
from turbognn_audit.model_config import load_model_spec
from turbognn_audit.results import build_fold_result
from turbognn_v2_models import TurboGNN

MODEL_CONTRACT = json.loads(
    (Path(__file__).parents[1] / "config" / "model_tds42.json").read_text(encoding="utf-8")
)
FEATURE_OPTIONS = MODEL_CONTRACT["feature_encoder"]
GAT_OPTIONS = MODEL_CONTRACT["gat"]


def test_bundled_model_contract_is_schema_validated_and_missing_fields_fail(tmp_path) -> None:
    contract_path = Path(__file__).parents[1] / "config" / "model_tds42.json"
    assert load_model_spec(contract_path).runtime["cuda_matmul_allow_tf32"] is False
    broken = json.loads(contract_path.read_text(encoding="utf-8"))
    del broken["gat"]["negative_slope"]
    broken_path = tmp_path / "broken_model.json"
    broken_path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_model_spec(broken_path)


def _undirected_degree(edge_index: np.ndarray, node_count: int) -> list[int]:
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    graph.add_edges_from(
        (int(source), int(target)) for source, target in edge_index.T if source != target
    )
    return [degree for _, degree in sorted(graph.degree())]


def test_edge_hash_is_order_and_duplicate_invariant() -> None:
    first = np.asarray([[0, 1, 1, 0], [1, 0, 0, 1]])
    second = np.asarray([[1, 0], [0, 1]])
    assert edge_hash(first, 2) == edge_hash(second, 2)


def test_random_graph_instances_are_deterministic_and_auditable() -> None:
    first = build_barabasi_albert_instance(20, 2, seed=101)
    repeat = build_barabasi_albert_instance(20, 2, seed=101)
    second = build_barabasi_albert_instance(20, 2, seed=102)
    assert edge_hash(first, 20) == edge_hash(repeat, 20)
    assert edge_hash(first, 20) != edge_hash(second, 20)
    provenance = graph_provenance(
        first,
        graph_type="barabasi_albert",
        graph_instance="barabasi_albert__seed_101",
        gene_panel_hash="genes",
        node_count=20,
        source="networkx",
        source_version=nx.__version__,
        parameters={"m": 2},
        seed=101,
    )
    assert provenance.edge_hash == edge_hash(first, 20)


def test_rewiring_preserves_every_node_degree() -> None:
    base = nx.random_regular_graph(4, 60, seed=7)
    reference = np.asarray(
        sorted(
            [(node, node) for node in base]
            + [(source, target) for source, target in base.edges()]
            + [(target, source) for source, target in base.edges()]
        ),
        dtype=np.int64,
    ).T
    rewired, audit = build_degree_preserving_rewired_instance(
        reference,
        60,
        seed=8,
        swaps_per_edge=10,
        max_attempts_per_rewirable_edge=200,
        minimum_rewirable_edge_fraction=0.90,
        maximum_rewirable_original_overlap=0.20,
        return_audit=True,
    )
    assert _undirected_degree(reference, 60) == _undirected_degree(rewired, 60)
    assert edge_hash(reference, 60) != edge_hash(rewired, 60)
    assert audit.accepted_swaps == audit.requested_swaps
    assert audit.proposal_count >= audit.accepted_swaps
    assert audit.rewirable_edge_fraction >= 0.90
    assert audit.rewirable_original_edge_overlap <= 0.20
    assert audit.rejection_rate == pytest.approx(
        (audit.proposal_count - audit.accepted_swaps) / audit.proposal_count
    )
    assert (
        audit.rejected_shared_endpoint
        + audit.rejected_self_loop
        + audit.rejected_duplicate
        + audit.rejected_component_disconnection
        == audit.proposal_count - audit.accepted_swaps
    )
    diagnostics = graph_diagnostics(rewired, 60, parent_edge_index=reference)
    assert diagnostics["curated_edge_overlap_fraction"] < 1.0
    assert diagnostics["parent_difference"]["mean_degree"] == pytest.approx(0.0)
    assert diagnostics["parent_difference"]["degree_variance"] == pytest.approx(0.0)
    assert "density" in diagnostics and "isolates" in diagnostics


def test_five_rewired_instances_are_distinct_and_component_preserving() -> None:
    graph = nx.cycle_graph(40)
    reference = np.asarray(
        sorted(
            [(node, node) for node in graph]
            + [(source, target) for source, target in graph.edges()]
            + [(target, source) for source, target in graph.edges()]
        ),
        dtype=np.int64,
    ).T
    hashes: set[str] = set()
    for seed in (101, 102, 103, 104, 105):
        rewired, audit = build_degree_preserving_rewired_instance(
            reference,
            40,
            seed=seed,
            swaps_per_edge=10,
            max_attempts_per_rewirable_edge=200,
            minimum_rewirable_edge_fraction=0.90,
            maximum_rewirable_original_overlap=0.20,
            return_audit=True,
        )
        rewired_graph = nx.Graph()
        rewired_graph.add_nodes_from(range(40))
        rewired_graph.add_edges_from(
            (int(source), int(target)) for source, target in rewired.T if source != target
        )
        assert nx.is_connected(rewired_graph)
        assert audit.immutable_component_node_sets == ()
        hashes.add(edge_hash(rewired, 40))
    assert len(hashes) == 5
    assert edge_hash(reference, 40) not in hashes


def test_self_loop_control_uses_exact_same_gat_architecture() -> None:
    node_count = 8
    self_loops = torch.tensor(self_loop_edge_index(node_count), dtype=torch.long)
    ring = torch.tensor(
        [[*range(node_count), *range(node_count)], [*range(node_count), *range(1, node_count), 0]],
        dtype=torch.long,
    )
    torch.manual_seed(42)
    control_model = TurboGNN(
        node_count,
        self_loops,
        hidden_dim=8,
        num_heads=2,
        dropout=0.1,
        gene_embedding_dim=32,
        feature_encoder_options=FEATURE_OPTIONS,
        gat_options=GAT_OPTIONS,
    )
    torch.manual_seed(42)
    graph_model = TurboGNN(
        node_count,
        ring,
        hidden_dim=8,
        num_heads=2,
        dropout=0.1,
        gene_embedding_dim=32,
        feature_encoder_options=FEATURE_OPTIONS,
        gat_options=GAT_OPTIONS,
    )
    assert type(control_model) is type(graph_model)
    control_parameters = {
        name: (tuple(parameter.shape), parameter.numel())
        for name, parameter in control_model.named_parameters()
    }
    graph_parameters = {
        name: (tuple(parameter.shape), parameter.numel())
        for name, parameter in graph_model.named_parameters()
    }
    assert control_parameters == graph_parameters
    assert model_parameter_hash(control_model) == model_parameter_hash(graph_model)
    torch.manual_seed(43)
    changed_seed_model = TurboGNN(
        node_count,
        self_loops,
        hidden_dim=8,
        num_heads=2,
        dropout=0.1,
        gene_embedding_dim=32,
        feature_encoder_options=FEATURE_OPTIONS,
        gat_options=GAT_OPTIONS,
    )
    assert model_parameter_hash(control_model) != model_parameter_hash(changed_seed_model)
    assert not control_model.conv1.add_self_loops
    assert (
        control_model.conv1.dropout
        == control_model.conv2.dropout
        == control_model.conv3.dropout
        == 0.0
    )
    control_model.eval()
    prediction = control_model(torch.zeros(node_count), torch.zeros(node_count))
    assert prediction.shape == (node_count,)


def test_initial_state_hash_includes_batchnorm_buffers_but_excludes_adjacency() -> None:
    node_count = 8
    self_loops = torch.tensor(self_loop_edge_index(node_count), dtype=torch.long)
    torch.manual_seed(42)
    model = TurboGNN(
        node_count,
        self_loops,
        hidden_dim=8,
        num_heads=2,
        dropout=0.1,
        gene_embedding_dim=32,
        feature_encoder_options=FEATURE_OPTIONS,
        gat_options=GAT_OPTIONS,
    )
    initial_hash = model_parameter_hash(model)
    with torch.no_grad():
        model.bn1.running_mean[0] = 1.0
    assert model_parameter_hash(model) != initial_hash
    with torch.no_grad():
        model.bn1.running_mean.zero_()
        model.edge_index = torch.flip(model.edge_index, dims=(1,))
    assert model_parameter_hash(model) == initial_hash


def test_self_loop_tds41_features_encode_gene_identity_and_broadcast_target_context() -> None:
    node_count = 4
    self_loops = torch.tensor(self_loop_edge_index(node_count), dtype=torch.long)
    torch.manual_seed(42)
    model = TurboGNN(
        node_count,
        self_loops,
        hidden_dim=16,
        num_heads=2,
        dropout=0.1,
        gene_embedding_dim=32,
        feature_encoder_options=FEATURE_OPTIONS,
        gat_options=GAT_OPTIONS,
    )
    model.eval()
    control_mean = torch.zeros(node_count)
    control_log_variance = torch.zeros(node_count)
    target_a = torch.tensor([True, False, False, False])
    target_b = torch.tensor([False, True, False, False])
    raw = model.feature_encoder.raw_features(control_mean, control_log_variance, target_a)
    assert raw.shape == (4, 68)
    raw_b = model.feature_encoder.raw_features(control_mean, control_log_variance, target_b)
    context_a = raw[0, 35:]
    context_b = raw_b[0, 35:]
    assert torch.max(torch.abs(context_a - context_b)).item() > 1e-8
    state_a = model.pre_message_features(control_mean, control_log_variance, target_a)
    state_b = model.pre_message_features(control_mean, control_log_variance, target_b)
    assert torch.max(torch.abs(state_a[2] - state_b[2])).item() > 1e-8
    assert torch.max(torch.abs(state_a[2] - state_a[3])).item() > 1e-8
    with torch.no_grad():
        output_a = model(control_mean, control_log_variance, target_a)
        output_b = model(control_mean, control_log_variance, target_b)
    assert abs(float(output_a[2] - output_b[2])) > 1e-8
    assert abs(float(output_a[2] - output_a[3])) > 1e-8
    with pytest.raises(ValueError, match="binary boolean"):
        model.feature_encoder.raw_features(
            control_mean,
            control_log_variance,
            target_a.to(dtype=torch.float32),
        )


def test_fold_result_persists_diagnostics_and_validates_schema() -> None:
    result = build_fold_result(
        run_identity={"dataset": "toy", "condition": "KO_A", "seed": 42},
        condition="KO_A",
        fold_index=0,
        gene_order=["A", "B"],
        y_true=[1.0, 2.0],
        y_pred=[1.1, 1.9],
        training_mean=[0.9, 1.8],
        control_profile=[0.0, 0.0],
        training_loss_by_epoch=[2.0, 1.0],
        metrics={"pearson_r": 1.0, "spearman_rho": 1.0, "jaccard": 1.0, "mse": 0.01},
        metric_states={"pearson_r": "valid", "spearman_rho": "valid"},
        metadata={"graph": {"edge_hash": "abc"}},
    )
    schema = json.loads(
        (Path(__file__).parents[1] / "schemas" / "fold_result.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(result.as_dict(), schema)
    assert result.gene_order == ("A", "B")
    assert result.gene_order_hash
    assert result.training_loss_by_epoch == (2.0, 1.0)
    assert result.vector_hashes["y_pred"]


def test_fold_result_rejects_nonfinite_predictions() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        build_fold_result(
            run_identity={"dataset": "toy"},
            condition="KO_A",
            fold_index=0,
            gene_order=["A"],
            y_true=[1.0],
            y_pred=[float("nan")],
            training_mean=[0.9],
            control_profile=[0.0],
            training_loss_by_epoch=[1.0],
            metrics={"mse": 0.0},
            metadata={},
        )

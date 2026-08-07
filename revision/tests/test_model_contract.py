from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cbac_revision.model import (
    MatchedGATConfig,
    build_model_input,
    initialize_matched_gat,
    state_dict_sha256,
)


def test_model_initialization_is_identical_for_same_seed() -> None:
    config = MatchedGATConfig(
        n_genes=3,
        hidden_dim=8,
        num_heads=2,
        num_layers=2,
        dropout=0.0,
        gene_identity_dim=4,
    )
    first = initialize_matched_gat(config, seed=42)
    second = initialize_matched_gat(config, seed=42)

    assert first.config.sha256() == second.config.sha256()
    assert state_dict_sha256(first) == state_dict_sha256(second)
    assert first.gene_identity.weight.shape == (3, 4)
    assert first.input_projection[0].in_features == 6


def test_model_config_has_no_graph_mode_parameter() -> None:
    fields = set(MatchedGATConfig.__dataclass_fields__)

    assert "graph_mode" not in fields
    assert "edge_index" not in fields


def test_hidden_normalization_is_per_node_layer_norm() -> None:
    config = MatchedGATConfig(n_genes=3, hidden_dim=8, num_heads=2, num_layers=2, dropout=0.0)
    model = initialize_matched_gat(config, seed=42)

    assert all(isinstance(layer, nn.LayerNorm) for layer in model.normalizations)
    assert not any(isinstance(layer, nn.BatchNorm1d) for layer in model.modules())


def test_self_loop_target_node_is_independent_of_other_node_inputs() -> None:
    config = MatchedGATConfig(
        n_genes=3,
        hidden_dim=8,
        num_heads=2,
        num_layers=2,
        dropout=0.0,
        gene_identity_dim=4,
    )
    model = initialize_matched_gat(config, seed=42).eval()
    self_loops = torch.arange(3, dtype=torch.long).repeat(2, 1)
    indicator = torch.tensor([True, False, False])
    first_control = torch.tensor([0.2, -0.1, 0.7])
    changed_other_nodes = torch.tensor([0.2, 50.0, -80.0])

    with torch.no_grad():
        first = model(first_control, self_loops, indicator)
        second = model(changed_other_nodes, self_loops, indicator)

    torch.testing.assert_close(first[0], second[0], rtol=0.0, atol=0.0)


def test_bool_and_float_target_indicators_are_equivalent() -> None:
    control = torch.tensor([0.2, -0.1, 0.7])
    boolean = torch.tensor([False, True, False])
    numeric = torch.tensor([0.0, 1.0, 0.0])

    bool_input = build_model_input(control, boolean)
    numeric_input = build_model_input(control, numeric)

    torch.testing.assert_close(bool_input, numeric_input)
    torch.testing.assert_close(bool_input[:, 0], control)
    torch.testing.assert_close(bool_input[:, 1], numeric)


def test_multi_target_indicator_preserves_all_control_expression() -> None:
    control = torch.tensor([0.2, -0.1, 0.7])
    model_input = build_model_input(control, torch.tensor([True, False, True]))

    torch.testing.assert_close(model_input[:, 0], control)
    assert model_input[:, 1].tolist() == [1.0, 0.0, 1.0]


def test_zero_target_indicator_is_reason_coded_failure() -> None:
    with pytest.raises(ValueError, match="ZERO_TARGET_INDICATOR"):
        build_model_input(torch.ones(3), torch.zeros(3))

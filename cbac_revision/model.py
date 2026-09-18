"""One graph-attention backbone shared by every adjacency arm.

This module is an optional model component. Importing the base package does not require
PyTorch; constructing ``MatchedGAT`` requires the ``model`` extra declared in
``pyproject.toml``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor
from torch_geometric.nn import GATConv


@dataclass(frozen=True)
class MatchedGATConfig:
    """Architecture parameters shared verbatim across all support modes."""

    n_genes: int
    hidden_dim: int = 128
    num_heads: int = 8
    num_layers: int = 3
    dropout: float = 0.1
    gene_identity_dim: int = 16

    def validate(self) -> None:
        """Validate a fixed, support-independent GAT configuration."""

        if (
            self.n_genes <= 0
            or self.hidden_dim <= 0
            or self.num_heads <= 0
            or self.num_layers <= 0
            or self.gene_identity_dim <= 0
        ):
            raise ValueError(
                "n_genes, hidden_dim, num_heads, num_layers, and gene_identity_dim must be positive"
            )
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def sha256(self) -> str:
        """Return the architecture fingerprint used in fold artifacts."""

        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class MatchedGAT(nn.Module):
    """A dynamic-support GAT whose parameterization never depends on graph mode."""

    def __init__(self, config: MatchedGATConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.gene_identity = nn.Embedding(config.n_genes, config.gene_identity_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(2 + config.gene_identity_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ELU(),
        )
        self.convolutions = nn.ModuleList(
            [
                GATConv(
                    config.hidden_dim,
                    config.hidden_dim,
                    heads=config.num_heads,
                    concat=False,
                    dropout=config.dropout,
                    add_self_loops=False,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.normalizations = nn.ModuleList(
            [nn.LayerNorm(config.hidden_dim) for _ in range(config.num_layers)]
        )
        self.residual_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.output_head = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        control_expression: Tensor,
        edge_index: Tensor,
        perturbation_indicator: Tensor,
    ) -> Tensor:
        """Predict one profile using the supplied support and the fixed architecture."""

        if control_expression.ndim != 1:
            raise ValueError("control_expression must have shape (n_genes,)")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, n_edges)")
        model_input = build_model_input(control_expression, perturbation_indicator)
        if len(control_expression) != self.config.n_genes:
            raise ValueError("control_expression length does not match config.n_genes")
        gene_order = torch.arange(self.config.n_genes, device=control_expression.device)
        model_input = torch.cat((model_input, self.gene_identity(gene_order)), dim=-1)
        features = self.input_projection(model_input)

        residual = self.residual_projection(features)
        for index, (convolution, normalization) in enumerate(
            zip(self.convolutions, self.normalizations, strict=True)
        ):
            features = normalization(convolution(features, edge_index))
            if index == len(self.convolutions) - 1:
                features = functional.elu(features + residual)
            else:
                features = functional.elu(features)
                features = functional.dropout(
                    features, p=self.config.dropout, training=self.training
                )
        return self.output_head(features).squeeze(-1)


def build_model_input(control_expression: Tensor, perturbation_indicator: Tensor) -> Tensor:
    """Create ``[control expression, binary target indicator]`` node features.

    Boolean and numeric 0/1 indicators are equivalent. At least one target is required;
    this prevents an unencodable perturbation from becoming a silent zero-signal input.
    """

    if control_expression.ndim != 1:
        raise ValueError("control_expression must have shape (n_genes,)")
    if perturbation_indicator.shape != control_expression.shape:
        raise ValueError("perturbation_indicator must match control_expression")
    if perturbation_indicator.dtype == torch.bool:
        indicator = perturbation_indicator.to(dtype=control_expression.dtype)
    else:
        if not torch.isfinite(perturbation_indicator).all():
            raise ValueError("perturbation_indicator must be finite")
        if not torch.all((perturbation_indicator == 0) | (perturbation_indicator == 1)):
            raise ValueError("perturbation_indicator must contain only 0 and 1")
        indicator = perturbation_indicator.to(dtype=control_expression.dtype)
    if not bool(torch.any(indicator == 1)):
        raise ValueError("[ZERO_TARGET_INDICATOR] at least one perturbation target is required")
    return torch.stack((control_expression, indicator), dim=-1)


def initialize_matched_gat(config: MatchedGATConfig, seed: int) -> MatchedGAT:
    """Initialize the shared architecture deterministically for one benchmark seed."""

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = MatchedGAT(config)
    return model


def state_dict_sha256(model: nn.Module) -> str:
    """Hash parameter names, shapes, dtypes, and bytes for initialization auditing."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        detached = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(detached.shape)).encode("ascii"))
        digest.update(str(detached.dtype).encode("ascii"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()

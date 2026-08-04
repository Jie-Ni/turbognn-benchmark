"""Frozen TDS-41/TDS-42 perturbation encoders and benchmark architectures."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class GenePerturbationEncoder(nn.Module):
    """Encode gene identity, control state, local targets, and broadcast perturbation context."""

    def __init__(
        self,
        num_genes: int,
        hidden_dim: int,
        gene_embedding_dim: int,
        options: Mapping[str, object],
    ) -> None:
        super().__init__()
        if num_genes < 1 or hidden_dim < 1 or gene_embedding_dim < 1:
            raise ValueError("Encoder dimensions must be positive")
        self.num_genes = num_genes
        self.gene_embedding_dim = gene_embedding_dim
        self.gene_embedding = nn.Embedding(
            num_genes,
            gene_embedding_dim,
            max_norm=(
                float(options["embedding_max_norm"])
                if options["embedding_max_norm"] is not None
                else None
            ),
            norm_type=float(options["embedding_norm_type"]),
            scale_grad_by_freq=bool(options["embedding_scale_grad_by_freq"]),
            sparse=bool(options["embedding_sparse"]),
        )
        if options["embedding_initialization"] != "normal_mean_0_std_1":
            raise ValueError("Unsupported gene-embedding initialization")
        nn.init.normal_(self.gene_embedding.weight, mean=0.0, std=1.0)
        input_dim = 2 * gene_embedding_dim + 4
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=bool(options["projection_linear_bias"])),
            nn.LayerNorm(
                hidden_dim,
                eps=float(options["layer_norm_eps"]),
                elementwise_affine=bool(options["layer_norm_elementwise_affine"]),
                bias=bool(options["layer_norm_bias"]),
            ),
            nn.ELU(alpha=float(options["elu_alpha"])),
        )

    def _target_mask(
        self,
        perturbation_mask: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if perturbation_mask is None:
            return torch.zeros(self.num_genes, dtype=reference.dtype, device=reference.device)
        if perturbation_mask.shape != (self.num_genes,):
            raise ValueError("perturbation_mask must have shape [num_genes]")
        mask = perturbation_mask.to(device=reference.device)
        if mask.dtype != torch.bool:
            raise ValueError("Canonical perturbation_mask must be a binary boolean target mask")
        return mask.to(dtype=reference.dtype)

    def raw_features(
        self,
        control_mean: torch.Tensor,
        control_log_variance: torch.Tensor,
        perturbation_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return the exact 68-column TDS-41 features before linear projection."""
        if control_mean.shape != (self.num_genes,) or control_log_variance.shape != (
            self.num_genes,
        ):
            raise ValueError("Control feature vectors must have shape [num_genes]")
        if not bool(torch.isfinite(control_mean).all()) or not bool(
            torch.isfinite(control_log_variance).all()
        ):
            raise ValueError("Control feature vectors must be finite")
        gene_ids = torch.arange(self.num_genes, device=control_mean.device)
        gene_embeddings = self.gene_embedding(gene_ids)
        target_strength = self._target_mask(perturbation_mask, control_mean)
        target_count = target_strength.sum()
        pooled = (gene_embeddings * target_strength.unsqueeze(-1)).sum(dim=0)
        pooled = pooled / target_count.clamp_min(1.0)
        pooled = torch.where(target_count > 0.0, pooled, torch.zeros_like(pooled))
        context = pooled.unsqueeze(0).expand(self.num_genes, -1)
        log_target_count = torch.log1p(target_count).expand(self.num_genes, 1)
        return torch.cat(
            [
                gene_embeddings,
                control_mean.unsqueeze(-1),
                control_log_variance.unsqueeze(-1),
                target_strength.unsqueeze(-1),
                context,
                log_target_count,
            ],
            dim=-1,
        )

    def forward(
        self,
        control_mean: torch.Tensor,
        control_log_variance: torch.Tensor,
        perturbation_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.input_projection(
            self.raw_features(control_mean, control_log_variance, perturbation_mask)
        )


class TurboGNN(nn.Module):
    """Same-backbone GAT whose only between-arm difference is the frozen adjacency."""

    def __init__(
        self,
        num_genes: int,
        edge_index: torch.Tensor,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        gene_embedding_dim: int,
        feature_encoder_options: Mapping[str, object],
        gat_options: Mapping[str, object],
    ) -> None:
        super().__init__()
        self.register_buffer("edge_index", edge_index.to(dtype=torch.long), persistent=True)
        self.num_genes = num_genes
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.feature_encoder = GenePerturbationEncoder(
            num_genes,
            hidden_dim,
            gene_embedding_dim,
            feature_encoder_options,
        )
        self.conv1 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=num_heads,
            concat=bool(gat_options["concat"]),
            dropout=float(gat_options["attention_coefficient_dropout"]),
            add_self_loops=bool(gat_options["add_self_loops"]),
            negative_slope=float(gat_options["negative_slope"]),
            edge_dim=gat_options["edge_dim"],
            fill_value=str(gat_options["fill_value"]),
            bias=bool(gat_options["bias"]),
            residual=bool(gat_options["residual"]),
            aggr=str(gat_options["aggregation"]),
            flow=str(gat_options["flow"]),
            decomposed_layers=int(gat_options["decomposed_layers"]),
        )
        batch_norm_kwargs = {
            "eps": float(gat_options["batch_norm_eps"]),
            "momentum": float(gat_options["batch_norm_momentum"]),
            "affine": bool(gat_options["batch_norm_affine"]),
            "track_running_stats": bool(gat_options["batch_norm_track_running_stats"]),
        }
        self.bn1 = nn.BatchNorm1d(hidden_dim, **batch_norm_kwargs)
        self.conv2 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=num_heads,
            concat=bool(gat_options["concat"]),
            dropout=float(gat_options["attention_coefficient_dropout"]),
            add_self_loops=bool(gat_options["add_self_loops"]),
            negative_slope=float(gat_options["negative_slope"]),
            edge_dim=gat_options["edge_dim"],
            fill_value=str(gat_options["fill_value"]),
            bias=bool(gat_options["bias"]),
            residual=bool(gat_options["residual"]),
            aggr=str(gat_options["aggregation"]),
            flow=str(gat_options["flow"]),
            decomposed_layers=int(gat_options["decomposed_layers"]),
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim, **batch_norm_kwargs)
        self.conv3 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=num_heads,
            concat=bool(gat_options["concat"]),
            dropout=float(gat_options["attention_coefficient_dropout"]),
            add_self_loops=bool(gat_options["add_self_loops"]),
            negative_slope=float(gat_options["negative_slope"]),
            edge_dim=gat_options["edge_dim"],
            fill_value=str(gat_options["fill_value"]),
            bias=bool(gat_options["bias"]),
            residual=bool(gat_options["residual"]),
            aggr=str(gat_options["aggregation"]),
            flow=str(gat_options["flow"]),
            decomposed_layers=int(gat_options["decomposed_layers"]),
        )
        self.bn3 = nn.BatchNorm1d(hidden_dim, **batch_norm_kwargs)
        self.residual_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=bool(gat_options["residual_linear_bias"]),
        )
        self.delta_decoder = nn.Linear(hidden_dim, 1, bias=bool(gat_options["decoder_linear_bias"]))
        self.elu_alpha = float(gat_options["elu_alpha"])
        if any(
            convolution.node_dim != int(gat_options["node_dim"])
            for convolution in (self.conv1, self.conv2, self.conv3)
        ):
            raise RuntimeError("Locked GATConv API did not retain the frozen node_dim")

    def pre_message_features(
        self,
        control_mean: torch.Tensor,
        control_log_variance: torch.Tensor,
        perturbation_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Expose the adjacency-independent state for E0 expressivity/parity tests."""
        return self.feature_encoder(control_mean, control_log_variance, perturbation_mask)

    def forward(
        self,
        control_mean: torch.Tensor,
        control_log_variance: torch.Tensor,
        perturbation_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.pre_message_features(control_mean, control_log_variance, perturbation_mask)
        residual = self.residual_projection(x)
        x = F.elu(self.bn1(self.conv1(x, self.edge_index)), alpha=self.elu_alpha)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn2(self.conv2(x, self.edge_index)), alpha=self.elu_alpha)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, self.edge_index)
        x = self.bn3(x)
        x = F.elu(x + residual, alpha=self.elu_alpha)
        return self.delta_decoder(x).squeeze(-1)


class TurboGCN(nn.Module):
    """Legacy GCN sensitivity using the same TDS-41 feature encoder."""

    def __init__(
        self,
        num_genes: int,
        edge_index: torch.Tensor,
        hidden_dim: int,
        dropout: float,
        gene_embedding_dim: int,
        feature_encoder_options: Mapping[str, object],
    ) -> None:
        super().__init__()
        self.register_buffer("edge_index", edge_index.to(dtype=torch.long), persistent=True)
        self.dropout = dropout
        self.feature_encoder = GenePerturbationEncoder(
            num_genes,
            hidden_dim,
            gene_embedding_dim,
            feature_encoder_options,
        )
        self.conv1 = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.delta_decoder = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        control_mean: torch.Tensor,
        control_log_variance: torch.Tensor,
        perturbation_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.feature_encoder(control_mean, control_log_variance, perturbation_mask)
        x = F.elu(self.bn1(self.conv1(x, self.edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn2(self.conv2(x, self.edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn3(self.conv3(x, self.edge_index)))
        return self.delta_decoder(x).squeeze(-1)


class SimpleTransformer(nn.Module):
    """Architecture sensitivity using the same TDS-41 node and perturbation features."""

    def __init__(
        self,
        num_genes: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        gene_embedding_dim: int,
        feature_encoder_options: Mapping[str, object],
        transformer_options: Mapping[str, object],
    ) -> None:
        super().__init__()
        self.feature_encoder = GenePerturbationEncoder(
            num_genes,
            d_model,
            gene_embedding_dim,
            feature_encoder_options,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=bool(transformer_options["batch_first"]),
            norm_first=bool(transformer_options["pre_norm"]),
            dropout=dropout,
            activation=str(transformer_options["activation"]).lower(),
            layer_norm_eps=float(transformer_options["layer_norm_eps"]),
            bias=bool(transformer_options["bias"]),
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=transformer_options["encoder_norm"],
            enable_nested_tensor=bool(transformer_options["encoder_enable_nested_tensor"]),
            mask_check=bool(transformer_options["encoder_mask_check"]),
        )
        self.delta_decoder = nn.Linear(
            d_model, 1, bias=bool(transformer_options["decoder_linear_bias"])
        )

    def forward(
        self,
        control_mean: torch.Tensor,
        control_log_variance: torch.Tensor,
        perturbation_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.feature_encoder(control_mean, control_log_variance, perturbation_mask)
        x = self.transformer_encoder(x.unsqueeze(0)).squeeze(0)
        return self.delta_decoder(x).squeeze(-1)

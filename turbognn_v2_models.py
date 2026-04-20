"""
TurboGNN v2: Fixed architecture for single-cell perturbation prediction.

Key fixes over v1:
    1. Uses control expression profile as node features (not learned embeddings)
    2. Proper input projection from expression values to hidden dim
    3. Still supports hard masking and dose-response simulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from typing import Optional


class TurboGNN(nn.Module):
    """
    Knowledge-Primed Graph Attention Network v2.

    Takes control expression profiles as input and predicts perturbed expression.
    Uses biological knowledge graph (PPI/GO) for message passing.

    Args:
        num_genes: Number of genes.
        edge_index: Graph edges [2, num_edges].
        hidden_dim: Hidden dimension (default: 128).
        num_heads: GAT attention heads (default: 8).
        dropout: Dropout rate (default: 0.1).
    """

    def __init__(self, num_genes, edge_index, hidden_dim=128, num_heads=8, dropout=0.1):
        super().__init__()
        self.edge_index = edge_index
        self.num_genes = num_genes
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Project scalar expression values to hidden dim
        self.input_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )

        # 3-layer GAT
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Residual projection (if dims differ)
        self.residual_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output head
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, ctrl_expr, perturbation_mask=None):
        """
        Args:
            ctrl_expr: Control expression profile [num_genes].
            perturbation_mask:
                - bool: True = knocked out gene (embedding zeroed)
                - float: 0.0 = full KO, 1.0 = WT (for dose-response)
                - None: wild-type prediction
        Returns:
            Predicted expression [num_genes].
        """
        # Project expression to hidden dim: [num_genes] -> [num_genes, hidden_dim]
        x = self.input_proj(ctrl_expr.unsqueeze(-1))

        # Hard masking: zero out perturbed gene representations
        if perturbation_mask is not None:
            x = x.clone()
            if perturbation_mask.dtype == torch.bool:
                x[perturbation_mask] = 0.0
            else:
                x = x * perturbation_mask.unsqueeze(-1)

        # Save for residual
        residual = self.residual_proj(x)

        # Layer 1
        x = self.conv1(x, self.edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, self.edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3 + residual
        x = self.conv3(x, self.edge_index)
        x = self.bn3(x)
        x = F.elu(x + residual)

        return self.head(x).squeeze(-1)


class TurboGCN(nn.Module):
    """Ablation: GCN instead of GAT, same expression-based input."""

    def __init__(self, num_genes, edge_index, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.edge_index = edge_index
        self.num_genes = num_genes
        self.dropout = dropout

        self.input_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, ctrl_expr, perturbation_mask=None):
        x = self.input_proj(ctrl_expr.unsqueeze(-1))
        if perturbation_mask is not None:
            x = x.clone()
            if perturbation_mask.dtype == torch.bool:
                x[perturbation_mask] = 0.0
            else:
                x = x * perturbation_mask.unsqueeze(-1)
        x = F.elu(self.bn1(self.conv1(x, self.edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn2(self.conv2(x, self.edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn3(self.conv3(x, self.edge_index)))
        return self.head(x).squeeze(-1)


class SimpleTransformer(nn.Module):
    """Sequence-based transformer baseline. Takes expression as input."""

    def __init__(self, num_genes, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.num_genes = num_genes

        self.input_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.ELU(),
        )
        # Learnable positional encoding for gene positions
        self.pos_enc = nn.Parameter(torch.randn(num_genes, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            batch_first=True, norm_first=True, dropout=0.1,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, ctrl_expr, mask=None):
        """
        Args:
            ctrl_expr: [num_genes] control expression.
            mask: bool tensor, True = knocked out.
        """
        x = self.input_proj(ctrl_expr.unsqueeze(-1))  # [num_genes, d_model]
        x = x + self.pos_enc

        if mask is not None:
            x = x.clone()
            x[mask] = 0.0

        x = x.unsqueeze(0)  # [1, num_genes, d_model]
        x = self.transformer_encoder(x)
        return self.head(x).squeeze()  # [num_genes]

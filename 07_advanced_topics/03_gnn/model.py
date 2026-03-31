"""
Graph Neural Networks in PyTorch
===================================

PyTorch GNN layers: GCN, GraphSAGE, GAT with configurable multi-layer
architectures for node and graph classification.

Message-passing framework:
    h_i' = UPDATE(h_i, AGGREGATE({MSG(h_i, h_j) : j ∈ N(i)}))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class GNNConfig:
    """GNN configuration."""
    in_features: int = 34
    hidden_dim: int = 32
    out_features: int = 2
    n_layers: int = 2
    gnn_type: str = "gcn"           # gcn, sage, gat
    n_heads: int = 4                # for GAT
    dropout: float = 0.5
    task: str = "node"              # node or graph classification


# =============================================================================
# GRAPH DATA
# =============================================================================

@dataclass
class GraphData:
    """Container for graph-structured data."""
    x: torch.Tensor                  # Node features (N, F)
    edge_index: torch.Tensor         # Edge list (2, E)
    labels: Optional[torch.Tensor] = None
    train_mask: Optional[torch.Tensor] = None
    test_mask: Optional[torch.Tensor] = None
    batch: Optional[torch.Tensor] = None  # For graph-level tasks

    @property
    def n_nodes(self) -> int:
        return self.x.size(0)

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        if self.labels is not None:
            self.labels = self.labels.to(device)
        if self.train_mask is not None:
            self.train_mask = self.train_mask.to(device)
        if self.test_mask is not None:
            self.test_mask = self.test_mask.to(device)
        if self.batch is not None:
            self.batch = self.batch.to(device)
        return self


def edge_index_to_adj(edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Convert edge list to adjacency matrix."""
    adj = torch.zeros(n_nodes, n_nodes, device=edge_index.device)
    adj[edge_index[0], edge_index[1]] = 1.0
    return adj


# =============================================================================
# GCN LAYER
# =============================================================================

class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer.

    H' = σ(D̃^(-1/2) Ã D̃^(-1/2) H W)

    Uses symmetric normalization of adjacency with self-loops.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        # Build adjacency with self-loops
        adj = edge_index_to_adj(edge_index, n) + torch.eye(n, device=x.device)

        # Symmetric normalization: D^(-1/2) A D^(-1/2)
        degree = adj.sum(dim=1).clamp(min=1)
        d_inv_sqrt = degree.pow(-0.5)
        adj_norm = adj * d_inv_sqrt.unsqueeze(0) * d_inv_sqrt.unsqueeze(1)

        # Propagate and transform
        return self.linear(adj_norm @ x)


# =============================================================================
# GRAPHSAGE LAYER
# =============================================================================

class SAGELayer(nn.Module):
    """
    GraphSAGE layer with mean aggregation.

    h_v' = σ(W · [h_v || MEAN({h_u : u ∈ N(v)})])

    Concatenates self-features with aggregated neighbor features.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(2 * in_features, out_features, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        adj = edge_index_to_adj(edge_index, n)

        # Mean aggregation of neighbors
        degree = adj.sum(dim=1, keepdim=True).clamp(min=1)
        neigh_agg = (adj @ x) / degree

        # Concatenate self + neighbor
        combined = torch.cat([x, neigh_agg], dim=1)
        out = self.linear(combined)

        # L2 normalize
        return F.normalize(out, p=2, dim=1)


# =============================================================================
# GAT LAYER
# =============================================================================

class GATLayer(nn.Module):
    """
    Graph Attention Network layer with multi-head attention.

    e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
    α_ij = softmax_j(e_ij)
    h_i' = σ(Σ_j α_ij W h_j)

    Multi-head: concatenate or average K independent attention heads.
    """

    def __init__(self, in_features: int, out_features: int,
                 n_heads: int = 4, concat: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.concat = concat
        self.out_per_head = out_features // n_heads if concat else out_features

        self.W = nn.Parameter(torch.empty(n_heads, in_features, self.out_per_head))
        self.a_src = nn.Parameter(torch.empty(n_heads, self.out_per_head, 1))
        self.a_dst = nn.Parameter(torch.empty(n_heads, self.out_per_head, 1))

        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        adj = edge_index_to_adj(edge_index, n) + torch.eye(n, device=x.device)

        # Transform per head: (K, N, F_out)
        Wh = torch.einsum('nf,khf->knh', x, self.W)

        # Attention scores
        e_src = torch.einsum('knh,kho->kno', Wh, self.a_src).squeeze(-1)  # (K, N)
        e_dst = torch.einsum('knh,kho->kno', Wh, self.a_dst).squeeze(-1)  # (K, N)

        # Pairwise: e_ij = e_src_i + e_dst_j
        e = e_src.unsqueeze(-1) + e_dst.unsqueeze(-2)  # (K, N, N)
        e = self.leaky_relu(e)

        # Mask non-edges
        mask = adj.unsqueeze(0)  # (1, N, N)
        e = e.masked_fill(mask == 0, float('-inf'))

        # Softmax attention
        alpha = F.softmax(e, dim=-1)  # (K, N, N)
        alpha = alpha.masked_fill(mask == 0, 0.0)

        # Aggregate
        out = torch.einsum('knm,kmh->knh', alpha, Wh)  # (K, N, F_out)

        if self.concat:
            return out.permute(1, 0, 2).reshape(n, -1)  # (N, K*F_out)
        else:
            return out.mean(dim=0)  # (N, F_out)


# =============================================================================
# GNN MODEL
# =============================================================================

class GNN(nn.Module):
    """
    Multi-layer GNN for node or graph classification.

    Supports GCN, GraphSAGE, and GAT layer types.
    """

    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config

        layers = []
        in_dim = config.in_features

        for i in range(config.n_layers - 1):
            out_dim = config.hidden_dim
            if config.gnn_type == "gcn":
                layers.append(GCNLayer(in_dim, out_dim))
            elif config.gnn_type == "sage":
                layers.append(SAGELayer(in_dim, out_dim))
            elif config.gnn_type == "gat":
                layers.append(GATLayer(in_dim, out_dim, config.n_heads, concat=True))
            in_dim = out_dim

        # Output layer
        if config.gnn_type == "gat":
            layers.append(GATLayer(in_dim, config.out_features,
                                    n_heads=1, concat=False))
        elif config.gnn_type == "sage":
            layers.append(SAGELayer(in_dim, config.out_features))
        else:
            layers.append(GCNLayer(in_dim, config.out_features))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, data: GraphData) -> torch.Tensor:
        x = data.x
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, data.edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.layers[-1](x, data.edge_index)

        if self.config.task == "graph" and data.batch is not None:
            # Global mean pooling for graph classification
            unique = data.batch.unique()
            pooled = torch.stack([x[data.batch == b].mean(dim=0) for b in unique])
            return pooled

        return x

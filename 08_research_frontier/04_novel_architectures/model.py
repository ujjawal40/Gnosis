"""
Novel Architectures: Geometric Attention
==========================================

Compositional networks with geometric attention mechanisms that
operate on structured inputs (graphs, point clouds, molecules).

Architecture:
    Nodes → GeometricAttention(distance-aware) → Message Passing → Pool → Classify
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class GeometricConfig:
    node_dim: int = 3        # Input node features
    hidden_dim: int = 64
    output_dim: int = 2
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1
    use_distance: bool = True  # Distance-aware attention


class GeometricAttention(nn.Module):
    """
    Distance-aware multi-head attention for geometric data.
    Modulates attention scores by pairwise distances between nodes.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1,
                 use_distance: bool = True):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.d_k = dim // n_heads
        self.use_distance = use_distance

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_o = nn.Linear(dim, dim)

        if use_distance:
            # Learn distance bias per head
            self.dist_bias = nn.Parameter(torch.randn(n_heads, 1, 1))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape

        Q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Distance modulation
        if self.use_distance and positions is not None:
            # Pairwise distances: (B, N, N)
            dists = torch.cdist(positions, positions)
            # RBF kernel: exp(-gamma * d^2)
            dist_weights = torch.exp(-self.dist_bias.abs() * dists.unsqueeze(1))
            attn = attn + torch.log(dist_weights + 1e-8)

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().reshape(B, N, C)
        return self.W_o(out)


class GeometricBlock(nn.Module):
    """Geometric transformer block with distance-aware attention."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1,
                 use_distance: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GeometricAttention(dim, n_heads, dropout, use_distance)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, positions=None, mask=None):
        x = x + self.attn(self.norm1(x), positions, mask)
        x = x + self.ffn(self.norm2(x))
        return x


class GeometricTransformer(nn.Module):
    """
    Transformer for geometric/structured data.

    Can process:
        - Point clouds (3D coordinates + features)
        - Molecular graphs (atom features + positions)
        - Any data with spatial structure

    Uses distance-aware attention to incorporate geometric inductive bias.
    """

    def __init__(self, config: GeometricConfig):
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.node_dim, config.hidden_dim)

        self.blocks = nn.ModuleList([
            GeometricBlock(
                config.hidden_dim, config.n_heads,
                config.dropout, config.use_distance,
            )
            for _ in range(config.n_layers)
        ])

        self.norm = nn.LayerNorm(config.hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(self, x: torch.Tensor, positions: torch.Tensor = None,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, n_nodes, node_dim) node features
            positions: (batch, n_nodes, 3) spatial coordinates
            mask: (batch, n_nodes) binary mask

        Returns:
            logits: (batch, output_dim)
        """
        h = self.input_proj(x)

        for block in self.blocks:
            h = block(h, positions, mask)

        h = self.norm(h)

        # Global pooling (mean over nodes)
        if mask is not None:
            h = (h * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            h = h.mean(dim=1)

        return self.classifier(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

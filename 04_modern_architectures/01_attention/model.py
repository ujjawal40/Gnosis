"""
Attention Mechanisms in PyTorch
=================================

Scaled dot-product attention and multi-head attention for
sequence classification (IMDB sentiment analysis).

Architecture:
    Tokens → Embedding → Positional Encoding → MultiHeadAttention × N → Pool → FC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass


@dataclass
class AttentionConfig:
    vocab_size: int = 30000
    embed_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 256
    max_seq_len: int = 512
    n_classes: int = 2
    dropout: float = 0.1


class ScaledDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    The scaling factor sqrt(d_k) prevents the dot products from growing
    large, which would push softmax into regions with tiny gradients.
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: torch.Tensor = None) -> tuple:
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention: run multiple attention operations in parallel,
    each learning different aspects of the input relationships.
    """

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.d_k = embed_dim // n_heads
        self.n_heads = n_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections and reshape to (batch, heads, seq, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Attention
        output, attn_weights = self.attention(Q, K, V, mask)

        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, -1,
                                                           self.n_heads * self.d_k)
        return self.W_o(output), attn_weights


class TransformerBlock(nn.Module):
    """Single transformer encoder block: MHA → Add&Norm → FFN → Add&Norm."""

    def __init__(self, embed_dim: int, n_heads: int, ff_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.mha(x, x, x, mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x, attn_weights


class AttentionClassifier(nn.Module):
    """
    Text classifier using self-attention.
    Stacks transformer blocks and pools for classification.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim,
                                       padding_idx=0)
        self.pos_encoding = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads,
                             config.ff_dim, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, config.n_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        x = self.embedding(x) + self.pos_encoding(positions)
        x = self.dropout(x)

        for layer in self.layers:
            x, _ = layer(x, mask)

        # Mean pooling (ignore padding)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

"""
Efficient Attention in PyTorch
=================================

PyTorch implementations of KV-Cache, MQA, GQA, and sliding window attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class EfficientAttnConfig:
    d_model: int = 256
    n_heads: int = 8
    n_kv_heads: int = 2          # for GQA; =1 for MQA, =n_heads for MHA
    max_len: int = 2048
    window_size: int = 256       # for sliding window
    dropout: float = 0.1


# =============================================================================
# KV-CACHE
# =============================================================================

class KVCache:
    """Key-Value cache for autoregressive generation."""

    def __init__(self, batch_size: int, n_heads: int, d_head: int,
                 max_len: int, device: torch.device):
        self.k = torch.zeros(batch_size, n_heads, max_len, d_head, device=device)
        self.v = torch.zeros(batch_size, n_heads, max_len, d_head, device=device)
        self.length = 0

    def update(self, k: torch.Tensor, v: torch.Tensor):
        """Update cache with new K, V and return full K, V."""
        seq_len = k.size(2)
        self.k[:, :, self.length:self.length + seq_len] = k
        self.v[:, :, self.length:self.length + seq_len] = v
        self.length += seq_len
        return self.k[:, :, :self.length], self.v[:, :, :self.length]


# =============================================================================
# GROUPED QUERY ATTENTION
# =============================================================================

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).

    Generalizes MHA/MQA:
        n_kv_heads = n_heads → standard MHA
        n_kv_heads = 1 → Multi-Query Attention
        1 < n_kv_heads < n_heads → Grouped Query Attention
    """

    def __init__(self, config: EfficientAttnConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_model // config.n_heads
        self.heads_per_group = config.n_heads // config.n_kv_heads

        self.W_q = nn.Linear(config.d_model, config.n_heads * self.d_head, bias=False)
        self.W_k = nn.Linear(config.d_model, config.n_kv_heads * self.d_head, bias=False)
        self.W_v = nn.Linear(config.d_model, config.n_kv_heads * self.d_head, bias=False)
        self.W_o = nn.Linear(config.n_heads * self.d_head, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                cache: Optional[KVCache] = None) -> torch.Tensor:
        B, L, _ = x.shape

        Q = self.W_q(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_kv_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_kv_heads, self.d_head).transpose(1, 2)

        if cache is not None:
            K, V = cache.update(K, V)

        # Repeat K, V for grouped heads
        if self.n_kv_heads != self.n_heads:
            K = K.repeat_interleave(self.heads_per_group, dim=1)
            V = V.repeat_interleave(self.heads_per_group, dim=1)

        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = (attn @ V).transpose(1, 2).reshape(B, -1, self.n_heads * self.d_head)
        return self.W_o(out)


# =============================================================================
# SLIDING WINDOW ATTENTION
# =============================================================================

class SlidingWindowAttention(nn.Module):
    """
    Sliding window attention for long sequences.

    Each position only attends to window_size neighbors.
    """

    def __init__(self, config: EfficientAttnConfig):
        super().__init__()
        self.gqa = GroupedQueryAttention(config)
        self.window_size = config.window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        # Create window mask
        positions = torch.arange(L, device=x.device)
        mask = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs() <= self.window_size // 2
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
        return self.gqa(x, mask=mask)

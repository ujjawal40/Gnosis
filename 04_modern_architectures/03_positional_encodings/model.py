"""
Positional Encodings in PyTorch
==================================

PyTorch implementations of all positional encoding methods.
"""

import math
import torch
import torch.nn as nn
from dataclasses import dataclass


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class PosEncConfig:
    d_model: int = 128
    max_len: int = 512
    n_heads: int = 8
    encoding_type: str = "sinusoidal"  # sinusoidal, learned, rope, alibi


# =============================================================================
# SINUSOIDAL
# =============================================================================

class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# =============================================================================
# LEARNED
# =============================================================================

class LearnedPE(nn.Module):
    """Learned absolute positional embeddings (BERT, GPT-2)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pe(positions)


# =============================================================================
# ROTARY POSITION EMBEDDING (RoPE)
# =============================================================================

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (Su et al., 2021).

    Encodes relative position in dot-product attention by rotating
    query and key vectors.
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        return torch.cat([freqs, freqs], dim=-1)


def apply_rotary_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to query or key tensor."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = freqs[..., :d].cos()
    sin = freqs[..., :d].sin()
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# =============================================================================
# ALiBi
# =============================================================================

class ALiBi(nn.Module):
    """
    Attention with Linear Biases (Press et al., 2022).

    No positional embeddings — just adds linear distance bias to attention.
    """

    def __init__(self, n_heads: int):
        super().__init__()
        ratio = 2 ** (-8.0 / n_heads)
        slopes = ratio ** torch.arange(1, n_heads + 1)
        self.register_buffer('slopes', slopes)

    def forward(self, seq_len: int) -> torch.Tensor:
        positions = torch.arange(seq_len)
        distances = -(positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()
        return self.slopes[:, None, None] * distances.unsqueeze(0)


# =============================================================================
# FACTORY
# =============================================================================

def get_positional_encoding(config: PosEncConfig) -> nn.Module:
    """Factory for positional encoding modules."""
    if config.encoding_type == "sinusoidal":
        return SinusoidalPE(config.d_model, config.max_len)
    elif config.encoding_type == "learned":
        return LearnedPE(config.d_model, config.max_len)
    elif config.encoding_type == "rope":
        return RotaryEmbedding(config.d_model // config.n_heads)
    elif config.encoding_type == "alibi":
        return ALiBi(config.n_heads)
    raise ValueError(f"Unknown: {config.encoding_type}")

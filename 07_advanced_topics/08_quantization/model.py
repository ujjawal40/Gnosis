"""
Model Quantization in PyTorch
================================

PyTorch implementations of post-training quantization,
dynamic quantization, and quantization-aware training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class QuantConfig:
    in_features: int = 784
    hidden_dims: list = None
    n_classes: int = 10
    n_bits: int = 8

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


# =============================================================================
# STANDARD MODEL
# =============================================================================

class SimpleMLP(nn.Module):
    """Standard MLP for quantization experiments."""

    def __init__(self, config: QuantConfig):
        super().__init__()
        layers = []
        in_dim = config.in_features
        for dim in config.hidden_dims:
            layers.extend([nn.Linear(in_dim, dim), nn.ReLU()])
            in_dim = dim
        layers.append(nn.Linear(in_dim, config.n_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)


# =============================================================================
# FAKE QUANTIZATION MODULE
# =============================================================================

class FakeQuantize(nn.Module):
    """Fake quantization for QAT — simulates quantization in forward pass."""

    def __init__(self, n_bits: int = 8):
        super().__init__()
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** n_bits - 1
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0))
        self.register_buffer('min_val', torch.tensor(float('inf')))
        self.register_buffer('max_val', torch.tensor(float('-inf')))

    def update_stats(self, x: torch.Tensor):
        """Update running min/max statistics."""
        self.min_val = torch.min(self.min_val, x.detach().min())
        self.max_val = torch.max(self.max_val, x.detach().max())
        self.scale = (self.max_val - self.min_val) / (self.qmax - self.qmin)
        self.scale = torch.clamp(self.scale, min=1e-8)
        self.zero_point = torch.round(-self.min_val / self.scale).clamp(
            self.qmin, self.qmax).to(torch.int32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.update_stats(x)

        # Fake quantize: quantize then immediately dequantize
        q = torch.clamp(
            torch.round(x / self.scale + self.zero_point.float()),
            self.qmin, self.qmax
        )
        return (q - self.zero_point.float()) * self.scale


# =============================================================================
# QUANTIZATION-AWARE TRAINING MODEL
# =============================================================================

class QATLinear(nn.Module):
    """Linear layer with fake quantization for QAT."""

    def __init__(self, in_features: int, out_features: int, n_bits: int = 8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.weight_fq = FakeQuantize(n_bits)
        self.activation_fq = FakeQuantize(n_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation_fq(x)
        # Fake quantize weights
        w_q = self.weight_fq(self.linear.weight)
        return F.linear(x, w_q, self.linear.bias)


class QATMLP(nn.Module):
    """MLP with quantization-aware training support."""

    def __init__(self, config: QuantConfig):
        super().__init__()
        layers = []
        in_dim = config.in_features
        for dim in config.hidden_dims:
            layers.extend([QATLinear(in_dim, dim, config.n_bits), nn.ReLU()])
            in_dim = dim
        layers.append(QATLinear(in_dim, config.n_classes, config.n_bits))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)


# =============================================================================
# POST-TRAINING QUANTIZATION
# =============================================================================

def quantize_model_weights(model: nn.Module, n_bits: int = 8):
    """
    Post-training weight quantization.

    Quantizes all linear layer weights to n-bit integers.
    Returns quantized state dict and scale factors.
    """
    quant_info = {}
    state = model.state_dict()

    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            w = param.data
            qmax = 2 ** (n_bits - 1) - 1
            abs_max = w.abs().max()
            scale = abs_max / qmax if abs_max > 0 else torch.tensor(1.0)
            q = torch.round(w / scale).clamp(-qmax, qmax).to(torch.int8)
            quant_info[name] = {'q_weight': q, 'scale': scale}

    return quant_info


def model_size_bytes(model: nn.Module, bits: int = 32) -> int:
    """Estimate model size in bytes."""
    n_params = sum(p.numel() for p in model.parameters())
    return n_params * bits // 8

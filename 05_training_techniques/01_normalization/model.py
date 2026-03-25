"""
Normalization Layers in PyTorch
=================================

PyTorch models with configurable normalization for comparing
BatchNorm, LayerNorm, GroupNorm, InstanceNorm, and RMSNorm.

Architecture:
    Input → [Linear → Norm → ReLU → Dropout] × N → Linear → Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class NormConfig:
    """Configuration for normalization comparison model."""
    in_features: int = 3072          # 3*32*32 for CIFAR-10 flattened
    n_classes: int = 10
    hidden_dims: List[int] = None
    norm_type: str = "batch"         # batch, layer, group, instance, rms, none
    num_groups: int = 8              # for GroupNorm
    dropout: float = 0.1
    eps: float = 1e-5

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]


# =============================================================================
# CUSTOM RMSNORM
# =============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm does not center the activations (no mean subtraction).
    It only rescales by the RMS statistic:
        x_hat = x / RMS(x)
        RMS(x) = sqrt(mean(x^2) + eps)

    Used in LLaMA, Gemma, and other modern architectures.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


# =============================================================================
# NORM LAYER FACTORY
# =============================================================================

def get_norm_layer(norm_type: str, num_features: int, num_groups: int = 8,
                   eps: float = 1e-5) -> Optional[nn.Module]:
    """Return the appropriate normalization layer."""
    if norm_type == "batch":
        return nn.BatchNorm1d(num_features, eps=eps)
    elif norm_type == "layer":
        return nn.LayerNorm(num_features, eps=eps)
    elif norm_type == "group":
        # Ensure num_groups divides num_features
        g = min(num_groups, num_features)
        while num_features % g != 0:
            g -= 1
        return nn.GroupNorm(g, num_features, eps=eps)
    elif norm_type == "instance":
        return nn.GroupNorm(num_features, num_features, eps=eps)
    elif norm_type == "rms":
        return RMSNorm(num_features, eps=eps)
    elif norm_type == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


# =============================================================================
# MODEL
# =============================================================================

class NormComparisonNet(nn.Module):
    """
    MLP with configurable normalization layers for fair comparison.

    Features:
        - Supports 6 normalization types including custom RMSNorm
        - Identical architecture across all norm types for fair comparison
        - Tracks activation statistics for analysis
    """

    def __init__(self, config: NormConfig):
        super().__init__()
        self.config = config

        layers = []
        in_dim = config.in_features
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(get_norm_layer(config.norm_type, hidden_dim,
                                         config.num_groups, config.eps))
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, config.n_classes)

        self._init_weights()

    def _init_weights(self):
        """He initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten if needed (image input)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.features(x)
        return self.classifier(x)

    def get_activation_stats(self, x: torch.Tensor) -> dict:
        """Track activation statistics through each layer."""
        stats = {}
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        for i, layer in enumerate(self.features):
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                stats[f"layer_{i}_mean"] = x.mean().item()
                stats[f"layer_{i}_std"] = x.std().item()
                stats[f"layer_{i}_dead_frac"] = (x == 0).float().mean().item()
        return stats

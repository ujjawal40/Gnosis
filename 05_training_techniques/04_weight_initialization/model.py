"""
Weight Initialization in PyTorch
===================================

PyTorch utilities for comparing initialization strategies on real networks.
Demonstrates how nn.init functions map to the from-scratch implementations.

Comparison:
    NumPy (implementation.py)      →  PyTorch (this file)
    ─────────────────────────────────────────────────────
    xavier_normal_init             →  nn.init.xavier_normal_
    he_normal_init                 →  nn.init.kaiming_normal_
    orthogonal_init                →  nn.init.orthogonal_
    lecun_normal_init              →  nn.init.normal_(std=sqrt(1/fan_in))
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Callable


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class InitConfig:
    """Configuration for initialization experiments."""
    in_features: int = 784
    hidden_dims: List[int] = None
    n_classes: int = 10
    activation: str = "relu"

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256] * 10  # 10 layers deep


# =============================================================================
# INITIALIZATION FUNCTIONS
# =============================================================================

INIT_METHODS = {
    "zeros": lambda w: nn.init.zeros_(w),
    "random_small": lambda w: nn.init.normal_(w, std=0.01),
    "random_large": lambda w: nn.init.normal_(w, std=1.0),
    "xavier_normal": lambda w: nn.init.xavier_normal_(w),
    "xavier_uniform": lambda w: nn.init.xavier_uniform_(w),
    "kaiming_normal": lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu'),
    "kaiming_uniform": lambda w: nn.init.kaiming_uniform_(w, nonlinearity='relu'),
    "orthogonal": lambda w: nn.init.orthogonal_(w),
}


def apply_init(model: nn.Module, init_name: str):
    """Apply initialization to all Linear layers in model."""
    init_fn = INIT_METHODS[init_name]
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init_fn(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# =============================================================================
# MODEL
# =============================================================================

class DeepMLP(nn.Module):
    """Deep MLP for initialization comparison (no batchnorm to isolate init effect)."""

    def __init__(self, config: InitConfig):
        super().__init__()
        self.config = config

        activation_map = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "selu": nn.SELU(),
        }
        act = activation_map.get(config.activation, nn.ReLU())

        layers = []
        in_dim = config.in_features
        for dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(act)
            in_dim = dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, config.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.classifier(self.features(x))

    @torch.no_grad()
    def get_layer_stats(self, x: torch.Tensor) -> dict:
        """Track activation stats through each layer."""
        stats = {}
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        layer_idx = 0
        for module in self.features:
            x = module(x)
            if isinstance(module, nn.Linear):
                stats[f"layer_{layer_idx}"] = {
                    "mean": x.mean().item(),
                    "std": x.std().item(),
                    "abs_mean": x.abs().mean().item(),
                }
                layer_idx += 1
        return stats

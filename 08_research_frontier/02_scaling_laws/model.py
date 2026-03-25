"""
Scaling Laws in PyTorch
=========================

Run actual scaling experiments: train models of varying sizes
and fit power laws to the results.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class ScalingConfig:
    """Configuration for scaling experiments."""
    in_features: int = 784
    n_classes: int = 10
    model_sizes: List[str] = None   # tiny, small, medium, large

    def __post_init__(self):
        if self.model_sizes is None:
            self.model_sizes = ["tiny", "small", "medium", "large"]


SIZE_CONFIGS = {
    "tiny":   [32],
    "small":  [64, 32],
    "medium": [128, 64, 32],
    "large":  [256, 128, 64, 32],
    "xlarge": [512, 256, 128, 64],
}


# =============================================================================
# MODEL
# =============================================================================

class ScalableMLP(nn.Module):
    """MLP with configurable size for scaling experiments."""

    def __init__(self, in_features: int, hidden_dims: List[int], n_classes: int):
        super().__init__()
        layers = []
        in_dim = in_features
        for dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, dim), nn.ReLU()])
            in_dim = dim
        layers.append(nn.Linear(in_dim, n_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(size: str, in_features: int = 784,
                n_classes: int = 10) -> ScalableMLP:
    """Build model of given size."""
    dims = SIZE_CONFIGS[size]
    return ScalableMLP(in_features, dims, n_classes)

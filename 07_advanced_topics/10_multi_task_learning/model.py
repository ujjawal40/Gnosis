"""
Multi-Task Learning in PyTorch
=================================

PyTorch implementations of hard/soft parameter sharing,
uncertainty weighting, and GradNorm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class MTLConfig:
    in_features: int = 784
    shared_dim: int = 256
    task_hidden_dim: int = 64
    task_outputs: list = None  # Output dims per task

    def __post_init__(self):
        if self.task_outputs is None:
            self.task_outputs = [10, 10]  # Two classification tasks


# =============================================================================
# SHARED BACKBONE
# =============================================================================

class SharedEncoder(nn.Module):
    """Shared feature extractor."""

    def __init__(self, in_features: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


# =============================================================================
# TASK HEADS
# =============================================================================

class TaskHead(nn.Module):
    """Task-specific prediction head."""

    def __init__(self, shared_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h):
        return self.net(h)


# =============================================================================
# HARD PARAMETER SHARING
# =============================================================================

class HardSharingMTL(nn.Module):
    """Multi-task model with hard parameter sharing."""

    def __init__(self, config: MTLConfig):
        super().__init__()
        self.encoder = SharedEncoder(config.in_features, config.shared_dim)
        self.heads = nn.ModuleList([
            TaskHead(config.shared_dim, config.task_hidden_dim, out_dim)
            for out_dim in config.task_outputs
        ])

    def forward(self, x) -> List[torch.Tensor]:
        h = self.encoder(x)
        return [head(h) for head in self.heads]

    def shared_parameters(self):
        return self.encoder.parameters()

    def task_parameters(self, task_idx: int):
        return self.heads[task_idx].parameters()


# =============================================================================
# UNCERTAINTY WEIGHTING
# =============================================================================

class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty Weighting (Kendall et al., 2018).

    Learns task weights via homoscedastic uncertainty.
    """

    def __init__(self, n_tasks: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        total = 0.0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]
        return total

    def get_weights(self) -> torch.Tensor:
        return torch.exp(-self.log_vars.data)


# =============================================================================
# GRADNORM
# =============================================================================

class GradNormWeights(nn.Module):
    """Learnable task weights for GradNorm."""

    def __init__(self, n_tasks: int, alpha: float = 1.5):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_tasks))
        self.alpha = alpha
        self.initial_losses = None

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        return sum(w * l for w, l in zip(self.weights, losses))

    def normalize(self):
        """Renormalize weights to sum to n_tasks."""
        with torch.no_grad():
            self.weights.data = self.weights.data * len(self.weights) / self.weights.data.sum()


# =============================================================================
# SINGLE-TASK BASELINE
# =============================================================================

class SingleTaskModel(nn.Module):
    """Single-task model for comparison."""

    def __init__(self, in_features: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)

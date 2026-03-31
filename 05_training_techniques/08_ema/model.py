"""
EMA & Model Averaging in PyTorch
====================================

PyTorch implementations of EMA, SWA, and model soup.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class EMAConfig:
    in_features: int = 784
    hidden_dims: list = None
    n_classes: int = 10
    ema_decay: float = 0.999

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


# =============================================================================
# MODEL
# =============================================================================

class SimpleMLP(nn.Module):
    def __init__(self, config: EMAConfig):
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
# EMA
# =============================================================================

class ModelEMA:
    """
    Exponential Moving Average of model parameters.

    Maintains a shadow copy updated as:
        shadow = decay * shadow + (1 - decay) * model
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.n_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.n_updates += 1
        decay = min(self.decay, (1 + self.n_updates) / (10 + self.n_updates))

        for s_param, m_param in zip(self.shadow.parameters(),
                                     model.parameters()):
            s_param.data.mul_(decay).add_(m_param.data, alpha=1 - decay)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shadow(x)


# =============================================================================
# SWA
# =============================================================================

class SWAModel:
    """
    Stochastic Weight Averaging.

    Averages model weights from multiple checkpoints.
    """

    def __init__(self, model: nn.Module):
        self.averaged = copy.deepcopy(model)
        self.averaged.eval()
        self.n_models = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        for a_param, m_param in zip(self.averaged.parameters(),
                                     model.parameters()):
            a_param.data.mul_(self.n_models).add_(m_param.data).div_(
                self.n_models + 1
            )
        self.n_models += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.averaged(x)


# =============================================================================
# MODEL SOUP
# =============================================================================

@torch.no_grad()
def uniform_soup(models: list) -> nn.Module:
    """Average weights of multiple models."""
    soup = copy.deepcopy(models[0])
    for param in soup.parameters():
        param.data.zero_()

    for model in models:
        for s_param, m_param in zip(soup.parameters(), model.parameters()):
            s_param.data.add_(m_param.data / len(models))

    return soup

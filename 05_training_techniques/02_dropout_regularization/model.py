"""
Regularized MLP in PyTorch
============================

PyTorch models with configurable regularization: Dropout, DropConnect,
L1/L2 penalties, elastic net, and max-norm constraints.

Architecture:
    Input → [Linear → BN → ReLU → Dropout/DropConnect] × N → Linear → Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class RegConfig:
    """Configuration for regularized model."""
    in_features: int = 784           # 28*28 for Fashion-MNIST
    n_classes: int = 10
    hidden_dims: List[int] = None
    dropout_rate: float = 0.0
    dropconnect_rate: float = 0.0
    l1_lambda: float = 0.0
    l2_lambda: float = 0.0
    max_norm: float = 0.0
    use_batch_norm: bool = True

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]


# =============================================================================
# CUSTOM DROPCONNECT LAYER
# =============================================================================

class DropConnect(nn.Module):
    """
    DropConnect: randomly zero out weights instead of activations.

    More general than Dropout - masks the weight matrix during forward pass.
    During inference, uses full weights (like dropout with inverted scaling).
    """

    def __init__(self, in_features: int, out_features: int, p: float = 0.5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0:
            mask = torch.bernoulli(
                torch.full_like(self.linear.weight, 1.0 - self.p)
            )
            weight = self.linear.weight * mask / (1.0 - self.p)
            return F.linear(x, weight, self.linear.bias)
        return self.linear(x)


# =============================================================================
# REGULARIZATION HELPERS
# =============================================================================

def compute_l1_penalty(model: nn.Module) -> torch.Tensor:
    """L1 penalty: sum of absolute values of all parameters."""
    l1 = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        l1 = l1 + p.abs().sum()
    return l1


def compute_l2_penalty(model: nn.Module) -> torch.Tensor:
    """L2 penalty: sum of squared values of all parameters."""
    l2 = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        l2 = l2 + (p ** 2).sum()
    return l2


def apply_max_norm(model: nn.Module, max_val: float = 3.0):
    """Clip weight norms after each update."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                norms = param.norm(dim=1, keepdim=True)
                desired = torch.clamp(norms, max=max_val)
                param.mul_(desired / (norms + 1e-8))


# =============================================================================
# MODEL
# =============================================================================

class RegularizedMLP(nn.Module):
    """
    MLP with configurable regularization for comparison experiments.

    Supports:
        - Standard Dropout
        - DropConnect
        - L1/L2/Elastic Net (computed externally, added to loss)
        - Max-norm constraint (applied after optimizer step)
        - Batch normalization
    """

    def __init__(self, config: RegConfig):
        super().__init__()
        self.config = config

        layers = []
        in_dim = config.in_features
        for hidden_dim in config.hidden_dims:
            if config.dropconnect_rate > 0:
                layers.append(DropConnect(in_dim, hidden_dim, config.dropconnect_rate))
            else:
                layers.append(nn.Linear(in_dim, hidden_dim))

            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())

            if config.dropout_rate > 0:
                layers.append(nn.Dropout(config.dropout_rate))

            in_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, config.n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.features(x)
        return self.classifier(x)

    def regularization_loss(self) -> torch.Tensor:
        """Compute regularization penalty to add to task loss."""
        reg = torch.tensor(0.0, device=next(self.parameters()).device)
        if self.config.l1_lambda > 0:
            reg = reg + self.config.l1_lambda * compute_l1_penalty(self)
        if self.config.l2_lambda > 0:
            reg = reg + self.config.l2_lambda * compute_l2_penalty(self)
        return reg

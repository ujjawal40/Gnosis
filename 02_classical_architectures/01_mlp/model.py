"""
Deep MLP for Tabular Data
===========================

Production-grade MLP with residual connections, batch normalization,
and configurable architecture for large-scale tabular classification.

Architecture:
    Input → [Linear → BN → Activation → Dropout (+ Residual)] × N → Linear → Output
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MLPConfig:
    """Configuration for the MLP model."""
    input_dim: int = 28
    hidden_dims: List[int] = None
    output_dim: int = 2
    activation: str = "relu"
    dropout: float = 0.1
    batch_norm: bool = True
    residual: bool = True
    weight_init: str = "kaiming"  # "kaiming", "xavier", "normal"

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128, 64]


class ResidualBlock(nn.Module):
    """Linear block with optional residual connection."""

    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module,
                 dropout: float = 0.1, batch_norm: bool = True,
                 residual: bool = True):
        super().__init__()
        self.residual = residual and (in_dim == out_dim)

        layers = [nn.Linear(in_dim, out_dim)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.residual:
            out = out + x
        return out


class DeepMLP(nn.Module):
    """
    Deep MLP with residual connections for tabular data.

    Features:
        - Configurable depth and width
        - Residual connections (when dimensions match)
        - Batch normalization
        - Dropout regularization
        - Flexible weight initialization
    """

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config

        activation_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "elu": nn.ELU(),
        }
        act = activation_map.get(config.activation, nn.ReLU())

        # Input projection
        layers = [nn.Linear(config.input_dim, config.hidden_dims[0])]
        if config.batch_norm:
            layers.append(nn.BatchNorm1d(config.hidden_dims[0]))
        layers.append(type(act)())
        if config.dropout > 0:
            layers.append(nn.Dropout(config.dropout))

        # Hidden blocks
        for i in range(len(config.hidden_dims) - 1):
            layers.append(ResidualBlock(
                config.hidden_dims[i], config.hidden_dims[i + 1],
                activation=type(act)(),
                dropout=config.dropout,
                batch_norm=config.batch_norm,
                residual=config.residual,
            ))

        # Output head
        layers.append(nn.Linear(config.hidden_dims[-1], config.output_dim))

        self.network = nn.Sequential(*layers)
        self._init_weights(config.weight_init)

    def _init_weights(self, method: str):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                elif method == "xavier":
                    nn.init.xavier_normal_(m.weight)
                elif method == "normal":
                    nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

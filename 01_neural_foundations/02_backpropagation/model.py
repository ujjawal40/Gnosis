"""
Backpropagation & MLP in PyTorch
==================================

PyTorch re-implementation of the backprop module from implementation.py.
Builds an MLP using nn.Module with gradient tracking and visualization.

Architecture:
    Input → [Linear → ReLU → Dropout] × N → Linear → Output

Comparison to NumPy version:
    Value class autograd → torch.autograd (built-in)
    Manual neuron/layer  → nn.Linear
    Manual backward()    → loss.backward()
    Manual zero_grad     → optimizer.zero_grad()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with configurable architecture.

    Args:
        layer_sizes: List of layer dimensions [input, hidden1, ..., output]
        activation: Activation function ('relu', 'tanh', 'sigmoid', 'gelu')
        dropout: Dropout rate between layers
        batch_norm: Whether to use batch normalization
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        super().__init__()

        self.layer_sizes = layer_sizes
        activation_fn = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "gelu": nn.GELU,
        }[activation]

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            # No activation/dropout/batchnorm on last layer
            if i < len(layer_sizes) - 2:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                layers.append(activation_fn())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """He initialization for ReLU-like activations."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_gradient_norms(self) -> dict:
        """Get per-layer gradient norms for gradient flow analysis."""
        norms = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                norms[name] = param.grad.norm().item()
        return norms


class XORNet(nn.Module):
    """
    Minimal 2-layer network for XOR.
    Demonstrates that depth is needed for non-linear decision boundaries.
    """

    def __init__(self, hidden_size: int = 4):
        super().__init__()
        self.layer1 = nn.Linear(2, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.layer1(x))
        return torch.sigmoid(self.layer2(h))

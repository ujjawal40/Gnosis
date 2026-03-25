"""
Gradient Clipping in PyTorch
================================

PyTorch implementations of gradient clipping methods
and gradient accumulation for effective large batch training.
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
class GradClipConfig:
    in_features: int = 784
    hidden_dims: list = None
    n_classes: int = 10
    clip_method: str = "norm"  # norm, value, agc, none
    clip_value: float = 1.0
    agc_factor: float = 0.01

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128, 64]


# =============================================================================
# DEEP MODEL (prone to gradient issues)
# =============================================================================

class DeepMLP(nn.Module):
    """Deep MLP where gradient clipping matters."""

    def __init__(self, config: GradClipConfig):
        super().__init__()
        layers = []
        in_dim = config.in_features
        for dim in config.hidden_dims:
            layers.extend([nn.Linear(in_dim, dim), nn.ReLU()])
            in_dim = dim
        layers.append(nn.Linear(in_dim, config.n_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# GRADIENT CLIPPING UTILITIES
# =============================================================================

def clip_gradients(model: nn.Module, method: str = "norm",
                   clip_value: float = 1.0, agc_factor: float = 0.01):
    """
    Apply gradient clipping to model parameters.

    Methods:
        - norm: Clip by global gradient norm (most common)
        - value: Clip each gradient element to [-clip_value, clip_value]
        - agc: Adaptive Gradient Clipping (NFNet)
    """
    if method == "norm":
        return torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    elif method == "value":
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
        return None
    elif method == "agc":
        return adaptive_gradient_clipping(model, agc_factor)
    return None


def adaptive_gradient_clipping(model: nn.Module, clip_factor: float = 0.01,
                                eps: float = 1e-3) -> float:
    """
    Adaptive Gradient Clipping (AGC) from NFNet.

    Clips based on ratio of gradient norm to parameter norm.
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        p_norm = param.data.norm().clamp(min=eps)
        g_norm = param.grad.data.norm().clamp(min=eps)
        max_norm = clip_factor * p_norm
        if g_norm > max_norm:
            param.grad.data.mul_(max_norm / g_norm)
        total_norm += param.grad.data.norm().item() ** 2
    return total_norm ** 0.5


# =============================================================================
# GRADIENT ACCUMULATOR
# =============================================================================

class GradientAccumulator:
    """
    Gradient accumulation for effective large batch training.

    Usage:
        accum = GradientAccumulator(optimizer, accumulation_steps=4)
        for X, y in loader:
            loss = criterion(model(X), y) / accum.steps
            loss.backward()
            if accum.step():  # True when it's time to update
                pass  # optimizer step happened inside
    """

    def __init__(self, optimizer: torch.optim.Optimizer,
                 accumulation_steps: int = 4,
                 clip_fn=None, clip_kwargs=None):
        self.optimizer = optimizer
        self.steps = accumulation_steps
        self.clip_fn = clip_fn
        self.clip_kwargs = clip_kwargs or {}
        self.count = 0

    def step(self) -> bool:
        """Increment counter. Returns True and updates if accumulated enough."""
        self.count += 1
        if self.count >= self.steps:
            if self.clip_fn:
                self.clip_fn(**self.clip_kwargs)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.count = 0
            return True
        return False

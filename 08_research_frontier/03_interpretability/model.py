"""
Neural Network Interpretability in PyTorch
=============================================

PyTorch implementations of gradient-based and perturbation-based
interpretability methods.
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
class InterpConfig:
    in_features: int = 784
    hidden_dims: list = None
    n_classes: int = 10

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]


# =============================================================================
# MODEL WITH HOOK SUPPORT
# =============================================================================

class InterpretableModel(nn.Module):
    """MLP with hooks for extracting intermediate activations and gradients."""

    def __init__(self, config: InterpConfig):
        super().__init__()
        layers = []
        in_dim = config.in_features
        for dim in config.hidden_dims:
            layers.extend([nn.Linear(in_dim, dim), nn.ReLU()])
            in_dim = dim
        layers.append(nn.Linear(in_dim, config.n_classes))
        self.model = nn.Sequential(*layers)
        self.activations = {}
        self.gradients = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)


# =============================================================================
# INTERPRETABILITY METHODS
# =============================================================================

def vanilla_saliency(model: nn.Module, x: torch.Tensor,
                     target_class: int) -> torch.Tensor:
    """Vanilla gradient saliency map."""
    x = x.detach().requires_grad_(True)
    output = model(x)
    output[0, target_class].backward()
    return x.grad.abs().detach()


def smooth_grad(model: nn.Module, x: torch.Tensor, target_class: int,
                n_samples: int = 50, noise_std: float = 0.1) -> torch.Tensor:
    """SmoothGrad: averaged noisy gradients."""
    saliency = torch.zeros_like(x)
    for _ in range(n_samples):
        noisy = (x + torch.randn_like(x) * noise_std).detach().requires_grad_(True)
        output = model(noisy)
        output[0, target_class].backward()
        saliency += noisy.grad.abs()
    return saliency / n_samples


def integrated_gradients(model: nn.Module, x: torch.Tensor,
                         target_class: int, baseline: Optional[torch.Tensor] = None,
                         n_steps: int = 50) -> torch.Tensor:
    """Integrated Gradients attribution."""
    if baseline is None:
        baseline = torch.zeros_like(x)

    grads = torch.zeros_like(x)
    for alpha in torch.linspace(0, 1, n_steps + 1):
        interp = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        output = model(interp)
        output[0, target_class].backward()
        grads += interp.grad

    return ((x - baseline) * grads / (n_steps + 1)).detach()


def occlusion_sensitivity(model: nn.Module, x: torch.Tensor,
                           target_class: int) -> torch.Tensor:
    """Occlusion-based feature importance."""
    with torch.no_grad():
        base_prob = F.softmax(model(x), dim=-1)[0, target_class].item()

    importance = torch.zeros(x.shape[-1])
    with torch.no_grad():
        for i in range(x.shape[-1]):
            occluded = x.clone()
            occluded[..., i] = 0
            new_prob = F.softmax(model(occluded), dim=-1)[0, target_class].item()
            importance[i] = base_prob - new_prob

    return importance

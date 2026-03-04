"""
Activation Functions in PyTorch
================================

All major activation functions with analysis tools.

Covered activations:
    ReLU, LeakyReLU, ELU, SELU, GELU, SiLU/Swish, Mish,
    Sigmoid, Tanh, Softmax, Softplus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Callable


# =============================================================================
# ACTIVATION REGISTRY
# =============================================================================

ACTIVATIONS: Dict[str, Callable] = {
    "relu":       nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.01),
    "elu":        nn.ELU,
    "selu":       nn.SELU,
    "gelu":       nn.GELU,
    "silu":       nn.SiLU,  # Swish
    "mish":       nn.Mish,
    "sigmoid":    nn.Sigmoid,
    "tanh":       nn.Tanh,
    "softplus":   nn.Softplus,
}


def get_activation(name: str) -> nn.Module:
    """Get activation by name."""
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]()


# =============================================================================
# ANALYSIS TOOLS
# =============================================================================

def activation_profile(name: str, x_range=(-5, 5), n_points=1000):
    """
    Compute activation values and gradients over a range.

    Returns:
        x: input values
        y: activation output
        dy: gradient of activation w.r.t. input
    """
    act = get_activation(name)
    x = torch.linspace(x_range[0], x_range[1], n_points, requires_grad=True)
    y = act(x)
    y.sum().backward()
    return x.detach(), y.detach(), x.grad.detach()


def dead_neuron_rate(model: nn.Module, x: torch.Tensor,
                     threshold: float = 1e-6) -> float:
    """
    Measure fraction of ReLU neurons that are always zero (dead neurons).
    Only meaningful for ReLU-based networks.
    """
    activations = []

    def hook_fn(module, input, output):
        if isinstance(module, nn.ReLU):
            activations.append((output <= threshold).float().mean().item())

    hooks = []
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    return sum(activations) / len(activations) if activations else 0.0


# =============================================================================
# CLASSIFIER WITH CONFIGURABLE ACTIVATION
# =============================================================================

class ActivationTestNet(nn.Module):
    """
    MLP for comparing activation functions on the same architecture.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer width
        n_classes: Number of output classes
        n_layers: Number of hidden layers
        activation: Name of activation function
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int,
                 n_layers: int = 3, activation: str = "relu"):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim), get_activation(activation)]
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                get_activation(activation),
            ])
        layers.append(nn.Linear(hidden_dim, n_classes))

        self.network = nn.Sequential(*layers)
        self._init_weights(activation)

    def _init_weights(self, activation: str):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if activation in ("relu", "leaky_relu", "elu", "gelu", "silu", "mish"):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                elif activation == "selu":
                    nn.init.normal_(m.weight, std=1.0 / m.weight.shape[1] ** 0.5)
                else:
                    nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

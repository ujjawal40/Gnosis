"""
Model Pruning in PyTorch
===========================

PyTorch implementations of magnitude, structured, and iterative pruning.
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
class PruneConfig:
    in_features: int = 784
    hidden_dims: list = None
    n_classes: int = 10
    sparsity: float = 0.5

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


# =============================================================================
# PRUNABLE MLP
# =============================================================================

class PrunableMLP(nn.Module):
    """MLP with pruning mask support."""

    def __init__(self, config: PruneConfig):
        super().__init__()
        layers = []
        in_dim = config.in_features
        for dim in config.hidden_dims:
            layers.extend([nn.Linear(in_dim, dim), nn.ReLU()])
            in_dim = dim
        layers.append(nn.Linear(in_dim, config.n_classes))
        self.model = nn.Sequential(*layers)
        self.masks: Dict[str, torch.Tensor] = {}

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        # Apply masks during forward
        self._apply_masks()
        return self.model(x)

    def _apply_masks(self):
        for name, param in self.named_parameters():
            if name in self.masks:
                param.data.mul_(self.masks[name])

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def active_params(self):
        total = 0
        for name, p in self.named_parameters():
            if name in self.masks:
                total += self.masks[name].sum().item()
            else:
                total += p.numel()
        return int(total)


# =============================================================================
# PRUNING METHODS
# =============================================================================

def magnitude_prune(model: PrunableMLP, sparsity: float):
    """Global unstructured magnitude pruning."""
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            all_weights.append(param.data.abs().flatten())

    all_weights = torch.cat(all_weights)
    threshold = torch.quantile(all_weights, sparsity)

    for name, param in model.named_parameters():
        if 'weight' in name:
            model.masks[name] = (param.data.abs() > threshold).float()


def structured_prune(model: PrunableMLP, sparsity: float):
    """Structured pruning: remove entire neurons by L2 norm."""
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2:
            norms = param.data.norm(dim=1)
            n_prune = int(len(norms) * sparsity)
            threshold = torch.sort(norms)[0][n_prune] if n_prune > 0 else 0
            channel_mask = (norms > threshold).float()
            model.masks[name] = channel_mask.unsqueeze(1).expand_as(param)


def iterative_magnitude_prune(model: PrunableMLP, initial_state: dict,
                               sparsity_per_round: float):
    """One round of IMP: prune + rewind to initial weights."""
    magnitude_prune(model, sparsity_per_round)

    # Rewind weights but keep masks
    masks = {k: v.clone() for k, v in model.masks.items()}
    model.load_state_dict(initial_state)
    model.masks = masks
    model._apply_masks()


# =============================================================================
# SPARSITY ANALYSIS
# =============================================================================

def analyze_sparsity(model: PrunableMLP) -> dict:
    """Analyze per-layer and total sparsity."""
    info = {}
    total_params, total_zeros = 0, 0

    for name, param in model.named_parameters():
        if name in model.masks:
            zeros = (model.masks[name] == 0).sum().item()
            total = param.numel()
            info[name] = {"sparsity": zeros / total, "total": total, "zeros": zeros}
            total_zeros += zeros
            total_params += total
        else:
            total_params += param.numel()

    info["total"] = {"sparsity": total_zeros / total_params if total_params > 0 else 0,
                     "total": total_params, "zeros": total_zeros}
    return info

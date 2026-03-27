"""
Label Smoothing & Advanced Losses in PyTorch
================================================

PyTorch implementations of label smoothing, focal loss,
and confidence penalty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class LossConfig:
    in_features: int = 784
    hidden_dims: list = None
    n_classes: int = 10
    label_smoothing: float = 0.1
    focal_gamma: float = 2.0

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


# =============================================================================
# MODEL
# =============================================================================

class SimpleMLP(nn.Module):
    def __init__(self, config: LossConfig):
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
# LOSSES
# =============================================================================

class LabelSmoothingCE(nn.Module):
    """Cross-entropy with label smoothing."""

    def __init__(self, n_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.n_classes = n_classes
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / self.n_classes)
            smooth_targets.scatter_(1, targets.unsqueeze(1),
                                     1 - self.smoothing + self.smoothing / self.n_classes)
        return -(smooth_targets * log_probs).sum(dim=-1).mean()


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, gamma: float = 2.0, alpha: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        ce = F.cross_entropy(logits, targets, reduction='none')
        p = torch.exp(-ce)  # Probability of correct class
        focal_weight = (1 - p) ** self.gamma
        return (self.alpha * focal_weight * ce).mean()


class ConfidencePenaltyCE(nn.Module):
    """Cross-entropy with confidence penalty (entropy regularization)."""

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        ce = F.cross_entropy(logits, targets)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        return ce - self.beta * entropy


class SymmetricCE(nn.Module):
    """Symmetric cross-entropy for noisy label robustness."""

    def __init__(self, n_classes: int, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.n_classes = n_classes
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = F.softmax(logits, dim=-1)
        one_hot = F.one_hot(targets, self.n_classes).float()

        ce = -(one_hot * torch.log(probs + 1e-8)).sum(-1).mean()
        rce = -(probs * torch.log(one_hot + 1e-4)).sum(-1).mean()

        return self.alpha * ce + self.beta * rce

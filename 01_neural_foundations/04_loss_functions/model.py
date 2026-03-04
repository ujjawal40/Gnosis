"""
Loss Functions in PyTorch
==========================

All major loss functions with gradient analysis and custom implementations.

Covered:
    Classification: CrossEntropy, BinaryCE, NLL, Focal, Hinge
    Regression: MSE, MAE, Huber, LogCosh
    Distribution: KL Divergence
    Custom: Label Smoothing CE, Weighted CE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# CUSTOM LOSS FUNCTIONS
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017): addresses class imbalance by
    down-weighting easy examples and focusing on hard ones.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    When gamma=0, this is standard cross-entropy.
    When gamma>0, easy examples (high p_t) contribute less.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


class LabelSmoothingCE(nn.Module):
    """
    Cross-entropy with label smoothing.
    Instead of one-hot targets, uses (1-ε)*one_hot + ε/K for K classes.
    Prevents overconfident predictions.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, label_smoothing=self.smoothing)


class LogCoshLoss(nn.Module):
    """
    Log-cosh loss: smooth approximation of Huber loss.
    log(cosh(x)) ≈ x²/2 for small x, |x| - log(2) for large x.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.log(torch.cosh(diff)))


# =============================================================================
# LOSS ANALYSIS TOOLS
# =============================================================================

def gradient_analysis(loss_fn, logits: torch.Tensor,
                      targets: torch.Tensor) -> torch.Tensor:
    """Compute gradient of loss w.r.t. logits."""
    logits = logits.clone().detach().requires_grad_(True)
    loss = loss_fn(logits, targets)
    loss.backward()
    return logits.grad.clone()


def loss_landscape(loss_fn, target_class: int, n_classes: int,
                   n_points: int = 100):
    """
    Sweep the predicted probability for target_class from 0 to 1
    and compute the loss at each point. Shows how the loss surface looks.
    """
    probs = torch.linspace(0.01, 0.99, n_points)
    losses = []

    for p in probs:
        # Build logits that give probability p for target class
        logit_target = torch.log(p / (1 - p))
        logits = torch.zeros(1, n_classes)
        logits[0, target_class] = logit_target
        target = torch.tensor([target_class])

        loss = loss_fn(logits, target)
        losses.append(loss.item())

    return probs, torch.tensor(losses)


# =============================================================================
# CLASSIFIER FOR LOSS COMPARISON
# =============================================================================

class LossComparisonNet(nn.Module):
    """Simple MLP for comparing loss functions on the same architecture."""

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

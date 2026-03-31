"""
Learning Rate Finder in PyTorch
==================================

PyTorch implementations of LR Range Test, Cyclical LR,
and 1cycle Policy with proper training integration.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class LRFinderConfig:
    in_features: int = 784
    hidden_dims: list = None
    n_classes: int = 10
    min_lr: float = 1e-7
    max_lr: float = 10.0
    num_test_steps: int = 100

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


# =============================================================================
# MODEL
# =============================================================================

class SimpleMLP(nn.Module):
    """Simple MLP for LR finder experiments."""

    def __init__(self, config: LRFinderConfig):
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


# =============================================================================
# LR RANGE TEST
# =============================================================================

class LRRangeTest:
    """
    Learning Rate Range Test (Smith, 2017).

    Exponentially increases LR from min_lr to max_lr over num_steps,
    recording loss at each step. The suggested LR is at the point of
    steepest descent divided by 10.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 criterion, min_lr: float = 1e-7, max_lr: float = 10.0,
                 num_steps: int = 100, smooth_factor: float = 0.05):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps
        self.smooth_factor = smooth_factor

        self.lrs: List[float] = []
        self.losses: List[float] = []
        self.smoothed_losses: List[float] = []
        self._initial_state = {k: v.clone() for k, v in model.state_dict().items()}

    def _get_lr_schedule(self):
        """Exponential schedule from min to max."""
        return torch.exp(torch.linspace(
            math.log(self.min_lr), math.log(self.max_lr), self.num_steps
        )).tolist()

    def run(self, train_loader, device: torch.device):
        """Run the LR range test."""
        lr_schedule = self._get_lr_schedule()
        data_iter = iter(train_loader)

        self.model.train()
        for step, lr in enumerate(lr_schedule):
            # Set LR
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr

            # Get batch (cycle if needed)
            try:
                X, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                X, y = next(data_iter)

            X, y = X.to(device), y.to(device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(X), y)
            loss.backward()
            self.optimizer.step()

            loss_val = loss.item()
            self.lrs.append(lr)
            self.losses.append(loss_val)

            # Smooth
            if len(self.smoothed_losses) == 0:
                self.smoothed_losses.append(loss_val)
            else:
                s = self.smooth_factor * loss_val + \
                    (1 - self.smooth_factor) * self.smoothed_losses[-1]
                self.smoothed_losses.append(s)

            # Check divergence
            if len(self.smoothed_losses) > 1:
                min_loss = min(self.smoothed_losses)
                if self.smoothed_losses[-1] > min_loss * 4:
                    break

        # Restore model
        self.model.load_state_dict(self._initial_state)

    def suggest_lr(self) -> float:
        """Suggest optimal LR (steepest descent / 10)."""
        if len(self.smoothed_losses) < 3:
            return self.min_lr
        losses = torch.tensor(self.smoothed_losses)
        grads = losses[1:] - losses[:-1]
        min_idx = grads.argmin().item()
        return self.lrs[min_idx] / 10


# =============================================================================
# CYCLICAL LR SCHEDULER
# =============================================================================

class CyclicalLRScheduler:
    """
    Cyclical Learning Rate scheduler (Smith, 2017).

    Wraps a PyTorch optimizer and cycles LR between base and max.
    """

    def __init__(self, optimizer: torch.optim.Optimizer,
                 base_lr: float = 1e-4, max_lr: float = 1e-2,
                 step_size: int = 2000, mode: str = "triangular"):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.step_count = 0

    def step(self):
        cycle = 1 + self.step_count // (2 * self.step_size)
        x = abs(self.step_count / self.step_size - 2 * cycle + 1)

        if self.mode == "triangular":
            scale = 1.0
        elif self.mode == "triangular2":
            scale = 1.0 / (2.0 ** (cycle - 1))
        else:
            scale = 1.0

        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x) * scale

        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

        self.step_count += 1

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

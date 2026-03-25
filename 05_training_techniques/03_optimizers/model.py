"""
Optimizer and Scheduler Utilities for PyTorch
================================================

Factory functions for building optimizers and LR schedulers,
plus a benchmark model for fair optimizer comparison.

Supports: SGD, SGD+Momentum, Nesterov, Adam, AdamW, RMSProp
Schedulers: StepLR, CosineAnnealing, WarmupCosine, OneCycleLR
"""

import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    optimizer_name: str = "adam"
    lr: float = 0.001
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    nesterov: bool = False
    scheduler_name: str = "cosine"
    warmup_epochs: int = 5
    step_size: int = 10
    gamma: float = 0.1


@dataclass
class BenchmarkConfig:
    """Benchmark model configuration."""
    in_features: int = 3072          # 3*32*32 for CIFAR-10
    n_classes: int = 10
    hidden_dims: List[int] = None

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]


# =============================================================================
# WARMUP COSINE SCHEDULER
# =============================================================================

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup followed by cosine annealing.

    LR schedule:
        warmup: lr * step / warmup_steps
        cosine: lr_min + 0.5 * (lr - lr_min) * (1 + cos(pi * t / T))
    """

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 min_lr: float = 1e-6, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs)
            return [self.min_lr + 0.5 * (base_lr - self.min_lr) *
                    (1 + math.cos(math.pi * progress))
                    for base_lr in self.base_lrs]


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def build_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    params = model.parameters()
    name = config.optimizer_name.lower()

    if name == "sgd":
        return torch.optim.SGD(params, lr=config.lr,
                                weight_decay=config.weight_decay)
    elif name == "sgd_momentum":
        return torch.optim.SGD(params, lr=config.lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)
    elif name == "nesterov":
        return torch.optim.SGD(params, lr=config.lr, momentum=config.momentum,
                                nesterov=True, weight_decay=config.weight_decay)
    elif name == "adam":
        return torch.optim.Adam(params, lr=config.lr,
                                 betas=(config.beta1, config.beta2),
                                 weight_decay=config.weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=config.lr,
                                  betas=(config.beta1, config.beta2),
                                  weight_decay=config.weight_decay)
    elif name == "rmsprop":
        return torch.optim.RMSprop(params, lr=config.lr,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: OptimizerConfig,
                    total_epochs: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Build LR scheduler from config."""
    name = config.scheduler_name.lower()

    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=config.gamma)
    elif name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs)
    elif name == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer, config.warmup_epochs, total_epochs)
    elif name == "onecycle":
        # OneCycleLR needs steps_per_epoch, use epochs as approximation
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.lr * 10, total_steps=total_epochs)
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {name}")


# =============================================================================
# BENCHMARK MODEL
# =============================================================================

class BenchmarkMLP(nn.Module):
    """
    Fixed architecture MLP for fair optimizer comparison.

    All experiments use identical architecture; only the optimizer changes.
    """

    def __init__(self, config: BenchmarkConfig):
        super().__init__()
        self.config = config

        layers = []
        in_dim = config.in_features
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
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
        return self.classifier(self.features(x))

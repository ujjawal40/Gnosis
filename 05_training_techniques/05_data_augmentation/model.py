"""
Data Augmentation in PyTorch
===============================

PyTorch augmentation transforms and training utilities for comparing
augmentation strategies on image classification.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List
from torchvision import transforms


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class AugConfig:
    """Augmentation experiment configuration."""
    strategy: str = "standard"       # none, standard, strong, autoaugment
    in_channels: int = 3
    n_classes: int = 10
    cutout_size: int = 8
    mixup_alpha: float = 0.2


# =============================================================================
# AUGMENTATION STRATEGIES
# =============================================================================

def get_transforms(strategy: str, train: bool = True):
    """Return transforms for a given augmentation strategy."""
    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))

    if not train:
        return transforms.Compose([transforms.ToTensor(), normalize])

    if strategy == "none":
        return transforms.Compose([transforms.ToTensor(), normalize])

    elif strategy == "standard":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])

    elif strategy == "strong":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        ])

    elif strategy == "autoaugment":
        return transforms.Compose([
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            normalize,
        ])

    raise ValueError(f"Unknown strategy: {strategy}")


# =============================================================================
# MODEL (simple CNN for fair comparison)
# =============================================================================

class SimpleCNN(nn.Module):
    """Small CNN for augmentation comparison."""

    def __init__(self, config: AugConfig):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(config.in_channels, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, config.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return self.classifier(x)

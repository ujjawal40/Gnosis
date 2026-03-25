"""
Transfer Learning in PyTorch
===============================

Feature extraction and fine-tuning with discriminative learning rates.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TransferConfig:
    backbone_dims: List[int] = None
    n_classes: int = 10
    freeze_backbone: bool = False
    discriminative_lr: bool = False
    base_lr: float = 0.001
    lr_decay: float = 0.3

    def __post_init__(self):
        if self.backbone_dims is None:
            self.backbone_dims = [784, 512, 256, 128]


# =============================================================================
# MODEL
# =============================================================================

class TransferModel(nn.Module):
    """
    Transfer learning model with freezable backbone.

    Supports:
        - Feature extraction (frozen backbone)
        - Fine-tuning (unfrozen)
        - Gradual unfreezing
        - Discriminative learning rates
    """

    def __init__(self, config: TransferConfig):
        super().__init__()
        self.config = config

        # Build backbone
        backbone_layers = []
        for i in range(len(config.backbone_dims) - 1):
            backbone_layers.extend([
                nn.Linear(config.backbone_dims[i], config.backbone_dims[i + 1]),
                nn.BatchNorm1d(config.backbone_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
        self.backbone = nn.Sequential(*backbone_layers)

        # New classification head
        self.head = nn.Linear(config.backbone_dims[-1], config.n_classes)

        if config.freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_from_layer(self, layer_idx: int):
        """Gradually unfreeze: only unfreeze layers >= layer_idx."""
        linear_idx = 0
        for module in self.backbone:
            if isinstance(module, nn.Linear):
                if linear_idx >= layer_idx:
                    module.weight.requires_grad = True
                    module.bias.requires_grad = True
                linear_idx += 1

    def get_param_groups(self) -> list:
        """Parameter groups with discriminative learning rates."""
        groups = []
        n_layers = sum(1 for m in self.backbone if isinstance(m, nn.Linear))

        layer_idx = 0
        for module in self.backbone:
            if isinstance(module, (nn.Linear, nn.BatchNorm1d)):
                lr_mult = self.config.lr_decay ** (n_layers - layer_idx)
                groups.append({
                    'params': module.parameters(),
                    'lr': self.config.base_lr * lr_mult,
                })
                if isinstance(module, nn.Linear):
                    layer_idx += 1

        groups.append({'params': self.head.parameters(), 'lr': self.config.base_lr})
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        features = self.backbone(x)
        return self.head(features)

"""
ResNet and Skip Connection Architectures in PyTorch
======================================================

Full ResNet-18/34/50 with BasicBlock and Bottleneck, plus
DenseNet-style blocks and Squeeze-and-Excitation enhancement.

Architectures:
    ResNet-18:  [2, 2, 2, 2] BasicBlocks
    ResNet-34:  [3, 4, 6, 3] BasicBlocks
    ResNet-50:  [3, 4, 6, 3] Bottlenecks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class ResNetConfig:
    """ResNet configuration."""
    in_channels: int = 3
    n_classes: int = 100
    architecture: str = "resnet18"    # resnet18, resnet34, resnet50
    dropout: float = 0.0
    use_se: bool = False              # Squeeze-and-Excitation


ARCHITECTURES = {
    "resnet18": {"blocks": [2, 2, 2, 2], "block_type": "basic"},
    "resnet34": {"blocks": [3, 4, 6, 3], "block_type": "basic"},
    "resnet50": {"blocks": [3, 4, 6, 3], "block_type": "bottleneck"},
}


# =============================================================================
# SQUEEZE-AND-EXCITATION
# =============================================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block (Hu et al., 2018).

    Channel attention mechanism:
        1. Squeeze: Global average pool → (B, C, 1, 1)
        2. Excitation: FC → ReLU → FC → Sigmoid → (B, C, 1, 1)
        3. Scale: multiply each channel by its attention weight

    This lets the network learn to emphasize informative channels
    and suppress less useful ones.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        w = self.squeeze(x).view(B, C)
        w = self.excitation(w).view(B, C, 1, 1)
        return x * w


# =============================================================================
# BASIC BLOCK (ResNet-18/34)
# =============================================================================

class BasicBlock(nn.Module):
    """
    Basic residual block: two 3x3 convolutions with skip connection.

        x ─────────────────────────────── (+) → ReLU → out
        │                                  ↑
        └→ Conv3x3 → BN → ReLU → Conv3x3 → BN [→ SE]
    """
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 use_se: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.se = SEBlock(out_ch) if use_se else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + self.shortcut(x))


# =============================================================================
# BOTTLENECK BLOCK (ResNet-50+)
# =============================================================================

class Bottleneck(nn.Module):
    """
    Bottleneck block: 1x1 → 3x3 → 1x1 with expansion factor 4.

        x ──────────────────────────────────────── (+) → ReLU → out
        │                                           ↑
        └→ 1x1(reduce) → BN → ReLU → 3x3 → BN → ReLU → 1x1(expand) → BN [→ SE]
    """
    expansion = 4

    def __init__(self, in_ch: int, mid_ch: int, stride: int = 1,
                 use_se: bool = False):
        super().__init__()
        out_ch = mid_ch * self.expansion

        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.se = SEBlock(out_ch) if use_se else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        return F.relu(out + self.shortcut(x))


# =============================================================================
# RESNET
# =============================================================================

class ResNet(nn.Module):
    """
    Full ResNet supporting 18/34/50 configurations.

    Architecture:
        Stem (3x3 conv) → Stage1 → Stage2 → Stage3 → Stage4 → AvgPool → FC
    """

    def __init__(self, config: ResNetConfig):
        super().__init__()
        self.config = config
        arch = ARCHITECTURES[config.architecture]
        blocks_per_stage = arch["blocks"]
        block_cls = BasicBlock if arch["block_type"] == "basic" else Bottleneck

        # Stem (for 32x32 images like CIFAR, use 3x3 instead of 7x7)
        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Stages
        channels = [64, 128, 256, 512]
        self.in_ch = 64
        self.stage1 = self._make_stage(block_cls, channels[0], blocks_per_stage[0],
                                        stride=1, use_se=config.use_se)
        self.stage2 = self._make_stage(block_cls, channels[1], blocks_per_stage[1],
                                        stride=2, use_se=config.use_se)
        self.stage3 = self._make_stage(block_cls, channels[2], blocks_per_stage[2],
                                        stride=2, use_se=config.use_se)
        self.stage4 = self._make_stage(block_cls, channels[3], blocks_per_stage[3],
                                        stride=2, use_se=config.use_se)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(channels[3] * block_cls.expansion, config.n_classes)

        self._init_weights()

    def _make_stage(self, block_cls, channels, n_blocks, stride, use_se):
        layers = []
        layers.append(block_cls(self.in_ch, channels, stride, use_se))
        self.in_ch = channels * block_cls.expansion
        for _ in range(1, n_blocks):
            layers.append(block_cls(self.in_ch, channels, 1, use_se))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


# =============================================================================
# PLAIN CNN (NO SKIP CONNECTIONS - FOR COMPARISON)
# =============================================================================

class PlainCNN(nn.Module):
    """Plain CNN without skip connections for comparison with ResNet."""

    def __init__(self, config: ResNetConfig):
        super().__init__()
        self.config = config

        layers = [
            nn.Conv2d(config.in_channels, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        channels = [64, 64, 128, 128, 256, 256, 512, 512]
        in_ch = 64
        for i, out_ch in enumerate(channels):
            stride = 2 if (i > 0 and channels[i] != channels[i-1]) else 1
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, config.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


# =============================================================================
# DENSENET-STYLE BLOCK
# =============================================================================

class DenseLayer(nn.Module):
    """Single DenseNet layer: BN → ReLU → Conv → concatenate."""

    def __init__(self, in_ch: int, growth_rate: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, growth_rate, 3, 1, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.layer(x)], dim=1)


class DenseBlock(nn.Module):
    """Stack of DenseNet layers with concatenation."""

    def __init__(self, in_ch: int, growth_rate: int, n_layers: int):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(DenseLayer(in_ch + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)
        self.out_channels = in_ch + n_layers * growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

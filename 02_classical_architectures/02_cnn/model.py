"""
Convolutional Neural Network for Image Classification
=======================================================

ResNet-style CNN with residual blocks, batch normalization,
and global average pooling for CIFAR-100 classification.

Architecture:
    Input(3x32x32) → Stem → [ResBlock] × N → GlobalAvgPool → FC → 100 classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List


@dataclass
class CNNConfig:
    """CNN configuration."""
    in_channels: int = 3
    n_classes: int = 100
    # Channels per stage: [64, 128, 256, 512]
    channels: List[int] = None
    # Blocks per stage: [2, 2, 2, 2]
    blocks_per_stage: List[int] = None
    dropout: float = 0.0

    def __post_init__(self):
        if self.channels is None:
            self.channels = [64, 128, 256]
        if self.blocks_per_stage is None:
            self.blocks_per_stage = [2, 2, 2]


class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """
    Basic residual block with skip connection.

    If dimensions don't match, uses 1x1 conv for projection.

        x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class CNN(nn.Module):
    """
    ResNet-style CNN for image classification.

    Features:
        - Configurable depth and width
        - Residual connections with projection shortcuts
        - Batch normalization throughout
        - Global average pooling (no large FC layers)
    """

    def __init__(self, config: CNNConfig):
        super().__init__()
        self.config = config

        # Stem: initial convolution
        self.stem = ConvBlock(config.in_channels, config.channels[0], 3, 1, 1)

        # Build stages
        stages = []
        in_ch = config.channels[0]
        for i, (out_ch, n_blocks) in enumerate(
                zip(config.channels, config.blocks_per_stage)):
            stride = 1 if i == 0 else 2  # Downsample after first stage
            blocks = [ResidualBlock(in_ch, out_ch, stride)]
            for _ in range(n_blocks - 1):
                blocks.append(ResidualBlock(out_ch, out_ch, 1))
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch

        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.channels[-1], config.n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

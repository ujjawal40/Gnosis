"""
DCGAN (Deep Convolutional GAN)
================================

Generator and Discriminator using convolutional architecture.
Follows DCGAN guidelines: BatchNorm, LeakyReLU in D, ReLU in G,
no fully connected layers in hidden, strided convolutions.

Architecture:
    Generator:     z(latent) → ConvTranspose → BN → ReLU → ... → Tanh → image
    Discriminator: image → Conv → BN → LeakyReLU → ... → Sigmoid → real/fake
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class GANConfig:
    latent_dim: int = 100
    n_channels: int = 1
    g_features: int = 64   # Generator feature map base
    d_features: int = 64   # Discriminator feature map base
    image_size: int = 28


class Generator(nn.Module):
    """
    DCGAN Generator: transforms random noise into images.

    z → FC → Reshape → ConvTranspose2d blocks → Tanh
    """

    def __init__(self, config: GANConfig):
        super().__init__()
        nf = config.g_features

        self.project = nn.Sequential(
            nn.Linear(config.latent_dim, nf * 4 * 7 * 7),
            nn.BatchNorm1d(nf * 4 * 7 * 7),
            nn.ReLU(True),
        )

        self.conv = nn.Sequential(
            # 7x7 → 14x14
            nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(True),
            # 14x14 → 28x28
            nn.ConvTranspose2d(nf * 2, config.n_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z)
        x = x.view(-1, self.conv[0].in_channels, 7, 7)
        return self.conv(x)


class Discriminator(nn.Module):
    """
    DCGAN Discriminator: classifies images as real or fake.

    image → Conv2d blocks → LeakyReLU → FC → Sigmoid
    """

    def __init__(self, config: GANConfig):
        super().__init__()
        nf = config.d_features

        self.conv = nn.Sequential(
            # 28x28 → 14x14
            nn.Conv2d(config.n_channels, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            # 14x14 → 7x7
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nf * 2 * 7 * 7, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return self.classifier(features)

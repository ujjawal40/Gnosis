"""
Autoencoders for Fashion-MNIST
================================

Vanilla, Denoising, and Sparse autoencoders with convolutional
encoder/decoder architecture.

Architecture:
    Encoder: Conv → BN → ReLU → Pool (repeated) → Flatten → FC → Latent
    Decoder: FC → Reshape → ConvTranspose → BN → ReLU (repeated) → Sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class AEConfig:
    in_channels: int = 1
    image_size: int = 28
    latent_dim: int = 32
    encoder_channels: list = None
    noise_factor: float = 0.3   # for denoising AE
    sparsity_weight: float = 0.001  # for sparse AE
    sparsity_target: float = 0.05

    def __post_init__(self):
        if self.encoder_channels is None:
            self.encoder_channels = [32, 64]


class Encoder(nn.Module):
    """Convolutional encoder: image → latent vector."""

    def __init__(self, config: AEConfig):
        super().__init__()
        layers = []
        in_ch = config.in_channels
        for out_ch in config.encoder_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            ])
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        # Compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, config.in_channels,
                                config.image_size, config.image_size)
            conv_out = self.conv(dummy)
            self.conv_shape = conv_out.shape[1:]
            self.flat_dim = conv_out.numel()

        self.fc = nn.Linear(self.flat_dim, config.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


class Decoder(nn.Module):
    """Convolutional decoder: latent vector → image."""

    def __init__(self, config: AEConfig, conv_shape: tuple, flat_dim: int):
        super().__init__()
        self.conv_shape = conv_shape
        self.fc = nn.Linear(config.latent_dim, flat_dim)

        layers = []
        channels = list(reversed(config.encoder_channels))
        for i, (in_ch, out_ch) in enumerate(zip(channels, channels[1:] + [config.in_channels])):
            layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(out_ch) if i < len(channels) - 1 else nn.Identity(),
                nn.ReLU() if i < len(channels) - 1 else nn.Sigmoid(),
            ])
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, *self.conv_shape)
        return self.deconv(x)


class Autoencoder(nn.Module):
    """Vanilla autoencoder: compress and reconstruct."""

    def __init__(self, config: AEConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config, self.encoder.conv_shape, self.encoder.flat_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DenoisingAutoencoder(Autoencoder):
    """Denoising autoencoder: add noise to input, reconstruct clean version."""

    def forward(self, x: torch.Tensor, add_noise: bool = True) -> tuple:
        if add_noise and self.training:
            noise = self.config.noise_factor * torch.randn_like(x)
            x_noisy = torch.clamp(x + noise, 0., 1.)
            z = self.encoder(x_noisy)
        else:
            z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


class SparseAutoencoder(Autoencoder):
    """
    Sparse autoencoder: adds KL penalty to encourage sparse latent codes.
    Encourages each latent dimension to be active for only a fraction of samples.
    """

    def sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        """KL divergence between average activation and target sparsity."""
        rho = self.config.sparsity_target
        rho_hat = torch.mean(torch.abs(z), dim=0)
        kl = rho * torch.log(rho / (rho_hat + 1e-8)) + \
             (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-8))
        return kl.sum()

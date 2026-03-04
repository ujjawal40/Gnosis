"""
Variational Autoencoder (VAE)
===============================

Learns a latent generative model using the reparameterization trick.
Optimizes ELBO = E[log p(x|z)] - KL(q(z|x) || p(z)).

Architecture:
    Encoder: Conv layers → mu, log_var (Gaussian posterior)
    Decoder: FC → ConvTranspose layers → Bernoulli likelihood
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class VAEConfig:
    in_channels: int = 1
    image_size: int = 28
    latent_dim: int = 16
    encoder_channels: list = None
    beta: float = 1.0  # KL weight (beta-VAE)

    def __post_init__(self):
        if self.encoder_channels is None:
            self.encoder_channels = [32, 64]


class VAE(nn.Module):
    """
    Convolutional VAE with reparameterization trick.

    The key insight: instead of sampling z ~ q(z|x) directly (non-differentiable),
    we sample ε ~ N(0,1) and compute z = μ + σ·ε (differentiable).
    """

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config

        # Encoder
        enc_layers = []
        in_ch = config.in_channels
        for out_ch in config.encoder_channels:
            enc_layers.extend([
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            ])
            in_ch = out_ch
        self.encoder_conv = nn.Sequential(*enc_layers)

        # Compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, config.in_channels, config.image_size, config.image_size)
            conv_out = self.encoder_conv(dummy)
            self.conv_shape = conv_out.shape[1:]
            flat_dim = conv_out.numel()

        self.fc_mu = nn.Linear(flat_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, config.latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(config.latent_dim, flat_dim)

        dec_layers = []
        channels = list(reversed(config.encoder_channels))
        for i, (in_c, out_c) in enumerate(zip(channels, channels[1:] + [config.in_channels])):
            dec_layers.extend([
                nn.ConvTranspose2d(in_c, out_c, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_c) if i < len(channels) - 1 else nn.Identity(),
                nn.ReLU() if i < len(channels) - 1 else nn.Sigmoid(),
            ])
        self.decoder_conv = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> tuple:
        h = self.encoder_conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * epsilon."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z).view(-1, *self.conv_shape)
        x_recon = self.decoder_conv(h)
        return x_recon

    def forward(self, x: torch.Tensor) -> tuple:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)

        # Crop if needed
        if x_recon.shape != x.shape:
            x_recon = x_recon[:, :, :x.shape[2], :x.shape[3]]

        return x_recon, mu, logvar

    def loss(self, x: torch.Tensor, x_recon: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor) -> tuple:
        """ELBO loss = Reconstruction + β * KL divergence."""
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum") / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        total = recon_loss + self.config.beta * kl_loss
        return total, recon_loss, kl_loss

    @torch.no_grad()
    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """Generate new images by sampling from prior p(z) = N(0,I)."""
        z = torch.randn(n_samples, self.config.latent_dim, device=device)
        return self.decode(z)

    @torch.no_grad()
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor,
                    n_steps: int = 10) -> torch.Tensor:
        """Interpolate between two images in latent space."""
        mu1, _ = self.encode(x1.unsqueeze(0))
        mu2, _ = self.encode(x2.unsqueeze(0))

        alphas = torch.linspace(0, 1, n_steps, device=x1.device)
        z_interp = torch.stack([mu1 * (1 - a) + mu2 * a for a in alphas]).squeeze(1)
        return self.decode(z_interp)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

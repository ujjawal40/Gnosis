"""
Denoising Diffusion Probabilistic Model (DDPM)
=================================================

Ho et al. (2020): learns to denoise images at each noise level.
Forward process: gradually add noise. Reverse process: learned denoising.

Architecture:
    UNet with time embedding: noisy_image, t → predicted_noise
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    image_channels: int = 1
    image_size: int = 28
    n_timesteps: int = 200  # Fewer for M3 8GB
    model_channels: int = 64
    time_emb_dim: int = 128
    beta_start: float = 1e-4
    beta_end: float = 0.02


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal position encoding for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    """Residual block with time embedding injection."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


class SimpleUNet(nn.Module):
    """
    Simplified UNet for denoising.
    Down → Middle → Up with skip connections and time conditioning.
    """

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        ch = config.model_channels
        t_dim = config.time_emb_dim

        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        # Encoder
        self.conv_in = nn.Conv2d(config.image_channels, ch, 3, padding=1)
        self.down1 = ResBlock(ch, ch, t_dim)
        self.down2 = ResBlock(ch, ch * 2, t_dim)
        self.pool = nn.AvgPool2d(2)

        # Middle
        self.mid = ResBlock(ch * 2, ch * 2, t_dim)

        # Decoder
        self.up2 = ResBlock(ch * 4, ch, t_dim)  # *4 because of skip
        self.up1 = ResBlock(ch * 2, ch, t_dim)   # *2 because of skip
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, config.image_channels, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)

        # Encoder
        h1 = self.conv_in(x)
        h1 = self.down1(h1, t_emb)
        h2 = self.pool(h1)
        h2 = self.down2(h2, t_emb)
        h3 = self.pool(h2)

        # Middle
        h3 = self.mid(h3, t_emb)

        # Decoder with skip connections
        h = self.upsample(h3)
        h = torch.cat([h, h2], dim=1)
        h = self.up2(h, t_emb)
        h = self.upsample(h)
        h = torch.cat([h, h1], dim=1)
        h = self.up1(h, t_emb)

        return self.conv_out(h)


class DDPM:
    """
    Denoising Diffusion Probabilistic Model.

    Forward process: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1-alpha_bar_t) * I)
    Reverse process: p(x_{t-1} | x_t) learned by UNet predicting noise.
    """

    def __init__(self, config: DiffusionConfig, device: torch.device):
        self.config = config
        self.device = device

        # Noise schedule
        betas = torch.linspace(config.beta_start, config.beta_end,
                               config.n_timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor,
                  noise: torch.Tensor = None) -> tuple:
        """Forward process: add noise to clean image at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_ab = self.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]

        x_t = sqrt_ab * x_0 + sqrt_one_minus_ab * noise
        return x_t, noise

    def loss(self, model: nn.Module, x_0: torch.Tensor) -> torch.Tensor:
        """Training loss: MSE between predicted and actual noise."""
        batch_size = x_0.size(0)
        t = torch.randint(0, self.config.n_timesteps, (batch_size,),
                          device=self.device)
        noise = torch.randn_like(x_0)
        x_t, _ = self.add_noise(x_0, t, noise)

        predicted_noise = model(x_t, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, model: nn.Module, n_samples: int) -> torch.Tensor:
        """Generate images via reverse diffusion process."""
        model.eval()
        shape = (n_samples, self.config.image_channels,
                 self.config.image_size, self.config.image_size)
        x = torch.randn(shape, device=self.device)

        for t in reversed(range(self.config.n_timesteps)):
            t_batch = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
            predicted_noise = model(x, t_batch)

            alpha = self.alphas[t]
            alpha_bar = self.alpha_bar[t]
            beta = self.betas[t]

            # Predicted mean
            mean = (1 / torch.sqrt(alpha)) * (
                x - beta / torch.sqrt(1 - alpha_bar) * predicted_noise
            )

            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta) * noise
            else:
                x = mean

        return x.clamp(-1, 1)

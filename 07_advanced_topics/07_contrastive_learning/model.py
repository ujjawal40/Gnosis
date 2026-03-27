"""
Contrastive & Self-Supervised Learning in PyTorch
====================================================

PyTorch implementations of SimCLR, BYOL, and Barlow Twins
for self-supervised representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class ContrastiveConfig:
    in_channels: int = 1
    hidden_dim: int = 256
    proj_dim: int = 128
    temperature: float = 0.5
    ema_tau: float = 0.996
    barlow_lambda: float = 0.005


# =============================================================================
# ENCODER BACKBONE
# =============================================================================

class ConvEncoder(nn.Module):
    """Simple CNN backbone for image representation learning."""

    def __init__(self, in_channels: int = 1, hidden_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))


# =============================================================================
# PROJECTION HEAD
# =============================================================================

class ProjectionHead(nn.Module):
    """MLP projection head used in contrastive methods."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# NT-Xent LOSS (SimCLR)
# =============================================================================

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = 0.5) -> torch.Tensor:
    """
    NT-Xent loss for SimCLR.

    z1, z2: (B, D) normalized embeddings from two augmented views.
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    z = F.normalize(z, dim=1)

    sim = z @ z.T / temperature  # (2B, 2B)

    # Mask out self-similarity
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)]).to(z.device)

    return F.cross_entropy(sim, labels)


# =============================================================================
# BYOL LOSS
# =============================================================================

def byol_loss(online: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BYOL loss: negative cosine similarity."""
    online = F.normalize(online, dim=-1)
    target = F.normalize(target, dim=-1)
    return 2 - 2 * (online * target).sum(dim=-1).mean()


# =============================================================================
# BARLOW TWINS LOSS
# =============================================================================

def barlow_twins_loss(z1: torch.Tensor, z2: torch.Tensor,
                       lambda_param: float = 0.005) -> torch.Tensor:
    """Barlow Twins loss: cross-correlation → identity."""
    B = z1.size(0)

    # Normalize along batch
    z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + 1e-4)
    z2_norm = (z2 - z2.mean(0)) / (z2.std(0) + 1e-4)

    # Cross-correlation
    C = (z1_norm.T @ z2_norm) / B

    on_diag = ((C.diag() - 1) ** 2).sum()
    off_diag = (C ** 2).sum() - (C.diag() ** 2).sum()

    return on_diag + lambda_param * off_diag


# =============================================================================
# SimCLR MODEL
# =============================================================================

class SimCLR(nn.Module):
    """SimCLR: Simple Contrastive Learning of Representations."""

    def __init__(self, config: ContrastiveConfig):
        super().__init__()
        self.encoder = ConvEncoder(config.in_channels, config.hidden_dim)
        self.projector = ProjectionHead(config.hidden_dim, config.hidden_dim,
                                        config.proj_dim)
        self.temperature = config.temperature

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        return nt_xent_loss(z1, z2, self.temperature)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get representations (without projection head)."""
        return self.encoder(x)


# =============================================================================
# BARLOW TWINS MODEL
# =============================================================================

class BarlowTwins(nn.Module):
    """Barlow Twins: Self-Supervised Learning via Redundancy Reduction."""

    def __init__(self, config: ContrastiveConfig):
        super().__init__()
        self.encoder = ConvEncoder(config.in_channels, config.hidden_dim)
        self.projector = ProjectionHead(config.hidden_dim, config.hidden_dim,
                                        config.proj_dim)
        self.lambda_param = config.barlow_lambda

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        return barlow_twins_loss(z1, z2, self.lambda_param)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# =============================================================================
# LINEAR EVALUATION
# =============================================================================

class LinearEvaluator(nn.Module):
    """Linear probe for evaluating learned representations."""

    def __init__(self, encoder: nn.Module, hidden_dim: int, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.linear = nn.Linear(hidden_dim, n_classes)

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.encoder(x)
        return self.linear(h)

"""
DCGAN Training on Fashion-MNIST
==================================

Adversarial training: Generator tries to fool Discriminator,
Discriminator tries to distinguish real from fake.

Uses standard GAN loss with label smoothing for stability.

Usage:
    python train.py
    python train.py --epochs 50 --latent_dim 100
"""

import sys
import os
import argparse
import time
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model import Generator, Discriminator, GANConfig


@dataclass
class TrainConfig:
    latent_dim: int = 100
    g_features: int = 64
    d_features: int = 64
    lr_g: float = 0.0002
    lr_d: float = 0.0002
    beta1: float = 0.5
    epochs: int = 50
    batch_size: int = 128
    d_steps: int = 1      # D updates per G update
    label_smooth: float = 0.1
    seed: int = 42
    device: str = "auto"


def get_device(config: TrainConfig) -> torch.device:
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    """GAN trainer with alternating D/G updates."""

    def __init__(self, gen: Generator, disc: Discriminator,
                 config: TrainConfig, device: torch.device):
        self.gen = gen.to(device)
        self.disc = disc.to(device)
        self.config = config
        self.device = device
        self.history: Dict[str, List[float]] = {
            "d_loss": [], "g_loss": [], "d_real": [], "d_fake": [],
        }

        # Fixed noise for tracking generation quality
        self.fixed_noise = torch.randn(64, config.latent_dim, device=device)

    def train_one_epoch(self, loader, opt_g, opt_d, loss_fn) -> tuple:
        self.gen.train()
        self.disc.train()

        total_d_loss = 0.0
        total_g_loss = 0.0
        d_real_acc = 0.0
        d_fake_acc = 0.0
        n = 0

        for real_images, _ in loader:
            batch_size = real_images.size(0)
            real_images = real_images.to(self.device)

            # Labels with smoothing
            real_label = (1.0 - self.config.label_smooth) * torch.ones(
                batch_size, 1, device=self.device)
            fake_label = torch.zeros(batch_size, 1, device=self.device)

            # ---- Train Discriminator ----
            for _ in range(self.config.d_steps):
                opt_d.zero_grad()

                # Real
                d_real = self.disc(real_images)
                d_loss_real = loss_fn(d_real, real_label)

                # Fake
                z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
                fake_images = self.gen(z).detach()
                d_fake = self.disc(fake_images)
                d_loss_fake = loss_fn(d_fake, fake_label)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                opt_d.step()

            # ---- Train Generator ----
            opt_g.zero_grad()
            z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
            fake_images = self.gen(z)
            d_fake = self.disc(fake_images)
            g_loss = loss_fn(d_fake, real_label)  # G wants D to say "real"
            g_loss.backward()
            opt_g.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            d_real_acc += torch.sigmoid(d_real).mean().item()
            d_fake_acc += torch.sigmoid(d_fake).mean().item()
            n += 1

        return (total_d_loss / n, total_g_loss / n,
                d_real_acc / n, d_fake_acc / n)

    def fit(self, train_loader):
        loss_fn = nn.BCEWithLogitsLoss()

        opt_g = torch.optim.Adam(self.gen.parameters(),
                                  lr=self.config.lr_g,
                                  betas=(self.config.beta1, 0.999))
        opt_d = torch.optim.Adam(self.disc.parameters(),
                                  lr=self.config.lr_d,
                                  betas=(self.config.beta1, 0.999))

        g_params = sum(p.numel() for p in self.gen.parameters())
        d_params = sum(p.numel() for p in self.disc.parameters())
        print(f"  Generator params: {g_params:,}")
        print(f"  Discriminator params: {d_params:,}")
        print()

        for epoch in range(self.config.epochs):
            start = time.time()
            d_loss, g_loss, d_real, d_fake = self.train_one_epoch(
                train_loader, opt_g, opt_d, loss_fn)
            elapsed = time.time() - start

            self.history["d_loss"].append(d_loss)
            self.history["g_loss"].append(g_loss)
            self.history["d_real"].append(d_real)
            self.history["d_fake"].append(d_fake)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{self.config.epochs} ({elapsed:.1f}s): "
                      f"D_loss={d_loss:.4f} G_loss={g_loss:.4f} "
                      f"D(real)={d_real:.3f} D(fake)={d_fake:.3f}")

        # Generate final samples
        self.gen.eval()
        with torch.no_grad():
            samples = self.gen(self.fixed_noise)
        print(f"\n  Generated {samples.shape[0]} samples, shape: {samples.shape[1:]}")
        print(f"  Pixel range: [{samples.min():.3f}, {samples.max():.3f}]")
        print()


def main():
    parser = argparse.ArgumentParser(description="DCGAN on Fashion-MNIST")
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--lr_g", type=float, default=0.0002)
    parser.add_argument("--lr_d", type=float, default=0.0002)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(**vars(args))
    torch.manual_seed(config.seed)
    device = get_device(config)

    print("=" * 60)
    print("DCGAN ON FASHION-MNIST")
    print(f"Device: {device}")
    print("=" * 60)

    from datasets.image_datasets import load_fashion_mnist
    train_loader, _ = load_fashion_mnist(
        batch_size=config.batch_size, augment=False)

    gan_config = GANConfig(
        latent_dim=config.latent_dim, n_channels=1,
        g_features=config.g_features, d_features=config.d_features,
    )
    gen = Generator(gan_config)
    disc = Discriminator(gan_config)

    trainer = Trainer(gen, disc, config, device)
    trainer.fit(train_loader)


if __name__ == "__main__":
    main()

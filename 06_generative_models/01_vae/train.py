"""
VAE Training on Fashion-MNIST
================================

Trains a Variational Autoencoder for image generation.
Evaluates reconstruction quality and generates new samples.

Usage:
    python train.py
    python train.py --latent_dim 32 --beta 1.0 --epochs 30
"""

import sys
import os
import argparse
import time
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model import VAE, VAEConfig


@dataclass
class TrainConfig:
    latent_dim: int = 16
    beta: float = 1.0
    lr: float = 0.001
    epochs: int = 30
    batch_size: int = 128
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
    def __init__(self, model: VAE, config: TrainConfig, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.history: Dict[str, List[float]] = {
            "total_loss": [], "recon_loss": [], "kl_loss": [],
        }

    def train_one_epoch(self, loader, optimizer) -> tuple:
        self.model.train()
        total, recon, kl = 0.0, 0.0, 0.0
        n = 0

        for x_batch, _ in loader:
            x_batch = x_batch.to(self.device)
            # Rescale to [0, 1] for BCE loss
            x_batch = (x_batch - x_batch.min()) / (x_batch.max() - x_batch.min() + 1e-8)

            optimizer.zero_grad()
            x_recon, mu, logvar = self.model(x_batch)
            loss, r_loss, k_loss = self.model.loss(x_batch, x_recon, mu, logvar)
            loss.backward()
            optimizer.step()

            total += loss.item()
            recon += r_loss.item()
            kl += k_loss.item()
            n += 1

        return total / n, recon / n, kl / n

    def fit(self, train_loader, val_loader):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        print(f"  Parameters: {self.model.count_parameters():,}")
        print(f"  Latent dim: {self.config.latent_dim}")
        print(f"  Beta (KL weight): {self.config.beta}")
        print()

        for epoch in range(self.config.epochs):
            start = time.time()
            total, recon, kl = self.train_one_epoch(train_loader, optimizer)
            elapsed = time.time() - start

            self.history["total_loss"].append(total)
            self.history["recon_loss"].append(recon)
            self.history["kl_loss"].append(kl)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{self.config.epochs} ({elapsed:.1f}s): "
                      f"total={total:.2f} recon={recon:.2f} kl={kl:.2f}")

        # Generate samples
        samples = self.model.sample(16, self.device)
        print(f"\n  Generated {samples.shape[0]} samples, shape: {samples.shape[1:]}")
        print(f"  Sample pixel range: [{samples.min():.3f}, {samples.max():.3f}]")
        print()


def main():
    parser = argparse.ArgumentParser(description="VAE on Fashion-MNIST")
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(**vars(args))
    torch.manual_seed(config.seed)
    device = get_device(config)

    print("=" * 60)
    print("VAE ON FASHION-MNIST")
    print(f"Device: {device}")
    print("=" * 60)

    from datasets.image_datasets import load_fashion_mnist
    train_loader, val_loader = load_fashion_mnist(
        batch_size=config.batch_size, augment=False)

    vae_config = VAEConfig(
        in_channels=1, image_size=28, latent_dim=config.latent_dim,
        beta=config.beta,
    )
    model = VAE(vae_config)

    trainer = Trainer(model, config, device)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()

"""
DDPM Training on Fashion-MNIST
=================================

Trains a denoising diffusion model to generate Fashion-MNIST images.
Uses simplified UNet and reduced timesteps for M3 8GB compatibility.

Usage:
    python train.py
    python train.py --n_timesteps 200 --epochs 50
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
from model import SimpleUNet, DDPM, DiffusionConfig


@dataclass
class TrainConfig:
    n_timesteps: int = 200
    model_channels: int = 64
    lr: float = 2e-4
    epochs: int = 50
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
    def __init__(self, model: nn.Module, ddpm: DDPM,
                 config: TrainConfig, device: torch.device):
        self.model = model.to(device)
        self.ddpm = ddpm
        self.config = config
        self.device = device
        self.history: Dict[str, List[float]] = {"loss": []}

    def train_one_epoch(self, loader, optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0

        for x_batch, _ in loader:
            x_batch = x_batch.to(self.device)

            optimizer.zero_grad()
            loss = self.ddpm.loss(self.model, x_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n += 1

        return total_loss / n

    def fit(self, train_loader):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Parameters: {n_params:,}")
        print(f"  Timesteps: {self.config.n_timesteps}")
        print()

        for epoch in range(self.config.epochs):
            start = time.time()
            loss = self.train_one_epoch(train_loader, optimizer)
            scheduler.step()
            elapsed = time.time() - start

            self.history["loss"].append(loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{self.config.epochs} ({elapsed:.1f}s): "
                      f"loss={loss:.6f}")

        # Generate samples
        print("\n  Generating samples...")
        samples = self.ddpm.sample(self.model, n_samples=16)
        print(f"  Generated {samples.shape[0]} samples, shape: {samples.shape[1:]}")
        print(f"  Pixel range: [{samples.min():.3f}, {samples.max():.3f}]")
        print()


def main():
    parser = argparse.ArgumentParser(description="DDPM on Fashion-MNIST")
    parser.add_argument("--n_timesteps", type=int, default=200)
    parser.add_argument("--model_channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(**vars(args))
    torch.manual_seed(config.seed)
    device = get_device(config)

    print("=" * 60)
    print("DDPM ON FASHION-MNIST")
    print(f"Device: {device}")
    print("=" * 60)

    from datasets.image_datasets import load_fashion_mnist
    train_loader, _ = load_fashion_mnist(
        batch_size=config.batch_size, augment=False)

    diff_config = DiffusionConfig(
        image_channels=1, image_size=28,
        n_timesteps=config.n_timesteps,
        model_channels=config.model_channels,
    )
    model = SimpleUNet(diff_config)
    ddpm = DDPM(diff_config, device)

    trainer = Trainer(model, ddpm, config, device)
    trainer.fit(train_loader)


if __name__ == "__main__":
    main()

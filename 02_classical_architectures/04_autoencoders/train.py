"""
Autoencoder Training on Fashion-MNIST
========================================

Trains vanilla, denoising, and sparse autoencoders for image reconstruction.
Evaluates reconstruction quality and latent space structure.

Usage:
    python train.py
    python train.py --variant denoising --latent_dim 16 --epochs 30
"""

import sys
import os
import argparse
import time
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model import Autoencoder, DenoisingAutoencoder, SparseAutoencoder, AEConfig


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TrainConfig:
    variant: str = "vanilla"  # "vanilla", "denoising", "sparse"
    latent_dim: int = 32
    lr: float = 0.001
    epochs: int = 30
    batch_size: int = 128
    noise_factor: float = 0.3
    sparsity_weight: float = 0.001
    seed: int = 42
    device: str = "auto"


def get_device(config: TrainConfig) -> torch.device:
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """Autoencoder training with reconstruction loss and variant-specific losses."""

    def __init__(self, model, config: TrainConfig, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
        }

    def compute_loss(self, x: torch.Tensor) -> tuple:
        """Compute reconstruction + variant-specific loss."""
        x_recon, z = self.model(x)

        # Crop if sizes don't match (due to conv/deconv rounding)
        if x_recon.shape != x.shape:
            x_recon = x_recon[:, :, :x.shape[2], :x.shape[3]]

        recon_loss = F.mse_loss(x_recon, x)

        if self.config.variant == "sparse" and isinstance(self.model, SparseAutoencoder):
            sparse_loss = self.model.sparsity_loss(z)
            total_loss = recon_loss + self.config.sparsity_weight * sparse_loss
        else:
            total_loss = recon_loss

        return total_loss, recon_loss.item()

    def train_one_epoch(self, loader: DataLoader, optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x_batch, _ in loader:  # Ignore labels
            x_batch = x_batch.to(self.device)

            optimizer.zero_grad()
            loss, _ = self.compute_loss(x_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for x_batch, _ in loader:
            x_batch = x_batch.to(self.device)
            _, recon_loss = self.compute_loss(x_batch)
            total_loss += recon_loss
            n_batches += 1

        return total_loss / n_batches

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)

        print(f"  Parameters: {self.model.count_parameters():,}")
        print(f"  Latent dim: {self.config.latent_dim}")
        print()

        for epoch in range(self.config.epochs):
            start = time.time()
            train_loss = self.train_one_epoch(train_loader, optimizer)
            val_loss = self.evaluate(val_loader)
            scheduler.step(val_loss)
            elapsed = time.time() - start

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch+1:3d}/{self.config.epochs} ({elapsed:.1f}s): "
                      f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                      f"lr={lr:.6f}")

        print(f"\n  Final val reconstruction loss: {self.history['val_loss'][-1]:.6f}")

        # Latent space analysis
        self.analyze_latent_space(val_loader)

    @torch.no_grad()
    def analyze_latent_space(self, loader: DataLoader):
        """Analyze the latent space structure."""
        self.model.eval()
        all_z = []
        all_labels = []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            _, z = self.model(x_batch)
            all_z.append(z.cpu())
            all_labels.append(y_batch)

        all_z = torch.cat(all_z, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Stats
        print(f"\n  Latent space statistics:")
        print(f"    Mean: {all_z.mean().item():.4f}")
        print(f"    Std:  {all_z.std().item():.4f}")
        print(f"    Active dims (std > 0.1): "
              f"{(all_z.std(dim=0) > 0.1).sum().item()}/{all_z.shape[1]}")

        # Per-class mean distance
        print(f"\n  Mean latent norm per class:")
        for cls in range(10):
            mask = all_labels == cls
            cls_mean = all_z[mask].mean(dim=0)
            print(f"    Class {cls}: norm={cls_mean.norm().item():.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Autoencoder on Fashion-MNIST")
    parser.add_argument("--variant", default="vanilla",
                        choices=["vanilla", "denoising", "sparse"])
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--noise_factor", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(**vars(args))
    torch.manual_seed(config.seed)
    device = get_device(config)

    print("=" * 60)
    print(f"AUTOENCODER ({config.variant.upper()}) ON FASHION-MNIST")
    print(f"Device: {device}")
    print("=" * 60)

    # Load data (no augmentation for autoencoders)
    from datasets.image_datasets import load_fashion_mnist
    train_loader, val_loader = load_fashion_mnist(
        batch_size=config.batch_size, augment=False)

    # Build model
    ae_config = AEConfig(
        in_channels=1, image_size=28, latent_dim=config.latent_dim,
        noise_factor=config.noise_factor,
        sparsity_weight=config.sparsity_weight,
    )

    model_classes = {
        "vanilla": Autoencoder,
        "denoising": DenoisingAutoencoder,
        "sparse": SparseAutoencoder,
    }
    model = model_classes[config.variant](ae_config)

    trainer = Trainer(model, config, device)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()

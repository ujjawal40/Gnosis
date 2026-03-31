"""
Normalization Comparison Training on CIFAR-10
================================================

Compares BatchNorm, LayerNorm, GroupNorm, InstanceNorm, and RMSNorm
on identical architectures to reveal convergence and accuracy differences.

Usage:
    python train.py
    python train.py --epochs 30 --norm_types batch,layer,rms
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
from model import NormComparisonNet, NormConfig


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 128
    lr: float = 0.001
    weight_decay: float = 1e-4
    norm_types: str = "batch,layer,group,instance,rms,none"
    hidden_dims: str = "512,256,128"
    dropout: float = 0.1
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
# DATA
# =============================================================================

def get_cifar10(batch_size: int):
    """Load CIFAR-10 using torchvision."""
    from torchvision import datasets, transforms

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_set = datasets.CIFAR10(root="./data", train=True,
                                  download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root="./data", train=False,
                                 download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                               shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                              shuffle=False, num_workers=2)
    return train_loader, test_loader


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """Training loop for normalization comparison."""

    def __init__(self, model: nn.Module, config: TrainConfig, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
        }

    def train_one_epoch(self, loader, optimizer, scheduler=None):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            logits = self.model(X)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += X.size(0)

        if scheduler:
            scheduler.step()

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            logits = self.model(X)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * X.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += X.size(0)

        return total_loss / total, correct / total

    def fit(self, train_loader, val_loader, optimizer, scheduler=None):
        for epoch in range(self.config.epochs):
            t0 = time.time()
            train_loss, train_acc = self.train_one_epoch(train_loader, optimizer, scheduler)
            val_loss, val_acc = self.evaluate(val_loader)
            dt = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{self.config.epochs} | "
                      f"Train: {train_acc:.3f} | Val: {val_acc:.3f} | "
                      f"Loss: {train_loss:.4f} | {dt:.1f}s")

        return self.history


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--norm_types", type=str, default="batch,layer,group,rms,none")
    args = parser.parse_args()

    config = TrainConfig(**vars(args))
    device = get_device(config)
    torch.manual_seed(config.seed)

    print("=" * 70)
    print("NORMALIZATION COMPARISON ON CIFAR-10")
    print(f"Device: {device} | Epochs: {config.epochs}")
    print("=" * 70)

    train_loader, test_loader = get_cifar10(config.batch_size)
    hidden_dims = [int(x) for x in config.hidden_dims.split(",")]
    norm_types = config.norm_types.split(",")

    results = {}
    for norm_type in norm_types:
        print(f"\n{'─' * 50}")
        print(f"Training with: {norm_type.upper()} normalization")
        print(f"{'─' * 50}")

        torch.manual_seed(config.seed)
        norm_config = NormConfig(
            in_features=3 * 32 * 32,
            n_classes=10,
            hidden_dims=hidden_dims,
            norm_type=norm_type,
            dropout=config.dropout,
        )
        model = NormComparisonNet(norm_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                      weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs)

        trainer = Trainer(model, config, device)
        history = trainer.fit(train_loader, test_loader, optimizer, scheduler)
        results[norm_type] = history

    # Comparison table
    print(f"\n{'=' * 70}")
    print("FINAL COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Norm Type':>12s} | {'Best Val Acc':>12s} | {'Final Val Acc':>13s} | "
          f"{'Final Train':>11s} | {'Conv Speed':>10s}")
    print("-" * 70)

    for norm_type, hist in results.items():
        best_val = max(hist["val_acc"])
        final_val = hist["val_acc"][-1]
        final_train = hist["train_acc"][-1]
        # Convergence speed: epoch to reach 40% val acc
        conv = next((i+1 for i, a in enumerate(hist["val_acc"]) if a > 0.40),
                     config.epochs)
        print(f"{norm_type:>12s} | {best_val:>11.3f}% | {final_val:>12.3f}% | "
              f"{final_train:>10.3f}% | {conv:>8d}ep")


if __name__ == "__main__":
    main()

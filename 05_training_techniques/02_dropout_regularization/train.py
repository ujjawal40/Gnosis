"""
Regularization Comparison Training on Fashion-MNIST
======================================================

Compares dropout, L2, dropout+L2, DropConnect, and no regularization
to measure generalization gaps (train acc - test acc).

Usage:
    python train.py
    python train.py --epochs 30 --batch_size 256
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
from model import RegularizedMLP, RegConfig, apply_max_norm


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 256
    lr: float = 0.001
    seed: int = 42
    device: str = "auto"
    patience: int = 10


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

def get_fashion_mnist(batch_size: int):
    """Load Fashion-MNIST."""
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    train_set = datasets.FashionMNIST(root="./data", train=True,
                                        download=True, transform=transform)
    test_set = datasets.FashionMNIST(root="./data", train=False,
                                       download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                               shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                              shuffle=False, num_workers=2)
    return train_loader, test_loader


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """Training loop with regularization support."""

    def __init__(self, model: RegularizedMLP, config: TrainConfig,
                 device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
        }

    def train_one_epoch(self, loader, optimizer):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            logits = self.model(X)
            loss = F.cross_entropy(logits, y)
            loss = loss + self.model.regularization_loss()
            loss.backward()
            optimizer.step()

            # Max-norm constraint
            if self.model.config.max_norm > 0:
                apply_max_norm(self.model, self.model.config.max_norm)

            total_loss += loss.item() * X.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += X.size(0)

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

    def fit(self, train_loader, val_loader, optimizer):
        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader, optimizer)
            val_loss, val_acc = self.evaluate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d} | Train: {train_acc:.3f} | "
                      f"Val: {val_acc:.3f} | Gap: {train_acc - val_acc:.3f}")

            if patience_counter >= self.config.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        return self.history


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    config = TrainConfig(**vars(args))
    device = get_device(config)
    torch.manual_seed(config.seed)

    print("=" * 70)
    print("REGULARIZATION COMPARISON ON FASHION-MNIST")
    print(f"Device: {device} | Epochs: {config.epochs}")
    print("=" * 70)

    train_loader, test_loader = get_fashion_mnist(config.batch_size)

    experiments = {
        "No Reg":       RegConfig(dropout_rate=0.0, l2_lambda=0.0),
        "Dropout(0.3)": RegConfig(dropout_rate=0.3),
        "Dropout(0.5)": RegConfig(dropout_rate=0.5),
        "L2(1e-3)":     RegConfig(l2_lambda=1e-3),
        "L1(1e-4)":     RegConfig(l1_lambda=1e-4),
        "Drop+L2":      RegConfig(dropout_rate=0.3, l2_lambda=1e-3),
        "DropConnect":   RegConfig(dropconnect_rate=0.3),
        "Elastic":       RegConfig(l1_lambda=5e-5, l2_lambda=5e-4),
        "MaxNorm":       RegConfig(max_norm=3.0, dropout_rate=0.3),
    }

    results = {}
    for name, reg_config in experiments.items():
        print(f"\n{'─' * 50}")
        print(f"Experiment: {name}")
        print(f"{'─' * 50}")

        torch.manual_seed(config.seed)
        model = RegularizedMLP(reg_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        trainer = Trainer(model, config, device)
        history = trainer.fit(train_loader, test_loader, optimizer)
        results[name] = history

    # Comparison table
    print(f"\n{'=' * 70}")
    print("FINAL COMPARISON — GENERALIZATION GAPS")
    print(f"{'=' * 70}")
    print(f"{'Experiment':>14s} | {'Best Val':>8s} | {'Final Val':>9s} | "
          f"{'Train':>6s} | {'Gap':>6s}")
    print("-" * 55)

    for name, hist in results.items():
        best_val = max(hist["val_acc"])
        final_val = hist["val_acc"][-1]
        final_train = hist["train_acc"][-1]
        gap = final_train - final_val
        print(f"{name:>14s} | {best_val:>7.3f} | {final_val:>8.3f} | "
              f"{final_train:>5.3f} | {gap:>5.3f}")


if __name__ == "__main__":
    main()

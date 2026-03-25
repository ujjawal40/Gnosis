"""
Data Augmentation Comparison Training on CIFAR-10
====================================================

Compares: no augmentation, standard (flip+crop), strong (color+rotate+erase),
and AutoAugment to measure generalization improvement.

Usage:
    python train.py
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
from torchvision import datasets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model import SimpleCNN, AugConfig, get_transforms


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 0.01
    weight_decay: float = 5e-4
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
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_acc": [], "train_acc": [],
        }

    def train_one_epoch(self, loader, optimizer):
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
        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        correct, total = 0, 0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            correct += (self.model(X).argmax(1) == y).sum().item()
            total += X.size(0)
        return correct / total

    def fit(self, train_loader, val_loader, optimizer, scheduler=None):
        for epoch in range(self.config.epochs):
            loss, train_acc = self.train_one_epoch(train_loader, optimizer)
            val_acc = self.evaluate(val_loader)
            if scheduler:
                scheduler.step()
            self.history["train_loss"].append(loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                gap = train_acc - val_acc
                print(f"  Epoch {epoch+1:3d} | Train: {train_acc:.3f} | "
                      f"Val: {val_acc:.3f} | Gap: {gap:.3f}")
        return self.history


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = TrainConfig()
    device = get_device(config)

    print("=" * 70)
    print("DATA AUGMENTATION COMPARISON ON CIFAR-10")
    print("=" * 70)

    strategies = ["none", "standard", "strong", "autoaugment"]
    test_transform = get_transforms("none", train=False)
    test_set = datasets.CIFAR10(root="./data", train=False,
                                 download=True, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=config.batch_size,
                              shuffle=False, num_workers=2)

    results = {}
    for strategy in strategies:
        print(f"\n{'─' * 50}")
        print(f"Strategy: {strategy.upper()}")
        print(f"{'─' * 50}")

        torch.manual_seed(config.seed)
        train_transform = get_transforms(strategy, train=True)
        train_set = datasets.CIFAR10(root="./data", train=True,
                                      download=True, transform=train_transform)
        train_loader = DataLoader(train_set, batch_size=config.batch_size,
                                   shuffle=True, num_workers=2)

        model = SimpleCNN(AugConfig())
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,
                                     momentum=0.9, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs)

        trainer = Trainer(model, config, device)
        history = trainer.fit(train_loader, test_loader, optimizer, scheduler)
        results[strategy] = history

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Strategy':>14s} | {'Best Val':>8s} | {'Train':>6s} | {'Gap':>6s}")
    print("-" * 42)
    for name, hist in results.items():
        best_val = max(hist["val_acc"])
        final_train = hist["train_acc"][-1]
        gap = final_train - hist["val_acc"][-1]
        print(f"{name:>14s} | {best_val:>7.3f} | {final_train:>5.3f} | {gap:>5.3f}")


if __name__ == "__main__":
    main()

"""
Weight Initialization Comparison Training
============================================

Compare how different initialization strategies affect training
convergence and final accuracy on Fashion-MNIST.

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model import DeepMLP, InitConfig, apply_init


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 256
    lr: float = 0.001
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

def get_fashion_mnist(batch_size: int):
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    train_set = datasets.FashionMNIST(root="./data", train=True,
                                        download=True, transform=transform)
    test_set = datasets.FashionMNIST(root="./data", train=False,
                                       download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_acc": [],
        }

    def train_one_epoch(self, loader, optimizer):
        self.model.train()
        total_loss, total = 0.0, 0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            loss = F.cross_entropy(self.model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * X.size(0)
            total += X.size(0)
        return total_loss / total

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        correct, total = 0, 0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            correct += (self.model(X).argmax(1) == y).sum().item()
            total += X.size(0)
        return correct / total

    def fit(self, train_loader, val_loader, optimizer):
        for epoch in range(self.config.epochs):
            loss = self.train_one_epoch(train_loader, optimizer)
            acc = self.evaluate(val_loader)
            self.history["train_loss"].append(loss)
            self.history["val_acc"].append(acc)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d} | Loss: {loss:.4f} | Val Acc: {acc:.3f}")
        return self.history


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = TrainConfig()
    device = get_device(config)
    torch.manual_seed(config.seed)

    print("=" * 70)
    print("WEIGHT INITIALIZATION COMPARISON ON FASHION-MNIST")
    print(f"Deep MLP: 10 hidden layers of 256 units (NO BatchNorm)")
    print("=" * 70)

    train_loader, test_loader = get_fashion_mnist(config.batch_size)

    init_methods = ["random_small", "random_large", "xavier_normal",
                    "kaiming_normal", "orthogonal"]

    results = {}
    for init_name in init_methods:
        print(f"\n{'─' * 40}")
        print(f"Init: {init_name}")
        print(f"{'─' * 40}")

        torch.manual_seed(config.seed)
        model = DeepMLP(InitConfig())
        apply_init(model, init_name)

        # Check initial stats
        sample = torch.randn(32, 784)
        stats = model.get_layer_stats(sample)
        first_std = stats["layer_0"]["std"]
        last_std = stats[f"layer_{len(stats)-1}"]["std"]
        print(f"  Initial: layer_0 std={first_std:.4f}, "
              f"layer_{len(stats)-1} std={last_std:.6f}")

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        trainer = Trainer(model, config, device)
        history = trainer.fit(train_loader, test_loader, optimizer)
        results[init_name] = history

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Init':>18s} | {'Best Val Acc':>12s} | {'Final Loss':>11s}")
    print("-" * 48)
    for name, hist in results.items():
        best = max(hist["val_acc"])
        final_loss = hist["train_loss"][-1]
        print(f"{name:>18s} | {best:>11.3f} | {final_loss:>10.4f}")


if __name__ == "__main__":
    main()

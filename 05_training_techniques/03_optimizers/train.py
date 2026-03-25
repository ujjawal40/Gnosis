"""
Optimizer Comparison Training on CIFAR-10
===========================================

Head-to-head comparison of SGD, SGD+Momentum, Adam, AdamW, RMSProp
with various LR schedulers on identical architectures.

Usage:
    python train.py
    python train.py --epochs 30 --lr 0.01
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
from model import BenchmarkMLP, BenchmarkConfig, OptimizerConfig
from model import build_optimizer, build_scheduler


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TrainConfig:
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


# =============================================================================
# DATA
# =============================================================================

def get_cifar10(batch_size: int):
    """Load CIFAR-10."""
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
    """Training loop with LR tracking."""

    def __init__(self, model: nn.Module, config: TrainConfig, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "lr": [],
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * X.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += X.size(0)

        # Record LR
        current_lr = optimizer.param_groups[0]['lr']
        self.history["lr"].append(current_lr)

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
            train_loss, train_acc = self.train_one_epoch(
                train_loader, optimizer, scheduler)
            val_loss, val_acc = self.evaluate(val_loader)
            dt = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                lr = self.history["lr"][-1]
                print(f"  Epoch {epoch+1:3d} | Train: {train_acc:.3f} | "
                      f"Val: {val_acc:.3f} | LR: {lr:.6f} | {dt:.1f}s")

        return self.history


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    config = TrainConfig(**vars(args))
    device = get_device(config)
    torch.manual_seed(config.seed)

    print("=" * 70)
    print("OPTIMIZER COMPARISON ON CIFAR-10")
    print(f"Device: {device} | Epochs: {config.epochs}")
    print("=" * 70)

    train_loader, test_loader = get_cifar10(config.batch_size)

    experiments = {
        "SGD (lr=0.01)":       OptimizerConfig(optimizer_name="sgd", lr=0.01,
                                                scheduler_name="none"),
        "SGD+Mom (lr=0.01)":   OptimizerConfig(optimizer_name="sgd_momentum", lr=0.01,
                                                scheduler_name="cosine"),
        "Nesterov (lr=0.01)":  OptimizerConfig(optimizer_name="nesterov", lr=0.01,
                                                scheduler_name="cosine"),
        "Adam (lr=1e-3)":      OptimizerConfig(optimizer_name="adam", lr=1e-3,
                                                scheduler_name="cosine"),
        "AdamW (lr=1e-3)":     OptimizerConfig(optimizer_name="adamw", lr=1e-3,
                                                weight_decay=0.01,
                                                scheduler_name="cosine"),
        "RMSProp (lr=1e-3)":   OptimizerConfig(optimizer_name="rmsprop", lr=1e-3,
                                                scheduler_name="cosine"),
        "AdamW+Warmup":        OptimizerConfig(optimizer_name="adamw", lr=1e-3,
                                                weight_decay=0.01,
                                                scheduler_name="warmup_cosine",
                                                warmup_epochs=5),
    }

    results = {}
    for name, opt_config in experiments.items():
        print(f"\n{'─' * 50}")
        print(f"Optimizer: {name}")
        print(f"{'─' * 50}")

        torch.manual_seed(config.seed)
        model = BenchmarkMLP(BenchmarkConfig())
        optimizer = build_optimizer(model, opt_config)
        scheduler = build_scheduler(optimizer, opt_config, config.epochs)

        trainer = Trainer(model, config, device)
        history = trainer.fit(train_loader, test_loader, optimizer, scheduler)
        results[name] = history

    # Comparison table
    print(f"\n{'=' * 70}")
    print("FINAL COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Optimizer':>20s} | {'Best Val':>8s} | {'Final Val':>9s} | "
          f"{'Final LR':>10s}")
    print("-" * 55)

    for name, hist in results.items():
        best_val = max(hist["val_acc"])
        final_val = hist["val_acc"][-1]
        final_lr = hist["lr"][-1]
        print(f"{name:>20s} | {best_val:>7.3f} | {final_val:>8.3f} | "
              f"{final_lr:>10.6f}")


if __name__ == "__main__":
    main()

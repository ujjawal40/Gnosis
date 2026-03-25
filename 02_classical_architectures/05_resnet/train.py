"""
ResNet Training on CIFAR-100
===============================

Compares Plain CNN (no skip connections), ResNet-18, and ResNet+SE
to demonstrate the impact of residual connections.

Usage:
    python train.py
    python train.py --epochs 50 --architecture resnet18
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
from model import ResNet, ResNetConfig, PlainCNN


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 128
    lr: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
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

def get_cifar100(batch_size: int):
    """Load CIFAR-100 with data augmentation."""
    from torchvision import datasets, transforms

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_set = datasets.CIFAR100(root="./data", train=True,
                                    download=True, transform=transform_train)
    test_set = datasets.CIFAR100(root="./data", train=False,
                                   download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                               shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                              shuffle=False, num_workers=2)
    return train_loader, test_loader


# =============================================================================
# GRADIENT MONITOR
# =============================================================================

class GradientMonitor:
    """Track gradient norms per layer during training."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.grad_norms: Dict[str, List[float]] = {}

    def record(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None and 'weight' in name:
                norm = param.grad.norm().item()
                if name not in self.grad_norms:
                    self.grad_norms[name] = []
                self.grad_norms[name].append(norm)

    def summary(self, n_layers: int = 5) -> str:
        """Return gradient norm summary for first and last layers."""
        if not self.grad_norms:
            return "No gradients recorded"

        names = list(self.grad_norms.keys())
        lines = []
        for name in names[:n_layers] + names[-n_layers:]:
            norms = self.grad_norms[name]
            avg = sum(norms[-10:]) / min(10, len(norms))
            lines.append(f"  {name[:40]:40s} | avg grad norm: {avg:.6f}")
        return "\n".join(lines)


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """ResNet training loop with gradient monitoring."""

    def __init__(self, model: nn.Module, config: TrainConfig, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.grad_monitor = GradientMonitor(model)
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

            self.grad_monitor.record()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
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
            train_loss, train_acc = self.train_one_epoch(
                train_loader, optimizer, scheduler)
            val_loss, val_acc = self.evaluate(val_loader)
            dt = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{self.config.epochs} | "
                      f"Train: {train_acc:.3f} | Val: {val_acc:.3f} | "
                      f"Loss: {train_loss:.4f} | {dt:.1f}s")

        return self.history


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    config = TrainConfig(**vars(args))
    device = get_device(config)
    torch.manual_seed(config.seed)

    print("=" * 70)
    print("RESNET VS PLAIN CNN ON CIFAR-100")
    print(f"Device: {device} | Epochs: {config.epochs}")
    print("=" * 70)

    train_loader, test_loader = get_cifar100(config.batch_size)

    experiments = {
        "Plain CNN": lambda: PlainCNN(ResNetConfig()),
        "ResNet-18": lambda: ResNet(ResNetConfig(architecture="resnet18")),
        "ResNet-18+SE": lambda: ResNet(ResNetConfig(architecture="resnet18", use_se=True)),
    }

    results = {}
    for name, model_fn in experiments.items():
        print(f"\n{'─' * 50}")
        print(f"Model: {name}")

        torch.manual_seed(config.seed)
        model = model_fn()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")
        print(f"{'─' * 50}")

        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,
                                     momentum=config.momentum,
                                     weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs)

        trainer = Trainer(model, config, device)
        history = trainer.fit(train_loader, test_loader, optimizer, scheduler)
        results[name] = {"history": history, "params": n_params,
                          "grad_monitor": trainer.grad_monitor}

    # Comparison
    print(f"\n{'=' * 70}")
    print("FINAL COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Model':>15s} | {'Params':>10s} | {'Best Val':>8s} | "
          f"{'Final Val':>9s}")
    print("-" * 50)

    for name, res in results.items():
        best_val = max(res["history"]["val_acc"])
        final_val = res["history"]["val_acc"][-1]
        print(f"{name:>15s} | {res['params']:>10,} | {best_val:>7.3f} | "
              f"{final_val:>8.3f}")

    # Gradient analysis
    print(f"\n{'=' * 70}")
    print("GRADIENT FLOW ANALYSIS (last 10 batches avg)")
    print(f"{'=' * 70}")
    for name, res in results.items():
        print(f"\n{name}:")
        print(res["grad_monitor"].summary())


if __name__ == "__main__":
    main()

"""
Mixture of Experts Training on Fashion-MNIST
===============================================

Compare MoE model vs dense equivalent to show capacity/compute tradeoff.

Usage:
    python train.py
"""

import sys
import os
import time
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model import MoEClassifier, MoEConfig, load_balancing_loss


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 256
    lr: float = 0.001
    balance_coef: float = 0.01
    seed: int = 42
    device: str = "auto"


def get_device(config: TrainConfig) -> torch.device:
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def get_fashion_mnist(batch_size):
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.2860,), (0.3530,))])
    train = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
    test = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2))


# =============================================================================
# DENSE BASELINE
# =============================================================================

class DenseBaseline(nn.Module):
    """Dense model with similar active compute as MoE."""
    def __init__(self, in_features=784, hidden=128, expert_dim=64, n_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(),
            nn.Linear(hidden, expert_dim), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(expert_dim, n_classes),
        )
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.model(x), None


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    def __init__(self, model, config, device, balance_coef=0.01):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.balance_coef = balance_coef
        self.history: Dict[str, List[float]] = {"val_acc": [], "train_loss": []}

    def train_one_epoch(self, loader, optimizer):
        self.model.train()
        total_loss, total = 0.0, 0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            logits, router_probs = self.model(X)
            loss = F.cross_entropy(logits, y)
            if router_probs is not None:
                loss = loss + self.balance_coef * load_balancing_loss(
                    router_probs, router_probs.size(-1))
            loss.backward()
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
            logits, _ = self.model(X)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        return correct / total

    def fit(self, train_loader, val_loader, optimizer):
        for epoch in range(self.config.epochs):
            loss = self.train_one_epoch(train_loader, optimizer)
            acc = self.evaluate(val_loader)
            self.history["train_loss"].append(loss)
            self.history["val_acc"].append(acc)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d} | Loss: {loss:.4f} | Val Acc: {acc:.4f}")
        return self.history


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = TrainConfig()
    device = get_device(config)
    torch.manual_seed(config.seed)

    print("=" * 70)
    print("MIXTURE OF EXPERTS vs DENSE ON FASHION-MNIST")
    print("=" * 70)

    train_loader, test_loader = get_fashion_mnist(config.batch_size)

    experiments = {
        "Dense Baseline": lambda: DenseBaseline(),
        "MoE (8 experts, top-2)": lambda: MoEClassifier(MoEConfig(n_experts=8, top_k=2)),
        "MoE (16 experts, top-2)": lambda: MoEClassifier(MoEConfig(n_experts=16, top_k=2)),
        "MoE (8 experts, top-1)": lambda: MoEClassifier(MoEConfig(n_experts=8, top_k=1)),
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

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        trainer = Trainer(model, config, device, config.balance_coef)
        history = trainer.fit(train_loader, test_loader, optimizer)
        results[name] = {"history": history, "params": n_params}

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':>25s} | {'Params':>10s} | {'Best Acc':>9s}")
    print("-" * 50)
    for name, r in results.items():
        best = max(r["history"]["val_acc"])
        print(f"{name:>25s} | {r['params']:>10,} | {best:>8.4f}")


if __name__ == "__main__":
    main()

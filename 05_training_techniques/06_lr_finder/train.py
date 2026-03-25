"""
Learning Rate Finder Experiments
===================================

Run LR range test on MNIST and compare training with different
LR schedules: constant, cyclical, and 1cycle.

Usage:
    python train.py
"""

import sys
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model import SimpleMLP, LRFinderConfig, LRRangeTest, CyclicalLRScheduler


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 256
    seed: int = 42
    device: str = "auto"


def get_device(config):
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def get_mnist(bs):
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST("./data", train=True, download=True, transform=t)
    test = datasets.MNIST("./data", train=False, download=True, transform=t)
    return DataLoader(train, bs, shuffle=True, num_workers=2), DataLoader(test, bs, num_workers=2)


def train_epoch(model, loader, optimizer, device, scheduler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            correct += (model(X).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    config = TrainConfig()
    device = get_device(config)
    torch.manual_seed(config.seed)

    print("=" * 70)
    print("LEARNING RATE FINDER EXPERIMENTS")
    print("=" * 70)

    train_loader, test_loader = get_mnist(config.batch_size)

    # Step 1: LR Range Test
    print("\n--- LR Range Test ---")
    model = SimpleMLP(LRFinderConfig()).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
    finder = LRRangeTest(model, optimizer, F.cross_entropy,
                         min_lr=1e-7, max_lr=10.0, num_steps=200)
    finder.run(train_loader, device)
    suggested_lr = finder.suggest_lr()
    print(f"  Tested {len(finder.lrs)} LR values")
    print(f"  Suggested LR: {suggested_lr:.6f}")

    # Step 2: Compare LR strategies
    print(f"\n{'=' * 70}")
    print("LR STRATEGY COMPARISON")
    print(f"{'=' * 70}")

    strategies = {
        "Constant (suggested)": {"lr": suggested_lr, "scheduler": None},
        "Constant (0.01)": {"lr": 0.01, "scheduler": None},
        "Constant (0.1)": {"lr": 0.1, "scheduler": None},
        "Cyclical (tri)": {"lr": suggested_lr, "scheduler": "cyclical"},
        "OneCycle": {"lr": suggested_lr, "scheduler": "onecycle"},
    }

    results = {}
    for name, cfg in strategies.items():
        torch.manual_seed(config.seed)
        model = SimpleMLP(LRFinderConfig()).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"],
                                     momentum=0.9)

        scheduler = None
        if cfg["scheduler"] == "cyclical":
            scheduler = CyclicalLRScheduler(
                optimizer, base_lr=cfg["lr"] / 10, max_lr=cfg["lr"] * 5,
                step_size=len(train_loader) * 2
            )
        elif cfg["scheduler"] == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=cfg["lr"] * 10,
                epochs=config.epochs, steps_per_epoch=len(train_loader)
            )

        t0 = time.time()
        for epoch in range(config.epochs):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, device, scheduler
            )

        test_acc = evaluate(model, test_loader, device)
        elapsed = time.time() - t0
        results[name] = {"acc": test_acc, "final_loss": train_loss, "time": elapsed}
        print(f"  {name:>22s} | acc: {test_acc:.4f} | "
              f"loss: {train_loss:.4f} | time: {elapsed:.1f}s")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    best = max(results.items(), key=lambda x: x[1]["acc"])
    print(f"  Best strategy: {best[0]} ({best[1]['acc']:.4f})")
    print(f"  LR Range Test suggested: {suggested_lr:.6f}")


if __name__ == "__main__":
    main()

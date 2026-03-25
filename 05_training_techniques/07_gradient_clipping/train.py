"""
Gradient Clipping Experiments
================================

Compare gradient clipping methods on Fashion-MNIST with a deep network.

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
from model import DeepMLP, GradClipConfig, clip_gradients, GradientAccumulator


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 256
    lr: float = 0.001
    seed: int = 42
    device: str = "auto"


def get_device(config):
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def get_fashion_mnist(bs):
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    train = datasets.FashionMNIST("./data", train=True, download=True, transform=t)
    test = datasets.FashionMNIST("./data", train=False, download=True, transform=t)
    return DataLoader(train, bs, shuffle=True, num_workers=2), DataLoader(test, bs, num_workers=2)


def train_with_clipping(model, train_loader, test_loader, epochs, lr, device,
                         clip_method="none", clip_value=1.0):
    """Train model with specified gradient clipping method."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    grad_norms = []

    for epoch in range(epochs):
        model.train()
        epoch_norms = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(X), y)
            loss.backward()

            # Record pre-clip gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm().item() ** 2
            epoch_norms.append(total_norm ** 0.5)

            # Apply clipping
            if clip_method != "none":
                clip_gradients(model, clip_method, clip_value)

            optimizer.step()

        grad_norms.append(sum(epoch_norms) / len(epoch_norms))

    # Evaluate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            correct += (model(X).argmax(1) == y).sum().item()
            total += y.size(0)

    return correct / total, grad_norms


def main():
    config = TrainConfig()
    device = get_device(config)

    print("=" * 70)
    print("GRADIENT CLIPPING EXPERIMENTS")
    print("=" * 70)

    train_loader, test_loader = get_fashion_mnist(config.batch_size)

    # Compare clipping methods
    methods = [
        ("No Clipping", "none", 1.0),
        ("Norm Clip (1.0)", "norm", 1.0),
        ("Norm Clip (5.0)", "norm", 5.0),
        ("Value Clip (0.5)", "value", 0.5),
        ("AGC (0.01)", "agc", 0.01),
    ]

    print(f"\n{'Method':>20s} | {'Accuracy':>8s} | {'Avg Grad Norm (e1)':>18s} | "
          f"{'Avg Grad Norm (last)':>20s}")
    print("-" * 78)

    results = {}
    for name, method, clip_val in methods:
        torch.manual_seed(config.seed)
        model_config = GradClipConfig()
        model = DeepMLP(model_config).to(device)

        acc, grad_norms = train_with_clipping(
            model, train_loader, test_loader, config.epochs,
            config.lr, device, method, clip_val
        )

        results[name] = {"acc": acc, "grad_norms": grad_norms}
        print(f"{name:>20s} | {acc:>7.4f}  | {grad_norms[0]:>17.4f}  | "
              f"{grad_norms[-1]:>19.4f}")

    # Gradient accumulation experiment
    print(f"\n{'=' * 70}")
    print("GRADIENT ACCUMULATION EXPERIMENT")
    print(f"{'=' * 70}")

    accum_configs = [
        ("Batch 256, accum 1", 256, 1),
        ("Batch 64, accum 4", 64, 4),
        ("Batch 32, accum 8", 32, 8),
    ]

    print(f"\n{'Config':>25s} | {'Eff. Batch':>10s} | {'Accuracy':>8s}")
    print("-" * 52)

    for name, bs, accum_steps in accum_configs:
        torch.manual_seed(config.seed)
        model = DeepMLP(GradClipConfig()).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        train_loader_small, _ = get_fashion_mnist(bs)

        accum = GradientAccumulator(optimizer, accum_steps)

        for epoch in range(config.epochs):
            model.train()
            optimizer.zero_grad()
            for X, y in train_loader_small:
                X, y = X.to(device), y.to(device)
                loss = F.cross_entropy(model(X), y) / accum_steps
                loss.backward()
                accum.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                correct += (model(X).argmax(1) == y).sum().item()
                total += y.size(0)

        acc = correct / total
        print(f"{name:>25s} | {bs * accum_steps:>10d} | {acc:>7.4f}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    best = max(results.items(), key=lambda x: x[1]["acc"])
    print(f"  Best clipping method: {best[0]} ({best[1]['acc']:.4f})")


if __name__ == "__main__":
    main()

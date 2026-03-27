"""
Pruning Experiments
======================

Compare pruning strategies on MNIST: magnitude, structured, and IMP.

Usage:
    python train.py
"""

import sys
import os
import copy
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model import (PrunableMLP, PruneConfig, magnitude_prune,
                   structured_prune, iterative_magnitude_prune,
                   analyze_sparsity)


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


def get_mnist(bs):
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST("./data", train=True, download=True, transform=t)
    test = datasets.MNIST("./data", train=False, download=True, transform=t)
    return DataLoader(train, bs, shuffle=True, num_workers=2), DataLoader(test, bs, num_workers=2)


def train_model(model, train_loader, epochs, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            F.cross_entropy(model(X), y).backward()
            optimizer.step()


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
    print("MODEL PRUNING EXPERIMENTS")
    print("=" * 70)

    train_loader, test_loader = get_mnist(config.batch_size)
    p_config = PruneConfig()

    # Train dense baseline
    print("\n--- Dense Baseline ---")
    torch.manual_seed(config.seed)
    model = PrunableMLP(p_config).to(device)
    initial_state = copy.deepcopy(model.state_dict())
    train_model(model, train_loader, config.epochs, config.lr, device)
    dense_acc = evaluate(model, test_loader, device)
    print(f"  Accuracy: {dense_acc:.4f} | Params: {model.n_params:,}")

    # Magnitude pruning at different sparsities
    print(f"\n--- Magnitude Pruning ---")
    print(f"{'Sparsity':>10s} | {'Active':>10s} | {'Accuracy':>8s} | {'Drop':>6s}")
    print("-" * 42)

    results = {}
    for sp in [0.3, 0.5, 0.7, 0.9, 0.95]:
        torch.manual_seed(config.seed)
        m = PrunableMLP(p_config).to(device)
        train_model(m, train_loader, config.epochs, config.lr, device)
        magnitude_prune(m, sp)
        acc = evaluate(m, test_loader, device)
        results[f"mag-{sp}"] = acc
        print(f"{sp:>10.0%} | {m.active_params:>10,} | {acc:>7.4f}  | "
              f"{dense_acc - acc:>+.4f}")

    # Structured pruning
    print(f"\n--- Structured Pruning ---")
    print(f"{'Sparsity':>10s} | {'Active':>10s} | {'Accuracy':>8s} | {'Drop':>6s}")
    print("-" * 42)

    for sp in [0.3, 0.5, 0.7]:
        torch.manual_seed(config.seed)
        m = PrunableMLP(p_config).to(device)
        train_model(m, train_loader, config.epochs, config.lr, device)
        structured_prune(m, sp)
        acc = evaluate(m, test_loader, device)
        results[f"struct-{sp}"] = acc
        print(f"{sp:>10.0%} | {m.active_params:>10,} | {acc:>7.4f}  | "
              f"{dense_acc - acc:>+.4f}")

    # IMP (3 rounds)
    print(f"\n--- Iterative Magnitude Pruning (3 rounds) ---")
    torch.manual_seed(config.seed)
    m = PrunableMLP(p_config).to(device)
    init_state = copy.deepcopy(m.state_dict())

    for round_i in range(3):
        train_model(m, train_loader, config.epochs, config.lr, device)
        iterative_magnitude_prune(m, init_state, 0.2 * (round_i + 1))
        acc = evaluate(m, test_loader, device)
        info = analyze_sparsity(m)
        sp = info["total"]["sparsity"]
        print(f"  Round {round_i+1} | Sparsity: {sp:.1%} | "
              f"Accuracy: {acc:.4f} | Drop: {dense_acc - acc:>+.4f}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Dense baseline: {dense_acc:.4f}")
    best_sparse = max(
        ((k, v) for k, v in results.items() if 'mag-0.9' in k or 'mag-0.7' in k),
        key=lambda x: x[1], default=None
    )
    if best_sparse:
        print(f"  Best high-sparsity: {best_sparse[0]} ({best_sparse[1]:.4f})")


if __name__ == "__main__":
    main()

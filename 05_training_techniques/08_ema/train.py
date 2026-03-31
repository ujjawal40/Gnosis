"""
EMA & Model Averaging Experiments
=====================================

Compare EMA, SWA, and model soup on Fashion-MNIST.

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
from model import SimpleMLP, ModelEMA, SWAModel, uniform_soup, EMAConfig


@dataclass
class TrainConfig:
    epochs: int = 15
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
    print("EMA & MODEL AVERAGING EXPERIMENTS")
    print("=" * 70)

    ema_config = EMAConfig()
    train_loader, test_loader = get_fashion_mnist(config.batch_size)

    # Train with EMA tracking
    print("\n--- Training with EMA and SWA ---")
    model = SimpleMLP(ema_config).to(device)
    ema = ModelEMA(model, decay=0.999)
    swa = SWAModel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            F.cross_entropy(model(X), y).backward()
            optimizer.step()
            ema.update(model)

        # SWA: collect from second half of training
        if epoch >= config.epochs // 2:
            swa.update(model)

        if (epoch + 1) % 5 == 0:
            raw_acc = evaluate(model, test_loader, device)
            ema_acc = evaluate(ema.shadow, test_loader, device)
            print(f"  Epoch {epoch+1:>3d} | Raw: {raw_acc:.4f} | EMA: {ema_acc:.4f}")

    # Final evaluation
    raw_acc = evaluate(model, test_loader, device)
    ema_acc = evaluate(ema.shadow, test_loader, device)
    swa_acc = evaluate(swa.averaged, test_loader, device)

    print(f"\n--- Final Results ---")
    print(f"  Raw model:  {raw_acc:.4f}")
    print(f"  EMA (0.999): {ema_acc:.4f}")
    print(f"  SWA:         {swa_acc:.4f}")

    # EMA decay comparison
    print(f"\n--- EMA Decay Comparison ---")
    decays = [0.9, 0.99, 0.999, 0.9999]
    for decay in decays:
        torch.manual_seed(config.seed)
        m = SimpleMLP(ema_config).to(device)
        e = ModelEMA(m, decay=decay)
        opt = torch.optim.Adam(m.parameters(), lr=config.lr)

        for epoch in range(config.epochs):
            m.train()
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                opt.zero_grad()
                F.cross_entropy(m(X), y).backward()
                opt.step()
                e.update(m)

        acc = evaluate(e.shadow, test_loader, device)
        print(f"  decay={decay:.4f} | Accuracy: {acc:.4f}")

    # Model Soup
    print(f"\n--- Model Soup ---")
    soup_models = []
    for i in range(3):
        torch.manual_seed(config.seed + i)
        m = SimpleMLP(ema_config).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=config.lr * (0.5 + 0.5 * i))
        for epoch in range(config.epochs):
            m.train()
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                opt.zero_grad()
                F.cross_entropy(m(X), y).backward()
                opt.step()
        acc = evaluate(m, test_loader, device)
        print(f"  Model {i+1} (lr={config.lr * (0.5 + 0.5 * i):.4f}): {acc:.4f}")
        soup_models.append(m)

    soup = uniform_soup(soup_models)
    soup_acc = evaluate(soup, test_loader, device)
    print(f"  Uniform soup: {soup_acc:.4f}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Raw:   {raw_acc:.4f}")
    print(f"  EMA:   {ema_acc:.4f} ({'+' if ema_acc > raw_acc else ''}{ema_acc - raw_acc:.4f})")
    print(f"  SWA:   {swa_acc:.4f} ({'+' if swa_acc > raw_acc else ''}{swa_acc - raw_acc:.4f})")
    print(f"  Soup:  {soup_acc:.4f} ({'+' if soup_acc > raw_acc else ''}{soup_acc - raw_acc:.4f})")


if __name__ == "__main__":
    main()

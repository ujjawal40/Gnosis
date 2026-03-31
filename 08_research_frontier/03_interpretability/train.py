"""
Interpretability Analysis on MNIST
=====================================

Train a model on MNIST, then apply interpretability methods
to understand which pixels drive predictions.

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
from model import (InterpretableModel, InterpConfig, vanilla_saliency,
                   smooth_grad, integrated_gradients, occlusion_sensitivity)


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


def main():
    config = TrainConfig()
    device = get_device(config)
    torch.manual_seed(config.seed)

    print("=" * 70)
    print("INTERPRETABILITY ANALYSIS ON MNIST")
    print("=" * 70)

    train_loader, test_loader = get_mnist(config.batch_size)

    # Train model
    model = InterpretableModel(InterpConfig()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            F.cross_entropy(model(X), y).backward()
            optimizer.step()

        if (epoch + 1) % 3 == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(device), y.to(device)
                    correct += (model(X).argmax(1) == y).sum().item()
                    total += y.size(0)
            print(f"  Epoch {epoch+1} | Acc: {correct/total:.4f}")

    # Interpretability analysis
    print(f"\n{'=' * 70}")
    print("FEATURE ATTRIBUTION ANALYSIS")
    print(f"{'=' * 70}")

    model = model.cpu().eval()

    # Get a test sample
    for X, y in test_loader:
        break
    x_sample = X[:1].view(1, -1)
    true_label = y[0].item()
    pred = model(x_sample).argmax(1).item()
    prob = F.softmax(model(x_sample), dim=-1)[0, pred].item()
    print(f"\nSample: true={true_label}, pred={pred}, prob={prob:.4f}")

    # Apply methods
    methods = {
        "Vanilla Gradient": vanilla_saliency(model, x_sample, pred),
        "SmoothGrad": smooth_grad(model, x_sample, pred),
        "Integrated Gradients": integrated_gradients(model, x_sample, pred).abs(),
        "Occlusion": occlusion_sensitivity(model, x_sample, pred),
    }

    print(f"\nTop-20 most important pixels per method:")
    for name, attr in methods.items():
        attr_flat = attr.view(-1)
        top20 = attr_flat.topk(20).indices.tolist()
        # Convert to (row, col)
        coords = [(p // 28, p % 28) for p in top20[:5]]
        total_attr = attr_flat.sum().item()
        print(f"  {name:>22s} | total={total_attr:.4f} | "
              f"top pixels: {coords}")

    # Analyze per-class
    print(f"\n{'=' * 70}")
    print("PER-DIGIT ATTRIBUTION ANALYSIS")
    print(f"{'=' * 70}")

    digit_importance = {}
    for X, y in test_loader:
        for i in range(min(100, len(X))):
            xi = X[i:i+1].view(1, -1)
            label = y[i].item()
            if label not in digit_importance:
                digit_importance[label] = []
            attr = vanilla_saliency(model, xi, label).view(-1)
            digit_importance[label].append(attr)
        break

    print(f"\n{'Digit':>6s} | {'Center %':>9s} | {'Edge %':>8s} | {'Total':>8s}")
    print("-" * 40)
    for digit in sorted(digit_importance.keys()):
        attrs = torch.stack(digit_importance[digit]).mean(0)
        total = attrs.sum().item()
        # Center region (rows 8-20, cols 8-20)
        center_mask = torch.zeros(784)
        for r in range(8, 20):
            for c in range(8, 20):
                center_mask[r * 28 + c] = 1
        center_pct = (attrs * center_mask).sum().item() / (total + 1e-8) * 100
        edge_pct = 100 - center_pct
        print(f"{digit:>6d} | {center_pct:>8.1f}% | {edge_pct:>7.1f}% | {total:>7.3f}")


if __name__ == "__main__":
    main()

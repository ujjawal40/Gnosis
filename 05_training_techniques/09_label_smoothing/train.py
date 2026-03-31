"""
Label Smoothing & Advanced Loss Experiments
===============================================

Compare loss functions on Fashion-MNIST.

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
from model import (SimpleMLP, LossConfig, LabelSmoothingCE,
                   FocalLoss, ConfidencePenaltyCE)


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


def train_and_eval(model, criterion, train_loader, test_loader, epochs, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            optimizer.step()

    # Evaluate accuracy and calibration
    model.eval()
    correct, total = 0, 0
    confidences, accuracies_per_bin = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            probs = F.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            confidences.extend(conf.cpu().tolist())

    acc = correct / total
    avg_conf = sum(confidences) / len(confidences)
    ece = abs(acc - avg_conf)  # Simplified ECE

    return acc, avg_conf, ece


def main():
    config = TrainConfig()
    device = get_device(config)

    print("=" * 70)
    print("LABEL SMOOTHING & ADVANCED LOSS EXPERIMENTS")
    print("=" * 70)

    loss_config = LossConfig()
    train_loader, test_loader = get_fashion_mnist(config.batch_size)

    criteria = {
        "Standard CE": nn.CrossEntropyLoss(),
        "Label Smooth (0.05)": LabelSmoothingCE(10, 0.05),
        "Label Smooth (0.1)": LabelSmoothingCE(10, 0.1),
        "Label Smooth (0.2)": LabelSmoothingCE(10, 0.2),
        "Focal (γ=1)": FocalLoss(gamma=1.0),
        "Focal (γ=2)": FocalLoss(gamma=2.0),
        "Confidence Pen": ConfidencePenaltyCE(beta=0.1),
    }

    print(f"\n{'Loss':>22s} | {'Accuracy':>8s} | {'Avg Conf':>9s} | {'ECE':>6s}")
    print("-" * 52)

    results = {}
    for name, criterion in criteria.items():
        torch.manual_seed(config.seed)
        model = SimpleMLP(loss_config).to(device)
        acc, avg_conf, ece = train_and_eval(
            model, criterion, train_loader, test_loader,
            config.epochs, config.lr, device
        )
        results[name] = {"acc": acc, "conf": avg_conf, "ece": ece}
        print(f"{name:>22s} | {acc:>7.4f}  | {avg_conf:>8.4f}  | {ece:>5.4f}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    best = max(results.items(), key=lambda x: x[1]["acc"])
    best_cal = min(results.items(), key=lambda x: x[1]["ece"])
    print(f"  Best accuracy:    {best[0]} ({best[1]['acc']:.4f})")
    print(f"  Best calibration: {best_cal[0]} (ECE={best_cal[1]['ece']:.4f})")


if __name__ == "__main__":
    main()

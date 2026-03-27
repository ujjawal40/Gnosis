"""
Contrastive Learning Experiments
===================================

Train SimCLR and Barlow Twins on Fashion-MNIST, then evaluate
with linear probing.

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
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model import (SimCLR, BarlowTwins, LinearEvaluator,
                   ContrastiveConfig)


@dataclass
class TrainConfig:
    pretrain_epochs: int = 20
    eval_epochs: int = 10
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


class AugmentedDataset(torch.utils.data.Dataset):
    """Dataset that returns two augmented views of each image."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return self.transform(img), self.transform(img), label


def get_fashion_mnist(bs, augment=False):
    base_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    train = datasets.FashionMNIST("./data", train=True, download=True, transform=base_t)
    test = datasets.FashionMNIST("./data", train=False, download=True, transform=base_t)

    if augment:
        aug_t = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.Normalize((0,), (1,)),  # identity (already normalized)
        ])
        train = AugmentedDataset(train, aug_t)

    return DataLoader(train, bs, shuffle=True, num_workers=2), \
           DataLoader(test, bs, num_workers=2)


def pretrain(model, train_loader, epochs, lr, device):
    """Pretrain contrastive model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x1, x2, _ = batch  # Ignore labels during pretraining
            x1, x2 = x1.to(device), x2.to(device)

            optimizer.zero_grad()
            loss = model(x1, x2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | Loss: {total_loss/n_batches:.4f}")


def linear_eval(encoder, hidden_dim, train_loader, test_loader,
                epochs, lr, device):
    """Linear evaluation protocol."""
    evaluator = LinearEvaluator(encoder, hidden_dim, 10).to(device)
    optimizer = torch.optim.Adam(evaluator.linear.parameters(), lr=lr)

    for epoch in range(epochs):
        evaluator.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            F.cross_entropy(evaluator(X), y).backward()
            optimizer.step()

    evaluator.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            correct += (evaluator(X).argmax(1) == y).sum().item()
            total += y.size(0)

    return correct / total


def supervised_baseline(train_loader, test_loader, epochs, lr, device, config):
    """Train a supervised model for comparison."""
    from model import ConvEncoder
    model = nn.Sequential(
        ConvEncoder(1, config.hidden_dim),
        nn.Linear(config.hidden_dim, 10)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            F.cross_entropy(model(X), y).backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            correct += (model(X).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    config = TrainConfig()
    device = get_device(config)
    torch.manual_seed(config.seed)
    cl_config = ContrastiveConfig()

    print("=" * 70)
    print("CONTRASTIVE LEARNING EXPERIMENTS")
    print("=" * 70)
    print(f"Dataset: Fashion-MNIST | Device: {device}")

    # Get data
    aug_train, _ = get_fashion_mnist(config.batch_size, augment=True)
    train_loader, test_loader = get_fashion_mnist(config.batch_size)

    results = {}

    # 1. SimCLR
    print("\n--- SimCLR Pretraining ---")
    torch.manual_seed(config.seed)
    simclr = SimCLR(cl_config).to(device)
    pretrain(simclr, aug_train, config.pretrain_epochs, config.lr, device)

    print("  Linear evaluation...")
    acc = linear_eval(simclr.encoder, cl_config.hidden_dim,
                      train_loader, test_loader, config.eval_epochs,
                      config.lr, device)
    results["SimCLR"] = acc
    print(f"  SimCLR linear probe accuracy: {acc:.4f}")

    # 2. Barlow Twins
    print("\n--- Barlow Twins Pretraining ---")
    torch.manual_seed(config.seed)
    barlow = BarlowTwins(cl_config).to(device)
    pretrain(barlow, aug_train, config.pretrain_epochs, config.lr, device)

    print("  Linear evaluation...")
    acc = linear_eval(barlow.encoder, cl_config.hidden_dim,
                      train_loader, test_loader, config.eval_epochs,
                      config.lr, device)
    results["Barlow Twins"] = acc
    print(f"  Barlow Twins linear probe accuracy: {acc:.4f}")

    # 3. Supervised baseline
    print("\n--- Supervised Baseline ---")
    torch.manual_seed(config.seed)
    acc = supervised_baseline(train_loader, test_loader,
                              config.pretrain_epochs + config.eval_epochs,
                              config.lr, device, cl_config)
    results["Supervised"] = acc
    print(f"  Supervised accuracy: {acc:.4f}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Method':>15s} | {'Accuracy':>8s}")
    print("-" * 28)
    for name, acc in results.items():
        print(f"{name:>15s} | {acc:>7.4f}")


if __name__ == "__main__":
    main()

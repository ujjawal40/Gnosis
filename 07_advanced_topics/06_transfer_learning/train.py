"""
Transfer Learning Training Pipeline
=======================================

Pretrain on MNIST, transfer to Fashion-MNIST to demonstrate
feature extraction, fine-tuning, and discriminative LR.

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
from model import TransferModel, TransferConfig


@dataclass
class TrainConfig:
    pretrain_epochs: int = 10
    transfer_epochs: int = 15
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


def get_fashion(bs):
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    train = datasets.FashionMNIST("./data", train=True, download=True, transform=t)
    test = datasets.FashionMNIST("./data", train=False, download=True, transform=t)
    return DataLoader(train, bs, shuffle=True, num_workers=2), DataLoader(test, bs, num_workers=2)


def train_loop(model, train_loader, test_loader, epochs, lr, device, label=""):
    model = model.to(device)
    if hasattr(model, 'get_param_groups') and model.config.discriminative_lr:
        optimizer = torch.optim.Adam(model.get_param_groups())
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                             model.parameters()), lr=lr)
    best_acc = 0
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
        acc = correct / total
        best_acc = max(best_acc, acc)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{label}] Epoch {epoch+1:3d} | Acc: {acc:.4f}")
    return best_acc


def main():
    config = TrainConfig()
    device = get_device(config)
    torch.manual_seed(config.seed)

    print("=" * 70)
    print("TRANSFER LEARNING: MNIST → FASHION-MNIST")
    print("=" * 70)

    mnist_train, mnist_test = get_mnist(config.batch_size)
    fmnist_train, fmnist_test = get_fashion(config.batch_size)

    # Step 1: Pretrain on MNIST
    print("\n--- Pretraining on MNIST ---")
    pretrained = TransferModel(TransferConfig())
    pretrain_acc = train_loop(pretrained, mnist_train, mnist_test,
                               config.pretrain_epochs, config.lr, device, "Pretrain")

    # Step 2: Transfer to Fashion-MNIST
    strategies = {}

    # 2a: From scratch
    print("\n--- From Scratch ---")
    torch.manual_seed(config.seed)
    scratch = TransferModel(TransferConfig())
    strategies["From scratch"] = train_loop(scratch, fmnist_train, fmnist_test,
                                            config.transfer_epochs, config.lr, device, "Scratch")

    # 2b: Feature extraction (frozen backbone)
    print("\n--- Feature Extraction ---")
    torch.manual_seed(config.seed)
    feat_model = TransferModel(TransferConfig(freeze_backbone=True))
    feat_model.backbone.load_state_dict(pretrained.backbone.state_dict())
    strategies["Feature extraction"] = train_loop(feat_model, fmnist_train, fmnist_test,
                                                   config.transfer_epochs, config.lr, device, "FeatExtr")

    # 2c: Fine-tuning
    print("\n--- Fine-tuning ---")
    torch.manual_seed(config.seed)
    ft_model = TransferModel(TransferConfig())
    ft_model.backbone.load_state_dict(pretrained.backbone.state_dict())
    strategies["Fine-tuning"] = train_loop(ft_model, fmnist_train, fmnist_test,
                                            config.transfer_epochs, config.lr * 0.1, device, "FineTune")

    # 2d: Discriminative LR
    print("\n--- Discriminative LR ---")
    torch.manual_seed(config.seed)
    dlr_model = TransferModel(TransferConfig(discriminative_lr=True, base_lr=config.lr))
    dlr_model.backbone.load_state_dict(pretrained.backbone.state_dict())
    strategies["Discriminative LR"] = train_loop(dlr_model, fmnist_train, fmnist_test,
                                                  config.transfer_epochs, config.lr, device, "DiscLR")

    print(f"\n{'=' * 70}")
    print(f"MNIST pretrain accuracy: {pretrain_acc:.4f}")
    print(f"\nFashion-MNIST Transfer Results:")
    print(f"{'Strategy':>22s} | {'Best Acc':>9s}")
    print("-" * 35)
    for name, acc in strategies.items():
        print(f"{name:>22s} | {acc:>8.4f}")


if __name__ == "__main__":
    main()

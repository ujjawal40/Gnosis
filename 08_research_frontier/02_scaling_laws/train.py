"""
Scaling Laws Experiments
==========================

Train models of varying sizes and data amounts, then fit scaling laws.

Usage:
    python train.py
"""

import sys
import os
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model import build_model, SIZE_CONFIGS


@dataclass
class TrainConfig:
    epochs_per_run: int = 15
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


def get_mnist(bs, subset_size=None):
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST("./data", train=True, download=True, transform=t)
    test = datasets.MNIST("./data", train=False, download=True, transform=t)
    if subset_size and subset_size < len(train):
        train = Subset(train, range(subset_size))
    return DataLoader(train, bs, shuffle=True, num_workers=2), DataLoader(test, bs, num_workers=2)


def train_and_eval(model, train_loader, test_loader, epochs, lr, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            F.cross_entropy(model(X), y).backward()
            optimizer.step()

    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += F.cross_entropy(logits, y, reduction='sum').item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total


def main():
    config = TrainConfig()
    device = get_device(config)

    print("=" * 70)
    print("SCALING LAW EXPERIMENTS")
    print("=" * 70)

    # Experiment 1: Scaling model size (fixed data)
    print("\n--- Model Size Scaling (fixed 60K data) ---")
    sizes = ["tiny", "small", "medium", "large", "xlarge"]
    train_loader, test_loader = get_mnist(config.batch_size)

    model_results = {}
    for size in sizes:
        torch.manual_seed(config.seed)
        model = build_model(size)
        n_params = model.n_params
        loss, acc = train_and_eval(model, train_loader, test_loader,
                                    config.epochs_per_run, config.lr, device)
        model_results[size] = {"params": n_params, "loss": loss, "acc": acc}
        print(f"  {size:>8s} | params: {n_params:>8,} | loss: {loss:.4f} | acc: {acc:.4f}")

    # Experiment 2: Scaling data size (fixed model)
    print("\n--- Data Size Scaling (fixed medium model) ---")
    data_sizes = [500, 1000, 2000, 5000, 10000, 30000, 60000]
    _, test_loader = get_mnist(config.batch_size)

    data_results = {}
    for ds in data_sizes:
        torch.manual_seed(config.seed)
        train_loader_sub, _ = get_mnist(config.batch_size, subset_size=ds)
        model = build_model("medium")
        loss, acc = train_and_eval(model, train_loader_sub, test_loader,
                                    config.epochs_per_run, config.lr, device)
        data_results[ds] = {"loss": loss, "acc": acc}
        print(f"  data: {ds:>6,} | loss: {loss:.4f} | acc: {acc:.4f}")

    # Fit power laws
    print(f"\n{'=' * 70}")
    print("SCALING LAW FITS")
    print(f"{'=' * 70}")

    params_arr = np.array([r["params"] for r in model_results.values()])
    losses_arr = np.array([r["loss"] for r in model_results.values()])

    log_p = np.log(params_arr)
    log_l = np.log(losses_arr)
    b = (len(log_p) * np.sum(log_p * log_l) - np.sum(log_p) * np.sum(log_l)) / \
        (len(log_p) * np.sum(log_p ** 2) - np.sum(log_p) ** 2)
    a = np.exp((np.sum(log_l) - b * np.sum(log_p)) / len(log_p))
    print(f"  L(N) = {a:.3f} * N^({b:.4f})")

    data_arr = np.array(list(data_results.keys()), dtype=float)
    losses_d = np.array([r["loss"] for r in data_results.values()])
    log_d = np.log(data_arr)
    log_ld = np.log(losses_d)
    b_d = (len(log_d) * np.sum(log_d * log_ld) - np.sum(log_d) * np.sum(log_ld)) / \
          (len(log_d) * np.sum(log_d ** 2) - np.sum(log_d) ** 2)
    a_d = np.exp((np.sum(log_ld) - b_d * np.sum(log_d)) / len(log_d))
    print(f"  L(D) = {a_d:.3f} * D^({b_d:.4f})")

    # Predictions
    print(f"\n  Predicted loss at 10M params: {a * (10e6 ** b):.4f}")
    print(f"  Predicted loss at 1M data:    {a_d * (1e6 ** b_d):.4f}")


if __name__ == "__main__":
    main()

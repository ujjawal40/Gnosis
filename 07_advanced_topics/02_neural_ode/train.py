"""
Neural ODE Training on Spiral Dataset
========================================

Learns continuous dynamics to classify clockwise vs counter-clockwise spirals.
Demonstrates how Neural ODE can learn smooth decision boundaries.

Usage:
    python train.py
    python train.py --n_steps 30 --hidden_dim 64 --epochs 200
"""

import sys
import os
import argparse
import time
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model import NeuralODE, NeuralODEConfig


@dataclass
class TrainConfig:
    hidden_dim: int = 64
    n_steps: int = 20
    solver: str = "rk4"
    lr: float = 0.01
    epochs: int = 200
    n_samples: int = 1000
    noise: float = 0.3
    seed: int = 42
    device: str = "auto"


def get_device(config: TrainConfig) -> torch.device:
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# SPIRAL DATASET
# =============================================================================

def make_spiral_data(n_samples: int = 1000, noise: float = 0.3,
                     device: torch.device = None) -> tuple:
    """
    Generate 2D spiral classification dataset.
    Two interleaved spirals (binary classification).
    """
    n = n_samples // 2
    theta = torch.linspace(0, 4 * np.pi, n)

    # Spiral 1 (class 0)
    r1 = theta / (4 * np.pi)
    x1 = r1 * torch.cos(theta) + noise * torch.randn(n)
    y1 = r1 * torch.sin(theta) + noise * torch.randn(n)

    # Spiral 2 (class 1) - rotated by pi
    x2 = r1 * torch.cos(theta + np.pi) + noise * torch.randn(n)
    y2 = r1 * torch.sin(theta + np.pi) + noise * torch.randn(n)

    X = torch.stack([torch.cat([x1, x2]), torch.cat([y1, y2])], dim=1)
    y = torch.cat([torch.zeros(n), torch.ones(n)]).long()

    # Shuffle
    perm = torch.randperm(n_samples)
    X, y = X[perm], y[perm]

    if device:
        X, y = X.to(device), y.to(device)

    return X, y


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    def __init__(self, model: NeuralODE, config: TrainConfig, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.history: Dict[str, List[float]] = {
            "loss": [], "acc": [],
        }

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor,
            X_test: torch.Tensor, y_test: torch.Tensor):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs)
        loss_fn = nn.CrossEntropyLoss()

        print(f"  Parameters: {self.model.count_parameters():,}")
        print(f"  ODE solver: {self.config.solver}, steps: {self.config.n_steps}")
        print()

        for epoch in range(self.config.epochs):
            self.model.train()
            optimizer.zero_grad()
            logits = self.model(X_train)
            loss = loss_fn(logits, y_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Evaluate
            self.model.eval()
            with torch.no_grad():
                test_logits = self.model(X_test)
                test_acc = (test_logits.argmax(dim=1) == y_test).float().mean().item()
                train_acc = (logits.argmax(dim=1) == y_train).float().mean().item()

            self.history["loss"].append(loss.item())
            self.history["acc"].append(test_acc)

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{self.config.epochs}: "
                      f"loss={loss.item():.4f} train_acc={train_acc:.4f} "
                      f"test_acc={test_acc:.4f}")

        print(f"\n  Final test accuracy: {self.history['acc'][-1]:.4f}")

        # Compare Euler vs RK4
        self.compare_solvers(X_test, y_test)

    def compare_solvers(self, X: torch.Tensor, y: torch.Tensor):
        """Compare Euler vs RK4 accuracy at different step counts."""
        print("\n  Solver comparison (same trained weights):")

        original_solver = self.model.config.solver
        original_steps = self.model.config.n_steps

        for solver in ["euler", "rk4"]:
            for steps in [5, 10, 20, 50]:
                self.model.config.solver = solver
                self.model.config.n_steps = steps

                with torch.no_grad():
                    logits = self.model(X)
                    acc = (logits.argmax(dim=1) == y).float().mean().item()
                print(f"    {solver:5s} (steps={steps:2d}): acc={acc:.4f}")

        self.model.config.solver = original_solver
        self.model.config.n_steps = original_steps
        print()


def main():
    parser = argparse.ArgumentParser(description="Neural ODE on Spirals")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--solver", default="rk4", choices=["euler", "rk4"])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(**vars(args))
    torch.manual_seed(config.seed)
    device = get_device(config)

    print("=" * 60)
    print("NEURAL ODE ON SPIRAL DATASET")
    print(f"Device: {device}")
    print("=" * 60)

    X, y = make_spiral_data(config.n_samples, config.noise, device)

    # Train/test split
    n_train = int(0.8 * len(y))
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    print(f"  Train: {n_train}, Test: {len(y) - n_train}")

    model_config = NeuralODEConfig(
        input_dim=2, hidden_dim=config.hidden_dim, output_dim=2,
        n_steps=config.n_steps, solver=config.solver,
    )
    model = NeuralODE(model_config)

    trainer = Trainer(model, config, device)
    trainer.fit(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()

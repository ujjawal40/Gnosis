"""
Geometric Transformer Training on Synthetic Point Cloud Data
===============================================================

Classifies 3D point cloud shapes using distance-aware attention.
Demonstrates how geometric inductive biases improve learning.

Usage:
    python train.py
    python train.py --n_layers 4 --hidden_dim 64 --epochs 100
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
from model import GeometricTransformer, GeometricConfig


@dataclass
class TrainConfig:
    hidden_dim: int = 64
    n_heads: int = 4
    n_layers: int = 3
    lr: float = 0.001
    epochs: int = 100
    n_samples: int = 2000
    n_points: int = 50     # Points per cloud
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
# SYNTHETIC POINT CLOUD DATASET
# =============================================================================

def make_point_clouds(n_samples: int = 2000, n_points: int = 50,
                      device: torch.device = None) -> tuple:
    """
    Generate synthetic 3D point clouds for classification.

    Classes:
        0: Sphere (points on unit sphere surface)
        1: Cube (points on unit cube surface)
        2: Torus (points on torus surface)
        3: Spiral (3D spiral)
    """
    n_per_class = n_samples // 4
    clouds = []
    labels = []

    for cls in range(4):
        for _ in range(n_per_class):
            if cls == 0:  # Sphere
                phi = torch.rand(n_points) * 2 * np.pi
                theta = torch.acos(2 * torch.rand(n_points) - 1)
                x = torch.sin(theta) * torch.cos(phi)
                y = torch.sin(theta) * torch.sin(phi)
                z = torch.cos(theta)

            elif cls == 1:  # Cube
                face = torch.randint(0, 6, (n_points,))
                u = torch.rand(n_points) * 2 - 1
                v = torch.rand(n_points) * 2 - 1
                x = torch.where(face < 2, torch.where(face == 0, torch.ones(n_points), -torch.ones(n_points)), u)
                y = torch.where((face >= 2) & (face < 4), torch.where(face == 2, torch.ones(n_points), -torch.ones(n_points)), v)
                z = torch.where(face >= 4, torch.where(face == 4, torch.ones(n_points), -torch.ones(n_points)), torch.rand(n_points) * 2 - 1)

            elif cls == 2:  # Torus (R=1, r=0.3)
                phi = torch.rand(n_points) * 2 * np.pi
                theta = torch.rand(n_points) * 2 * np.pi
                R, r = 1.0, 0.3
                x = (R + r * torch.cos(theta)) * torch.cos(phi)
                y = (R + r * torch.cos(theta)) * torch.sin(phi)
                z = r * torch.sin(theta)

            else:  # 3D Spiral
                t = torch.linspace(0, 4 * np.pi, n_points) + 0.1 * torch.randn(n_points)
                x = t / (4 * np.pi) * torch.cos(t)
                y = t / (4 * np.pi) * torch.sin(t)
                z = t / (4 * np.pi)

            # Add noise
            noise = 0.05 * torch.randn(n_points, 3)
            points = torch.stack([x, y, z], dim=1) + noise
            clouds.append(points)
            labels.append(cls)

    X = torch.stack(clouds)
    y = torch.tensor(labels, dtype=torch.long)

    # Shuffle
    perm = torch.randperm(len(y))
    X, y = X[perm], y[perm]

    if device:
        X, y = X.to(device), y.to(device)

    return X, y


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    def __init__(self, model: GeometricTransformer, config: TrainConfig,
                 device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.history: Dict[str, List[float]] = {"loss": [], "acc": []}

    def fit(self, X_train, y_train, X_test, y_test):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs)
        loss_fn = nn.CrossEntropyLoss()

        print(f"  Parameters: {self.model.count_parameters():,}")
        print(f"  Shapes: {X_train.shape[0]} classes × {X_train.shape[1]} points × 3D")
        print()

        best_acc = 0.0
        for epoch in range(self.config.epochs):
            # Train (full batch - small dataset)
            self.model.train()
            optimizer.zero_grad()
            logits = self.model(X_train, positions=X_train)
            loss = loss_fn(logits, y_train)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Eval
            self.model.eval()
            with torch.no_grad():
                test_logits = self.model(X_test, positions=X_test)
                test_acc = (test_logits.argmax(dim=1) == y_test).float().mean().item()

            self.history["loss"].append(loss.item())
            self.history["acc"].append(test_acc)
            best_acc = max(best_acc, test_acc)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{self.config.epochs}: "
                      f"loss={loss.item():.4f} test_acc={test_acc:.4f}")

        print(f"\n  Best test accuracy: {best_acc:.4f}")

        # Compare with vs without distance attention
        self.ablation_study(X_test, y_test)

    def ablation_study(self, X_test, y_test):
        """Compare geometric attention vs standard attention."""
        print("\n  Ablation: Geometric vs Standard attention:")

        # Current model (with distance)
        self.model.eval()
        with torch.no_grad():
            logits_geo = self.model(X_test, positions=X_test)
            acc_geo = (logits_geo.argmax(1) == y_test).float().mean().item()

            # Without positions (standard attention)
            logits_std = self.model(X_test, positions=None)
            acc_std = (logits_std.argmax(1) == y_test).float().mean().item()

        print(f"    With distance-aware attention: {acc_geo:.4f}")
        print(f"    Without distance (standard):   {acc_std:.4f}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Geometric Transformer")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--n_points", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(**vars(args))
    torch.manual_seed(config.seed)
    device = get_device(config)

    print("=" * 60)
    print("GEOMETRIC TRANSFORMER ON 3D POINT CLOUDS")
    print(f"Device: {device}")
    print("=" * 60)

    X, y = make_point_clouds(config.n_samples, config.n_points, device)

    n_train = int(0.8 * len(y))
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    model_config = GeometricConfig(
        node_dim=3, hidden_dim=config.hidden_dim, output_dim=4,
        n_heads=config.n_heads, n_layers=config.n_layers,
    )
    model = GeometricTransformer(model_config)

    trainer = Trainer(model, config, device)
    trainer.fit(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()

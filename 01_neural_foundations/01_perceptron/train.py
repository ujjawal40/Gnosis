"""
Perceptron Training Pipeline
==============================

Trains perceptron models on:
1. Boolean gates (AND, OR, XOR) - demonstrates XOR failure
2. California Housing (regression with single linear unit)

Usage:
    python train.py
    python train.py --task housing --epochs 100
"""

import sys
import os
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model import Perceptron, SmoothPerceptron, MultiClassPerceptron


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    task: str = "boolean"         # "boolean" or "housing"
    lr: float = 0.01
    epochs: int = 200
    batch_size: int = 32
    seed: int = 42
    device: str = "auto"


def get_device(config: Config) -> torch.device:
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# DATASETS
# =============================================================================

def get_boolean_data():
    """AND, OR, XOR gate datasets."""
    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    gates = {
        "AND": torch.tensor([[0.], [0.], [0.], [1.]]),
        "OR":  torch.tensor([[0.], [1.], [1.], [1.]]),
        "XOR": torch.tensor([[0.], [1.], [1.], [0.]]),
    }
    return X, gates


def get_housing_data():
    """Load California Housing as PyTorch tensors."""
    from datasets.tabular_datasets import load_california_housing
    X_train, y_train, X_test, y_test = load_california_housing()
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """Training loop with logging and evaluation."""

    def __init__(self, model: nn.Module, config: Config, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.history = {"train_loss": [], "val_loss": []}

    def train_boolean_gates(self):
        """Train perceptron on AND/OR/XOR using Rosenblatt rule."""
        X, gates = get_boolean_data()
        X = X.to(self.device)

        print("=" * 50)
        print("BOOLEAN GATES (Rosenblatt Learning Rule)")
        print("=" * 50)

        for gate_name, y in gates.items():
            y = y.to(self.device)
            model = Perceptron(n_features=2).to(self.device)

            converged = False
            for epoch in range(self.config.epochs):
                model.perceptron_update(X, y, lr=self.config.lr)
                preds = model.predict(X)
                accuracy = (preds == y).float().mean().item()

                if accuracy == 1.0:
                    print(f"  {gate_name}: Converged at epoch {epoch+1}")
                    converged = True
                    break

            if not converged:
                preds = model.predict(X)
                accuracy = (preds == y).float().mean().item()
                print(f"  {gate_name}: Failed to converge. Accuracy={accuracy:.2f}")
                if gate_name == "XOR":
                    print(f"    → XOR is NOT linearly separable (expected failure)")

            # Show predictions
            with torch.no_grad():
                for i in range(4):
                    print(f"    {X[i].tolist()} → {preds[i].item():.0f} "
                          f"(target: {y[i].item():.0f})")
        print()

    def train_smooth_boolean(self):
        """Train smooth perceptron with backprop on AND gate."""
        print("=" * 50)
        print("SMOOTH PERCEPTRON (Sigmoid + Backprop)")
        print("=" * 50)

        X, gates = get_boolean_data()
        X, y = X.to(self.device), gates["AND"].to(self.device)

        model = SmoothPerceptron(n_features=2).to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        loss_fn = nn.BCELoss()

        for epoch in range(self.config.epochs):
            optimizer.zero_grad()
            preds = model(X)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1:3d}: loss={loss.item():.4f}")

        with torch.no_grad():
            preds = model(X)
            print(f"  Final predictions:")
            for i in range(4):
                print(f"    {X[i].tolist()} → {preds[i].item():.4f} "
                      f"(target: {y[i].item():.0f})")
        print()

    def train_housing(self):
        """Train single-layer model on California Housing regression."""
        print("=" * 50)
        print("CALIFORNIA HOUSING (Linear Regression)")
        print("=" * 50)

        X_train, y_train, X_test, y_test = get_housing_data()
        X_train, y_train = X_train.to(self.device), y_train.to(self.device)
        X_test, y_test = X_test.to(self.device), y_test.to(self.device)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size,
                                  shuffle=True)

        model = nn.Linear(X_train.shape[1], 1).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        loss_fn = nn.MSELoss()

        for epoch in range(self.config.epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                preds = model(X_batch).squeeze()
                loss = loss_fn(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 20 == 0:
                model.eval()
                with torch.no_grad():
                    train_loss = epoch_loss / n_batches
                    val_preds = model(X_test).squeeze()
                    val_loss = loss_fn(val_preds, y_test).item()
                    self.history["train_loss"].append(train_loss)
                    self.history["val_loss"].append(val_loss)
                    print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
                          f"val_loss={val_loss:.4f}")

        # Final evaluation
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test).squeeze()
            mse = loss_fn(test_preds, y_test).item()
            rmse = mse ** 0.5
            # R² score
            ss_res = ((y_test - test_preds) ** 2).sum()
            ss_tot = ((y_test - y_test.mean()) ** 2).sum()
            r2 = 1 - (ss_res / ss_tot).item()
            print(f"\n  Test RMSE: {rmse:.4f}")
            print(f"  Test R²:   {r2:.4f}")
        print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Perceptron Training")
    parser.add_argument("--task", default="all", choices=["boolean", "housing", "all"])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Config(
        task=args.task, lr=args.lr, epochs=args.epochs,
        batch_size=args.batch_size, seed=args.seed,
    )

    torch.manual_seed(config.seed)
    device = get_device(config)
    print(f"Device: {device}")

    trainer = Trainer(model=nn.Identity(), config=config, device=device)

    if config.task in ("boolean", "all"):
        trainer.train_boolean_gates()
        trainer.train_smooth_boolean()

    if config.task in ("housing", "all"):
        trainer.train_housing()


if __name__ == "__main__":
    main()

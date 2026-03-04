"""
Activation Function Comparison Training
==========================================

Trains the same architecture with different activation functions on
Forest CoverType to compare convergence speed, final accuracy, and
gradient flow properties.

Usage:
    python train.py
    python train.py --activations relu,gelu,silu --epochs 30
"""

import sys
import os
import argparse
from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model import ActivationTestNet, activation_profile, dead_neuron_rate, ACTIVATIONS


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    activations: str = "relu,leaky_relu,elu,gelu,silu,tanh"
    lr: float = 0.001
    epochs: int = 20
    batch_size: int = 256
    hidden_dim: int = 128
    n_layers: int = 3
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
# TRAINER
# =============================================================================

class ActivationComparisonTrainer:
    """Trains multiple models with different activations and compares results."""

    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.results: Dict[str, dict] = {}

    def load_data(self):
        """Load Forest CoverType dataset."""
        from datasets.tabular_datasets import load_covertype

        X_train, y_train, X_test, y_test = load_covertype()

        self.train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long),
            ),
            batch_size=self.config.batch_size, shuffle=True,
            num_workers=2, pin_memory=True,
        )
        self.test_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long),
            ),
            batch_size=self.config.batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        )
        self.input_dim = X_train.shape[1]
        self.n_classes = 7

    def train_single(self, activation: str) -> dict:
        """Train one model and return metrics."""
        torch.manual_seed(self.config.seed)

        model = ActivationTestNet(
            input_dim=self.input_dim,
            hidden_dim=self.config.hidden_dim,
            n_classes=self.n_classes,
            n_layers=self.config.n_layers,
            activation=activation,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        loss_fn = nn.CrossEntropyLoss()

        train_losses, val_accs = [], []

        for epoch in range(self.config.epochs):
            # Train
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for X_b, y_b in self.train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(model(X_b), y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            train_losses.append(epoch_loss / n_batches)

            # Eval
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X_b, y_b in self.test_loader:
                    X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                    preds = model(X_b).argmax(dim=1)
                    correct += (preds == y_b).sum().item()
                    total += y_b.size(0)
            val_accs.append(correct / total)

        # Dead neuron analysis
        sample_batch = next(iter(self.test_loader))[0].to(self.device)
        dead_rate = dead_neuron_rate(model, sample_batch)

        return {
            "train_losses": train_losses,
            "val_accs": val_accs,
            "best_acc": max(val_accs),
            "final_acc": val_accs[-1],
            "dead_neuron_rate": dead_rate,
            "n_params": sum(p.numel() for p in model.parameters()),
        }

    def run_comparison(self):
        """Train all activations and print comparison."""
        activations = self.config.activations.split(",")

        print("=" * 70)
        print("ACTIVATION FUNCTION COMPARISON")
        print(f"Architecture: {self.input_dim} → "
              f"[{self.config.hidden_dim}] × {self.config.n_layers} → {self.n_classes}")
        print(f"Dataset: Forest CoverType (581K samples, 7 classes)")
        print("=" * 70)

        for act in activations:
            print(f"\n  Training with {act}...")
            self.results[act] = self.train_single(act)

        # Summary table
        print("\n" + "=" * 70)
        print(f"{'Activation':<15} {'Best Acc':>10} {'Final Acc':>10} "
              f"{'Dead Neurons':>13} {'Final Loss':>11}")
        print("-" * 70)

        for act in activations:
            r = self.results[act]
            print(f"{act:<15} {r['best_acc']:>10.4f} {r['final_acc']:>10.4f} "
                  f"{r['dead_neuron_rate']:>12.4f} {r['train_losses'][-1]:>11.4f}")

        print()

    def print_activation_properties(self):
        """Print mathematical properties of each activation."""
        print("=" * 70)
        print("ACTIVATION FUNCTION PROPERTIES")
        print("=" * 70)

        activations = self.config.activations.split(",")
        for act in activations:
            x, y, dy = activation_profile(act)

            # Properties
            zero_grad = (dy.abs() < 1e-6).float().mean().item()
            max_grad = dy.max().item()
            min_val = y.min().item()
            max_val = y.max().item()

            print(f"  {act:<15}: output=[{min_val:.2f}, {max_val:.2f}], "
                  f"max_grad={max_grad:.4f}, zero_grad_pct={zero_grad*100:.1f}%")
        print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Activation Function Comparison")
    parser.add_argument("--activations", default="relu,leaky_relu,elu,gelu,silu,tanh")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Config(**vars(args))
    device = get_device(config)
    print(f"Device: {device}\n")

    trainer = ActivationComparisonTrainer(config, device)
    trainer.load_data()
    trainer.print_activation_properties()
    trainer.run_comparison()


if __name__ == "__main__":
    main()

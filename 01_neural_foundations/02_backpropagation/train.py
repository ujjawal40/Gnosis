"""
Backpropagation Training Pipeline
===================================

Trains MLP on:
1. XOR gate (demonstrates need for hidden layers)
2. Forest CoverType (581K samples, 7-class classification)

Shows gradient flow analysis, learning dynamics, and convergence.

Usage:
    python train.py
    python train.py --task covertype --epochs 50 --batch_size 256
"""

import sys
import os
import argparse
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model import MLP, XORNet


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    task: str = "all"
    lr: float = 0.001
    epochs: int = 50
    batch_size: int = 256
    hidden_sizes: str = "256,128,64"
    dropout: float = 0.1
    activation: str = "relu"
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

class Trainer:
    """Industry-standard training loop with logging, checkpointing, gradient analysis."""

    def __init__(self, model: nn.Module, config: Config, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
        }
        self.gradient_history: List[dict] = []

    def train_epoch(self, loader: DataLoader, optimizer, loss_fn,
                    track_gradients: bool = False) -> tuple:
        """Train for one epoch, return (avg_loss, accuracy)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()

            if track_gradients:
                self.gradient_history.append(self.model.get_gradient_norms())

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, loss_fn) -> tuple:
        """Evaluate model, return (avg_loss, accuracy)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            logits = self.model(X_batch)
            loss = loss_fn(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

        return total_loss / total, correct / total

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            optimizer, scheduler=None, loss_fn=None):
        """Full training loop with validation and early stopping."""
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, loss_fn,
                track_gradients=(epoch == 0),
            )
            val_loss, val_acc = self.evaluate(val_loader, loss_fn)

            if scheduler:
                scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch+1:3d}/{self.config.epochs}: "
                      f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                      f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

    def print_gradient_analysis(self):
        """Print gradient norm analysis from first epoch."""
        if not self.gradient_history:
            return

        print("\n  Gradient norms (first batch):")
        first_batch = self.gradient_history[0]
        for name, norm in first_batch.items():
            status = "⚠ VANISHING" if norm < 1e-6 else \
                     "⚠ EXPLODING" if norm > 100 else "OK"
            print(f"    {name:40s}: {norm:.6f} [{status}]")


# =============================================================================
# TASK: XOR
# =============================================================================

def train_xor(config: Config, device: torch.device):
    """Train XORNet to solve XOR problem."""
    print("=" * 60)
    print("XOR (Demonstrating need for hidden layers)")
    print("=" * 60)

    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], device=device)
    y = torch.tensor([[0.], [1.], [1.], [0.]], device=device)

    model = XORNet(hidden_size=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    for epoch in range(1000):
        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.6f}")

    with torch.no_grad():
        preds = model(X)
        print(f"\n  Final predictions:")
        for i in range(4):
            print(f"    {X[i].tolist()} → {preds[i].item():.4f} "
                  f"(target: {y[i].item():.0f})")

    # Show that single layer CANNOT solve XOR
    print(f"\n  Single-layer attempt:")
    linear = nn.Linear(2, 1).to(device)
    opt2 = torch.optim.Adam(linear.parameters(), lr=0.01)
    for _ in range(1000):
        opt2.zero_grad()
        p = torch.sigmoid(linear(X))
        loss = loss_fn(p, y)
        loss.backward()
        opt2.step()

    with torch.no_grad():
        p = torch.sigmoid(linear(X))
        print(f"    Loss after 1000 epochs: {loss.item():.4f} (can't go below ~0.693)")
        print(f"    → Single layer fails on XOR (not linearly separable)")
    print()


# =============================================================================
# TASK: FOREST COVERTYPE
# =============================================================================

def train_covertype(config: Config, device: torch.device):
    """Train MLP on Forest CoverType (581K samples, 7 classes)."""
    print("=" * 60)
    print("FOREST COVERTYPE (7-class, 54 features, 581K samples)")
    print("=" * 60)

    from datasets.tabular_datasets import load_covertype

    X_train, y_train, X_test, y_test = load_covertype()

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    hidden = [int(x) for x in config.hidden_sizes.split(",")]
    layer_sizes = [54] + hidden + [7]

    model = MLP(
        layer_sizes=layer_sizes,
        activation=config.activation,
        dropout=config.dropout,
        batch_norm=True,
    )
    print(f"  Model: {layer_sizes}")
    print(f"  Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs)

    trainer = Trainer(model, config, device)
    trainer.fit(train_loader, test_loader, optimizer, scheduler)
    trainer.print_gradient_analysis()

    print(f"\n  Best val accuracy: {max(trainer.history['val_acc']):.4f}")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Backprop/MLP Training")
    parser.add_argument("--task", default="all", choices=["xor", "covertype", "all"])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_sizes", default="256,128,64")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Config(**vars(args))
    torch.manual_seed(config.seed)
    device = get_device(config)
    print(f"Device: {device}\n")

    if config.task in ("xor", "all"):
        train_xor(config, device)
    if config.task in ("covertype", "all"):
        train_covertype(config, device)


if __name__ == "__main__":
    main()

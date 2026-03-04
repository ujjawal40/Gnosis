"""
Deep MLP Training on Higgs Boson Dataset
==========================================

Binary classification on the UCI Higgs Boson dataset (up to 11M samples).
Particle physics: distinguish signal (Higgs boson) from background events.

Features:
    - DataLoader with proper batching
    - Cosine annealing with warmup
    - Early stopping
    - Gradient clipping
    - Checkpointing
    - Metrics logging

Usage:
    python train.py
    python train.py --n_samples 1000000 --epochs 30 --batch_size 512
"""

import sys
import os
import argparse
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model import DeepMLP, MLPConfig


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TrainConfig:
    # Data
    n_samples: int = 500_000
    test_size: float = 0.2
    # Model
    hidden_dims: str = "512,256,128,64"
    dropout: float = 0.1
    activation: str = "relu"
    residual: bool = True
    # Training
    lr: float = 0.001
    weight_decay: float = 1e-4
    epochs: int = 30
    batch_size: int = 512
    warmup_epochs: int = 3
    # System
    seed: int = 42
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    patience: int = 7


def get_device(config: TrainConfig) -> torch.device:
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# DATASET
# =============================================================================

class HiggsDataset(torch.utils.data.Dataset):
    """Wrapper for Higgs Boson data as a proper PyTorch Dataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """Production training loop for the MLP."""

    def __init__(self, model: nn.Module, config: TrainConfig, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "lr": [],
        }

    def train_one_epoch(self, loader: DataLoader, optimizer,
                        loss_fn: nn.Module) -> tuple:
        """Train for one epoch. Returns (avg_loss, accuracy)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, loss_fn: nn.Module) -> tuple:
        """Evaluate. Returns (avg_loss, accuracy, auc)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            logits = self.model(X_batch)
            loss = loss_fn(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.cpu())
            all_labels.append(y_batch.cpu())

        return total_loss / total, correct / total

    def save_checkpoint(self, path: str, epoch: int, optimizer):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }, path)

    def load_checkpoint(self, path: str, optimizer=None) -> int:
        """Load checkpoint. Returns epoch."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if optimizer:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_val_loss = ckpt["best_val_loss"]
        self.history = ckpt["history"]
        return ckpt["epoch"]

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop."""
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # Warmup + cosine schedule
        warmup_steps = self.config.warmup_epochs * len(train_loader)
        total_steps = self.config.epochs * len(train_loader)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        print(f"  Parameters: {self.model.count_parameters():,}")
        print(f"  Train samples: {len(train_loader.dataset):,}")
        print(f"  Val samples: {len(val_loader.dataset):,}")
        print(f"  Batches/epoch: {len(train_loader)}")
        print()

        for epoch in range(self.config.epochs):
            start_time = time.time()

            train_loss, train_acc = self.train_one_epoch(
                train_loader, optimizer, loss_fn)
            # Step scheduler per batch is handled in train_one_epoch
            # For simplicity, step once per epoch here
            scheduler.step()

            val_loss, val_acc = self.evaluate(val_loader, loss_fn)

            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(lr)

            print(f"  Epoch {epoch+1:3d}/{self.config.epochs} ({elapsed:.1f}s): "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr:.6f}")

            # Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.save_checkpoint(
                    os.path.join(self.config.checkpoint_dir, "best_model.pt"),
                    epoch, optimizer,
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        print(f"\n  Best val accuracy: {self.best_val_acc:.4f}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MLP on Higgs Boson")
    parser.add_argument("--n_samples", type=int, default=500_000)
    parser.add_argument("--hidden_dims", default="512,256,128,64")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()

    config = TrainConfig(**vars(args))
    torch.manual_seed(config.seed)
    device = get_device(config)

    print("=" * 60)
    print("DEEP MLP ON HIGGS BOSON")
    print(f"Device: {device}")
    print("=" * 60)

    # Load data
    from datasets.tabular_datasets import load_higgs
    X_train, y_train, X_test, y_test = load_higgs(n_samples=config.n_samples)

    train_loader = DataLoader(
        HiggsDataset(X_train, y_train),
        batch_size=config.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        HiggsDataset(X_test, y_test),
        batch_size=config.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Build model
    hidden = [int(x) for x in config.hidden_dims.split(",")]
    model_config = MLPConfig(
        input_dim=X_train.shape[1],
        hidden_dims=hidden,
        output_dim=2,
        activation=config.activation,
        dropout=config.dropout,
        residual=True,
    )

    model = DeepMLP(model_config)
    print(f"\n  Architecture: {X_train.shape[1]} → {hidden} → 2")

    # Train
    trainer = Trainer(model, config, device)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()

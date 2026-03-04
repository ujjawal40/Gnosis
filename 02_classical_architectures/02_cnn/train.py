"""
CNN Training on CIFAR-100
===========================

100-class image classification with data augmentation, cosine LR,
mixup, and gradient clipping.

Usage:
    python train.py
    python train.py --epochs 50 --batch_size 128 --lr 0.1
"""

import sys
import os
import argparse
import time
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model import CNN, CNNConfig


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 128
    lr: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    channels: str = "64,128,256"
    blocks: str = "2,2,2"
    dropout: float = 0.0
    mixup_alpha: float = 0.2
    seed: int = 42
    device: str = "auto"
    patience: int = 15


def get_device(config: TrainConfig) -> torch.device:
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# MIXUP AUGMENTATION
# =============================================================================

def mixup_data(x, y, alpha=0.2):
    """Mixup: convex combinations of training examples."""
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss: weighted combination of losses."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """CNN training loop with mixup, cosine LR, and checkpointing."""

    def __init__(self, model: nn.Module, config: TrainConfig, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.best_acc = 0.0
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
        }

    def train_one_epoch(self, loader, optimizer, loss_fn) -> tuple:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            # Mixup
            if self.config.mixup_alpha > 0:
                X_mixed, y_a, y_b, lam = mixup_data(
                    X_batch, y_batch, self.config.mixup_alpha)
                logits = self.model(X_mixed)
                loss = mixup_criterion(loss_fn, logits, y_a, y_b, lam)
            else:
                logits = self.model(X_batch)
                loss = loss_fn(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            # Accuracy on unmixed labels
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, loader, loss_fn) -> tuple:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            logits = self.model(X_batch)
            loss = loss_fn(logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

        return total_loss / total, correct / total

    def fit(self, train_loader, val_loader):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs)

        print(f"  Parameters: {self.model.count_parameters():,}")
        print(f"  Batches/epoch: {len(train_loader)}")
        print()

        patience_counter = 0

        for epoch in range(self.config.epochs):
            start = time.time()
            train_loss, train_acc = self.train_one_epoch(
                train_loader, optimizer, loss_fn)
            val_loss, val_acc = self.evaluate(val_loader, loss_fn)
            scheduler.step()

            elapsed = time.time() - start
            lr = optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            marker = ""
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                patience_counter = 0
                marker = " ★"
            else:
                patience_counter += 1

            print(f"  Epoch {epoch+1:3d}/{self.config.epochs} ({elapsed:.1f}s): "
                  f"train={train_loss:.4f}/{train_acc:.4f} "
                  f"val={val_loss:.4f}/{val_acc:.4f} lr={lr:.4f}{marker}")

            if patience_counter >= self.config.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        print(f"\n  Best validation accuracy: {self.best_acc:.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CNN on CIFAR-100")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--channels", default="64,128,256")
    parser.add_argument("--blocks", default="2,2,2")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(**vars(args))
    torch.manual_seed(config.seed)
    device = get_device(config)

    print("=" * 60)
    print("CNN ON CIFAR-100 (100 classes)")
    print(f"Device: {device}")
    print("=" * 60)

    from datasets.image_datasets import load_cifar100
    train_loader, val_loader = load_cifar100(
        batch_size=config.batch_size, num_workers=2)

    channels = [int(x) for x in config.channels.split(",")]
    blocks = [int(x) for x in config.blocks.split(",")]

    model_config = CNNConfig(
        in_channels=3, n_classes=100,
        channels=channels, blocks_per_stage=blocks,
        dropout=config.dropout,
    )
    model = CNN(model_config)
    print(f"\n  Architecture: channels={channels}, blocks={blocks}")

    trainer = Trainer(model, config, device)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()

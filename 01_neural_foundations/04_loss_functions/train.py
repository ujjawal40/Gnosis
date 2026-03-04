"""
Loss Function Comparison Training
====================================

Compares different loss functions by training the same model on
Forest CoverType and analyzing convergence and gradient behavior.

Usage:
    python train.py
    python train.py --losses ce,focal,label_smooth --epochs 20
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model import (LossComparisonNet, FocalLoss, LabelSmoothingCE,
                   LogCoshLoss, gradient_analysis)


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    losses: str = "ce,focal,label_smooth,nll"
    lr: float = 0.001
    epochs: int = 20
    batch_size: int = 256
    hidden_dim: int = 128
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
# LOSS REGISTRY
# =============================================================================

LOSS_FUNCTIONS = {
    "ce":           ("CrossEntropy",       nn.CrossEntropyLoss()),
    "focal":        ("Focal(γ=2)",         FocalLoss(gamma=2.0)),
    "focal_g5":     ("Focal(γ=5)",         FocalLoss(gamma=5.0)),
    "label_smooth": ("LabelSmoothing(0.1)", LabelSmoothingCE(smoothing=0.1)),
    "nll":          ("NLLLoss",            nn.NLLLoss()),
}


# =============================================================================
# TRAINER
# =============================================================================

class LossComparisonTrainer:
    """Trains same model with different losses and compares."""

    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.results: Dict[str, dict] = {}

    def load_data(self):
        """Load Forest CoverType."""
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

    def train_single(self, loss_key: str) -> dict:
        """Train one model with a specific loss function."""
        torch.manual_seed(self.config.seed)

        loss_name, loss_fn = LOSS_FUNCTIONS[loss_key]
        needs_log_softmax = (loss_key == "nll")

        model = LossComparisonNet(
            input_dim=self.input_dim,
            hidden_dim=self.config.hidden_dim,
            n_classes=self.n_classes,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        train_losses, val_accs = [], []

        for epoch in range(self.config.epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_b, y_b in self.train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                logits = model(X_b)

                if needs_log_softmax:
                    loss = loss_fn(F.log_softmax(logits, dim=1), y_b)
                else:
                    loss = loss_fn(logits, y_b)

                loss.backward()
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

        return {
            "name": loss_name,
            "train_losses": train_losses,
            "val_accs": val_accs,
            "best_acc": max(val_accs),
            "final_acc": val_accs[-1],
        }

    def run_comparison(self):
        """Train all loss functions and print comparison."""
        losses = self.config.losses.split(",")

        print("=" * 65)
        print("LOSS FUNCTION COMPARISON")
        print(f"Dataset: Forest CoverType | Epochs: {self.config.epochs}")
        print("=" * 65)

        for loss_key in losses:
            if loss_key not in LOSS_FUNCTIONS:
                print(f"  Unknown loss: {loss_key}, skipping")
                continue
            name = LOSS_FUNCTIONS[loss_key][0]
            print(f"\n  Training with {name}...")
            self.results[loss_key] = self.train_single(loss_key)

            # Print last few epochs
            r = self.results[loss_key]
            for e in range(max(0, self.config.epochs - 3), self.config.epochs):
                print(f"    Epoch {e+1}: loss={r['train_losses'][e]:.4f}, "
                      f"acc={r['val_accs'][e]:.4f}")

        # Summary
        print("\n" + "=" * 65)
        print(f"{'Loss Function':<25} {'Best Acc':>10} {'Final Acc':>10} "
              f"{'Final Loss':>11}")
        print("-" * 65)

        for loss_key in losses:
            if loss_key not in self.results:
                continue
            r = self.results[loss_key]
            print(f"{r['name']:<25} {r['best_acc']:>10.4f} {r['final_acc']:>10.4f} "
                  f"{r['train_losses'][-1]:>11.4f}")
        print()

    def gradient_comparison(self):
        """Compare gradient magnitudes for correct vs wrong predictions."""
        print("=" * 65)
        print("GRADIENT ANALYSIS: CE vs MSE")
        print("=" * 65)

        # Confident wrong prediction
        logits = torch.tensor([[0.1, 0.1, 5.0]])
        target = torch.tensor([0])

        ce_grad = gradient_analysis(F.cross_entropy, logits.clone(), target)
        print(f"  Wrong confident prediction (logits=[0.1, 0.1, 5.0], target=0):")
        print(f"    CE gradient norm:  {ce_grad.norm().item():.4f}")

        # MSE gradient (through softmax)
        logits_mse = logits.clone().detach().requires_grad_(True)
        probs = F.softmax(logits_mse, dim=1)
        target_oh = F.one_hot(target, 3).float()
        mse = F.mse_loss(probs, target_oh)
        mse.backward()
        print(f"    MSE gradient norm: {logits_mse.grad.norm().item():.4f}")
        print(f"    → CE gives stronger signal for wrong predictions")

        # Correct confident prediction
        logits2 = torch.tensor([[5.0, 0.1, 0.1]])
        target2 = torch.tensor([0])

        ce_grad2 = gradient_analysis(F.cross_entropy, logits2.clone(), target2)
        print(f"\n  Correct confident prediction (logits=[5.0, 0.1, 0.1], target=0):")
        print(f"    CE gradient norm:  {ce_grad2.norm().item():.4f}")
        print(f"    → Small gradient when already correct (desirable)")
        print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Loss Function Comparison")
    parser.add_argument("--losses", default="ce,focal,label_smooth,nll")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Config(**vars(args))
    device = get_device(config)
    print(f"Device: {device}\n")

    trainer = LossComparisonTrainer(config, device)
    trainer.load_data()
    trainer.gradient_comparison()
    trainer.run_comparison()


if __name__ == "__main__":
    main()

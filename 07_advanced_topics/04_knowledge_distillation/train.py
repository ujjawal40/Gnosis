"""
Knowledge Distillation Training on MNIST
===========================================

Train a large teacher, then distill into a small student.
Compare: student from scratch vs distilled student.

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
from model import TeacherModel, StudentModel, DistillationLoss, KDConfig


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TrainConfig:
    teacher_epochs: int = 15
    student_epochs: int = 15
    batch_size: int = 256
    lr: float = 0.001
    temperature: float = 4.0
    alpha: float = 0.7
    seed: int = 42
    device: str = "auto"


def get_device(config: TrainConfig) -> torch.device:
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def get_mnist(batch_size):
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test = datasets.MNIST("./data", train=False, download=True, transform=transform)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2))


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_model(model, train_loader, test_loader, epochs, lr, device, label=""):
    """Standard training loop."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"val_acc": []}

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
        history["val_acc"].append(acc)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{label}] Epoch {epoch+1:3d} | Val Acc: {acc:.4f}")

    return history


def distill_model(student, teacher, train_loader, test_loader,
                  epochs, lr, temperature, alpha, device):
    """Distillation training loop."""
    student = student.to(device)
    teacher = teacher.to(device).eval()
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    kd_loss = DistillationLoss(temperature, alpha)
    history = {"val_acc": []}

    for epoch in range(epochs):
        student.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher(X)
            student_logits = student(X)
            loss = kd_loss(student_logits, teacher_logits, y)
            loss.backward()
            optimizer.step()

        student.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                correct += (student(X).argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / total
        history["val_acc"].append(acc)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [Distilled] Epoch {epoch+1:3d} | Val Acc: {acc:.4f}")

    return history


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = TrainConfig()
    device = get_device(config)
    torch.manual_seed(config.seed)

    print("=" * 70)
    print("KNOWLEDGE DISTILLATION ON MNIST")
    print("=" * 70)

    train_loader, test_loader = get_mnist(config.batch_size)
    kd_config = KDConfig()

    # 1. Train teacher
    print("\n--- Training Teacher ---")
    teacher = TeacherModel(kd_config.teacher_dims)
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher params: {n_teacher:,}")
    teacher_hist = train_model(teacher, train_loader, test_loader,
                                config.teacher_epochs, config.lr, device, "Teacher")

    # 2. Train student from scratch
    print("\n--- Training Student (scratch) ---")
    torch.manual_seed(config.seed)
    student_scratch = StudentModel(kd_config.student_dims)
    n_student = sum(p.numel() for p in student_scratch.parameters())
    print(f"Student params: {n_student:,} ({n_student/n_teacher*100:.1f}% of teacher)")
    scratch_hist = train_model(student_scratch, train_loader, test_loader,
                                config.student_epochs, config.lr, device, "Scratch")

    # 3. Distill student
    print("\n--- Training Student (distilled) ---")
    torch.manual_seed(config.seed)
    student_kd = StudentModel(kd_config.student_dims)
    kd_hist = distill_model(student_kd, teacher, train_loader, test_loader,
                            config.student_epochs, config.lr,
                            config.temperature, config.alpha, device)

    # Summary
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Model':>20s} | {'Params':>10s} | {'Best Acc':>9s}")
    print("-" * 45)
    print(f"{'Teacher':>20s} | {n_teacher:>10,} | {max(teacher_hist['val_acc']):>8.4f}")
    print(f"{'Student (scratch)':>20s} | {n_student:>10,} | {max(scratch_hist['val_acc']):>8.4f}")
    print(f"{'Student (distilled)':>20s} | {n_student:>10,} | {max(kd_hist['val_acc']):>8.4f}")
    improvement = max(kd_hist['val_acc']) - max(scratch_hist['val_acc'])
    print(f"\nDistillation improvement: {improvement*100:+.2f}%")


if __name__ == "__main__":
    main()

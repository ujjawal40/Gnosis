"""
Multi-Task Learning Experiments
==================================

Train MTL model on two related tasks from Fashion-MNIST:
    Task 1: Classify item type (10 classes)
    Task 2: Classify coarse category (tops, bottoms, shoes, bags, accessories)

Usage:
    python train.py
"""

import sys
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model import (HardSharingMTL, SingleTaskModel, MTLConfig,
                   UncertaintyWeightedLoss)


@dataclass
class TrainConfig:
    epochs: int = 15
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


# Fashion-MNIST coarse categories
# 0:T-shirt, 1:Trouser, 2:Pullover, 3:Dress, 4:Coat,
# 5:Sandal, 6:Shirt, 7:Sneaker, 8:Bag, 9:Ankle boot
COARSE_MAP = {
    0: 0, 2: 0, 4: 0, 6: 0,  # Tops
    1: 1, 3: 1,                # Bottoms/Dress
    5: 2, 7: 2, 9: 2,          # Footwear
    8: 3,                       # Bags
}


def get_fashion_mnist_mtl(bs):
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    train = datasets.FashionMNIST("./data", train=True, download=True, transform=t)
    test = datasets.FashionMNIST("./data", train=False, download=True, transform=t)

    def make_mtl_dataset(dataset):
        images, fine_labels, coarse_labels = [], [], []
        for img, label in dataset:
            images.append(img)
            fine_labels.append(label)
            coarse_labels.append(COARSE_MAP[label])
        return TensorDataset(
            torch.stack(images),
            torch.tensor(fine_labels),
            torch.tensor(coarse_labels)
        )

    train_ds = make_mtl_dataset(train)
    test_ds = make_mtl_dataset(test)
    return (DataLoader(train_ds, bs, shuffle=True, num_workers=2),
            DataLoader(test_ds, bs, num_workers=2))


def main():
    config = TrainConfig()
    device = get_device(config)
    torch.manual_seed(config.seed)

    print("=" * 70)
    print("MULTI-TASK LEARNING EXPERIMENTS")
    print("=" * 70)
    print("  Task 1: Fine-grained classification (10 classes)")
    print("  Task 2: Coarse category (4 classes)")

    train_loader, test_loader = get_fashion_mnist_mtl(config.batch_size)

    results = {}

    # 1. Single-task baselines
    print("\n--- Single-Task Baselines ---")
    for task_name, task_idx, n_classes in [("Fine (10)", 0, 10), ("Coarse (4)", 1, 4)]:
        torch.manual_seed(config.seed)
        model = SingleTaskModel(784, 256, n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        for epoch in range(config.epochs):
            model.train()
            for batch in train_loader:
                X, y_fine, y_coarse = [b.to(device) for b in batch]
                y = y_fine if task_idx == 0 else y_coarse
                optimizer.zero_grad()
                F.cross_entropy(model(X), y).backward()
                optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                X, y_fine, y_coarse = [b.to(device) for b in batch]
                y = y_fine if task_idx == 0 else y_coarse
                correct += (model(X).argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / total
        results[f"single_{task_name}"] = acc
        print(f"  {task_name}: {acc:.4f}")

    # 2. MTL with uniform weighting
    print("\n--- MTL: Uniform Weighting ---")
    torch.manual_seed(config.seed)
    mtl_config = MTLConfig(task_outputs=[10, 4])
    model = HardSharingMTL(mtl_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        model.train()
        for batch in train_loader:
            X, y_fine, y_coarse = [b.to(device) for b in batch]
            outputs = model(X)
            loss = F.cross_entropy(outputs[0], y_fine) + F.cross_entropy(outputs[1], y_coarse)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    fine_correct, coarse_correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in test_loader:
            X, y_fine, y_coarse = [b.to(device) for b in batch]
            outputs = model(X)
            fine_correct += (outputs[0].argmax(1) == y_fine).sum().item()
            coarse_correct += (outputs[1].argmax(1) == y_coarse).sum().item()
            total += X.size(0)
    fine_acc = fine_correct / total
    coarse_acc = coarse_correct / total
    results["mtl_uniform_fine"] = fine_acc
    results["mtl_uniform_coarse"] = coarse_acc
    print(f"  Fine: {fine_acc:.4f} | Coarse: {coarse_acc:.4f}")

    # 3. MTL with uncertainty weighting
    print("\n--- MTL: Uncertainty Weighting ---")
    torch.manual_seed(config.seed)
    model = HardSharingMTL(mtl_config).to(device)
    uw_loss = UncertaintyWeightedLoss(2).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(uw_loss.parameters()), lr=config.lr
    )

    for epoch in range(config.epochs):
        model.train()
        for batch in train_loader:
            X, y_fine, y_coarse = [b.to(device) for b in batch]
            outputs = model(X)
            losses = [
                F.cross_entropy(outputs[0], y_fine),
                F.cross_entropy(outputs[1], y_coarse),
            ]
            loss = uw_loss(losses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    fine_correct, coarse_correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in test_loader:
            X, y_fine, y_coarse = [b.to(device) for b in batch]
            outputs = model(X)
            fine_correct += (outputs[0].argmax(1) == y_fine).sum().item()
            coarse_correct += (outputs[1].argmax(1) == y_coarse).sum().item()
            total += X.size(0)
    fine_acc = fine_correct / total
    coarse_acc = coarse_correct / total
    weights = uw_loss.get_weights()
    results["mtl_uw_fine"] = fine_acc
    results["mtl_uw_coarse"] = coarse_acc
    print(f"  Fine: {fine_acc:.4f} | Coarse: {coarse_acc:.4f}")
    print(f"  Learned weights: [{weights[0]:.3f}, {weights[1]:.3f}]")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Method':>25s} | {'Fine (10)':>9s} | {'Coarse (4)':>10s}")
    print("-" * 50)
    print(f"{'Single-task':>25s} | {results['single_Fine (10)']:>8.4f}  | "
          f"{results['single_Coarse (4)']:>9.4f}")
    print(f"{'MTL Uniform':>25s} | {results['mtl_uniform_fine']:>8.4f}  | "
          f"{results['mtl_uniform_coarse']:>9.4f}")
    print(f"{'MTL Uncertainty':>25s} | {results['mtl_uw_fine']:>8.4f}  | "
          f"{results['mtl_uw_coarse']:>9.4f}")


if __name__ == "__main__":
    main()

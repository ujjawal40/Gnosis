"""
Quantization Experiments
===========================

Compare standard vs QAT models on MNIST.

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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model import (SimpleMLP, QATMLP, QuantConfig,
                   quantize_model_weights, model_size_bytes)


@dataclass
class TrainConfig:
    epochs: int = 10
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


def get_mnist(bs):
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST("./data", train=True, download=True, transform=t)
    test = datasets.MNIST("./data", train=False, download=True, transform=t)
    return DataLoader(train, bs, shuffle=True, num_workers=2), DataLoader(test, bs, num_workers=2)


def train_model(model, train_loader, epochs, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            F.cross_entropy(model(X), y).backward()
            optimizer.step()


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            correct += (model(X).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    config = TrainConfig()
    device = get_device(config)
    torch.manual_seed(config.seed)

    print("=" * 70)
    print("MODEL QUANTIZATION EXPERIMENTS")
    print("=" * 70)

    train_loader, test_loader = get_mnist(config.batch_size)

    # 1. Train standard FP32 model
    print("\n--- Standard FP32 Model ---")
    q_config = QuantConfig()
    model_fp32 = SimpleMLP(q_config).to(device)
    train_model(model_fp32, train_loader, config.epochs, config.lr, device)
    acc_fp32 = evaluate(model_fp32, test_loader, device)
    size_fp32 = model_size_bytes(model_fp32, 32)
    print(f"  Accuracy: {acc_fp32:.4f}")
    print(f"  Size: {size_fp32 / 1024:.1f} KB")

    # 2. Post-training quantization (simulated)
    print("\n--- Post-Training Quantization ---")
    results = {}
    for bits in [8, 4, 2]:
        quant_info = quantize_model_weights(model_fp32, bits)
        # Simulate quantized inference by dequantizing
        model_ptq = SimpleMLP(q_config).to(device)
        model_ptq.load_state_dict(model_fp32.state_dict())

        with torch.no_grad():
            for name, info in quant_info.items():
                param_path = name.split('.')
                module = model_ptq
                for p in param_path[:-1]:
                    module = getattr(module, p)
                # Dequantize and set
                dequant_w = info['q_weight'].float() * info['scale']
                getattr(module, param_path[-1]).copy_(dequant_w)

        acc = evaluate(model_ptq, test_loader, device)
        size = model_size_bytes(model_ptq, bits)
        results[f"PTQ-{bits}bit"] = {"acc": acc, "size": size}
        print(f"  {bits}-bit | Accuracy: {acc:.4f} | "
              f"Size: {size / 1024:.1f} KB | "
              f"Acc drop: {acc_fp32 - acc:.4f}")

    # 3. Quantization-Aware Training
    print("\n--- Quantization-Aware Training ---")
    for bits in [8, 4]:
        torch.manual_seed(config.seed)
        qat_config = QuantConfig(n_bits=bits)
        model_qat = QATMLP(qat_config).to(device)
        train_model(model_qat, train_loader, config.epochs, config.lr, device)
        acc_qat = evaluate(model_qat, test_loader, device)
        size_qat = model_size_bytes(model_qat, bits)
        results[f"QAT-{bits}bit"] = {"acc": acc_qat, "size": size_qat}
        print(f"  {bits}-bit QAT | Accuracy: {acc_qat:.4f} | "
              f"Size: {size_qat / 1024:.1f} KB")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Method':>15s} | {'Accuracy':>8s} | {'Size (KB)':>10s} | {'Compression':>12s}")
    print("-" * 52)
    print(f"{'FP32':>15s} | {acc_fp32:>7.4f}  | {size_fp32/1024:>9.1f}  | {'1.0x':>12s}")
    for name, r in results.items():
        comp = size_fp32 / max(r['size'], 1)
        print(f"{name:>15s} | {r['acc']:>7.4f}  | {r['size']/1024:>9.1f}  | {comp:>10.1f}x")


if __name__ == "__main__":
    main()

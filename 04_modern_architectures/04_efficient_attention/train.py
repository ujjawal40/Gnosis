"""
Efficient Attention Experiments
==================================

Compare MHA, MQA, GQA, and Sliding Window attention on a
sequence modeling task using MNIST rows as sequences.

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
from model import GroupedQueryAttention, SlidingWindowAttention, EfficientAttnConfig


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


def get_mnist(bs):
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST("./data", train=True, download=True, transform=t)
    test = datasets.MNIST("./data", train=False, download=True, transform=t)
    return DataLoader(train, bs, shuffle=True, num_workers=2), DataLoader(test, bs, num_workers=2)


# =============================================================================
# CLASSIFIER WITH CONFIGURABLE ATTENTION
# =============================================================================

class AttnClassifier(nn.Module):
    """Classifier using configurable attention mechanism."""

    def __init__(self, attn_config: EfficientAttnConfig,
                 feat_dim: int = 28, n_classes: int = 10):
        super().__init__()
        self.proj = nn.Linear(feat_dim, attn_config.d_model)
        self.norm1 = nn.LayerNorm(attn_config.d_model)
        self.attn = GroupedQueryAttention(attn_config)
        self.norm2 = nn.LayerNorm(attn_config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(attn_config.d_model, attn_config.d_model * 4),
            nn.GELU(),
            nn.Linear(attn_config.d_model * 4, attn_config.d_model),
        )
        self.head = nn.Linear(attn_config.d_model, n_classes)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)  # (B, 28, 28)
        x = self.proj(x)
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return self.head(x.mean(dim=1))


class SlidingWindowClassifier(nn.Module):
    """Classifier using sliding window attention."""

    def __init__(self, attn_config: EfficientAttnConfig,
                 feat_dim: int = 28, n_classes: int = 10):
        super().__init__()
        self.proj = nn.Linear(feat_dim, attn_config.d_model)
        self.norm1 = nn.LayerNorm(attn_config.d_model)
        self.attn = SlidingWindowAttention(attn_config)
        self.norm2 = nn.LayerNorm(attn_config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(attn_config.d_model, attn_config.d_model * 4),
            nn.GELU(),
            nn.Linear(attn_config.d_model * 4, attn_config.d_model),
        )
        self.head = nn.Linear(attn_config.d_model, n_classes)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        x = self.proj(x)
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return self.head(x.mean(dim=1))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    config = TrainConfig()
    device = get_device(config)
    d_model = 64

    print("=" * 70)
    print("EFFICIENT ATTENTION EXPERIMENTS")
    print("=" * 70)
    print(f"Task: MNIST as sequence (28 rows × 28 features)")

    train_loader, test_loader = get_mnist(config.batch_size)

    # Define attention configs
    attention_variants = {
        "MHA (8 heads)": EfficientAttnConfig(
            d_model=d_model, n_heads=8, n_kv_heads=8, dropout=0.1),
        "MQA (1 KV head)": EfficientAttnConfig(
            d_model=d_model, n_heads=8, n_kv_heads=1, dropout=0.1),
        "GQA (2 KV heads)": EfficientAttnConfig(
            d_model=d_model, n_heads=8, n_kv_heads=2, dropout=0.1),
        "GQA (4 KV heads)": EfficientAttnConfig(
            d_model=d_model, n_heads=8, n_kv_heads=4, dropout=0.1),
        "Sliding (w=10)": EfficientAttnConfig(
            d_model=d_model, n_heads=8, n_kv_heads=8,
            window_size=10, dropout=0.1),
    }

    results = {}
    for name, attn_cfg in attention_variants.items():
        torch.manual_seed(config.seed)

        if "Sliding" in name:
            model = SlidingWindowClassifier(attn_cfg).to(device)
        else:
            model = AttnClassifier(attn_cfg).to(device)

        n_params = count_params(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        t0 = time.time()
        for epoch in range(config.epochs):
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
        elapsed = time.time() - t0
        results[name] = {"acc": acc, "params": n_params, "time": elapsed}
        print(f"  {name:>20s} | params: {n_params:>7,} | "
              f"acc: {acc:.4f} | time: {elapsed:.1f}s")

    # KV efficiency analysis
    print(f"\n{'=' * 70}")
    print("KV-CACHE MEMORY ANALYSIS")
    print(f"{'=' * 70}")

    seq_len = 28
    batch = 1
    d_head = d_model // 8

    print(f"\n{'Variant':>20s} | {'KV Heads':>8s} | {'KV Memory':>10s} | {'Savings':>8s}")
    print("-" * 55)
    for name, attn_cfg in attention_variants.items():
        if "Sliding" in name:
            continue
        kv_mem = batch * attn_cfg.n_kv_heads * seq_len * d_head * 2 * 4  # float32
        mha_mem = batch * 8 * seq_len * d_head * 2 * 4
        savings = (1 - kv_mem / mha_mem) * 100
        print(f"{name:>20s} | {attn_cfg.n_kv_heads:>8d} | {kv_mem:>8d} B | {savings:>6.1f}%")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    best = max(results.items(), key=lambda x: x[1]["acc"])
    most_efficient = min(
        ((n, r) for n, r in results.items() if "MQA" in n or "GQA" in n),
        key=lambda x: x[1]["params"]
    )
    print(f"  Best accuracy: {best[0]} ({best[1]['acc']:.4f})")
    print(f"  Most efficient: {most_efficient[0]} ({most_efficient[1]['acc']:.4f}, "
          f"{most_efficient[1]['params']:,} params)")


if __name__ == "__main__":
    main()

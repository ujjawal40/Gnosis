"""
Positional Encoding Experiments
==================================

Compare positional encoding methods on a sequence classification task.
Uses MNIST rows as a sequence (28 timesteps of 28 features).

Usage:
    python train.py
"""

import sys
import os
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model import (SinusoidalPE, LearnedPE, RotaryEmbedding,
                   ALiBi, apply_rotary_emb, PosEncConfig)


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 256
    lr: float = 0.001
    d_model: int = 64
    n_heads: int = 4
    seq_len: int = 28
    feat_dim: int = 28
    n_classes: int = 10
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
# MODELS WITH DIFFERENT POSITIONAL ENCODINGS
# =============================================================================

class SelfAttentionBlock(nn.Module):
    """Single self-attention block."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_bias=None):
        if attn_bias is not None:
            out, _ = self.attn(x, x, x, attn_mask=attn_bias.squeeze(0) if attn_bias.dim() == 4 else attn_bias)
        else:
            out, _ = self.attn(x, x, x)
        return self.norm(x + out)


class SeqClassifier(nn.Module):
    """Sequence classifier with configurable positional encoding."""

    def __init__(self, config: TrainConfig, pe_type: str = "sinusoidal"):
        super().__init__()
        self.pe_type = pe_type
        self.proj = nn.Linear(config.feat_dim, config.d_model)

        if pe_type == "sinusoidal":
            self.pe = SinusoidalPE(config.d_model, config.seq_len + 10)
        elif pe_type == "learned":
            self.pe = LearnedPE(config.d_model, config.seq_len + 10)
        elif pe_type == "rope":
            self.rope = RotaryEmbedding(config.d_model // config.n_heads)
        elif pe_type == "alibi":
            self.alibi = ALiBi(config.n_heads)
        # "none" = no positional encoding

        self.attn = SelfAttentionBlock(config.d_model, config.n_heads)
        self.head = nn.Linear(config.d_model, config.n_classes)

    def forward(self, x):
        # x: (B, 1, 28, 28) -> (B, 28, 28) treat rows as sequence
        if x.dim() == 4:
            x = x.squeeze(1)
        x = self.proj(x)  # (B, 28, d_model)

        if self.pe_type in ("sinusoidal", "learned"):
            x = self.pe(x)
            out = self.attn(x)
        elif self.pe_type == "alibi":
            bias = self.alibi(x.size(1))
            # Average across heads for nn.MultiheadAttention compatibility
            bias_avg = bias.mean(0)
            out = self.attn(x, attn_bias=bias_avg)
        else:
            out = self.attn(x)

        # Mean pool over sequence
        return self.head(out.mean(dim=1))


def main():
    config = TrainConfig()
    device = get_device(config)

    print("=" * 70)
    print("POSITIONAL ENCODING EXPERIMENTS")
    print("=" * 70)
    print(f"Task: MNIST as sequence (28 rows × 28 features)")
    print(f"Model: d_model={config.d_model}, heads={config.n_heads}")

    train_loader, test_loader = get_mnist(config.batch_size)

    pe_types = ["none", "sinusoidal", "learned", "alibi"]

    results = {}
    for pe_type in pe_types:
        torch.manual_seed(config.seed)
        model = SeqClassifier(config, pe_type).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        n_params = sum(p.numel() for p in model.parameters())

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
        results[pe_type] = {"acc": acc, "params": n_params, "time": elapsed}
        print(f"  {pe_type:>12s} | params: {n_params:>7,} | "
              f"acc: {acc:.4f} | time: {elapsed:.1f}s")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Type':>12s} | {'Accuracy':>8s} | {'Params':>8s}")
    print("-" * 36)
    for pe_type in pe_types:
        r = results[pe_type]
        print(f"{pe_type:>12s} | {r['acc']:>7.4f}  | {r['params']:>7,}")

    best = max(results.items(), key=lambda x: x[1]["acc"])
    none_acc = results["none"]["acc"]
    print(f"\n  Best: {best[0]} ({best[1]['acc']:.4f})")
    print(f"  Gain over no-PE: +{best[1]['acc'] - none_acc:.4f}")


if __name__ == "__main__":
    main()

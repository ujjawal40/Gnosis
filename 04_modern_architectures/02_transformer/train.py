"""
Transformer Language Model Training on WikiText-2
====================================================

GPT-style decoder-only transformer for next-token prediction.
Evaluates using perplexity and generates sample text.

Usage:
    python train.py
    python train.py --n_layers 4 --embed_dim 256 --epochs 30
"""

import sys
import os
import argparse
import time
import math
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model import TransformerLM, TransformerConfig


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TrainConfig:
    embed_dim: int = 256
    n_heads: int = 4
    n_layers: int = 4
    ff_dim: int = 512
    max_seq_len: int = 128
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    epochs: int = 30
    batch_size: int = 32
    max_vocab: int = 30000
    clip: float = 1.0
    seed: int = 42
    device: str = "auto"


def get_device(config: TrainConfig) -> torch.device:
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# DATA
# =============================================================================

class LMDataset:
    """Chunk text into fixed-length sequences for transformer LM."""

    def __init__(self, token_ids: List[int], seq_len: int, device: torch.device):
        self.seq_len = seq_len
        # Cut into complete sequences
        n_seqs = len(token_ids) // (seq_len + 1)
        token_ids = token_ids[:n_seqs * (seq_len + 1)]
        self.data = torch.tensor(token_ids, dtype=torch.long, device=device)
        self.data = self.data.view(n_seqs, seq_len + 1)

    def __len__(self):
        return self.data.size(0)

    def get_batch(self, indices):
        """Get input/target pairs: input = data[:-1], target = data[1:]."""
        batch = self.data[indices]
        x = batch[:, :-1]
        y = batch[:, 1:]
        return x, y


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """Transformer LM trainer with warmup + cosine decay."""

    def __init__(self, model: TransformerLM, config: TrainConfig,
                 device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.step = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_ppl": [],
            "val_loss": [], "val_ppl": [],
        }

    def get_lr(self) -> float:
        """Warmup + cosine decay schedule."""
        if self.step < self.config.warmup_steps:
            return self.config.lr * self.step / self.config.warmup_steps
        progress = (self.step - self.config.warmup_steps) / \
                   max(1, self.total_steps - self.config.warmup_steps)
        return self.config.lr * 0.5 * (1 + math.cos(math.pi * progress))

    def train_one_epoch(self, dataset: LMDataset, optimizer, loss_fn) -> float:
        self.model.train()
        indices = torch.randperm(len(dataset))
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(indices), self.config.batch_size):
            batch_idx = indices[i:i + self.config.batch_size]
            if len(batch_idx) == 0:
                break

            x, y = dataset.get_batch(batch_idx)

            # Update LR
            lr = self.get_lr()
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            self.step += 1

            optimizer.zero_grad()
            logits = self.model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    @torch.no_grad()
    def evaluate(self, dataset: LMDataset, loss_fn) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(dataset), self.config.batch_size):
            batch_idx = torch.arange(i, min(i + self.config.batch_size, len(dataset)))
            x, y = dataset.get_batch(batch_idx)

            logits = self.model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def fit(self, train_data: LMDataset, val_data: LMDataset, vocab):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )

        n_batches_per_epoch = len(train_data) // self.config.batch_size
        self.total_steps = self.config.epochs * n_batches_per_epoch

        print(f"  Parameters: {self.model.count_parameters():,}")
        print(f"  Train sequences: {len(train_data):,}")
        print(f"  Val sequences: {len(val_data):,}")
        print(f"  Total steps: {self.total_steps:,}")
        print()

        best_val_loss = float("inf")

        for epoch in range(self.config.epochs):
            start = time.time()
            train_loss = self.train_one_epoch(train_data, optimizer, loss_fn)
            val_loss = self.evaluate(val_data, loss_fn)
            elapsed = time.time() - start

            train_ppl = math.exp(min(train_loss, 20))
            val_ppl = math.exp(min(val_loss, 20))

            self.history["train_loss"].append(train_loss)
            self.history["train_ppl"].append(train_ppl)
            self.history["val_loss"].append(val_loss)
            self.history["val_ppl"].append(val_ppl)

            marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                marker = " ★"

            lr = self.get_lr()
            print(f"  Epoch {epoch+1:3d}/{self.config.epochs} ({elapsed:.1f}s): "
                  f"train_ppl={train_ppl:.1f} val_ppl={val_ppl:.1f} "
                  f"lr={lr:.6f}{marker}")

        best_ppl = math.exp(min(best_val_loss, 20))
        print(f"\n  Best validation perplexity: {best_ppl:.1f}")

        # Generate sample
        self.generate_sample(vocab)

    def generate_sample(self, vocab, prompt="the", n_tokens=50):
        """Generate text from the trained model."""
        tokens = [vocab.word2idx.get(w, vocab.word2idx["<unk>"])
                  for w in prompt.split()]
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        generated = self.model.generate(
            input_ids, max_len=n_tokens, temperature=0.8, top_k=40)

        text = vocab.decode(generated[0].tolist())
        print(f"\n  Generated (prompt='{prompt}'):")
        print(f"    {text}")
        print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Transformer LM on WikiText-2")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=512)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(**vars(args))
    torch.manual_seed(config.seed)
    device = get_device(config)

    print("=" * 60)
    print("TRANSFORMER LANGUAGE MODEL ON WIKITEXT-2")
    print(f"Device: {device}")
    print("=" * 60)

    from datasets.text_datasets import load_wikitext2

    train_tokens, val_tokens, _, vocab = load_wikitext2(
        max_vocab_size=config.max_vocab)

    unk_id = vocab.word2idx["<unk>"]
    train_ids = [vocab.word2idx.get(t, unk_id) for t in train_tokens]
    val_ids = [vocab.word2idx.get(t, unk_id) for t in val_tokens]

    train_data = LMDataset(train_ids, config.max_seq_len, device)
    val_data = LMDataset(val_ids, config.max_seq_len, device)

    model_config = TransformerConfig(
        vocab_size=len(vocab), embed_dim=config.embed_dim,
        n_heads=config.n_heads, n_layers=config.n_layers,
        ff_dim=config.ff_dim, max_seq_len=config.max_seq_len,
        dropout=config.dropout, tie_weights=True,
    )
    model = TransformerLM(model_config)

    trainer = Trainer(model, config, device)
    trainer.fit(train_data, val_data, vocab)


if __name__ == "__main__":
    main()

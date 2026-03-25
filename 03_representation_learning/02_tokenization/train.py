"""
Tokenization Comparison Training
===================================

Compare character, word, and BPE tokenization on a text classification task.

Usage:
    python train.py
"""

import sys
import os
import argparse
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from implementation import CharTokenizer, WordTokenizer, BPETokenizer
from model import TextClassifier, TokenizedBatch, collate_sequences


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 32
    lr: float = 0.001
    embed_dim: int = 64
    hidden_dim: int = 128
    max_length: int = 64
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
# SYNTHETIC DATASET
# =============================================================================

def make_sentiment_data(n_samples: int = 2000):
    """Create synthetic sentiment data for tokenization comparison."""
    np.random.seed(42)

    positive_words = ["good", "great", "excellent", "amazing", "wonderful",
                      "fantastic", "brilliant", "love", "best", "perfect",
                      "beautiful", "outstanding", "superb", "awesome", "happy"]
    negative_words = ["bad", "terrible", "awful", "horrible", "worst",
                      "hate", "disappointing", "poor", "boring", "ugly",
                      "mediocre", "annoying", "dreadful", "pathetic", "sad"]
    neutral_words = ["the", "movie", "was", "this", "film", "is", "a",
                     "very", "quite", "rather", "somewhat", "really",
                     "truly", "incredibly", "absolutely"]

    texts, labels = [], []
    for _ in range(n_samples):
        length = np.random.randint(5, 15)
        label = np.random.randint(2)
        words = []
        for _ in range(length):
            r = np.random.rand()
            if r < 0.4:
                pool = positive_words if label == 1 else negative_words
            elif r < 0.6:
                pool = negative_words if label == 1 else positive_words
            else:
                pool = neutral_words
            words.append(np.random.choice(pool))
        texts.append(" ".join(words))
        labels.append(label)

    # Split
    split = int(0.8 * n_samples)
    return texts[:split], labels[:split], texts[split:], labels[split:]


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = TrainConfig()
    device = get_device(config)
    torch.manual_seed(config.seed)

    print("=" * 70)
    print("TOKENIZATION COMPARISON ON SENTIMENT CLASSIFICATION")
    print("=" * 70)

    train_texts, train_labels, test_texts, test_labels = make_sentiment_data()
    all_texts = train_texts + test_texts

    tokenizers = {
        "Character": CharTokenizer(),
        "Word": WordTokenizer(max_vocab=500),
        "BPE (100)": BPETokenizer(num_merges=100),
        "BPE (500)": BPETokenizer(num_merges=500),
    }

    results = {}
    for tok_name, tokenizer in tokenizers.items():
        print(f"\n{'─' * 50}")
        print(f"Tokenizer: {tok_name}")

        tokenizer.fit(all_texts)
        print(f"Vocab size: {tokenizer.vocab_size}")

        # Encode all data
        train_ids = [tokenizer.encode(t) for t in train_texts]
        test_ids = [tokenizer.encode(t) for t in test_texts]

        avg_len = np.mean([len(ids) for ids in train_ids])
        print(f"Avg sequence length: {avg_len:.1f}")

        # Build model
        model = TextClassifier(
            vocab_size=tokenizer.vocab_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            n_classes=2,
            max_length=config.max_length,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        history = {"train_loss": [], "val_acc": []}

        # Training loop (manual batching with collate)
        for epoch in range(config.epochs):
            model.train()
            indices = np.random.permutation(len(train_ids))
            total_loss = 0
            n_batches = 0

            for i in range(0, len(indices), config.batch_size):
                batch_idx = indices[i:i+config.batch_size]
                batch_ids = [train_ids[j] for j in batch_idx]
                batch_labels = torch.tensor([train_labels[j] for j in batch_idx],
                                             dtype=torch.long, device=device)

                batch = collate_sequences(batch_ids, config.max_length).to(device)
                optimizer.zero_grad()
                logits = model(batch)
                loss = F.cross_entropy(logits, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            # Evaluate
            model.eval()
            with torch.no_grad():
                test_batch = collate_sequences(test_ids, config.max_length).to(device)
                test_labels_t = torch.tensor(test_labels, dtype=torch.long, device=device)
                preds = model(test_batch).argmax(1)
                acc = (preds == test_labels_t).float().mean().item()

            history["train_loss"].append(total_loss / n_batches)
            history["val_acc"].append(acc)

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d} | Loss: {total_loss/n_batches:.4f} | "
                      f"Val Acc: {acc:.3f}")

        results[tok_name] = {"acc": max(history["val_acc"]),
                              "vocab": tokenizer.vocab_size,
                              "avg_len": avg_len}

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Tokenizer':>15s} | {'Vocab':>6s} | {'Avg Len':>7s} | {'Best Acc':>8s}")
    print("-" * 45)
    for name, r in results.items():
        print(f"{name:>15s} | {r['vocab']:>6d} | {r['avg_len']:>6.1f} | {r['acc']:>7.3f}")


if __name__ == "__main__":
    main()

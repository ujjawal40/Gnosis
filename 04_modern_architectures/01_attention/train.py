"""
Attention Classifier Training on IMDB
========================================

Binary sentiment classification using self-attention mechanism.
Demonstrates how attention learns to focus on sentiment-bearing words.

Usage:
    python train.py
    python train.py --epochs 10 --embed_dim 128 --n_heads 4
"""

import sys
import os
import argparse
import time
import collections
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model import AttentionClassifier, AttentionConfig


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TrainConfig:
    embed_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 256
    max_seq_len: int = 256
    max_vocab: int = 25000
    lr: float = 0.0003
    epochs: int = 10
    batch_size: int = 32
    dropout: float = 0.1
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
# DATASET
# =============================================================================

class IMDBDataset(Dataset):
    """IMDB reviews as token index sequences."""

    def __init__(self, texts: List[str], labels: List[int],
                 word2idx: Dict[str, int], max_len: int = 256):
        self.labels = labels
        self.max_len = max_len
        self.word2idx = word2idx
        self.unk_idx = word2idx.get("<unk>", 1)

        # Tokenize and encode
        self.encoded = []
        for text in texts:
            tokens = text.lower().split()[:max_len]
            ids = [word2idx.get(t, self.unk_idx) for t in tokens]
            self.encoded.append(ids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded[idx], dtype=torch.long), self.labels[idx]


def collate_fn(batch):
    """Pad sequences to same length within batch."""
    sequences, labels = zip(*batch)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    mask = (padded != 0).long()
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, mask, labels


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """Attention classifier trainer."""

    def __init__(self, model: nn.Module, config: TrainConfig, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
        }

    def train_one_epoch(self, loader, optimizer, loss_fn) -> tuple:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for padded, mask, labels in loader:
            padded = padded.to(self.device)
            mask = mask.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            logits = self.model(padded, mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, loader, loss_fn) -> tuple:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for padded, mask, labels in loader:
            padded = padded.to(self.device)
            mask = mask.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(padded, mask)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    def fit(self, train_loader, val_loader):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs)

        print(f"  Parameters: {self.model.count_parameters():,}")
        print()

        best_acc = 0.0
        for epoch in range(self.config.epochs):
            start = time.time()
            train_loss, train_acc = self.train_one_epoch(
                train_loader, optimizer, loss_fn)
            val_loss, val_acc = self.evaluate(val_loader, loss_fn)
            scheduler.step()
            elapsed = time.time() - start

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            marker = " ★" if val_acc > best_acc else ""
            best_acc = max(best_acc, val_acc)

            print(f"  Epoch {epoch+1:2d}/{self.config.epochs} ({elapsed:.1f}s): "
                  f"train={train_loss:.4f}/{train_acc:.4f} "
                  f"val={val_loss:.4f}/{val_acc:.4f}{marker}")

        print(f"\n  Best validation accuracy: {best_acc:.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Attention on IMDB")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(**vars(args))
    torch.manual_seed(config.seed)
    device = get_device(config)

    print("=" * 60)
    print("ATTENTION CLASSIFIER ON IMDB")
    print(f"Device: {device}")
    print("=" * 60)

    # Load data
    from datasets.text_datasets import load_imdb

    train_texts, train_labels, test_texts, test_labels = load_imdb()

    # Build vocabulary from training data
    word_counts = collections.Counter()
    for text in train_texts:
        word_counts.update(text.lower().split())

    vocab_words = ["<pad>", "<unk>"] + [w for w, c in word_counts.most_common(config.max_vocab - 2)]
    word2idx = {w: i for i, w in enumerate(vocab_words)}

    # Create datasets
    train_dataset = IMDBDataset(train_texts, train_labels, word2idx, config.max_seq_len)
    test_dataset = IMDBDataset(test_texts, test_labels, word2idx, config.max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=collate_fn,
                             num_workers=2, pin_memory=True)

    print(f"  Vocab size: {len(vocab_words)}")
    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Build model
    model_config = AttentionConfig(
        vocab_size=len(vocab_words), embed_dim=config.embed_dim,
        n_heads=config.n_heads, n_layers=config.n_layers,
        ff_dim=config.ff_dim, max_seq_len=config.max_seq_len,
        n_classes=2, dropout=config.dropout,
    )
    model = AttentionClassifier(model_config)

    # Train
    trainer = Trainer(model, config, device)
    trainer.fit(train_loader, test_loader)


if __name__ == "__main__":
    main()

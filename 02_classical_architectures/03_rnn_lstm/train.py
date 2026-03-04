"""
LSTM Language Model Training on WikiText-2
============================================

Word-level language modeling: predict the next word given context.
Evaluates using perplexity (lower is better).

Features:
    - BPTT (Backpropagation Through Time) with configurable sequence length
    - Hidden state carried across batches (detached to prevent gradient explosion)
    - Gradient clipping (essential for RNNs)
    - Perplexity evaluation
    - Text generation with temperature and top-k sampling

Usage:
    python train.py
    python train.py --epochs 30 --bptt 35 --hidden_dim 512
"""

import sys
import os
import argparse
import time
import math
from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model import WordLSTM, LSTMConfig


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TrainConfig:
    embed_dim: int = 256
    hidden_dim: int = 256
    n_layers: int = 2
    dropout: float = 0.3
    tie_weights: bool = True
    lr: float = 20.0  # SGD with high LR is standard for LSTM LMs
    clip: float = 0.25
    epochs: int = 30
    batch_size: int = 20
    bptt: int = 35  # sequence length for BPTT
    seed: int = 42
    device: str = "auto"
    max_vocab: int = 30000


def get_device(config: TrainConfig) -> torch.device:
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# DATA PREPARATION
# =============================================================================

class LMDataset:
    """Prepares language modeling data in BPTT-friendly batches."""

    def __init__(self, tokens: List[int], batch_size: int, device: torch.device):
        # Reshape into (batch_size, -1)
        n_tokens = len(tokens)
        n_batches = n_tokens // batch_size
        tokens = tokens[:n_batches * batch_size]
        self.data = torch.tensor(tokens, dtype=torch.long, device=device)
        self.data = self.data.view(batch_size, -1)

    def get_batch(self, i: int, bptt: int):
        """Get a batch starting at position i."""
        seq_len = min(bptt, self.data.size(1) - 1 - i)
        x = self.data[:, i:i + seq_len]
        y = self.data[:, i + 1:i + 1 + seq_len]
        return x, y

    @property
    def n_tokens(self):
        return self.data.size(1)


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """LSTM language model trainer."""

    def __init__(self, model: WordLSTM, config: TrainConfig, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_ppl": [],
            "val_loss": [], "val_ppl": [],
        }

    def train_one_epoch(self, dataset: LMDataset, optimizer, loss_fn) -> float:
        """Train one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_tokens = 0

        hidden = self.model.init_hidden(self.config.batch_size, self.device)

        for i in range(0, dataset.n_tokens - 1, self.config.bptt):
            x, y = dataset.get_batch(i, self.config.bptt)
            if x.size(1) == 0:
                break

            # Detach hidden state (truncated BPTT)
            hidden = tuple(h.detach() for h in hidden)

            optimizer.zero_grad()
            logits, hidden = self.model(x, hidden)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()

            # Gradient clipping (critical for RNNs)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.clip)
            optimizer.step()

            total_loss += loss.item() * y.numel()
            n_tokens += y.numel()

        return total_loss / n_tokens

    @torch.no_grad()
    def evaluate(self, dataset: LMDataset, loss_fn) -> float:
        """Evaluate. Returns average loss."""
        self.model.eval()
        total_loss = 0.0
        n_tokens = 0

        hidden = self.model.init_hidden(self.config.batch_size, self.device)

        for i in range(0, dataset.n_tokens - 1, self.config.bptt):
            x, y = dataset.get_batch(i, self.config.bptt)
            if x.size(1) == 0:
                break

            logits, hidden = self.model(x, hidden)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item() * y.numel()
            n_tokens += y.numel()

        return total_loss / n_tokens

    def fit(self, train_data: LMDataset, val_data: LMDataset, vocab):
        """Full training loop."""
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.lr)

        best_val_loss = float("inf")

        print(f"  Parameters: {self.model.count_parameters():,}")
        print(f"  Vocab size: {len(vocab)}")
        print()

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

            print(f"  Epoch {epoch+1:3d}/{self.config.epochs} ({elapsed:.1f}s): "
                  f"train_ppl={train_ppl:.1f} val_ppl={val_ppl:.1f}{marker}")

            # Anneal LR on plateau
            if epoch > 5 and val_loss >= best_val_loss:
                for pg in optimizer.param_groups:
                    pg["lr"] /= 2.0
                    print(f"    LR reduced to {pg['lr']:.4f}")

        best_ppl = math.exp(min(best_val_loss, 20))
        print(f"\n  Best validation perplexity: {best_ppl:.1f}")

        # Generate sample text
        self.generate_sample(vocab)

    def generate_sample(self, vocab, prompt="the", n_tokens=50):
        """Generate text from the trained model."""
        print(f"\n  Generated text (prompt='{prompt}'):")

        tokens = vocab.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        generated = self.model.generate(
            input_ids, max_len=n_tokens, temperature=0.8, top_k=40)

        text = vocab.decode(generated[0].tolist())
        print(f"    {text}")
        print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LSTM LM on WikiText-2")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=20.0)
    parser.add_argument("--clip", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--bptt", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(**vars(args))
    torch.manual_seed(config.seed)
    device = get_device(config)

    print("=" * 60)
    print("LSTM LANGUAGE MODEL ON WIKITEXT-2")
    print(f"Device: {device}")
    print("=" * 60)

    # Load data
    from datasets.text_datasets import load_wikitext2
    from datasets.utils import Vocabulary

    train_tokens, val_tokens, test_tokens, vocab = load_wikitext2(
        max_vocab_size=config.max_vocab)

    # Encode tokens to indices
    train_ids = [vocab.word2idx.get(t, vocab.word2idx["<unk>"]) for t in train_tokens]
    val_ids = [vocab.word2idx.get(t, vocab.word2idx["<unk>"]) for t in val_tokens]

    train_data = LMDataset(train_ids, config.batch_size, device)
    val_data = LMDataset(val_ids, config.batch_size, device)

    # Build model
    model_config = LSTMConfig(
        vocab_size=len(vocab),
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        dropout=config.dropout,
        tie_weights=config.tie_weights,
    )
    model = WordLSTM(model_config)

    # Train
    trainer = Trainer(model, config, device)
    trainer.fit(train_data, val_data, vocab)


if __name__ == "__main__":
    main()

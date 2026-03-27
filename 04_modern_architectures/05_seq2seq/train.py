"""
Seq2Seq Training: Number Reversal Task
==========================================

Train an encoder-decoder model to reverse sequences of numbers.
Compare greedy vs beam search decoding.

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
from model import Seq2Seq, Seq2SeqConfig


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 0.001
    n_train: int = 10000
    n_test: int = 1000
    max_seq_len: int = 8
    seed: int = 42
    device: str = "auto"


def get_device(config):
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def make_reversal_dataset(n_samples, max_len, vocab_size=20,
                           sos=0, eos=1, pad=2):
    """Create number reversal dataset: [3,7,5] -> [5,7,3]"""
    src_seqs, tgt_seqs = [], []

    for _ in range(n_samples):
        length = torch.randint(3, max_len + 1, (1,)).item()
        # Random tokens from 3 to vocab_size-1 (avoid special tokens)
        seq = torch.randint(3, vocab_size, (length,))

        # Source: seq + EOS + padding
        src = torch.full((max_len + 1,), pad, dtype=torch.long)
        src[:length] = seq
        src[length] = eos

        # Target: SOS + reversed seq + EOS + padding
        tgt = torch.full((max_len + 2,), pad, dtype=torch.long)
        tgt[0] = sos
        tgt[1:length + 1] = seq.flip(0)
        tgt[length + 1] = eos

        src_seqs.append(src)
        tgt_seqs.append(tgt)

    return torch.stack(src_seqs), torch.stack(tgt_seqs)


def sequence_accuracy(pred: torch.Tensor, tgt: torch.Tensor,
                       pad_token: int = 2) -> float:
    """Compute full-sequence accuracy (ignoring padding)."""
    mask = tgt != pad_token
    correct = ((pred == tgt) | ~mask).all(dim=1).float().mean().item()
    return correct


def main():
    config = TrainConfig()
    device = get_device(config)
    torch.manual_seed(config.seed)

    print("=" * 70)
    print("SEQ2SEQ WITH BEAM SEARCH: NUMBER REVERSAL")
    print("=" * 70)

    seq2seq_config = Seq2SeqConfig()

    # Create data
    src_train, tgt_train = make_reversal_dataset(
        config.n_train, config.max_seq_len)
    src_test, tgt_test = make_reversal_dataset(
        config.n_test, config.max_seq_len)

    train_ds = TensorDataset(src_train, tgt_train)
    test_ds = TensorDataset(src_test, tgt_test)
    train_loader = DataLoader(train_ds, config.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, config.batch_size)

    print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}")
    print(f"  Max seq len: {config.max_seq_len}")

    # Train
    model = Seq2Seq(seq2seq_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    print(f"\n--- Training ---")
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()

            # Teacher forcing schedule: decrease over epochs
            tf_ratio = max(0.1, 1.0 - epoch / config.epochs)
            output = model(src, tgt, teacher_forcing=tf_ratio)

            # Flatten for cross-entropy
            loss = F.cross_entropy(
                output.reshape(-1, seq2seq_config.tgt_vocab_size),
                tgt[:, 1:output.size(1)+1].reshape(-1),
                ignore_index=seq2seq_config.pad_token
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1:>3d} | Loss: {avg_loss:.4f} | "
                  f"TF ratio: {tf_ratio:.2f}")

    # Evaluate
    print(f"\n{'=' * 70}")
    print("DECODING COMPARISON")
    print(f"{'=' * 70}")

    model.eval()

    # Show examples
    print("\n--- Example Predictions ---")
    for i in range(5):
        src_seq = src_test[i:i+1].to(device)
        tgt_seq = tgt_test[i]

        greedy_out = model.greedy_decode(src_seq, config.max_seq_len + 2)
        beam_out = model.beam_search(src_seq, beam_width=5,
                                      max_len=config.max_seq_len + 2)

        # Extract non-padding tokens
        src_tokens = src_seq[0][src_seq[0] > 2].tolist()
        tgt_tokens = tgt_seq[tgt_seq > 2].tolist()
        greedy_tokens = greedy_out[0][(greedy_out[0] > 2) & (greedy_out[0] != 1)].tolist()
        beam_tokens = [t for t in beam_out if t > 2 and t != 1]

        print(f"  Source:  {src_tokens}")
        print(f"  Target:  {tgt_tokens}")
        print(f"  Greedy:  {greedy_tokens}")
        print(f"  Beam(5): {beam_tokens}")
        print()

    # Full evaluation
    print("--- Full Test Evaluation ---")
    greedy_correct, beam_correct, total = 0, 0, 0

    for src, tgt in test_loader:
        src = src.to(device)
        greedy_out = model.greedy_decode(src, config.max_seq_len + 2)

        # Pad greedy output to match target length
        tgt_trimmed = tgt[:, 1:greedy_out.size(1)+1]
        mask = tgt_trimmed != seq2seq_config.pad_token
        greedy_match = ((greedy_out.cpu() == tgt_trimmed) | ~mask).all(dim=1)
        greedy_correct += greedy_match.sum().item()
        total += src.size(0)

    # Beam search on subset (slower)
    beam_total = min(100, len(test_ds))
    beam_correct = 0
    for i in range(beam_total):
        src_seq = src_test[i:i+1].to(device)
        tgt_tokens = tgt_test[i][tgt_test[i] > 2].tolist()
        beam_out = model.beam_search(src_seq, beam_width=5,
                                      max_len=config.max_seq_len + 2)
        beam_tokens = [t for t in beam_out if t > 2 and t != 1]
        if beam_tokens == tgt_tokens:
            beam_correct += 1

    print(f"  Greedy accuracy:  {greedy_correct/total:.4f} ({greedy_correct}/{total})")
    print(f"  Beam(5) accuracy: {beam_correct/beam_total:.4f} ({beam_correct}/{beam_total})")


if __name__ == "__main__":
    main()

"""
Word2Vec Training on WikiText-2
=================================

Skip-gram with negative sampling for learning word embeddings.
Evaluates on word similarity and analogy tasks.

Usage:
    python train.py
    python train.py --embed_dim 200 --epochs 10 --window_size 5
"""

import sys
import os
import argparse
import time
import collections
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model import SkipGramNS, Word2VecConfig


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TrainConfig:
    embed_dim: int = 100
    window_size: int = 5
    n_negatives: int = 5
    min_count: int = 5
    lr: float = 0.001
    epochs: int = 10
    batch_size: int = 512
    subsample_t: float = 1e-3
    max_vocab: int = 30000
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

class SkipGramDataset(Dataset):
    """
    Generates (center, context) pairs from tokenized text.
    Applies subsampling to discard frequent words.
    """

    def __init__(self, token_ids: List[int], word_freqs: Dict[int, float],
                 window_size: int = 5, subsample_t: float = 1e-3):
        self.pairs = []
        n = len(token_ids)

        for i in range(n):
            center = token_ids[i]
            freq = word_freqs.get(center, 0)

            # Subsampling: discard frequent words with probability
            if freq > subsample_t:
                prob_discard = 1 - (subsample_t / freq) ** 0.5
                if random.random() < prob_discard:
                    continue

            # Random window size (1 to window_size)
            actual_window = random.randint(1, window_size)
            start = max(0, i - actual_window)
            end = min(n, i + actual_window + 1)

            for j in range(start, end):
                if j != i:
                    self.pairs.append((center, token_ids[j]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


class NegativeSampler:
    """Sample negatives according to unigram distribution raised to 0.75 power."""

    def __init__(self, word_counts: Dict[int, int], power: float = 0.75):
        words = list(word_counts.keys())
        freqs = np.array([word_counts[w] for w in words], dtype=np.float64)
        freqs = freqs ** power
        freqs /= freqs.sum()

        self.words = np.array(words)
        self.probs = freqs

        # Pre-sample a large table for efficiency
        self.table_size = 10_000_000
        self.table = np.random.choice(self.words, size=self.table_size, p=self.probs)
        self.idx = 0

    def sample(self, n: int) -> np.ndarray:
        if self.idx + n > self.table_size:
            self.idx = 0
        result = self.table[self.idx:self.idx + n]
        self.idx += n
        return result


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """Word2Vec training loop."""

    def __init__(self, model: SkipGramNS, config: TrainConfig,
                 device: torch.device, neg_sampler: NegativeSampler):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.neg_sampler = neg_sampler

    def train_one_epoch(self, loader: DataLoader, optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            center, context = batch
            center = center.to(self.device)
            context = context.to(self.device)

            # Sample negatives
            neg_np = self.neg_sampler.sample(
                center.size(0) * self.config.n_negatives)
            negatives = torch.tensor(
                neg_np.reshape(center.size(0), self.config.n_negatives),
                dtype=torch.long, device=self.device)

            optimizer.zero_grad()
            loss = self.model(center, context, negatives)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def fit(self, loader: DataLoader, vocab_words: List[str],
            word2idx: Dict[str, int]):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        print(f"  Parameters: {self.model.count_parameters():,}")
        print(f"  Training pairs: {len(loader.dataset):,}")
        print()

        for epoch in range(self.config.epochs):
            start = time.time()
            loss = self.train_one_epoch(loader, optimizer)
            elapsed = time.time() - start
            print(f"  Epoch {epoch+1:2d}/{self.config.epochs} ({elapsed:.1f}s): "
                  f"loss={loss:.4f}")

        # Evaluation
        self.evaluate_embeddings(vocab_words, word2idx)

    @torch.no_grad()
    def evaluate_embeddings(self, vocab_words: List[str],
                            word2idx: Dict[str, int]):
        """Evaluate with word similarity and analogy examples."""
        print("\n  Word Similarity Examples:")

        test_words = ["the", "king", "good", "computer", "water"]
        for word in test_words:
            if word not in word2idx:
                continue
            idx = word2idx[word]
            similar = self.model.most_similar(idx, top_k=5)
            sim_words = [f"{vocab_words[i]}({s:.3f})" for i, s in similar
                         if i < len(vocab_words)]
            print(f"    {word}: {', '.join(sim_words)}")

        # Analogy tests
        print("\n  Analogy Tests (a:b :: c:?):")
        analogies = [
            ("man", "woman", "king"),    # king - man + woman = queen?
            ("good", "better", "bad"),   # bad - good + better = worse?
        ]

        for a, b, c in analogies:
            if all(w in word2idx for w in [a, b, c]):
                results = self.model.analogy(
                    word2idx[a], word2idx[b], word2idx[c], top_k=3)
                answers = [f"{vocab_words[i]}({s:.3f})" for i, s in results
                           if i < len(vocab_words)]
                print(f"    {a}:{b} :: {c}:? → {', '.join(answers)}")
        print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Word2Vec on WikiText-2")
    parser.add_argument("--embed_dim", type=int, default=100)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--n_negatives", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(**vars(args))
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    device = get_device(config)

    print("=" * 60)
    print("WORD2VEC (Skip-gram + Negative Sampling)")
    print(f"Device: {device}")
    print("=" * 60)

    # Load data
    from datasets.text_datasets import load_wikitext2
    from datasets.utils import Vocabulary

    train_tokens, _, _, vocab = load_wikitext2(max_vocab_size=config.max_vocab)

    # Build word frequency table
    word_counts = collections.Counter()
    unk_id = vocab.word2idx["<unk>"]
    token_ids = []
    for t in train_tokens:
        idx = vocab.word2idx.get(t, unk_id)
        token_ids.append(idx)
        word_counts[idx] += 1

    total = sum(word_counts.values())
    word_freqs = {w: c / total for w, c in word_counts.items()}

    # Create dataset and sampler
    dataset = SkipGramDataset(
        token_ids, word_freqs,
        window_size=config.window_size,
        subsample_t=config.subsample_t,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)

    neg_sampler = NegativeSampler(word_counts)

    # Build model
    model_config = Word2VecConfig(
        vocab_size=len(vocab), embed_dim=config.embed_dim,
        window_size=config.window_size, n_negatives=config.n_negatives,
    )
    model = SkipGramNS(model_config)

    # Train
    trainer = Trainer(model, config, device, neg_sampler)
    trainer.fit(loader, vocab.idx2word, vocab.word2idx)


if __name__ == "__main__":
    main()

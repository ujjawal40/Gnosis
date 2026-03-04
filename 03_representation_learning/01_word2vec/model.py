"""
Word2Vec in PyTorch (Skip-gram with Negative Sampling)
========================================================

Learns dense word embeddings by predicting context words given
a center word, using negative sampling for efficiency.

Architecture:
    Center word → Embedding → dot product with context/negative embeddings → sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List


@dataclass
class Word2VecConfig:
    vocab_size: int = 30000
    embed_dim: int = 100
    window_size: int = 5
    n_negatives: int = 5
    min_count: int = 5
    subsample_threshold: float = 1e-3


class SkipGramNS(nn.Module):
    """
    Skip-gram with Negative Sampling (Mikolov et al., 2013).

    Instead of softmax over entire vocab (expensive), uses binary
    classification: is this word a real context word or a random negative?

    Two embedding matrices:
        - center_emb: embeddings for center words
        - context_emb: embeddings for context/negative words
    """

    def __init__(self, config: Word2VecConfig):
        super().__init__()
        self.config = config
        self.center_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.context_emb = nn.Embedding(config.vocab_size, config.embed_dim)

        # Initialize with small uniform weights (original paper)
        init_range = 0.5 / config.embed_dim
        self.center_emb.weight.data.uniform_(-init_range, init_range)
        self.context_emb.weight.data.zero_()

    def forward(self, center: torch.Tensor, context: torch.Tensor,
                negatives: torch.Tensor) -> torch.Tensor:
        """
        Args:
            center: (batch,) center word indices
            context: (batch,) positive context word indices
            negatives: (batch, n_neg) negative sample indices

        Returns:
            loss: scalar negative sampling loss
        """
        # Center embeddings: (batch, embed_dim)
        center_vec = self.center_emb(center)

        # Positive context: (batch, embed_dim)
        context_vec = self.context_emb(context)

        # Positive score: dot product → should be high
        pos_score = torch.sum(center_vec * context_vec, dim=1)
        pos_loss = F.logsigmoid(pos_score)

        # Negative scores: should be low
        neg_vec = self.context_emb(negatives)  # (batch, n_neg, embed_dim)
        neg_score = torch.bmm(neg_vec, center_vec.unsqueeze(2)).squeeze(2)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)

        # Total loss (maximize positive, minimize negative)
        return -(pos_loss + neg_loss).mean()

    def get_embeddings(self) -> torch.Tensor:
        """Return learned embeddings (average of center + context)."""
        return (self.center_emb.weight.data + self.context_emb.weight.data) / 2

    def most_similar(self, word_idx: int, top_k: int = 10) -> List[tuple]:
        """Find most similar words by cosine similarity."""
        embeddings = self.get_embeddings()
        word_vec = embeddings[word_idx].unsqueeze(0)

        # Cosine similarity
        norms = embeddings.norm(dim=1, keepdim=True)
        normalized = embeddings / (norms + 1e-8)
        word_normalized = word_vec / (word_vec.norm() + 1e-8)

        similarities = (normalized @ word_normalized.T).squeeze()
        values, indices = similarities.topk(top_k + 1)

        # Exclude the word itself
        results = [(idx.item(), val.item()) for idx, val in zip(indices, values)
                    if idx.item() != word_idx]
        return results[:top_k]

    def analogy(self, a: int, b: int, c: int, top_k: int = 5) -> List[tuple]:
        """
        Solve analogy: a is to b as c is to ?
        Example: king - man + woman = queen
        """
        embeddings = self.get_embeddings()
        vec = embeddings[b] - embeddings[a] + embeddings[c]

        norms = embeddings.norm(dim=1, keepdim=True)
        normalized = embeddings / (norms + 1e-8)
        vec_normalized = vec / (vec.norm() + 1e-8)

        similarities = (normalized @ vec_normalized).squeeze()
        # Exclude input words
        for idx in [a, b, c]:
            similarities[idx] = -float("inf")

        values, indices = similarities.topk(top_k)
        return [(idx.item(), val.item()) for idx, val in zip(indices, values)]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

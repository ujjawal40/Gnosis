"""
Tokenization in PyTorch
=========================

PyTorch-compatible tokenizers and embedding layers.
Wraps tokenizer output for use with nn.Embedding and Transformer models.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TokenizerConfig:
    vocab_size: int = 8000
    max_length: int = 128
    pad_token_id: int = 0
    unk_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3


# =============================================================================
# TOKENIZED BATCH
# =============================================================================

@dataclass
class TokenizedBatch:
    """Container for a batch of tokenized sequences."""
    input_ids: torch.Tensor        # (B, L)
    attention_mask: torch.Tensor   # (B, L) - 1 for real, 0 for pad
    lengths: torch.Tensor          # (B,) - actual lengths

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.lengths = self.lengths.to(device)
        return self


def collate_sequences(token_ids_list: List[List[int]],
                      max_length: int = 128,
                      pad_id: int = 0) -> TokenizedBatch:
    """Pad variable-length sequences into a batch."""
    batch_size = len(token_ids_list)
    lengths = [min(len(ids), max_length) for ids in token_ids_list]
    max_len = max(lengths)

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

    for i, (ids, length) in enumerate(zip(token_ids_list, lengths)):
        input_ids[i, :length] = torch.tensor(ids[:length])
        attention_mask[i, :length] = 1

    return TokenizedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        lengths=torch.tensor(lengths),
    )


# =============================================================================
# TOKEN EMBEDDING MODEL
# =============================================================================

class TokenEmbedding(nn.Module):
    """
    Token embedding with optional positional encoding.

    Converts token IDs to dense vectors:
        embedding(token_id) + position_encoding(position)
    """

    def __init__(self, vocab_size: int, embed_dim: int, max_length: int = 512,
                 dropout: float = 0.1, padding_idx: int = 0):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos_embed = nn.Embedding(max_length, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        return self.dropout(self.token_embed(x) + self.pos_embed(positions))


class TextClassifier(nn.Module):
    """Simple text classifier: Embedding → Mean Pool → MLP → Classes."""

    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_dim: int = 256, n_classes: int = 2,
                 max_length: int = 256):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, embed_dim, max_length)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, batch: TokenizedBatch) -> torch.Tensor:
        embeds = self.embedding(batch.input_ids)  # (B, L, D)
        # Masked mean pooling
        mask = batch.attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        pooled = (embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)

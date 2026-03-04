"""
RNN/LSTM for Language Modeling
================================

Word-level LSTM language model with embedding, multi-layer LSTM,
tied weights, and text generation capabilities.

Architecture:
    Token → Embedding → LSTM × N layers → Linear → Vocab logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class LSTMConfig:
    vocab_size: int = 30000
    embed_dim: int = 256
    hidden_dim: int = 256
    n_layers: int = 2
    dropout: float = 0.3
    tie_weights: bool = True


class WordLSTM(nn.Module):
    """
    Word-level LSTM language model.

    Features:
        - Learned word embeddings
        - Multi-layer LSTM with dropout between layers
        - Optional weight tying (embedding = output projection)
        - Supports variable-length sequences via BPTT

    Weight tying (Press & Wolf, 2017): sharing embedding and output
    weights reduces parameters and often improves perplexity.
    """

    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.drop = nn.Dropout(config.dropout)

        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.n_layers,
            dropout=config.dropout if config.n_layers > 1 else 0,
            batch_first=True,
        )

        self.output_proj = nn.Linear(config.hidden_dim, config.vocab_size)

        # Weight tying
        if config.tie_weights and config.embed_dim == config.hidden_dim:
            self.output_proj.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.output_proj.bias.data.zero_()
        if not self.config.tie_weights:
            self.output_proj.weight.data.uniform_(-init_range, init_range)

    def forward(self, x: torch.Tensor,
                hidden=None) -> tuple:
        """
        Args:
            x: (batch, seq_len) token indices
            hidden: optional (h_0, c_0) LSTM state

        Returns:
            logits: (batch, seq_len, vocab_size)
            hidden: (h_n, c_n) for continuing sequence
        """
        emb = self.drop(self.embedding(x))
        output, hidden = self.lstm(emb, hidden)
        output = self.drop(output)
        logits = self.output_proj(output)
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize LSTM hidden state with zeros."""
        h = torch.zeros(self.config.n_layers, batch_size,
                        self.config.hidden_dim, device=device)
        c = torch.zeros_like(h)
        return (h, c)

    @torch.no_grad()
    def generate(self, start_tokens: torch.Tensor, max_len: int = 100,
                 temperature: float = 1.0, top_k: int = 0) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            start_tokens: (1, seq_len) initial token indices
            max_len: max tokens to generate
            temperature: sampling temperature (lower = more conservative)
            top_k: if > 0, only sample from top-k tokens
        """
        self.eval()
        device = start_tokens.device
        generated = start_tokens.clone()
        hidden = self.init_hidden(1, device)

        # Process prompt
        logits, hidden = self(start_tokens, hidden)

        for _ in range(max_len):
            next_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                # Top-k filtering
                values, _ = torch.topk(next_logits, top_k)
                min_val = values[:, -1].unsqueeze(-1)
                next_logits[next_logits < min_val] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=1)

            logits, hidden = self(next_token, hidden)

        return generated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

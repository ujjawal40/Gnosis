"""
Sequence-to-Sequence with Beam Search in PyTorch
====================================================

PyTorch implementation of encoder-decoder with attention
and beam search decoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Optional


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Seq2SeqConfig:
    src_vocab_size: int = 20
    tgt_vocab_size: int = 20
    embed_dim: int = 64
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.1
    sos_token: int = 0
    eos_token: int = 1
    pad_token: int = 2


# =============================================================================
# ENCODER
# =============================================================================

class Encoder(nn.Module):
    """Bidirectional GRU encoder."""

    def __init__(self, config: Seq2SeqConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.src_vocab_size, config.embed_dim,
                                       padding_idx=config.pad_token)
        self.rnn = nn.GRU(config.embed_dim, config.hidden_dim,
                          num_layers=config.n_layers, bidirectional=True,
                          batch_first=True, dropout=config.dropout if config.n_layers > 1 else 0)
        self.fc = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

    def forward(self, src: torch.Tensor):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)

        # Combine bidirectional hidden states
        hidden = hidden.view(self.rnn.num_layers, 2, -1, self.rnn.hidden_size)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1)
        hidden = torch.tanh(self.fc(hidden))

        return outputs, hidden


# =============================================================================
# ATTENTION
# =============================================================================

class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention."""

    def __init__(self, hidden_dim: int, enc_dim: int):
        super().__init__()
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(enc_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, query: torch.Tensor, keys: torch.Tensor):
        # query: (batch, hidden), keys: (batch, seq, enc_dim)
        q = self.W_q(query).unsqueeze(1)  # (batch, 1, hidden)
        k = self.W_k(keys)                # (batch, seq, hidden)
        scores = self.v(torch.tanh(q + k)).squeeze(-1)  # (batch, seq)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)
        return context, attn_weights


# =============================================================================
# DECODER
# =============================================================================

class Decoder(nn.Module):
    """GRU decoder with attention."""

    def __init__(self, config: Seq2SeqConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.tgt_vocab_size, config.embed_dim,
                                       padding_idx=config.pad_token)
        self.attention = BahdanauAttention(config.hidden_dim,
                                           config.hidden_dim * 2)
        self.rnn = nn.GRU(config.embed_dim + config.hidden_dim * 2,
                          config.hidden_dim, num_layers=config.n_layers,
                          batch_first=True,
                          dropout=config.dropout if config.n_layers > 1 else 0)
        self.fc = nn.Linear(config.hidden_dim, config.tgt_vocab_size)

    def forward_step(self, token: torch.Tensor, hidden: torch.Tensor,
                     enc_outputs: torch.Tensor):
        """Single decoding step."""
        embedded = self.embedding(token)  # (batch, 1, embed)
        context, attn_w = self.attention(hidden[-1], enc_outputs)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        output, hidden = self.rnn(rnn_input, hidden)
        logits = self.fc(output.squeeze(1))
        return logits, hidden, attn_w


# =============================================================================
# SEQ2SEQ MODEL
# =============================================================================

class Seq2Seq(nn.Module):
    """Full encoder-decoder with attention."""

    def __init__(self, config: Seq2SeqConfig):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.config = config

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                teacher_forcing: float = 0.5):
        """Training forward pass with teacher forcing."""
        B, tgt_len = tgt.shape
        enc_outputs, hidden = self.encoder(src)

        outputs = []
        token = tgt[:, 0:1]  # SOS

        for t in range(1, tgt_len):
            logits, hidden, _ = self.decoder.forward_step(
                token, hidden, enc_outputs
            )
            outputs.append(logits)

            if torch.rand(1).item() < teacher_forcing:
                token = tgt[:, t:t+1]
            else:
                token = logits.argmax(-1).unsqueeze(1)

        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, max_len: int = 50):
        """Greedy decoding."""
        enc_outputs, hidden = self.encoder(src)
        B = src.size(0)
        token = torch.full((B, 1), self.config.sos_token,
                           device=src.device, dtype=torch.long)
        outputs = []

        for _ in range(max_len):
            logits, hidden, _ = self.decoder.forward_step(
                token, hidden, enc_outputs
            )
            token = logits.argmax(-1).unsqueeze(1)
            outputs.append(token.squeeze(1))
            if (token == self.config.eos_token).all():
                break

        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def beam_search(self, src: torch.Tensor, beam_width: int = 5,
                    max_len: int = 50, length_penalty: float = 0.6):
        """Beam search decoding (batch_size=1)."""
        enc_outputs, hidden = self.encoder(src)

        # Each beam: (log_prob, tokens, hidden)
        sos = torch.tensor([[self.config.sos_token]], device=src.device)
        beams = [(0.0, [self.config.sos_token], hidden)]
        completed = []

        for _ in range(max_len):
            candidates = []
            for score, tokens, h in beams:
                if tokens[-1] == self.config.eos_token:
                    completed.append((score, tokens))
                    continue

                token = torch.tensor([[tokens[-1]]], device=src.device)
                logits, new_h, _ = self.decoder.forward_step(
                    token, h, enc_outputs
                )
                log_probs = F.log_softmax(logits, dim=-1)[0]
                top_k = log_probs.topk(beam_width)

                for lp, tok in zip(top_k.values, top_k.indices):
                    candidates.append(
                        (score + lp.item(), tokens + [tok.item()], new_h)
                    )

            if not candidates:
                break

            candidates.sort(
                key=lambda x: x[0] / (len(x[1]) ** length_penalty),
                reverse=True
            )
            beams = candidates[:beam_width]

            if len(completed) >= beam_width:
                break

        all_results = [(s, t) for s, t, _ in beams] + completed
        all_results.sort(
            key=lambda x: x[0] / (len(x[1]) ** length_penalty),
            reverse=True
        )
        return all_results[0][1]

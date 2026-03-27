"""
Sequence-to-Sequence with Beam Search: From Scratch
======================================================

Encoder-decoder architecture with attention and beam search decoding.

Components:
    1. Encoder (bidirectional RNN)
    2. Decoder with attention
    3. Greedy decoding
    4. Beam search decoding
    5. Length normalization

All code uses only NumPy. No frameworks.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SAVE_DIR = Path(__file__).parent / "plots"
SAVE_DIR.mkdir(exist_ok=True)

np.random.seed(42)


# =============================================================================
# PART 1: RNN CELLS
# =============================================================================

class GRUCell:
    """GRU cell from scratch."""

    def __init__(self, input_dim: int, hidden_dim: int):
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_z = np.random.randn(hidden_dim, input_dim + hidden_dim) * scale
        self.b_z = np.zeros(hidden_dim)
        self.W_r = np.random.randn(hidden_dim, input_dim + hidden_dim) * scale
        self.b_r = np.zeros(hidden_dim)
        self.W_h = np.random.randn(hidden_dim, input_dim + hidden_dim) * scale
        self.b_h = np.zeros(hidden_dim)

    def forward(self, x, h_prev):
        """x: (batch, input_dim), h_prev: (batch, hidden_dim)"""
        concat = np.concatenate([x, h_prev], axis=-1)

        z = self._sigmoid(concat @ self.W_z.T + self.b_z)  # Update gate
        r = self._sigmoid(concat @ self.W_r.T + self.b_r)  # Reset gate

        concat_r = np.concatenate([x, r * h_prev], axis=-1)
        h_cand = np.tanh(concat_r @ self.W_h.T + self.b_h)

        h_new = (1 - z) * h_prev + z * h_cand
        return h_new

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


# =============================================================================
# PART 2: ATTENTION
# =============================================================================

def bahdanau_attention(query, keys, values):
    """
    Bahdanau (additive) attention.

    query: (batch, hidden_dim)
    keys: (batch, seq_len, hidden_dim)
    values: (batch, seq_len, hidden_dim)
    """
    # score = v^T tanh(W_q q + W_k k)
    # Simplified: dot product attention
    scores = np.einsum('bh,bsh->bs', query, keys)
    scores = scores / np.sqrt(keys.shape[-1])

    # Softmax
    exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn_weights = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-8)

    # Weighted sum of values
    context = np.einsum('bs,bsh->bh', attn_weights, values)
    return context, attn_weights


# =============================================================================
# PART 3: ENCODER
# =============================================================================

class Encoder:
    """Bidirectional GRU encoder."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        self.embedding = np.random.randn(vocab_size, embed_dim) * 0.1
        self.fwd_gru = GRUCell(embed_dim, hidden_dim)
        self.bwd_gru = GRUCell(embed_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, tokens):
        """tokens: (batch, seq_len) integer indices."""
        batch, seq_len = tokens.shape
        embedded = self.embedding[tokens]  # (batch, seq_len, embed_dim)

        # Forward pass
        h_fwd = np.zeros((batch, self.hidden_dim))
        fwd_outputs = []
        for t in range(seq_len):
            h_fwd = self.fwd_gru.forward(embedded[:, t], h_fwd)
            fwd_outputs.append(h_fwd)

        # Backward pass
        h_bwd = np.zeros((batch, self.hidden_dim))
        bwd_outputs = [None] * seq_len
        for t in range(seq_len - 1, -1, -1):
            h_bwd = self.bwd_gru.forward(embedded[:, t], h_bwd)
            bwd_outputs[t] = h_bwd

        # Concatenate forward and backward (sum for simplicity)
        enc_outputs = np.stack(
            [f + b for f, b in zip(fwd_outputs, bwd_outputs)], axis=1
        )
        final_hidden = fwd_outputs[-1] + bwd_outputs[0]
        return enc_outputs, final_hidden


# =============================================================================
# PART 4: DECODER WITH ATTENTION
# =============================================================================

class Decoder:
    """GRU decoder with attention."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        self.embedding = np.random.randn(vocab_size, embed_dim) * 0.1
        self.gru = GRUCell(embed_dim + hidden_dim, hidden_dim)
        self.output_proj = np.random.randn(vocab_size, hidden_dim) * 0.1
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

    def step(self, token, hidden, enc_outputs):
        """One decoding step."""
        embedded = self.embedding[token]  # (batch, embed_dim)

        # Attention
        context, attn_weights = bahdanau_attention(hidden, enc_outputs, enc_outputs)

        # GRU input = [embedding; context]
        gru_input = np.concatenate([embedded, context], axis=-1)
        hidden = self.gru.forward(gru_input, hidden)

        # Output logits
        logits = hidden @ self.output_proj.T
        return logits, hidden, attn_weights


# =============================================================================
# PART 5: BEAM SEARCH
# =============================================================================

def greedy_decode(decoder, enc_outputs, initial_hidden, sos_token: int,
                  eos_token: int, max_len: int = 50):
    """Simple greedy decoding."""
    batch = enc_outputs.shape[0]
    token = np.full((batch,), sos_token, dtype=int)
    hidden = initial_hidden
    outputs = []

    for _ in range(max_len):
        logits, hidden, _ = decoder.step(token, hidden, enc_outputs)
        token = logits.argmax(axis=-1)
        outputs.append(token)
        if np.all(token == eos_token):
            break

    return np.stack(outputs, axis=1)


def beam_search_decode(decoder, enc_outputs, initial_hidden, sos_token: int,
                       eos_token: int, beam_width: int = 5,
                       max_len: int = 50, length_penalty: float = 0.6):
    """
    Beam Search Decoding.

    Maintains top-K hypotheses at each step, expanding each and
    keeping the best K overall. Length normalization prevents
    bias toward shorter sequences.

    Score(Y) = log P(Y) / |Y|^α  (α = length_penalty)
    """
    # Only handle batch_size=1 for clarity
    enc = enc_outputs[0:1]  # (1, seq_len, hidden)

    # Each beam: (score, tokens, hidden)
    beams = [(0.0, [sos_token], initial_hidden[0:1])]
    completed = []

    for step in range(max_len):
        candidates = []

        for score, tokens, hidden in beams:
            if tokens[-1] == eos_token:
                completed.append((score, tokens, hidden))
                continue

            token = np.array([tokens[-1]])
            logits, new_hidden, _ = decoder.step(token, hidden, enc)

            # Log probabilities
            log_probs = logits[0] - np.log(np.sum(np.exp(logits[0] -
                                            logits[0].max())) + 1e-8) - logits[0].max()
            # Correction for log-sum-exp
            log_probs = logits[0] - (logits[0].max() +
                        np.log(np.sum(np.exp(logits[0] - logits[0].max()))))

            # Get top-K tokens
            top_k = np.argsort(log_probs)[-beam_width:]

            for tok in top_k:
                new_score = score + log_probs[tok]
                candidates.append((new_score, tokens + [tok], new_hidden))

        if not candidates:
            break

        # Keep top beam_width candidates (normalized by length)
        def norm_score(c):
            return c[0] / (len(c[1]) ** length_penalty)

        candidates.sort(key=norm_score, reverse=True)
        beams = candidates[:beam_width]

        if len(completed) >= beam_width:
            break

    # Return best
    all_results = completed + beams
    all_results.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty),
                     reverse=True)
    return all_results[0][1]  # Best sequence


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate seq2seq with beam search on a reverse task."""
    print("=" * 70)
    print("SEQUENCE-TO-SEQUENCE WITH BEAM SEARCH")
    print("=" * 70)

    vocab_size = 20
    embed_dim = 16
    hidden_dim = 32
    SOS, EOS = 0, 1

    encoder = Encoder(vocab_size, embed_dim, hidden_dim)
    decoder = Decoder(vocab_size, embed_dim, hidden_dim)

    # Test with random input
    src = np.array([[2, 5, 8, 3, 7]])  # batch=1
    enc_outputs, enc_hidden = encoder.forward(src)

    print(f"\nSource: {src[0].tolist()}")

    # Greedy
    greedy_out = greedy_decode(decoder, enc_outputs, enc_hidden, SOS, EOS, 10)
    print(f"Greedy output: {greedy_out[0].tolist()}")

    # Beam search
    for bw in [1, 3, 5]:
        beam_out = beam_search_decode(
            decoder, enc_outputs, enc_hidden, SOS, EOS,
            beam_width=bw, max_len=10
        )
        print(f"Beam (k={bw}) output: {beam_out}")

    # Attention visualization
    print("\n--- Attention Visualization ---")
    hidden = enc_hidden.copy()
    token = np.array([SOS])
    attn_matrix = []

    for _ in range(5):
        logits, hidden, attn_w = decoder.step(token, hidden, enc_outputs)
        attn_matrix.append(attn_w[0])
        token = logits.argmax(axis=-1)

    attn_matrix = np.stack(attn_matrix)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(attn_matrix, cmap='Blues', aspect='auto')
    ax.set_xlabel("Source Position")
    ax.set_ylabel("Target Step")
    ax.set_title("Attention Weights")
    plt.colorbar(ax.images[0])
    plt.savefig(SAVE_DIR / "seq2seq_attention.png", dpi=100)
    plt.close()
    print("  Attention plot saved.")


if __name__ == "__main__":
    demo()

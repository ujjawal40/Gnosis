"""
Transformer from Scratch
=========================

Complete implementation of the Transformer architecture in NumPy.
Includes positional encoding, multi-head attention, feed-forward network,
layer normalization, and residual connections.

This is the architecture behind GPT, BERT, and every modern language model.
"""

import numpy as np


# ==============================================================================
# Part 1: Building Blocks
# ==============================================================================

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def gelu(x):
    """GELU activation (used in modern transformers instead of ReLU)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class LayerNorm:
    """
    Layer Normalization: normalize across features (last dimension).
    Unlike BatchNorm, this normalizes per-example, so it works with
    variable-length sequences and doesn't depend on batch size.
    """

    def __init__(self, d_model, eps=1e-5):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def forward(self, x):
        self.x = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_norm + self.beta


class PositionalEncoding:
    """
    Sinusoidal positional encoding from 'Attention Is All You Need'.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Each dimension oscillates at a different frequency, creating a unique
    signature for each position that the model can learn to decode.
    """

    def __init__(self, d_model, max_len=512):
        pe = np.zeros((max_len, d_model))
        position = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe[np.newaxis, :, :]  # (1, max_len, d_model)

    def forward(self, x):
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]


class MultiHeadAttention:
    """Multi-head self-attention with optional causal mask."""

    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        scale = np.sqrt(1.0 / d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape

        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        # Split heads
        Q = Q.reshape(batch, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Attention scores
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores + mask

        self.attention_weights = softmax(scores)

        # Weighted values
        attended = self.attention_weights @ V

        # Combine heads
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)

        return attended @ self.W_O


class FeedForward:
    """
    Position-wise feed-forward network.
    FFN(x) = GELU(xW_1 + b_1)W_2 + b_2

    Inner dimension is typically 4x model dimension.
    """

    def __init__(self, d_model, d_ff=None):
        if d_ff is None:
            d_ff = 4 * d_model

        scale1 = np.sqrt(2.0 / d_model)
        scale2 = np.sqrt(2.0 / d_ff)
        self.W1 = np.random.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        self.h = gelu(x @ self.W1 + self.b1)
        return self.h @ self.W2 + self.b2


# ==============================================================================
# Part 2: Transformer Block and Full Model
# ==============================================================================

class TransformerBlock:
    """
    One Transformer layer (pre-norm variant):
        x = x + MHA(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(self, d_model, n_heads, d_ff=None):
        self.ln1 = LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x, mask=None):
        # Self-attention with residual
        normed = self.ln1.forward(x)
        attended = self.mha.forward(normed, mask)
        x = x + attended

        # Feed-forward with residual
        normed = self.ln2.forward(x)
        fed = self.ffn.forward(normed)
        x = x + fed

        return x


class TransformerLM:
    """
    Transformer Language Model (GPT-style decoder-only).

    Token embeddings + positional encoding → N transformer blocks → logits
    """

    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2,
                 max_seq_len=128):
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Token embedding
        self.token_embed = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Transformer blocks
        self.blocks = [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]

        # Final layer norm and output projection
        self.ln_final = LayerNorm(d_model)
        self.output_proj = np.random.randn(d_model, vocab_size) * 0.02

    def forward(self, token_ids):
        """
        Args:
            token_ids: (batch, seq_len) integer token indices
        Returns:
            logits: (batch, seq_len, vocab_size) unnormalized predictions
        """
        batch, seq_len = token_ids.shape

        # Embed tokens
        x = self.token_embed[token_ids]  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoding.forward(x)

        # Create causal mask
        mask = np.triu(np.ones((seq_len, seq_len)) * (-1e9), k=1)
        mask = mask[np.newaxis, np.newaxis, :, :]  # (1, 1, seq_len, seq_len)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)

        # Final norm and project to vocabulary
        x = self.ln_final.forward(x)
        logits = x @ self.output_proj

        return logits

    def generate(self, start_tokens, max_new_tokens=20, temperature=1.0):
        """Autoregressive generation: predict one token at a time."""
        tokens = list(start_tokens)

        for _ in range(max_new_tokens):
            # Forward pass on current sequence
            input_ids = np.array([tokens])
            logits = self.forward(input_ids)

            # Get logits for last position only
            next_logits = logits[0, -1, :] / temperature

            # Sample from distribution
            probs = softmax(next_logits)
            next_token = np.random.choice(len(probs), p=probs)
            tokens.append(next_token)

        return tokens


# ==============================================================================
# Part 3: Experiments
# ==============================================================================

def experiment_positional_encoding():
    """Visualize positional encodings."""
    print("=" * 60)
    print("EXPERIMENT 1: Positional Encoding")
    print("=" * 60)

    pe = PositionalEncoding(d_model=16, max_len=20)

    print("\nPositional encoding for first 6 positions, first 8 dimensions:")
    print(f"{'Pos':>4}" + "".join(f"{'d'+str(i):>8}" for i in range(8)))
    for pos in range(6):
        vals = pe.pe[0, pos, :8]
        print(f"{pos:4d}" + "".join(f"{v:8.4f}" for v in vals))

    print("\nLow dimensions: slow oscillation (captures global position)")
    print("High dimensions: fast oscillation (captures local patterns)")

    # Show that relative positions are preserved
    print("\nDot product similarity between positions:")
    for i in range(5):
        sims = []
        for j in range(5):
            sim = np.dot(pe.pe[0, i], pe.pe[0, j])
            sim /= np.linalg.norm(pe.pe[0, i]) * np.linalg.norm(pe.pe[0, j])
            sims.append(f"{sim:.2f}")
        print(f"  Pos {i}: {' '.join(sims)}")

    print("\nNearby positions have higher similarity — position is encoded!")


def experiment_transformer_forward():
    """Run a forward pass and inspect attention patterns."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Transformer Forward Pass")
    print("=" * 60)

    np.random.seed(42)
    vocab_size = 50
    model = TransformerLM(vocab_size=vocab_size, d_model=32, n_heads=4, n_layers=2)

    # Random input sequence
    tokens = np.array([[5, 12, 3, 8, 22, 15]])
    logits = model.forward(tokens)

    print(f"\nInput tokens: {tokens[0]}")
    print(f"Logits shape: {logits.shape}  (batch, seq_len, vocab_size)")

    # Show predicted next token probabilities for each position
    print("\nPredicted next tokens (top 3 per position):")
    for pos in range(tokens.shape[1]):
        probs = softmax(logits[0, pos])
        top3 = np.argsort(probs)[-3:][::-1]
        top3_probs = probs[top3]
        print(f"  After token {tokens[0, pos]:2d}: "
              f"next = {top3[0]:2d} ({top3_probs[0]:.3f}), "
              f"{top3[1]:2d} ({top3_probs[1]:.3f}), "
              f"{top3[2]:2d} ({top3_probs[2]:.3f})")

    # Show attention weights from first layer
    print("\nAttention weights (Layer 0, Head 0) — who attends to whom:")
    weights = model.blocks[0].mha.attention_weights[0, 0]
    for i in range(weights.shape[0]):
        bars = "".join("█" if w > 0.2 else "▒" if w > 0.1 else "░"
                       for w in weights[i, :i+1])
        print(f"  Token {i}: {bars}")

    print("  (Causal mask: each token only sees itself and previous tokens)")


def experiment_sequence_learning():
    """Train transformer to learn a simple pattern."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Learning a Simple Sequence Pattern")
    print("=" * 60)
    print("Task: Learn to repeat the pattern [A, B, C, A, B, C, ...]")

    np.random.seed(42)
    vocab_size = 10
    pattern = [1, 2, 3]  # A=1, B=2, C=3

    # Generate training data: sequences of repeating pattern
    def make_data(n_samples=50, seq_len=12):
        data = np.zeros((n_samples, seq_len), dtype=int)
        for i in range(n_samples):
            offset = np.random.randint(0, len(pattern))
            for t in range(seq_len):
                data[i, t] = pattern[(t + offset) % len(pattern)]
        return data

    model = TransformerLM(vocab_size=vocab_size, d_model=32, n_heads=4, n_layers=2)
    train_data = make_data(100, seq_len=9)
    lr = 0.001

    print(f"\nTraining on {len(train_data)} sequences of length 9")
    print(f"Pattern: {pattern} repeating\n")

    for epoch in range(201):
        # Forward pass
        logits = model.forward(train_data)

        # Cross-entropy loss: predict next token
        # Input: [1,2,3,1,2,3,1,2,3] → Target: [2,3,1,2,3,1,2,3,?]
        targets = np.roll(train_data, -1, axis=1)
        targets[:, -1] = train_data[:, 0]  # Wrap around

        # Compute loss and gradient
        probs = softmax(logits.reshape(-1, vocab_size))
        target_flat = targets.flatten()
        loss = -np.mean(np.log(probs[np.arange(len(target_flat)), target_flat] + 1e-10))

        # Simple gradient: adjust output projection
        # (Full backprop through transformer is complex — we do a simplified version)
        grad = probs.copy()
        grad[np.arange(len(target_flat)), target_flat] -= 1
        grad = grad.reshape(logits.shape) / len(train_data)

        # Update output projection (simplified training)
        x_final = model.ln_final.forward(
            model.blocks[-1].forward(
                model.pos_encoding.forward(model.token_embed[train_data]),
                np.triu(np.ones((9, 9)) * (-1e9), k=1)[np.newaxis, np.newaxis, :, :]
            )
        )
        dW = np.mean([x_final[i].T @ grad[i] for i in range(len(train_data))], axis=0)
        model.output_proj -= lr * dW
        model.token_embed -= lr * 0.1 * np.random.randn(*model.token_embed.shape)  # Small perturbation

        if epoch % 50 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f}")

    # Test generation
    print("\nGeneration test (starting with [1, 2]):")
    generated = model.generate([1, 2], max_new_tokens=10, temperature=0.5)
    print(f"  Generated: {generated}")
    print(f"  Expected:  {[pattern[i % 3] for i in range(12)]}")


def experiment_architecture_size():
    """Show how model size affects the transformer."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Transformer Architecture Sizes")
    print("=" * 60)

    configs = [
        ("Tiny",   {"d_model": 16,  "n_heads": 2, "n_layers": 1}),
        ("Small",  {"d_model": 32,  "n_heads": 4, "n_layers": 2}),
        ("Medium", {"d_model": 64,  "n_heads": 8, "n_layers": 4}),
        ("Large",  {"d_model": 128, "n_heads": 8, "n_layers": 6}),
    ]

    print(f"\n{'Name':>8} {'d_model':>8} {'heads':>6} {'layers':>7} {'params':>12}")
    print("-" * 50)

    for name, cfg in configs:
        model = TransformerLM(vocab_size=100, **cfg)

        # Count parameters
        n_params = model.token_embed.size + model.output_proj.size
        for block in model.blocks:
            n_params += (block.mha.W_Q.size + block.mha.W_K.size +
                         block.mha.W_V.size + block.mha.W_O.size)
            n_params += block.ffn.W1.size + block.ffn.W2.size
            n_params += block.ln1.gamma.size * 2 + block.ln2.gamma.size * 2

        print(f"{name:>8} {cfg['d_model']:>8} {cfg['n_heads']:>6} "
              f"{cfg['n_layers']:>7} {n_params:>12,}")

    print("\nFor reference:")
    print("  GPT-2:    d=768,  12 heads, 12 layers, 117M params")
    print("  GPT-3:    d=12288, 96 heads, 96 layers, 175B params")
    print("  LLaMA-7B: d=4096, 32 heads, 32 layers, 7B params")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("TRANSFORMER FROM SCRATCH")
    print("The architecture behind GPT, BERT, and modern AI\n")

    experiment_positional_encoding()
    experiment_transformer_forward()
    experiment_sequence_learning()
    experiment_architecture_size()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. Transformer = attention + FFN + residual + layer norm, stacked N times.
2. Positional encoding injects order (attention is permutation-invariant).
3. Causal mask makes it autoregressive (GPT-style generation).
4. Multi-head attention captures different relationship types simultaneously.
5. Feed-forward layers store "factual knowledge" as key-value patterns.
6. Residual connections enable training of very deep models.
7. Scaling: more layers, wider model, more heads = better performance.

This is the foundation of modern AI. Everything else builds on this.
Next: Encoder models (BERT) and decoder models (GPT) are specializations.
""")

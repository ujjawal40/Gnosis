"""
Attention Mechanisms from Scratch
==================================

Implements scaled dot-product attention and multi-head attention
using only NumPy. These are the building blocks of the Transformer.
"""

import numpy as np


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


# ==============================================================================
# Part 1: Scaled Dot-Product Attention
# ==============================================================================

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Attention(Q, K, V) = softmax(QK^T / √d_k) · V

    Args:
        Q: queries (batch, seq_len_q, d_k)
        K: keys (batch, seq_len_k, d_k)
        V: values (batch, seq_len_k, d_v)
        mask: optional mask (batch, seq_len_q, seq_len_k) — -inf where masked

    Returns:
        output: (batch, seq_len_q, d_v)
        weights: attention weights (batch, seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]

    # Compute attention scores
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)  # (batch, q_len, k_len)

    # Apply mask (for causal attention)
    if mask is not None:
        scores = scores + mask  # mask contains -inf for blocked positions

    # Softmax over keys
    weights = softmax(scores, axis=-1)

    # Weighted sum of values
    output = weights @ V  # (batch, q_len, d_v)

    return output, weights


# ==============================================================================
# Part 2: Multi-Head Attention
# ==============================================================================

class MultiHeadAttention:
    """
    Multi-head attention: run multiple attention functions in parallel,
    each with different learned projections.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
    head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
    """

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

    def forward(self, Q, K, V, mask=None):
        batch, seq_len, _ = Q.shape

        # Project to Q, K, V
        Q_proj = Q @ self.W_Q  # (batch, seq, d_model)
        K_proj = K @ self.W_K
        V_proj = V @ self.W_V

        # Split into heads: (batch, seq, d_model) -> (batch*n_heads, seq, d_k)
        Q_heads = self._split_heads(Q_proj)
        K_heads = self._split_heads(K_proj)
        V_heads = self._split_heads(V_proj)

        # Expand mask for multiple heads
        if mask is not None:
            mask = np.tile(mask, (self.n_heads, 1, 1))

        # Attention per head
        attended, weights = scaled_dot_product_attention(Q_heads, K_heads, V_heads, mask)

        # Combine heads: (batch*n_heads, seq, d_k) -> (batch, seq, d_model)
        combined = self._combine_heads(attended, batch)

        # Final projection
        output = combined @ self.W_O

        # Reshape weights for inspection
        self.attention_weights = weights.reshape(batch, self.n_heads, seq_len, -1)

        return output

    def _split_heads(self, x):
        """(batch, seq, d_model) -> (batch*n_heads, seq, d_k)"""
        batch, seq, _ = x.shape
        x = x.reshape(batch, seq, self.n_heads, self.d_k)
        x = x.transpose(0, 2, 1, 3)  # (batch, n_heads, seq, d_k)
        return x.reshape(batch * self.n_heads, seq, self.d_k)

    def _combine_heads(self, x, batch):
        """(batch*n_heads, seq, d_k) -> (batch, seq, d_model)"""
        seq = x.shape[1]
        x = x.reshape(batch, self.n_heads, seq, self.d_k)
        x = x.transpose(0, 2, 1, 3)  # (batch, seq, n_heads, d_k)
        return x.reshape(batch, seq, self.d_model)


def create_causal_mask(seq_len):
    """Create mask for autoregressive (decoder) attention."""
    mask = np.triu(np.ones((seq_len, seq_len)) * (-1e9), k=1)
    return mask[np.newaxis, :, :]  # (1, seq_len, seq_len)


# ==============================================================================
# Experiments
# ==============================================================================

def experiment_attention_basics():
    """Demonstrate how attention works on a simple example."""
    print("=" * 60)
    print("EXPERIMENT 1: How Attention Works")
    print("=" * 60)

    np.random.seed(42)

    # Simple example: 3 tokens, 4-dimensional
    # Token 0 and Token 2 are similar, Token 1 is different
    Q = np.array([[[1.0, 0.0, 1.0, 0.0],    # Query: similar to token 0,2
                    [0.0, 1.0, 0.0, 1.0],    # Query: similar to token 1
                    [1.0, 0.0, 1.0, 0.0]]])  # Query: similar to token 0,2

    K = np.array([[[1.0, 0.0, 1.0, 0.0],    # Key 0
                    [0.0, 1.0, 0.0, 1.0],    # Key 1
                    [0.9, 0.1, 0.9, 0.1]]])  # Key 2 (similar to key 0)

    V = np.array([[[1.0, 0.0, 0.0, 0.0],    # Value 0: "red"
                    [0.0, 1.0, 0.0, 0.0],    # Value 1: "green"
                    [0.0, 0.0, 1.0, 0.0]]])  # Value 2: "blue"

    output, weights = scaled_dot_product_attention(Q, K, V)

    print("\nAttention weights (which keys each query attends to):")
    print(f"  Query 0 -> Key weights: {weights[0, 0].round(3)}")
    print(f"  Query 1 -> Key weights: {weights[0, 1].round(3)}")
    print(f"  Query 2 -> Key weights: {weights[0, 2].round(3)}")

    print("\nOutput (weighted combination of values):")
    for i in range(3):
        print(f"  Token {i}: {output[0, i].round(3)}")

    print("\nQuery 0 (similar to keys 0,2) gets a mix of values 0 (red) and 2 (blue)")
    print("Query 1 (similar to key 1) gets mostly value 1 (green)")


def experiment_causal_mask():
    """Show how causal masking works for autoregressive models."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Causal (Autoregressive) Masking")
    print("=" * 60)

    np.random.seed(42)

    seq_len = 4
    d = 8
    Q = np.random.randn(1, seq_len, d)
    K = np.random.randn(1, seq_len, d)
    V = np.random.randn(1, seq_len, d)

    # Without mask: all tokens see all tokens
    _, weights_full = scaled_dot_product_attention(Q, K, V)
    print("\nAttention weights WITHOUT mask (bidirectional):")
    for i in range(seq_len):
        print(f"  Token {i}: {weights_full[0, i].round(3)}")

    # With causal mask: each token only sees itself and previous tokens
    mask = create_causal_mask(seq_len)
    _, weights_causal = scaled_dot_product_attention(Q, K, V, mask)
    print("\nAttention weights WITH causal mask (autoregressive):")
    for i in range(seq_len):
        print(f"  Token {i}: {weights_causal[0, i].round(3)}")

    print("\nToken 0 can only see itself: weight = [1, 0, 0, 0]")
    print("Token 3 can see all previous: non-zero weights for positions 0-3")
    print("This prevents 'cheating' — tokens can't look at the future!")


def experiment_multihead():
    """Show that multiple heads attend to different patterns."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Multi-Head Attention")
    print("=" * 60)

    np.random.seed(42)

    batch, seq_len, d_model = 1, 6, 16
    n_heads = 4

    X = np.random.randn(batch, seq_len, d_model)

    mha = MultiHeadAttention(d_model, n_heads)
    output = mha.forward(X, X, X)  # Self-attention

    print(f"\nInput shape:  {X.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Heads: {n_heads}, d_per_head: {d_model // n_heads}")

    print("\nAttention patterns per head (for token 0):")
    for h in range(n_heads):
        weights = mha.attention_weights[0, h, 0]
        max_pos = np.argmax(weights)
        print(f"  Head {h}: max attention at position {max_pos}, "
              f"weights = {weights.round(3)}")

    print("\nDifferent heads attend to different positions!")
    print("This lets the model capture multiple types of relationships simultaneously.")


def experiment_attention_as_soft_lookup():
    """Show attention as a differentiable key-value store."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Attention as Soft Dictionary Lookup")
    print("=" * 60)

    # Store: key-value pairs representing [animal:sound]
    # Keys encode the animal, values encode the sound
    K = np.array([[[1, 0, 0],    # cat
                    [0, 1, 0],    # dog
                    [0, 0, 1]]])  # bird

    V = np.array([[[1, 0],       # meow
                    [0, 1],       # bark
                    [0.5, 0.5]]]) # tweet

    print("\nKey-Value Store:")
    print("  cat  -> meow")
    print("  dog  -> bark")
    print("  bird -> tweet")

    # Query: exact match
    Q_exact = np.array([[[1, 0, 0]]])  # Looking for "cat"
    out, w = scaled_dot_product_attention(Q_exact, K, V)
    print(f"\nExact query 'cat': weights = {w[0,0].round(3)}, output = {out[0,0].round(3)}")
    print("  -> Gets 'meow' (exact lookup)")

    # Query: partial match
    Q_partial = np.array([[[0.5, 0.5, 0]]])  # Between cat and dog
    out, w = scaled_dot_product_attention(Q_partial, K, V)
    print(f"\nPartial query (cat+dog)/2: weights = {w[0,0].round(3)}, output = {out[0,0].round(3)}")
    print("  -> Gets mix of 'meow' and 'bark' (interpolated lookup)")

    print("\nThis soft lookup is what makes attention powerful:")
    print("  It can retrieve blended information based on partial similarity.")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("ATTENTION MECHANISMS FROM SCRATCH\n")

    experiment_attention_basics()
    experiment_causal_mask()
    experiment_multihead()
    experiment_attention_as_soft_lookup()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. Attention = soft dictionary lookup. Q asks, K answers, V provides.
2. Scaled dot product: QK^T/√d prevents softmax saturation.
3. Multi-head: different heads capture different relationship types.
4. Causal mask: prevents tokens from seeing the future (for generation).
5. Self-attention: every token attends to every other token — O(n²) cost.
6. This replaces the RNN bottleneck with direct token-to-token interaction.

Next: The Transformer combines attention with feed-forward layers,
layer norm, and residual connections into the dominant architecture.
""")

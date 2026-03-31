"""
Positional Encodings: From Scratch Implementation
====================================================

How transformers encode position information since self-attention
is permutation-invariant by default.

Techniques:
    1. Sinusoidal (fixed) — original Transformer
    2. Learned absolute — BERT, GPT-2
    3. Rotary Position Embedding (RoPE) — LLaMA, GPT-NeoX
    4. ALiBi (Attention with Linear Biases) — BLOOM
    5. Relative Position Bias — T5

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
# PART 1: SINUSOIDAL POSITIONAL ENCODING
# =============================================================================

def sinusoidal_encoding(max_len: int, d_model: int) -> np.ndarray:
    """
    Sinusoidal positional encoding (Vaswani et al., 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Properties:
        - Fixed (no learned parameters)
        - Can theoretically generalize to longer sequences
        - PE(pos+k) is a linear function of PE(pos) for any fixed k
    """
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


# =============================================================================
# PART 2: LEARNED POSITIONAL ENCODING
# =============================================================================

class LearnedPositionalEncoding:
    """
    Learned absolute positional embeddings.

    Simply a lookup table: position → embedding vector.
    Used by BERT, GPT-2.

    Pros: Can learn task-specific patterns
    Cons: Fixed maximum length, no extrapolation
    """

    def __init__(self, max_len: int, d_model: int):
        self.embeddings = np.random.randn(max_len, d_model) * 0.02
        self.max_len = max_len
        self.d_model = d_model

    def forward(self, seq_len: int) -> np.ndarray:
        return self.embeddings[:seq_len]


# =============================================================================
# PART 3: ROTARY POSITION EMBEDDING (RoPE)
# =============================================================================

def rope_frequencies(d_model: int, base: float = 10000.0) -> np.ndarray:
    """Compute RoPE frequency bands."""
    freqs = 1.0 / (base ** (np.arange(0, d_model, 2).astype(float) / d_model))
    return freqs


def apply_rope(x: np.ndarray, positions: np.ndarray,
               freqs: np.ndarray) -> np.ndarray:
    """
    Apply Rotary Position Embedding (Su et al., 2021).

    Instead of adding position to the embedding, RoPE rotates pairs of
    dimensions by position-dependent angles:

        [x_2i, x_2i+1] → R(θ_i * pos) @ [x_2i, x_2i+1]

    where R(θ) is a 2D rotation matrix.

    Key properties:
        - Relative position information encoded in dot products
        - q·k depends only on relative position (m-n), not absolute
        - Natural length extrapolation
        - Used by LLaMA, GPT-NeoX, Mistral, PaLM
    """
    seq_len, d = x.shape
    angles = np.outer(positions, freqs)  # (seq_len, d/2)

    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    # Split into pairs and rotate
    x_even = x[:, 0::2]  # (seq_len, d/2)
    x_odd = x[:, 1::2]

    rotated_even = x_even * cos_angles - x_odd * sin_angles
    rotated_odd = x_even * sin_angles + x_odd * cos_angles

    # Interleave back
    result = np.zeros_like(x)
    result[:, 0::2] = rotated_even
    result[:, 1::2] = rotated_odd
    return result


def rope_dot_product(q: np.ndarray, k: np.ndarray,
                     q_pos: np.ndarray, k_pos: np.ndarray,
                     freqs: np.ndarray) -> np.ndarray:
    """Compute attention scores with RoPE: score depends on relative position."""
    q_rot = apply_rope(q, q_pos, freqs)
    k_rot = apply_rope(k, k_pos, freqs)
    return q_rot @ k_rot.T


# =============================================================================
# PART 4: ALiBi (Attention with Linear Biases)
# =============================================================================

def alibi_slopes(n_heads: int) -> np.ndarray:
    """
    Compute ALiBi slopes for each attention head.

    Slopes follow geometric sequence: 2^(-8/n_heads), 2^(-16/n_heads), ...
    """
    ratio = 2 ** (-8.0 / n_heads)
    return ratio ** np.arange(1, n_heads + 1)


def alibi_bias(seq_len: int, n_heads: int) -> np.ndarray:
    """
    ALiBi: Attention with Linear Biases (Press et al., 2022).

    Instead of positional embeddings, add a linear bias to attention scores:
        attention = softmax(QK^T / sqrt(d) + m * |i - j|)

    where m is a head-specific slope and |i-j| is the distance between
    query position i and key position j.

    Properties:
        - No positional embeddings at all (no extra parameters)
        - Naturally extrapolates to longer sequences
        - Each head attends to different distance scales
        - Used by BLOOM, MPT
    """
    slopes = alibi_slopes(n_heads)
    # Distance matrix: -|i - j| (negative because it's a penalty)
    positions = np.arange(seq_len)
    distances = -np.abs(positions[:, None] - positions[None, :])

    # Each head gets its own slope
    bias = slopes[:, None, None] * distances[None, :, :]
    return bias  # (n_heads, seq_len, seq_len)


# =============================================================================
# PART 5: RELATIVE POSITION BIAS (T5)
# =============================================================================

def t5_relative_position_bucket(relative_position: np.ndarray,
                                 num_buckets: int = 32,
                                 max_distance: int = 128) -> np.ndarray:
    """
    T5-style relative position buckets.

    Maps relative positions to a fixed number of buckets:
        - Small distances map to individual buckets
        - Large distances share buckets (logarithmic)
        - Separate buckets for positive and negative positions

    This allows learning relative position patterns without
    an entry for every possible distance.
    """
    ret = np.zeros_like(relative_position)
    n = -relative_position
    n = np.maximum(n, 0)

    half_buckets = num_buckets // 2
    is_small = n < half_buckets

    # Log buckets for large distances
    val_if_large = half_buckets + (
        np.log(n.astype(float).clip(min=1) / half_buckets) /
        np.log(max_distance / half_buckets) * (half_buckets - 1)
    ).astype(int)
    val_if_large = np.minimum(val_if_large, num_buckets - 1)

    ret = np.where(is_small, n, val_if_large)
    return ret


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Compare positional encoding methods."""
    print("=" * 70)
    print("POSITIONAL ENCODING COMPARISON")
    print("=" * 70)

    seq_len, d_model = 64, 32
    n_heads = 8

    # 1. Sinusoidal
    pe_sin = sinusoidal_encoding(seq_len, d_model)
    print(f"\nSinusoidal: shape={pe_sin.shape}")

    # Verify relative position property
    dot_products = pe_sin @ pe_sin.T
    print(f"  Dot product[0,1] = {dot_products[0,1]:.4f}")
    print(f"  Dot product[5,6] = {dot_products[5,6]:.4f}")
    print(f"  (Should be similar — relative distance is same)")

    # 2. Learned
    learned = LearnedPositionalEncoding(seq_len, d_model)
    pe_learned = learned.forward(seq_len)
    print(f"\nLearned: shape={pe_learned.shape}")

    # 3. RoPE
    freqs = rope_frequencies(d_model)
    positions = np.arange(seq_len).astype(float)
    x = np.random.randn(seq_len, d_model)
    x_rotated = apply_rope(x, positions, freqs)
    print(f"\nRoPE: rotated shape={x_rotated.shape}")

    # Show relative position in dot products
    q = np.random.randn(seq_len, d_model) * 0.1
    k = np.random.randn(seq_len, d_model) * 0.1
    scores = rope_dot_product(q, k, positions, positions, freqs)
    print(f"  Attention scores shape: {scores.shape}")

    # 4. ALiBi
    bias = alibi_bias(seq_len, n_heads)
    print(f"\nALiBi: bias shape={bias.shape}")
    print(f"  Head 0 slope: {alibi_slopes(n_heads)[0]:.6f}")
    print(f"  Head {n_heads-1} slope: {alibi_slopes(n_heads)[-1]:.6f}")

    # 5. T5 relative position buckets
    rel_pos = np.arange(seq_len)[None, :] - np.arange(seq_len)[:, None]
    buckets = t5_relative_position_bucket(rel_pos)
    print(f"\nT5 relative buckets: shape={buckets.shape}, "
          f"unique={len(np.unique(buckets))}")

    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(pe_sin, aspect='auto', cmap='RdBu')
    axes[0, 0].set_title("Sinusoidal PE")
    axes[0, 0].set_xlabel("Dimension")
    axes[0, 0].set_ylabel("Position")

    axes[0, 1].imshow(dot_products, cmap='viridis')
    axes[0, 1].set_title("Sinusoidal Dot Products")

    axes[0, 2].imshow(bias[0], cmap='RdBu')
    axes[0, 2].set_title("ALiBi Bias (Head 0)")

    axes[1, 0].imshow(bias[-1], cmap='RdBu')
    axes[1, 0].set_title(f"ALiBi Bias (Head {n_heads-1})")

    axes[1, 1].imshow(buckets, cmap='viridis')
    axes[1, 1].set_title("T5 Relative Position Buckets")

    axes[1, 2].imshow(scores, cmap='viridis')
    axes[1, 2].set_title("RoPE Attention Scores")

    plt.tight_layout()
    plt.savefig(SAVE_DIR / "positional_encodings.png", dpi=100)
    plt.close()
    print(f"\nPlot saved to {SAVE_DIR / 'positional_encodings.png'}")


if __name__ == "__main__":
    demo()

"""
Efficient Attention Mechanisms: From Scratch
===============================================

Standard attention is O(n²) in sequence length. These techniques
reduce that cost for long sequences.

Techniques:
    1. Standard Multi-Head Attention (baseline)
    2. KV-Cache for autoregressive inference
    3. Multi-Query Attention (MQA)
    4. Grouped Query Attention (GQA) — LLaMA 2
    5. Sliding Window Attention — Mistral
    6. Linear Attention (kernel approximation)

All code uses only NumPy. No frameworks.
"""

import numpy as np
import time
from typing import Optional, Tuple

np.random.seed(42)


# =============================================================================
# PART 1: STANDARD MULTI-HEAD ATTENTION
# =============================================================================

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray,
                                  V: np.ndarray,
                                  mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Standard attention: softmax(QK^T / sqrt(d_k)) V

    Complexity: O(n² d) where n = sequence length, d = head dimension
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ V


def multi_head_attention(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray,
                         W_v: np.ndarray, W_o: np.ndarray,
                         n_heads: int, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Standard Multi-Head Attention."""
    seq_len, d_model = x.shape
    d_head = d_model // n_heads

    # Project
    Q = x @ W_q  # (seq_len, d_model)
    K = x @ W_k
    V = x @ W_v

    # Reshape to heads: (n_heads, seq_len, d_head)
    Q = Q.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    K = K.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    V = V.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)

    # Attention per head
    out = scaled_dot_product_attention(Q, K, V, mask)  # (n_heads, seq_len, d_head)

    # Concatenate heads
    out = out.transpose(1, 0, 2).reshape(seq_len, d_model)
    return out @ W_o


# =============================================================================
# PART 2: KV-CACHE
# =============================================================================

class KVCache:
    """
    Key-Value Cache for autoregressive generation.

    During generation, we only need to compute Q for the new token,
    but reuse K and V from all previous tokens. Without cache,
    generation is O(n² * T) for T tokens. With cache, it's O(n * T).

    This is THE optimization that makes LLM inference practical.
    """

    def __init__(self, n_heads: int, d_head: int, max_len: int = 2048):
        self.n_heads = n_heads
        self.d_head = d_head
        self.k_cache = np.zeros((n_heads, max_len, d_head))
        self.v_cache = np.zeros((n_heads, max_len, d_head))
        self.length = 0

    def update(self, k: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Add new K, V and return full cached K, V."""
        seq_len = k.shape[1]
        self.k_cache[:, self.length:self.length + seq_len] = k
        self.v_cache[:, self.length:self.length + seq_len] = v
        self.length += seq_len

        return (self.k_cache[:, :self.length].copy(),
                self.v_cache[:, :self.length].copy())

    def reset(self):
        self.length = 0


def attention_with_kv_cache(x_new: np.ndarray, cache: KVCache,
                            W_q: np.ndarray, W_k: np.ndarray,
                            W_v: np.ndarray, W_o: np.ndarray,
                            n_heads: int) -> np.ndarray:
    """Attention with KV-Cache for single new token."""
    d_model = x_new.shape[-1]
    d_head = d_model // n_heads

    Q_new = (x_new @ W_q).reshape(1, n_heads, d_head).transpose(1, 0, 2)
    K_new = (x_new @ W_k).reshape(1, n_heads, d_head).transpose(1, 0, 2)
    V_new = (x_new @ W_v).reshape(1, n_heads, d_head).transpose(1, 0, 2)

    K_full, V_full = cache.update(K_new, V_new)
    out = scaled_dot_product_attention(Q_new, K_full, V_full)
    out = out.transpose(1, 0, 2).reshape(1, d_model)
    return out @ W_o


# =============================================================================
# PART 3: MULTI-QUERY ATTENTION (MQA)
# =============================================================================

def multi_query_attention(x: np.ndarray, W_q: np.ndarray,
                          W_k_shared: np.ndarray, W_v_shared: np.ndarray,
                          W_o: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Multi-Query Attention (Shazeer, 2019).

    All heads share the same K and V projections (only Q differs per head).
    Reduces KV-cache memory by n_heads×, with minimal quality loss.

    Memory: O(n * d) instead of O(n * d * n_heads) for KV cache
    Used by: PaLM, Falcon
    """
    seq_len, d_model = x.shape
    d_head = d_model // n_heads

    Q = x @ W_q  # (seq_len, d_model) - per head
    K = x @ W_k_shared  # (seq_len, d_head) - shared
    V = x @ W_v_shared  # (seq_len, d_head) - shared

    Q = Q.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)

    # Broadcast K, V to all heads
    K = K[np.newaxis, :, :]  # (1, seq_len, d_head)
    V = V[np.newaxis, :, :]

    out = scaled_dot_product_attention(Q, K, V)
    out = out.transpose(1, 0, 2).reshape(seq_len, d_model)
    return out @ W_o


# =============================================================================
# PART 4: GROUPED QUERY ATTENTION (GQA)
# =============================================================================

def grouped_query_attention(x: np.ndarray, W_q: np.ndarray,
                            W_k: np.ndarray, W_v: np.ndarray,
                            W_o: np.ndarray, n_heads: int,
                            n_kv_groups: int) -> np.ndarray:
    """
    Grouped Query Attention (Ainslie et al., 2023).

    Middle ground between MHA and MQA: share K,V across groups of heads.

    n_kv_groups = 1 → MQA
    n_kv_groups = n_heads → MHA

    Used by: LLaMA 2, Mistral
    """
    seq_len, d_model = x.shape
    d_head = d_model // n_heads
    heads_per_group = n_heads // n_kv_groups

    Q = x @ W_q  # (seq_len, d_model)
    Q = Q.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)

    d_kv = d_head * n_kv_groups
    K = x @ W_k[:, :d_kv]  # (seq_len, n_kv_groups * d_head)
    V = x @ W_v[:, :d_kv]

    K = K.reshape(seq_len, n_kv_groups, d_head).transpose(1, 0, 2)
    V = V.reshape(seq_len, n_kv_groups, d_head).transpose(1, 0, 2)

    # Repeat K, V for each head in the group
    K = np.repeat(K, heads_per_group, axis=0)
    V = np.repeat(V, heads_per_group, axis=0)

    out = scaled_dot_product_attention(Q, K, V)
    out = out.transpose(1, 0, 2).reshape(seq_len, d_model)
    return out @ W_o


# =============================================================================
# PART 5: SLIDING WINDOW ATTENTION
# =============================================================================

def sliding_window_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                              window_size: int = 256) -> np.ndarray:
    """
    Sliding Window Attention (Beltagy et al., 2020; Mistral).

    Each token only attends to its local window of neighbors.
    Reduces complexity from O(n²) to O(n * w) where w = window_size.

    Despite only local attention, information propagates across layers:
    With L layers and window W, effective receptive field = L * W.

    Used by: Longformer, Mistral, BigBird
    """
    n_heads, seq_len, d_k = Q.shape
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)

    # Create sliding window mask
    mask = np.zeros((seq_len, seq_len), dtype=bool)
    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        mask[i, start:end] = True

    scores = np.where(mask[np.newaxis, :, :], scores, -1e9)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ V


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Benchmark efficient attention variants."""
    print("=" * 70)
    print("EFFICIENT ATTENTION COMPARISON")
    print("=" * 70)

    d_model = 64
    n_heads = 8
    d_head = d_model // n_heads

    # Weight matrices
    W_q = np.random.randn(d_model, d_model) * 0.02
    W_k = np.random.randn(d_model, d_model) * 0.02
    W_v = np.random.randn(d_model, d_model) * 0.02
    W_o = np.random.randn(d_model, d_model) * 0.02
    W_k_shared = np.random.randn(d_model, d_head) * 0.02
    W_v_shared = np.random.randn(d_model, d_head) * 0.02

    for seq_len in [64, 128, 256]:
        print(f"\n--- Sequence length: {seq_len} ---")
        x = np.random.randn(seq_len, d_model) * 0.1

        # Standard MHA
        t0 = time.time()
        out_mha = multi_head_attention(x, W_q, W_k, W_v, W_o, n_heads)
        t_mha = time.time() - t0

        # MQA
        t0 = time.time()
        out_mqa = multi_query_attention(x, W_q, W_k_shared, W_v_shared, W_o, n_heads)
        t_mqa = time.time() - t0

        # GQA (2 groups)
        t0 = time.time()
        out_gqa = grouped_query_attention(x, W_q, W_k, W_v, W_o, n_heads, 2)
        t_gqa = time.time() - t0

        print(f"  MHA:  {t_mha*1000:.2f}ms  output={out_mha.shape}")
        print(f"  MQA:  {t_mqa*1000:.2f}ms  output={out_mqa.shape}")
        print(f"  GQA:  {t_gqa*1000:.2f}ms  output={out_gqa.shape}")

    # KV-Cache demo
    print(f"\n{'=' * 70}")
    print("KV-CACHE DEMO (autoregressive generation)")
    print(f"{'=' * 70}")

    cache = KVCache(n_heads, d_head)
    x_init = np.random.randn(1, d_model) * 0.1

    # Simulate generating 20 tokens
    for step in range(20):
        out = attention_with_kv_cache(x_init, cache, W_q, W_k, W_v, W_o, n_heads)
        x_init = out  # Feed output as next input
        if step < 5 or step >= 18:
            print(f"  Step {step+1}: cache_len={cache.length}, output_norm={np.linalg.norm(out):.4f}")


if __name__ == "__main__":
    demo()

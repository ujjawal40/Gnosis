"""
Novel Architecture Prototypes
===============================

Experimental implementations of research ideas from the Gnosis research plan.
These are starting points for exploration, not finished products.

1. Geometric Attention: attention on a learned metric space
2. Hierarchical Compositional Embeddings: multi-level representations
3. Simple State Space Model: linear recurrence for sequences
"""

import numpy as np


# ==============================================================================
# 1. Geometric Attention
# ==============================================================================

class GeometricAttention:
    """
    Attention using learned distance on a metric space instead of dot product.

    Standard:  score = QK^T / √d       (flat Euclidean assumption)
    Geometric: score = -d_M(Q,K)² / τ  (learned Mahalanobis metric)

    d_M(q,k)² = (q-k)^T M (q-k) where M is a learned positive-definite matrix.
    """

    def __init__(self, d_model, temperature=1.0):
        self.d_model = d_model
        self.temperature = temperature

        # Learn metric M = L^T L (guaranteed positive semi-definite)
        self.L = np.eye(d_model) + np.random.randn(d_model, d_model) * 0.01

    @property
    def M(self):
        """Metric matrix (positive semi-definite)."""
        return self.L.T @ self.L

    def distance_squared(self, Q, K):
        """Compute Mahalanobis distance between all Q-K pairs."""
        # Q: (batch, seq_q, d), K: (batch, seq_k, d)
        # diff: (batch, seq_q, seq_k, d)
        diff = Q[:, :, np.newaxis, :] - K[:, np.newaxis, :, :]
        # d² = diff^T M diff
        M = self.M
        # (batch, seq_q, seq_k, d) @ (d, d) -> (batch, seq_q, seq_k, d)
        transformed = diff @ M
        # Sum over last dim: (batch, seq_q, seq_k)
        return np.sum(transformed * diff, axis=-1)

    def forward(self, Q, K, V, mask=None):
        """Geometric attention using learned distance."""
        d_sq = self.distance_squared(Q, K)  # (batch, seq_q, seq_k)

        # Attention scores (negative distance = closer = higher score)
        scores = -d_sq / self.temperature

        if mask is not None:
            scores = scores + mask

        # Softmax
        weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = weights / (np.sum(weights, axis=-1, keepdims=True) + 1e-8)

        self.weights = weights
        return weights @ V


def experiment_geometric_attention():
    """Compare geometric vs standard attention."""
    print("=" * 60)
    print("PROTOTYPE 1: Geometric Attention")
    print("=" * 60)

    np.random.seed(42)
    batch, seq_len, d = 1, 6, 8

    # Create data where semantically similar tokens are nearby
    # but have different dot products
    X = np.random.randn(batch, seq_len, d)

    # Standard dot-product attention
    def standard_attention(Q, K, V):
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d)
        weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights /= np.sum(weights, axis=-1, keepdims=True)
        return weights @ V, weights

    out_standard, w_standard = standard_attention(X, X, X)

    # Geometric attention
    geo_attn = GeometricAttention(d, temperature=1.0)
    out_geometric = geo_attn.forward(X, X, X)

    print(f"\nInput shape: {X.shape}")
    print(f"\nStandard attention weights (token 0):")
    print(f"  {w_standard[0, 0].round(3)}")
    print(f"\nGeometric attention weights (token 0):")
    print(f"  {geo_attn.weights[0, 0].round(3)}")

    print(f"\nDifference: geometric attention uses distance (proximity)")
    print(f"instead of dot product (angular similarity).")
    print(f"With a learned metric M, this captures non-Euclidean structure.")


# ==============================================================================
# 2. Hierarchical Compositional Embeddings
# ==============================================================================

class HierarchicalEmbedding:
    """
    Multi-level embedding where each level captures a different abstraction.

    Level 0: broadest (e.g., "entity type")
    Level 1: medium (e.g., "category")
    Level 2: specific (e.g., "individual identity")

    Composition: final embedding = concat(v0, v1, v2) or learned combination.
    """

    def __init__(self, vocab_size, level_dims=(64, 32, 16)):
        self.n_levels = len(level_dims)
        self.level_dims = level_dims
        self.total_dim = sum(level_dims)

        # Separate embedding table per level
        self.embeddings = []
        for dim in level_dims:
            self.embeddings.append(np.random.randn(vocab_size, dim) * 0.1)

        # Composition weights (learn how to combine levels)
        self.compose_W = np.random.randn(self.total_dim, self.total_dim) * 0.1
        self.compose_b = np.zeros(self.total_dim)

    def embed(self, token_ids):
        """Get hierarchical embedding for tokens."""
        level_embeds = []
        for level in range(self.n_levels):
            level_embeds.append(self.embeddings[level][token_ids])

        # Concatenate all levels
        concat = np.concatenate(level_embeds, axis=-1)

        # Optional: learned composition
        composed = np.tanh(concat @ self.compose_W + self.compose_b)
        return composed, level_embeds

    def similarity_per_level(self, idx1, idx2):
        """Compare two tokens at each hierarchical level."""
        sims = []
        for level in range(self.n_levels):
            v1 = self.embeddings[level][idx1]
            v2 = self.embeddings[level][idx2]
            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            sims.append(cos_sim)
        return sims


def experiment_hierarchical_embeddings():
    """Demonstrate hierarchical embedding structure."""
    print("\n" + "=" * 60)
    print("PROTOTYPE 2: Hierarchical Compositional Embeddings")
    print("=" * 60)

    np.random.seed(42)
    vocab_size = 20
    he = HierarchicalEmbedding(vocab_size, level_dims=(32, 16, 8))

    # Simulate hierarchy: tokens 0-4 are "animals", 5-9 are "vehicles"
    # Within animals: 0-2 are "mammals", 3-4 are "birds"
    animals = list(range(5))
    vehicles = list(range(5, 10))

    # Force broad-level similarity for same category
    for i in animals:
        he.embeddings[0][i] = np.random.randn(32) * 0.1 + np.array([1]*16 + [0]*16)
    for i in vehicles:
        he.embeddings[0][i] = np.random.randn(32) * 0.1 + np.array([0]*16 + [1]*16)

    names = {0: "cat", 1: "dog", 2: "horse", 3: "eagle", 4: "parrot",
             5: "car", 6: "truck", 7: "bus", 8: "plane", 9: "boat"}

    print(f"\nHierarchical similarity (Level 0=broad, Level 2=specific):")
    pairs = [(0, 1), (0, 5), (0, 3), (5, 6)]
    for i, j in pairs:
        sims = he.similarity_per_level(i, j)
        print(f"  {names[i]:6s} vs {names[j]:6s}: "
              f"L0={sims[0]:.3f}  L1={sims[1]:.3f}  L2={sims[2]:.3f}")

    print(f"\n  Level 0 (broad): cat-dog similar (both animals), cat-car different")
    print(f"  Level 2 (specific): even cat-dog are different (different animals)")
    print(f"\n  This hierarchy enables compositional generalization:")
    print(f"  'I know about cats (L2) and I know they're animals (L0)'")
    print(f"  → 'New animal X shares L0 properties but has unique L2'")


# ==============================================================================
# 3. Simple State Space Model
# ==============================================================================

class SimpleSSM:
    """
    Linear State Space Model for sequence processing.

    h_t = A·h_{t-1} + B·x_t     (state update)
    y_t = C·h_t + D·x_t         (output)

    With diagonal A (Mamba-style), this is O(n) in sequence length.
    """

    def __init__(self, input_dim, state_dim, output_dim):
        self.state_dim = state_dim

        # Diagonal state matrix (stable: eigenvalues < 1)
        self.A_diag = -np.exp(np.random.randn(state_dim) * 0.5)  # Negative real part
        self.B = np.random.randn(state_dim, input_dim) * 0.1
        self.C = np.random.randn(output_dim, state_dim) * 0.1
        self.D = np.random.randn(output_dim, input_dim) * 0.01

    def forward(self, x_seq):
        """
        Process a sequence: x_seq of shape (seq_len, input_dim).
        Returns: outputs of shape (seq_len, output_dim).
        """
        seq_len = x_seq.shape[0]
        h = np.zeros(self.state_dim)
        outputs = []

        for t in range(seq_len):
            # State update: h = A*h + B*x
            h = self.A_diag * h + self.B @ x_seq[t]
            # Output: y = C*h + D*x
            y = self.C @ h + self.D @ x_seq[t]
            outputs.append(y)

        return np.array(outputs)

    def forward_parallel(self, x_seq):
        """
        Parallel scan (conceptual — actual implementation uses associative scan).
        This shows the key insight: linear recurrence can be parallelized.
        """
        # For diagonal A, the solution at time t is:
        # h_t = Σ_{s=0}^{t} A^{t-s} B x_s
        # This is a convolution, computable with FFT in O(n log n)!

        seq_len = x_seq.shape[0]
        # Build convolution kernel
        kernel = np.zeros((seq_len, self.state_dim))
        for t in range(seq_len):
            kernel[t] = self.A_diag ** t

        # Apply (simplified — real implementation uses FFT)
        outputs = []
        for t in range(seq_len):
            h = np.zeros(self.state_dim)
            for s in range(t + 1):
                h += kernel[t - s] * (self.B @ x_seq[s])
            y = self.C @ h + self.D @ x_seq[t]
            outputs.append(y)

        return np.array(outputs)


def experiment_ssm():
    """Demonstrate state space model on sequence processing."""
    print("\n" + "=" * 60)
    print("PROTOTYPE 3: State Space Model (S4/Mamba-style)")
    print("=" * 60)

    np.random.seed(42)

    # Generate sine wave sequence
    t = np.linspace(0, 4 * np.pi, 100)
    x_seq = np.sin(t).reshape(-1, 1)

    ssm = SimpleSSM(input_dim=1, state_dim=16, output_dim=1)

    # Sequential processing
    out_seq = ssm.forward(x_seq)

    print(f"\nInput: sine wave, 100 timesteps")
    print(f"State dimension: 16")
    print(f"\nOutput (first 10 steps, sequential):")
    for i in range(10):
        print(f"  t={i:2d}: input={x_seq[i,0]:.4f}  output={out_seq[i,0]:.4f}")

    # Verify parallel gives same result
    out_par = ssm.forward_parallel(x_seq)
    max_diff = np.max(np.abs(out_seq - out_par))
    print(f"\nSequential vs parallel max difference: {max_diff:.2e}")
    print(f"(Should be ~0: both compute the same function)")

    print(f"\nKey insight: linear recurrence h=A*h+B*x can be:")
    print(f"  - Sequential: O(n) time, O(1) memory")
    print(f"  - Parallel:   O(n log n) time with parallel scan/FFT")
    print(f"  - Attention:  O(n²) time — SSMs are asymptotically faster!")

    print(f"\nComplexity comparison for sequence length n:")
    for n in [100, 1000, 10000, 100000]:
        attn = n * n
        ssm_c = n * 16  # n * state_dim
        ratio = attn / ssm_c
        print(f"  n={n:>6d}: Attention={attn:>12,} | SSM={ssm_c:>8,} | {ratio:.0f}x faster")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("NOVEL ARCHITECTURE PROTOTYPES")
    print("Research starting points for breakthroughs\n")

    experiment_geometric_attention()
    experiment_hierarchical_embeddings()
    experiment_ssm()

    print("\n" + "=" * 60)
    print("RESEARCH DIRECTIONS")
    print("=" * 60)
    print("""
Three promising directions from our foundations:

1. GEOMETRIC ATTENTION
   Replace dot product with learned distance metric.
   Potential: O(n log n) attention with semantic geometry.
   Next step: Learn the metric M during training and test on NLP tasks.

2. HIERARCHICAL EMBEDDINGS
   Multi-level representations with compositional structure.
   Potential: Better generalization, continual learning, interpretability.
   Next step: Design loss function that encourages hierarchy.

3. STATE SPACE MODELS
   Linear recurrence for O(n) sequence processing.
   Potential: Handle very long sequences efficiently.
   Next step: Hybrid SSM-attention architectures.

FINDING THE BREAKTHROUGH:
- Pick the direction that excites you most
- Find a specific task where current methods FAIL
- Show your approach succeeds where they don't
- That's your proof of concept → paper → startup
""")

"""
Contrastive & Self-Supervised Learning: From Scratch
======================================================

Core techniques for learning representations without labels.

Methods:
    1. Contrastive loss (pair-based)
    2. Triplet loss
    3. NT-Xent (SimCLR)
    4. BYOL-style (no negatives)
    5. Barlow Twins (redundancy reduction)

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
# PART 1: CONTRASTIVE LOSS
# =============================================================================

def contrastive_loss(z1: np.ndarray, z2: np.ndarray, label: int,
                     margin: float = 1.0) -> float:
    """
    Contrastive Loss (Chopra et al., 2005).

    L = (1-Y) * 0.5 * D^2 + Y * 0.5 * max(0, margin - D)^2

    Where Y=0 for similar pairs, Y=1 for dissimilar pairs,
    and D is the Euclidean distance.
    """
    dist = np.sqrt(np.sum((z1 - z2) ** 2) + 1e-8)
    if label == 0:  # similar
        return 0.5 * dist ** 2
    else:  # dissimilar
        return 0.5 * max(0, margin - dist) ** 2


def contrastive_loss_batch(z1: np.ndarray, z2: np.ndarray,
                            labels: np.ndarray, margin: float = 1.0):
    """Batch contrastive loss with gradients."""
    B = z1.shape[0]
    diff = z1 - z2
    dist = np.sqrt(np.sum(diff ** 2, axis=1) + 1e-8)

    # Similar pairs
    pos_loss = (1 - labels) * 0.5 * dist ** 2
    # Dissimilar pairs
    neg_loss = labels * 0.5 * np.maximum(0, margin - dist) ** 2

    loss = np.mean(pos_loss + neg_loss)

    # Gradients w.r.t. z1
    grad_pos = (1 - labels)[:, None] * diff / (dist[:, None] + 1e-8) * dist[:, None]
    mask = (margin - dist > 0).astype(float)
    grad_neg = -labels[:, None] * mask[:, None] * diff / (dist[:, None] + 1e-8) * (margin - dist)[:, None]
    grad_z1 = (grad_pos + grad_neg) / B

    return loss, grad_z1


# =============================================================================
# PART 2: TRIPLET LOSS
# =============================================================================

def triplet_loss(anchor: np.ndarray, positive: np.ndarray,
                 negative: np.ndarray, margin: float = 0.2):
    """
    Triplet Loss (Schroff et al., 2015).

    L = max(0, ||a - p||^2 - ||a - n||^2 + margin)

    Pulls anchors closer to positives and away from negatives.
    """
    dist_pos = np.sum((anchor - positive) ** 2, axis=-1)
    dist_neg = np.sum((anchor - negative) ** 2, axis=-1)
    loss = np.maximum(0, dist_pos - dist_neg + margin)
    return np.mean(loss)


def hard_negative_mining(anchor: np.ndarray, positives: np.ndarray,
                          negatives: np.ndarray) -> int:
    """Select hardest negative (closest to anchor)."""
    dists = np.sum((negatives - anchor) ** 2, axis=1)
    return np.argmin(dists)


# =============================================================================
# PART 3: NT-Xent (SimCLR)
# =============================================================================

def nt_xent_loss(z: np.ndarray, temperature: float = 0.5):
    """
    Normalized Temperature-scaled Cross Entropy (NT-Xent).

    Used in SimCLR (Chen et al., 2020).

    Given 2N embeddings (N pairs of augmented views), the loss
    encourages each pair (2i, 2i+1) to be similar while pushing
    apart all other pairs.

    L = -log(exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ))
    """
    N = z.shape[0] // 2  # Number of pairs

    # Normalize
    z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)

    # Cosine similarity matrix
    sim = z_norm @ z_norm.T / temperature

    # Mask out self-similarity
    mask = np.eye(2 * N, dtype=bool)
    sim[mask] = -1e9

    total_loss = 0.0
    for i in range(2 * N):
        # Positive pair index
        j = i + 1 if i % 2 == 0 else i - 1

        # Numerator: similarity with positive pair
        pos_sim = sim[i, j]

        # Denominator: sum over all negatives (excluding self)
        log_sum_exp = np.log(np.sum(np.exp(sim[i])) + 1e-8)

        total_loss += -pos_sim + log_sum_exp

    return total_loss / (2 * N)


# =============================================================================
# PART 4: BYOL LOSS (NO NEGATIVES)
# =============================================================================

def byol_loss(online_proj: np.ndarray, target_proj: np.ndarray):
    """
    BYOL loss (Grill et al., 2020).

    No negative pairs needed! Just minimize cosine distance
    between online and target network projections.

    L = 2 - 2 * <q, z> / (||q|| * ||z||)

    Uses stop-gradient on target and EMA update.
    """
    # Normalize
    q = online_proj / (np.linalg.norm(online_proj, axis=-1, keepdims=True) + 1e-8)
    z = target_proj / (np.linalg.norm(target_proj, axis=-1, keepdims=True) + 1e-8)

    return np.mean(2 - 2 * np.sum(q * z, axis=-1))


def ema_update(online_params: list, target_params: list,
               tau: float = 0.996) -> list:
    """Exponential Moving Average update for target network."""
    return [tau * tp + (1 - tau) * op
            for tp, op in zip(target_params, online_params)]


# =============================================================================
# PART 5: BARLOW TWINS
# =============================================================================

def barlow_twins_loss(z1: np.ndarray, z2: np.ndarray,
                       lambda_param: float = 0.005):
    """
    Barlow Twins (Zbontar et al., 2021).

    Makes the cross-correlation matrix of two augmented view
    embeddings close to identity. This avoids collapse without
    needing negative pairs, asymmetric networks, or momentum.

    L = Σ_i (1 - C_ii)^2 + λ Σ_{i≠j} C_ij^2
    """
    B = z1.shape[0]

    # Normalize along batch dimension
    z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + 1e-8)
    z2_norm = (z2 - z2.mean(0)) / (z2.std(0) + 1e-8)

    # Cross-correlation matrix
    C = (z1_norm.T @ z2_norm) / B

    D = C.shape[0]

    # On-diagonal: want to be 1
    on_diag = np.sum((np.diag(C) - 1) ** 2)

    # Off-diagonal: want to be 0
    off_diag = np.sum(C ** 2) - np.sum(np.diag(C) ** 2)

    return on_diag + lambda_param * off_diag


# =============================================================================
# PART 6: SIMPLE ENCODER
# =============================================================================

class SimpleEncoder:
    """MLP encoder for contrastive learning experiments."""

    def __init__(self, in_dim: int, hidden_dim: int, proj_dim: int):
        scale1 = np.sqrt(2.0 / in_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = np.random.randn(hidden_dim, in_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(proj_dim, hidden_dim) * scale2
        self.b2 = np.zeros(proj_dim)

    def forward(self, x):
        self.h = np.maximum(0, x @ self.W1.T + self.b1)
        return self.h @ self.W2.T + self.b2


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate contrastive learning methods."""
    print("=" * 70)
    print("CONTRASTIVE & SELF-SUPERVISED LEARNING")
    print("=" * 70)

    dim = 32
    n_samples = 100

    # Create clustered data (3 clusters)
    centers = np.array([[2, 0], [0, 2], [-2, -2]])
    X = np.vstack([np.random.randn(n_samples, 2) * 0.5 + c for c in centers])
    labels = np.repeat(np.arange(3), n_samples)

    # Project to higher dim
    W_proj = np.random.randn(dim, 2)
    Z = X @ W_proj.T

    # Test NT-Xent
    print("\n--- NT-Xent Loss ---")
    # Create positive pairs (augmented views = same + noise)
    z_pairs = np.zeros((20, dim))
    for i in range(10):
        z_pairs[2 * i] = Z[i]
        z_pairs[2 * i + 1] = Z[i] + np.random.randn(dim) * 0.1
    loss = nt_xent_loss(z_pairs, temperature=0.5)
    print(f"  NT-Xent loss (similar pairs): {loss:.4f}")

    random_pairs = Z[np.random.permutation(20)]
    loss_random = nt_xent_loss(random_pairs, temperature=0.5)
    print(f"  NT-Xent loss (random pairs):  {loss_random:.4f}")

    # Test Triplet
    print("\n--- Triplet Loss ---")
    anchors = Z[:10]
    positives = Z[:10] + np.random.randn(10, dim) * 0.1
    negatives = Z[n_samples:n_samples + 10]  # Different cluster
    loss_t = triplet_loss(anchors, positives, negatives)
    print(f"  Triplet loss (correct): {loss_t:.4f}")

    # Test Barlow Twins
    print("\n--- Barlow Twins ---")
    z1 = Z[:50]
    z2 = z1 + np.random.randn(50, dim) * 0.1  # Augmented views
    loss_bt = barlow_twins_loss(z1, z2)
    print(f"  Barlow Twins loss: {loss_bt:.4f}")

    z2_random = Z[np.random.permutation(50)]
    loss_bt_random = barlow_twins_loss(z1, z2_random)
    print(f"  Barlow Twins (random): {loss_bt_random:.4f}")

    # Test BYOL
    print("\n--- BYOL Loss ---")
    online = Z[:50]
    target = online + np.random.randn(50, dim) * 0.1
    loss_byol = byol_loss(online, target)
    print(f"  BYOL loss (similar): {loss_byol:.4f}")

    target_random = Z[np.random.permutation(50)]
    loss_byol_random = byol_loss(online, target_random)
    print(f"  BYOL loss (random):  {loss_byol_random:.4f}")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    temps = [0.1, 0.5, 2.0]
    for ax, temp in zip(axes, temps):
        losses_sim, losses_rand = [], []
        for _ in range(20):
            pairs = np.zeros((20, dim))
            for i in range(10):
                pairs[2*i] = Z[np.random.randint(len(Z))]
                pairs[2*i+1] = pairs[2*i] + np.random.randn(dim) * 0.1
            losses_sim.append(nt_xent_loss(pairs, temp))
            losses_rand.append(nt_xent_loss(Z[np.random.permutation(20)], temp))

        ax.boxplot([losses_sim, losses_rand], labels=["Similar", "Random"])
        ax.set_title(f"NT-Xent (τ={temp})")
        ax.set_ylabel("Loss")

    plt.tight_layout()
    plt.savefig(SAVE_DIR / "contrastive_learning.png", dpi=100)
    plt.close()
    print(f"\nPlot saved.")


if __name__ == "__main__":
    demo()

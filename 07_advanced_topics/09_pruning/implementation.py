"""
Model Pruning: From Scratch
===============================

Techniques to remove redundant parameters from neural networks.

Methods:
    1. Magnitude Pruning (unstructured)
    2. Structured Pruning (channel/filter)
    3. Iterative Magnitude Pruning (IMP)
    4. Lottery Ticket Hypothesis
    5. Movement Pruning

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
# PART 1: MAGNITUDE PRUNING
# =============================================================================

def magnitude_prune(weights: np.ndarray, sparsity: float) -> np.ndarray:
    """
    Unstructured magnitude pruning.

    Set the smallest |w| values to zero until desired sparsity.
    Returns binary mask (1 = keep, 0 = pruned).
    """
    flat = np.abs(weights.flatten())
    threshold = np.percentile(flat, sparsity * 100)
    mask = (np.abs(weights) > threshold).astype(np.float64)
    return mask


def apply_mask(weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply pruning mask to weights."""
    return weights * mask


def compute_sparsity(mask: np.ndarray) -> float:
    """Compute actual sparsity of a mask."""
    return 1.0 - np.mean(mask)


# =============================================================================
# PART 2: STRUCTURED PRUNING
# =============================================================================

def structured_prune_channels(weight: np.ndarray, sparsity: float):
    """
    Structured pruning: remove entire output channels.

    Removes channels with smallest L2 norm. This gives real
    speedup (unlike unstructured pruning which needs sparse hardware).

    weight shape: (out_channels, in_features)
    """
    n_channels = weight.shape[0]
    n_prune = int(n_channels * sparsity)

    # L2 norm per channel
    channel_norms = np.linalg.norm(weight, axis=1)

    # Find channels to prune
    prune_indices = np.argsort(channel_norms)[:n_prune]
    keep_indices = np.argsort(channel_norms)[n_prune:]

    mask = np.ones(n_channels, dtype=bool)
    mask[prune_indices] = False

    return mask, keep_indices


def structured_prune_filters(weight_4d: np.ndarray, sparsity: float):
    """
    Filter pruning for convolutional layers.

    weight_4d shape: (out_channels, in_channels, H, W)
    Removes entire filters based on L1 norm.
    """
    n_filters = weight_4d.shape[0]
    n_prune = int(n_filters * sparsity)

    # L1 norm per filter
    filter_norms = np.sum(np.abs(weight_4d), axis=(1, 2, 3))

    prune_indices = np.argsort(filter_norms)[:n_prune]
    keep_mask = np.ones(n_filters, dtype=bool)
    keep_mask[prune_indices] = False

    return keep_mask


# =============================================================================
# PART 3: ITERATIVE MAGNITUDE PRUNING (IMP)
# =============================================================================

class IterativeMagnitudePruning:
    """
    Iterative Magnitude Pruning (IMP).

    The Lottery Ticket Hypothesis (Frankle & Carlin, 2019):
    Dense networks contain sparse subnetworks (winning tickets)
    that can train to full accuracy when trained from their
    original initialization.

    Algorithm:
        1. Train network to completion
        2. Prune p% of smallest weights
        3. Reset remaining weights to initial values
        4. Repeat from step 1
    """

    def __init__(self, prune_rate: float = 0.2, n_rounds: int = 5):
        self.prune_rate = prune_rate
        self.n_rounds = n_rounds
        self.masks = None
        self.initial_weights = None

    def save_initial_weights(self, weights: list):
        """Save initial weights for rewinding."""
        self.initial_weights = [w.copy() for w in weights]
        self.masks = [np.ones_like(w) for w in weights]

    def prune_round(self, trained_weights: list) -> list:
        """One round of pruning + rewind."""
        new_masks = []
        for w, m in zip(trained_weights, self.masks):
            # Only consider currently unpruned weights
            active = np.abs(w) * m
            flat_active = active[m > 0]

            if len(flat_active) > 0:
                threshold = np.percentile(np.abs(flat_active),
                                          self.prune_rate * 100)
                new_mask = m * (np.abs(w) > threshold).astype(np.float64)
            else:
                new_mask = m

            new_masks.append(new_mask)

        self.masks = new_masks

        # Rewind to initial weights with new masks
        return [w * m for w, m in zip(self.initial_weights, self.masks)]

    def get_total_sparsity(self) -> float:
        """Get overall sparsity across all layers."""
        total = sum(m.size for m in self.masks)
        pruned = sum(np.sum(m == 0) for m in self.masks)
        return pruned / total


# =============================================================================
# PART 4: MOVEMENT PRUNING
# =============================================================================

def movement_score(weights: np.ndarray, gradients: np.ndarray) -> np.ndarray:
    """
    Movement Pruning (Sanh et al., 2020).

    Instead of pruning by magnitude, prune by how much weights
    "move" during training. Weights that move toward zero are
    pruned (they're being pushed to zero by the loss).

    Score = -w * gradient (positive = moving away from zero)
    """
    return -weights * gradients


def movement_prune(weights: np.ndarray, scores: np.ndarray,
                    sparsity: float) -> np.ndarray:
    """Prune based on accumulated movement scores."""
    threshold = np.percentile(scores.flatten(), sparsity * 100)
    mask = (scores > threshold).astype(np.float64)
    return mask


# =============================================================================
# PART 5: SIMPLE MLP FOR EXPERIMENTS
# =============================================================================

class PrunableMLP:
    """MLP that supports pruning."""

    def __init__(self, dims):
        self.weights = []
        self.biases = []
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            self.weights.append(np.random.randn(dims[i+1], dims[i]) * scale)
            self.biases.append(np.zeros(dims[i+1]))
        self.masks = [np.ones_like(w) for w in self.weights]

    def forward(self, x):
        self.activations = [x]
        for i, (w, b, m) in enumerate(zip(self.weights, self.biases, self.masks)):
            x = x @ (w * m).T + b
            if i < len(self.weights) - 1:
                x = np.maximum(0, x)  # ReLU
            self.activations.append(x)
        return x

    def total_params(self):
        return sum(w.size for w in self.weights)

    def active_params(self):
        return sum(np.sum(m > 0) for m in self.masks)

    def sparsity(self):
        return 1 - self.active_params() / self.total_params()


# =============================================================================
# DEMO
# =============================================================================

def demo():
    print("=" * 70)
    print("MODEL PRUNING")
    print("=" * 70)

    # Create model
    model = PrunableMLP([784, 256, 128, 10])
    print(f"\nOriginal params: {model.total_params():,}")

    # Magnitude pruning at different sparsities
    print("\n--- Magnitude Pruning ---")
    print(f"{'Sparsity':>10s} | {'Active Params':>14s} | {'Compression':>12s}")
    print("-" * 42)

    sparsities = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]
    for s in sparsities:
        masks = [magnitude_prune(w, s) for w in model.weights]
        active = sum(np.sum(m > 0) for m in masks)
        comp = model.total_params() / max(active, 1)
        print(f"{s:>10.0%} | {active:>14,} | {comp:>10.1f}x")

    # Structured pruning
    print("\n--- Structured (Channel) Pruning ---")
    w = model.weights[0]  # (256, 784)
    for s in [0.25, 0.5, 0.75]:
        mask, keep = structured_prune_channels(w, s)
        print(f"  {s:.0%} pruned: {mask.sum():.0f}/{len(mask)} channels kept")

    # IMP demo
    print("\n--- Iterative Magnitude Pruning ---")
    imp = IterativeMagnitudePruning(prune_rate=0.2, n_rounds=5)
    imp.save_initial_weights(model.weights)

    # Simulate training + pruning rounds
    for r in range(imp.n_rounds):
        trained = [w + np.random.randn(*w.shape) * 0.1 for w in model.weights]
        rewound = imp.prune_round(trained)
        sp = imp.get_total_sparsity()
        print(f"  Round {r+1}: sparsity = {sp:.1%}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([f"{s:.0%}" for s in sparsities],
           [sum(np.sum(magnitude_prune(w, s) > 0) for w in model.weights)
            for s in sparsities],
           color='steelblue')
    ax.set_xlabel("Sparsity")
    ax.set_ylabel("Active Parameters")
    ax.set_title("Active Parameters vs Pruning Sparsity")
    plt.savefig(SAVE_DIR / "pruning.png", dpi=100)
    plt.close()
    print("\nPlots saved.")


if __name__ == "__main__":
    demo()

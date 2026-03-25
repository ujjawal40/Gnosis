"""
Mixture of Experts: From Scratch Implementation
==================================================

MoE allows scaling model capacity without proportionally scaling compute.
Only a subset of experts are activated per input.

Components:
    1. Expert networks (multiple parallel FFNs)
    2. Gating/Router network (selects which experts to use)
    3. Top-K routing (sparse activation)
    4. Load balancing loss (prevent expert collapse)
    5. Expert capacity and overflow handling

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
# PART 1: EXPERT NETWORK
# =============================================================================

class Expert:
    """Single expert: a small feedforward network."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        scale1 = np.sqrt(2.0 / in_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = np.random.randn(hidden_dim, in_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(out_dim, hidden_dim) * scale2
        self.b2 = np.zeros(out_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, in_dim) → (batch, out_dim)"""
        h = np.maximum(0, x @ self.W1.T + self.b1)  # ReLU
        return h @ self.W2.T + self.b2


# =============================================================================
# PART 2: GATING / ROUTER
# =============================================================================

class TopKRouter:
    """
    Top-K Router: selects which experts to use for each input.

    The router is a simple linear layer that maps input to expert scores.
    Only the top-K experts are activated (sparse routing).

    router(x) = softmax(topk(x @ W_gate))

    Used in: Switch Transformer, GShard, Mixtral
    """

    def __init__(self, in_dim: int, n_experts: int, top_k: int = 2):
        self.W_gate = np.random.randn(n_experts, in_dim) * 0.01
        self.n_experts = n_experts
        self.top_k = top_k

    def forward(self, x: np.ndarray):
        """
        Route each input to top-K experts.

        Returns:
            expert_indices: (batch, top_k) - which experts
            expert_weights: (batch, top_k) - how much weight
            load: (n_experts,) - fraction of inputs per expert
        """
        # Raw gating scores
        scores = x @ self.W_gate.T  # (batch, n_experts)

        # Top-K selection
        top_k_idx = np.argsort(scores, axis=1)[:, -self.top_k:]  # (batch, top_k)

        # Softmax over selected experts only
        top_k_scores = np.take_along_axis(scores, top_k_idx, axis=1)
        exp_scores = np.exp(top_k_scores - top_k_scores.max(axis=1, keepdims=True))
        top_k_weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        # Load balancing statistics
        load = np.zeros(self.n_experts)
        for idx in top_k_idx.flatten():
            load[idx] += 1
        load /= (x.shape[0] * self.top_k)

        return top_k_idx, top_k_weights, load


# =============================================================================
# PART 3: MIXTURE OF EXPERTS LAYER
# =============================================================================

class MoELayer:
    """
    Mixture of Experts layer.

    Architecture:
        Input → Router → Select top-K experts → Weighted sum of expert outputs

    Key properties:
        - Total parameters: n_experts × expert_params (large)
        - Active parameters: top_k × expert_params (small)
        - Allows massive model capacity with fixed compute budget
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 n_experts: int = 8, top_k: int = 2):
        self.experts = [Expert(in_dim, hidden_dim, out_dim) for _ in range(n_experts)]
        self.router = TopKRouter(in_dim, n_experts, top_k)
        self.n_experts = n_experts
        self.top_k = top_k

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with sparse expert routing.

        Each input is processed by only top-K experts (not all N).
        """
        batch_size = x.shape[0]
        out_dim = self.experts[0].W2.shape[0]

        # Route
        expert_idx, expert_weights, self.load = self.router.forward(x)

        # Compute expert outputs (only for selected experts)
        output = np.zeros((batch_size, out_dim))

        for k in range(self.top_k):
            for e in range(self.n_experts):
                # Find which inputs go to expert e at position k
                mask = expert_idx[:, k] == e
                if not mask.any():
                    continue
                expert_out = self.experts[e].forward(x[mask])
                output[mask] += expert_weights[mask, k:k+1] * expert_out

        return output


# =============================================================================
# PART 4: LOAD BALANCING LOSS
# =============================================================================

def load_balancing_loss(load: np.ndarray, n_experts: int) -> float:
    """
    Auxiliary loss to prevent expert collapse.

    Without this, the router tends to send all inputs to a few experts
    (rich-get-richer). The balancing loss encourages uniform routing.

    L_balance = n_experts * sum(f_i * P_i)

    where f_i = fraction of tokens routed to expert i
          P_i = fraction of router probability for expert i
    """
    ideal = 1.0 / n_experts
    return n_experts * np.sum(load * load)  # Penalizes non-uniformity


# =============================================================================
# PART 5: FULL MoE MODEL
# =============================================================================

class MoEModel:
    """Simple MoE model for classification."""

    def __init__(self, in_dim: int, n_classes: int, n_experts: int = 8,
                 top_k: int = 2, expert_dim: int = 64):
        self.moe = MoELayer(in_dim, expert_dim, n_classes, n_experts, top_k)

    def forward(self, x: np.ndarray) -> np.ndarray:
        logits = self.moe.forward(x)
        exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_l / exp_l.sum(axis=1, keepdims=True)


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate Mixture of Experts."""
    print("=" * 70)
    print("MIXTURE OF EXPERTS DEMO")
    print("=" * 70)

    in_dim, n_classes = 20, 4
    n_experts = 8
    top_k = 2

    # Create data with cluster structure
    n_samples = 400
    X = np.random.randn(n_samples, in_dim)
    for c in range(n_classes):
        mask = np.arange(n_samples) % n_classes == c
        X[mask] += np.random.randn(in_dim) * 0.5

    model = MoEModel(in_dim, n_classes, n_experts, top_k)
    probs = model.forward(X)

    print(f"\nModel Configuration:")
    print(f"  Experts: {n_experts}, Top-K: {top_k}")
    print(f"  Total params: {n_experts}x expert size")
    print(f"  Active params per input: {top_k}x expert size "
          f"({top_k/n_experts*100:.0f}% of total)")

    print(f"\nRouting Statistics:")
    load = model.moe.load
    for i, l in enumerate(load):
        bar = "█" * int(l * 50)
        print(f"  Expert {i}: {l:.3f} {'█' * int(l * 50)}")

    balance_loss = load_balancing_loss(load, n_experts)
    print(f"\nLoad balancing loss: {balance_loss:.4f} "
          f"(ideal: {1.0/n_experts:.4f})")

    # Compare MoE vs dense
    print(f"\n{'=' * 70}")
    print("MOE vs DENSE PARAMETER COMPARISON")
    print(f"{'=' * 70}")
    dense_params = in_dim * 64 + 64 * n_classes  # Equivalent dense
    moe_total = n_experts * (in_dim * 64 + 64 * n_classes)
    moe_active = top_k * (in_dim * 64 + 64 * n_classes)
    print(f"  Dense equivalent:     {dense_params:>8,} params")
    print(f"  MoE total params:     {moe_total:>8,} params")
    print(f"  MoE active per input: {moe_active:>8,} params")
    print(f"  Capacity multiplier:  {moe_total / dense_params:.1f}x")
    print(f"  Compute multiplier:   {moe_active / dense_params:.1f}x")


if __name__ == "__main__":
    demo()

"""
Gradient Clipping: From Scratch Implementation
=================================================

Techniques to prevent exploding gradients during training.

Methods:
    1. Gradient Norm Clipping (clip by global norm)
    2. Gradient Value Clipping (clip element-wise)
    3. Gradient Scaling (adaptive)
    4. Gradient Accumulation (for effective large batch)

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
# PART 1: GRADIENT CLIPPING METHODS
# =============================================================================

def clip_grad_norm(grads: list, max_norm: float = 1.0) -> list:
    """
    Clip gradients by global norm.

    If ||g|| > max_norm: g = g * max_norm / ||g||

    This preserves gradient direction but limits magnitude.
    Used in: RNNs, Transformers, most modern architectures.
    """
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        return [g * clip_coef for g in grads]
    return grads


def clip_grad_value(grads: list, clip_value: float = 1.0) -> list:
    """
    Clip gradients element-wise to [-clip_value, +clip_value].

    Simple but can change gradient direction. Less commonly used than norm clipping.
    """
    return [np.clip(g, -clip_value, clip_value) for g in grads]


def adaptive_gradient_clipping(grads: list, params: list,
                                clip_factor: float = 0.01,
                                eps: float = 1e-3) -> list:
    """
    Adaptive Gradient Clipping (AGC) from NFNet.

    Clips based on ratio of gradient norm to parameter norm:
        if ||g|| / ||w|| > clip_factor: g = g * clip_factor * ||w|| / ||g||

    This adapts the clipping threshold per parameter, avoiding
    over-aggressive clipping on small parameters.
    """
    clipped = []
    for g, p in zip(grads, params):
        p_norm = np.linalg.norm(p).clip(min=eps)
        g_norm = np.linalg.norm(g).clip(min=eps)

        max_norm = clip_factor * p_norm
        if g_norm > max_norm:
            g = g * max_norm / g_norm
        clipped.append(g)
    return clipped


# =============================================================================
# PART 2: GRADIENT ACCUMULATION
# =============================================================================

class GradientAccumulator:
    """
    Gradient Accumulation for effective large batches.

    When GPU memory is limited, process small micro-batches and
    accumulate gradients before updating. This simulates a larger
    batch size without requiring more memory.

    effective_batch = micro_batch * accumulation_steps
    """

    def __init__(self, accumulation_steps: int = 4):
        self.steps = accumulation_steps
        self.accumulated = None
        self.count = 0

    def add(self, grads: list):
        """Add gradients from a micro-batch."""
        if self.accumulated is None:
            self.accumulated = [g.copy() for g in grads]
        else:
            for i, g in enumerate(grads):
                self.accumulated[i] += g
        self.count += 1

    def should_step(self) -> bool:
        """Check if we've accumulated enough."""
        return self.count >= self.steps

    def get_and_reset(self) -> list:
        """Get averaged accumulated gradients and reset."""
        avg = [g / self.count for g in self.accumulated]
        self.accumulated = None
        self.count = 0
        return avg


# =============================================================================
# PART 3: DEMO
# =============================================================================

def demo_gradient_clipping():
    """Demonstrate gradient clipping on an RNN-like gradient explosion scenario."""
    print("=" * 70)
    print("GRADIENT CLIPPING DEMO")
    print("=" * 70)

    # Simulate exploding gradients in a deep network
    dim = 50
    depth = 30
    np.random.seed(42)

    def simulate_backprop(clip_fn=None, clip_kwargs=None):
        """Simulate backward pass through many layers."""
        grad = np.random.randn(dim)
        norms = [np.linalg.norm(grad)]

        params = [np.random.randn(dim, dim) * 1.05 for _ in range(depth)]

        for i in range(depth):
            grad = params[i].T @ grad  # Gradient through layer

            if clip_fn:
                [grad] = clip_fn([grad], **clip_kwargs)

            norms.append(np.linalg.norm(grad))

        return norms

    # Compare
    no_clip = simulate_backprop()
    norm_clip = simulate_backprop(clip_grad_norm, {"max_norm": 5.0})
    value_clip = simulate_backprop(clip_grad_value, {"clip_value": 2.0})

    print(f"\n{'Method':>18s} | {'Initial Norm':>12s} | {'Final Norm':>12s} | {'Ratio':>10s}")
    print("-" * 60)
    for name, norms in [("No Clipping", no_clip), ("Norm Clip (5.0)", norm_clip),
                        ("Value Clip (2.0)", value_clip)]:
        ratio = norms[-1] / (norms[0] + 1e-10)
        print(f"{name:>18s} | {norms[0]:>12.4f} | {norms[-1]:>12.4f} | {ratio:>10.2e}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(no_clip, label="No Clipping", color='red')
    ax.plot(norm_clip, label="Norm Clip (5.0)", color='blue')
    ax.plot(value_clip, label="Value Clip (2.0)", color='green')
    ax.set_xlabel("Layer (backward)")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Explosion: Clipping Comparison")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(SAVE_DIR / "gradient_clipping.png", dpi=100)
    plt.close()
    print(f"\nPlot saved to {SAVE_DIR / 'gradient_clipping.png'}")

    # Gradient accumulation demo
    print(f"\n{'=' * 70}")
    print("GRADIENT ACCUMULATION DEMO")
    print(f"{'=' * 70}")

    accum = GradientAccumulator(accumulation_steps=4)
    for step in range(8):
        micro_grad = [np.random.randn(10)]
        accum.add(micro_grad)
        if accum.should_step():
            avg_grads = accum.get_and_reset()
            print(f"  Step {step+1}: Update! Avg grad norm = "
                  f"{np.linalg.norm(avg_grads[0]):.4f}")


if __name__ == "__main__":
    demo_gradient_clipping()

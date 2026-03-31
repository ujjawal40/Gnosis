"""
Multi-Task Learning: From Scratch
====================================

Training a single model on multiple related tasks simultaneously.

Concepts:
    1. Hard parameter sharing (shared backbone)
    2. Soft parameter sharing (cross-stitch)
    3. Task weighting strategies
    4. Gradient balancing (GradNorm)
    5. Uncertainty weighting

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
# PART 1: HARD PARAMETER SHARING
# =============================================================================

class SharedBackbone:
    """
    Hard parameter sharing: all tasks share the same backbone,
    with task-specific heads on top.
    """

    def __init__(self, in_dim: int, shared_dim: int, task_dims: list):
        scale = np.sqrt(2.0 / in_dim)
        self.W_shared = np.random.randn(shared_dim, in_dim) * scale
        self.b_shared = np.zeros(shared_dim)

        self.heads = []
        head_scale = np.sqrt(2.0 / shared_dim)
        for out_dim in task_dims:
            W = np.random.randn(out_dim, shared_dim) * head_scale
            b = np.zeros(out_dim)
            self.heads.append((W, b))

    def forward(self, x):
        # Shared representation
        h = np.maximum(0, x @ self.W_shared.T + self.b_shared)

        # Task-specific outputs
        outputs = []
        for W, b in self.heads:
            outputs.append(h @ W.T + b)
        return outputs, h


# =============================================================================
# PART 2: TASK WEIGHTING
# =============================================================================

def uniform_weighting(losses: list) -> np.ndarray:
    """Equal weight for all tasks."""
    n = len(losses)
    return np.ones(n) / n


def loss_ratio_weighting(losses: list, initial_losses: list) -> np.ndarray:
    """Weight inversely proportional to training progress."""
    ratios = np.array([l / (l0 + 1e-8) for l, l0 in zip(losses, initial_losses)])
    return ratios / ratios.sum()


class UncertaintyWeighting:
    """
    Uncertainty Weighting (Kendall et al., 2018).

    Learn task weights through homoscedastic uncertainty:
    L_total = Σ (1/(2σ²_t)) * L_t + log(σ_t)

    σ (or log_var) is learned per task, automatically balancing losses.
    """

    def __init__(self, n_tasks: int):
        self.log_vars = np.zeros(n_tasks)  # log(σ²)

    def weighted_loss(self, losses: list) -> float:
        total = 0.0
        for i, loss in enumerate(losses):
            precision = np.exp(-self.log_vars[i])  # 1/σ²
            total += precision * loss + self.log_vars[i]
        return total

    def update(self, losses: list, lr: float = 0.01):
        """Update log_var via gradient descent."""
        for i, loss in enumerate(losses):
            grad = -np.exp(-self.log_vars[i]) * loss + 1.0
            self.log_vars[i] -= lr * grad

    def get_weights(self) -> np.ndarray:
        return np.exp(-self.log_vars)


# =============================================================================
# PART 3: GRADNORM
# =============================================================================

class GradNorm:
    """
    GradNorm (Chen et al., 2018).

    Dynamically balances gradient magnitudes across tasks by
    adjusting task weights so that all tasks train at similar rates.
    """

    def __init__(self, n_tasks: int, alpha: float = 1.5):
        self.weights = np.ones(n_tasks)
        self.alpha = alpha
        self.initial_losses = None

    def update(self, losses: list, grad_norms: list, lr: float = 0.01):
        """
        Update task weights based on gradient norms.

        grad_norms: list of ||∇_shared w_i * L_i|| for each task
        """
        if self.initial_losses is None:
            self.initial_losses = np.array(losses)
            return

        # Training rates
        loss_ratios = np.array(losses) / (self.initial_losses + 1e-8)
        avg_ratio = loss_ratios.mean()
        relative_rates = loss_ratios / (avg_ratio + 1e-8)

        # Target gradient norms
        mean_norm = np.mean(grad_norms)
        targets = mean_norm * (relative_rates ** self.alpha)

        # Update weights to match target norms
        for i in range(len(self.weights)):
            grad_w = (grad_norms[i] - targets[i]) * grad_norms[i]
            self.weights[i] -= lr * grad_w

        # Normalize
        self.weights = self.weights * len(self.weights) / self.weights.sum()

    def get_weights(self) -> np.ndarray:
        return self.weights.copy()


# =============================================================================
# PART 4: CROSS-STITCH NETWORKS (SOFT SHARING)
# =============================================================================

class CrossStitch:
    """
    Cross-Stitch Networks (Misra et al., 2016).

    Soft parameter sharing: each task has its own backbone,
    but representations are linearly combined at each layer.

    h_A' = α_AA * h_A + α_AB * h_B
    h_B' = α_BA * h_A + α_BB * h_B
    """

    def __init__(self, n_tasks: int):
        # Cross-stitch matrix (initialized near identity)
        self.alpha = np.eye(n_tasks) * 0.9 + 0.1 / n_tasks

    def stitch(self, representations: list) -> list:
        """Combine representations from different tasks."""
        stacked = np.stack(representations)  # (n_tasks, batch, dim)
        n_tasks, batch, dim = stacked.shape

        result = []
        for i in range(n_tasks):
            combined = np.zeros((batch, dim))
            for j in range(n_tasks):
                combined += self.alpha[i, j] * stacked[j]
            result.append(combined)
        return result


# =============================================================================
# DEMO
# =============================================================================

def demo():
    print("=" * 70)
    print("MULTI-TASK LEARNING")
    print("=" * 70)

    # Synthetic multi-task problem: shared linear + task-specific
    dim = 20
    n_samples = 500
    X = np.random.randn(n_samples, dim)

    # Shared features + task-specific noise
    shared_w = np.random.randn(dim)
    w1 = shared_w + np.random.randn(dim) * 0.3
    w2 = shared_w + np.random.randn(dim) * 0.3
    w3 = shared_w + np.random.randn(dim) * 0.3

    y1 = X @ w1 + np.random.randn(n_samples) * 0.5
    y2 = X @ w2 + np.random.randn(n_samples) * 0.5
    y3 = X @ w3 + np.random.randn(n_samples) * 0.5

    # Compare weighting strategies
    print("\n--- Task Weighting Strategies ---")

    n_tasks = 3
    uw = UncertaintyWeighting(n_tasks)
    gn = GradNorm(n_tasks)

    losses = [2.0, 0.5, 5.0]  # Simulated losses at different scales

    print(f"\n  Initial losses: {losses}")
    print(f"  Uniform weights: {uniform_weighting(losses)}")

    # Simulate training
    for step in range(50):
        uw.update(losses, lr=0.1)
        gn.update(losses, [1.0, 0.5, 2.0])
        # Simulate loss decrease
        losses = [l * 0.95 + np.random.randn() * 0.1 for l in losses]

    print(f"  Uncertainty weights: {uw.get_weights().round(3)}")
    print(f"  GradNorm weights:    {gn.get_weights().round(3)}")

    # Cross-stitch demo
    print("\n--- Cross-Stitch Networks ---")
    cs = CrossStitch(2)
    h_a = np.random.randn(4, 8)
    h_b = np.random.randn(4, 8)
    stitched = cs.stitch([h_a, h_b])
    print(f"  Input norms:  task_A={np.linalg.norm(h_a):.3f}, "
          f"task_B={np.linalg.norm(h_b):.3f}")
    print(f"  Output norms: task_A={np.linalg.norm(stitched[0]):.3f}, "
          f"task_B={np.linalg.norm(stitched[1]):.3f}")
    print(f"  Cross-stitch matrix:\n{cs.alpha.round(3)}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    losses_hist = [2.0, 0.5, 5.0]
    uw2 = UncertaintyWeighting(3)
    w_history = []
    for _ in range(100):
        uw2.update(losses_hist, lr=0.05)
        w_history.append(uw2.get_weights().copy())
        losses_hist = [l * 0.98 for l in losses_hist]

    w_history = np.array(w_history)
    for i in range(3):
        ax1.plot(w_history[:, i], label=f"Task {i+1}")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Weight")
    ax1.set_title("Uncertainty Weighting Evolution")
    ax1.legend()

    # Multi-task vs single-task (simulated)
    methods = ["Single-Task\n(avg)", "Hard\nSharing", "Uncertainty\nWeighting", "GradNorm"]
    accs = [0.82, 0.85, 0.87, 0.86]
    ax2.bar(methods, accs, color=['gray', 'steelblue', 'coral', 'green'])
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Multi-Task Methods (simulated)")
    ax2.set_ylim(0.75, 0.9)

    plt.tight_layout()
    plt.savefig(SAVE_DIR / "multi_task_learning.png", dpi=100)
    plt.close()
    print("\nPlots saved.")


if __name__ == "__main__":
    demo()

"""
Learning Rate Finder: From Scratch Implementation
====================================================

The LR Range Test (Smith, 2017): exponentially increase LR during training
and find the sweet spot where loss decreases fastest.

Techniques:
    1. LR Range Test (exponential sweep)
    2. Cyclical Learning Rates
    3. 1cycle Policy
    4. Warmup strategies

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
# PART 1: LR RANGE TEST
# =============================================================================

class LRRangeTest:
    """
    Learning Rate Range Test.

    Gradually increase LR from min_lr to max_lr over num_steps.
    Track the loss at each step. The best LR is where loss
    decreases most steeply (steepest negative gradient).

    Usage:
        1. Start with very small LR (e.g., 1e-7)
        2. Exponentially increase each step
        3. Stop when loss starts increasing (diverging)
        4. Pick LR one order of magnitude below the minimum loss
    """

    def __init__(self, min_lr: float = 1e-7, max_lr: float = 10.0,
                 num_steps: int = 100, smooth_factor: float = 0.05):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps
        self.smooth_factor = smooth_factor
        self.lrs = []
        self.losses = []
        self.smoothed_losses = []

    def get_lr_schedule(self) -> np.ndarray:
        """Exponential LR schedule from min to max."""
        return np.exp(np.linspace(np.log(self.min_lr),
                                   np.log(self.max_lr),
                                   self.num_steps))

    def record(self, lr: float, loss: float):
        """Record a (lr, loss) pair."""
        self.lrs.append(lr)
        self.losses.append(loss)

        # Exponential moving average for smoothing
        if len(self.smoothed_losses) == 0:
            self.smoothed_losses.append(loss)
        else:
            smoothed = self.smooth_factor * loss + \
                       (1 - self.smooth_factor) * self.smoothed_losses[-1]
            self.smoothed_losses.append(smoothed)

    def suggest_lr(self) -> float:
        """Suggest optimal LR (steepest descent point / 10)."""
        if len(self.smoothed_losses) < 3:
            return self.min_lr

        losses = np.array(self.smoothed_losses)
        # Find point of steepest descent
        gradients = np.gradient(losses)
        min_idx = np.argmin(gradients)

        # Return LR at steepest descent / 10 (safety margin)
        suggested = self.lrs[min_idx] / 10
        return suggested

    def is_diverging(self, factor: float = 4.0) -> bool:
        """Check if loss has increased too much from minimum."""
        if len(self.smoothed_losses) < 2:
            return False
        min_loss = min(self.smoothed_losses)
        return self.smoothed_losses[-1] > min_loss * factor


# =============================================================================
# PART 2: CYCLICAL LEARNING RATES
# =============================================================================

class CyclicalLR:
    """
    Cyclical Learning Rates (Smith, 2017).

    Cycle LR between min and max. This allows the optimizer to explore
    different parts of the loss landscape and potentially escape
    sharp minima (which generalize poorly).

    Modes:
        - triangular: linear cycle
        - triangular2: halve max_lr each cycle
        - exp_range: exponential decay of max_lr
    """

    def __init__(self, base_lr: float = 1e-4, max_lr: float = 1e-2,
                 step_size: int = 2000, mode: str = "triangular",
                 gamma: float = 0.99994):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self.step_count = 0

    def get_lr(self) -> float:
        """Get current LR based on step."""
        cycle = 1 + self.step_count // (2 * self.step_size)
        x = abs(self.step_count / self.step_size - 2 * cycle + 1)

        if self.mode == "triangular":
            scale = 1.0
        elif self.mode == "triangular2":
            scale = 1.0 / (2.0 ** (cycle - 1))
        elif self.mode == "exp_range":
            scale = self.gamma ** self.step_count
        else:
            scale = 1.0

        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x) * scale
        return lr

    def step(self):
        self.step_count += 1


# =============================================================================
# PART 3: 1CYCLE POLICY
# =============================================================================

class OneCycleLR:
    """
    1cycle Policy (Smith & Topin, 2019).

    Two phases:
        1. Warmup: LR increases from base_lr to max_lr (30% of training)
        2. Annealing: LR decreases from max_lr to min_lr (70% of training)

    Combined with momentum annealing in opposite direction.
    """

    def __init__(self, max_lr: float, total_steps: int,
                 pct_start: float = 0.3, div_factor: float = 25.0,
                 final_div_factor: float = 1e4):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.initial_lr = max_lr / div_factor
        self.min_lr = max_lr / final_div_factor
        self.step_count = 0

    def get_lr(self) -> float:
        pct = self.step_count / self.total_steps
        if pct <= self.pct_start:
            # Phase 1: linear warmup
            t = pct / self.pct_start
            return self.initial_lr + (self.max_lr - self.initial_lr) * t
        else:
            # Phase 2: cosine annealing
            t = (pct - self.pct_start) / (1 - self.pct_start)
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                   (1 + np.cos(np.pi * t))

    def step(self):
        self.step_count += 1


# =============================================================================
# DEMO
# =============================================================================

def demo_lr_finder():
    """Simulate LR range test on a quadratic loss."""
    print("=" * 70)
    print("LEARNING RATE FINDER DEMO")
    print("=" * 70)

    # Simulate: loss = (w - 3)^2, gradient = 2(w - 3)
    w = 10.0
    finder = LRRangeTest(min_lr=1e-5, max_lr=10.0, num_steps=100)
    lrs = finder.get_lr_schedule()

    for lr in lrs:
        loss = (w - 3.0) ** 2
        grad = 2 * (w - 3.0)
        finder.record(lr, loss)

        if finder.is_diverging():
            print(f"  Diverged at LR = {lr:.6f}")
            break

        w -= lr * grad

    suggested = finder.suggest_lr()
    print(f"  Suggested LR: {suggested:.6f}")
    print(f"  Total steps recorded: {len(finder.lrs)}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(finder.lrs, finder.smoothed_losses, 'b-', linewidth=2)
    ax.axvline(suggested, color='r', linestyle='--', label=f'Suggested: {suggested:.5f}')
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Loss')
    ax.set_title('LR Range Test')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(SAVE_DIR / "lr_range_test.png", dpi=100)
    plt.close()

    # Demo cyclical and 1cycle
    print("\nLR Schedule Comparison:")
    total = 1000
    clr = CyclicalLR(1e-4, 1e-2, step_size=200)
    one = OneCycleLR(1e-2, total)

    clr_lrs, one_lrs = [], []
    for _ in range(total):
        clr_lrs.append(clr.get_lr())
        one_lrs.append(one.get_lr())
        clr.step()
        one.step()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(clr_lrs)
    ax1.set_title("Cyclical LR")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("LR")

    ax2.plot(one_lrs)
    ax2.set_title("1cycle Policy")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("LR")

    plt.tight_layout()
    plt.savefig(SAVE_DIR / "lr_schedules.png", dpi=100)
    plt.close()
    print("  Plots saved.")


if __name__ == "__main__":
    demo_lr_finder()

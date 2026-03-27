"""
Label Smoothing & Advanced Loss Techniques: From Scratch
===========================================================

Regularization techniques applied at the loss level.

Methods:
    1. Label Smoothing (Szegedy et al., 2016)
    2. Focal Loss (Lin et al., 2017)
    3. Knowledge Distillation Loss (soft targets)
    4. Mixup Loss
    5. Confidence Penalty

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
# PART 1: LABEL SMOOTHING
# =============================================================================

def label_smoothing(labels: np.ndarray, n_classes: int,
                     smoothing: float = 0.1) -> np.ndarray:
    """
    Label Smoothing (Szegedy et al., 2016).

    Instead of hard targets [0, 0, 1, 0]:
        smooth = [ε/K, ε/K, 1-ε+ε/K, ε/K]

    Where ε = smoothing, K = n_classes.

    Prevents overconfident predictions and improves calibration.
    """
    one_hot = np.eye(n_classes)[labels]
    smooth = one_hot * (1 - smoothing) + smoothing / n_classes
    return smooth


def smooth_cross_entropy(logits: np.ndarray, labels: np.ndarray,
                          n_classes: int, smoothing: float = 0.1):
    """Cross-entropy with label smoothing."""
    smooth_targets = label_smoothing(labels, n_classes, smoothing)

    # Softmax
    exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    log_probs = np.log(probs + 1e-8)

    loss = -np.sum(smooth_targets * log_probs, axis=-1)
    return np.mean(loss)


# =============================================================================
# PART 2: FOCAL LOSS
# =============================================================================

def focal_loss(logits: np.ndarray, labels: np.ndarray,
               gamma: float = 2.0, alpha: float = 0.25):
    """
    Focal Loss (Lin et al., 2017).

    FL = -α(1-p)^γ log(p)

    Down-weights easy examples (high p) and focuses on hard ones.
    Used in: object detection (RetinaNet), imbalanced classification.

    γ=0 → standard cross-entropy
    γ=2 → typical value, strongly focuses on hard examples
    """
    # Softmax
    exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    # Get probability for correct class
    batch_size = labels.shape[0]
    p_correct = probs[np.arange(batch_size), labels]

    # Focal weight
    focal_weight = (1 - p_correct) ** gamma

    loss = -alpha * focal_weight * np.log(p_correct + 1e-8)
    return np.mean(loss)


# =============================================================================
# PART 3: CONFIDENCE PENALTY
# =============================================================================

def confidence_penalty(logits: np.ndarray, beta: float = 0.1):
    """
    Confidence Penalty (Pereyra et al., 2017).

    Adds negative entropy to the loss to prevent overconfidence:
    L = CE + β * H(p)

    Where H(p) = -Σ p_i log(p_i) is the entropy of predictions.
    Higher entropy = less confident = more regularized.
    """
    exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    entropy = -np.sum(probs * np.log(probs + 1e-8), axis=-1)
    return -beta * np.mean(entropy)  # Negative because we maximize entropy


# =============================================================================
# PART 4: SYMMETRIC CROSS-ENTROPY
# =============================================================================

def symmetric_cross_entropy(logits: np.ndarray, labels: np.ndarray,
                             n_classes: int, alpha: float = 1.0,
                             beta: float = 1.0):
    """
    Symmetric Cross-Entropy (Wang et al., 2019).

    Robust to noisy labels:
    L = α * CE(p, q) + β * CE(q, p)

    The reverse CE term acts as a regularizer against label noise.
    """
    one_hot = np.eye(n_classes)[labels]

    exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    # Standard CE
    ce = -np.sum(one_hot * np.log(probs + 1e-8), axis=-1)

    # Reverse CE
    rce = -np.sum(probs * np.log(one_hot + 1e-4), axis=-1)

    return np.mean(alpha * ce + beta * rce)


# =============================================================================
# DEMO
# =============================================================================

def demo():
    print("=" * 70)
    print("LABEL SMOOTHING & ADVANCED LOSSES")
    print("=" * 70)

    n_classes = 10
    batch_size = 32

    # Simulated logits and labels
    logits = np.random.randn(batch_size, n_classes)
    labels = np.random.randint(0, n_classes, batch_size)

    # Standard CE
    exp_l = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=-1, keepdims=True)
    ce = -np.mean(np.log(probs[np.arange(batch_size), labels] + 1e-8))

    print(f"\n--- Loss Comparison ---")
    print(f"  Standard CE:          {ce:.4f}")
    print(f"  Label Smooth (0.1):   {smooth_cross_entropy(logits, labels, n_classes, 0.1):.4f}")
    print(f"  Label Smooth (0.2):   {smooth_cross_entropy(logits, labels, n_classes, 0.2):.4f}")
    print(f"  Focal (γ=0):          {focal_loss(logits, labels, gamma=0):.4f}")
    print(f"  Focal (γ=2):          {focal_loss(logits, labels, gamma=2):.4f}")
    print(f"  Focal (γ=5):          {focal_loss(logits, labels, gamma=5):.4f}")
    print(f"  Symmetric CE:         {symmetric_cross_entropy(logits, labels, n_classes):.4f}")

    # Show label smoothing effect
    print(f"\n--- Label Smoothing Visualization ---")
    hard = np.eye(n_classes)[3]
    smooth_01 = label_smoothing(np.array([3]), n_classes, 0.1)[0]
    smooth_03 = label_smoothing(np.array([3]), n_classes, 0.3)[0]
    print(f"  Hard:     {hard.round(3)}")
    print(f"  ε=0.1:    {smooth_01.round(3)}")
    print(f"  ε=0.3:    {smooth_03.round(3)}")

    # Focal loss behavior
    print(f"\n--- Focal Loss: Easy vs Hard Examples ---")
    easy_logit = np.array([[5.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
    hard_logit = np.array([[0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
    label = np.array([0])

    for gamma in [0, 1, 2, 5]:
        easy_fl = focal_loss(easy_logit, label, gamma=gamma)
        hard_fl = focal_loss(hard_logit, label, gamma=gamma)
        print(f"  γ={gamma}: easy={easy_fl:.4f}, hard={hard_fl:.4f}, "
              f"ratio={hard_fl/(easy_fl+1e-8):.1f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    gammas = np.arange(0, 5.1, 0.5)
    p_range = np.linspace(0.01, 0.99, 100)
    for gamma in [0, 1, 2, 5]:
        fl = -((1 - p_range) ** gamma) * np.log(p_range)
        ax1.plot(p_range, fl, label=f"γ={gamma}")
    ax1.set_xlabel("p (correct class probability)")
    ax1.set_ylabel("Loss")
    ax1.set_title("Focal Loss vs Probability")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    smoothings = [0, 0.05, 0.1, 0.2, 0.3]
    x = np.arange(n_classes)
    for s in smoothings:
        targets = label_smoothing(np.array([3]), n_classes, s)[0]
        ax2.bar(x + s * 2, targets, width=0.15, label=f"ε={s}")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Target probability")
    ax2.set_title("Label Smoothing Targets (true class=3)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(SAVE_DIR / "label_smoothing.png", dpi=100)
    plt.close()
    print("\nPlots saved.")


if __name__ == "__main__":
    demo()

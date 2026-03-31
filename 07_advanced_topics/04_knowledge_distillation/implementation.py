"""
Knowledge Distillation: From Scratch Implementation
======================================================

Train a small "student" network to mimic a large "teacher" network.
The teacher's soft probabilities contain more information than hard labels.

Techniques:
    1. Vanilla KD (Hinton et al., 2015) - soft target matching
    2. Feature-based KD - intermediate layer matching
    3. Self-distillation - model distills into itself
    4. Temperature scaling analysis

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
# PART 1: TEMPERATURE-SCALED SOFTMAX
# =============================================================================

def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Softmax with temperature scaling.

    Higher temperature → softer (more uniform) distribution
    Lower temperature → sharper (more peaked) distribution
    T=1 → standard softmax

    The key insight of KD: soft targets at high temperature reveal
    "dark knowledge" — which classes the teacher thinks are similar.
    Example: a teacher predicting "7" might assign 0.01 to "1" and
    0.001 to "9", revealing that 7 looks more like 1 than 9.
    """
    scaled = logits / temperature
    exp_s = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    return exp_s / exp_s.sum(axis=-1, keepdims=True)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) = sum(p * log(p/q))."""
    return np.sum(p * np.log((p + 1e-8) / (q + 1e-8)))


# =============================================================================
# PART 2: SIMPLE TEACHER AND STUDENT MODELS
# =============================================================================

class SimpleNetwork:
    """Simple MLP for distillation experiments."""

    def __init__(self, dims: list):
        self.layers = []
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            W = np.random.randn(dims[i + 1], dims[i]) * scale
            b = np.zeros(dims[i + 1])
            self.layers.append((W, b))

    def forward(self, x: np.ndarray, return_features: bool = False):
        """Forward pass. Returns logits (and optionally intermediate features)."""
        features = []
        h = x
        for i, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if i < len(self.layers) - 1:
                h = np.maximum(0, h)
                features.append(h.copy())

        if return_features:
            return h, features
        return h

    def backward(self, x, y_target, lr=0.01):
        """Simple backprop for training."""
        # Forward
        activations = [x]
        pre_acts = []
        h = x
        for i, (W, b) in enumerate(self.layers):
            z = h @ W.T + b
            pre_acts.append(z)
            if i < len(self.layers) - 1:
                h = np.maximum(0, z)
            else:
                h = z
            activations.append(h)

        # Softmax loss
        probs = softmax(h)
        grad = (probs - y_target) / x.shape[0]

        # Backward
        for i in range(len(self.layers) - 1, -1, -1):
            W, b = self.layers[i]
            grad_W = grad.T @ activations[i]
            grad_b = grad.sum(axis=0)

            if i > 0:
                grad = grad @ W
                grad *= (pre_acts[i - 1] > 0).astype(float)

            W -= lr * grad_W
            b -= lr * grad_b
            self.layers[i] = (W, b)

        return np.sum(-y_target * np.log(probs + 1e-8)) / x.shape[0]


# =============================================================================
# PART 3: DISTILLATION LOSS
# =============================================================================

def distillation_loss(student_logits: np.ndarray, teacher_logits: np.ndarray,
                      hard_labels: np.ndarray, temperature: float = 4.0,
                      alpha: float = 0.7) -> float:
    """
    Knowledge Distillation loss (Hinton et al., 2015).

    L = α * T² * KL(soft_teacher || soft_student) + (1-α) * CE(hard_labels, student)

    The T² factor compensates for the gradient magnitude change from temperature scaling.

    Args:
        alpha: weight for soft targets (0.7 = mostly teacher knowledge)
        temperature: softness of distributions (higher = softer)
    """
    n_classes = student_logits.shape[-1]

    # Soft targets from teacher
    soft_teacher = softmax(teacher_logits, temperature)
    soft_student = softmax(student_logits, temperature)

    # KL divergence on soft targets
    soft_loss = np.sum(soft_teacher * np.log((soft_teacher + 1e-8) / (soft_student + 1e-8)))
    soft_loss /= student_logits.shape[0]

    # Hard target cross-entropy
    student_probs = softmax(student_logits, 1.0)
    hard_loss = -np.sum(hard_labels * np.log(student_probs + 1e-8))
    hard_loss /= student_logits.shape[0]

    # Combined
    return alpha * temperature ** 2 * soft_loss + (1 - alpha) * hard_loss


# =============================================================================
# PART 4: FEATURE DISTILLATION
# =============================================================================

def feature_distillation_loss(student_features: list, teacher_features: list,
                               projectors: list = None) -> float:
    """
    Feature-based Knowledge Distillation.

    Match intermediate representations, not just output distributions.
    Uses MSE loss between student and teacher feature maps.
    If dimensions differ, apply a projection matrix.
    """
    total_loss = 0.0
    for i, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
        if projectors and projectors[i] is not None:
            s_feat = s_feat @ projectors[i]
        # Normalize features
        s_norm = s_feat / (np.linalg.norm(s_feat, axis=-1, keepdims=True) + 1e-8)
        t_norm = t_feat / (np.linalg.norm(t_feat, axis=-1, keepdims=True) + 1e-8)
        total_loss += np.mean((s_norm - t_norm) ** 2)
    return total_loss


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate knowledge distillation."""
    print("=" * 70)
    print("KNOWLEDGE DISTILLATION DEMO")
    print("=" * 70)

    # Create synthetic data: 3 classes, 20 features
    n_train, n_test = 500, 100
    n_features, n_classes = 20, 5
    X_train = np.random.randn(n_train, n_features)
    W_true = np.random.randn(n_features, n_classes)
    y_train_idx = np.argmax(X_train @ W_true, axis=1)
    y_train = np.eye(n_classes)[y_train_idx]

    X_test = np.random.randn(n_test, n_features)
    y_test_idx = np.argmax(X_test @ W_true, axis=1)

    # 1. Train large teacher
    print("\nTraining Teacher (large model)...")
    teacher = SimpleNetwork([n_features, 128, 64, n_classes])
    for epoch in range(200):
        teacher.backward(X_train, y_train, lr=0.01)
    teacher_logits = teacher.forward(X_test)
    teacher_preds = teacher_logits.argmax(axis=1)
    teacher_acc = (teacher_preds == y_test_idx).mean()
    print(f"  Teacher accuracy: {teacher_acc:.3f}")

    # 2. Train small student from scratch
    print("\nTraining Student (small model, from scratch)...")
    np.random.seed(42)
    student_scratch = SimpleNetwork([n_features, 32, n_classes])
    for epoch in range(200):
        student_scratch.backward(X_train, y_train, lr=0.01)
    scratch_logits = student_scratch.forward(X_test)
    scratch_preds = scratch_logits.argmax(axis=1)
    scratch_acc = (scratch_preds == y_test_idx).mean()
    print(f"  Student (scratch) accuracy: {scratch_acc:.3f}")

    # 3. Train student with distillation
    print("\nTraining Student (with distillation)...")
    np.random.seed(42)
    student_kd = SimpleNetwork([n_features, 32, n_classes])
    teacher_train_logits = teacher.forward(X_train)

    for epoch in range(200):
        student_logits_train = student_kd.forward(X_train)
        soft_teacher = softmax(teacher_train_logits, temperature=4.0)
        soft_target = 0.7 * soft_teacher + 0.3 * y_train
        student_kd.backward(X_train, soft_target, lr=0.01)

    kd_logits = student_kd.forward(X_test)
    kd_preds = kd_logits.argmax(axis=1)
    kd_acc = (kd_preds == y_test_idx).mean()
    print(f"  Student (distilled) accuracy: {kd_acc:.3f}")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"  Teacher (128+64):      {teacher_acc:.3f}")
    print(f"  Student (32, scratch): {scratch_acc:.3f}")
    print(f"  Student (32, KD):      {kd_acc:.3f}")
    print(f"  Improvement from KD:   +{(kd_acc - scratch_acc)*100:.1f}%")

    # Temperature analysis
    print(f"\n{'=' * 50}")
    print("Temperature Analysis:")
    teacher_test = teacher.forward(X_test[:1])
    for T in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
        probs = softmax(teacher_test, T)
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        print(f"  T={T:5.1f} | probs={np.round(probs[0], 3)} | "
              f"entropy={entropy:.3f}")


if __name__ == "__main__":
    demo()

"""
Transfer Learning: From Scratch Implementation
=================================================

How pre-trained representations transfer knowledge between tasks.

Concepts:
    1. Feature extraction (freeze backbone, train classifier)
    2. Fine-tuning (unfreeze and train end-to-end)
    3. Gradual unfreezing (unfreeze layers progressively)
    4. Discriminative learning rates (lower LR for earlier layers)
    5. Domain adaptation analysis

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
# PART 1: PRETRAINED FEATURE EXTRACTOR
# =============================================================================

class PretrainedMLP:
    """
    Simulates a pretrained network.

    In real transfer learning, this would be a model trained on
    a large dataset (ImageNet, BERT pretraining, etc.).
    The key insight: early layers learn general features (edges, textures),
    later layers learn task-specific features.
    """

    def __init__(self, dims: list):
        self.layers = []
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            W = np.random.randn(dims[i + 1], dims[i]) * scale
            b = np.zeros(dims[i + 1])
            self.layers.append((W, b))
        self.frozen = [False] * len(self.layers)

    def forward(self, x: np.ndarray, return_features: bool = False) -> np.ndarray:
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

    def freeze_layers(self, n_layers: int):
        """Freeze first n layers (prevent weight updates)."""
        for i in range(min(n_layers, len(self.layers))):
            self.frozen[i] = True

    def unfreeze_all(self):
        self.frozen = [False] * len(self.layers)

    def backward(self, x, y_target, lr=0.01, layer_lrs=None):
        """Train with optional discriminative learning rates."""
        # Forward
        activations = [x]
        pre_acts = []
        h = x
        for i, (W, b) in enumerate(self.layers):
            z = h @ W.T + b
            pre_acts.append(z)
            h = np.maximum(0, z) if i < len(self.layers) - 1 else z
            activations.append(h)

        # Softmax
        exp_h = np.exp(h - h.max(axis=1, keepdims=True))
        probs = exp_h / exp_h.sum(axis=1, keepdims=True)
        grad = (probs - y_target) / x.shape[0]

        # Backward
        for i in range(len(self.layers) - 1, -1, -1):
            if self.frozen[i]:
                break

            W, b = self.layers[i]
            grad_W = grad.T @ activations[i]
            grad_b = grad.sum(axis=0)

            # Discriminative LR
            layer_lr = layer_lrs[i] if layer_lrs else lr

            W -= layer_lr * grad_W
            b -= layer_lr * grad_b
            self.layers[i] = (W, b)

            if i > 0:
                grad = grad @ W
                grad *= (pre_acts[i - 1] > 0).astype(float)

        loss = -np.sum(y_target * np.log(probs + 1e-8)) / x.shape[0]
        return loss


# =============================================================================
# PART 2: TRANSFER STRATEGIES
# =============================================================================

def feature_extraction(pretrained, X_train, y_train, X_test, y_test, n_classes):
    """
    Feature Extraction: freeze backbone, only train new classifier.

    Fastest to train. Works well when target task is similar to pretrained task.
    """
    # Extract features from pretrained (frozen)
    _, train_features = pretrained.forward(X_train, return_features=True)
    _, test_features = pretrained.forward(X_test, return_features=True)

    # Use last hidden layer features
    train_feat = train_features[-1]
    test_feat = test_features[-1]

    # Train simple linear classifier on features
    feat_dim = train_feat.shape[1]
    scale = np.sqrt(2.0 / feat_dim)
    W = np.random.randn(n_classes, feat_dim) * scale
    b = np.zeros(n_classes)

    for epoch in range(100):
        logits = train_feat @ W.T + b
        exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)
        grad = (probs - y_train) / len(y_train)
        W -= 0.1 * grad.T @ train_feat
        b -= 0.1 * grad.sum(axis=0)

    test_logits = test_feat @ W.T + b
    return (test_logits.argmax(axis=1) == y_test.argmax(axis=1)).mean()


def fine_tuning(pretrained, X_train, y_train, X_test, y_test, lr=0.001):
    """Fine-tune entire network with small learning rate."""
    pretrained.unfreeze_all()
    for epoch in range(100):
        pretrained.backward(X_train, y_train, lr=lr)

    logits = pretrained.forward(X_test)
    return (logits.argmax(axis=1) == y_test.argmax(axis=1)).mean()


def discriminative_lr(pretrained, X_train, y_train, X_test, y_test, base_lr=0.01):
    """
    Discriminative Learning Rates (Howard & Ruder, 2018).

    Earlier layers get smaller LR (they have good general features).
    Later layers get larger LR (need more adaptation to new task).
    """
    pretrained.unfreeze_all()
    n = len(pretrained.layers)
    layer_lrs = [base_lr * (0.3 ** (n - 1 - i)) for i in range(n)]

    for epoch in range(100):
        pretrained.backward(X_train, y_train, layer_lrs=layer_lrs)

    logits = pretrained.forward(X_test)
    return (logits.argmax(axis=1) == y_test.argmax(axis=1)).mean()


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Compare transfer learning strategies."""
    print("=" * 70)
    print("TRANSFER LEARNING COMPARISON")
    print("=" * 70)

    n_features = 20
    n_classes_source = 5
    n_classes_target = 3

    # Source task: generate and "pretrain"
    np.random.seed(42)
    X_source = np.random.randn(1000, n_features)
    W_source = np.random.randn(n_features, n_classes_source)
    y_source_idx = np.argmax(X_source @ W_source, axis=1)
    y_source = np.eye(n_classes_source)[y_source_idx]

    pretrained = PretrainedMLP([n_features, 64, 32, n_classes_source])
    for _ in range(300):
        pretrained.backward(X_source, y_source, lr=0.01)

    source_logits = pretrained.forward(X_source)
    source_acc = (source_logits.argmax(1) == y_source_idx).mean()
    print(f"Source task accuracy: {source_acc:.3f}")

    # Target task: related but different
    np.random.seed(123)
    X_train = np.random.randn(200, n_features)
    X_test = np.random.randn(200, n_features)
    W_target = W_source[:, :n_classes_target] + np.random.randn(n_features, n_classes_target) * 0.3
    y_train_idx = np.argmax(X_train @ W_target, axis=1)
    y_test_idx = np.argmax(X_test @ W_target, axis=1)
    y_train = np.eye(n_classes_target)[y_train_idx]
    y_test = np.eye(n_classes_target)[y_test_idx]

    # 1. Train from scratch
    np.random.seed(42)
    scratch = PretrainedMLP([n_features, 64, 32, n_classes_target])
    for _ in range(200):
        scratch.backward(X_train, y_train, lr=0.01)
    scratch_logits = scratch.forward(X_test)
    scratch_acc = (scratch_logits.argmax(1) == y_test_idx).mean()

    # 2. Feature extraction
    np.random.seed(42)
    import copy
    feat_acc = feature_extraction(pretrained, X_train, y_train, X_test, y_test, n_classes_target)

    # 3. Fine-tuning
    np.random.seed(42)
    ft_model = PretrainedMLP([n_features, 64, 32, n_classes_target])
    ft_model.layers = [(W.copy(), b.copy()) for W, b in pretrained.layers[:-1]]
    scale = np.sqrt(2.0 / 32)
    ft_model.layers.append((np.random.randn(n_classes_target, 32) * scale, np.zeros(n_classes_target)))
    ft_acc = fine_tuning(ft_model, X_train, y_train, X_test, y_test)

    # 4. Discriminative LR
    np.random.seed(42)
    dlr_model = PretrainedMLP([n_features, 64, 32, n_classes_target])
    dlr_model.layers = [(W.copy(), b.copy()) for W, b in pretrained.layers[:-1]]
    dlr_model.layers.append((np.random.randn(n_classes_target, 32) * scale, np.zeros(n_classes_target)))
    dlr_acc = discriminative_lr(dlr_model, X_train, y_train, X_test, y_test)

    print(f"\nTarget Task Results (200 train samples):")
    print(f"{'Strategy':>25s} | {'Accuracy':>8s}")
    print("-" * 38)
    print(f"{'From scratch':>25s} | {scratch_acc:>7.3f}")
    print(f"{'Feature extraction':>25s} | {feat_acc:>7.3f}")
    print(f"{'Fine-tuning':>25s} | {ft_acc:>7.3f}")
    print(f"{'Discriminative LR':>25s} | {dlr_acc:>7.3f}")


if __name__ == "__main__":
    demo()

"""
Neural Network Interpretability: From Scratch
================================================

Techniques for understanding what neural networks learn and why
they make specific predictions.

Methods:
    1. Gradient-based saliency maps
    2. Integrated Gradients
    3. Class Activation Maps (CAM)
    4. Feature visualization (activation maximization)
    5. SHAP-like attribution (simplified)
    6. Probing classifiers

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
# PART 1: SIMPLE NETWORK FOR INTERPRETABILITY
# =============================================================================

class InterpretableMLP:
    """MLP with gradient tracking for interpretability methods."""

    def __init__(self, dims: list):
        self.layers = []
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            W = np.random.randn(dims[i + 1], dims[i]) * scale
            b = np.zeros(dims[i + 1])
            self.layers.append((W, b))

    def forward(self, x: np.ndarray):
        self.activations = [x.copy()]
        self.pre_activations = []
        h = x
        for i, (W, b) in enumerate(self.layers):
            z = h @ W.T + b
            self.pre_activations.append(z)
            if i < len(self.layers) - 1:
                h = np.maximum(0, z)
            else:
                exp_z = np.exp(z - z.max(axis=-1, keepdims=True))
                h = exp_z / exp_z.sum(axis=-1, keepdims=True)
            self.activations.append(h)
        return h

    def backward_to_input(self, target_class: int):
        """Compute gradient of target class output w.r.t. input."""
        # Gradient at output
        grad = np.zeros_like(self.activations[-1])
        grad[:, target_class] = 1.0

        for i in range(len(self.layers) - 1, -1, -1):
            W, b = self.layers[i]
            if i < len(self.layers) - 1:
                grad *= (self.pre_activations[i] > 0).astype(float)
            grad = grad @ W  # (batch, in_dim)

        return grad


# =============================================================================
# PART 2: SALIENCY MAPS
# =============================================================================

def vanilla_gradient_saliency(model: InterpretableMLP, x: np.ndarray,
                               target_class: int) -> np.ndarray:
    """
    Vanilla Gradient Saliency (Simonyan et al., 2014).

    Simply compute |∂y_c / ∂x| — how much does each input feature
    affect the target class output?

    Simple but can be noisy.
    """
    model.forward(x)
    grad = model.backward_to_input(target_class)
    return np.abs(grad)


def smooth_gradient_saliency(model: InterpretableMLP, x: np.ndarray,
                              target_class: int, n_samples: int = 50,
                              noise_std: float = 0.1) -> np.ndarray:
    """
    SmoothGrad (Smilkov et al., 2017).

    Average gradients over noisy versions of the input.
    This reduces noise in saliency maps by smoothing.
    """
    saliency = np.zeros_like(x)
    for _ in range(n_samples):
        noisy_x = x + np.random.randn(*x.shape) * noise_std
        model.forward(noisy_x)
        grad = model.backward_to_input(target_class)
        saliency += np.abs(grad)
    return saliency / n_samples


# =============================================================================
# PART 3: INTEGRATED GRADIENTS
# =============================================================================

def integrated_gradients(model: InterpretableMLP, x: np.ndarray,
                         target_class: int, baseline: np.ndarray = None,
                         n_steps: int = 50) -> np.ndarray:
    """
    Integrated Gradients (Sundararajan et al., 2017).

    Attribute importance by integrating gradients along the path
    from a baseline (e.g., zero input) to the actual input:

        IG(x) = (x - x') × ∫₀¹ ∂F(x' + α(x - x'))/∂x dα

    Properties:
        - Completeness: attributions sum to output difference
        - Sensitivity: if feature matters, it gets attribution
        - Implementation invariance: same for functionally equivalent models
    """
    if baseline is None:
        baseline = np.zeros_like(x)

    # Generate interpolated inputs
    alphas = np.linspace(0, 1, n_steps + 1)
    gradients = np.zeros_like(x)

    for alpha in alphas:
        interpolated = baseline + alpha * (x - baseline)
        model.forward(interpolated)
        grad = model.backward_to_input(target_class)
        gradients += grad

    # Average and scale
    avg_gradients = gradients / (n_steps + 1)
    integrated = (x - baseline) * avg_gradients
    return integrated


# =============================================================================
# PART 4: OCCLUSION SENSITIVITY
# =============================================================================

def occlusion_sensitivity(model: InterpretableMLP, x: np.ndarray,
                           target_class: int, patch_size: int = 1) -> np.ndarray:
    """
    Occlusion Sensitivity (Zeiler & Fergus, 2014).

    Slide a "patch" (set features to zero) across the input and measure
    how much the target class probability drops.

    Large drop = that region is important.
    """
    base_prob = model.forward(x)[0, target_class]
    importance = np.zeros(x.shape[1])

    for i in range(0, x.shape[1], patch_size):
        occluded = x.copy()
        end = min(i + patch_size, x.shape[1])
        occluded[:, i:end] = 0.0
        new_prob = model.forward(occluded)[0, target_class]
        importance[i:end] = base_prob - new_prob

    return importance


# =============================================================================
# PART 5: SIMPLE SHAPLEY-LIKE ATTRIBUTION
# =============================================================================

def permutation_importance(model: InterpretableMLP, x: np.ndarray,
                            target_class: int, n_permutations: int = 100,
                            X_background: np.ndarray = None) -> np.ndarray:
    """
    Permutation-based feature importance (approximation of Shapley values).

    For each feature, randomly replace its value with values from
    background data and measure how much the prediction changes.
    """
    if X_background is None:
        X_background = np.random.randn(100, x.shape[1])

    base_prob = model.forward(x)[0, target_class]
    importance = np.zeros(x.shape[1])

    for feat_idx in range(x.shape[1]):
        prob_drops = []
        for _ in range(n_permutations):
            perturbed = x.copy()
            rand_idx = np.random.randint(len(X_background))
            perturbed[:, feat_idx] = X_background[rand_idx, feat_idx]
            new_prob = model.forward(perturbed)[0, target_class]
            prob_drops.append(base_prob - new_prob)
        importance[feat_idx] = np.mean(prob_drops)

    return importance


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate interpretability methods."""
    print("=" * 70)
    print("NEURAL NETWORK INTERPRETABILITY")
    print("=" * 70)

    # Create a model where we KNOW feature 0-4 are important
    n_features = 20
    n_classes = 3
    np.random.seed(42)

    # Train model where first 5 features determine the class
    model = InterpretableMLP([n_features, 32, 16, n_classes])
    X = np.random.randn(500, n_features)
    W_true = np.zeros((n_features, n_classes))
    W_true[:5] = np.random.randn(5, n_classes) * 2  # Only first 5 matter
    y_idx = np.argmax(X @ W_true, axis=1)
    y = np.eye(n_classes)[y_idx]

    # Train
    for epoch in range(200):
        probs = model.forward(X)
        grad = (probs - y) / len(X)
        for i in range(len(model.layers) - 1, -1, -1):
            W, b = model.layers[i]
            W -= 0.01 * grad.T @ model.activations[i]
            b -= 0.01 * grad.sum(axis=0)
            model.layers[i] = (W, b)
            if i > 0:
                grad = grad @ W
                grad *= (model.pre_activations[i-1] > 0).astype(float)

    # Interpret a single prediction
    x_test = np.random.randn(1, n_features)
    pred = model.forward(x_test)
    target = pred.argmax()
    print(f"\nPrediction: class {target} (prob={pred[0, target]:.3f})")
    print(f"True important features: 0, 1, 2, 3, 4")

    # Run all methods
    saliency = vanilla_gradient_saliency(model, x_test, target)[0]
    smooth = smooth_gradient_saliency(model, x_test, target)[0]
    ig = integrated_gradients(model, x_test, target)[0]
    occlusion = occlusion_sensitivity(model, x_test, target)
    perm = permutation_importance(model, x_test, target, X_background=X)

    print(f"\nTop-5 features by each method:")
    for name, scores in [("Vanilla Grad", saliency), ("SmoothGrad", smooth),
                          ("Integrated Grad", np.abs(ig)),
                          ("Occlusion", occlusion), ("Permutation", perm)]:
        top5 = np.argsort(scores)[-5:][::-1]
        print(f"  {name:>15s}: {top5.tolist()}")

    # Plot
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    methods = [("Vanilla", saliency), ("SmoothGrad", smooth),
               ("IntGrad", np.abs(ig)), ("Occlusion", occlusion),
               ("Permutation", perm)]

    for ax, (name, scores) in zip(axes, methods):
        colors = ['red' if i < 5 else 'blue' for i in range(n_features)]
        ax.bar(range(n_features), scores, color=colors, alpha=0.7)
        ax.set_title(name)
        ax.set_xlabel("Feature")

    plt.tight_layout()
    plt.savefig(SAVE_DIR / "interpretability.png", dpi=100)
    plt.close()
    print(f"\nPlot saved (red = truly important features)")


if __name__ == "__main__":
    demo()

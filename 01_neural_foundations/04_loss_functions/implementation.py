"""
Loss Functions from Scratch
===========================

Every loss function is a negative log-likelihood under some probability distribution.
This module implements each loss, derives its gradient, and demonstrates why
cross-entropy is superior to MSE for classification.
"""

import numpy as np


# ==============================================================================
# Part 1: Loss Functions and Their Gradients
# ==============================================================================

def mse_loss(y_true, y_pred):
    """
    Mean Squared Error: L = (1/n) Σ (y - ŷ)²

    Assumes Gaussian noise: y = f(x) + N(0, σ²)
    Minimizing MSE = Maximum Likelihood under Gaussian assumption.
    """
    return np.mean((y_true - y_pred) ** 2)


def mse_gradient(y_true, y_pred):
    """dL/dŷ = -2(y - ŷ) / n = 2(ŷ - y) / n"""
    n = len(y_true)
    return 2 * (y_pred - y_true) / n


def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    BCE: L = -(1/n) Σ [y log(ŷ) + (1-y) log(1-ŷ)]

    Assumes Bernoulli distribution: y ~ Bernoulli(ŷ)
    Minimizing BCE = Maximum Likelihood under Bernoulli assumption.
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def bce_gradient(y_true, y_pred, eps=1e-15):
    """dL/dŷ = -(y/ŷ - (1-y)/(1-ŷ)) / n"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    n = len(y_true)
    return (-(y_true / y_pred) + (1 - y_true) / (1 - y_pred)) / n


def categorical_cross_entropy(y_true_onehot, y_pred_probs, eps=1e-15):
    """
    CCE: L = -(1/n) Σᵢ Σⱼ yij log(ŷij)

    Assumes Categorical distribution. y_true is one-hot, y_pred is softmax output.
    """
    y_pred_probs = np.clip(y_pred_probs, eps, 1.0)
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred_probs), axis=1))


def softmax(z):
    """Numerically stable softmax: shift by max to prevent overflow."""
    e = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def softmax_cce_gradient(y_true_onehot, logits):
    """
    Combined softmax + CCE gradient: dL/dz = softmax(z) - y

    This beautiful simplification is why softmax and cross-entropy are always paired.
    """
    probs = softmax(logits)
    return (probs - y_true_onehot) / len(y_true_onehot)


def hinge_loss(y_true, y_pred):
    """
    Hinge: L = (1/n) Σ max(0, 1 - y·ŷ)

    y ∈ {-1, +1}. From SVM — creates max-margin classifier.
    """
    return np.mean(np.maximum(0, 1 - y_true * y_pred))


def hinge_gradient(y_true, y_pred):
    """dL/dŷ = -y if y·ŷ < 1, else 0"""
    n = len(y_true)
    grad = np.where(y_true * y_pred < 1, -y_true, 0.0)
    return grad / n


def huber_loss(y_true, y_pred, delta=1.0):
    """
    Huber: MSE for small errors, MAE for large errors.
    Robust to outliers while smooth near zero.
    """
    error = np.abs(y_true - y_pred)
    quadratic = np.minimum(error, delta)
    linear = error - quadratic
    return np.mean(0.5 * quadratic ** 2 + delta * linear)


def huber_gradient(y_true, y_pred, delta=1.0):
    """Gradient transitions from (ŷ-y) to ±δ at the threshold."""
    error = y_pred - y_true
    n = len(y_true)
    grad = np.where(np.abs(error) <= delta, error, delta * np.sign(error))
    return grad / n


# ==============================================================================
# Part 2: Why Cross-Entropy Beats MSE for Classification
# ==============================================================================

def sigmoid(z):
    """Sigmoid activation: maps ℝ -> (0, 1)."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def demonstrate_mse_vs_bce():
    """
    Show why BCE is better than MSE for classification with sigmoid output.

    The key issue: sigmoid saturates for large |z|, making MSE gradients tiny.
    BCE cancels the saturation, keeping gradients useful.
    """
    print("=" * 60)
    print("MSE vs CROSS-ENTROPY FOR CLASSIFICATION")
    print("=" * 60)

    # Scenario: model outputs logit z, we apply sigmoid to get p, target is 1
    # Compare dL/dz for MSE vs BCE

    print("\nTarget y = 1. Varying model logit z (before sigmoid):")
    print(f"{'z':>8} {'sigmoid(z)':>12} {'MSE grad':>12} {'BCE grad':>12}")
    print("-" * 48)

    for z_val in [-5, -3, -1, 0, 1, 3, 5]:
        z = np.array([float(z_val)])
        p = sigmoid(z)
        y = np.array([1.0])

        # MSE gradient through sigmoid: dL/dz = dL/dp * dp/dz
        # dL/dp = 2(p - y), dp/dz = p(1-p)
        mse_grad = 2 * (p - y) * p * (1 - p)

        # BCE gradient through sigmoid: dL/dz = p - y
        # (The sigmoid derivative cancels with the 1/p from log)
        bce_grad = p - y

        print(f"{z_val:8.1f} {p[0]:12.6f} {mse_grad[0]:12.6f} {bce_grad[0]:12.6f}")

    print("\nKey observation:")
    print("  When z = -5, the model is VERY wrong (p ≈ 0, target = 1)")
    print("  MSE gradient ≈ 0 (sigmoid is saturated -> tiny gradient)")
    print("  BCE gradient ≈ -1 (strong signal to fix the mistake)")
    print("\n  This is why BCE converges faster for classification.")


# ==============================================================================
# Part 3: Training Comparison
# ==============================================================================

def train_comparison():
    """
    Train a simple 1-layer network on a binary classification task
    using MSE vs BCE. Show BCE converges faster.
    """
    print("\n" + "=" * 60)
    print("TRAINING COMPARISON: MSE vs BCE")
    print("=" * 60)

    np.random.seed(42)

    # Generate linearly separable 2D data
    n = 100
    X_pos = np.random.randn(n // 2, 2) + np.array([1.5, 1.5])
    X_neg = np.random.randn(n // 2, 2) + np.array([-1.5, -1.5])
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(n // 2), np.zeros(n // 2)])

    results = {}

    for loss_name in ['MSE', 'BCE']:
        # Initialize same weights for fair comparison
        np.random.seed(0)
        w = np.random.randn(2) * 0.01
        b = 0.0
        lr = 1.0
        losses = []

        for epoch in range(200):
            # Forward: logit -> sigmoid -> loss
            z = X @ w + b
            p = sigmoid(z)

            if loss_name == 'MSE':
                loss = np.mean((y - p) ** 2)
                # dL/dz = dL/dp * dp/dz = 2(p-y) * p*(1-p)
                dz = 2 * (p - y) * p * (1 - p)
            else:
                p_clip = np.clip(p, 1e-15, 1 - 1e-15)
                loss = -np.mean(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip))
                # dL/dz = p - y (the beautiful simplification)
                dz = p - y

            losses.append(loss)

            # Backward: dL/dw = X^T · dL/dz, dL/db = sum(dL/dz)
            dw = X.T @ dz / n
            db = np.mean(dz)

            w -= lr * dw
            b -= lr * db

        # Accuracy
        preds = sigmoid(X @ w + b) > 0.5
        acc = np.mean(preds == y)
        results[loss_name] = {'losses': losses, 'accuracy': acc}

        print(f"\n{loss_name}:")
        print(f"  Final loss:     {losses[-1]:.6f}")
        print(f"  Final accuracy: {acc * 100:.1f}%")
        print(f"  Loss at epoch 10:  {losses[9]:.6f}")
        print(f"  Loss at epoch 50:  {losses[49]:.6f}")

    print("\nConclusion: BCE converges faster because its gradient doesn't")
    print("vanish when the sigmoid saturates.")


# ==============================================================================
# Part 4: The Probabilistic View
# ==============================================================================

def probabilistic_interpretation():
    """
    Show that training with cross-entropy IS doing maximum likelihood estimation.

    We fit a logistic regression model and show the learned probabilities
    match the empirical frequencies.
    """
    print("\n" + "=" * 60)
    print("LOSS = NEGATIVE LOG-LIKELIHOOD")
    print("Training with BCE = Maximum Likelihood Estimation")
    print("=" * 60)

    np.random.seed(42)

    # Generate data from a known logistic model
    # True model: P(y=1|x) = sigmoid(2x - 1)
    n = 1000
    x = np.random.randn(n)
    true_probs = sigmoid(2 * x - 1)
    y = (np.random.rand(n) < true_probs).astype(float)

    # Fit with MLE (= minimize BCE)
    w, b = 0.0, 0.0
    lr = 0.1

    for _ in range(500):
        z = w * x + b
        p = sigmoid(z)
        dz = p - y
        w -= lr * np.mean(dz * x)
        b -= lr * np.mean(dz)

    print(f"\n  True parameters:    w = 2.000, b = -1.000")
    print(f"  Learned parameters: w = {w:.3f}, b = {b:.3f}")
    print(f"\n  This confirms: minimizing cross-entropy recovers the true")
    print(f"  data-generating parameters. Training IS statistical inference.")

    # Show calibration: predicted probabilities match actual frequencies
    print("\n  Calibration check (predicted probability vs actual frequency):")
    print(f"  {'Predicted':>12} {'Actual':>12} {'Count':>8}")
    print("  " + "-" * 36)

    p_final = sigmoid(w * x + b)
    for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        mask = (p_final >= lo) & (p_final < hi)
        if np.sum(mask) > 0:
            avg_pred = np.mean(p_final[mask])
            avg_actual = np.mean(y[mask])
            count = np.sum(mask)
            print(f"  {avg_pred:12.3f} {avg_actual:12.3f} {count:8d}")

    print("\n  Well-calibrated: predicted probabilities ≈ actual frequencies.")


# ==============================================================================
# Part 5: Gradient Magnitude Analysis
# ==============================================================================

def gradient_analysis():
    """
    Compare gradient magnitudes for different losses.
    Shows how each loss behaves as the prediction varies.
    """
    print("\n" + "=" * 60)
    print("GRADIENT MAGNITUDE ANALYSIS")
    print("=" * 60)

    y_true = np.array([1.0])
    preds = np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])

    print(f"\nTarget = 1.0. Gradient dL/dŷ for different predictions:")
    print(f"{'ŷ':>8} {'MSE':>12} {'BCE':>12} {'Huber':>12}")
    print("-" * 48)

    for p in preds:
        y_p = np.array([p])
        mse_g = mse_gradient(y_true, y_p)[0]
        bce_g = bce_gradient(y_true, y_p)[0]
        hub_g = huber_gradient(y_true, y_p)[0]

        print(f"{p:8.2f} {mse_g:12.6f} {bce_g:12.6f} {hub_g:12.6f}")

    print("\nObservations:")
    print("  MSE: gradient proportional to error (symmetric)")
    print("  BCE: very large gradient when ŷ→0 (strong correction when very wrong)")
    print("  Huber: capped gradient for large errors (robust to outliers)")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("LOSS FUNCTIONS FROM SCRATCH")
    print("Every loss is a negative log-likelihood.\n")

    # 1. Core demonstration: why BCE beats MSE for classification
    demonstrate_mse_vs_bce()

    # 2. Training comparison
    train_comparison()

    # 3. Prove: training with BCE = doing MLE
    probabilistic_interpretation()

    # 4. Gradient analysis
    gradient_analysis()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. MSE assumes Gaussian noise -> good for regression
2. Cross-entropy assumes Bernoulli/Categorical -> good for classification
3. BCE + sigmoid has gradient (p - y): no saturation, fast learning
4. Training with cross-entropy IS maximum likelihood estimation
5. Loss choice = probabilistic assumption about your data
6. Hinge loss creates max-margin classifiers (SVM connection)
7. Huber loss is robust to outliers (capped gradient)

Next: Module 02 builds a complete MLP using these components.
""")

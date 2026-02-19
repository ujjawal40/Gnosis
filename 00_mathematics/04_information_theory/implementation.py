"""
Information Theory: From-Scratch Implementation
================================================

Everything here uses NumPy only. We build every concept from the ground up:
  1. Entropy computation
  2. Cross-entropy computation
  3. KL Divergence (with asymmetry demonstration)
  4. Mutual information estimation
  5. Cross-entropy as a loss function (equivalence to NLL, comparison with MSE)
  6. Information Bottleneck toy example
  7. Huffman coding from scratch (with compression ratio analysis)

Run this file directly: python implementation.py
"""

import numpy as np
from collections import Counter


# =============================================================================
# 1. ENTROPY
# =============================================================================

def entropy(p, base=2):
    """
    Compute Shannon entropy H(X) = -sum p(x) log p(x).

    Args:
        p: Probability distribution (array-like, must sum to 1).
        base: Logarithm base. 2 for bits, e for nats, 10 for hartleys.

    Returns:
        Entropy value.
    """
    p = np.asarray(p, dtype=np.float64)
    assert np.allclose(p.sum(), 1.0), f"Distribution must sum to 1, got {p.sum()}"
    assert np.all(p >= 0), "Probabilities must be non-negative"

    # Convention: 0 * log(0) = 0 (by continuity, since lim x->0+ x*log(x) = 0)
    mask = p > 0
    h = -np.sum(p[mask] * np.log(p[mask]) / np.log(base))
    return h


def joint_entropy(p_xy, base=2):
    """
    Compute joint entropy H(X, Y) = -sum_{x,y} p(x,y) log p(x,y).

    Args:
        p_xy: Joint probability table (2D array).

    Returns:
        Joint entropy value.
    """
    p_xy = np.asarray(p_xy, dtype=np.float64)
    assert np.allclose(p_xy.sum(), 1.0), "Joint distribution must sum to 1"

    mask = p_xy > 0
    h = -np.sum(p_xy[mask] * np.log(p_xy[mask]) / np.log(base))
    return h


def conditional_entropy(p_xy, base=2):
    """
    Compute conditional entropy H(X|Y) = H(X,Y) - H(Y).

    Here X indexes rows and Y indexes columns of the joint distribution.
    """
    p_xy = np.asarray(p_xy, dtype=np.float64)
    p_y = p_xy.sum(axis=0)  # marginal of Y
    return joint_entropy(p_xy, base) - entropy(p_y, base)


def demo_entropy():
    """Demonstrate entropy computation with various distributions."""
    print("=" * 70)
    print("1. ENTROPY")
    print("=" * 70)

    # Fair coin
    p_fair = np.array([0.5, 0.5])
    print(f"\nFair coin:     p = {p_fair}")
    print(f"  H = {entropy(p_fair):.4f} bits  (maximum for 2 outcomes)")

    # Biased coin
    p_biased = np.array([0.9, 0.1])
    print(f"\nBiased coin:   p = {p_biased}")
    print(f"  H = {entropy(p_biased):.4f} bits  (more predictable -> lower entropy)")

    # Deterministic
    p_det = np.array([1.0, 0.0])
    print(f"\nDeterministic: p = {p_det}")
    print(f"  H = {entropy(p_det):.4f} bits  (no uncertainty at all)")

    # Fair die
    p_die = np.ones(6) / 6
    print(f"\nFair die:      p = {np.round(p_die, 4)}")
    print(f"  H = {entropy(p_die):.4f} bits  (= log2(6))")

    # Loaded die
    p_loaded = np.array([0.4, 0.2, 0.15, 0.1, 0.1, 0.05])
    print(f"\nLoaded die:    p = {p_loaded}")
    print(f"  H = {entropy(p_loaded):.4f} bits  (less than fair die)")

    # Entropy vs number of equally-likely outcomes
    print("\n  Entropy of uniform distribution over n outcomes:")
    for n in [2, 4, 8, 16, 64, 256]:
        p_uniform = np.ones(n) / n
        print(f"    n = {n:>3d}: H = {entropy(p_uniform):.4f} bits (= log2({n}))")

    # Entropy as a function of binary probability
    print("\n  Binary entropy function H(p) for p in [0, 1]:")
    for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        p_bin = np.array([p, 1 - p]) if p > 0 and p < 1 else (
            np.array([1.0, 0.0]) if p == 1.0 else np.array([0.0, 1.0])
        )
        # Handle edge cases for display
        if p == 0.0:
            p_bin = np.array([1e-15, 1.0 - 1e-15])
            h = 0.0
        elif p == 1.0:
            p_bin = np.array([1.0 - 1e-15, 1e-15])
            h = 0.0
        else:
            p_bin = np.array([p, 1 - p])
            h = entropy(p_bin)
        print(f"    p = {p:.1f}: H = {h:.4f} bits")

    print()


# =============================================================================
# 2. CROSS-ENTROPY
# =============================================================================

def cross_entropy(p, q, base=2):
    """
    Compute cross-entropy H(p, q) = -sum p(x) log q(x).

    Measures the average bits needed to encode samples from p using code
    optimized for q.

    Args:
        p: True distribution.
        q: Model distribution.
        base: Logarithm base.

    Returns:
        Cross-entropy value.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    assert np.allclose(p.sum(), 1.0), "p must sum to 1"
    assert np.allclose(q.sum(), 1.0), "q must sum to 1"

    # Where p > 0, q must also be > 0 (otherwise cross-entropy is infinite)
    mask = p > 0
    if np.any(q[mask] <= 0):
        return np.inf

    h = -np.sum(p[mask] * np.log(q[mask]) / np.log(base))
    return h


def demo_cross_entropy():
    """Demonstrate cross-entropy and its relationship to entropy and KL divergence."""
    print("=" * 70)
    print("2. CROSS-ENTROPY")
    print("=" * 70)

    p = np.array([0.25, 0.25, 0.25, 0.25])  # True: uniform

    # Perfect model
    q_perfect = np.array([0.25, 0.25, 0.25, 0.25])
    print(f"\nTrue distribution p = {p}")
    print(f"Perfect model    q = {q_perfect}")
    print(f"  H(p)    = {entropy(p):.4f} bits  (entropy of truth)")
    print(f"  H(p, q) = {cross_entropy(p, q_perfect):.4f} bits  (cross-entropy)")
    print(f"  Wasted  = {cross_entropy(p, q_perfect) - entropy(p):.4f} bits  (= KL divergence)")

    # Bad model
    q_bad = np.array([0.7, 0.1, 0.1, 0.1])
    print(f"\nBad model        q = {q_bad}")
    print(f"  H(p, q) = {cross_entropy(p, q_bad):.4f} bits")
    print(f"  Wasted  = {cross_entropy(p, q_bad) - entropy(p):.4f} bits")

    # Even worse model
    q_worse = np.array([0.97, 0.01, 0.01, 0.01])
    print(f"\nTerrible model   q = {q_worse}")
    print(f"  H(p, q) = {cross_entropy(p, q_worse):.4f} bits")
    print(f"  Wasted  = {cross_entropy(p, q_worse) - entropy(p):.4f} bits")

    # Key identity: H(p, q) = H(p) + D_KL(p || q)
    print("\n  Key identity: H(p, q) = H(p) + D_KL(p || q)")
    print(f"  So cross-entropy >= entropy, with equality iff p = q")

    # One-hot example (classification)
    print("\n  Classification example (one-hot true distribution):")
    p_onehot = np.array([0, 1, 0, 0], dtype=np.float64)
    # Adjust to avoid log(0) issues in display -- use small epsilon only for entropy calc
    for confidence in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        remaining = (1 - confidence) / 3
        q_model = np.array([remaining, confidence, remaining, remaining])
        ce = cross_entropy(p_onehot, q_model)
        nll = -np.log2(confidence)
        print(f"    q(correct class) = {confidence:.2f}: "
              f"H(p,q) = {ce:.4f} bits = -log2({confidence:.2f}) = {nll:.4f} bits")

    print()


# =============================================================================
# 3. KL DIVERGENCE
# =============================================================================

def kl_divergence(p, q, base=2):
    """
    Compute KL divergence D_KL(p || q) = sum p(x) log(p(x)/q(x)).

    Measures the extra bits needed when using q instead of p.

    Args:
        p: True distribution.
        q: Approximate distribution.
        base: Logarithm base.

    Returns:
        KL divergence (always >= 0).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    mask = p > 0
    if np.any(q[mask] <= 0):
        return np.inf

    kl = np.sum(p[mask] * np.log(p[mask] / q[mask]) / np.log(base))
    return kl


def demo_kl_divergence():
    """Demonstrate KL divergence properties, especially asymmetry."""
    print("=" * 70)
    print("3. KL DIVERGENCE")
    print("=" * 70)

    # Basic properties
    p = np.array([0.3, 0.5, 0.2])
    q = np.array([0.1, 0.6, 0.3])

    print(f"\np = {p}")
    print(f"q = {q}")
    print(f"\n  D_KL(p || q) = {kl_divergence(p, q):.6f} bits")
    print(f"  D_KL(q || p) = {kl_divergence(q, p):.6f} bits")
    print(f"  These are NOT equal! KL divergence is asymmetric.")

    # Verify: D_KL(p || q) = H(p, q) - H(p)
    ce = cross_entropy(p, q)
    h = entropy(p)
    kl = kl_divergence(p, q)
    print(f"\n  Verification: H(p,q) - H(p) = {ce:.6f} - {h:.6f} = {ce - h:.6f}")
    print(f"  D_KL(p || q)                = {kl:.6f}")
    print(f"  Match: {np.isclose(ce - h, kl)}")

    # Non-negativity
    print(f"\n  Non-negativity:")
    print(f"  D_KL(p || q) = {kl:.6f} >= 0: {kl >= -1e-10}")
    print(f"  D_KL(p || p) = {kl_divergence(p, p):.6f} (zero for identical distributions)")

    # Asymmetry demonstration: forward vs reverse KL
    print("\n" + "-" * 50)
    print("  ASYMMETRY IN DEPTH: Forward vs Reverse KL")
    print("-" * 50)

    # True distribution: bimodal
    p_bimodal = np.array([0.0, 0.4, 0.1, 0.0, 0.0, 0.1, 0.4, 0.0])
    p_bimodal = p_bimodal / p_bimodal.sum()

    # Forward KL minimizer: tries to cover both modes (mean-seeking)
    q_forward = np.array([0.05, 0.2, 0.15, 0.1, 0.1, 0.15, 0.2, 0.05])

    # Reverse KL minimizer: locks onto one mode (mode-seeking)
    q_reverse = np.array([0.0, 0.5, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0])
    q_reverse = q_reverse / q_reverse.sum()

    print(f"\n  True (bimodal):  p = {np.round(p_bimodal, 3)}")
    print(f"  Mean-seeking:    q = {np.round(q_forward, 3)}")
    print(f"  Mode-seeking:    q = {np.round(q_reverse, 3)}")

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p_safe = np.clip(p_bimodal, eps, 1.0)
    p_safe /= p_safe.sum()
    q_fwd_safe = np.clip(q_forward, eps, 1.0)
    q_fwd_safe /= q_fwd_safe.sum()
    q_rev_safe = np.clip(q_reverse, eps, 1.0)
    q_rev_safe /= q_rev_safe.sum()

    print(f"\n  Forward KL D_KL(p || q):")
    print(f"    Mean-seeking: {kl_divergence(p_safe, q_fwd_safe):.4f} bits  (lower = better)")
    print(f"    Mode-seeking: {kl_divergence(p_safe, q_rev_safe):.4f} bits  (penalized for missing a mode)")

    print(f"\n  Reverse KL D_KL(q || p):")
    print(f"    Mean-seeking: {kl_divergence(q_fwd_safe, p_safe):.4f} bits")
    print(f"    Mode-seeking: {kl_divergence(q_rev_safe, p_safe):.4f} bits  (lower: not penalized for ignoring a mode)")

    print(f"\n  Forward KL penalizes missing modes (mean-seeking wins).")
    print(f"  Reverse KL penalizes placing mass where p is low (mode-seeking wins).")
    print(f"  This is why VAEs (reverse KL) produce blurry outputs -- they mode-seek.")

    print()


# =============================================================================
# 4. MUTUAL INFORMATION
# =============================================================================

def mutual_information(p_xy, base=2):
    """
    Compute mutual information I(X; Y) = H(X) + H(Y) - H(X, Y).

    Also equals D_KL(p(x,y) || p(x)p(y)).

    Args:
        p_xy: Joint probability table (2D array). Rows = X, Cols = Y.

    Returns:
        Mutual information value.
    """
    p_xy = np.asarray(p_xy, dtype=np.float64)
    p_x = p_xy.sum(axis=1)  # marginal of X
    p_y = p_xy.sum(axis=0)  # marginal of Y

    h_x = entropy(p_x, base)
    h_y = entropy(p_y, base)
    h_xy = joint_entropy(p_xy, base)

    mi = h_x + h_y - h_xy
    return mi


def mutual_information_from_data(x, y, bins=20):
    """
    Estimate mutual information from data samples using binning.

    This is a simple histogram-based estimator. For real applications, use
    k-nearest-neighbor estimators (KSG) or neural estimators (MINE).

    Args:
        x: Samples of variable X (1D array).
        y: Samples of variable Y (1D array).
        bins: Number of bins for histogram.

    Returns:
        Estimated mutual information in bits.
    """
    # Create 2D histogram (joint distribution)
    hist_xy, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    p_xy = hist_xy / hist_xy.sum()

    # Marginals
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    # Compute I(X;Y) = sum p(x,y) log(p(x,y) / (p(x)p(y)))
    mi = 0.0
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))

    return mi


def demo_mutual_information():
    """Demonstrate mutual information computation and estimation."""
    print("=" * 70)
    print("4. MUTUAL INFORMATION")
    print("=" * 70)

    # Example 1: Perfect dependence
    # X and Y always agree: p(0,0) = p(1,1) = 0.5
    p_xy_perfect = np.array([[0.5, 0.0],
                              [0.0, 0.5]])
    print(f"\nPerfect dependence (X = Y):")
    print(f"  Joint distribution:\n{p_xy_perfect}")
    mi = mutual_information(p_xy_perfect)
    print(f"  I(X;Y) = {mi:.4f} bits")
    print(f"  H(X) = {entropy(p_xy_perfect.sum(axis=1)):.4f}, "
          f"H(Y) = {entropy(p_xy_perfect.sum(axis=0)):.4f}")
    print(f"  I(X;Y) = H(X) = H(Y) because knowing one determines the other.")

    # Example 2: Independence
    p_xy_indep = np.array([[0.25, 0.25],
                            [0.25, 0.25]])
    print(f"\nIndependence:")
    print(f"  Joint distribution:\n{p_xy_indep}")
    mi = mutual_information(p_xy_indep)
    print(f"  I(X;Y) = {mi:.4f} bits  (zero = independent)")

    # Example 3: Partial dependence
    p_xy_partial = np.array([[0.4, 0.1],
                              [0.1, 0.4]])
    print(f"\nPartial dependence:")
    print(f"  Joint distribution:\n{p_xy_partial}")
    mi = mutual_information(p_xy_partial)
    print(f"  I(X;Y) = {mi:.4f} bits")
    p_x = p_xy_partial.sum(axis=1)
    print(f"  H(X) = {entropy(p_x):.4f} bits")
    print(f"  So X and Y share {mi / entropy(p_x) * 100:.1f}% of X's information.")

    # Example 4: Estimating MI from continuous data
    print("\n  Estimating MI from continuous data samples:")
    rng = np.random.default_rng(42)
    n_samples = 10000

    # Independent Gaussians
    x_indep = rng.normal(0, 1, n_samples)
    y_indep = rng.normal(0, 1, n_samples)
    mi_indep = mutual_information_from_data(x_indep, y_indep, bins=30)
    print(f"    Independent Gaussians: I(X;Y) ~= {mi_indep:.4f} bits (should be ~0)")

    # Correlated Gaussians (rho = 0.8)
    rho = 0.8
    x_corr = rng.normal(0, 1, n_samples)
    y_corr = rho * x_corr + np.sqrt(1 - rho ** 2) * rng.normal(0, 1, n_samples)
    mi_corr = mutual_information_from_data(x_corr, y_corr, bins=30)
    # Analytical MI for bivariate Gaussian: I = -0.5 * log2(1 - rho^2)
    mi_analytical = -0.5 * np.log2(1 - rho ** 2)
    print(f"    Correlated Gaussians (rho={rho}): I(X;Y) ~= {mi_corr:.4f} bits "
          f"(analytical: {mi_analytical:.4f})")

    # Deterministic relationship: Y = X^2
    x_det = rng.uniform(-2, 2, n_samples)
    y_det = x_det ** 2
    mi_det = mutual_information_from_data(x_det, y_det, bins=30)
    print(f"    Y = X^2 (deterministic): I(X;Y) ~= {mi_det:.4f} bits (high, nonlinear dep)")

    # Show symmetry
    print(f"\n  Symmetry check:")
    print(f"    I(X;Y) = {mutual_information_from_data(x_corr, y_corr, bins=30):.4f}")
    print(f"    I(Y;X) = {mutual_information_from_data(y_corr, x_corr, bins=30):.4f}")
    print(f"    Unlike KL divergence, mutual information IS symmetric.")

    print()


# =============================================================================
# 5. CROSS-ENTROPY AS A LOSS FUNCTION
# =============================================================================

def softmax(logits):
    """Numerically stable softmax."""
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy_loss(logits, targets):
    """
    Compute cross-entropy loss for classification.

    This IS the cross-entropy from information theory, applied to:
      p = one-hot true distribution
      q = softmax model distribution

    For one-hot p: H(p, q) = -log q(correct class) = NLL

    Args:
        logits: Raw model outputs, shape (N, C).
        targets: True class indices, shape (N,).

    Returns:
        Average cross-entropy loss (in nats, since we use ln).
    """
    probs = softmax(logits)
    n = len(targets)
    # -log p(correct class) for each sample
    log_probs = -np.log(probs[np.arange(n), targets] + 1e-15)
    return log_probs.mean()


def mse_loss_for_classification(logits, targets, n_classes):
    """MSE loss between softmax outputs and one-hot targets."""
    probs = softmax(logits)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(targets)), targets] = 1.0
    return ((probs - one_hot) ** 2).mean()


def demo_cross_entropy_loss():
    """Show cross-entropy loss IS information theory, and why it beats MSE."""
    print("=" * 70)
    print("5. CROSS-ENTROPY AS A LOSS FUNCTION")
    print("=" * 70)

    # Part A: CE loss = Negative log-likelihood
    print("\n--- Part A: Cross-entropy loss = Negative log-likelihood ---")

    logits = np.array([[2.0, 1.0, 0.1],
                        [0.5, 2.5, 0.3],
                        [0.1, 0.3, 3.0]])
    targets = np.array([0, 1, 2])  # correct classes

    probs = softmax(logits)
    print(f"\nLogits:\n{logits}")
    print(f"Softmax probabilities:\n{np.round(probs, 4)}")
    print(f"True classes: {targets}")

    # Method 1: Cross-entropy loss (standard)
    ce = cross_entropy_loss(logits, targets)
    print(f"\nCross-entropy loss: {ce:.6f}")

    # Method 2: Negative log-likelihood (manual)
    nll = -np.mean([np.log(probs[i, targets[i]] + 1e-15) for i in range(len(targets))])
    print(f"Negative log-likelihood: {nll:.6f}")
    print(f"They are identical: {np.isclose(ce, nll)}")

    # Method 3: Information-theoretic cross-entropy
    ce_info = 0
    for i in range(len(targets)):
        p = np.zeros(3)
        p[targets[i]] = 1.0
        q = probs[i]
        ce_info += cross_entropy(p, q, base=np.e)  # use nats for comparison
    ce_info /= len(targets)
    print(f"Info-theoretic H(p,q): {ce_info:.6f}")
    print(f"All three are the same quantity!")

    # Part B: Why CE works better than MSE for classification
    print("\n--- Part B: Why cross-entropy beats MSE for classification ---")
    print("\nGradient analysis when model is VERY WRONG:")

    # Scenario: true class is 0, but model is very confident it's class 1
    logits_wrong = np.array([[0.1, 5.0, 0.1]])
    target_wrong = np.array([0])
    probs_wrong = softmax(logits_wrong)

    print(f"\n  True class: 0")
    print(f"  Model probs: {np.round(probs_wrong[0], 6)}")
    print(f"  p(correct): {probs_wrong[0, 0]:.6f}  (model is VERY wrong)")

    # CE loss and gradient
    ce_val = cross_entropy_loss(logits_wrong, target_wrong)
    # Gradient of CE w.r.t. logits: (softmax - one_hot)
    one_hot = np.array([[1.0, 0.0, 0.0]])
    ce_grad = probs_wrong - one_hot
    print(f"\n  Cross-entropy loss: {ce_val:.4f}")
    print(f"  CE gradient w.r.t. logits: {np.round(ce_grad[0], 6)}")
    print(f"  CE gradient magnitude: {np.linalg.norm(ce_grad):.6f}")

    # MSE loss and gradient
    mse_val = ((probs_wrong - one_hot) ** 2).mean()
    # Gradient of MSE through softmax is more complex and saturates
    mse_grad = 2 * (probs_wrong - one_hot) / 3  # simplified
    # The actual gradient through softmax involves the Jacobian
    # For softmax output s, ds/dz = diag(s) - s*s^T
    s = probs_wrong[0]
    jacobian = np.diag(s) - np.outer(s, s)
    mse_grad_actual = (2 / 3) * jacobian @ (s - one_hot[0])
    print(f"\n  MSE loss: {mse_val:.4f}")
    print(f"  MSE gradient w.r.t. logits: {np.round(mse_grad_actual, 6)}")
    print(f"  MSE gradient magnitude: {np.linalg.norm(mse_grad_actual):.6f}")

    print(f"\n  KEY INSIGHT:")
    print(f"  CE gradient magnitude: {np.linalg.norm(ce_grad):.6f}")
    print(f"  MSE gradient magnitude: {np.linalg.norm(mse_grad_actual):.6f}")
    print(f"  Ratio (CE/MSE): {np.linalg.norm(ce_grad) / np.linalg.norm(mse_grad_actual):.1f}x")
    print(f"\n  When the model is very wrong, CE gives MUCH stronger gradients.")
    print(f"  MSE gradients saturate because softmax is nearly flat at extremes.")
    print(f"  This is why CE converges faster for classification.")

    # Part C: Training comparison
    print("\n--- Part C: Mini training comparison ---")

    rng = np.random.default_rng(42)
    n_classes = 3
    n_features = 4
    n_samples = 100

    # Generate synthetic data
    X = rng.normal(0, 1, (n_samples, n_features))
    true_W = rng.normal(0, 1, (n_features, n_classes))
    true_logits = X @ true_W
    y = true_logits.argmax(axis=1)

    # Train with CE loss
    W_ce = rng.normal(0, 0.1, (n_features, n_classes))
    lr = 0.01
    ce_losses = []
    for step in range(200):
        logits = X @ W_ce
        probs = softmax(logits)
        one_hot_y = np.zeros((n_samples, n_classes))
        one_hot_y[np.arange(n_samples), y] = 1.0
        # CE gradient w.r.t. W: X^T @ (softmax - one_hot) / N
        grad = X.T @ (probs - one_hot_y) / n_samples
        W_ce -= lr * grad
        loss = cross_entropy_loss(logits, y)
        ce_losses.append(loss)

    # Train with MSE loss
    W_mse = rng.normal(0, 0.1, (n_features, n_classes))
    mse_losses = []
    for step in range(200):
        logits = X @ W_mse
        probs = softmax(logits)
        one_hot_y = np.zeros((n_samples, n_classes))
        one_hot_y[np.arange(n_samples), y] = 1.0
        # MSE gradient through softmax
        diff = probs - one_hot_y
        # For each sample, gradient = Jacobian^T @ 2*diff / (N * C)
        grad_W = np.zeros_like(W_mse)
        for i in range(n_samples):
            s = probs[i]
            jac = np.diag(s) - np.outer(s, s)
            grad_logits = (2 / n_classes) * jac @ diff[i]
            grad_W += np.outer(X[i], grad_logits)
        grad_W /= n_samples
        W_mse -= lr * grad_W
        mse_loss = ((probs - one_hot_y) ** 2).mean()
        mse_losses.append(mse_loss)

    # Compare accuracy
    ce_acc = (softmax(X @ W_ce).argmax(axis=1) == y).mean()
    mse_acc = (softmax(X @ W_mse).argmax(axis=1) == y).mean()

    print(f"\n  After 200 steps (lr={lr}):")
    print(f"    CE training  -> accuracy: {ce_acc * 100:.1f}%, final loss: {ce_losses[-1]:.4f}")
    print(f"    MSE training -> accuracy: {mse_acc * 100:.1f}%, final loss: {mse_losses[-1]:.6f}")
    print(f"\n  CE converges faster because its gradients don't saturate.")

    print()


# =============================================================================
# 6. INFORMATION BOTTLENECK TOY EXAMPLE
# =============================================================================

def demo_information_bottleneck():
    """
    Information Bottleneck: compress X while preserving info about Y.

    We solve:  min I(X;T) - beta * I(T;Y)

    Using the iterative Blahut-Arimoto-like algorithm for the IB.
    """
    print("=" * 70)
    print("6. INFORMATION BOTTLENECK")
    print("=" * 70)

    # Setup: X has 8 states, Y has 2 states (binary classification)
    # We want to compress X into T with fewer states while keeping info about Y
    rng = np.random.default_rng(42)

    n_x = 8   # input states
    n_y = 2   # target states
    n_t = 3   # compressed states (bottleneck)

    # Define p(x) -- prior over inputs
    p_x = np.array([0.15, 0.10, 0.13, 0.12, 0.10, 0.15, 0.10, 0.15])
    p_x /= p_x.sum()

    # Define p(y|x) -- how X relates to Y
    # States 0-3 mostly predict Y=0, states 4-7 mostly predict Y=1
    p_y_given_x = np.array([
        [0.9, 0.1],   # x=0 -> mostly y=0
        [0.8, 0.2],   # x=1 -> mostly y=0
        [0.85, 0.15],  # x=2 -> mostly y=0
        [0.7, 0.3],   # x=3 -> mostly y=0 (noisier)
        [0.2, 0.8],   # x=4 -> mostly y=1
        [0.1, 0.9],   # x=5 -> mostly y=1
        [0.15, 0.85],  # x=6 -> mostly y=1
        [0.3, 0.7],   # x=7 -> mostly y=1 (noisier)
    ])

    # Compute joint p(x, y)
    p_xy = p_y_given_x * p_x[:, None]
    p_y = p_xy.sum(axis=0)

    print(f"\nSetup:")
    print(f"  X has {n_x} states, Y has {n_y} states, T has {n_t} states")
    print(f"  p(x) = {np.round(p_x, 3)}")
    print(f"  p(y) = {np.round(p_y, 3)}")
    print(f"  I(X; Y) = {mutual_information(p_xy):.4f} bits")
    print(f"  H(X) = {entropy(p_x):.4f} bits")

    # IB iterative algorithm (Blahut-Arimoto style)
    # Variables: p(t|x) -- the encoder (stochastic mapping from X to T)

    print(f"\n  Running Information Bottleneck for different beta values...")
    print(f"  {'beta':>6s}  {'I(X;T)':>8s}  {'I(T;Y)':>8s}  {'Compression':>12s}  {'Prediction':>12s}")
    print(f"  {'-' * 52}")

    for beta in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
        # Initialize p(t|x) randomly
        p_t_given_x = rng.dirichlet(np.ones(n_t), size=n_x)

        # Iterate
        for iteration in range(500):
            # Compute p(t) = sum_x p(t|x) p(x)
            p_t = p_t_given_x.T @ p_x  # shape (n_t,)
            p_t = np.clip(p_t, 1e-15, None)

            # Compute p(y|t) = sum_x p(y|x) p(x|t)
            # p(x|t) = p(t|x) p(x) / p(t)
            p_x_given_t = (p_t_given_x * p_x[:, None]).T  # shape (n_t, n_x), unnorm
            p_x_given_t = p_x_given_t / p_x_given_t.sum(axis=1, keepdims=True)

            p_y_given_t = p_x_given_t @ p_y_given_x  # shape (n_t, n_y)

            # Update p(t|x) using the IB update rule
            # p(t|x) proportional to p(t) * exp(-beta * D_KL(p(y|x) || p(y|t)))
            log_p_t_given_x_new = np.zeros((n_x, n_t))
            for x in range(n_x):
                for t in range(n_t):
                    # D_KL(p(y|x) || p(y|t))
                    kl = 0.0
                    for y_idx in range(n_y):
                        if p_y_given_x[x, y_idx] > 1e-15:
                            kl += p_y_given_x[x, y_idx] * np.log(
                                p_y_given_x[x, y_idx] / max(p_y_given_t[t, y_idx], 1e-15))
                    log_p_t_given_x_new[x, t] = np.log(max(p_t[t], 1e-15)) - beta * kl

            # Normalize (softmax over t for each x)
            log_p_t_given_x_new -= log_p_t_given_x_new.max(axis=1, keepdims=True)
            p_t_given_x = np.exp(log_p_t_given_x_new)
            p_t_given_x /= p_t_given_x.sum(axis=1, keepdims=True)

        # Compute final quantities
        # p(x, t)
        p_xt = p_t_given_x * p_x[:, None]
        i_xt = mutual_information(p_xt)

        # p(t, y)
        p_t_final = p_t_given_x.T @ p_x
        p_ty = np.zeros((n_t, n_y))
        for t in range(n_t):
            for y_idx in range(n_y):
                p_ty[t, y_idx] = sum(
                    p_t_given_x[x, t] * p_x[x] * p_y_given_x[x, y_idx]
                    for x in range(n_x))
        i_ty = mutual_information(p_ty)

        i_xy = mutual_information(p_xy)
        compression = 1 - i_xt / entropy(p_x) if entropy(p_x) > 0 else 0
        prediction = i_ty / i_xy if i_xy > 0 else 0

        print(f"  {beta:6.1f}  {i_xt:8.4f}  {i_ty:8.4f}  "
              f"{compression * 100:10.1f}%  {prediction * 100:10.1f}%")

    print(f"\n  Interpretation:")
    print(f"    - Small beta: heavy compression, lose info about Y.")
    print(f"    - Large beta: preserve prediction, less compression.")
    print(f"    - The tradeoff curve traces the information bottleneck bound.")
    print(f"    - This is the fundamental tradeoff in representation learning:")
    print(f"      compress the input while keeping what matters for the task.")

    print()


# =============================================================================
# 7. HUFFMAN CODING FROM SCRATCH
# =============================================================================

class HuffmanNode:
    """Node in a Huffman tree."""

    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None


def build_huffman_tree(symbol_freq):
    """
    Build a Huffman tree from symbol frequencies.

    The algorithm (greedy, optimal prefix code):
    1. Create a leaf node for each symbol.
    2. Repeatedly merge the two lowest-frequency nodes.
    3. The root gives the optimal prefix-free binary code.

    Args:
        symbol_freq: Dict mapping symbol -> frequency/probability.

    Returns:
        Root node of the Huffman tree.
    """
    # Create leaf nodes and use a simple list as a priority queue
    nodes = [HuffmanNode(symbol=s, freq=f) for s, f in symbol_freq.items()]

    while len(nodes) > 1:
        # Sort by frequency (a real implementation would use a heap)
        nodes.sort(key=lambda n: n.freq)

        # Merge two lowest frequency nodes
        left = nodes.pop(0)
        right = nodes.pop(0)
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        nodes.append(merged)

    return nodes[0]


def build_codebook(root):
    """
    Build codebook (symbol -> binary string) from Huffman tree via DFS.

    Returns:
        Dict mapping symbol -> binary code string.
    """
    codebook = {}

    def traverse(node, code=""):
        if node is None:
            return
        if node.is_leaf():
            # Handle edge case: single symbol
            codebook[node.symbol] = code if code else "0"
            return
        traverse(node.left, code + "0")
        traverse(node.right, code + "1")

    traverse(root)
    return codebook


def huffman_encode(message, codebook):
    """Encode a message using the Huffman codebook."""
    return "".join(codebook[symbol] for symbol in message)


def huffman_decode(bitstring, root):
    """Decode a bitstring using the Huffman tree."""
    decoded = []
    node = root
    for bit in bitstring:
        if bit == "0":
            node = node.left
        else:
            node = node.right
        if node.is_leaf():
            decoded.append(node.symbol)
            node = root
    return decoded


def demo_huffman_coding():
    """
    Full Huffman coding demo: build tree, encode, decode, analyze compression.
    Show that compression ratio approaches entropy.
    """
    print("=" * 70)
    print("7. HUFFMAN CODING FROM SCRATCH")
    print("=" * 70)

    # ----- Example 1: Simple message -----
    print("\n--- Example 1: Simple message ---")
    message = list("abracadabra")
    freq = Counter(message)
    total = len(message)

    print(f"\n  Message: '{''.join(message)}' (length {total})")
    print(f"  Symbol frequencies:")
    for symbol, count in sorted(freq.items(), key=lambda x: -x[1]):
        print(f"    '{symbol}': {count}/{total} = {count / total:.4f}")

    # Build Huffman tree and codebook
    tree = build_huffman_tree(freq)
    codebook = build_codebook(tree)

    print(f"\n  Huffman codes:")
    for symbol, code in sorted(codebook.items()):
        print(f"    '{symbol}' -> {code}  (length {len(code)})")

    # Encode
    encoded = huffman_encode(message, codebook)
    print(f"\n  Encoded: {encoded}")
    print(f"  Encoded length: {len(encoded)} bits")

    # Decode and verify
    decoded = huffman_decode(encoded, tree)
    assert decoded == message, "Decoding failed!"
    print(f"  Decoded: '{''.join(decoded)}'")
    print(f"  Decode matches original: {decoded == message}")

    # Compression analysis
    fixed_bits = total * np.ceil(np.log2(len(freq)))  # fixed-length code
    huffman_bits = len(encoded)
    probs = np.array([count / total for count in freq.values()])
    h = entropy(probs)
    avg_code_length = huffman_bits / total

    print(f"\n  Compression analysis:")
    print(f"    Fixed-length code: {int(fixed_bits)} bits "
          f"({np.ceil(np.log2(len(freq))):.0f} bits/symbol x {total} symbols)")
    print(f"    Huffman code: {huffman_bits} bits "
          f"({avg_code_length:.3f} bits/symbol average)")
    print(f"    Entropy H: {h:.4f} bits/symbol (theoretical minimum)")
    print(f"    Huffman overhead: {avg_code_length - h:.4f} bits/symbol above entropy")
    print(f"    Compression ratio: {huffman_bits / fixed_bits:.3f} "
          f"({(1 - huffman_bits / fixed_bits) * 100:.1f}% space saved)")

    # ----- Example 2: Larger text showing convergence to entropy -----
    print("\n--- Example 2: Compression ratio approaches entropy ---")

    # Generate text from a known distribution
    rng = np.random.default_rng(42)
    symbols = ['a', 'b', 'c', 'd', 'e']
    true_probs = np.array([0.40, 0.25, 0.15, 0.12, 0.08])
    h_true = entropy(true_probs)

    print(f"\n  True distribution: {dict(zip(symbols, true_probs))}")
    print(f"  True entropy: {h_true:.4f} bits/symbol")
    print(f"\n  {'Length':>8s}  {'Huffman bits/sym':>18s}  {'Entropy':>10s}  {'Overhead':>10s}")
    print(f"  {'-' * 50}")

    for n in [50, 100, 500, 1000, 5000, 10000, 50000]:
        # Generate random message from true distribution
        indices = rng.choice(len(symbols), size=n, p=true_probs)
        msg = [symbols[i] for i in indices]

        # Build Huffman code from empirical frequencies
        freq = Counter(msg)
        tree = build_huffman_tree(freq)
        codebook = build_codebook(tree)
        encoded = huffman_encode(msg, codebook)

        avg_bits = len(encoded) / n
        overhead = avg_bits - h_true

        print(f"  {n:>8d}  {avg_bits:>18.4f}  {h_true:>10.4f}  {overhead:>+10.4f}")

    print(f"\n  As message length grows, Huffman average bits/symbol -> entropy.")
    print(f"  This is Shannon's source coding theorem in action:")
    print(f"  No lossless code can do better than H bits/symbol on average.")

    # ----- Example 3: Extreme distributions -----
    print("\n--- Example 3: Extreme distributions ---")

    # Very skewed distribution
    print("\n  Highly skewed distribution (one dominant symbol):")
    symbols_skew = ['a', 'b', 'c', 'd']
    probs_skew = np.array([0.90, 0.05, 0.03, 0.02])
    h_skew = entropy(probs_skew)

    indices = rng.choice(len(symbols_skew), size=1000, p=probs_skew)
    msg_skew = [symbols_skew[i] for i in indices]
    freq_skew = Counter(msg_skew)
    tree_skew = build_huffman_tree(freq_skew)
    codebook_skew = build_codebook(tree_skew)
    encoded_skew = huffman_encode(msg_skew, codebook_skew)

    print(f"    Distribution: {dict(zip(symbols_skew, probs_skew))}")
    print(f"    Entropy: {h_skew:.4f} bits/symbol")
    print(f"    Huffman: {len(encoded_skew) / 1000:.4f} bits/symbol")
    print(f"    Fixed:   {np.ceil(np.log2(4)):.4f} bits/symbol")
    print(f"    Huffman saves {(1 - len(encoded_skew) / (1000 * 2)) * 100:.1f}% vs fixed-length")

    # Uniform distribution (hardest to compress)
    print("\n  Uniform distribution (incompressible):")
    probs_uniform = np.array([0.25, 0.25, 0.25, 0.25])
    h_uniform = entropy(probs_uniform)

    indices = rng.choice(4, size=1000, p=probs_uniform)
    msg_uniform = [symbols_skew[i] for i in indices]
    freq_uniform = Counter(msg_uniform)
    tree_uniform = build_huffman_tree(freq_uniform)
    codebook_uniform = build_codebook(tree_uniform)
    encoded_uniform = huffman_encode(msg_uniform, codebook_uniform)

    print(f"    Distribution: {dict(zip(symbols_skew, probs_uniform))}")
    print(f"    Entropy: {h_uniform:.4f} bits/symbol")
    print(f"    Huffman: {len(encoded_uniform) / 1000:.4f} bits/symbol")
    print(f"    Fixed:   {np.ceil(np.log2(4)):.4f} bits/symbol")
    print(f"    Huffman saves ~{(1 - len(encoded_uniform) / (1000 * 2)) * 100:.1f}% "
          f"(nearly nothing -- already at max entropy)")

    print()


# =============================================================================
# 8. DATA PROCESSING INEQUALITY DEMONSTRATION
# =============================================================================

def demo_data_processing_inequality():
    """
    Demonstrate that processing can only lose information.
    X -> Y -> Z  implies  I(X;Z) <= I(X;Y)
    """
    print("=" * 70)
    print("8. DATA PROCESSING INEQUALITY")
    print("=" * 70)

    rng = np.random.default_rng(42)
    n_samples = 50000

    # X: original signal
    x = rng.normal(0, 1, n_samples)

    # Y = f(X) + noise: noisy observation
    y = 2 * x + rng.normal(0, 0.5, n_samples)

    # Z = g(Y): further processing of Y (X -> Y -> Z is Markov)
    z = np.sign(y)  # quantize to binary: loses information

    mi_xy = mutual_information_from_data(x, y, bins=30)
    mi_xz = mutual_information_from_data(x, z, bins=30)
    mi_yz = mutual_information_from_data(y, z, bins=30)

    print(f"\n  Markov chain: X -> Y -> Z")
    print(f"  X = Gaussian noise")
    print(f"  Y = 2X + noise  (noisy linear transform)")
    print(f"  Z = sign(Y)     (binary quantization)")

    print(f"\n  I(X; Y) = {mi_xy:.4f} bits  (X and its noisy observation)")
    print(f"  I(X; Z) = {mi_xz:.4f} bits  (X and the quantized version)")
    print(f"  I(Y; Z) = {mi_yz:.4f} bits  (Y and its quantization)")

    print(f"\n  Data Processing Inequality: I(X;Z) <= I(X;Y)")
    print(f"  {mi_xz:.4f} <= {mi_xy:.4f}: {mi_xz <= mi_xy + 0.01}")  # small tolerance for estimation
    print(f"\n  Quantizing Y to get Z destroyed information about X.")
    print(f"  No amount of processing of Z can recover what was lost.")

    # Multiple processing steps
    print(f"\n  Successive processing steps:")
    signals = [x]
    names = ["X (original)"]
    current = x.copy()
    transformations = [
        ("Y = 2X + noise(0.5)", lambda s: 2 * s + rng.normal(0, 0.5, len(s))),
        ("Z = Y + noise(1.0)", lambda s: s + rng.normal(0, 1.0, len(s))),
        ("W = round(Z)", lambda s: np.round(s)),
        ("V = clip(W, -2, 2)", lambda s: np.clip(s, -2, 2)),
    ]

    print(f"    {'Step':<25s}  {'I(X; step)':>12s}  {'Info retained':>14s}")
    print(f"    {'-' * 55}")
    mi_original = entropy(np.histogram(x, bins=30, density=True)[0].clip(1e-15))

    for name, transform in transformations:
        current = transform(current)
        mi = mutual_information_from_data(x, current, bins=30)
        print(f"    {name:<25s}  {mi:>12.4f}  {'(decreasing)':>14s}")

    print(f"\n  Each step can only lose information about X -- never gain it.")

    print()


# =============================================================================
# MAIN: RUN ALL DEMONSTRATIONS
# =============================================================================

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  INFORMATION THEORY: FROM-SCRATCH IMPLEMENTATION")
    print("#  Everything built with NumPy only.")
    print("#" * 70 + "\n")

    demo_entropy()
    demo_cross_entropy()
    demo_kl_divergence()
    demo_mutual_information()
    demo_cross_entropy_loss()
    demo_information_bottleneck()
    demo_huffman_coding()
    demo_data_processing_inequality()

    print("=" * 70)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  1. Information = surprise = -log(probability)")
    print("  2. Entropy = expected surprise = minimum bits for compression")
    print("  3. Cross-entropy loss IS information theory (not just inspired by it)")
    print("  4. KL divergence is asymmetric: forward vs reverse give different behaviors")
    print("  5. Mutual information measures shared knowledge between variables")
    print("  6. The Information Bottleneck: good representations compress irrelevant info")
    print("  7. Huffman coding achieves rates approaching entropy (source coding theorem)")
    print("  8. Data Processing Inequality: processing can only destroy information")
    print()

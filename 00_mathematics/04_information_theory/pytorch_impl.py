"""
Information Theory with PyTorch
================================

PyTorch re-implementation of information theory from implementation.py.

Comparison:
    NumPy (implementation.py)         →  PyTorch (this file)
    ───────────────────────────────────────────────────────────
    Manual entropy calculation        →  torch + distributions.entropy()
    Manual cross-entropy              →  F.cross_entropy (fused + stable)
    Manual KL divergence              →  F.kl_div + dist.kl_divergence
    Manual softmax + CE               →  F.cross_entropy (logits directly)
    Manual Huffman coding             →  Same algorithm with torch tensors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np


# =============================================================================
# PART 1: ENTROPY AND CROSS-ENTROPY
# =============================================================================

def entropy(probs: torch.Tensor) -> torch.Tensor:
    """Shannon entropy: H(X) = -sum(p * log(p))."""
    # Mask zeros to avoid log(0)
    mask = probs > 0
    return -(probs[mask] * torch.log2(probs[mask])).sum()


def cross_entropy(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Cross-entropy: H(p, q) = -sum(p * log(q))."""
    mask = p > 0
    return -(p[mask] * torch.log2(q[mask])).sum()


def demo_entropy():
    """Demonstrate entropy and cross-entropy."""
    print("=" * 60)
    print("ENTROPY & CROSS-ENTROPY")
    print("=" * 60)

    # Fair vs biased coin
    fair = torch.tensor([0.5, 0.5])
    biased = torch.tensor([0.9, 0.1])
    certain = torch.tensor([1.0, 0.0])

    print(f"Fair coin entropy: {entropy(fair):.4f} bits (max=1.0)")
    print(f"Biased coin (0.9): {entropy(biased):.4f} bits")
    print(f"Certain outcome:   {entropy(certain):.4f} bits")

    # Cross-entropy: using wrong distribution
    print(f"\nH(fair, biased) = {cross_entropy(fair, biased):.4f}")
    print(f"H(fair, fair) = {cross_entropy(fair, fair):.4f}")
    print(f"Cross-entropy >= entropy (Gibbs' inequality)")
    print()


# =============================================================================
# PART 2: KL DIVERGENCE
# =============================================================================

def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) = sum(p * log(p/q)). Measures information lost using q instead of p."""
    mask = (p > 0) & (q > 0)
    return (p[mask] * torch.log(p[mask] / q[mask])).sum()


def demo_kl_divergence():
    """Demonstrate KL divergence properties and asymmetry."""
    print("=" * 60)
    print("KL DIVERGENCE")
    print("=" * 60)

    p = torch.tensor([0.4, 0.3, 0.2, 0.1])
    q = torch.tensor([0.25, 0.25, 0.25, 0.25])

    print(f"p = {p.tolist()}")
    print(f"q = {q.tolist()} (uniform)")
    print(f"KL(p||q) = {kl_divergence(p, q):.4f}")
    print(f"KL(q||p) = {kl_divergence(q, p):.4f}")
    print(f"KL is asymmetric: KL(p||q) ≠ KL(q||p)")
    print(f"KL(p||p) = {kl_divergence(p, p):.6f} (should be 0)")

    # Forward KL vs Reverse KL on bimodal
    print(f"\nForward KL (mean-seeking) vs Reverse KL (mode-seeking):")
    # Bimodal target
    target = torch.tensor([0.5, 0.0, 0.0, 0.0, 0.5])
    # Mode-covering approximation (spread out)
    q_cover = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
    # Mode-seeking approximation (concentrate on one mode)
    q_seek = torch.tensor([0.9, 0.025, 0.025, 0.025, 0.025])

    print(f"  Target (bimodal): {target.tolist()}")
    print(f"  KL(target||q_cover) = {kl_divergence(target, q_cover):.4f} (forward KL favors this)")
    print(f"  KL(target||q_seek)  = {kl_divergence(target, q_seek):.4f}")
    print()


# =============================================================================
# PART 3: CROSS-ENTROPY LOSS IN NEURAL NETWORKS
# =============================================================================

def demo_cross_entropy_loss():
    """Show why cross-entropy is used instead of MSE for classification."""
    print("=" * 60)
    print("CROSS-ENTROPY LOSS (Neural Networks)")
    print("=" * 60)

    # Compare gradient magnitude: CE vs MSE for wrong confident prediction
    # Simulate: true label = class 0, model predicts class 2 with high confidence
    logits = torch.tensor([[0.1, 0.1, 5.0]], requires_grad=True)
    target = torch.tensor([0])

    # Cross-entropy loss
    ce_loss = F.cross_entropy(logits, target)
    ce_loss.backward()
    ce_grad_norm = logits.grad.norm().item()

    print(f"Logits: {logits.data.tolist()[0]}")
    print(f"Target: class 0")
    print(f"Softmax: {F.softmax(logits, dim=1).data.tolist()[0]}")
    print(f"\nCross-entropy loss: {ce_loss.item():.4f}")
    print(f"CE gradient norm:   {ce_grad_norm:.4f}")

    # MSE loss
    logits2 = torch.tensor([[0.1, 0.1, 5.0]], requires_grad=True)
    probs = F.softmax(logits2, dim=1)
    target_onehot = torch.tensor([[1.0, 0.0, 0.0]])
    mse_loss = F.mse_loss(probs, target_onehot)
    mse_loss.backward()
    mse_grad_norm = logits2.grad.norm().item()

    print(f"\nMSE loss:           {mse_loss.item():.4f}")
    print(f"MSE gradient norm:  {mse_grad_norm:.4f}")
    print(f"\n→ CE gives {ce_grad_norm/mse_grad_norm:.1f}x stronger gradient for wrong predictions")
    print(f"→ This is why CE is standard for classification")
    print()

    # Label smoothing
    print("LABEL SMOOTHING:")
    logits3 = torch.tensor([[2.0, 0.5, 0.1]], requires_grad=True)
    target3 = torch.tensor([0])

    loss_hard = F.cross_entropy(logits3, target3)
    print(f"  Hard labels CE: {loss_hard.item():.4f}")

    logits4 = logits3.clone().detach().requires_grad_(True)
    loss_smooth = F.cross_entropy(logits4, target3, label_smoothing=0.1)
    print(f"  Smooth (ε=0.1) CE: {loss_smooth.item():.4f}")
    print(f"  → Smoothing prevents overconfidence")
    print()


# =============================================================================
# PART 4: MUTUAL INFORMATION
# =============================================================================

def demo_mutual_information():
    """Estimate mutual information I(X;Y) from samples."""
    print("=" * 60)
    print("MUTUAL INFORMATION")
    print("=" * 60)

    torch.manual_seed(42)

    # Correlated variables: Y = X + noise
    n = 10000
    X = torch.randn(n)
    noise_levels = [0.01, 0.1, 0.5, 1.0, 5.0]

    for noise_std in noise_levels:
        Y = X + noise_std * torch.randn(n)

        # Estimate MI via binning
        n_bins = 30
        x_bins = torch.linspace(X.min(), X.max(), n_bins + 1)
        y_bins = torch.linspace(Y.min(), Y.max(), n_bins + 1)

        # Joint histogram
        joint = torch.zeros(n_bins, n_bins)
        x_idx = torch.clamp(torch.bucketize(X, x_bins) - 1, 0, n_bins - 1)
        y_idx = torch.clamp(torch.bucketize(Y, y_bins) - 1, 0, n_bins - 1)
        for i in range(n):
            joint[x_idx[i], y_idx[i]] += 1
        joint = joint / joint.sum()

        # Marginals
        px = joint.sum(dim=1)
        py = joint.sum(dim=0)

        # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += joint[i, j] * torch.log(joint[i, j] / (px[i] * py[j]))

        print(f"  noise_std={noise_std:.2f}: MI ≈ {mi.item():.4f} nats")

    print(f"  → More noise = less mutual information")
    print()


# =============================================================================
# PART 5: TEMPERATURE SCALING
# =============================================================================

def demo_temperature():
    """Show effect of temperature on softmax distribution."""
    print("=" * 60)
    print("TEMPERATURE SCALING")
    print("=" * 60)

    logits = torch.tensor([2.0, 1.0, 0.5, 0.1])

    for T in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        probs = F.softmax(logits / T, dim=0)
        H = entropy(probs)
        print(f"  T={T:4.1f}: probs={[f'{p:.3f}' for p in probs.tolist()]}, "
              f"entropy={H:.3f} bits")

    print(f"  → Low T = sharp (confident), High T = uniform (uncertain)")
    print()


if __name__ == "__main__":
    demo_entropy()
    demo_kl_divergence()
    demo_cross_entropy_loss()
    demo_mutual_information()
    demo_temperature()

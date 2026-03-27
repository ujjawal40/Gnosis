"""
Model Quantization: From Scratch
===================================

Techniques to reduce model size and speed up inference by
using lower-precision arithmetic.

Methods:
    1. Post-Training Quantization (PTQ)
    2. Dynamic Quantization
    3. Quantization-Aware Training (QAT)
    4. Mixed-Precision concepts

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
# PART 1: QUANTIZATION FUNDAMENTALS
# =============================================================================

def compute_scale_zero_point(min_val: float, max_val: float,
                              n_bits: int = 8):
    """
    Compute quantization parameters for affine quantization.

    Maps [min_val, max_val] -> [0, 2^n_bits - 1]

    scale = (max_val - min_val) / (2^n_bits - 1)
    zero_point = round(-min_val / scale)
    """
    qmin, qmax = 0, 2 ** n_bits - 1
    scale = (max_val - min_val) / (qmax - qmin)
    scale = max(scale, 1e-8)  # Avoid division by zero
    zero_point = int(round(-min_val / scale))
    zero_point = max(qmin, min(qmax, zero_point))
    return scale, zero_point


def quantize(x: np.ndarray, scale: float, zero_point: int,
             n_bits: int = 8) -> np.ndarray:
    """Quantize float tensor to n-bit integer."""
    qmin, qmax = 0, 2 ** n_bits - 1
    q = np.round(x / scale + zero_point).astype(np.int32)
    return np.clip(q, qmin, qmax)


def dequantize(q: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Dequantize integer tensor back to float."""
    return (q.astype(np.float64) - zero_point) * scale


def symmetric_quantize(x: np.ndarray, n_bits: int = 8):
    """
    Symmetric quantization (no zero point, maps around 0).

    Used for weights where distribution is roughly symmetric.
    """
    qmax = 2 ** (n_bits - 1) - 1
    abs_max = np.abs(x).max()
    scale = abs_max / qmax if abs_max > 0 else 1.0
    q = np.round(x / scale).astype(np.int32)
    q = np.clip(q, -qmax, qmax)
    return q, scale


def symmetric_dequantize(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float64) * scale


# =============================================================================
# PART 2: PER-CHANNEL QUANTIZATION
# =============================================================================

def per_channel_quantize(weight: np.ndarray, n_bits: int = 8):
    """
    Per-channel quantization for conv/linear weights.

    Each output channel gets its own scale, giving better accuracy
    than per-tensor quantization.
    """
    n_channels = weight.shape[0]
    scales = np.zeros(n_channels)
    q_weight = np.zeros_like(weight, dtype=np.int32)
    qmax = 2 ** (n_bits - 1) - 1

    for c in range(n_channels):
        channel = weight[c].flatten()
        abs_max = np.abs(channel).max()
        scales[c] = abs_max / qmax if abs_max > 0 else 1.0
        q_weight[c] = np.round(weight[c] / scales[c]).astype(np.int32)
        q_weight[c] = np.clip(q_weight[c], -qmax, qmax)

    return q_weight, scales


def per_channel_dequantize(q_weight: np.ndarray, scales: np.ndarray):
    """Dequantize per-channel quantized weights."""
    result = np.zeros_like(q_weight, dtype=np.float64)
    for c in range(q_weight.shape[0]):
        result[c] = q_weight[c].astype(np.float64) * scales[c]
    return result


# =============================================================================
# PART 3: QUANTIZED OPERATIONS
# =============================================================================

def quantized_linear(x_q, x_scale, x_zp, w_q, w_scale, w_zp, bias=None):
    """
    Quantized matrix multiplication.

    y = scale_y * (q_x - zp_x) @ (q_w - zp_w)^T
    """
    # Integer-only matmul (in practice done in int32)
    x_int = x_q.astype(np.int32) - x_zp
    w_int = w_q.astype(np.int32) - w_zp
    y_int = x_int @ w_int.T

    # Rescale to float
    y = y_int.astype(np.float64) * (x_scale * w_scale)

    if bias is not None:
        y += bias

    return y


def quantized_relu(x_q, zero_point):
    """Quantized ReLU: just clamp to zero_point."""
    return np.maximum(x_q, zero_point)


# =============================================================================
# PART 4: QUANTIZATION ERROR ANALYSIS
# =============================================================================

def quantization_error(original: np.ndarray, n_bits: int = 8,
                        method: str = "affine"):
    """Compute quantization error for different bit widths."""
    if method == "affine":
        scale, zp = compute_scale_zero_point(original.min(), original.max(), n_bits)
        q = quantize(original, scale, zp, n_bits)
        reconstructed = dequantize(q, scale, zp)
    else:  # symmetric
        q, scale = symmetric_quantize(original, n_bits)
        reconstructed = symmetric_dequantize(q, scale)

    mse = np.mean((original - reconstructed) ** 2)
    max_err = np.max(np.abs(original - reconstructed))
    snr = 10 * np.log10(np.mean(original ** 2) / (mse + 1e-10))

    return {"mse": mse, "max_error": max_err, "snr_db": snr}


# =============================================================================
# PART 5: FAKE QUANTIZATION (QAT)
# =============================================================================

def fake_quantize(x: np.ndarray, n_bits: int = 8):
    """
    Fake quantization for Quantization-Aware Training.

    Simulates quantization during forward pass but keeps float values
    for gradient computation. Uses Straight-Through Estimator (STE)
    for backward pass.

    x_fq = dequantize(quantize(x))
    """
    scale, zp = compute_scale_zero_point(x.min(), x.max(), n_bits)
    q = quantize(x, scale, zp, n_bits)
    return dequantize(q, scale, zp)


class FakeQuantizedLinear:
    """Linear layer with fake quantization for QAT."""

    def __init__(self, in_dim, out_dim, n_bits=8):
        self.weight = np.random.randn(out_dim, in_dim) * np.sqrt(2.0 / in_dim)
        self.bias = np.zeros(out_dim)
        self.n_bits = n_bits

    def forward(self, x):
        # Fake quantize weights and activations
        w_fq = fake_quantize(self.weight, self.n_bits)
        x_fq = fake_quantize(x, self.n_bits)
        return x_fq @ w_fq.T + self.bias


# =============================================================================
# DEMO
# =============================================================================

def demo():
    print("=" * 70)
    print("MODEL QUANTIZATION")
    print("=" * 70)

    # Create sample weights (normally distributed)
    weights = np.random.randn(128, 64) * 0.5

    # Error vs bit width
    print("\n--- Quantization Error vs Bit Width ---")
    print(f"{'Bits':>6s} | {'MSE':>12s} | {'Max Error':>12s} | {'SNR (dB)':>10s}")
    print("-" * 48)

    bit_widths = [2, 4, 8, 16]
    errors = []
    for bits in bit_widths:
        err = quantization_error(weights, bits, "symmetric")
        errors.append(err)
        print(f"{bits:>6d} | {err['mse']:>12.8f} | {err['max_error']:>12.8f} | "
              f"{err['snr_db']:>10.2f}")

    # Per-tensor vs per-channel
    print("\n--- Per-tensor vs Per-channel (8-bit) ---")
    w = np.random.randn(32, 64)
    # Make channels have different scales
    w[0:8] *= 0.1
    w[8:16] *= 1.0
    w[16:24] *= 5.0
    w[24:32] *= 0.01

    # Per-tensor
    q_pt, s_pt = symmetric_quantize(w, 8)
    recon_pt = symmetric_dequantize(q_pt, s_pt)
    mse_pt = np.mean((w - recon_pt) ** 2)

    # Per-channel
    q_pc, s_pc = per_channel_quantize(w, 8)
    recon_pc = per_channel_dequantize(q_pc, s_pc)
    mse_pc = np.mean((w - recon_pc) ** 2)

    print(f"  Per-tensor MSE:  {mse_pt:.8f}")
    print(f"  Per-channel MSE: {mse_pc:.8f}")
    print(f"  Improvement:     {mse_pt/mse_pc:.1f}x")

    # Model size comparison
    print("\n--- Model Size Reduction ---")
    n_params = 10_000_000  # 10M params
    for bits in [32, 16, 8, 4, 2]:
        size_mb = n_params * bits / 8 / 1e6
        ratio = 32 / bits
        print(f"  {bits:>2d}-bit: {size_mb:>6.1f} MB ({ratio:.0f}x compression)")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar([str(b) for b in bit_widths],
            [e['snr_db'] for e in errors], color='steelblue')
    ax1.set_xlabel("Bit Width")
    ax1.set_ylabel("SNR (dB)")
    ax1.set_title("Quantization Quality vs Bit Width")

    # Show distribution before/after quantization
    flat_w = weights.flatten()
    q_w, s = symmetric_quantize(weights, 4)
    recon = symmetric_dequantize(q_w, s).flatten()
    ax2.hist(flat_w, bins=50, alpha=0.5, label="Original (FP32)", density=True)
    ax2.hist(recon, bins=50, alpha=0.5, label="Quantized (4-bit)", density=True)
    ax2.set_title("Weight Distribution: Original vs 4-bit")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(SAVE_DIR / "quantization.png", dpi=100)
    plt.close()
    print("\nPlots saved.")


if __name__ == "__main__":
    demo()

"""
Weight Initialization: From Scratch Implementation
=====================================================

How you initialize weights determines whether a network can train at all.
Bad initialization → vanishing/exploding activations from the first forward pass.

Techniques:
    1. Zero initialization (the trap)
    2. Random normal / uniform
    3. Xavier/Glorot (sigmoid/tanh networks)
    4. He/Kaiming (ReLU networks)
    5. LeCun initialization (SELU)
    6. Orthogonal initialization (RNNs)
    7. LSUV (Layer-Sequential Unit-Variance)

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
# PART 1: INITIALIZATION METHODS
# =============================================================================

def zeros_init(shape):
    """Zero initialization. NEVER use for hidden layers (symmetry problem)."""
    return np.zeros(shape)


def random_normal_init(shape, std=0.01):
    """Small random normal. Common default but often too small for deep nets."""
    return np.random.randn(*shape) * std


def random_uniform_init(shape, low=-0.1, high=0.1):
    """Random uniform initialization."""
    return np.random.uniform(low, high, shape)


def xavier_normal_init(shape):
    """
    Xavier/Glorot Normal initialization (Glorot & Bengio, 2010).

    Designed for sigmoid and tanh activations. Keeps variance constant
    through layers by using: std = sqrt(2 / (fan_in + fan_out))

    Derivation: Var(output) = Var(input) when:
        Var(W) = 2 / (fan_in + fan_out)
    """
    fan_in, fan_out = shape[0], shape[1]
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(*shape) * std


def xavier_uniform_init(shape):
    """Xavier/Glorot Uniform: U(-a, a) where a = sqrt(6 / (fan_in + fan_out))."""
    fan_in, fan_out = shape[0], shape[1]
    a = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-a, a, shape)


def he_normal_init(shape):
    """
    He/Kaiming Normal initialization (He et al., 2015).

    Designed for ReLU activations. ReLU zeros out half the values,
    so we need 2x the variance: std = sqrt(2 / fan_in)

    This is THE standard for modern deep learning with ReLU.
    """
    fan_in = shape[0]
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape) * std


def he_uniform_init(shape):
    """He/Kaiming Uniform: U(-a, a) where a = sqrt(6 / fan_in)."""
    fan_in = shape[0]
    a = np.sqrt(6.0 / fan_in)
    return np.random.uniform(-a, a, shape)


def lecun_normal_init(shape):
    """
    LeCun Normal initialization.

    For SELU activation: std = sqrt(1 / fan_in)
    Ensures self-normalizing property of SELU networks.
    """
    fan_in = shape[0]
    std = np.sqrt(1.0 / fan_in)
    return np.random.randn(*shape) * std


def orthogonal_init(shape, gain=1.0):
    """
    Orthogonal initialization (Saxe et al., 2014).

    Creates an orthogonal matrix via QR decomposition of random matrix.
    Preserves gradient norm during backprop. Essential for RNNs.
    """
    flat_shape = (shape[0], np.prod(shape[1:]) if len(shape) > 1 else shape[0])
    a = np.random.randn(*flat_shape)
    q, r = np.linalg.qr(a)
    # Make Q uniform
    d = np.diag(r)
    q *= np.sign(d)
    if flat_shape[0] < flat_shape[1]:
        q = q.T
    return (gain * q[:shape[0], :shape[1] if len(shape) > 1 else shape[0]])


# =============================================================================
# PART 2: FORWARD PASS ANALYSIS
# =============================================================================

def analyze_forward_pass(init_fn, name, n_layers=20, dim=256,
                         activation="relu"):
    """Track activation statistics through a deep network."""
    x = np.random.randn(32, dim)  # batch of 32
    means, stds = [], []

    for _ in range(n_layers):
        W = init_fn((dim, dim))
        x = x @ W

        if activation == "relu":
            x = np.maximum(0, x)
        elif activation == "tanh":
            x = np.tanh(x)
        elif activation == "sigmoid":
            x = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

        means.append(float(np.mean(np.abs(x))))
        stds.append(float(np.std(x)))

    return means, stds


def demo_initialization():
    """Compare all initialization methods on a deep network."""
    print("=" * 70)
    print("WEIGHT INITIALIZATION COMPARISON")
    print("=" * 70)

    inits_relu = {
        "Random(0.01)": lambda s: random_normal_init(s, 0.01),
        "Random(1.0)": lambda s: random_normal_init(s, 1.0),
        "Xavier Normal": xavier_normal_init,
        "He Normal": he_normal_init,
        "Orthogonal": orthogonal_init,
    }

    print("\nWith ReLU activation (20 layers, dim=256):")
    print(f"{'Init Method':>20s} | {'Final Mean |x|':>15s} | {'Final Std':>12s} | {'Status':>10s}")
    print("-" * 65)

    results = {}
    for name, fn in inits_relu.items():
        means, stds = analyze_forward_pass(fn, name, activation="relu")
        final_mean = means[-1]
        final_std = stds[-1]

        if final_mean < 1e-6:
            status = "VANISHED"
        elif final_mean > 1e6 or np.isnan(final_mean):
            status = "EXPLODED"
        else:
            status = "HEALTHY"

        results[name] = (means, stds)
        print(f"{name:>20s} | {final_mean:>15.6f} | {final_std:>12.6f} | {status:>10s}")

    # Xavier with tanh
    print("\nWith Tanh activation (20 layers):")
    inits_tanh = {"Xavier Normal": xavier_normal_init, "He Normal": he_normal_init}
    for name, fn in inits_tanh.items():
        means, stds = analyze_forward_pass(fn, name, activation="tanh")
        print(f"  {name:>20s} | Final Std: {stds[-1]:.6f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, (means, stds) in results.items():
        ax1.plot(means, label=name, linewidth=2)
        ax2.plot(stds, label=name, linewidth=2)

    ax1.set_title("Activation Magnitude Through Layers (ReLU)")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Mean |activation|")
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Activation Std Through Layers (ReLU)")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Std(activation)")
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(SAVE_DIR / "initialization_comparison.png", dpi=100)
    plt.close()
    print(f"\nPlot saved to {SAVE_DIR / 'initialization_comparison.png'}")


# =============================================================================
# PART 3: LSUV (Layer-Sequential Unit-Variance)
# =============================================================================

def lsuv_init(weights_list, x_sample, target_std=1.0, max_iters=10, tol=0.1):
    """
    Layer-Sequential Unit-Variance initialization (Mishkin & Matas, 2016).

    Data-driven initialization that iteratively rescales each layer's weights
    until the output variance is approximately 1.0.

    Algorithm:
        1. Initialize with orthogonal
        2. For each layer: forward pass, measure output std, rescale weights
        3. Repeat until convergence
    """
    print("\nLSUV Initialization:")
    for i, W in enumerate(weights_list):
        h = x_sample
        # Forward through previous layers
        for j in range(i):
            h = np.maximum(0, h @ weights_list[j])

        for iteration in range(max_iters):
            out = np.maximum(0, h @ W)
            current_std = out.std()
            if abs(current_std - target_std) < tol:
                break
            W *= target_std / (current_std + 1e-8)
            weights_list[i] = W

        print(f"  Layer {i}: std = {out.std():.4f} (after {iteration+1} iters)")

    return weights_list


if __name__ == "__main__":
    demo_initialization()

    # LSUV demo
    print("\n" + "=" * 70)
    dim = 128
    weights = [orthogonal_init((dim, dim)) for _ in range(10)]
    x = np.random.randn(64, dim)
    weights = lsuv_init(weights, x)

"""
Residual Networks and Skip Connections: From Scratch
======================================================

Implementation of residual learning, bottleneck blocks, dense connections,
highway networks, and gradient flow analysis - all from first principles.

Key insight: Skip connections allow gradients to flow directly through
the network, solving the vanishing gradient problem in deep networks.

    Plain:    x → F(x)
    Residual: x → F(x) + x    (learn the residual F(x) = H(x) - x)

All code uses only NumPy. No frameworks.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

SAVE_DIR = Path(__file__).parent / "plots"
SAVE_DIR.mkdir(exist_ok=True)

np.random.seed(42)


# =============================================================================
# PART 1: BUILDING BLOCKS
# =============================================================================

def conv2d_forward(x, W, b, stride=1, padding=0):
    """
    2D convolution forward pass.

    Args:
        x: input (N, C_in, H, W)
        W: filters (C_out, C_in, kH, kW)
        b: bias (C_out,)
        stride: convolution stride
        padding: zero padding

    Returns:
        output: (N, C_out, H_out, W_out)
    """
    N, C_in, H, Wi = x.shape
    C_out, _, kH, kW = W.shape

    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (Wi + 2 * padding - kW) // stride + 1
    out = np.zeros((N, C_out, H_out, W_out))

    for n in range(N):
        for co in range(C_out):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride
                    w_start = w * stride
                    patch = x[n, :, h_start:h_start+kH, w_start:w_start+kW]
                    out[n, co, h, w] = np.sum(patch * W[co]) + b[co]
    return out


def batch_norm_forward(x, gamma, beta, running_mean, running_var,
                       training=True, momentum=0.1, eps=1e-5):
    """Batch normalization for convolutional layers (per-channel)."""
    if x.ndim == 4:
        N, C, H, W = x.shape
        # Per-channel statistics
        if training:
            mean = x.mean(axis=(0, 2, 3), keepdims=True)
            var = x.var(axis=(0, 2, 3), keepdims=True)
            running_mean[:] = (1 - momentum) * running_mean + momentum * mean.reshape(C)
            running_var[:] = (1 - momentum) * running_var + momentum * var.reshape(C)
        else:
            mean = running_mean.reshape(1, C, 1, 1)
            var = running_var.reshape(1, C, 1, 1)

        x_norm = (x - mean) / np.sqrt(var + eps)
        gamma_r = gamma.reshape(1, C, 1, 1)
        beta_r = beta.reshape(1, C, 1, 1)
        return gamma_r * x_norm + beta_r
    else:
        # Fully connected
        if training:
            mean = x.mean(axis=0, keepdims=True)
            var = x.var(axis=0, keepdims=True)
        else:
            mean = running_mean
            var = running_var
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta


def relu(x):
    """ReLU activation."""
    return np.maximum(0, x)


def global_avg_pool(x):
    """Global average pooling: (N, C, H, W) → (N, C)."""
    return x.mean(axis=(2, 3))


# =============================================================================
# PART 2: RESIDUAL BLOCK
# =============================================================================

class BasicResidualBlock:
    """
    Basic residual block: two conv layers with skip connection.

        x ─────────────────────────── (+) → ReLU → out
        │                              ↑
        └→ Conv → BN → ReLU → Conv → BN

    The key insight: instead of learning H(x), learn F(x) = H(x) - x.
    If the optimal mapping is close to identity, it's easier to learn
    a small residual than to learn the full mapping.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        self.stride = stride
        self.need_projection = (stride != 1) or (in_channels != out_channels)

        # First conv: 3x3
        scale1 = np.sqrt(2.0 / (in_channels * 9))
        self.W1 = np.random.randn(out_channels, in_channels, 3, 3) * scale1
        self.b1 = np.zeros(out_channels)
        self.gamma1 = np.ones(out_channels)
        self.beta1 = np.zeros(out_channels)
        self.rm1 = np.zeros(out_channels)
        self.rv1 = np.ones(out_channels)

        # Second conv: 3x3
        scale2 = np.sqrt(2.0 / (out_channels * 9))
        self.W2 = np.random.randn(out_channels, out_channels, 3, 3) * scale2
        self.b2 = np.zeros(out_channels)
        self.gamma2 = np.ones(out_channels)
        self.beta2 = np.zeros(out_channels)
        self.rm2 = np.zeros(out_channels)
        self.rv2 = np.ones(out_channels)

        # Projection shortcut (1x1 conv)
        if self.need_projection:
            scale_p = np.sqrt(2.0 / in_channels)
            self.W_proj = np.random.randn(out_channels, in_channels, 1, 1) * scale_p
            self.b_proj = np.zeros(out_channels)
            self.gamma_p = np.ones(out_channels)
            self.beta_p = np.zeros(out_channels)
            self.rm_p = np.zeros(out_channels)
            self.rv_p = np.ones(out_channels)

    def forward(self, x, training=True):
        # Main path
        out = conv2d_forward(x, self.W1, self.b1, stride=self.stride, padding=1)
        out = batch_norm_forward(out, self.gamma1, self.beta1, self.rm1, self.rv1, training)
        out = relu(out)
        out = conv2d_forward(out, self.W2, self.b2, stride=1, padding=1)
        out = batch_norm_forward(out, self.gamma2, self.beta2, self.rm2, self.rv2, training)

        # Shortcut path
        if self.need_projection:
            shortcut = conv2d_forward(x, self.W_proj, self.b_proj, stride=self.stride, padding=0)
            shortcut = batch_norm_forward(shortcut, self.gamma_p, self.beta_p,
                                          self.rm_p, self.rv_p, training)
        else:
            shortcut = x

        # Add and activate
        return relu(out + shortcut)


# =============================================================================
# PART 3: BOTTLENECK BLOCK
# =============================================================================

class BottleneckBlock:
    """
    Bottleneck residual block: 1x1 → 3x3 → 1x1 with expansion.

        x ────────────────────────────────────────── (+) → ReLU → out
        │                                             ↑
        └→ 1x1(reduce) → BN → ReLU → 3x3 → BN → ReLU → 1x1(expand) → BN

    The 1x1 convolutions reduce and restore dimensions, making the 3x3
    convolution operate on a lower-dimensional space (bottleneck).
    """
    EXPANSION = 4

    def __init__(self, in_channels: int, mid_channels: int, stride: int = 1):
        out_channels = mid_channels * self.EXPANSION

        # 1x1 reduce
        scale1 = np.sqrt(2.0 / in_channels)
        self.W1 = np.random.randn(mid_channels, in_channels, 1, 1) * scale1
        self.b1 = np.zeros(mid_channels)
        self.gamma1 = np.ones(mid_channels)
        self.beta1 = np.zeros(mid_channels)
        self.rm1 = np.zeros(mid_channels)
        self.rv1 = np.ones(mid_channels)

        # 3x3 spatial
        scale2 = np.sqrt(2.0 / (mid_channels * 9))
        self.W2 = np.random.randn(mid_channels, mid_channels, 3, 3) * scale2
        self.b2 = np.zeros(mid_channels)
        self.gamma2 = np.ones(mid_channels)
        self.beta2 = np.zeros(mid_channels)
        self.rm2 = np.zeros(mid_channels)
        self.rv2 = np.ones(mid_channels)

        # 1x1 expand
        scale3 = np.sqrt(2.0 / mid_channels)
        self.W3 = np.random.randn(out_channels, mid_channels, 1, 1) * scale3
        self.b3 = np.zeros(out_channels)
        self.gamma3 = np.ones(out_channels)
        self.beta3 = np.zeros(out_channels)
        self.rm3 = np.zeros(out_channels)
        self.rv3 = np.ones(out_channels)

        # Projection shortcut
        self.need_projection = (stride != 1) or (in_channels != out_channels)
        if self.need_projection:
            scale_p = np.sqrt(2.0 / in_channels)
            self.W_proj = np.random.randn(out_channels, in_channels, 1, 1) * scale_p
            self.b_proj = np.zeros(out_channels)
            self.gamma_p = np.ones(out_channels)
            self.beta_p = np.zeros(out_channels)
            self.rm_p = np.zeros(out_channels)
            self.rv_p = np.ones(out_channels)

    def forward(self, x, training=True):
        out = conv2d_forward(x, self.W1, self.b1, stride=1, padding=0)
        out = batch_norm_forward(out, self.gamma1, self.beta1, self.rm1, self.rv1, training)
        out = relu(out)

        out = conv2d_forward(out, self.W2, self.b2, stride=1, padding=1)
        out = batch_norm_forward(out, self.gamma2, self.beta2, self.rm2, self.rv2, training)
        out = relu(out)

        out = conv2d_forward(out, self.W3, self.b3, stride=1, padding=0)
        out = batch_norm_forward(out, self.gamma3, self.beta3, self.rm3, self.rv3, training)

        if self.need_projection:
            shortcut = conv2d_forward(x, self.W_proj, self.b_proj, stride=1, padding=0)
            shortcut = batch_norm_forward(shortcut, self.gamma_p, self.beta_p,
                                          self.rm_p, self.rv_p, training)
        else:
            shortcut = x

        return relu(out + shortcut)


# =============================================================================
# PART 4: DENSE BLOCK (DenseNet-style)
# =============================================================================

class DenseLayer:
    """
    Dense layer: concatenates input with output (DenseNet-style).

        x → [x, F(x)] → next layer gets all previous features

    Unlike ResNet (addition), DenseNet concatenates, enabling feature reuse
    and reducing the number of parameters needed per layer.
    """

    def __init__(self, in_channels: int, growth_rate: int):
        scale = np.sqrt(2.0 / (in_channels * 9))
        self.W = np.random.randn(growth_rate, in_channels, 3, 3) * scale
        self.b = np.zeros(growth_rate)
        self.gamma = np.ones(growth_rate)
        self.beta = np.zeros(growth_rate)
        self.rm = np.zeros(growth_rate)
        self.rv = np.ones(growth_rate)

    def forward(self, x, training=True):
        out = conv2d_forward(x, self.W, self.b, stride=1, padding=1)
        out = batch_norm_forward(out, self.gamma, self.beta, self.rm, self.rv, training)
        out = relu(out)
        return np.concatenate([x, out], axis=1)


class DenseBlock:
    """Stack of dense layers with feature concatenation."""

    def __init__(self, in_channels: int, growth_rate: int, n_layers: int):
        self.layers = []
        ch = in_channels
        for _ in range(n_layers):
            self.layers.append(DenseLayer(ch, growth_rate))
            ch += growth_rate
        self.out_channels = ch

    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training)
        return x


# =============================================================================
# PART 5: HIGHWAY NETWORK
# =============================================================================

class HighwayBlock:
    """
    Highway Network: learned gating of skip connections.

        T(x) = sigmoid(W_T @ x + b_T)     (transform gate)
        H(x) = ReLU(W_H @ x + b_H)        (transformation)
        y = T(x) * H(x) + (1 - T(x)) * x  (gated output)

    When T(x) → 0: output ≈ x (highway, pass through)
    When T(x) → 1: output ≈ H(x) (transform)
    The network learns how much to transform vs pass through.
    """

    def __init__(self, dim: int):
        scale = np.sqrt(2.0 / dim)
        self.W_H = np.random.randn(dim, dim) * scale
        self.b_H = np.zeros(dim)
        self.W_T = np.random.randn(dim, dim) * scale
        self.b_T = np.full(dim, -2.0)  # Bias towards pass-through initially

    def forward(self, x):
        H = relu(x @ self.W_H.T + self.b_H)
        T = 1.0 / (1.0 + np.exp(-(x @ self.W_T.T + self.b_T)))
        return T * H + (1.0 - T) * x


# =============================================================================
# PART 6: GRADIENT FLOW ANALYSIS
# =============================================================================

def analyze_gradient_flow():
    """
    Compare gradient flow through plain vs residual networks.

    Shows that skip connections maintain gradient magnitude through deep networks,
    preventing the vanishing gradient problem.
    """
    print("=" * 70)
    print("GRADIENT FLOW ANALYSIS: PLAIN vs RESIDUAL")
    print("=" * 70)

    depths = [5, 10, 20, 50]
    dim = 32
    results = {"plain": {}, "residual": {}, "highway": {}}

    for depth in depths:
        # Plain network
        np.random.seed(42)
        gradient = np.random.randn(1, dim)
        plain_norms = [np.linalg.norm(gradient)]
        for _ in range(depth):
            W = np.random.randn(dim, dim) * np.sqrt(2.0 / dim)
            # Simulate: gradient * W * relu_derivative
            mask = (np.random.rand(1, dim) > 0.5).astype(float)
            gradient = (gradient @ W) * mask
            plain_norms.append(np.linalg.norm(gradient))
        results["plain"][depth] = plain_norms

        # Residual network
        np.random.seed(42)
        gradient = np.random.randn(1, dim)
        res_norms = [np.linalg.norm(gradient)]
        for _ in range(depth):
            W = np.random.randn(dim, dim) * np.sqrt(2.0 / dim)
            mask = (np.random.rand(1, dim) > 0.5).astype(float)
            grad_through_block = (gradient @ W) * mask
            gradient = gradient + grad_through_block  # Skip connection!
            res_norms.append(np.linalg.norm(gradient))
        results["residual"][depth] = res_norms

        # Highway network
        np.random.seed(42)
        gradient = np.random.randn(1, dim)
        hw_norms = [np.linalg.norm(gradient)]
        for _ in range(depth):
            W = np.random.randn(dim, dim) * np.sqrt(2.0 / dim)
            mask = (np.random.rand(1, dim) > 0.5).astype(float)
            gate = 0.5  # Simplified gate value
            grad_through = (gradient @ W) * mask
            gradient = gate * grad_through + (1 - gate) * gradient
            hw_norms.append(np.linalg.norm(gradient))
        results["highway"][depth] = hw_norms

        ratio_plain = plain_norms[-1] / (plain_norms[0] + 1e-10)
        ratio_res = res_norms[-1] / (res_norms[0] + 1e-10)
        ratio_hw = hw_norms[-1] / (hw_norms[0] + 1e-10)
        print(f"Depth {depth:3d} | Plain: {ratio_plain:.6f} | "
              f"Residual: {ratio_res:.4f} | Highway: {ratio_hw:.4f}")

    # Plot
    fig, axes = plt.subplots(1, len(depths), figsize=(16, 4))
    for ax, depth in zip(axes, depths):
        ax.plot(results["plain"][depth], label="Plain", color='red', alpha=0.8)
        ax.plot(results["residual"][depth], label="Residual", color='blue', alpha=0.8)
        ax.plot(results["highway"][depth], label="Highway", color='green', alpha=0.8)
        ax.set_title(f"Depth = {depth}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Gradient Norm")
        ax.legend(fontsize=8)
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(SAVE_DIR / "gradient_flow.png", dpi=100)
    plt.close()
    print(f"\nPlot saved to {SAVE_DIR / 'gradient_flow.png'}")


# =============================================================================
# PART 7: DEMO
# =============================================================================

def demo_blocks():
    """Demonstrate each block type."""
    print("=" * 70)
    print("RESIDUAL BLOCK DEMO")
    print("=" * 70)

    # Small input: (batch=2, channels=16, height=8, width=8)
    x = np.random.randn(2, 16, 8, 8).astype(np.float64)

    # Basic residual block (same dimensions)
    block = BasicResidualBlock(16, 16)
    out = block.forward(x)
    print(f"BasicResidualBlock: {x.shape} → {out.shape}")

    # Basic residual block (downsample)
    block_ds = BasicResidualBlock(16, 32, stride=2)
    out_ds = block_ds.forward(x)
    print(f"BasicResidualBlock (stride=2): {x.shape} → {out_ds.shape}")

    # Bottleneck block
    bottleneck = BottleneckBlock(16, 8)
    out_bn = bottleneck.forward(x)
    print(f"BottleneckBlock: {x.shape} → {out_bn.shape}")

    # Dense block
    dense = DenseBlock(16, growth_rate=8, n_layers=3)
    out_dense = dense.forward(x)
    print(f"DenseBlock (3 layers, k=8): {x.shape} → {out_dense.shape}")

    # Highway block (FC)
    x_fc = np.random.randn(4, 32)
    highway = HighwayBlock(32)
    out_hw = highway.forward(x_fc)
    print(f"HighwayBlock: {x_fc.shape} → {out_hw.shape}")
    print()


if __name__ == "__main__":
    demo_blocks()
    analyze_gradient_flow()

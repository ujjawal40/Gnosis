"""
Normalization Techniques from Scratch
======================================

Implements Batch, Layer, Group, Instance, and RMS normalization
using only NumPy. Each includes forward and backward passes with
full gradient computation.

All normalization techniques share a common pattern:
    y = gamma * (x - mean) / sqrt(var + eps) + beta

They differ in WHICH dimensions they compute mean/variance over:
    - Batch Norm:    across batch dimension (per channel)
    - Layer Norm:    across feature dimensions (per sample)
    - Group Norm:    across groups of channels (per sample)
    - Instance Norm: across spatial dimensions (per sample, per channel)
    - RMS Norm:      like Layer Norm but without mean centering

All code uses only NumPy. No frameworks.
"""

import numpy as np


# ==============================================================================
# Part 1: Batch Normalization
# ==============================================================================

class BatchNorm:
    """
    Batch Normalization (Ioffe & Szegedy, 2015).

    Normalizes across the batch dimension for each feature/channel:
        mu    = (1/N) * sum(x_i)           — mean over batch
        var   = (1/N) * sum((x_i - mu)^2)  — variance over batch
        x_hat = (x - mu) / sqrt(var + eps) — normalize
        y     = gamma * x_hat + beta       — scale and shift

    During training, tracks running mean/variance with momentum for inference.
    At test time, uses running statistics instead of batch statistics.

    Supports 2D input (batch, features) and 4D input (batch, channels, H, W).
    """

    def __init__(self, num_features: int, eps: float = 1e-5,
                 momentum: float = 0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        # Running statistics for inference
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        # Cache for backward pass
        self.cache = None
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: input array, shape (N, C) or (N, C, H, W)

        Returns:
            Normalized output, same shape as input
        """
        is_4d = (x.ndim == 4)

        if self.training:
            if is_4d:
                # Compute stats over (N, H, W) for each channel
                mean = x.mean(axis=(0, 2, 3))
                var = x.var(axis=(0, 2, 3))
            else:
                # Compute stats over batch dimension
                mean = x.mean(axis=0)
                var = x.var(axis=0)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Reshape for broadcasting
        if is_4d:
            shape = (1, self.num_features, 1, 1)
        else:
            shape = (1, self.num_features)

        mean = mean.reshape(shape)
        var = var.reshape(shape)
        gamma = self.gamma.reshape(shape)
        beta = self.beta.reshape(shape)

        # Normalize
        std = np.sqrt(var + self.eps)
        x_hat = (x - mean) / std
        out = gamma * x_hat + beta

        # Cache for backward
        if self.training:
            self.cache = (x, x_hat, mean, std, gamma)

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass using cached values from forward.

        Args:
            dout: gradient of loss w.r.t. output, same shape as input

        Returns:
            dx: gradient w.r.t. input
        """
        x, x_hat, mean, std, gamma = self.cache
        is_4d = (x.ndim == 4)

        if is_4d:
            N = x.shape[0] * x.shape[2] * x.shape[3]
            reduce_axes = (0, 2, 3)
            shape = (1, self.num_features, 1, 1)
        else:
            N = x.shape[0]
            reduce_axes = (0,)
            shape = (1, self.num_features)

        # Gradients of learnable parameters
        self.dgamma = (dout * x_hat).sum(axis=reduce_axes)
        self.dbeta = dout.sum(axis=reduce_axes)

        # Gradient w.r.t. input
        dx_hat = dout * gamma
        dvar = (-0.5 * dx_hat * (x - mean) / (std ** 3)).sum(
            axis=reduce_axes).reshape(shape)
        dmean = (-dx_hat / std).sum(axis=reduce_axes).reshape(shape) + \
            dvar * (-2.0 / N) * (x - mean).sum(axis=reduce_axes).reshape(shape)

        dx = dx_hat / std + dvar * 2.0 * (x - mean) / N + dmean / N
        return dx


# ==============================================================================
# Part 2: Layer Normalization
# ==============================================================================

class LayerNorm:
    """
    Layer Normalization (Ba et al., 2016).

    Normalizes across feature dimensions for each sample independently:
        mu    = (1/D) * sum(x_d)           — mean over features
        var   = (1/D) * sum((x_d - mu)^2)  — variance over features
        x_hat = (x - mu) / sqrt(var + eps) — normalize
        y     = gamma * x_hat + beta       — scale and shift

    Unlike BatchNorm, statistics are computed per-sample, not per-batch.
    This makes it suitable for variable-length sequences and small batches.

    Supports 2D (batch, features) and 3D (batch, seq, features) inputs.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Learnable parameters
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)

        # Cache for backward
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: input array, shape (..., normalized_shape)

        Returns:
            Normalized output, same shape as input
        """
        # Normalize over the last dimension
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)

        std = np.sqrt(var + self.eps)
        x_hat = (x - mean) / std
        out = self.gamma * x_hat + self.beta

        self.cache = (x, x_hat, mean, std)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            dout: gradient of loss w.r.t. output

        Returns:
            dx: gradient w.r.t. input
        """
        x, x_hat, mean, std = self.cache
        D = self.normalized_shape

        # Parameter gradients — sum over all dims except last
        sum_axes = tuple(range(dout.ndim - 1))
        self.dgamma = (dout * x_hat).sum(axis=sum_axes)
        self.dbeta = dout.sum(axis=sum_axes)

        # Input gradient
        dx_hat = dout * self.gamma
        dvar = (-0.5 * dx_hat * (x - mean) / (std ** 3)).sum(
            axis=-1, keepdims=True)
        dmean = (-dx_hat / std).sum(axis=-1, keepdims=True) + \
            dvar * (-2.0 / D) * (x - mean).sum(axis=-1, keepdims=True)

        dx = dx_hat / std + dvar * 2.0 * (x - mean) / D + dmean / D
        return dx


# ==============================================================================
# Part 3: Group Normalization
# ==============================================================================

class GroupNorm:
    """
    Group Normalization (Wu & He, 2018).

    Divides channels into G groups, normalizes within each group:
        For group g containing channels [c_start, c_end]:
            mu_g  = mean over (channels_in_g, H, W) per sample
            var_g = var over (channels_in_g, H, W) per sample
            x_hat = (x - mu_g) / sqrt(var_g + eps)
            y     = gamma * x_hat + beta

    Interpolates between LayerNorm (G=1) and InstanceNorm (G=C).
    Works well with small batch sizes where BatchNorm fails.

    Input shape: (N, C, H, W). C must be divisible by num_groups.
    """

    def __init__(self, num_groups: int, num_channels: int,
                 eps: float = 1e-5):
        assert num_channels % num_groups == 0, \
            f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

        # Learnable parameters (per channel)
        self.gamma = np.ones(num_channels)
        self.beta = np.zeros(num_channels)

        # Cache for backward
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: input array, shape (N, C, H, W)

        Returns:
            Normalized output, same shape as input
        """
        N, C, H, W = x.shape
        G = self.num_groups
        channels_per_group = C // G

        # Reshape to (N, G, C//G, H, W)
        x_grouped = x.reshape(N, G, channels_per_group, H, W)

        # Compute mean and var over (C//G, H, W) per sample per group
        mean = x_grouped.mean(axis=(2, 3, 4), keepdims=True)
        var = x_grouped.var(axis=(2, 3, 4), keepdims=True)

        std = np.sqrt(var + self.eps)
        x_hat_grouped = (x_grouped - mean) / std

        # Reshape back to (N, C, H, W)
        x_hat = x_hat_grouped.reshape(N, C, H, W)

        # Scale and shift (per channel)
        gamma = self.gamma.reshape(1, C, 1, 1)
        beta = self.beta.reshape(1, C, 1, 1)
        out = gamma * x_hat + beta

        self.cache = (x, x_hat, x_grouped, mean, std, gamma)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            dout: gradient of loss w.r.t. output, shape (N, C, H, W)

        Returns:
            dx: gradient w.r.t. input
        """
        x, x_hat, x_grouped, mean, std, gamma = self.cache
        N, C, H, W = x.shape
        G = self.num_groups
        cpg = C // G  # channels per group

        # Parameter gradients
        self.dgamma = (dout * x_hat).sum(axis=(0, 2, 3))
        self.dbeta = dout.sum(axis=(0, 2, 3))

        # dx_hat in original shape
        dx_hat = (dout * gamma).reshape(N, G, cpg, H, W)
        D = cpg * H * W  # elements per group

        dvar = (-0.5 * dx_hat * (x_grouped - mean) / (std ** 3)).sum(
            axis=(2, 3, 4), keepdims=True)
        dmean = (-dx_hat / std).sum(axis=(2, 3, 4), keepdims=True) + \
            dvar * (-2.0 / D) * (x_grouped - mean).sum(
                axis=(2, 3, 4), keepdims=True)

        dx = dx_hat / std + dvar * 2.0 * (x_grouped - mean) / D + dmean / D
        return dx.reshape(N, C, H, W)


# ==============================================================================
# Part 4: Instance Normalization
# ==============================================================================

class InstanceNorm:
    """
    Instance Normalization (Ulyanov et al., 2016).

    Normalizes across spatial dimensions (H, W) per sample per channel:
        mu_{n,c}  = mean over (H, W) for sample n, channel c
        var_{n,c} = var over (H, W) for sample n, channel c
        x_hat     = (x - mu) / sqrt(var + eps)
        y         = gamma * x_hat + beta

    Equivalent to GroupNorm with num_groups = num_channels.
    Originally designed for style transfer — removes instance-specific
    contrast from content, enabling style-independent features.

    Input shape: (N, C, H, W).
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        self.num_features = num_features
        self.eps = eps

        # Learnable parameters (per channel)
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        # Cache for backward
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: input array, shape (N, C, H, W)

        Returns:
            Normalized output, same shape as input
        """
        # Normalize over spatial dims (H, W) per sample per channel
        mean = x.mean(axis=(2, 3), keepdims=True)
        var = x.var(axis=(2, 3), keepdims=True)

        std = np.sqrt(var + self.eps)
        x_hat = (x - mean) / std

        gamma = self.gamma.reshape(1, self.num_features, 1, 1)
        beta = self.beta.reshape(1, self.num_features, 1, 1)
        out = gamma * x_hat + beta

        self.cache = (x, x_hat, mean, std, gamma)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            dout: gradient of loss w.r.t. output, shape (N, C, H, W)

        Returns:
            dx: gradient w.r.t. input
        """
        x, x_hat, mean, std, gamma = self.cache
        H, W = x.shape[2], x.shape[3]
        D = H * W

        # Parameter gradients
        self.dgamma = (dout * x_hat).sum(axis=(0, 2, 3))
        self.dbeta = dout.sum(axis=(0, 2, 3))

        # Input gradient
        dx_hat = dout * gamma
        dvar = (-0.5 * dx_hat * (x - mean) / (std ** 3)).sum(
            axis=(2, 3), keepdims=True)
        dmean = (-dx_hat / std).sum(axis=(2, 3), keepdims=True) + \
            dvar * (-2.0 / D) * (x - mean).sum(axis=(2, 3), keepdims=True)

        dx = dx_hat / std + dvar * 2.0 * (x - mean) / D + dmean / D
        return dx


# ==============================================================================
# Part 5: RMS Normalization
# ==============================================================================

class RMSNorm:
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Simplifies LayerNorm by removing mean centering:
        rms   = sqrt((1/D) * sum(x_d^2) + eps)
        x_hat = x / rms
        y     = gamma * x_hat

    No beta parameter — the re-centering from LayerNorm is deemed
    unnecessary. Computationally cheaper and used in LLaMA, Gemma, etc.

    Supports 2D (batch, features) and 3D (batch, seq, features) inputs.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Learnable scale parameter (no bias/beta)
        self.gamma = np.ones(normalized_shape)

        # Cache for backward
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: input array, shape (..., normalized_shape)

        Returns:
            Normalized output, same shape as input
        """
        # RMS = sqrt(mean(x^2) + eps)
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        x_hat = x / rms
        out = self.gamma * x_hat

        self.cache = (x, x_hat, rms)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            dout: gradient of loss w.r.t. output

        Returns:
            dx: gradient w.r.t. input
        """
        x, x_hat, rms = self.cache
        D = self.normalized_shape

        # Parameter gradient
        sum_axes = tuple(range(dout.ndim - 1))
        self.dgamma = (dout * x_hat).sum(axis=sum_axes)

        # Input gradient
        # d(RMS)/dx = x / (D * RMS)
        # d(x/RMS)/dx = 1/RMS - x^2 / (D * RMS^3)
        dx_hat = dout * self.gamma
        dx = dx_hat / rms - (dx_hat * x).sum(
            axis=-1, keepdims=True) * x / (D * rms ** 3)
        return dx


# ==============================================================================
# Experiments
# ==============================================================================

def experiment_normalization_comparison():
    """Compare how each normalization technique transforms data."""
    print("=" * 60)
    print("EXPERIMENT 1: How Each Norm Transforms Data")
    print("=" * 60)

    np.random.seed(42)

    # Simulate a mini-batch of feature maps: (N, C, H, W)
    N, C, H, W = 8, 16, 4, 4
    # Deliberately create data with different scales per channel
    x = np.random.randn(N, C, H, W)
    for c in range(C):
        x[:, c, :, :] = x[:, c, :, :] * (c + 1) + c * 2  # scale + shift

    print(f"\nInput shape: ({N}, {C}, {H}, {W})")
    print(f"Input mean per channel (sample 0): "
          f"[{x[0,:,0,0].mean():.2f}, ...] range: "
          f"[{x.min():.2f}, {x.max():.2f}]")

    # --- Batch Norm ---
    bn = BatchNorm(num_features=C)
    bn_out = bn.forward(x)
    print(f"\nBatch Norm:")
    print(f"  Normalizes over: batch (N, H, W) per channel")
    print(f"  Output mean per channel: {bn_out.mean(axis=(0,2,3))[:4].round(4)}")
    print(f"  Output var  per channel: {bn_out.var(axis=(0,2,3))[:4].round(4)}")

    # --- Layer Norm ---
    # Reshape to (N, C*H*W) for LayerNorm
    x_flat = x.reshape(N, C * H * W)
    ln = LayerNorm(normalized_shape=C * H * W)
    ln_out = ln.forward(x_flat)
    print(f"\nLayer Norm:")
    print(f"  Normalizes over: all features per sample")
    print(f"  Output mean per sample: {ln_out.mean(axis=-1)[:4].round(6)}")
    print(f"  Output var  per sample: {ln_out.var(axis=-1)[:4].round(4)}")

    # --- Group Norm ---
    gn = GroupNorm(num_groups=4, num_channels=C)
    gn_out = gn.forward(x)
    # Check stats within groups
    gn_grouped = gn_out.reshape(N, 4, C // 4, H, W)
    print(f"\nGroup Norm (G=4):")
    print(f"  Normalizes over: channel groups per sample")
    print(f"  Output mean per group (sample 0): "
          f"{gn_grouped[0].mean(axis=(1,2,3)).round(4)}")
    print(f"  Output var  per group (sample 0): "
          f"{gn_grouped[0].var(axis=(1,2,3)).round(4)}")

    # --- Instance Norm ---
    inst = InstanceNorm(num_features=C)
    inst_out = inst.forward(x)
    print(f"\nInstance Norm:")
    print(f"  Normalizes over: spatial (H, W) per sample per channel")
    print(f"  Output mean (sample 0, ch 0): "
          f"{inst_out[0, 0].mean():.6f}")
    print(f"  Output var  (sample 0, ch 0): "
          f"{inst_out[0, 0].var():.4f}")

    # --- RMS Norm ---
    rms = RMSNorm(normalized_shape=C * H * W)
    rms_out = rms.forward(x_flat)
    print(f"\nRMS Norm:")
    print(f"  Normalizes over: all features per sample (no mean centering)")
    print(f"  Output RMS per sample: "
          f"{np.sqrt((rms_out**2).mean(axis=-1))[:4].round(4)}")
    print(f"  Output mean per sample (NOT zero): "
          f"{rms_out.mean(axis=-1)[:4].round(4)}")


def experiment_backward_pass():
    """Verify backward passes produce correct gradients via finite differences."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Gradient Check (Finite Differences)")
    print("=" * 60)

    np.random.seed(42)
    eps_fd = 1e-5  # finite difference epsilon

    def numerical_gradient(layer_fn, x, h=1e-5):
        """Compute numerical gradient via central differences."""
        grad = np.zeros_like(x)
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old_val = x[idx]

            x[idx] = old_val + h
            fp = layer_fn(x).sum()

            x[idx] = old_val - h
            fm = layer_fn(x).sum()

            grad[idx] = (fp - fm) / (2 * h)
            x[idx] = old_val
            it.iternext()
        return grad

    checks = []

    # Batch Norm gradient check
    x = np.random.randn(4, 8) * 3 + 2
    bn = BatchNorm(num_features=8)
    out = bn.forward(x)
    dout = np.ones_like(out)
    dx_analytical = bn.backward(dout)
    dx_numerical = numerical_gradient(lambda z: BatchNorm(8).forward(z), x.copy())
    err = np.max(np.abs(dx_analytical - dx_numerical))
    checks.append(("BatchNorm", err))

    # Layer Norm gradient check
    x = np.random.randn(4, 16) * 2 + 1
    ln = LayerNorm(normalized_shape=16)
    out = ln.forward(x)
    dout = np.ones_like(out)
    dx_analytical = ln.backward(dout)
    dx_numerical = numerical_gradient(lambda z: LayerNorm(16).forward(z), x.copy())
    err = np.max(np.abs(dx_analytical - dx_numerical))
    checks.append(("LayerNorm", err))

    # Group Norm gradient check
    x = np.random.randn(2, 8, 3, 3) * 2
    gn = GroupNorm(num_groups=2, num_channels=8)
    out = gn.forward(x)
    dout = np.ones_like(out)
    dx_analytical = gn.backward(dout)
    dx_numerical = numerical_gradient(
        lambda z: GroupNorm(2, 8).forward(z), x.copy())
    err = np.max(np.abs(dx_analytical - dx_numerical))
    checks.append(("GroupNorm", err))

    # Instance Norm gradient check
    x = np.random.randn(2, 4, 3, 3) * 3
    inst = InstanceNorm(num_features=4)
    out = inst.forward(x)
    dout = np.ones_like(out)
    dx_analytical = inst.backward(dout)
    dx_numerical = numerical_gradient(
        lambda z: InstanceNorm(4).forward(z), x.copy())
    err = np.max(np.abs(dx_analytical - dx_numerical))
    checks.append(("InstanceNorm", err))

    # RMS Norm gradient check
    x = np.random.randn(4, 12) * 2
    rms = RMSNorm(normalized_shape=12)
    out = rms.forward(x)
    dout = np.ones_like(out)
    dx_analytical = rms.backward(dout)
    dx_numerical = numerical_gradient(
        lambda z: RMSNorm(12).forward(z), x.copy())
    err = np.max(np.abs(dx_analytical - dx_numerical))
    checks.append(("RMSNorm", err))

    print()
    for name, err in checks:
        status = "PASS" if err < 1e-4 else "FAIL"
        print(f"  {name:15s}  max error: {err:.2e}  [{status}]")


def experiment_training_vs_inference():
    """Show BatchNorm behavior difference between training and inference."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: BatchNorm Training vs Inference Mode")
    print("=" * 60)

    np.random.seed(42)

    bn = BatchNorm(num_features=4)
    bn.training = True

    # Simulate several training batches to build up running stats
    print("\n  Training batches (running stats accumulate):")
    for i in range(5):
        x = np.random.randn(32, 4) * (i + 1) + i  # increasing scale/shift
        out = bn.forward(x)
        print(f"    Batch {i}: input mean={x.mean(axis=0)[:2].round(3)}, "
              f"output mean={out.mean(axis=0)[:2].round(5)}")

    print(f"\n  Running mean: {bn.running_mean.round(4)}")
    print(f"  Running var:  {bn.running_var.round(4)}")

    # Switch to eval mode
    bn.training = False
    x_test = np.random.randn(4, 4) * 3 + 2
    out_eval = bn.forward(x_test)
    print(f"\n  Inference mode (uses running stats):")
    print(f"    Input mean:  {x_test.mean(axis=0).round(4)}")
    print(f"    Output mean: {out_eval.mean(axis=0).round(4)}")
    print(f"    (Not exactly zero — running stats are approximate)")


def experiment_norm_dimensions():
    """Visualize which dimensions each normalization type operates on."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Normalization Dimension Summary")
    print("=" * 60)

    print("""
    For input tensor shape (N, C, H, W):

    +-----------+------------------+------------------------------+
    | Norm Type | Compute Mean/Var | Use Case                     |
    +-----------+------------------+------------------------------+
    | Batch     | over (N, H, W)   | CNN training (large batches) |
    | Layer     | over (C, H, W)   | Transformers, RNNs           |
    | Instance  | over (H, W)      | Style transfer               |
    | Group     | over (C/G, H, W) | Small batches, detection     |
    | RMS       | over (C, H, W)*  | LLMs (LLaMA, Gemma)         |
    +-----------+------------------+------------------------------+
    * RMS uses x^2 mean, no centering

    Key insight: They all do x_hat = (x - mu) / sigma (or x / RMS),
    they just differ in WHICH dimensions define the "population" for
    computing those statistics.
    """)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("NORMALIZATION TECHNIQUES FROM SCRATCH\n")

    experiment_normalization_comparison()
    experiment_backward_pass()
    experiment_training_vs_inference()
    experiment_norm_dimensions()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. All norms share the pattern: y = gamma * (x - mu) / sigma + beta.
2. BatchNorm: normalizes across batch — great for CNNs, bad for small batches.
3. LayerNorm: normalizes across features — batch-size independent, used in Transformers.
4. GroupNorm: compromise between Layer and Instance — works with any batch size.
5. InstanceNorm: per-channel per-sample — strips style information for transfer.
6. RMSNorm: drops mean centering for speed — used in modern LLMs.
7. BatchNorm has training/inference split (running stats); others don't need it.
8. Backward passes follow standard chain rule through the normalization graph.

Next: Dropout and regularization techniques.
""")

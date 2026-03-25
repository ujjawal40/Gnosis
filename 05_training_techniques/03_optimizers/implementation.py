"""
Optimizers from Scratch
========================

Every optimizer is a strategy for navigating the loss landscape. This module
builds each one from first principles in NumPy: from vanilla SGD through Adam
and AdamW, plus learning rate schedulers that modulate step sizes over time.

Key insight: all optimizers maintain some form of memory (momentum, squared
gradients) to make smarter updates than naive gradient descent.

Sections:
    1. Base Optimizer
    2. SGD (Vanilla)
    3. SGD with Momentum
    4. Nesterov Accelerated Gradient (NAG)
    5. AdaGrad
    6. RMSProp
    7. Adam
    8. AdamW (Decoupled Weight Decay)
    9. Learning Rate Schedulers
    10. Rosenbrock Trajectory Demo
    11. MLP Convergence Comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

# Create output directory for plots
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# =============================================================================
# Section 1: Base Optimizer
# =============================================================================

class Optimizer:
    """
    Base class for all optimizers.

    All optimizers share the same interface:
        - __init__: set hyperparameters
        - step(params, grads): update params in-place and return new params
        - reset(): clear internal state
    """

    def __init__(self, lr: float = 0.01):
        self.lr = lr
        self.t = 0  # step counter

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self):
        self.t = 0


# =============================================================================
# Section 2: SGD (Vanilla)
# =============================================================================

class SGD(Optimizer):
    """
    Vanilla Stochastic Gradient Descent.

    Update rule:
        θ_{t+1} = θ_t - η * ∇L(θ_t)

    The simplest possible optimizer: step in the negative gradient direction.
    Fast per-step but zig-zags in narrow valleys.
    """

    def __init__(self, lr: float = 0.01):
        super().__init__(lr)

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        self.t += 1
        return params - self.lr * grads

    def reset(self):
        super().reset()


# =============================================================================
# Section 3: SGD with Momentum
# =============================================================================

class SGDMomentum(Optimizer):
    """
    SGD with Momentum (Polyak, 1964).

    Update rule:
        v_t = β * v_{t-1} + ∇L(θ_t)
        θ_{t+1} = θ_t - η * v_t

    Momentum accumulates a velocity vector in directions of persistent gradient,
    damping oscillations in high-curvature directions. Think of a ball rolling
    downhill, gaining speed in consistent directions.
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.velocity = None

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        self.t += 1
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        self.velocity = self.momentum * self.velocity + grads
        return params - self.lr * self.velocity

    def reset(self):
        super().reset()
        self.velocity = None


# =============================================================================
# Section 4: Nesterov Accelerated Gradient (NAG)
# =============================================================================

class NesterovMomentum(Optimizer):
    """
    Nesterov Accelerated Gradient (Nesterov, 1983).

    Update rule:
        v_t = β * v_{t-1} + ∇L(θ_t - η * β * v_{t-1})
        θ_{t+1} = θ_t - η * v_t

    In practice (equivalent reformulation):
        v_t = β * v_{t-1} + g_t
        θ_{t+1} = θ_t - η * (β * v_t + g_t)

    The "look-ahead" trick: compute gradient at the approximate future position.
    This correction reduces overshooting, especially near minima.
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.velocity = None

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        self.t += 1
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        self.velocity = self.momentum * self.velocity + grads
        # Nesterov correction: step using the look-ahead velocity
        return params - self.lr * (self.momentum * self.velocity + grads)

    def reset(self):
        super().reset()
        self.velocity = None


# =============================================================================
# Section 5: AdaGrad
# =============================================================================

class AdaGrad(Optimizer):
    """
    Adaptive Gradient (Duchi et al., 2011).

    Update rule:
        G_t = G_{t-1} + g_t²
        θ_{t+1} = θ_t - η * g_t / (√G_t + ε)

    Adapts learning rate per-parameter: frequently updated parameters get
    smaller steps. Excellent for sparse data, but the monotonically
    increasing denominator can kill learning prematurely.
    """

    def __init__(self, lr: float = 0.01, eps: float = 1e-8):
        super().__init__(lr)
        self.eps = eps
        self.sum_sq_grads = None

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        self.t += 1
        if self.sum_sq_grads is None:
            self.sum_sq_grads = np.zeros_like(params)
        self.sum_sq_grads += grads ** 2
        return params - self.lr * grads / (np.sqrt(self.sum_sq_grads) + self.eps)

    def reset(self):
        super().reset()
        self.sum_sq_grads = None


# =============================================================================
# Section 6: RMSProp
# =============================================================================

class RMSProp(Optimizer):
    """
    Root Mean Square Propagation (Hinton, 2012, unpublished lecture).

    Update rule:
        E[g²]_t = ρ * E[g²]_{t-1} + (1 - ρ) * g_t²
        θ_{t+1} = θ_t - η * g_t / (√E[g²]_t + ε)

    Fixes AdaGrad's dying learning rate by using an exponential moving average
    of squared gradients instead of the full sum. The ρ parameter controls the
    window length of the moving average.
    """

    def __init__(self, lr: float = 0.001, rho: float = 0.9, eps: float = 1e-8):
        super().__init__(lr)
        self.rho = rho
        self.eps = eps
        self.ema_sq_grads = None

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        self.t += 1
        if self.ema_sq_grads is None:
            self.ema_sq_grads = np.zeros_like(params)
        self.ema_sq_grads = (self.rho * self.ema_sq_grads +
                             (1 - self.rho) * grads ** 2)
        return params - self.lr * grads / (np.sqrt(self.ema_sq_grads) + self.eps)

    def reset(self):
        super().reset()
        self.ema_sq_grads = None


# =============================================================================
# Section 7: Adam
# =============================================================================

class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Kingma & Ba, 2015).

    Update rule:
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t           (1st moment: mean)
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²           (2nd moment: variance)
        m̂_t = m_t / (1 - β₁^t)                          (bias correction)
        v̂_t = v_t / (1 - β₂^t)                          (bias correction)
        θ_{t+1} = θ_t - η * m̂_t / (√v̂_t + ε)

    Combines momentum (1st moment) with adaptive learning rates (2nd moment).
    Bias correction compensates for the zero initialization of moments.
    Default hyperparameters (β₁=0.9, β₂=0.999) work remarkably well.
    """

    def __init__(self, lr: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None  # 1st moment
        self.v = None  # 2nd moment

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        # Update biased moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self):
        super().reset()
        self.m = None
        self.v = None


# =============================================================================
# Section 8: AdamW (Decoupled Weight Decay)
# =============================================================================

class AdamW(Optimizer):
    """
    Adam with Decoupled Weight Decay (Loshchilov & Hutter, 2019).

    The key insight: L2 regularization ≠ weight decay for adaptive optimizers.

    Standard Adam with L2: gradient includes λθ term, which gets scaled by
    the adaptive learning rate → inconsistent regularization.

    AdamW: applies weight decay DIRECTLY to parameters, bypassing the
    adaptive scaling entirely:
        θ_{t+1} = (1 - η * λ) * θ_t - η * m̂_t / (√v̂_t + ε)

    This decoupling ensures consistent regularization regardless of gradient
    magnitude, which is why AdamW outperforms Adam+L2 in practice.
    """

    def __init__(self, lr: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8,
                 weight_decay: float = 0.01):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = None
        self.v = None

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        # Update biased moments (NO weight decay in gradient)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Decoupled weight decay: applied directly to params, not through gradient
        params = params * (1 - self.lr * self.weight_decay)

        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self):
        super().reset()
        self.m = None
        self.v = None


# =============================================================================
# Section 9: Learning Rate Schedulers
# =============================================================================

class LRScheduler:
    """Base class for learning rate schedulers."""

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.step_count = 0

    def step(self):
        """Advance one step and update the optimizer's learning rate."""
        self.step_count += 1
        self.optimizer.lr = self.get_lr()

    def get_lr(self) -> float:
        raise NotImplementedError


class StepLR(LRScheduler):
    """
    Step Decay: multiply LR by gamma every step_size steps.

    lr_t = base_lr * gamma^(floor(t / step_size))

    Simple and predictable. Common in older training recipes (e.g., ResNet
    drops LR by 10x at epochs 30 and 60).
    """

    def __init__(self, optimizer: Optimizer, step_size: int = 30,
                 gamma: float = 0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self) -> float:
        return self.base_lr * (self.gamma ** (self.step_count // self.step_size))


class CosineAnnealingLR(LRScheduler):
    """
    Cosine Annealing: smoothly decay LR following a half-cosine.

    lr_t = η_min + 0.5 * (η_max - η_min) * (1 + cos(π * t / T))

    No abrupt jumps. The slow start and slow end spend more time near
    the minimum learning rate, which helps fine-tune near convergence.
    """

    def __init__(self, optimizer: Optimizer, T_max: int = 100,
                 eta_min: float = 0.0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self) -> float:
        t = min(self.step_count, self.T_max)
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1 + np.cos(np.pi * t / self.T_max)
        )


class WarmupCosineAnnealingLR(LRScheduler):
    """
    Linear warmup followed by cosine annealing.

    Warmup phase (t < warmup_steps):
        lr_t = base_lr * (t / warmup_steps)

    Cosine phase (t >= warmup_steps):
        lr_t = η_min + 0.5 * (base_lr - η_min) * (1 + cos(π * progress))
        where progress = (t - warmup_steps) / (T_max - warmup_steps)

    Warmup prevents unstable early updates when loss landscape is rough.
    Standard in modern training (Transformers, ViTs, etc.).
    """

    def __init__(self, optimizer: Optimizer, warmup_steps: int = 10,
                 T_max: int = 100, eta_min: float = 0.0):
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self) -> float:
        if self.step_count < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (self.step_count / self.warmup_steps)
        # Cosine annealing
        progress = (self.step_count - self.warmup_steps) / (
            self.T_max - self.warmup_steps
        )
        progress = min(progress, 1.0)
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1 + np.cos(np.pi * progress)
        )


class OneCycleLR(LRScheduler):
    """
    One-Cycle Policy (Smith & Topin, 2019).

    Phase 1 (0 → pct_start): ramp LR from initial_lr to max_lr (linear)
    Phase 2 (pct_start → 1): anneal LR from max_lr to min_lr (cosine)

    The aggressive warmup phase explores broadly; the long annealing phase
    fine-tunes. Enables super-convergence: train in fewer epochs at higher LR.
    """

    def __init__(self, optimizer: Optimizer, max_lr: float = 0.01,
                 total_steps: int = 100, pct_start: float = 0.3,
                 div_factor: float = 25.0, final_div_factor: float = 1e4):
        super().__init__(optimizer)
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.initial_lr = max_lr / div_factor
        self.min_lr = max_lr / final_div_factor

    def get_lr(self) -> float:
        pct = self.step_count / self.total_steps
        pct = min(pct, 1.0)

        if pct <= self.pct_start:
            # Phase 1: linear warmup
            progress = pct / self.pct_start
            return self.initial_lr + (self.max_lr - self.initial_lr) * progress
        else:
            # Phase 2: cosine annealing
            progress = (pct - self.pct_start) / (1.0 - self.pct_start)
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress)
            )


# =============================================================================
# Section 10: Rosenbrock Trajectory Demo
# =============================================================================

def rosenbrock(xy):
    """
    Rosenbrock function: f(x, y) = (1 - x)² + 100(y - x²)²
    Global minimum at (1, 1) with f(1, 1) = 0.
    """
    x, y = xy[0], xy[1]
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosenbrock_grad(xy):
    """Analytical gradient of the Rosenbrock function."""
    x, y = xy[0], xy[1]
    dfdx = -2 * (1 - x) + 200 * (y - x ** 2) * (-2 * x)
    dfdy = 200 * (y - x ** 2)
    return np.array([dfdx, dfdy])


def demo_rosenbrock_trajectories():
    """
    Run all optimizers on the Rosenbrock function and plot their trajectories.

    This visualization reveals each optimizer's character:
    - SGD zig-zags in the narrow valley
    - Momentum overshoots but eventually converges faster
    - Adam adapts step sizes and navigates the valley smoothly
    """
    print("=" * 60)
    print("ROSENBROCK TRAJECTORY COMPARISON")
    print("=" * 60)

    start = np.array([-1.5, 1.5])
    n_steps = 2000

    optimizers = {
        "SGD (lr=0.001)": SGD(lr=0.001),
        "Momentum (lr=0.001)": SGDMomentum(lr=0.001, momentum=0.9),
        "Nesterov (lr=0.001)": NesterovMomentum(lr=0.001, momentum=0.9),
        "AdaGrad (lr=0.1)": AdaGrad(lr=0.1),
        "RMSProp (lr=0.001)": RMSProp(lr=0.001, rho=0.9),
        "Adam (lr=0.01)": Adam(lr=0.01),
        "AdamW (lr=0.01)": AdamW(lr=0.01, weight_decay=0.01),
    }

    # Collect trajectories
    trajectories = {}
    for name, opt in optimizers.items():
        opt.reset()
        pos = start.copy()
        path = [pos.copy()]
        for _ in range(n_steps):
            grad = rosenbrock_grad(pos)
            pos = opt.step(pos, grad)
            path.append(pos.copy())
        trajectories[name] = np.array(path)
        final_val = rosenbrock(pos)
        print(f"  {name:25s} → f = {final_val:.6f}  at ({pos[0]:.4f}, {pos[1]:.4f})")

    # Plot contours + trajectories
    fig, ax = plt.subplots(figsize=(12, 9))
    x_grid = np.linspace(-2, 2, 300)
    y_grid = np.linspace(-1, 3, 300)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2

    ax.contour(X, Y, Z, levels=np.logspace(-1, 3.5, 30), norm=LogNorm(),
               cmap="viridis", alpha=0.6)

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
              "#ff7f00", "#a65628", "#f781bf"]
    for (name, path), color in zip(trajectories.items(), colors):
        ax.plot(path[:, 0], path[:, 1], "-", color=color, alpha=0.8,
                linewidth=1.5, label=name)
        ax.plot(path[0, 0], path[0, 1], "o", color=color, markersize=6)
        ax.plot(path[-1, 0], path[-1, 1], "*", color=color, markersize=10)

    ax.plot(1, 1, "r*", markersize=20, zorder=10, label="Minimum (1,1)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Optimizer Trajectories on Rosenbrock Function")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "rosenbrock_trajectories.png"), dpi=150)
    plt.close()
    print(f"\n  Plot saved to {PLOT_DIR}/rosenbrock_trajectories.png")


# =============================================================================
# Section 11: MLP Convergence Comparison
# =============================================================================

class TinyMLP:
    """
    A small 2-layer MLP in NumPy for testing optimizer convergence.

    Architecture: input_dim → hidden_dim → output_dim
    Uses ReLU activation and cross-entropy loss.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 seed: int = 42):
        rng = np.random.RandomState(seed)
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = rng.randn(input_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, output_dim) * scale2
        self.b2 = np.zeros(output_dim)

    def get_params(self) -> np.ndarray:
        """Flatten all parameters into a single vector."""
        return np.concatenate([
            self.W1.ravel(), self.b1.ravel(),
            self.W2.ravel(), self.b2.ravel(),
        ])

    def set_params(self, flat: np.ndarray):
        """Restore parameters from a flat vector."""
        idx = 0
        W1_size = self.W1.size
        self.W1 = flat[idx:idx + W1_size].reshape(self.W1.shape)
        idx += W1_size
        b1_size = self.b1.size
        self.b1 = flat[idx:idx + b1_size]
        idx += b1_size
        W2_size = self.W2.size
        self.W2 = flat[idx:idx + W2_size].reshape(self.W2.shape)
        idx += W2_size
        self.b2 = flat[idx:idx + self.b2.size]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass returning logits."""
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.logits = self.a1 @ self.W2 + self.b2
        return self.logits

    def softmax(self, z: np.ndarray) -> np.ndarray:
        e = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    def cross_entropy_loss(self, logits: np.ndarray,
                           y: np.ndarray) -> float:
        """Compute cross-entropy loss. y is integer labels."""
        probs = self.softmax(logits)
        n = len(y)
        log_probs = -np.log(probs[np.arange(n), y] + 1e-15)
        return np.mean(log_probs)

    def backward(self, y: np.ndarray) -> np.ndarray:
        """Backward pass returning flat gradient vector."""
        n = len(y)
        probs = self.softmax(self.logits)
        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1
        dlogits /= n

        # Gradients for W2, b2
        dW2 = self.a1.T @ dlogits
        db2 = np.sum(dlogits, axis=0)

        # Backprop through ReLU
        da1 = dlogits @ self.W2.T
        dz1 = da1 * (self.z1 > 0).astype(float)

        # Gradients for W1, b1
        dW1 = self.X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return np.concatenate([dW1.ravel(), db1.ravel(),
                               dW2.ravel(), db2.ravel()])


def generate_spiral_data(n_samples: int = 300, n_classes: int = 3,
                         seed: int = 42) -> tuple:
    """Generate synthetic spiral dataset for classification."""
    rng = np.random.RandomState(seed)
    X = np.zeros((n_samples * n_classes, 2))
    y = np.zeros(n_samples * n_classes, dtype=int)
    for j in range(n_classes):
        ix = range(n_samples * j, n_samples * (j + 1))
        r = np.linspace(0.0, 1, n_samples)
        t = np.linspace(j * 4, (j + 1) * 4, n_samples) + rng.randn(n_samples) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return X, y


def demo_mlp_convergence():
    """
    Train a tiny MLP with each optimizer on synthetic spiral data.
    Compare convergence speed and final loss.
    """
    print("\n" + "=" * 60)
    print("MLP CONVERGENCE COMPARISON (Spiral Dataset)")
    print("=" * 60)

    X, y = generate_spiral_data(n_samples=200, n_classes=3, seed=42)
    n_epochs = 150
    batch_size = 64

    optimizer_configs = {
        "SGD": SGD(lr=0.1),
        "Momentum": SGDMomentum(lr=0.05, momentum=0.9),
        "Nesterov": NesterovMomentum(lr=0.05, momentum=0.9),
        "AdaGrad": AdaGrad(lr=0.05),
        "RMSProp": RMSProp(lr=0.005),
        "Adam": Adam(lr=0.005),
        "AdamW": AdamW(lr=0.005, weight_decay=0.01),
    }

    # Store initial params so every optimizer starts identically
    ref_model = TinyMLP(input_dim=2, hidden_dim=64, output_dim=3, seed=42)
    init_params = ref_model.get_params().copy()

    rng = np.random.RandomState(0)
    loss_curves = {}

    for name, opt in optimizer_configs.items():
        opt.reset()
        model = TinyMLP(input_dim=2, hidden_dim=64, output_dim=3)
        model.set_params(init_params.copy())
        losses = []

        for epoch in range(n_epochs):
            # Shuffle
            perm = rng.permutation(len(X))
            X_shuf, y_shuf = X[perm], y[perm]
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(X), batch_size):
                X_b = X_shuf[i:i + batch_size]
                y_b = y_shuf[i:i + batch_size]

                logits = model.forward(X_b)
                loss = model.cross_entropy_loss(logits, y_b)
                grads = model.backward(y_b)

                params = model.get_params()
                params = opt.step(params, grads)
                model.set_params(params)

                epoch_loss += loss
                n_batches += 1

            losses.append(epoch_loss / n_batches)

        loss_curves[name] = losses
        final = losses[-1]
        print(f"  {name:12s}: final_loss = {final:.4f}")

    # Plot convergence curves
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
              "#ff7f00", "#a65628", "#f781bf"]
    for (name, curve), color in zip(loss_curves.items(), colors):
        ax.plot(curve, label=name, color=color, linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Optimizer Convergence on Spiral Classification")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "mlp_convergence.png"), dpi=150)
    plt.close()
    print(f"\n  Plot saved to {PLOT_DIR}/mlp_convergence.png")

    # Plot LR scheduler demos
    demo_lr_schedulers()


def demo_lr_schedulers():
    """Visualize all learning rate schedules."""
    print("\n" + "=" * 60)
    print("LEARNING RATE SCHEDULER VISUALIZATION")
    print("=" * 60)

    total_steps = 200
    dummy_opt = SGD(lr=0.1)

    schedulers = {
        "StepLR (step=50, γ=0.5)": StepLR(
            SGD(lr=0.1), step_size=50, gamma=0.5),
        "CosineAnnealing": CosineAnnealingLR(
            SGD(lr=0.1), T_max=total_steps),
        "WarmupCosine (wu=20)": WarmupCosineAnnealingLR(
            SGD(lr=0.1), warmup_steps=20, T_max=total_steps),
        "OneCycleLR": OneCycleLR(
            SGD(lr=0.1), max_lr=0.1, total_steps=total_steps, pct_start=0.3),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    for (name, sched), color in zip(schedulers.items(), colors):
        lrs = []
        for _ in range(total_steps):
            sched.step()
            lrs.append(sched.optimizer.lr)
        ax.plot(lrs, label=name, color=color, linewidth=1.5)
        print(f"  {name:30s}: final_lr = {lrs[-1]:.6f}")

    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedules")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "lr_schedules.png"), dpi=150)
    plt.close()
    print(f"\n  Plot saved to {PLOT_DIR}/lr_schedules.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    demo_rosenbrock_trajectories()
    demo_mlp_convergence()

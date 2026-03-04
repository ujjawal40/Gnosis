"""
Activation Functions: From Scratch Implementation and Analysis
=============================================================

Every activation function implemented with its derivative, then tested
on real problems to reveal their properties.

All code uses only NumPy and Matplotlib. No frameworks.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

SAVE_DIR = Path(__file__).parent / "plots"
SAVE_DIR.mkdir(exist_ok=True)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Part 1: Activation Functions and Their Derivatives
# ---------------------------------------------------------------------------

def sigmoid(x):
    """Logistic sigmoid: 1 / (1 + exp(-x))."""
    # Numerically stable version
    pos = np.where(x >= 0, 1.0, 0.0)
    neg = 1.0 - pos
    z = np.zeros_like(x, dtype=np.float64)
    z[pos.astype(bool)] = np.exp(-x[pos.astype(bool)])
    z[neg.astype(bool)] = np.exp(x[neg.astype(bool)])
    result = np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))
    return result


def sigmoid_derivative(x):
    """sigma'(x) = sigma(x) * (1 - sigma(x))."""
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh_fn(x):
    """Hyperbolic tangent."""
    return np.tanh(x)


def tanh_derivative(x):
    """tanh'(x) = 1 - tanh^2(x)."""
    t = np.tanh(x)
    return 1.0 - t ** 2


def relu(x):
    """Rectified Linear Unit: max(0, x)."""
    return np.maximum(0.0, x)


def relu_derivative(x):
    """ReLU'(x) = 1 if x > 0, else 0 (subgradient at 0)."""
    return (x > 0).astype(np.float64)


def leaky_relu(x, alpha=0.01):
    """Leaky ReLU: x if x > 0, else alpha * x."""
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.01):
    """Leaky ReLU derivative."""
    return np.where(x > 0, 1.0, alpha)


def elu(x, alpha=1.0):
    """Exponential Linear Unit: x if x > 0, else alpha * (exp(x) - 1)."""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1.0))


def elu_derivative(x, alpha=1.0):
    """ELU'(x) = 1 if x > 0, else ELU(x) + alpha."""
    return np.where(x > 0, 1.0, elu(x, alpha) + alpha)


def selu(x):
    """Scaled Exponential Linear Unit with analytically derived constants."""
    lam = 1.0507009873554805
    alpha = 1.6732632423543772
    return lam * np.where(x > 0, x, alpha * (np.exp(x) - 1.0))


def selu_derivative(x):
    """SELU derivative."""
    lam = 1.0507009873554805
    alpha = 1.6732632423543772
    return lam * np.where(x > 0, 1.0, alpha * np.exp(x))


def gelu(x):
    """Gaussian Error Linear Unit: x * Phi(x)."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_derivative(x):
    """GELU derivative (using the tanh approximation)."""
    # d/dx [0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))]
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x ** 3)
    t = np.tanh(inner)
    sech2 = 1.0 - t ** 2
    inner_deriv = c * (1.0 + 3.0 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + t) + 0.5 * x * sech2 * inner_deriv


def swish(x):
    """Swish / SiLU: x * sigmoid(x)."""
    return x * sigmoid(x)


def swish_derivative(x):
    """Swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))."""
    s = sigmoid(x)
    return s + x * s * (1.0 - s)


def softmax(x, temperature=1.0):
    """
    Softmax with temperature scaling.
    x: array of shape (N,) or (batch, N)
    """
    z = x / temperature
    if z.ndim == 1:
        z = z - np.max(z)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)
    else:
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def softmax_jacobian(p):
    """
    Jacobian of softmax: diag(p) - p @ p.T
    p: softmax output, shape (K,)
    Returns: shape (K, K)
    """
    return np.diag(p) - np.outer(p, p)


# ---------------------------------------------------------------------------
# Part 2: Visualization — Plot Each Function and Its Derivative
# ---------------------------------------------------------------------------

def plot_all_activations():
    """Plot every activation function alongside its derivative."""
    x = np.linspace(-5, 5, 1000)

    activations = [
        ("Sigmoid", sigmoid, sigmoid_derivative),
        ("Tanh", tanh_fn, tanh_derivative),
        ("ReLU", relu, relu_derivative),
        ("Leaky ReLU (alpha=0.01)", lambda z: leaky_relu(z, 0.01),
         lambda z: leaky_relu_derivative(z, 0.01)),
        ("ELU (alpha=1)", lambda z: elu(z, 1.0), lambda z: elu_derivative(z, 1.0)),
        ("SELU", selu, selu_derivative),
        ("GELU", gelu, gelu_derivative),
        ("Swish / SiLU", swish, swish_derivative),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    fig.suptitle("Activation Functions and Their Derivatives", fontsize=16, y=0.98)

    for idx, (name, fn, fn_deriv) in enumerate(activations):
        ax = axes[idx // 2, idx % 2]
        y = fn(x)
        dy = fn_deriv(x)

        ax.plot(x, y, 'b-', linewidth=2.0, label=f'{name}')
        ax.plot(x, dy, 'r--', linewidth=1.5, label=f"Derivative")
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
        ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='-')
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_xlim(-5, 5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(SAVE_DIR / "01_all_activations.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {SAVE_DIR / '01_all_activations.png'}")


# ---------------------------------------------------------------------------
# Part 3: Demonstrate Saturation — Gradient Magnitude for Extreme Inputs
# ---------------------------------------------------------------------------

def demonstrate_saturation():
    """Show how gradient magnitude varies with input magnitude for each activation."""
    x = np.linspace(-10, 10, 2000)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Saturation: Gradient Magnitude vs Input", fontsize=15, y=0.98)

    # Panel 1: Saturating activations
    ax = axes[0, 0]
    ax.semilogy(x, np.abs(sigmoid_derivative(x)) + 1e-30, label='Sigmoid', linewidth=2)
    ax.semilogy(x, np.abs(tanh_derivative(x)) + 1e-30, label='Tanh', linewidth=2)
    ax.set_title("Saturating: Sigmoid & Tanh", fontweight='bold')
    ax.set_ylabel("|gradient| (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: ReLU family
    ax = axes[0, 1]
    ax.plot(x, relu_derivative(x), label='ReLU', linewidth=2)
    ax.plot(x, leaky_relu_derivative(x, 0.01), label='Leaky ReLU', linewidth=2, linestyle='--')
    ax.plot(x, elu_derivative(x), label='ELU', linewidth=2, linestyle='-.')
    ax.set_title("Non-Saturating: ReLU Family", fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Gradient after N layers (backprop simulation)
    ax = axes[1, 0]
    layers = np.arange(1, 51)
    # For sigmoid, max gradient per layer is 0.25; for tanh, 1.0; for ReLU, 1.0
    grad_sigmoid = 0.25 ** layers
    grad_tanh_best = 1.0 ** layers  # Best case
    grad_tanh_typical = 0.75 ** layers  # Typical case (inputs not perfectly at 0)
    grad_relu = 1.0 ** layers  # Active neurons

    ax.semilogy(layers, grad_sigmoid, label='Sigmoid (max)', linewidth=2)
    ax.semilogy(layers, grad_tanh_typical, label='Tanh (typical)', linewidth=2)
    ax.semilogy(layers, grad_relu, label='ReLU (active)', linewidth=2)
    ax.set_title("Gradient Magnitude After N Layers", fontweight='bold')
    ax.set_xlabel("Number of layers")
    ax.set_ylabel("Gradient magnitude (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Modern activations — gradient near zero
    ax = axes[1, 1]
    x_zoom = np.linspace(-3, 3, 1000)
    ax.plot(x_zoom, gelu_derivative(x_zoom), label='GELU', linewidth=2)
    ax.plot(x_zoom, swish_derivative(x_zoom), label='Swish', linewidth=2, linestyle='--')
    ax.plot(x_zoom, relu_derivative(x_zoom), label='ReLU', linewidth=2, alpha=0.7)
    ax.set_title("Gradients Near Zero: Modern vs ReLU", fontweight='bold')
    ax.set_xlabel("Input x")
    ax.set_ylabel("Gradient")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(SAVE_DIR / "02_saturation_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {SAVE_DIR / '02_saturation_analysis.png'}")


# ---------------------------------------------------------------------------
# Part 4: Dying ReLU Demonstration
# ---------------------------------------------------------------------------

class SimpleNetwork:
    """
    A minimal fully-connected network for demonstration.
    Architecture: input -> hidden1 -> hidden2 -> output
    """

    def __init__(self, layer_sizes, activation='relu', lr=0.01):
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.activation_name = activation
        self.n_layers = len(layer_sizes) - 1

        # He initialization for ReLU, Xavier for others
        self.weights = []
        self.biases = []
        for i in range(self.n_layers):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            if activation in ('relu', 'leaky_relu'):
                std = np.sqrt(2.0 / fan_in)
            else:
                std = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(np.random.randn(fan_in, fan_out) * std)
            self.biases.append(np.zeros(fan_out))

    def _activate(self, x):
        if self.activation_name == 'relu':
            return relu(x)
        elif self.activation_name == 'leaky_relu':
            return leaky_relu(x)
        elif self.activation_name == 'sigmoid':
            return sigmoid(x)
        elif self.activation_name == 'tanh':
            return tanh_fn(x)
        elif self.activation_name == 'elu':
            return elu(x)
        elif self.activation_name == 'gelu':
            return gelu(x)
        elif self.activation_name == 'swish':
            return swish(x)
        elif self.activation_name == 'selu':
            return selu(x)

    def _activate_derivative(self, x):
        if self.activation_name == 'relu':
            return relu_derivative(x)
        elif self.activation_name == 'leaky_relu':
            return leaky_relu_derivative(x)
        elif self.activation_name == 'sigmoid':
            return sigmoid_derivative(x)
        elif self.activation_name == 'tanh':
            return tanh_derivative(x)
        elif self.activation_name == 'elu':
            return elu_derivative(x)
        elif self.activation_name == 'gelu':
            return gelu_derivative(x)
        elif self.activation_name == 'swish':
            return swish_derivative(x)
        elif self.activation_name == 'selu':
            return selu_derivative(x)

    def forward(self, x):
        """Forward pass. Stores pre-activations and activations for backprop."""
        self.pre_activations = []
        self.activations = [x]

        current = x
        for i in range(self.n_layers - 1):
            z = current @ self.weights[i] + self.biases[i]
            self.pre_activations.append(z)
            current = self._activate(z)
            self.activations.append(current)

        # Output layer: no activation (linear)
        z = current @ self.weights[-1] + self.biases[-1]
        self.pre_activations.append(z)
        self.activations.append(z)
        return z

    def backward(self, y_true, loss_type='mse'):
        """Backward pass with MSE or cross-entropy loss."""
        batch_size = y_true.shape[0]
        output = self.activations[-1]

        if loss_type == 'mse':
            # dL/d(output) = 2*(output - y_true) / batch_size
            delta = 2.0 * (output - y_true) / batch_size
        elif loss_type == 'cross_entropy':
            # For softmax + cross-entropy, the gradient is simply (p - y)
            probs = softmax(output)
            delta = (probs - y_true) / batch_size
        else:
            delta = 2.0 * (output - y_true) / batch_size

        grad_weights = []
        grad_biases = []

        for i in range(self.n_layers - 1, -1, -1):
            # Gradient for weights and biases at layer i
            gw = self.activations[i].T @ delta
            gb = np.sum(delta, axis=0)
            grad_weights.insert(0, gw)
            grad_biases.insert(0, gb)

            if i > 0:
                # Propagate delta to previous layer
                delta = delta @ self.weights[i].T
                delta = delta * self._activate_derivative(self.pre_activations[i - 1])

        # Update
        for i in range(self.n_layers):
            self.weights[i] -= self.lr * grad_weights[i]
            self.biases[i] -= self.lr * grad_biases[i]

    def train_step(self, x, y, loss_type='mse'):
        """One forward-backward pass. Returns loss."""
        output = self.forward(x)
        if loss_type == 'mse':
            loss = np.mean((output - y) ** 2)
        elif loss_type == 'cross_entropy':
            probs = softmax(output)
            loss = -np.mean(np.sum(y * np.log(probs + 1e-12), axis=1))
        else:
            loss = np.mean((output - y) ** 2)
        self.backward(y, loss_type)
        return loss

    def count_dead_neurons(self, x):
        """Count neurons that output zero for ALL inputs in x."""
        self.forward(x)
        dead_counts = []
        for i, act in enumerate(self.activations[1:-1]):
            # A neuron is dead if it outputs 0 (or very near 0) for every sample
            dead = np.all(np.abs(act) < 1e-8, axis=0)
            dead_counts.append(np.sum(dead))
        return dead_counts


def demonstrate_dying_relu():
    """Train a network and show how ReLU neurons die during training."""
    # Create a classification dataset (spiral)
    N = 200  # points per class
    D = 2    # dimensions
    K = 3    # classes
    X = np.zeros((N * K, D))
    y = np.zeros(N * K, dtype=int)

    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    # One-hot encode
    Y = np.zeros((N * K, K))
    Y[np.arange(N * K), y] = 1

    # --- Experiment: ReLU with high learning rate causes dying neurons ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Dying ReLU Demonstration", fontsize=15, y=0.98)

    # Train with ReLU and high LR (causes dying neurons)
    net_relu_high_lr = SimpleNetwork([2, 128, 128, K], activation='relu', lr=0.1)
    # Train with ReLU and normal LR
    net_relu_normal = SimpleNetwork([2, 128, 128, K], activation='relu', lr=0.01)
    # Train with Leaky ReLU and high LR (no dying)
    net_leaky = SimpleNetwork([2, 128, 128, K], activation='leaky_relu', lr=0.1)

    epochs = 300
    dead_relu_high = []
    dead_relu_normal = []
    dead_leaky = []
    losses_relu_high = []
    losses_relu_normal = []
    losses_leaky = []

    for epoch in range(epochs):
        l1 = net_relu_high_lr.train_step(X, Y, loss_type='cross_entropy')
        l2 = net_relu_normal.train_step(X, Y, loss_type='cross_entropy')
        l3 = net_leaky.train_step(X, Y, loss_type='cross_entropy')

        losses_relu_high.append(l1)
        losses_relu_normal.append(l2)
        losses_leaky.append(l3)

        d1 = net_relu_high_lr.count_dead_neurons(X)
        d2 = net_relu_normal.count_dead_neurons(X)
        d3 = net_leaky.count_dead_neurons(X)

        dead_relu_high.append(sum(d1))
        dead_relu_normal.append(sum(d2))
        dead_leaky.append(sum(d3))

    # Plot dead neuron count over training
    ax = axes[0, 0]
    ax.plot(dead_relu_high, label='ReLU (lr=0.1)', linewidth=2, color='red')
    ax.plot(dead_relu_normal, label='ReLU (lr=0.01)', linewidth=2, color='blue')
    ax.plot(dead_leaky, label='Leaky ReLU (lr=0.1)', linewidth=2, color='green')
    ax.set_title("Dead Neurons During Training", fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Number of dead neurons (out of 256)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot training loss
    ax = axes[0, 1]
    ax.plot(losses_relu_high, label='ReLU (lr=0.1)', linewidth=2, color='red')
    ax.plot(losses_relu_normal, label='ReLU (lr=0.01)', linewidth=2, color='blue')
    ax.plot(losses_leaky, label='Leaky ReLU (lr=0.1)', linewidth=2, color='green')
    ax.set_title("Training Loss", fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Visualize activation patterns — heatmap of hidden layer outputs
    ax = axes[1, 0]
    net_relu_high_lr.forward(X)
    # Show first hidden layer activations for the high-LR ReLU network
    h1 = net_relu_high_lr.activations[1]  # shape (N*K, 128)
    activity = np.mean(np.abs(h1) > 1e-8, axis=0)  # fraction of inputs each neuron is active for
    ax.bar(range(len(activity)), np.sort(activity)[::-1], color='steelblue', width=1.0)
    ax.axhline(y=0.01, color='red', linestyle='--', label='Dead threshold')
    ax.set_title("ReLU (lr=0.1): Neuron Activity (Layer 1)", fontweight='bold')
    ax.set_xlabel("Neuron (sorted by activity)")
    ax.set_ylabel("Fraction of inputs where neuron is active")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Same for Leaky ReLU
    ax = axes[1, 1]
    net_leaky.forward(X)
    h1_leaky = net_leaky.activations[1]
    activity_leaky = np.mean(np.abs(h1_leaky) > 1e-8, axis=0)
    ax.bar(range(len(activity_leaky)), np.sort(activity_leaky)[::-1], color='forestgreen', width=1.0)
    ax.axhline(y=0.01, color='red', linestyle='--', label='Dead threshold')
    ax.set_title("Leaky ReLU (lr=0.1): Neuron Activity (Layer 1)", fontweight='bold')
    ax.set_xlabel("Neuron (sorted by activity)")
    ax.set_ylabel("Fraction of inputs where neuron is active")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(SAVE_DIR / "03_dying_relu.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {SAVE_DIR / '03_dying_relu.png'}")


# ---------------------------------------------------------------------------
# Part 5: Compare Activations — Same Network, Different Activations
# ---------------------------------------------------------------------------

def compare_activations_convergence():
    """Train the same network architecture with different activations and compare."""
    # Generate a non-linear regression dataset
    np.random.seed(42)
    N = 500
    X = np.random.randn(N, 4)
    # Target: a genuinely non-linear function
    y_true = (np.sin(X[:, 0] * X[:, 1]) + np.cos(X[:, 2]) + 0.5 * X[:, 3] ** 2
              + 0.1 * np.random.randn(N))
    y_true = y_true.reshape(-1, 1)

    # Normalize inputs and outputs
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    y_mean, y_std = y_true.mean(), y_true.std()
    y_true = (y_true - y_mean) / (y_std + 1e-8)

    activations_to_test = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'gelu', 'swish']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf']

    results = {}
    epochs = 500

    for act_name in activations_to_test:
        np.random.seed(42)
        net = SimpleNetwork([4, 64, 64, 1], activation=act_name, lr=0.005)
        losses = []
        for epoch in range(epochs):
            loss = net.train_step(X, y_true, loss_type='mse')
            losses.append(loss)
        results[act_name] = losses

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Activation Function Comparison: Convergence on Non-Linear Regression",
                 fontsize=14, y=1.02)

    # Full training curve
    ax = axes[0]
    for (name, losses), color in zip(results.items(), colors):
        ax.plot(losses, label=name, linewidth=1.8, color=color)
    ax.set_title("Training Loss (Full)", fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Zoomed in (last 200 epochs)
    ax = axes[1]
    for (name, losses), color in zip(results.items(), colors):
        ax.plot(range(300, epochs), losses[300:], label=name, linewidth=1.8, color=color)
    ax.set_title("Training Loss (Epoch 300-500, Zoomed)", fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(SAVE_DIR / "04_activation_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {SAVE_DIR / '04_activation_comparison.png'}")

    # Print final losses
    print("\n  Final MSE Loss after 500 epochs:")
    print("  " + "-" * 35)
    for name in activations_to_test:
        print(f"  {name:>12s}: {results[name][-1]:.6f}")


# ---------------------------------------------------------------------------
# Part 6: Softmax with Temperature
# ---------------------------------------------------------------------------

def demonstrate_softmax_temperature():
    """Show how temperature scaling affects the softmax output distribution."""
    logits = np.array([2.0, 1.0, 0.5, -0.5, -1.0])
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0, 20.0]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Softmax Temperature Scaling", fontsize=15, y=0.98)

    x_labels = [f"z={z}" for z in logits]
    x_pos = np.arange(len(logits))

    for idx, T in enumerate(temperatures):
        ax = axes[idx // 3, idx % 3]
        probs = softmax(logits, temperature=T)

        bars = ax.bar(x_pos, probs, color='steelblue', alpha=0.8, edgecolor='navy')
        ax.set_title(f"T = {T}", fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis='y')

        # Annotate bars with probability values
        for bar, p in zip(bars, probs):
            if p > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                        f'{p:.3f}', ha='center', va='bottom', fontsize=9)

        # Show entropy
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        max_entropy = np.log(len(logits))
        ax.text(0.95, 0.85, f'H = {entropy:.2f}\n(max={max_entropy:.2f})',
                transform=ax.transAxes, ha='right', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(SAVE_DIR / "05_softmax_temperature.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {SAVE_DIR / '05_softmax_temperature.png'}")

    # Also demonstrate the Jacobian
    print("\n  Softmax Jacobian verification:")
    p = softmax(logits)
    J = softmax_jacobian(p)
    print(f"  Logits: {logits}")
    print(f"  Softmax output: {np.round(p, 4)}")
    print(f"  Jacobian (5x5):\n  {np.array2string(J, precision=4, prefix='  ')}")
    print(f"  Row sums of Jacobian (should be ~0): {np.round(J.sum(axis=1), 8)}")


# ---------------------------------------------------------------------------
# Main: Run All Experiments
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  ACTIVATION FUNCTIONS: Implementation and Analysis")
    print("=" * 70)

    print("\n[1/5] Plotting all activation functions and derivatives...")
    plot_all_activations()

    print("\n[2/5] Analyzing saturation and gradient flow...")
    demonstrate_saturation()

    print("\n[3/5] Demonstrating dying ReLU problem...")
    demonstrate_dying_relu()

    print("\n[4/5] Comparing activations on convergence task...")
    compare_activations_convergence()

    print("\n[5/5] Demonstrating softmax temperature scaling...")
    demonstrate_softmax_temperature()

    print("\n" + "=" * 70)
    print(f"  All plots saved to: {SAVE_DIR.resolve()}")
    print("=" * 70)

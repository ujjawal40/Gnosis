"""
Multi-Layer Perceptron from Scratch
====================================

A complete neural network framework using only NumPy.
No PyTorch, no TensorFlow — just matrix multiplication and calculus.

This implementation includes:
- Dense layers with forward and backward passes
- Weight initialization (Xavier, He)
- Batch normalization
- Dropout
- SGD with momentum and Adam optimizer
- Training on synthetic datasets

Every computation is explicit. Nothing is hidden.
"""

import numpy as np


# ==============================================================================
# Part 1: Activation Functions (Vectorized)
# ==============================================================================

class ReLU:
    """f(x) = max(0, x). Gradient: 1 if x > 0, else 0."""

    def forward(self, x):
        self.mask = (x > 0).astype(float)
        return x * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask


class Sigmoid:
    """f(x) = 1/(1+e^-x). Gradient: f(x)(1-f(x))."""

    def forward(self, x):
        self.out = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return self.out

    def backward(self, grad_output):
        return grad_output * self.out * (1 - self.out)


class Tanh:
    """f(x) = tanh(x). Gradient: 1 - tanh²(x)."""

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad_output):
        return grad_output * (1 - self.out ** 2)


class Softmax:
    """Softmax for multi-class. Used with cross-entropy loss."""

    def forward(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.out = e / np.sum(e, axis=1, keepdims=True)
        return self.out

    def backward(self, grad_output):
        # When paired with cross-entropy, the combined gradient is (softmax - y)
        # which is passed directly. This backward is for standalone use.
        return grad_output  # Simplified: combined with loss


# ==============================================================================
# Part 2: Layers
# ==============================================================================

class Dense:
    """
    Fully connected layer: y = xW + b

    Forward: z = xW + b  (matrix multiply + broadcast bias)
    Backward: dL/dW = x^T · dL/dz,  dL/db = sum(dL/dz),  dL/dx = dL/dz · W^T
    """

    def __init__(self, n_inputs, n_outputs, init='xavier'):
        """
        Initialization matters enormously:
        - Random: var(output) grows with n_inputs -> exploding activations
        - Xavier: var(W) = 1/n_in -> keeps variance ~1 (good for sigmoid/tanh)
        - He: var(W) = 2/n_in -> accounts for ReLU killing half the values
        """
        if init == 'xavier':
            scale = np.sqrt(1.0 / n_inputs)
        elif init == 'he':
            scale = np.sqrt(2.0 / n_inputs)
        elif init == 'random':
            scale = 0.01
        else:
            raise ValueError(f"Unknown init: {init}")

        self.W = np.random.randn(n_inputs, n_outputs) * scale
        self.b = np.zeros((1, n_outputs))

        # Gradient storage
        self.dW = None
        self.db = None

        # For optimizers (momentum/Adam state)
        self.mW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x  # Cache for backward
        return x @ self.W + self.b

    def backward(self, grad_output):
        n = self.x.shape[0]
        self.dW = self.x.T @ grad_output / n
        self.db = np.sum(grad_output, axis=0, keepdims=True) / n
        return grad_output @ self.W.T


class BatchNorm:
    """
    Batch Normalization: normalize activations to zero mean, unit variance.

    Forward (training):  x_hat = (x - μ_batch) / √(σ²_batch + ε)
                         y = γ · x_hat + β

    Forward (eval):      use running mean/variance instead of batch stats

    Why it works (several theories):
    1. Reduces internal covariate shift (original claim)
    2. Smooths the loss landscape (more recent understanding)
    3. Acts as a regularizer (noise from batch statistics)
    """

    def __init__(self, n_features, momentum=0.9, eps=1e-5):
        self.gamma = np.ones((1, n_features))
        self.beta = np.zeros((1, n_features))
        self.eps = eps
        self.momentum = momentum

        # Running statistics for eval mode
        self.running_mean = np.zeros((1, n_features))
        self.running_var = np.ones((1, n_features))

        # Gradient storage
        self.dgamma = None
        self.dbeta = None

        self.training = True

        # Optimizer state
        self.mgamma = np.zeros_like(self.gamma)
        self.mbeta = np.zeros_like(self.beta)
        self.vgamma = np.zeros_like(self.gamma)
        self.vbeta = np.zeros_like(self.beta)

    def forward(self, x):
        if self.training:
            self.mean = np.mean(x, axis=0, keepdims=True)
            self.var = np.var(x, axis=0, keepdims=True)

            # Update running stats
            self.running_mean = (self.momentum * self.running_mean +
                                 (1 - self.momentum) * self.mean)
            self.running_var = (self.momentum * self.running_var +
                                (1 - self.momentum) * self.var)
        else:
            self.mean = self.running_mean
            self.var = self.running_var

        self.x = x
        self.x_hat = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_hat + self.beta

    def backward(self, grad_output):
        n = grad_output.shape[0]
        self.dgamma = np.sum(grad_output * self.x_hat, axis=0, keepdims=True) / n
        self.dbeta = np.sum(grad_output, axis=0, keepdims=True) / n

        # Backprop through normalization (complex but well-defined)
        dx_hat = grad_output * self.gamma
        inv_std = 1.0 / np.sqrt(self.var + self.eps)

        dx = (1.0 / n) * inv_std * (
            n * dx_hat
            - np.sum(dx_hat, axis=0, keepdims=True)
            - self.x_hat * np.sum(dx_hat * self.x_hat, axis=0, keepdims=True)
        )
        return dx


class Dropout:
    """
    Dropout: randomly zero out neurons during training.

    Interpretation: training an ensemble of 2^n sub-networks simultaneously.
    At test time, use all neurons but scale by (1-p) — or use inverted dropout
    (scale during training) so test time is unchanged.
    """

    def __init__(self, p=0.5):
        self.p = p  # probability of dropping
        self.training = True

    def forward(self, x):
        if self.training:
            # Inverted dropout: scale during training so test is unchanged
            self.mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
            return x * self.mask
        return x

    def backward(self, grad_output):
        if self.training:
            return grad_output * self.mask
        return grad_output


# ==============================================================================
# Part 3: Loss Functions (Vectorized)
# ==============================================================================

class MSELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        n = self.y_pred.shape[0]
        return 2 * (self.y_pred - self.y_true) / n


class BCELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.y_true = y_true
        return -np.mean(y_true * np.log(self.y_pred) +
                        (1 - y_true) * np.log(1 - self.y_pred))

    def backward(self):
        n = self.y_pred.shape[0]
        return (-(self.y_true / self.y_pred) +
                (1 - self.y_true) / (1 - self.y_pred)) / n


class CrossEntropyLoss:
    """Combined softmax + cross-entropy for numerical stability."""

    def forward(self, logits, y_true_onehot):
        # Stable softmax
        e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = e / np.sum(e, axis=1, keepdims=True)
        self.y_true = y_true_onehot

        # Cross entropy
        return -np.mean(np.sum(y_true_onehot * np.log(self.probs + 1e-15), axis=1))

    def backward(self):
        # The beautiful gradient: softmax output - one-hot target
        return (self.probs - self.y_true) / self.y_true.shape[0]


# ==============================================================================
# Part 4: Optimizers
# ==============================================================================

class SGD:
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def step(self, layers):
        for layer in layers:
            if not isinstance(layer, Dense):
                continue
            if self.momentum > 0:
                layer.mW = self.momentum * layer.mW - self.lr * layer.dW
                layer.mb = self.momentum * layer.mb - self.lr * layer.db
                layer.W += layer.mW
                layer.b += layer.mb
            else:
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db


class Adam:
    """
    Adam: Adaptive Moment Estimation.

    Combines momentum (first moment) with RMSProp (second moment).
    Arguably the most widely used optimizer in deep learning.

    m = β1*m + (1-β1)*grad        (momentum)
    v = β2*v + (1-β2)*grad²       (RMSProp)
    m_hat = m / (1-β1^t)          (bias correction)
    v_hat = v / (1-β2^t)          (bias correction)
    w -= lr * m_hat / (√v_hat + ε)
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def step(self, layers):
        self.t += 1
        for layer in layers:
            if isinstance(layer, Dense):
                params = [('W', layer.dW), ('b', layer.db)]
                for name, grad in params:
                    w = getattr(layer, name)
                    m = getattr(layer, f'm{name}')
                    v = getattr(layer, f'v{name}')

                    m = self.beta1 * m + (1 - self.beta1) * grad
                    v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

                    m_hat = m / (1 - self.beta1 ** self.t)
                    v_hat = v / (1 - self.beta2 ** self.t)

                    setattr(layer, name, w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
                    setattr(layer, f'm{name}', m)
                    setattr(layer, f'v{name}', v)

            elif isinstance(layer, BatchNorm):
                for name in ['gamma', 'beta']:
                    w = getattr(layer, name)
                    grad = getattr(layer, f'd{name}')
                    m = getattr(layer, f'm{name}')
                    v = getattr(layer, f'v{name}')

                    m = self.beta1 * m + (1 - self.beta1) * grad
                    v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

                    m_hat = m / (1 - self.beta1 ** self.t)
                    v_hat = v / (1 - self.beta2 ** self.t)

                    setattr(layer, name, w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
                    setattr(layer, f'm{name}', m)
                    setattr(layer, f'v{name}', v)


# ==============================================================================
# Part 5: The MLP Class
# ==============================================================================

class MLP:
    """
    Multi-Layer Perceptron: stack of Dense layers with activations.

    This is the complete neural network. It handles:
    - Forward pass (input -> hidden layers -> output)
    - Backward pass (backpropagation through all layers)
    - Training loop with mini-batches
    """

    def __init__(self, layer_dims, activations=None, init='xavier',
                 use_batchnorm=False, dropout_rate=0.0):
        """
        Args:
            layer_dims: list of layer sizes, e.g. [2, 64, 32, 1]
            activations: list of activation names for each layer transition
            init: weight initialization method
            use_batchnorm: whether to add batch norm after each hidden layer
            dropout_rate: dropout probability (0 = no dropout)
        """
        n_layers = len(layer_dims) - 1

        if activations is None:
            activations = ['relu'] * (n_layers - 1) + ['sigmoid']

        self.layers = []

        for i in range(n_layers):
            # Dense layer
            self.layers.append(Dense(layer_dims[i], layer_dims[i + 1], init=init))

            # Batch norm (before activation, after dense)
            if use_batchnorm and i < n_layers - 1:
                self.layers.append(BatchNorm(layer_dims[i + 1]))

            # Activation
            act_name = activations[i]
            if act_name == 'relu':
                self.layers.append(ReLU())
            elif act_name == 'sigmoid':
                self.layers.append(Sigmoid())
            elif act_name == 'tanh':
                self.layers.append(Tanh())
            elif act_name == 'softmax':
                self.layers.append(Softmax())
            elif act_name == 'linear':
                pass  # No activation
            else:
                raise ValueError(f"Unknown activation: {act_name}")

            # Dropout (after activation)
            if dropout_rate > 0 and i < n_layers - 1:
                self.layers.append(Dropout(dropout_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def train_mode(self):
        for layer in self.layers:
            if isinstance(layer, (BatchNorm, Dropout)):
                layer.training = True

    def eval_mode(self):
        for layer in self.layers:
            if isinstance(layer, (BatchNorm, Dropout)):
                layer.training = False

    def count_parameters(self):
        count = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                count += layer.W.size + layer.b.size
            elif isinstance(layer, BatchNorm):
                count += layer.gamma.size + layer.beta.size
        return count


# ==============================================================================
# Part 6: Data Generation
# ==============================================================================

def make_moons(n_samples=200, noise=0.1, random_state=42):
    """Generate two interleaving half circles (non-linearly separable)."""
    np.random.seed(random_state)
    n = n_samples // 2

    # Upper moon
    theta1 = np.linspace(0, np.pi, n)
    x1 = np.column_stack([np.cos(theta1), np.sin(theta1)])

    # Lower moon (shifted)
    theta2 = np.linspace(0, np.pi, n)
    x2 = np.column_stack([1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5])

    X = np.vstack([x1, x2]) + np.random.randn(n_samples, 2) * noise
    y = np.hstack([np.zeros(n), np.ones(n)])

    # Shuffle
    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]


def make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42):
    """Generate two concentric circles (non-linearly separable)."""
    np.random.seed(random_state)
    n = n_samples // 2

    # Outer circle
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    outer = np.column_stack([np.cos(theta), np.sin(theta)])

    # Inner circle
    inner = factor * np.column_stack([np.cos(theta), np.sin(theta)])

    X = np.vstack([outer, inner]) + np.random.randn(n_samples, 2) * noise
    y = np.hstack([np.zeros(n), np.ones(n)])

    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]


# ==============================================================================
# Part 7: Training Loop
# ==============================================================================

def train(model, X, y, loss_fn, optimizer, epochs=200, batch_size=32, verbose=True):
    """Standard mini-batch training loop."""
    n = X.shape[0]
    history = []

    for epoch in range(epochs):
        model.train_mode()

        # Shuffle data each epoch
        idx = np.random.permutation(n)
        X_shuffled = X[idx]
        y_shuffled = y[idx]

        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            if len(y_batch.shape) == 1:
                y_batch = y_batch.reshape(-1, 1)

            # Forward
            pred = model.forward(X_batch)
            loss = loss_fn.forward(pred, y_batch)

            # Backward
            grad = loss_fn.backward()
            model.backward(grad)

            # Update
            optimizer.step(model.layers)

            epoch_loss += loss
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        history.append(avg_loss)

        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            # Compute accuracy
            model.eval_mode()
            pred = model.forward(X)
            if pred.shape[1] == 1:
                acc = np.mean((pred.flatten() > 0.5) == y)
            else:
                acc = np.mean(np.argmax(pred, axis=1) == y)
            print(f"  Epoch {epoch:4d} | Loss: {avg_loss:.6f} | Accuracy: {acc * 100:.1f}%")

    return history


# ==============================================================================
# Part 8: Experiments
# ==============================================================================

def experiment_basic():
    """Train MLP on moons dataset — the fundamental test."""
    print("=" * 60)
    print("EXPERIMENT 1: MLP on Moons Dataset")
    print("=" * 60)

    X, y = make_moons(n_samples=300, noise=0.15)
    print(f"Data: {X.shape[0]} samples, 2 features, 2 classes\n")

    model = MLP([2, 16, 8, 1], activations=['relu', 'relu', 'sigmoid'], init='he')
    print(f"Network: 2 -> 16 -> 8 -> 1 ({model.count_parameters()} parameters)")
    print(f"Activation: ReLU (hidden), Sigmoid (output)")
    print(f"Loss: Binary Cross-Entropy")
    print(f"Optimizer: Adam (lr=0.01)\n")

    loss_fn = BCELoss()
    optimizer = Adam(lr=0.01)
    history = train(model, X, y, loss_fn, optimizer, epochs=300, batch_size=32)

    # Final evaluation
    model.eval_mode()
    pred = model.forward(X)
    acc = np.mean((pred.flatten() > 0.5) == y)
    print(f"\nFinal accuracy: {acc * 100:.1f}%")


def experiment_depth():
    """Compare networks of different depths on the same problem."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Effect of Depth")
    print("Does deeper = better?")
    print("=" * 60)

    X, y = make_circles(n_samples=300, noise=0.08)
    print(f"Data: Concentric circles (300 samples)\n")

    configs = [
        ("1 hidden (16)", [2, 16, 1]),
        ("2 hidden (16, 8)", [2, 16, 8, 1]),
        ("4 hidden (32,16,8,4)", [2, 32, 16, 8, 4, 1]),
    ]

    for name, dims in configs:
        print(f"\n--- {name} ---")
        np.random.seed(42)
        n_hidden = len(dims) - 2
        acts = ['relu'] * n_hidden + ['sigmoid']
        model = MLP(dims, activations=acts, init='he')
        loss_fn = BCELoss()
        optimizer = Adam(lr=0.01)
        train(model, X, y, loss_fn, optimizer, epochs=300, batch_size=32, verbose=True)

        model.eval_mode()
        pred = model.forward(X)
        acc = np.mean((pred.flatten() > 0.5) == y)
        print(f"  Final: {acc * 100:.1f}% ({model.count_parameters()} params)")


def experiment_init():
    """Show why initialization matters."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Effect of Initialization")
    print("=" * 60)

    X, y = make_moons(n_samples=300, noise=0.15)

    for init_name in ['random', 'xavier', 'he']:
        print(f"\n--- {init_name.upper()} initialization ---")
        np.random.seed(42)
        model = MLP([2, 32, 16, 1], activations=['relu', 'relu', 'sigmoid'], init=init_name)
        loss_fn = BCELoss()
        optimizer = Adam(lr=0.01)
        history = train(model, X, y, loss_fn, optimizer, epochs=200, batch_size=32)

        model.eval_mode()
        pred = model.forward(X)
        acc = np.mean((pred.flatten() > 0.5) == y)
        print(f"  Final: {acc * 100:.1f}% | Start loss: {history[0]:.4f} | End loss: {history[-1]:.4f}")


def experiment_batchnorm():
    """Show batch normalization stabilizes training."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Batch Normalization")
    print("=" * 60)

    X, y = make_circles(n_samples=400, noise=0.08)

    for use_bn in [False, True]:
        label = "With BatchNorm" if use_bn else "Without BatchNorm"
        print(f"\n--- {label} ---")
        np.random.seed(42)
        model = MLP([2, 64, 32, 16, 1],
                     activations=['relu', 'relu', 'relu', 'sigmoid'],
                     init='he', use_batchnorm=use_bn)
        loss_fn = BCELoss()
        optimizer = Adam(lr=0.01)
        history = train(model, X, y, loss_fn, optimizer, epochs=200, batch_size=32)

        model.eval_mode()
        pred = model.forward(X)
        acc = np.mean((pred.flatten() > 0.5) == y)
        print(f"  Final: {acc * 100:.1f}% | Params: {model.count_parameters()}")


def experiment_dropout():
    """Show dropout as regularization."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Dropout Regularization")
    print("=" * 60)

    # Small dataset + large network = overfitting
    X, y = make_moons(n_samples=100, noise=0.2)

    for drop_rate in [0.0, 0.3, 0.5]:
        label = f"Dropout = {drop_rate}"
        print(f"\n--- {label} ---")
        np.random.seed(42)
        model = MLP([2, 64, 64, 1],
                     activations=['relu', 'relu', 'sigmoid'],
                     init='he', dropout_rate=drop_rate)
        loss_fn = BCELoss()
        optimizer = Adam(lr=0.01)
        history = train(model, X, y, loss_fn, optimizer, epochs=300, batch_size=32)

        model.eval_mode()
        pred = model.forward(X)
        acc = np.mean((pred.flatten() > 0.5) == y)
        print(f"  Final: {acc * 100:.1f}% | End loss: {history[-1]:.4f}")


def experiment_optimizer_comparison():
    """Compare SGD vs Momentum vs Adam."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: Optimizer Comparison")
    print("=" * 60)

    X, y = make_moons(n_samples=300, noise=0.15)

    optimizers = [
        ("SGD (lr=0.1)", SGD(lr=0.1)),
        ("SGD+Momentum (lr=0.1, m=0.9)", SGD(lr=0.1, momentum=0.9)),
        ("Adam (lr=0.01)", Adam(lr=0.01)),
    ]

    for name, opt in optimizers:
        print(f"\n--- {name} ---")
        np.random.seed(42)
        model = MLP([2, 32, 16, 1], activations=['relu', 'relu', 'sigmoid'], init='he')
        loss_fn = BCELoss()
        history = train(model, X, y, loss_fn, opt, epochs=200, batch_size=32)

        model.eval_mode()
        pred = model.forward(X)
        acc = np.mean((pred.flatten() > 0.5) == y)
        print(f"  Final: {acc * 100:.1f}%")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("MULTI-LAYER PERCEPTRON FROM SCRATCH")
    print("Complete neural network using only NumPy")
    print("=" * 60)

    experiment_basic()
    experiment_depth()
    experiment_init()
    experiment_batchnorm()
    experiment_dropout()
    experiment_optimizer_comparison()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. An MLP is just: matrix multiply -> activation -> repeat -> loss -> backprop -> update
2. Depth matters: deeper networks learn compositional features
3. Initialization matters: Xavier for sigmoid/tanh, He for ReLU
4. Batch normalization stabilizes training in deeper networks
5. Dropout regularizes by training an implicit ensemble
6. Adam usually converges faster than plain SGD

Everything here is what PyTorch does under the hood.
The difference: PyTorch uses tensors on GPUs. The math is identical.

Next: CNNs add spatial structure (convolution). RNNs add temporal structure (recurrence).
Both are just specialized versions of this same framework.
""")

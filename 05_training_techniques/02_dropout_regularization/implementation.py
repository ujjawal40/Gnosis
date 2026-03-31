"""
Dropout and Regularization: From Scratch Implementation
=========================================================

Every regularization technique implemented from first principles in NumPy.
Demonstrates how each technique prevents overfitting through different mechanisms.

Techniques:
    1. Dropout (inverted scaling)
    2. DropConnect (weight masking)
    3. L1 Regularization (sparsity-inducing)
    4. L2 Regularization (weight decay)
    5. Elastic Net (L1 + L2)
    6. Early Stopping (validation-based)
    7. Max-Norm Constraint

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
# PART 1: DROPOUT
# =============================================================================

class Dropout:
    """
    Inverted Dropout Layer.

    During training, randomly zeroes elements with probability p and scales
    remaining elements by 1/(1-p) so expected value remains unchanged.

    During inference, no dropout is applied (identity function).

    Math:
        Training:  mask ~ Bernoulli(1-p), output = x * mask / (1-p)
        Inference: output = x
    """

    def __init__(self, p: float = 0.5):
        self.p = p
        self.mask = None
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0:
            return x
        self.mask = (np.random.rand(*x.shape) > self.p).astype(np.float64)
        return x * self.mask / (1.0 - self.p)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0:
            return grad_output
        return grad_output * self.mask / (1.0 - self.p)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class DropConnect:
    """
    DropConnect: drops weights instead of activations.

    Instead of masking activations, DropConnect masks the weight matrix
    during the forward pass. This is a more general form of dropout.

    Math:
        Training:  W_masked = W * mask, output = W_masked @ x + b
        Inference: output = W @ x + b (full weights)
    """

    def __init__(self, in_features: int, out_features: int, p: float = 0.5):
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(out_features, in_features) * scale
        self.b = np.zeros(out_features)
        self.p = p
        self.mask = None
        self.x = None
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        if self.training and self.p > 0:
            self.mask = (np.random.rand(*self.W.shape) > self.p).astype(np.float64)
            W_masked = self.W * self.mask / (1.0 - self.p)
        else:
            W_masked = self.W
        return x @ W_masked.T + self.b

    def backward(self, grad_output: np.ndarray):
        if self.training and self.p > 0:
            W_masked = self.W * self.mask / (1.0 - self.p)
        else:
            W_masked = self.W
        grad_x = grad_output @ W_masked
        grad_W = grad_output.T @ self.x
        grad_b = grad_output.sum(axis=0)
        return grad_x, grad_W, grad_b


# =============================================================================
# PART 2: WEIGHT REGULARIZATION
# =============================================================================

def l1_penalty(weights_list: list, lambda_: float) -> float:
    """
    L1 regularization (Lasso): lambda * sum(|w|)

    Induces sparsity by pushing small weights to exactly zero.
    The gradient is lambda * sign(w), which is constant regardless of w magnitude.
    """
    total = 0.0
    for W in weights_list:
        total += lambda_ * np.abs(W).sum()
    return total


def l1_gradient(W: np.ndarray, lambda_: float) -> np.ndarray:
    """Gradient of L1 penalty: lambda * sign(w)."""
    return lambda_ * np.sign(W)


def l2_penalty(weights_list: list, lambda_: float) -> float:
    """
    L2 regularization (Ridge / Weight Decay): lambda/2 * sum(w^2)

    Shrinks all weights proportionally. Never pushes weights to exactly zero.
    The gradient is lambda * w, so larger weights get stronger gradients.
    """
    total = 0.0
    for W in weights_list:
        total += 0.5 * lambda_ * np.sum(W ** 2)
    return total


def l2_gradient(W: np.ndarray, lambda_: float) -> np.ndarray:
    """Gradient of L2 penalty: lambda * w."""
    return lambda_ * W


def elastic_net_penalty(weights_list: list, l1_lambda: float,
                        l2_lambda: float) -> float:
    """Elastic Net: combines L1 sparsity with L2 smoothness."""
    return l1_penalty(weights_list, l1_lambda) + l2_penalty(weights_list, l2_lambda)


def elastic_net_gradient(W: np.ndarray, l1_lambda: float,
                         l2_lambda: float) -> np.ndarray:
    """Gradient of elastic net penalty."""
    return l1_gradient(W, l1_lambda) + l2_gradient(W, l2_lambda)


# =============================================================================
# PART 3: MAX-NORM CONSTRAINT
# =============================================================================

def max_norm_constraint(W: np.ndarray, max_val: float = 3.0) -> np.ndarray:
    """
    Max-norm constraint: clip weight norms to max_val.

    After each parameter update, rescale each weight vector (row) so its
    L2 norm doesn't exceed max_val. This bounds the size of the weight space.

    Often used with dropout for complementary regularization.
    """
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    desired = np.clip(norms, 0, max_val)
    scale = desired / (norms + 1e-8)
    return W * scale


# =============================================================================
# PART 4: EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """
    Early Stopping: halt training when validation loss stops improving.

    Monitors validation loss and stops if no improvement for `patience` epochs.
    Optionally saves the best model weights.

    This prevents overfitting by stopping before the model memorizes training data.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.should_stop = False

    def check(self, val_loss: float, weights: list = None) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if weights is not None:
                self.best_weights = [w.copy() for w in weights]
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# =============================================================================
# PART 5: REGULARIZED MLP
# =============================================================================

class RegularizedMLP:
    """
    MLP with all regularization techniques for comparison.

    Supports: dropout, L1, L2, elastic net, max-norm, early stopping.
    """

    def __init__(self, layer_dims: list, dropout_rate: float = 0.0,
                 l1_lambda: float = 0.0, l2_lambda: float = 0.0,
                 max_norm: float = 0.0):
        self.layers = []
        self.dropout_rate = dropout_rate
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.max_norm = max_norm
        self.dropouts = []

        # Initialize weights
        for i in range(len(layer_dims) - 1):
            scale = np.sqrt(2.0 / layer_dims[i])
            W = np.random.randn(layer_dims[i + 1], layer_dims[i]) * scale
            b = np.zeros(layer_dims[i + 1])
            self.layers.append((W, b))
            if i < len(layer_dims) - 2:
                self.dropouts.append(Dropout(dropout_rate))

        self.training = True

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass with dropout."""
        self.activations = [X]
        self.pre_activations = []
        h = X

        for i, (W, b) in enumerate(self.layers):
            z = h @ W.T + b
            self.pre_activations.append(z)

            if i < len(self.layers) - 1:
                h = np.maximum(0, z)  # ReLU
                if self.training and i < len(self.dropouts):
                    h = self.dropouts[i].forward(h)
            else:
                # Softmax for output
                exp_z = np.exp(z - z.max(axis=1, keepdims=True))
                h = exp_z / exp_z.sum(axis=1, keepdims=True)

            self.activations.append(h)
        return h

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Cross-entropy loss with regularization penalties."""
        n = y_true.shape[0]
        ce = -np.sum(y_true * np.log(y_pred + 1e-8)) / n

        # Add regularization
        weights = [W for W, b in self.layers]
        reg = 0.0
        if self.l1_lambda > 0:
            reg += l1_penalty(weights, self.l1_lambda)
        if self.l2_lambda > 0:
            reg += l2_penalty(weights, self.l2_lambda)

        return ce + reg

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray,
                 lr: float = 0.01):
        """Backpropagation with regularization gradients."""
        n = y_true.shape[0]
        delta = (y_pred - y_true) / n

        for i in range(len(self.layers) - 1, -1, -1):
            W, b = self.layers[i]
            grad_W = delta.T @ self.activations[i]
            grad_b = delta.sum(axis=0)

            # Add regularization gradients
            if self.l1_lambda > 0:
                grad_W += l1_gradient(W, self.l1_lambda)
            if self.l2_lambda > 0:
                grad_W += l2_gradient(W, self.l2_lambda)

            # Update
            W -= lr * grad_W
            b -= lr * grad_b
            self.layers[i] = (W, b)

            # Max-norm constraint
            if self.max_norm > 0:
                W = max_norm_constraint(W, self.max_norm)
                self.layers[i] = (W, b)

            if i > 0:
                delta = delta @ W
                # ReLU derivative
                delta *= (self.pre_activations[i - 1] > 0).astype(np.float64)
                # Dropout backward
                if i - 1 < len(self.dropouts) and self.training:
                    delta = self.dropouts[i - 1].backward(delta)

    def train_mode(self):
        self.training = True
        for d in self.dropouts:
            d.train()

    def eval_mode(self):
        self.training = False
        for d in self.dropouts:
            d.eval()

    def get_weights(self):
        return [W.copy() for W, b in self.layers]


# =============================================================================
# PART 6: DEMONSTRATION
# =============================================================================

def make_overfit_dataset(n_train=200, n_test=500, n_features=20, noise=0.3):
    """Create a dataset that's easy to overfit (small train, many features)."""
    np.random.seed(42)
    W_true = np.random.randn(n_features)
    W_true[10:] = 0  # Only first 10 features matter

    X_train = np.random.randn(n_train, n_features)
    y_train = (X_train @ W_true + noise * np.random.randn(n_train) > 0).astype(int)

    X_test = np.random.randn(n_test, n_features)
    y_test = (X_test @ W_true + noise * np.random.randn(n_test) > 0).astype(int)

    # One-hot encode
    y_train_oh = np.eye(2)[y_train]
    y_test_oh = np.eye(2)[y_test]

    return X_train, y_train_oh, X_test, y_test_oh, y_train, y_test


def demo_regularization():
    """Compare regularization techniques on an overfit-prone dataset."""
    print("=" * 70)
    print("REGULARIZATION COMPARISON")
    print("=" * 70)

    X_train, y_train_oh, X_test, y_test_oh, y_train, y_test = make_overfit_dataset()
    n_features = X_train.shape[1]

    configs = {
        "No Regularization":  {"dropout_rate": 0.0, "l1_lambda": 0.0, "l2_lambda": 0.0},
        "Dropout (0.3)":      {"dropout_rate": 0.3, "l1_lambda": 0.0, "l2_lambda": 0.0},
        "L1 (1e-3)":          {"dropout_rate": 0.0, "l1_lambda": 1e-3, "l2_lambda": 0.0},
        "L2 (1e-3)":          {"dropout_rate": 0.0, "l1_lambda": 0.0, "l2_lambda": 1e-3},
        "Dropout + L2":       {"dropout_rate": 0.3, "l1_lambda": 0.0, "l2_lambda": 1e-3},
        "Elastic Net":        {"dropout_rate": 0.0, "l1_lambda": 5e-4, "l2_lambda": 5e-4},
    }

    results = {}
    for name, kwargs in configs.items():
        np.random.seed(42)
        model = RegularizedMLP([n_features, 64, 32, 2], **kwargs)
        early_stop = EarlyStopping(patience=20)

        train_losses, test_losses = [], []
        for epoch in range(300):
            model.train_mode()
            y_pred = model.forward(X_train)
            loss = model.compute_loss(y_pred, y_train_oh)
            model.backward(y_pred, y_train_oh, lr=0.01)
            train_losses.append(loss)

            model.eval_mode()
            y_pred_test = model.forward(X_test)
            test_loss = model.compute_loss(y_pred_test, y_test_oh)
            test_losses.append(test_loss)

            if early_stop.check(test_loss, model.get_weights()):
                break

        # Final accuracy
        model.eval_mode()
        train_pred = model.forward(X_train).argmax(axis=1)
        test_pred = model.forward(X_test).argmax(axis=1)
        train_acc = (train_pred == y_train).mean()
        test_acc = (test_pred == y_test).mean()
        gap = train_acc - test_acc

        results[name] = {
            "train_acc": train_acc, "test_acc": test_acc,
            "gap": gap, "epochs": len(train_losses),
            "train_losses": train_losses, "test_losses": test_losses
        }

        print(f"{name:25s} | Train: {train_acc:.3f} | Test: {test_acc:.3f} | "
              f"Gap: {gap:.3f} | Epochs: {len(train_losses)}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (name, res) in zip(axes.flat, results.items()):
        ax.plot(res["train_losses"], label="Train", alpha=0.8)
        ax.plot(res["test_losses"], label="Test", alpha=0.8)
        ax.set_title(f"{name}\nGap: {res['gap']:.3f}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

    plt.tight_layout()
    plt.savefig(SAVE_DIR / "regularization_comparison.png", dpi=100)
    plt.close()
    print(f"\nPlot saved to {SAVE_DIR / 'regularization_comparison.png'}")


if __name__ == "__main__":
    demo_regularization()

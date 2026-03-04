"""
Perceptron in PyTorch
======================

PyTorch re-implementation of the perceptron from implementation.py.
Includes both the classical perceptron (hard threshold) and a smooth
differentiable version (sigmoid) for gradient-based training.

Architecture:
    Input (n_features) → Linear → Activation → Output (1)
"""

import torch
import torch.nn as nn


class Perceptron(nn.Module):
    """
    Classical Perceptron using nn.Linear.

    The perceptron computes: y = step(w @ x + b)
    Since step is not differentiable, we use the Rosenblatt update rule
    (geometric correction) instead of backprop.
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.linear(x) >= 0).float()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x)

    def perceptron_update(self, x: torch.Tensor, y_true: torch.Tensor,
                          lr: float = 1.0):
        """
        Rosenblatt learning rule (not gradient descent):
            w += lr * (y_true - y_pred) * x
            b += lr * (y_true - y_pred)
        """
        with torch.no_grad():
            y_pred = self.forward(x)
            error = y_true - y_pred  # (batch, 1)
            # Weight update: average over batch
            self.linear.weight += lr * (error.T @ x) / x.shape[0]
            self.linear.bias += lr * error.mean()


class SmoothPerceptron(nn.Module):
    """
    Differentiable perceptron using sigmoid activation.
    Can be trained with standard backpropagation and gradient descent.

    Computes: y = sigmoid(w @ x + b)
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))


class MultiClassPerceptron(nn.Module):
    """
    Multi-class perceptron for classification tasks.
    Uses softmax for multi-class output.

    Architecture: Input → Linear → Softmax → Class probabilities
    """

    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)  # raw logits; use with F.cross_entropy

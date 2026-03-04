"""
Calculus with PyTorch: Autograd, Gradients, and Optimization
=============================================================

PyTorch re-implementation of the calculus module from implementation.py.
Compares our hand-built Value autograd engine with torch.autograd.

Comparison:
    NumPy (implementation.py)        →  PyTorch (this file)
    ─────────────────────────────────────────────────────────
    Value class (custom autograd)    →  torch.autograd (built-in)
    numerical_derivative             →  torch.autograd.grad
    manual chain rule                →  .backward() auto chain rule
    numerical Jacobian               →  torch.autograd.functional.jacobian
    numerical Hessian                →  torch.autograd.functional.hessian
    gradient_descent loop            →  torch.optim.SGD
"""

import torch
import torch.nn as nn
import numpy as np


# =============================================================================
# PART 1: AUTOMATIC DIFFERENTIATION WITH TORCH.AUTOGRAD
# =============================================================================

def demo_basic_autograd():
    """Show basic autograd usage: scalar expressions."""
    print("=" * 60)
    print("BASIC AUTOGRAD")
    print("=" * 60)

    # Simple expression: f(x) = x^2 + 3x + 1
    x = torch.tensor(2.0, requires_grad=True)
    f = x**2 + 3*x + 1
    f.backward()
    print(f"f(x) = x² + 3x + 1")
    print(f"f(2) = {f.item():.4f}")
    print(f"f'(2) = {x.grad.item():.4f}  (expected: 2*2+3 = 7)")

    # Multi-variable: f(x,y) = x*y + sin(x)
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(2.0, requires_grad=True)
    f = x * y + torch.sin(x)
    f.backward()
    print(f"\nf(x,y) = xy + sin(x)")
    print(f"∂f/∂x = y + cos(x) = {x.grad.item():.4f} (expected: {2 + np.cos(1):.4f})")
    print(f"∂f/∂y = x = {y.grad.item():.4f} (expected: 1.0)")
    print()


def demo_computation_graph():
    """Show how torch builds and traverses the computation graph."""
    print("=" * 60)
    print("COMPUTATION GRAPH")
    print("=" * 60)

    a = torch.tensor(3.0, requires_grad=True)
    b = torch.tensor(4.0, requires_grad=True)

    # Build graph: f = (a + b) * (a - b) = a² - b²
    c = a + b      # AddBackward
    d = a - b      # SubBackward
    f = c * d      # MulBackward

    print(f"a = {a.item()}, b = {b.item()}")
    print(f"c = a + b = {c.item()}")
    print(f"d = a - b = {d.item()}")
    print(f"f = c * d = {f.item()}")
    print(f"f.grad_fn: {f.grad_fn}")

    f.backward()
    print(f"∂f/∂a = 2a = {a.grad.item():.4f} (expected: {2*3:.4f})")
    print(f"∂f/∂b = -2b = {b.grad.item():.4f} (expected: {-2*4:.4f})")
    print()


# =============================================================================
# PART 2: HIGHER-ORDER DERIVATIVES
# =============================================================================

def demo_higher_order():
    """Compute second derivatives and Hessians using create_graph=True."""
    print("=" * 60)
    print("HIGHER-ORDER DERIVATIVES")
    print("=" * 60)

    # Second derivative of f(x) = x^4
    x = torch.tensor(2.0, requires_grad=True)
    f = x**4
    # First derivative
    df_dx = torch.autograd.grad(f, x, create_graph=True)[0]
    print(f"f(x) = x⁴")
    print(f"f'(x) = 4x³ = {df_dx.item():.4f} (expected: {4*8:.4f})")

    # Second derivative
    d2f_dx2 = torch.autograd.grad(df_dx, x, create_graph=True)[0]
    print(f"f''(x) = 12x² = {d2f_dx2.item():.4f} (expected: {12*4:.4f})")

    # Third derivative
    d3f_dx3 = torch.autograd.grad(d2f_dx2, x)[0]
    print(f"f'''(x) = 24x = {d3f_dx3.item():.4f} (expected: {24*2:.4f})")
    print()


def demo_jacobian_hessian():
    """Compute Jacobians and Hessians using torch.autograd.functional."""
    print("=" * 60)
    print("JACOBIAN & HESSIAN")
    print("=" * 60)

    # f: R² → R², f(x,y) = (x*y, x+y²)
    def f(x):
        return torch.stack([x[0] * x[1], x[0] + x[1]**2])

    x = torch.tensor([1.0, 2.0])

    J = torch.autograd.functional.jacobian(f, x)
    print(f"f(x,y) = [xy, x+y²]")
    print(f"Jacobian at (1,2):\n{J}")
    print(f"Expected: [[y, x], [1, 2y]] = [[2, 1], [1, 4]]")

    # Hessian of g: R² → R, g(x,y) = x²y + y³
    def g(x):
        return x[0]**2 * x[1] + x[1]**3

    H = torch.autograd.functional.hessian(g, x)
    print(f"\ng(x,y) = x²y + y³")
    print(f"Hessian at (1,2):\n{H}")
    print(f"Expected: [[2y, 2x], [2x, 6y]] = [[4, 2], [2, 12]]")
    print()


# =============================================================================
# PART 3: CHAIN RULE IN DEEP NETWORKS
# =============================================================================

def demo_chain_rule_depth():
    """Visualize gradient flow through a deep chain of operations."""
    print("=" * 60)
    print("CHAIN RULE DEPTH TEST")
    print("=" * 60)

    # f = tanh(tanh(tanh(...tanh(x)...))) - deep composition
    x = torch.tensor(1.0, requires_grad=True)

    for depth in [1, 5, 10, 20, 50]:
        x_clone = x.clone().detach().requires_grad_(True)
        val = x_clone
        for _ in range(depth):
            val = torch.tanh(val)
        val.backward()
        print(f"  depth={depth:3d}: f(1)={val.item():.6f}, "
              f"df/dx={x_clone.grad.item():.2e}")
    print("  → Gradient vanishes with depth (tanh saturation)")
    print()


# =============================================================================
# PART 4: TRAINING A NEURON (AND GATE)
# =============================================================================

def demo_train_neuron():
    """Train a single neuron on AND gate, matching the NumPy version."""
    print("=" * 60)
    print("TRAIN SINGLE NEURON (AND GATE)")
    print("=" * 60)

    # AND gate data
    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = torch.tensor([[0.], [0.], [0.], [1.]])

    # Single neuron
    model = nn.Linear(2, 1)
    nn.init.normal_(model.weight, std=0.1)
    nn.init.zeros_(model.bias)

    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    loss_fn = nn.MSELoss()

    for epoch in range(200):
        pred = torch.sigmoid(model(X))
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d}: loss={loss.item():.4f}")

    # Final predictions
    with torch.no_grad():
        preds = torch.sigmoid(model(X))
        print(f"\nFinal predictions:")
        for i in range(4):
            print(f"  {X[i].tolist()} → {preds[i].item():.4f} (target: {y[i].item():.0f})")
    print()


# =============================================================================
# PART 5: GRADIENT DESCENT ON TEST FUNCTIONS
# =============================================================================

def demo_gradient_descent():
    """Gradient descent on Rosenbrock using torch.autograd (no manual gradients)."""
    print("=" * 60)
    print("GRADIENT DESCENT ON ROSENBROCK (autograd)")
    print("=" * 60)

    def rosenbrock(xy):
        x, y = xy[0], xy[1]
        return (1 - x)**2 + 100 * (y - x**2)**2

    # Using raw autograd
    xy = torch.tensor([-1.0, 1.0], requires_grad=True)
    lr = 0.001

    for step in range(5001):
        loss = rosenbrock(xy)
        loss.backward()

        with torch.no_grad():
            xy -= lr * xy.grad
            xy.grad.zero_()

        if step % 1000 == 0:
            print(f"  Step {step:5d}: f({xy[0].item():.4f}, {xy[1].item():.4f}) "
                  f"= {loss.item():.6f}")

    # Using torch.optim.Adam
    print("\nWith Adam optimizer:")
    xy = torch.tensor([-1.0, 1.0], requires_grad=True)
    optimizer = torch.optim.Adam([xy], lr=0.01)

    for step in range(3001):
        optimizer.zero_grad()
        loss = rosenbrock(xy)
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"  Step {step:5d}: f({xy[0].item():.4f}, {xy[1].item():.4f}) "
                  f"= {loss.item():.6f}")
    print()


if __name__ == "__main__":
    demo_basic_autograd()
    demo_computation_graph()
    demo_higher_order()
    demo_jacobian_hessian()
    demo_chain_rule_depth()
    demo_train_neuron()
    demo_gradient_descent()

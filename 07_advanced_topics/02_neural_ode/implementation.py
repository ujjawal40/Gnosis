"""
Neural ODE from Scratch
========================

Treats neural network layers as a continuous dynamical system.
Uses simple Euler and RK4 ODE solvers, plus the adjoint method for gradients.
"""

import numpy as np


# ==============================================================================
# Part 1: ODE Solvers
# ==============================================================================

def euler_solve(f, y0, t_span, n_steps=100):
    """Euler method: y_{n+1} = y_n + h * f(y_n, t_n)"""
    t0, t1 = t_span
    h = (t1 - t0) / n_steps
    y = y0.copy()
    t = t0
    trajectory = [y.copy()]

    for _ in range(n_steps):
        y = y + h * f(y, t)
        t += h
        trajectory.append(y.copy())

    return y, trajectory


def rk4_solve(f, y0, t_span, n_steps=50):
    """4th-order Runge-Kutta: more accurate than Euler."""
    t0, t1 = t_span
    h = (t1 - t0) / n_steps
    y = y0.copy()
    t = t0
    trajectory = [y.copy()]

    for _ in range(n_steps):
        k1 = f(y, t)
        k2 = f(y + h/2 * k1, t + h/2)
        k3 = f(y + h/2 * k2, t + h/2)
        k4 = f(y + h * k3, t + h)
        y = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        trajectory.append(y.copy())

    return y, trajectory


# ==============================================================================
# Part 2: Neural ODE Layer
# ==============================================================================

class NeuralODEFunc:
    """
    The vector field f(h, t; θ) defining the continuous dynamics.
    dh/dt = f(h, t; θ)

    This is a simple MLP that takes (h, t) and outputs dh/dt.
    """

    def __init__(self, dim, hidden_dim=32):
        self.dim = dim
        s = np.sqrt(2.0 / (dim + 1))
        self.W1 = np.random.randn(dim + 1, hidden_dim) * s  # +1 for time
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(dim)

    def __call__(self, h, t):
        """Compute dh/dt given current state h and time t."""
        # Concatenate time as a feature
        t_feat = np.full((h.shape[0], 1), t) if len(h.shape) > 1 else np.array([t])
        if len(h.shape) == 1:
            h = h.reshape(1, -1)
            t_feat = t_feat.reshape(1, -1)

        inp = np.hstack([h, t_feat])
        hidden = np.tanh(inp @ self.W1 + self.b1)
        return hidden @ self.W2 + self.b2


class NeuralODE:
    """
    Neural ODE layer: transforms input by solving an ODE.

    Forward:  h(T) = h(0) + ∫₀ᵀ f(h(t), t; θ) dt
    Backward: Uses adjoint method for memory-efficient gradients
    """

    def __init__(self, dim, hidden_dim=32, t_span=(0, 1), n_steps=20):
        self.func = NeuralODEFunc(dim, hidden_dim)
        self.t_span = t_span
        self.n_steps = n_steps

    def forward(self, x):
        """Solve ODE forward in time."""
        self.x = x
        self.output, self.trajectory = euler_solve(
            self.func, x, self.t_span, self.n_steps
        )
        return self.output

    def backward(self, grad_output, lr=0.01):
        """
        Simplified adjoint method.
        In full implementation, this solves a reverse ODE.
        Here we use finite differences for the parameter gradients.
        """
        h = 1e-5

        # Gradient w.r.t. W1
        for i in range(self.func.W1.shape[0]):
            for j in range(self.func.W1.shape[1]):
                self.func.W1[i, j] += h
                out_plus, _ = euler_solve(self.func, self.x, self.t_span, self.n_steps)
                self.func.W1[i, j] -= 2 * h
                out_minus, _ = euler_solve(self.func, self.x, self.t_span, self.n_steps)
                self.func.W1[i, j] += h

                dW = np.sum(grad_output * (out_plus - out_minus) / (2 * h))
                self.func.W1[i, j] -= lr * dW

        # Gradient w.r.t. W2
        for i in range(self.func.W2.shape[0]):
            for j in range(self.func.W2.shape[1]):
                self.func.W2[i, j] += h
                out_plus, _ = euler_solve(self.func, self.x, self.t_span, self.n_steps)
                self.func.W2[i, j] -= 2 * h
                out_minus, _ = euler_solve(self.func, self.x, self.t_span, self.n_steps)
                self.func.W2[i, j] += h

                dW = np.sum(grad_output * (out_plus - out_minus) / (2 * h))
                self.func.W2[i, j] -= lr * dW


# ==============================================================================
# Experiments
# ==============================================================================

def experiment_continuous_dynamics():
    """Show how Neural ODE transforms data through continuous flow."""
    print("=" * 60)
    print("EXPERIMENT 1: Continuous Dynamics")
    print("=" * 60)

    np.random.seed(42)

    # Create 2D data points
    x = np.array([[1.0, 0.0],
                   [0.0, 1.0],
                   [-1.0, 0.0],
                   [0.0, -1.0]])

    node = NeuralODE(dim=2, hidden_dim=16, t_span=(0, 1), n_steps=10)

    print(f"\nInitial points:")
    for i in range(len(x)):
        print(f"  ({x[i, 0]:.3f}, {x[i, 1]:.3f})")

    output = node.forward(x)

    print(f"\nAfter Neural ODE flow (t=0 → t=1):")
    for i in range(len(output)):
        print(f"  ({output[i, 0]:.3f}, {output[i, 1]:.3f})")

    print(f"\nTrajectory of first point through time:")
    for step in range(0, len(node.trajectory), 2):
        t = step / (len(node.trajectory) - 1)
        pt = node.trajectory[step][0]
        print(f"  t={t:.2f}: ({pt[0]:.3f}, {pt[1]:.3f})")

    print("\nThe points flow continuously — like fluid dynamics on the data!")


def experiment_classification():
    """Use Neural ODE for a simple classification task."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Neural ODE for Classification")
    print("=" * 60)

    np.random.seed(42)

    # Generate concentric circles
    n = 100
    theta_inner = np.random.uniform(0, 2*np.pi, n//2)
    theta_outer = np.random.uniform(0, 2*np.pi, n//2)
    X_inner = 0.5 * np.column_stack([np.cos(theta_inner), np.sin(theta_inner)])
    X_outer = 1.5 * np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])
    X_inner += np.random.randn(n//2, 2) * 0.1
    X_outer += np.random.randn(n//2, 2) * 0.1

    X = np.vstack([X_inner, X_outer])
    y = np.hstack([np.zeros(n//2), np.ones(n//2)])

    # Simple model: NeuralODE -> Linear -> Sigmoid
    node = NeuralODE(dim=2, hidden_dim=16, t_span=(0, 1), n_steps=10)
    W_out = np.random.randn(2, 1) * 0.1
    b_out = np.zeros(1)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    print(f"\nData: {n} points, 2 concentric circles")
    print(f"Model: NeuralODE(2D, 16 hidden, 10 steps) -> Linear -> Sigmoid\n")

    for epoch in range(51):
        # Forward
        h = node.forward(X)
        logits = h @ W_out + b_out
        pred = sigmoid(logits).flatten()

        # BCE loss
        pred_clip = np.clip(pred, 1e-8, 1-1e-8)
        loss = -np.mean(y * np.log(pred_clip) + (1-y) * np.log(1-pred_clip))

        # Backward (simplified: only update output layer for speed)
        grad = (pred - y).reshape(-1, 1) / n
        W_out -= 0.1 * (h.T @ grad)
        b_out -= 0.1 * np.sum(grad)

        if epoch % 10 == 0:
            acc = np.mean((pred > 0.5) == y)
            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Acc: {acc*100:.1f}%")


def experiment_ode_vs_discrete():
    """Compare Neural ODE with discrete ResNet-style layers."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Continuous vs Discrete Depth")
    print("=" * 60)

    np.random.seed(42)
    x = np.random.randn(5, 2)

    print("\nSame network evaluated at different 'depths' (integration steps):")
    func = NeuralODEFunc(2, 16)

    for n_steps in [1, 5, 10, 20, 50]:
        output, _ = euler_solve(func, x, (0, 1), n_steps)
        norm = np.mean(np.linalg.norm(output, axis=1))
        print(f"  Steps={n_steps:2d}: mean output norm = {norm:.4f}")

    print("\nMore steps = more accurate ODE solution (converges as steps increase)")
    print("This is continuous depth: the 'number of layers' is adaptive!")
    print("\nResNet connection: ResNet's h_{l+1} = h_l + f(h_l) is one Euler step")
    print("Neural ODE = ResNet with infinitely many infinitesimal steps")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("NEURAL ODE FROM SCRATCH")
    print("Neural networks as continuous dynamical systems\n")

    experiment_continuous_dynamics()
    experiment_classification()
    experiment_ode_vs_discrete()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. Neural ODE: dh/dt = f(h,t;θ) — continuous analog of ResNet
2. Depth is continuous: ODE solver adaptively decides number of steps
3. Memory efficient: adjoint method avoids storing intermediate states
4. ResNet is a special case: one Euler step of a Neural ODE
5. Connects deep learning to dynamical systems theory
6. Foundation for treating networks as flows on manifolds

RESEARCH CONNECTION:
Neural ODEs formalize the "geometric compositional learning" vision.
The vector field f defines how representations flow on a learned manifold.
Making f respect the manifold's geometry could be revolutionary.
""")

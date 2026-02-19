"""
Calculus from Scratch: Numerical Differentiation, Automatic Differentiation, and Gradient Descent.

This module builds the mathematical machinery that makes neural networks learn.
Everything is implemented from first principles using only NumPy.

Structure:
    Part 1: Numerical Differentiation — approximate derivatives with finite differences
    Part 2: Automatic Differentiation — build a miniature autograd engine (the heart of this module)
    Part 3: Jacobians and Hessians — higher-order derivative structures
    Part 4: Gradient Descent — use everything above to optimize functions

The autograd engine (Part 2) is the most important piece. It implements the same algorithm
that powers PyTorch's autograd and TensorFlow's GradientTape: reverse-mode automatic
differentiation on a dynamically constructed computation graph.
"""

import numpy as np
import math


# =============================================================================
# PART 1: NUMERICAL DIFFERENTIATION
# =============================================================================
# Approximate derivatives using finite differences. These are simple but inexact —
# we use them primarily to VERIFY our autograd engine's exact gradients.

def numerical_derivative_forward(f, x, h=1e-7):
    """
    Forward difference approximation of f'(x).

    Formula: f'(x) ≈ (f(x + h) - f(x)) / h

    This is the most direct implementation of the derivative definition.
    Error is O(h): halving h roughly halves the error.

    Args:
        f: Scalar function of a scalar
        x: Point at which to evaluate the derivative
        h: Step size (default 1e-7)

    Returns:
        Approximate value of f'(x)
    """
    return (f(x + h) - f(x)) / h


def numerical_derivative_backward(f, x, h=1e-7):
    """
    Backward difference approximation of f'(x).

    Formula: f'(x) ≈ (f(x) - f(x - h)) / h

    Same accuracy as forward difference — O(h) error.
    """
    return (f(x) - f(x - h)) / h


def numerical_derivative_central(f, x, h=1e-5):
    """
    Central difference approximation of f'(x).

    Formula: f'(x) ≈ (f(x + h) - f(x - h)) / (2h)

    This is significantly more accurate than forward/backward difference.
    Error is O(h²): halving h quarters the error.

    Why: The Taylor expansion of f(x+h) and f(x-h) are:
        f(x+h) = f(x) + f'(x)h + f''(x)h²/2 + O(h³)
        f(x-h) = f(x) - f'(x)h + f''(x)h²/2 + O(h³)

    Subtracting cancels the h² term:
        f(x+h) - f(x-h) = 2f'(x)h + O(h³)

    So (f(x+h) - f(x-h)) / 2h = f'(x) + O(h²)

    Args:
        f: Scalar function of a scalar
        x: Point at which to evaluate the derivative
        h: Step size (default 1e-5, larger than forward/backward because the
           higher accuracy allows it)

    Returns:
        Approximate value of f'(x)
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_partial_derivative(f, x, i, h=1e-5):
    """
    Central difference approximation of ∂f/∂xᵢ.

    Perturbs only the i-th component of x, keeping all others fixed.

    Args:
        f: Scalar function of a vector (numpy array)
        x: Point at which to evaluate (numpy array)
        i: Index of the variable to differentiate with respect to
        h: Step size

    Returns:
        Approximate value of ∂f/∂xᵢ
    """
    x_plus = x.copy().astype(float)
    x_minus = x.copy().astype(float)
    x_plus[i] += h
    x_minus[i] -= h
    return (f(x_plus) - f(x_minus)) / (2 * h)


def numerical_gradient(f, x, h=1e-5):
    """
    Compute the full gradient ∇f numerically using central differences.

    Args:
        f: Scalar function of a vector (numpy array)
        x: Point at which to evaluate (numpy array)
        h: Step size

    Returns:
        Gradient vector (numpy array of same shape as x)
    """
    x = np.array(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        grad[i] = numerical_partial_derivative(f, x, i, h)
    return grad


def demonstrate_numerical_differentiation():
    """Compare numerical differentiation methods against known analytical derivatives."""
    print("=" * 70)
    print("PART 1: NUMERICAL DIFFERENTIATION")
    print("=" * 70)

    # Test function: f(x) = x³ + 2x² - 5x + 3
    # Analytical derivative: f'(x) = 3x² + 4x - 5
    f = lambda x: x**3 + 2*x**2 - 5*x + 3
    f_prime_exact = lambda x: 3*x**2 + 4*x - 5

    x = 2.0
    exact = f_prime_exact(x)

    print(f"\nf(x) = x³ + 2x² - 5x + 3")
    print(f"f'(x) = 3x² + 4x - 5")
    print(f"f'({x}) = {exact} (exact)")
    print()

    forward = numerical_derivative_forward(f, x)
    backward = numerical_derivative_backward(f, x)
    central = numerical_derivative_central(f, x)

    print(f"Forward difference:  {forward:.10f}  (error: {abs(forward - exact):.2e})")
    print(f"Backward difference: {backward:.10f}  (error: {abs(backward - exact):.2e})")
    print(f"Central difference:  {central:.10f}  (error: {abs(central - exact):.2e})")
    print()
    print("Note: Central difference error is ~10⁵x smaller — O(h²) vs O(h).")

    # Demonstrate gradient of a multivariable function
    print(f"\n--- Gradient of f(x,y) = x²y + sin(y) ---")
    f_multi = lambda v: v[0]**2 * v[1] + np.sin(v[1])
    point = np.array([3.0, np.pi/4])
    grad = numerical_gradient(f_multi, point)
    # Analytical: ∂f/∂x = 2xy, ∂f/∂y = x² + cos(y)
    exact_grad = np.array([
        2 * point[0] * point[1],
        point[0]**2 + np.cos(point[1])
    ])
    print(f"At point ({point[0]:.2f}, {point[1]:.4f}):")
    print(f"  Numerical gradient: [{grad[0]:.8f}, {grad[1]:.8f}]")
    print(f"  Exact gradient:     [{exact_grad[0]:.8f}, {exact_grad[1]:.8f}]")
    print(f"  Max error:          {np.max(np.abs(grad - exact_grad)):.2e}")


# =============================================================================
# PART 2: AUTOMATIC DIFFERENTIATION — THE AUTOGRAD ENGINE
# =============================================================================
# This is the core of the module. We build a miniature version of PyTorch's autograd.
#
# The key idea:
#   1. Every computation creates a node in a directed acyclic graph (DAG).
#   2. Each node stores its value AND how to compute its local gradient.
#   3. To get gradients, we traverse the graph backward (topological sort in reverse),
#      applying the chain rule at each node.
#
# This is reverse-mode automatic differentiation, a.k.a. backpropagation.

class Value:
    """
    A scalar value that tracks its computation history for automatic differentiation.

    This is the fundamental building block of our autograd engine. Each Value stores:
      - data: the actual numerical value
      - grad: the gradient of the final output with respect to this value (∂output/∂self)
      - _backward: a function that propagates gradients to this node's parents
      - _prev: the set of Values that were used to produce this one
      - _op: a string label for the operation that produced this value (for debugging)

    The gradient is accumulated during the backward pass via the chain rule.

    Usage:
        a = Value(2.0)
        b = Value(3.0)
        c = a * b + a  # builds computation graph: c = 2*3 + 2 = 8
        c.backward()    # computes gradients: dc/da = b + 1 = 4, dc/db = a = 2
        print(a.grad)   # 4.0
        print(b.grad)   # 2.0
    """

    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0  # gradient accumulates here during backward pass
        self._backward = lambda: None  # no-op by default (leaf nodes)
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"

    # ------------------------------------------------------------------
    # Arithmetic operations — each builds a new node in the computation graph
    # and defines the local backward function (local gradient computation).
    # ------------------------------------------------------------------

    def __add__(self, other):
        """
        Addition: z = self + other

        Local gradients:
            ∂z/∂self = 1
            ∂z/∂other = 1

        Chain rule (backward):
            ∂L/∂self += ∂L/∂z * ∂z/∂self = ∂L/∂z * 1
            ∂L/∂other += ∂L/∂z * ∂z/∂other = ∂L/∂z * 1

        Gradients distribute equally — makes sense because adding 1 to either
        input increases the output by 1.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        """
        Multiplication: z = self * other

        Local gradients:
            ∂z/∂self = other
            ∂z/∂other = self

        Chain rule (backward):
            ∂L/∂self += ∂L/∂z * other.data
            ∂L/∂other += ∂L/∂z * self.data

        Each input's gradient is the OTHER input's value — this is the product rule.
        If self is large, a small change in other causes a large change in z.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        """
        Power: z = self^exponent (exponent is a constant, not a Value)

        Local gradient:
            ∂z/∂self = exponent * self^(exponent - 1)

        This is the power rule. We only support constant exponents here
        because variable exponents require d/dx(x^y) = x^y * (y/x + y'*ln(x)),
        which adds complexity we don't need.
        """
        assert isinstance(exponent, (int, float)), "Only constant exponents supported"
        out = Value(self.data ** exponent, (self,), f'**{exponent}')

        def _backward():
            self.grad += (exponent * self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        """Negation: -self = self * (-1)"""
        return self * -1

    def __sub__(self, other):
        """Subtraction: self - other = self + (-other)"""
        return self + (-other)

    def __truediv__(self, other):
        """Division: self / other = self * other^(-1)"""
        return self * (other ** -1)

    # Reverse operations (so that int/float OP Value works)
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Value(other) - self

    def __rtruediv__(self, other):
        return Value(other) / self

    # ------------------------------------------------------------------
    # Activation functions — critical nonlinearities for neural networks
    # ------------------------------------------------------------------

    def relu(self):
        """
        Rectified Linear Unit: z = max(0, self)

        Local gradient:
            ∂z/∂self = 1 if self > 0, else 0

        ReLU is piecewise linear: it passes positive values through unchanged
        and kills negative values. Its gradient is the simplest possible:
        it's either fully on (1) or fully off (0).

        This simplicity is exactly why ReLU works so well — during backpropagation,
        the gradient either flows through unchanged or is blocked entirely.
        No multiplicative shrinkage like sigmoid/tanh, which is why ReLU largely
        solved the vanishing gradient problem in deep networks.
        """
        out = Value(max(0.0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        """
        Sigmoid: z = 1 / (1 + exp(-self))

        Local gradient:
            ∂z/∂self = z * (1 - z)

        The sigmoid squashes any input to the range (0, 1). Its gradient is
        elegant: it depends only on the output, not the input.

        Problem: when |self| is large, z ≈ 0 or z ≈ 1, so the gradient
        z*(1-z) ≈ 0. This is the vanishing gradient problem for sigmoid.
        The maximum gradient is 0.25 (at self = 0), which means each sigmoid
        layer can shrink gradients by at least 4x.
        """
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            self.grad += (s * (1.0 - s)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        """
        Hyperbolic tangent: z = tanh(self) = (e^self - e^(-self)) / (e^self + e^(-self))

        Local gradient:
            ∂z/∂self = 1 - z²

        Like sigmoid but centered at 0, mapping to (-1, 1) instead of (0, 1).
        Preferred over sigmoid as a hidden-layer activation because its outputs
        are zero-centered, which helps gradient flow.

        Still suffers from vanishing gradients for large |self|, where tanh
        saturates and the gradient 1 - z² → 0.
        """
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        """
        Exponential: z = e^self

        Local gradient:
            ∂z/∂self = e^self = z

        The exponential function is its own derivative — the defining property of e.
        """
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        """
        Natural logarithm: z = ln(self)

        Local gradient:
            ∂z/∂self = 1 / self

        Appears throughout ML: log-likelihood, cross-entropy loss, log-probabilities.
        """
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # The backward pass — the algorithm that makes everything work
    # ------------------------------------------------------------------

    def backward(self):
        """
        Compute gradients of this value with respect to all ancestor values
        using reverse-mode automatic differentiation (backpropagation).

        Algorithm:
            1. Build a topological ordering of the computation graph
               (every node appears after all nodes that depend on it).
            2. Set this node's gradient to 1.0 (∂self/∂self = 1).
            3. Traverse the topological order in reverse, calling each
               node's _backward() to propagate gradients to its parents.

        Why topological sort?
            A node's gradient is the SUM of gradient contributions from all
            paths to the output. We need all contributions to arrive before
            we propagate further. Reverse topological order guarantees this:
            by the time we process a node, all nodes that depend on it have
            already propagated their gradient contributions.

        Time complexity: O(number of operations in the graph)
            — same order as the forward pass, regardless of the number of parameters.
            This is the key efficiency advantage of reverse-mode AD.
        """
        # Step 1: Topological sort via depth-first search
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Step 2: Set output gradient to 1 (base case of the chain rule)
        self.grad = 1.0

        # Step 3: Backward pass in reverse topological order
        for node in reversed(topo):
            node._backward()


def demonstrate_autograd():
    """Build and differentiate several computation graphs, verifying against numerical gradients."""
    print("\n" + "=" * 70)
    print("PART 2: AUTOMATIC DIFFERENTIATION (AUTOGRAD ENGINE)")
    print("=" * 70)

    # --- Example 1: Simple expression ---
    print("\n--- Example 1: f = a*b + c ---")
    a = Value(2.0)
    b = Value(3.0)
    c = Value(4.0)
    f = a * b + c  # 2*3 + 4 = 10

    f.backward()
    print(f"f = a*b + c = {f.data}")
    print(f"  ∂f/∂a = b = {a.grad}  (expected: 3.0)")
    print(f"  ∂f/∂b = a = {b.grad}  (expected: 2.0)")
    print(f"  ∂f/∂c = 1 = {c.grad}  (expected: 1.0)")

    # --- Example 2: Nonlinear expression with reuse ---
    print("\n--- Example 2: f = (a + b) * (a + b) = (a + b)² ---")
    a = Value(3.0)
    b = Value(2.0)
    c = a + b       # c = 5
    f = c * c        # f = 25, df/da = 2*(a+b) = 10, df/db = 2*(a+b) = 10

    f.backward()
    print(f"f = (a + b)² = {f.data}")
    print(f"  ∂f/∂a = 2(a+b) = {a.grad}  (expected: 10.0)")
    print(f"  ∂f/∂b = 2(a+b) = {b.grad}  (expected: 10.0)")

    # --- Example 3: Chain of operations (testing chain rule depth) ---
    print("\n--- Example 3: f = sigmoid(a*b + c) ---")
    a = Value(1.0)
    b = Value(2.0)
    c = Value(-1.0)
    z = a * b + c    # z = 2 - 1 = 1
    f = z.sigmoid()   # f = sigmoid(1) ≈ 0.7311

    f.backward()
    # sigmoid'(z) = sigmoid(z)*(1-sigmoid(z))
    sig_z = 1.0 / (1.0 + math.exp(-1.0))
    dsig = sig_z * (1 - sig_z)
    print(f"f = sigmoid(a*b + c) = {f.data:.6f}")
    print(f"  ∂f/∂a = sigmoid'(z) * b = {a.grad:.6f}  (expected: {dsig * 2.0:.6f})")
    print(f"  ∂f/∂b = sigmoid'(z) * a = {b.grad:.6f}  (expected: {dsig * 1.0:.6f})")
    print(f"  ∂f/∂c = sigmoid'(z) * 1 = {c.grad:.6f}  (expected: {dsig:.6f})")

    # --- Example 4: Power and division ---
    print("\n--- Example 4: f = (x² + y²) / (x + y) ---")
    x = Value(3.0)
    y = Value(4.0)
    f = (x**2 + y**2) / (x + y)
    # f = (9+16)/(3+4) = 25/7 ≈ 3.5714
    # ∂f/∂x = (2x(x+y) - (x²+y²)) / (x+y)²
    #        = (2*3*7 - 25) / 49 = (42-25)/49 = 17/49 ≈ 0.3469

    f.backward()
    expected_dfx = (2 * 3.0 * 7.0 - 25.0) / 49.0
    expected_dfy = (2 * 4.0 * 7.0 - 25.0) / 49.0
    print(f"f = (x² + y²) / (x + y) = {f.data:.6f}")
    print(f"  ∂f/∂x = {x.grad:.6f}  (expected: {expected_dfx:.6f})")
    print(f"  ∂f/∂y = {y.grad:.6f}  (expected: {expected_dfy:.6f})")

    # --- Example 5: ReLU ---
    print("\n--- Example 5: ReLU activation ---")
    for val in [2.0, -3.0, 0.0]:
        x = Value(val)
        f = x.relu()
        f.backward()
        print(f"  ReLU({val:5.1f}) = {f.data:5.1f}, grad = {x.grad:.1f}"
              f"  (expected: {'1.0' if val > 0 else '0.0'})")

    # --- Example 6: tanh ---
    print("\n--- Example 6: tanh activation ---")
    x = Value(0.5)
    f = x.tanh()
    f.backward()
    expected = 1 - math.tanh(0.5)**2
    print(f"  tanh(0.5) = {f.data:.6f}, grad = {x.grad:.6f}  (expected: {expected:.6f})")

    # --- Verification against numerical gradients ---
    print("\n--- Verification: Autograd vs Numerical Gradients ---")
    print("Testing f(a,b,c) = tanh(a² * sigmoid(b) + c³) ...")

    def f_numerical(vals):
        a_val, b_val, c_val = vals
        sig_b = 1.0 / (1.0 + np.exp(-b_val))
        return np.tanh(a_val**2 * sig_b + c_val**3)

    # Autograd computation
    a = Value(1.5)
    b = Value(-0.5)
    c = Value(0.7)
    result = (a**2 * b.sigmoid() + c**3).tanh()
    result.backward()
    autograd_grads = [a.grad, b.grad, c.grad]

    # Numerical computation
    point = np.array([1.5, -0.5, 0.7])
    numerical_grads = numerical_gradient(f_numerical, point)

    print(f"  Autograd:  [{autograd_grads[0]:.8f}, {autograd_grads[1]:.8f}, {autograd_grads[2]:.8f}]")
    print(f"  Numerical: [{numerical_grads[0]:.8f}, {numerical_grads[1]:.8f}, {numerical_grads[2]:.8f}]")
    max_err = max(abs(a - n) for a, n in zip(autograd_grads, numerical_grads))
    print(f"  Max error: {max_err:.2e}")
    print(f"  Match: {'YES' if max_err < 1e-5 else 'NO'}")


# =============================================================================
# PART 3: JACOBIANS AND HESSIANS (NUMERICAL)
# =============================================================================
# These are higher-order derivative structures computed numerically.
# In practice, frameworks use autograd for these too, but numerical computation
# makes the concepts concrete.

def numerical_jacobian(F, x, h=1e-5):
    """
    Compute the Jacobian matrix of F: Rⁿ → Rᵐ at point x.

    The Jacobian J has shape (m, n), where:
        J[i][j] = ∂Fᵢ/∂xⱼ

    We compute it column by column: perturbing xⱼ gives us column j.

    Args:
        F: Vector function (takes numpy array, returns numpy array)
        x: Point at which to evaluate (numpy array of length n)
        h: Step size for central differences

    Returns:
        Jacobian matrix (numpy array of shape (m, n))
    """
    x = np.array(x, dtype=float)
    f0 = np.array(F(x), dtype=float)
    n = len(x)
    m = len(f0)
    J = np.zeros((m, n))

    for j in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += h
        x_minus[j] -= h
        J[:, j] = (np.array(F(x_plus)) - np.array(F(x_minus))) / (2 * h)

    return J


def numerical_hessian(f, x, h=1e-5):
    """
    Compute the Hessian matrix of f: Rⁿ → R at point x.

    The Hessian H has shape (n, n), where:
        H[i][j] = ∂²f / ∂xᵢ∂xⱼ

    We compute it using the second-order central difference formula:
        ∂²f/∂xᵢ∂xⱼ ≈ (f(x+hᵢ+hⱼ) - f(x+hᵢ-hⱼ) - f(x-hᵢ+hⱼ) + f(x-hᵢ-hⱼ)) / (4h²)

    For diagonal entries (i = j):
        ∂²f/∂xᵢ² ≈ (f(x+hᵢ) - 2f(x) + f(x-hᵢ)) / h²

    Args:
        f: Scalar function of a vector (numpy array)
        x: Point at which to evaluate (numpy array of length n)
        h: Step size

    Returns:
        Hessian matrix (numpy array of shape (n, n))
    """
    x = np.array(x, dtype=float)
    n = len(x)
    H = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal: second derivative using central difference
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                H[i, i] = (f(x_plus) - 2 * f(x) + f(x_minus)) / (h ** 2)
            else:
                # Off-diagonal: mixed partial derivative
                x_pp = x.copy()  # +h_i, +h_j
                x_pm = x.copy()  # +h_i, -h_j
                x_mp = x.copy()  # -h_i, +h_j
                x_mm = x.copy()  # -h_i, -h_j
                x_pp[i] += h; x_pp[j] += h
                x_pm[i] += h; x_pm[j] -= h
                x_mp[i] -= h; x_mp[j] += h
                x_mm[i] -= h; x_mm[j] -= h
                H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h ** 2)

    return H


def demonstrate_jacobian_hessian():
    """Compute and interpret Jacobians and Hessians of example functions."""
    print("\n" + "=" * 70)
    print("PART 3: JACOBIANS AND HESSIANS")
    print("=" * 70)

    # --- Jacobian Example ---
    # F(x, y) = [x²y, x + sin(y)]  — maps R² → R²
    print("\n--- Jacobian of F(x,y) = [x²y, x + sin(y)] ---")

    def F(v):
        x, y = v
        return np.array([x**2 * y, x + np.sin(y)])

    point = np.array([2.0, np.pi / 3])
    J = numerical_jacobian(F, point)

    # Analytical Jacobian:
    # J = [[2xy,  x²    ],
    #      [1,    cos(y) ]]
    x_val, y_val = point
    J_exact = np.array([
        [2 * x_val * y_val, x_val**2],
        [1.0, np.cos(y_val)]
    ])

    print(f"At point ({x_val:.2f}, {y_val:.4f}):")
    print(f"\nNumerical Jacobian:")
    for row in J:
        print(f"  [{row[0]:10.6f}, {row[1]:10.6f}]")
    print(f"\nExact Jacobian:")
    for row in J_exact:
        print(f"  [{row[0]:10.6f}, {row[1]:10.6f}]")
    print(f"\nMax error: {np.max(np.abs(J - J_exact)):.2e}")

    # --- Hessian Example ---
    # f(x, y) = x³ + x²y + y² — a simple function with clear curvature
    print("\n--- Hessian of f(x,y) = x³ + x²y + y² ---")

    def f_hess(v):
        x, y = v
        return x**3 + x**2 * y + y**2

    point = np.array([1.0, 2.0])
    H = numerical_hessian(f_hess, point)

    # Analytical Hessian:
    # ∂f/∂x = 3x² + 2xy → ∂²f/∂x² = 6x + 2y,  ∂²f/∂x∂y = 2x
    # ∂f/∂y = x² + 2y   → ∂²f/∂y∂x = 2x,       ∂²f/∂y² = 2
    x_val, y_val = point
    H_exact = np.array([
        [6 * x_val + 2 * y_val, 2 * x_val],
        [2 * x_val, 2.0]
    ])

    print(f"At point ({x_val:.1f}, {y_val:.1f}):")
    print(f"\nNumerical Hessian:")
    for row in H:
        print(f"  [{row[0]:10.6f}, {row[1]:10.6f}]")
    print(f"\nExact Hessian:")
    for row in H_exact:
        print(f"  [{row[0]:10.6f}, {row[1]:10.6f}]")
    print(f"\nMax error: {np.max(np.abs(H - H_exact)):.2e}")

    # Eigenvalue analysis
    eigenvalues = np.linalg.eigvalsh(H)
    print(f"\nEigenvalues of Hessian: {eigenvalues}")
    if np.all(eigenvalues > 0):
        print("All eigenvalues positive → local minimum (positive definite)")
    elif np.all(eigenvalues < 0):
        print("All eigenvalues negative → local maximum (negative definite)")
    else:
        print("Mixed eigenvalues → saddle point (indefinite)")
    print(f"Condition number: {max(abs(eigenvalues)) / min(abs(eigenvalues)):.2f}")


# =============================================================================
# PART 4: GRADIENT DESCENT
# =============================================================================
# Bring everything together: use our autograd engine to minimize a function.

def gradient_descent_numerical(f, x0, learning_rate=0.1, num_steps=50):
    """
    Gradient descent using numerical gradients.

    Simple but illustrative: at each step, compute the gradient numerically
    and take a step in the negative gradient direction.

    Args:
        f: Scalar function of a vector
        x0: Starting point (numpy array)
        learning_rate: Step size (α)
        num_steps: Number of gradient descent steps

    Returns:
        history: List of (x, f(x)) tuples showing the optimization trajectory
    """
    x = np.array(x0, dtype=float)
    history = [(x.copy(), f(x))]

    for step in range(num_steps):
        grad = numerical_gradient(f, x)
        x = x - learning_rate * grad
        history.append((x.copy(), f(x)))

    return history


def gradient_descent_autograd(f_builder, params_init, learning_rate=0.1, num_steps=50):
    """
    Gradient descent using our autograd engine.

    Instead of numerical gradients, we get exact gradients from the computation graph.
    This is how real ML frameworks work.

    Args:
        f_builder: A function that takes Value parameters and returns a Value (the loss).
                   Called fresh each step to build a new computation graph.
        params_init: List of initial parameter values (floats)
        learning_rate: Step size
        num_steps: Number of gradient descent steps

    Returns:
        history: List of (params, loss) tuples
    """
    params = list(params_init)
    history = [(list(params), None)]

    for step in range(num_steps):
        # Create fresh Value objects for this step's computation graph
        param_values = [Value(p) for p in params]

        # Forward pass: build computation graph and compute loss
        loss = f_builder(param_values)
        history[-1] = (list(params), loss.data)

        # Backward pass: compute all gradients
        loss.backward()

        # Gradient descent update
        for i in range(len(params)):
            params[i] -= learning_rate * param_values[i].grad

        history.append((list(params), None))

    # Compute final loss
    param_values = [Value(p) for p in params]
    final_loss = f_builder(param_values)
    history[-1] = (list(params), final_loss.data)

    return history


def demonstrate_gradient_descent():
    """Minimize functions using gradient descent with both numerical and autograd gradients."""
    print("\n" + "=" * 70)
    print("PART 4: GRADIENT DESCENT")
    print("=" * 70)

    # --- Function 1: f(x, y) = x² + y² ---
    # Minimum at (0, 0). A simple bowl.
    print("\n--- Minimizing f(x,y) = x² + y² ---")
    print("    (Minimum at origin, starting from (3.0, 4.0))")
    print()

    # Method A: Numerical gradients
    f_bowl = lambda v: v[0]**2 + v[1]**2
    x0 = np.array([3.0, 4.0])
    history_num = gradient_descent_numerical(f_bowl, x0, learning_rate=0.1, num_steps=50)

    print("Using NUMERICAL gradients:")
    for step in [0, 1, 5, 10, 25, 50]:
        x, fx = history_num[step]
        print(f"  Step {step:3d}: x = ({x[0]:8.5f}, {x[1]:8.5f}), f(x) = {fx:.8f}")

    # Method B: Autograd
    print()
    f_bowl_autograd = lambda params: params[0] ** 2 + params[1] ** 2
    history_auto = gradient_descent_autograd(
        f_bowl_autograd, [3.0, 4.0], learning_rate=0.1, num_steps=50
    )

    print("Using AUTOGRAD gradients:")
    for step in [0, 1, 5, 10, 25, 50]:
        p, loss = history_auto[step]
        print(f"  Step {step:3d}: x = ({p[0]:8.5f}, {p[1]:8.5f}), f(x) = {loss:.8f}")

    # --- Function 2: Rosenbrock function ---
    # f(x, y) = (1-x)² + 100(y-x²)² — a classic optimization benchmark
    # Minimum at (1, 1). Much harder than the bowl.
    print("\n--- Minimizing the Rosenbrock function ---")
    print("    f(x,y) = (1-x)² + 100(y-x²)²")
    print("    (Minimum at (1,1), starting from (-1.0, 1.0))")
    print()

    def rosenbrock_autograd(params):
        x, y = params
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    history_rosen = gradient_descent_autograd(
        rosenbrock_autograd, [-1.0, 1.0], learning_rate=0.001, num_steps=10000
    )

    print("Using AUTOGRAD gradients (lr=0.001, 10000 steps):")
    for step in [0, 1, 100, 1000, 5000, 10000]:
        p, loss = history_rosen[step]
        print(f"  Step {step:5d}: x = ({p[0]:8.5f}, {p[1]:8.5f}), f(x) = {loss:.8f}")

    # --- Demonstrate: training a tiny "neuron" ---
    print("\n--- Training a single neuron with autograd ---")
    print("    Model: y = sigmoid(w1*x1 + w2*x2 + b)")
    print("    Target: learn AND gate (both inputs must be 1)")
    print()

    # AND gate training data
    data = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 0.0),
        ([1.0, 0.0], 0.0),
        ([1.0, 1.0], 1.0),
    ]

    # Initialize weights
    w1, w2, b = 0.1, -0.2, 0.0
    lr = 1.0

    print(f"Initial weights: w1={w1}, w2={w2}, b={b}")
    print()

    for epoch in range(200):
        total_loss_val = 0.0

        for inputs, target in data:
            # Create fresh computation graph
            w1_v = Value(w1)
            w2_v = Value(w2)
            b_v = Value(b)

            # Forward pass
            x1 = Value(inputs[0])
            x2 = Value(inputs[1])
            z = w1_v * x1 + w2_v * x2 + b_v
            pred = z.sigmoid()

            # Mean squared error loss (for one sample)
            diff = pred - Value(target)
            loss = diff ** 2

            # Backward pass
            loss.backward()

            # Accumulate loss for reporting
            total_loss_val += loss.data

            # Update weights
            w1 -= lr * w1_v.grad
            w2 -= lr * w2_v.grad
            b -= lr * b_v.grad

        if epoch % 40 == 0 or epoch == 199:
            print(f"  Epoch {epoch:3d}: loss = {total_loss_val:.6f}, "
                  f"w1 = {w1:.4f}, w2 = {w2:.4f}, b = {b:.4f}")

    # Test the trained neuron
    print("\nTrained neuron predictions:")
    for inputs, target in data:
        w1_v = Value(w1)
        w2_v = Value(w2)
        b_v = Value(b)
        z = w1_v * Value(inputs[0]) + w2_v * Value(inputs[1]) + b_v
        pred = z.sigmoid()
        print(f"  Input: {inputs} → Prediction: {pred.data:.4f} (target: {target})")


# =============================================================================
# PART 5: ADVANCED DEMONSTRATIONS
# =============================================================================

def demonstrate_computation_graph():
    """
    Visualize what the computation graph looks like by tracing through a complete
    forward and backward pass manually.
    """
    print("\n" + "=" * 70)
    print("PART 5: COMPUTATION GRAPH WALKTHROUGH")
    print("=" * 70)

    print("""
We trace through f(a, b) = (a + b) * sigmoid(a) step by step.

Let a = 2.0, b = 1.0.
    """)

    a = Value(2.0)
    b = Value(1.0)

    # Step-by-step forward pass
    c = a + b                    # c = 3.0
    s = a.sigmoid()              # s = sigmoid(2) = 0.8808
    f = c * s                    # f = 3.0 * 0.8808 = 2.6424

    print("FORWARD PASS (building the graph):")
    print(f"  c = a + b = {c.data:.4f}")
    print(f"  s = sigmoid(a) = {s.data:.4f}")
    print(f"  f = c * s = {f.data:.4f}")

    print("""
Computation graph:

    a ──────┬──── [+] ──── c ────┐
            │                    │
    b ──────┘                  [*] ──── f
            │                    │
    a ─── [sig] ──── s ─────────┘

Note: 'a' appears TWICE in the graph (once in the sum, once in sigmoid).
The chain rule must sum gradient contributions from BOTH paths.
    """)

    # Backward pass
    f.backward()

    sig_a = 1.0 / (1.0 + math.exp(-2.0))
    dsig_a = sig_a * (1 - sig_a)

    print("BACKWARD PASS (propagating gradients):")
    print(f"  ∂f/∂f = 1.0 (base case)")
    print()
    print(f"  f = c * s:")
    print(f"    ∂f/∂c = s = {s.data:.4f}")
    print(f"    ∂f/∂s = c = {c.data:.4f}")
    print()
    print(f"  s = sigmoid(a):")
    print(f"    ∂f/∂a (via sigmoid path) = ∂f/∂s * sigmoid'(a)")
    print(f"                              = {c.data:.4f} * {dsig_a:.4f}")
    print(f"                              = {c.data * dsig_a:.4f}")
    print()
    print(f"  c = a + b:")
    print(f"    ∂f/∂a (via addition path) = ∂f/∂c * 1 = {s.data:.4f}")
    print(f"    ∂f/∂b = ∂f/∂c * 1 = {s.data:.4f}")
    print()
    print(f"  Total ∂f/∂a = (via +) + (via sigmoid)")
    print(f"              = {s.data:.4f} + {c.data * dsig_a:.4f}")
    print(f"              = {s.data + c.data * dsig_a:.4f}")
    print()
    print(f"  Autograd result:  ∂f/∂a = {a.grad:.4f}")
    print(f"  Autograd result:  ∂f/∂b = {b.grad:.4f}")
    print()

    # Numerical verification
    def f_check(vals):
        a_v, b_v = vals
        sig = 1.0 / (1.0 + np.exp(-a_v))
        return (a_v + b_v) * sig

    num_grad = numerical_gradient(f_check, np.array([2.0, 1.0]))
    print(f"  Numerical check: ∂f/∂a = {num_grad[0]:.4f}, ∂f/∂b = {num_grad[1]:.4f}")


def demonstrate_autograd_vs_numerical_comprehensive():
    """
    Run autograd through a gauntlet of test cases to build confidence
    that the implementation is correct.
    """
    print("\n" + "=" * 70)
    print("COMPREHENSIVE AUTOGRAD VERIFICATION")
    print("=" * 70)

    tests_passed = 0
    tests_total = 0

    def check(name, f_autograd, f_numpy, params, tol=1e-4):
        """Test autograd gradients against numerical gradients."""
        nonlocal tests_passed, tests_total
        tests_total += 1

        # Autograd
        values = [Value(p) for p in params]
        result = f_autograd(values)
        result.backward()
        auto_grads = [v.grad for v in values]

        # Numerical
        num_grads = numerical_gradient(f_numpy, np.array(params))

        max_err = max(abs(a - n) for a, n in zip(auto_grads, num_grads))
        passed = max_err < tol

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name:45s} max_error = {max_err:.2e}")

        if passed:
            tests_passed += 1
        return passed

    print()

    # Test 1: Addition
    check("a + b",
          lambda v: v[0] + v[1],
          lambda v: v[0] + v[1],
          [3.0, 4.0])

    # Test 2: Multiplication
    check("a * b",
          lambda v: v[0] * v[1],
          lambda v: v[0] * v[1],
          [3.0, 4.0])

    # Test 3: Power
    check("a^3",
          lambda v: v[0] ** 3,
          lambda v: v[0] ** 3,
          [2.5])

    # Test 4: Nested operations
    check("(a + b) * (a - b)",
          lambda v: (v[0] + v[1]) * (v[0] - v[1]),
          lambda v: (v[0] + v[1]) * (v[0] - v[1]),
          [3.0, 2.0])

    # Test 5: Division
    check("a / b",
          lambda v: v[0] / v[1],
          lambda v: v[0] / v[1],
          [6.0, 3.0])

    # Test 6: Sigmoid
    check("sigmoid(a)",
          lambda v: v[0].sigmoid(),
          lambda v: 1.0 / (1.0 + np.exp(-v[0])),
          [1.5])

    # Test 7: Tanh
    check("tanh(a)",
          lambda v: v[0].tanh(),
          lambda v: np.tanh(v[0]),
          [-0.7])

    # Test 8: ReLU
    check("relu(a) with a > 0",
          lambda v: v[0].relu(),
          lambda v: max(0, v[0]),
          [2.0])

    check("relu(a) with a < 0",
          lambda v: v[0].relu(),
          lambda v: max(0, v[0]),
          [-2.0])

    # Test 9: Exp
    check("exp(a)",
          lambda v: v[0].exp(),
          lambda v: np.exp(v[0]),
          [1.0])

    # Test 10: Log
    check("log(a)",
          lambda v: v[0].log(),
          lambda v: np.log(v[0]),
          [2.0])

    # Test 11: Complex composition
    check("sigmoid(a*b + c^2)",
          lambda v: (v[0] * v[1] + v[2] ** 2).sigmoid(),
          lambda v: 1.0 / (1.0 + np.exp(-(v[0]*v[1] + v[2]**2))),
          [1.0, -2.0, 0.5])

    # Test 12: Variable reuse (gradient accumulation)
    check("a * a (= a²)",
          lambda v: v[0] * v[0],
          lambda v: v[0] ** 2,
          [3.0])

    # Test 13: Deep composition
    check("tanh(sigmoid(a * b))",
          lambda v: (v[0] * v[1]).sigmoid().tanh(),
          lambda v: np.tanh(1.0 / (1.0 + np.exp(-(v[0]*v[1])))),
          [0.5, 1.5])

    # Test 14: Polynomial
    check("a^3 + 2*a^2 - 5*a + 3",
          lambda v: v[0]**3 + 2*v[0]**2 - 5*v[0] + 3,
          lambda v: v[0]**3 + 2*v[0]**2 - 5*v[0] + 3,
          [2.0])

    # Test 15: Multi-path with exp and log
    check("log(exp(a) + exp(b)) (logsumexp)",
          lambda v: (v[0].exp() + v[1].exp()).log(),
          lambda v: np.log(np.exp(v[0]) + np.exp(v[1])),
          [1.0, 2.0])

    # Test 16: Chain of tanh (deep chain rule)
    check("tanh(tanh(tanh(a)))",
          lambda v: v[0].tanh().tanh().tanh(),
          lambda v: np.tanh(np.tanh(np.tanh(v[0]))),
          [0.5])

    print(f"\nResults: {tests_passed}/{tests_total} tests passed.")


# =============================================================================
# MAIN — Run all demonstrations
# =============================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          CALCULUS FROM SCRATCH: IMPLEMENTATION MODULE              ║")
    print("║                                                                    ║")
    print("║  Numerical differentiation, automatic differentiation (autograd),  ║")
    print("║  Jacobians, Hessians, and gradient descent — all from first        ║")
    print("║  principles.                                                       ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    demonstrate_numerical_differentiation()
    demonstrate_autograd()
    demonstrate_jacobian_hessian()
    demonstrate_gradient_descent()
    demonstrate_computation_graph()
    demonstrate_autograd_vs_numerical_comprehensive()

    print("\n" + "=" * 70)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)
    print("""
Key takeaways:
  1. Numerical differentiation is simple but approximate — useful for verification.
  2. Automatic differentiation gives EXACT gradients by tracking operations
     and applying the chain rule on the computation graph.
  3. The Value class is a miniature autograd engine — the same algorithm that
     powers PyTorch, TensorFlow, and JAX, just for scalars instead of tensors.
  4. Gradient descent uses these gradients to iteratively minimize functions.
  5. Training a neural network = gradient descent on a loss function, with
     gradients computed by backpropagation (reverse-mode autodiff).

Next: 03_probability_statistics — the mathematical language of uncertainty.
""")

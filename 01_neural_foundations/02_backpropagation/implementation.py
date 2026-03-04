"""
Backpropagation from Scratch — A Micrograd-Style Autograd Engine
================================================================

The core insight: backpropagation is just the chain rule applied systematically
through a computation graph. Every operation records how to compute its local
gradient, and backward() chains them all together.

This implementation builds a complete automatic differentiation engine in pure Python,
then uses it to train a neural network on XOR — proving that multi-layer networks
with backprop solve what single perceptrons cannot.
"""

import math
import random


# ==============================================================================
# Part 1: The Value Class — The Heart of Autograd
# ==============================================================================

class Value:
    """
    A scalar value that tracks its computation history and can compute gradients.

    Every Value knows:
    - Its data (the actual number)
    - Its gradient (dL/d(self), computed during backward pass)
    - Its children (what Values produced it)
    - Its backward function (how to propagate gradients to children)

    This is the same principle behind PyTorch's autograd, JAX's autodiff, etc.
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None  # no-op by default
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # --- Arithmetic Operations ---
    # Each one: (1) computes the forward pass, (2) defines the backward pass

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1
            # Gradients accumulate (+=) because a value might be used multiple times
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return Value(other) * (self ** -1)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supporting int/float powers"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # d(a^n)/da = n * a^(n-1)
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    # --- Activation Functions ---

    def relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')

        def _backward():
            # d(relu(x))/dx = 1 if x > 0, else 0
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            # d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d(tanh(x))/dx = 1 - tanh(x)^2
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            # d(e^x)/dx = e^x
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            # d(ln(x))/dx = 1/x
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    # --- The Backward Pass ---

    def backward(self):
        """
        Compute gradients for all Values in the computation graph.

        Algorithm:
        1. Topological sort: order nodes so children come before parents
        2. Set output gradient to 1.0 (dL/dL = 1)
        3. Walk backward, calling each node's _backward()

        This is backpropagation. The topological sort ensures we always have
        the full gradient before propagating further back.
        """
        # Topological sort using DFS
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Set output gradient and propagate backward
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# ==============================================================================
# Part 2: Neural Network Built on the Autograd Engine
# ==============================================================================

class Neuron:
    """A single neuron: weighted sum + bias + activation."""

    def __init__(self, n_inputs, activation='tanh'):
        # Xavier initialization: weights ~ Uniform(-1/sqrt(n), 1/sqrt(n))
        scale = 1.0 / math.sqrt(n_inputs)
        self.w = [Value(random.uniform(-scale, scale), label=f'w{i}')
                   for i in range(n_inputs)]
        self.b = Value(0.0, label='b')
        self.activation = activation

    def __call__(self, x):
        # Weighted sum: w·x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        # Apply activation
        if self.activation == 'tanh':
            return act.tanh()
        elif self.activation == 'sigmoid':
            return act.sigmoid()
        elif self.activation == 'relu':
            return act.relu()
        elif self.activation == 'linear':
            return act
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def parameters(self):
        return self.w + [self.b]


class Layer:
    """A layer of neurons."""

    def __init__(self, n_inputs, n_outputs, activation='tanh'):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_outputs)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP_Autograd:
    """
    Multi-layer perceptron built entirely on our autograd engine.
    No NumPy, no PyTorch — just the Value class above.
    """

    def __init__(self, n_inputs, layer_sizes, activations=None):
        sizes = [n_inputs] + layer_sizes
        if activations is None:
            # tanh for hidden layers, sigmoid for output
            activations = ['tanh'] * (len(layer_sizes) - 1) + ['sigmoid']
        self.layers = [Layer(sizes[i], sizes[i+1], activations[i])
                       for i in range(len(layer_sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


# ==============================================================================
# Part 3: Gradient Verification
# ==============================================================================

def numerical_gradient(f, x, h=1e-7):
    """
    Compute gradient numerically using central differences.

    This is slow but correct — we use it to verify our autograd.

    df/dx ≈ (f(x+h) - f(x-h)) / (2h)
    """
    grad = []
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad


def verify_gradients():
    """Compare autograd gradients with numerical gradients."""
    print("=" * 60)
    print("GRADIENT VERIFICATION")
    print("Comparing autograd vs numerical gradients")
    print("=" * 60)

    # Test 1: Simple expression f = (a*b + c)^2
    print("\nTest 1: f = (a*b + c)^2")
    a = Value(2.0, label='a')
    b = Value(3.0, label='b')
    c = Value(4.0, label='c')
    f = (a * b + c) ** 2
    f.backward()

    print(f"  Autograd:  df/da = {a.grad:.6f}, df/db = {b.grad:.6f}, df/dc = {c.grad:.6f}")

    # Numerical verification
    def f_func(vals):
        return (vals[0] * vals[1] + vals[2]) ** 2

    num_grad = numerical_gradient(f_func, [2.0, 3.0, 4.0])
    print(f"  Numerical: df/da = {num_grad[0]:.6f}, df/db = {num_grad[1]:.6f}, df/dc = {num_grad[2]:.6f}")

    match = all(abs(ag - ng) < 1e-4 for ag, ng in zip([a.grad, b.grad, c.grad], num_grad))
    print(f"  Match: {'YES' if match else 'NO'}")

    # Test 2: Sigmoid composition
    print("\nTest 2: f = sigmoid(a*b) + tanh(c)")
    a = Value(1.0, label='a')
    b = Value(-2.0, label='b')
    c = Value(0.5, label='c')
    f = (a * b).sigmoid() + c.tanh()
    f.backward()

    print(f"  Autograd:  df/da = {a.grad:.6f}, df/db = {b.grad:.6f}, df/dc = {c.grad:.6f}")

    def f_func2(vals):
        s = 1.0 / (1.0 + math.exp(-(vals[0] * vals[1])))
        return s + math.tanh(vals[2])

    num_grad2 = numerical_gradient(f_func2, [1.0, -2.0, 0.5])
    print(f"  Numerical: df/da = {num_grad2[0]:.6f}, df/db = {num_grad2[1]:.6f}, df/dc = {num_grad2[2]:.6f}")

    match = all(abs(ag - ng) < 1e-4 for ag, ng in zip([a.grad, b.grad, c.grad], num_grad2))
    print(f"  Match: {'YES' if match else 'NO'}")

    # Test 3: Reuse of variables (gradient accumulation)
    print("\nTest 3: f = a*a + a*b (a is used multiple times)")
    a = Value(3.0, label='a')
    b = Value(2.0, label='b')
    f = a * a + a * b  # df/da = 2a + b = 8, df/db = a = 3
    f.backward()

    print(f"  Autograd:  df/da = {a.grad:.6f} (expected 8.0), df/db = {b.grad:.6f} (expected 3.0)")
    print(f"  Correct: {'YES' if abs(a.grad - 8.0) < 1e-6 and abs(b.grad - 3.0) < 1e-6 else 'NO'}")


# ==============================================================================
# Part 4: Training on XOR — Proving Backprop Works
# ==============================================================================

def train_xor():
    """
    Train a neural network on XOR using our autograd engine.

    XOR cannot be solved by a single perceptron (it's not linearly separable).
    A 2-layer network with backprop can solve it — this is the fundamental
    demonstration that multi-layer networks + gradient-based learning work.

    XOR Truth Table:
        (0,0) -> 0
        (0,1) -> 1
        (1,0) -> 1
        (1,1) -> 0
    """
    print("\n" + "=" * 60)
    print("TRAINING ON XOR")
    print("Proving multi-layer networks + backprop solve non-linear problems")
    print("=" * 60)

    # Dataset
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]

    # Network: 2 inputs -> 4 hidden (tanh) -> 1 output (sigmoid)
    random.seed(42)
    net = MLP_Autograd(2, [4, 1], activations=['tanh', 'sigmoid'])

    n_params = len(net.parameters())
    print(f"\nNetwork: 2 -> 4 -> 1 ({n_params} parameters)")
    print(f"Hidden activation: tanh, Output activation: sigmoid")
    print(f"Loss: Binary cross-entropy\n")

    learning_rate = 0.5

    for epoch in range(501):
        # Forward pass: compute predictions and loss
        total_loss = Value(0.0)

        for xi, yi in zip(X, y):
            # Convert inputs to Values
            x_vals = [Value(v) for v in xi]
            pred = net(x_vals)

            # Binary cross-entropy: L = -(y*log(p) + (1-y)*log(1-p))
            # Clip predictions to avoid log(0)
            eps = 1e-7
            p_clipped = pred if pred.data > eps else Value(eps)
            p_clipped = p_clipped if p_clipped.data < (1 - eps) else Value(1 - eps)

            if yi == 1:
                loss = -(p_clipped.log())
            else:
                loss = -((1 - p_clipped).log())

            total_loss = total_loss + loss

        # Average loss
        avg_loss = total_loss * (1.0 / len(X))

        # Backward pass
        net.zero_grad()
        avg_loss.backward()

        # Update parameters (SGD)
        for p in net.parameters():
            p.data -= learning_rate * p.grad

        # Print progress
        if epoch % 100 == 0:
            predictions = []
            for xi in X:
                x_vals = [Value(v) for v in xi]
                pred = net(x_vals)
                predictions.append(pred.data)

            print(f"  Epoch {epoch:4d} | Loss: {avg_loss.data:.6f}")
            for xi, yi, pi in zip(X, y, predictions):
                correct = "OK" if (pi > 0.5) == (yi == 1) else "WRONG"
                print(f"    {xi} -> {pi:.4f} (target: {yi}) [{correct}]")
            print()

    # Final results
    print("FINAL PREDICTIONS:")
    all_correct = True
    for xi, yi in zip(X, y):
        x_vals = [Value(v) for v in xi]
        pred = net(x_vals)
        correct = (pred.data > 0.5) == (yi == 1)
        all_correct = all_correct and correct
        print(f"  {xi} -> {pred.data:.4f} (target: {yi}) {'CORRECT' if correct else 'WRONG'}")

    print(f"\nXOR solved: {'YES' if all_correct else 'NO'}")
    if all_correct:
        print("This proves: backpropagation enables multi-layer networks to learn non-linear functions.")


# ==============================================================================
# Part 5: Visualizing Gradient Flow
# ==============================================================================

def trace_gradient_flow():
    """
    Show exactly how gradients flow backward through a computation graph.

    This makes the chain rule concrete: each node multiplies by its local
    gradient and passes the result to its children.
    """
    print("\n" + "=" * 60)
    print("GRADIENT FLOW VISUALIZATION")
    print("Tracing the chain rule through a computation graph")
    print("=" * 60)

    # Simple network: 2 inputs, 1 hidden neuron, 1 output
    # Forward: h = tanh(w1*x1 + w2*x2 + b1), o = sigmoid(w3*h + b2)

    x1 = Value(1.0, label='x1')
    x2 = Value(0.5, label='x2')
    w1 = Value(0.3, label='w1')
    w2 = Value(-0.5, label='w2')
    b1 = Value(0.1, label='b1')
    w3 = Value(0.7, label='w3')
    b2 = Value(-0.2, label='b2')

    # Forward pass (showing each step)
    print("\n--- Forward Pass ---")
    z1 = w1 * x1
    print(f"  z1 = w1*x1 = {w1.data:.4f} * {x1.data:.4f} = {z1.data:.4f}")

    z2 = w2 * x2
    print(f"  z2 = w2*x2 = {w2.data:.4f} * {x2.data:.4f} = {z2.data:.4f}")

    z3 = z1 + z2 + b1
    print(f"  z3 = z1 + z2 + b1 = {z1.data:.4f} + {z2.data:.4f} + {b1.data:.4f} = {z3.data:.4f}")

    h = z3.tanh()
    print(f"  h  = tanh(z3) = tanh({z3.data:.4f}) = {h.data:.4f}")

    z4 = w3 * h + b2
    print(f"  z4 = w3*h + b2 = {w3.data:.4f} * {h.data:.4f} + {b2.data:.4f} = {z4.data:.4f}")

    o = z4.sigmoid()
    print(f"  o  = sigmoid(z4) = sigmoid({z4.data:.4f}) = {o.data:.4f}")

    # Target = 1, so loss = -log(o)
    target = 1.0
    loss = -(o.log())
    print(f"  loss = -log(o) = -log({o.data:.4f}) = {loss.data:.4f}")

    # Backward pass
    loss.backward()

    print("\n--- Backward Pass (Chain Rule) ---")
    print(f"  dL/dL    = {loss.grad:.6f}")
    print(f"  dL/do    = {o.grad:.6f}  (= -1/o = -1/{o.data:.4f})")
    print(f"  dL/dz4   = {z4.grad:.6f}  (= dL/do * sigmoid'(z4))")
    print(f"  dL/dw3   = {w3.grad:.6f}  (= dL/dz4 * h)")
    print(f"  dL/dh    = {h.grad:.6f}  (= dL/dz4 * w3)")
    print(f"  dL/dz3   = {z3.grad:.6f}  (= dL/dh * tanh'(z3))")
    print(f"  dL/dw1   = {w1.grad:.6f}  (= dL/dz3 * x1)")
    print(f"  dL/dw2   = {w2.grad:.6f}  (= dL/dz3 * x2)")
    print(f"  dL/db1   = {b1.grad:.6f}  (= dL/dz3 * 1)")
    print(f"  dL/db2   = {b2.grad:.6f}  (= dL/dz4 * 1)")

    print("\nKey insight: each gradient is the product of ALL local gradients")
    print("along the path from the loss to that parameter. THAT'S the chain rule.")
    print(f"\nFor example, dL/dw1 = dL/do * do/dz4 * dz4/dh * dh/dz3 * dz3/dw1")
    print(f"Each factor is a local derivative. The product is the full gradient.")


# ==============================================================================
# Part 6: Why Deep Networks Have Gradient Problems
# ==============================================================================

def gradient_problems():
    """
    Demonstrate vanishing and exploding gradients in deep networks.

    With sigmoid/tanh activations, gradients shrink exponentially with depth
    because sigmoid'(x) <= 0.25 and tanh'(x) <= 1. After many layers,
    the gradient product approaches zero.
    """
    print("\n" + "=" * 60)
    print("VANISHING & EXPLODING GRADIENTS")
    print("Why deep networks are hard to train")
    print("=" * 60)

    print("\n--- Vanishing Gradients (tanh) ---")
    print("Each tanh derivative is at most 1.0, typically < 1.0")
    print("After many layers, gradient = product of many small numbers -> 0\n")

    random.seed(42)
    for depth in [2, 5, 10, 20]:
        # Create a chain: x -> tanh -> tanh -> ... -> tanh -> output
        x = Value(0.5)
        current = x
        for _ in range(depth):
            w = Value(random.uniform(-1, 1))
            current = (current * w).tanh()

        loss = (current - Value(1.0)) ** 2
        loss.backward()

        print(f"  Depth {depth:2d}: |dL/dx| = {abs(x.grad):.10f}")

    print("\n  Observation: gradient shrinks exponentially with depth!")
    print("  This is why ReLU was revolutionary: relu'(x) = 1 for x > 0")

    print("\n--- ReLU Helps ---")
    random.seed(42)
    for depth in [2, 5, 10, 20]:
        x = Value(0.5)
        current = x
        for _ in range(depth):
            w = Value(random.uniform(0.5, 1.5))  # positive weights
            current = (current * w).relu()

        loss = (current - Value(1.0)) ** 2
        loss.backward()

        print(f"  Depth {depth:2d}: |dL/dx| = {abs(x.grad):.10f}")

    print("\n  ReLU gradients don't vanish (but can explode with large weights).")
    print("  This is why initialization matters: keep gradients ~1 across layers.")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("BACKPROPAGATION FROM SCRATCH")
    print("A complete autograd engine + neural network training")
    print("No PyTorch, no TensorFlow — just Python and math.\n")

    # 1. Verify our autograd computes correct gradients
    verify_gradients()

    # 2. Show gradient flow through a computation graph
    trace_gradient_flow()

    # 3. Train on XOR — the fundamental test
    train_xor()

    # 4. Demonstrate gradient problems in deep networks
    gradient_problems()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. Backprop IS the chain rule, applied systematically through a computation graph.
2. Each Value tracks how it was created and how to compute its local gradient.
3. backward() walks the graph in reverse topological order, multiplying local gradients.
4. This is exactly what PyTorch/TensorFlow do — just with tensors instead of scalars.
5. Deep networks suffer from vanishing gradients (tanh/sigmoid) — ReLU helps.
6. Multi-layer networks + backprop can solve XOR, proving non-linear learning works.

Next: See the MLP module for a full NumPy-based implementation with matrix operations.
""")

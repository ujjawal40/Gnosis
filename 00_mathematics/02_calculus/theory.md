# Calculus for Machine Learning

**From limits to backpropagation — the mathematics of change that makes learning possible.**

Calculus is the language of change. Neural networks learn by changing their parameters to reduce error. Every step of that process — computing how wrong the network is, figuring out which direction to adjust each parameter, deciding how far to move — is calculus. Without calculus, there is no learning.

This module derives everything from first principles: what a derivative actually is, why the chain rule works, what gradients mean geometrically, and how all of it comes together in the algorithms that train every neural network ever built.

---

## 1. Derivatives: The Rate of Change

### 1.1 The Fundamental Definition

The derivative of a function f at a point x is defined as:

```
f'(x) = lim    f(x + h) - f(x)
        h→0  ─────────────────────
                      h
```

This is the **limit of the difference quotient**. Before the limit, the expression `(f(x+h) - f(x)) / h` is the slope of the **secant line** between points `(x, f(x))` and `(x+h, f(x+h))`. As `h → 0`, the secant line becomes the **tangent line**, and the difference quotient becomes the **instantaneous rate of change**.

The derivative exists if and only if this limit exists and is finite. Functions where this limit does not exist at a point are **not differentiable** at that point (e.g., |x| at x=0, where the left and right limits disagree).

### 1.2 Geometric Interpretation

The derivative f'(a) is the **slope of the tangent line** to y = f(x) at x = a.

The tangent line itself is the **best linear approximation** to f near a:

```
f(x) ≈ f(a) + f'(a)(x - a)     for x near a
```

This is not just a visual aid — it is the foundation of optimization. Gradient descent works because, locally, every smooth function looks like a line (in 1D) or a hyperplane (in higher dimensions), and we know how to optimize linear functions.

### 1.3 Computing Derivatives: The Basic Rules

From the limit definition, we can derive all the standard rules:

**Power Rule:** If f(x) = xⁿ, then f'(x) = nxⁿ⁻¹.

Proof sketch:
```
f'(x) = lim  (x+h)ⁿ - xⁿ
        h→0  ─────────────
                   h
```
Expand (x+h)ⁿ using the binomial theorem. All terms except the first two have h² or higher powers, which vanish as h → 0, leaving nxⁿ⁻¹.

**Sum Rule:** (f + g)' = f' + g' — the derivative of a sum is the sum of derivatives.

**Product Rule:** (fg)' = f'g + fg' — differentiation does not distribute over multiplication.

**Quotient Rule:** (f/g)' = (f'g - fg') / g² — derived from the product rule applied to f · g⁻¹.

**Exponential:** If f(x) = eˣ, then f'(x) = eˣ. The exponential function is its own derivative — this is actually what *defines* the number e.

**Logarithm:** If f(x) = ln(x), then f'(x) = 1/x. Follows from differentiating both sides of e^(ln x) = x.

### 1.4 Why Derivatives Matter for ML

Every loss function in ML takes the form L(θ) where θ are the model's parameters. Training means solving:

```
θ* = argmin L(θ)
         θ
```

The derivative ∂L/∂θ tells us the direction and rate at which the loss changes when we change a parameter. This is the entire basis of gradient-based optimization.

---

## 2. Partial Derivatives: Functions of Multiple Variables

### 2.1 Definition

Real-world functions depend on many variables. A neural network with millions of parameters defines a loss function L(θ₁, θ₂, ..., θₙ) over millions of dimensions.

The **partial derivative** of f with respect to xᵢ, holding all other variables fixed:

```
∂f        f(x₁, ..., xᵢ + h, ..., xₙ) - f(x₁, ..., xᵢ, ..., xₙ)
──── = lim ──────────────────────────────────────────────────────────
∂xᵢ   h→0                            h
```

It is literally just an ordinary derivative, taken with respect to one variable while pretending all others are constants.

### 2.2 Example

Let f(x, y) = x²y + sin(y).

```
∂f/∂x = 2xy        (treat y as a constant, differentiate w.r.t. x)
∂f/∂y = x² + cos(y) (treat x as a constant, differentiate w.r.t. y)
```

### 2.3 Geometric Interpretation

For f: R² → R, the partial derivative ∂f/∂x at point (a,b) is the slope of f along the x-direction, holding y fixed at b. Visually, if you slice the surface z = f(x,y) with the plane y = b, you get a curve; ∂f/∂x is the slope of that curve.

### 2.4 Higher-Order Partial Derivatives

We can differentiate again:

```
∂²f/∂x² = ∂/∂x (∂f/∂x)     — second derivative w.r.t. x
∂²f/∂x∂y = ∂/∂x (∂f/∂y)    — mixed partial derivative
```

**Clairaut's Theorem (Schwarz's Theorem):** If the mixed partials are continuous, then the order of differentiation does not matter:

```
∂²f/∂x∂y = ∂²f/∂y∂x
```

This symmetry is important — it means the Hessian matrix (Section 6) is symmetric for "nice" functions.

---

## 3. The Chain Rule

**This is the single most important concept in deep learning.**

### 3.1 Single-Variable Chain Rule

If y = f(g(x)), meaning we compose two functions, then:

```
dy     dy   dg
── = ── · ──
dx     dg   dx
```

Or equivalently: (f ∘ g)'(x) = f'(g(x)) · g'(x).

**Intuition:** If g changes x by a factor of 3, and f changes g's output by a factor of 2, then the overall change is 2 × 3 = 6. Rates of change multiply through composition.

### 3.2 Proof of the Chain Rule

Let y = f(u) where u = g(x). We want dy/dx.

```
dy     Δy         Δy   Δu
── = lim ── = lim (── · ──)
dx   Δx→0 Δx   Δx→0 Δu   Δx
```

As Δx → 0, Δu → 0 (because g is continuous), so:

```
    Δy        Δu       dy   du
lim ── · lim ── = ── · ──
Δu→0 Δu   Δx→0 Δx    du   dx
```

(The formal proof requires care when Δu = 0, but the intuition is clean.)

### 3.3 Multivariable Chain Rule

If z = f(x, y) where x = x(t) and y = y(t), then:

```
dz     ∂f dx     ∂f dy
── = ── · ── + ── · ──
dt     ∂x dt     ∂y dt
```

More generally, if z = f(x₁, x₂, ..., xₙ) and each xᵢ = xᵢ(t₁, t₂, ..., tₘ), then:

```
∂z       n   ∂f   ∂xᵢ
─── = Σ  ── · ───
∂tⱼ    i=1  ∂xᵢ  ∂tⱼ
```

This is the sum-over-paths rule: to find how z changes with tⱼ, sum over all paths through which tⱼ can influence z.

### 3.4 The Chain Rule in Neural Networks

A neural network is a composition of functions:

```
output = fₙ(fₙ₋₁(...f₂(f₁(input))))
```

Each fᵢ is a layer: a linear transformation followed by a nonlinear activation. The chain rule lets us compute how the loss changes with respect to parameters in any layer:

```
∂L       ∂L     ∂fₙ     ∂fₙ₋₁         ∂f₁
──── = ──── · ──── · ────── · ... · ────
∂θ₁     ∂fₙ    ∂fₙ₋₁   ∂fₙ₋₂         ∂θ₁
```

**This is backpropagation.** The chain rule, applied to a computation graph, gives us an efficient algorithm to compute all gradients in a single backward pass. The "back" in backpropagation refers to applying the chain rule from the output backward through each layer.

### 3.5 Chain Rule on Computation Graphs

Any mathematical expression can be represented as a **directed acyclic graph (DAG)** where:
- Leaf nodes are inputs and parameters
- Internal nodes are operations (+, ×, exp, etc.)
- Edges represent data flow

The chain rule on a graph says: the gradient of the output with respect to any node is the **sum over all paths** from that node to the output, where each path contributes the product of local derivatives along the path.

In practice, we compute this efficiently with **reverse-mode automatic differentiation** (backpropagation):

1. **Forward pass:** Compute all intermediate values, building the graph.
2. **Backward pass:** Starting from the output, propagate gradients backward. At each node, multiply the incoming gradient by the local derivative and pass it to each input.

For a node that computes z = f(x, y):
```
∂L/∂x = (∂L/∂z) · (∂z/∂x)     — chain rule
∂L/∂y = (∂L/∂z) · (∂z/∂y)     — chain rule
```

The term ∂L/∂z is the gradient flowing in from above (already computed). The terms ∂z/∂x and ∂z/∂y are **local derivatives** that each operation knows how to compute.

This is the algorithm implemented in PyTorch's autograd, TensorFlow's GradientTape, and the autograd engine we build in implementation.py.

---

## 4. Gradients

### 4.1 Definition

The **gradient** of a scalar-valued function f: Rⁿ → R is the vector of all partial derivatives:

```
∇f = (∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ)
```

The gradient is a **vector** that lives in the same space as the input. If the input is a point in Rⁿ, the gradient is also a vector in Rⁿ.

### 4.2 Direction of Steepest Ascent

The **directional derivative** of f in direction u (a unit vector) is:

```
Dᵤf = ∇f · u = |∇f| cos(θ)
```

where θ is the angle between ∇f and u. This is maximized when θ = 0, i.e., when u points in the same direction as ∇f.

Therefore: **the gradient points in the direction of steepest ascent.** The negative gradient points in the direction of **steepest descent** — which is exactly the direction gradient descent moves.

### 4.3 Magnitude of the Gradient

The magnitude |∇f| tells us **how steep** the function is at that point. If |∇f| is large, the function is changing rapidly; if |∇f| ≈ 0, the function is nearly flat (we may be near a minimum, maximum, or saddle point).

### 4.4 Gradient Descent

The update rule:

```
θ ← θ - α · ∇L(θ)
```

where α is the **learning rate**. This moves parameters in the direction of steepest descent on the loss surface. Each step reduces the loss (for small enough α) because:

```
L(θ - α∇L) ≈ L(θ) - α|∇L|²
```

Since α > 0 and |∇L|² ≥ 0, the loss decreases. (This approximation comes from the first-order Taylor expansion — see Section 7.)

### 4.5 Gradients and Level Sets

The gradient ∇f at a point is **perpendicular to the level set** (contour line) of f passing through that point. This is because movement along a level set does not change f, so the directional derivative along the level set is zero, meaning ∇f must be orthogonal to the level set.

This is visible in contour plots of loss functions: gradient descent moves perpendicular to the contour lines.

---

## 5. Jacobian Matrices

### 5.1 When Functions Map Vectors to Vectors

A gradient handles functions f: Rⁿ → R (many inputs, one output). But what about functions F: Rⁿ → Rᵐ (many inputs, many outputs)?

Each output Fᵢ has its own gradient. Stack them into a matrix and you get the **Jacobian**:

```
        ┌ ∂F₁/∂x₁  ∂F₁/∂x₂  ...  ∂F₁/∂xₙ ┐
        │ ∂F₂/∂x₁  ∂F₂/∂x₂  ...  ∂F₂/∂xₙ │
J_F =   │    ⋮         ⋮       ⋱      ⋮     │
        └ ∂Fₘ/∂x₁  ∂Fₘ/∂x₂  ...  ∂Fₘ/∂xₙ ┘
```

The Jacobian is an m × n matrix. Entry (i,j) is ∂Fᵢ/∂xⱼ — how much the i-th output changes when the j-th input changes.

### 5.2 The Jacobian as a Linear Map

The Jacobian is the **best linear approximation** to F near a point:

```
F(x + Δx) ≈ F(x) + J_F · Δx
```

This is the multivariable generalization of f(x+h) ≈ f(x) + f'(x)h.

### 5.3 Jacobians in Neural Networks

Each layer of a neural network is a function from Rⁿ → Rᵐ (mapping one activation vector to the next). The derivative of a layer is its Jacobian.

The chain rule for composed vector functions uses Jacobians:

```
J_{F∘G} = J_F · J_G
```

Backpropagation computes the product of these Jacobians — but it never forms the full matrices. Instead, it computes **Jacobian-vector products** (JVPs) or **vector-Jacobian products** (VJPs), which are far more efficient.

In reverse-mode autodiff (backpropagation), we compute VJPs:

```
vᵀ · J_F
```

where v is the gradient flowing backward. This costs O(nm) per layer, not O(nm × mn) for forming the full Jacobian product.

### 5.4 Special Cases

- If F: R → R (scalar to scalar), the Jacobian is a 1×1 matrix — just the derivative f'(x).
- If f: Rⁿ → R (vector to scalar), the Jacobian is a 1×n matrix — the gradient (as a row vector).
- If F: R → Rᵐ (scalar to vector), the Jacobian is an m×1 matrix — a column vector of derivatives.

### 5.5 The Jacobian Determinant

For a square Jacobian (n = m), the **determinant** measures how F changes volumes locally:

```
|det(J_F)| = local volume scaling factor
```

If |det(J)| > 1, volumes expand. If |det(J)| < 1, volumes contract. If det(J) = 0, the function collapses a dimension (is locally non-invertible).

This appears in:
- **Normalizing flows:** The change-of-variables formula requires computing log|det(J)|. The entire design of flow architectures (RealNVP, GLOW, etc.) revolves around making this determinant efficient to compute.
- **Information geometry:** How parameter transformations affect Fisher information.

---

## 6. Hessian Matrices

### 6.1 Definition: Second-Order Information

The Hessian of a scalar function f: Rⁿ → R is the matrix of all second partial derivatives:

```
        ┌ ∂²f/∂x₁²     ∂²f/∂x₁∂x₂  ...  ∂²f/∂x₁∂xₙ ┐
        │ ∂²f/∂x₂∂x₁   ∂²f/∂x₂²    ...  ∂²f/∂x₂∂xₙ │
H_f =   │    ⋮              ⋮         ⋱       ⋮        │
        └ ∂²f/∂xₙ∂x₁   ∂²f/∂xₙ∂x₂  ...  ∂²f/∂xₙ²   ┘
```

The Hessian is the **Jacobian of the gradient**: H = J(∇f).

By Clairaut's theorem, the Hessian is **symmetric** (for sufficiently smooth functions).

### 6.2 Curvature

The Hessian encodes **curvature** — how the gradient itself changes as we move.

- The gradient tells us the slope (first-order information).
- The Hessian tells us how the slope changes (second-order information).

The **second-order Taylor approximation** is:

```
f(x + Δx) ≈ f(x) + ∇f(x)ᵀΔx + ½ Δxᵀ H Δx
```

The quadratic term ½ΔxᵀHΔx captures curvature: whether the function curves up (bowl-shaped) or down (hill-shaped) or has mixed curvature (saddle-shaped).

### 6.3 Eigenvalues and Curvature Directions

The eigenvalues of the Hessian correspond to the curvature along each eigenvector direction:

- **All eigenvalues > 0** → local minimum (curves up in every direction, positive definite)
- **All eigenvalues < 0** → local maximum (curves down in every direction, negative definite)
- **Mixed signs** → saddle point (curves up in some directions, down in others)
- **Some eigenvalues ≈ 0** → flat directions (the function is nearly linear)

The **condition number** κ = λ_max / λ_min measures how much the curvature varies across directions. High condition number means the loss landscape is shaped like a long narrow valley, which makes gradient descent inefficient (it zigzags down the valley instead of going straight to the minimum).

### 6.4 Newton's Method

Newton's method uses the Hessian to take the curvature into account:

```
θ ← θ - H⁻¹ · ∇L
```

Instead of moving in the gradient direction with a fixed step size, Newton's method moves in the direction H⁻¹∇L, which accounts for curvature. In a quadratic function, Newton's method converges in one step.

However, computing and inverting the Hessian is O(n²) storage and O(n³) computation, which is completely infeasible for neural networks with millions of parameters. This is why first-order methods (gradient descent, Adam) dominate in practice, and why second-order methods remain a research frontier.

### 6.5 Hessians in Deep Learning

Even though we rarely compute the full Hessian, understanding it matters:

- **Loss landscape analysis:** The eigenspectrum of the Hessian reveals whether minima are sharp or flat, which correlates with generalization.
- **Adaptive optimizers:** Adam and AdaGrad approximate diagonal Hessian information.
- **Natural gradient descent:** Uses the Fisher information matrix (related to the Hessian of the KL divergence) to account for the geometry of parameter space.
- **Pruning:** Hessian eigenvalues help identify which parameters can be removed with minimal impact (Optimal Brain Damage, Optimal Brain Surgeon).

---

## 7. Taylor Series

### 7.1 The Idea: Local Polynomial Approximation

A Taylor series approximates a function near a point using polynomials. The n-th order Taylor expansion of f around point a is:

```
f(x) = f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + f'''(a)(x-a)³/3! + ...
```

Or compactly:

```
           ∞   f⁽ⁿ⁾(a)
f(x) = Σ   ────── (x-a)ⁿ
          n=0   n!
```

Each term adds more detail:
- 0th order: f(x) ≈ f(a) — constant approximation (value only)
- 1st order: f(x) ≈ f(a) + f'(a)(x-a) — linear approximation (tangent line)
- 2nd order: f(x) ≈ f(a) + f'(a)(x-a) + ½f''(a)(x-a)² — quadratic (adds curvature)

### 7.2 Multivariable Taylor Expansion

For f: Rⁿ → R around point a:

```
f(x) ≈ f(a) + ∇f(a)ᵀ(x-a) + ½(x-a)ᵀ H(a) (x-a) + ...
```

This connects gradient (first order) and Hessian (second order) into a single framework.

### 7.3 Taylor Series and Common Functions

Some important Taylor series (around a = 0):

```
eˣ = 1 + x + x²/2! + x³/3! + ...                    (converges everywhere)
sin(x) = x - x³/3! + x⁵/5! - ...                     (converges everywhere)
cos(x) = 1 - x²/2! + x⁴/4! - ...                     (converges everywhere)
1/(1-x) = 1 + x + x² + x³ + ...                       (converges for |x| < 1)
ln(1+x) = x - x²/2 + x³/3 - x⁴/4 + ...              (converges for |x| ≤ 1)
```

### 7.4 Connection to Optimization

Gradient descent is optimization using the **1st-order Taylor approximation**: we assume the function is locally linear and step downhill.

Newton's method uses the **2nd-order Taylor approximation**: we assume the function is locally quadratic and jump to the minimum of the quadratic.

Higher-order methods could use the 3rd-order approximation, but the cost grows combinatorially and the benefit diminishes. In practice, 1st-order methods (SGD, Adam) work remarkably well for neural networks, partly because the stochasticity of minibatch gradients provides implicit regularization.

### 7.5 Approximation Error

The Taylor remainder theorem says the error of an n-th order approximation is:

```
R_n(x) = f⁽ⁿ⁺¹⁾(c) · (x-a)ⁿ⁺¹ / (n+1)!
```

for some c between a and x. This bounds how wrong the approximation can be — crucial for understanding when gradient descent's linear approximation is trustworthy (i.e., within the "trust region" where the step size is small enough).

---

## 8. Why This All Matters for Machine Learning

### 8.1 Gradient Descent IS Calculus

The training loop of every neural network is:

```python
for each batch:
    predictions = model(batch)          # forward pass (function evaluation)
    loss = loss_fn(predictions, targets) # compute scalar loss
    gradients = backward(loss)           # chain rule (backpropagation)
    for param in model.parameters():
        param -= learning_rate * grad    # gradient descent step
```

Every line of this is calculus:
- `model(batch)`: evaluating a composition of differentiable functions
- `loss_fn`: a scalar-valued function of predictions and targets
- `backward`: applying the chain rule on the computation graph to get ∇L w.r.t. all parameters
- `param -= lr * grad`: taking a step in the negative gradient direction

### 8.2 Backpropagation IS the Chain Rule

Backpropagation is not a separate algorithm — it is an efficient implementation of the chain rule on a computation graph using reverse-mode automatic differentiation.

Key insight: computing the gradient of a scalar loss with respect to N parameters costs roughly the same as a single forward pass (up to a small constant factor), regardless of N. This is what makes training networks with millions of parameters feasible.

Forward-mode differentiation would require N forward passes (one per parameter). Reverse-mode requires one backward pass. This asymmetry is why we use reverse mode for training.

### 8.3 The Vanishing/Exploding Gradient Problem

The chain rule multiplies local derivatives along a path:

```
∂L/∂θ₁ = (∂L/∂fₙ) · (∂fₙ/∂fₙ₋₁) · ... · (∂f₂/∂f₁) · (∂f₁/∂θ₁)
```

If each factor |∂fᵢ/∂fᵢ₋₁| < 1, the product shrinks exponentially → **vanishing gradients** (early layers barely learn).

If each factor |∂fᵢ/∂fᵢ₋₁| > 1, the product grows exponentially → **exploding gradients** (training becomes unstable).

This is why:
- **ReLU** replaced sigmoid/tanh: its derivative is exactly 0 or 1, avoiding the < 1 multiplication problem.
- **Residual connections** (ResNets) add skip connections: x + f(x) has gradient 1 + f'(x), providing a gradient highway through the network.
- **Layer normalization** and **careful initialization** (Xavier, He) keep activation scales stable.
- **LSTMs** use gates specifically designed to control gradient flow through time.

### 8.4 The Landscape View

The loss function L(θ) defines a surface over parameter space. Everything we do in training is navigating this landscape:

- **Gradient descent** follows the steepest downhill direction
- **Momentum** adds inertia to avoid oscillation in narrow valleys
- **Adam** adapts the step size per-parameter using running estimates of the gradient's first and second moments (approximating diagonal Hessian information)
- **Learning rate schedules** (warmup, cosine decay) navigate different phases: explore broadly first, then settle into a minimum
- **Saddle points** (not local minima) are the main obstacle in high dimensions — the Hessian has mixed eigenvalues
- **Flat minima** tend to generalize better than sharp minima — this connects the Hessian eigenspectrum to generalization

### 8.5 Beyond First-Order Gradients

Active research areas that require deeper calculus:

- **Natural gradient descent:** Uses the Fisher information matrix (the expected Hessian of the log-likelihood) to account for the Riemannian geometry of probability distributions.
- **Implicit differentiation:** Computes gradients through the solution of optimization problems (used in meta-learning, bilevel optimization).
- **Neural ODEs:** Treat the network as a continuous dynamical system, using the adjoint method (continuous-time backpropagation) for gradient computation.
- **Differentiable programming:** Making entire programs differentiable — physics simulators, renderers, sorting algorithms — to learn their parameters with gradient descent.

---

## Summary

| Concept | What It Is | Role in ML |
|---------|-----------|------------|
| Derivative | Rate of change of a function | How loss changes with a parameter |
| Partial derivative | Rate of change w.r.t. one variable | Gradient component for one parameter |
| Chain rule | Derivative of composed functions | Backpropagation — gradient flow through layers |
| Gradient | Vector of all partial derivatives | Direction of steepest ascent/descent |
| Jacobian | Matrix of all partial derivatives (vector → vector) | Layer-wise derivative in neural networks |
| Hessian | Matrix of second derivatives | Curvature of loss landscape |
| Taylor series | Local polynomial approximation | Justification for gradient descent (linear approx.) |

The path forward: in `implementation.py`, we build all of this from scratch — numerical differentiation, a working autograd engine that implements the chain rule on computation graphs, and gradient descent that finds minima. This is not abstract mathematics. This is the engine that makes every neural network learn.

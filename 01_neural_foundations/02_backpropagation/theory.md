# Backpropagation: How Networks Learn

*"How does a network with millions of weights know which one to change?"*

This is the **credit assignment problem** --- the central challenge of learning in deep networks. Backpropagation is the answer: an algorithm that computes the gradient of a loss function with respect to every weight in an arbitrarily deep network, using nothing more than the chain rule applied systematically through a computation graph.

Backpropagation is not a learning algorithm itself. It is a method for computing gradients efficiently. Combined with gradient descent, it becomes the engine of all modern deep learning.

---

## 1. The Credit Assignment Problem

Consider a 10-layer neural network that makes a wrong prediction. Which weights caused the error?

- **The output layer weights** directly produced the wrong answer, but they operated on features computed by layer 9.
- **Layer 9 weights** produced those features, but from features computed by layer 8.
- **Layer 1 weights** shaped the very first representation --- a tiny change there ripples through all subsequent layers.

Every weight in the network shares some fraction of the blame. The credit assignment problem asks: *how much blame does each weight deserve, and in what direction should it change?*

The mathematical answer is the **gradient** $\frac{\partial \mathcal{L}}{\partial w_i}$ for every weight $w_i$ --- a number that tells us both the direction and magnitude of each weight's contribution to the error.

Computing this naively would require $O(N)$ forward passes for $N$ parameters (perturbing each weight individually). Backpropagation computes **all** gradients in a single backward pass --- $O(1)$ passes regardless of the number of parameters. This is what makes training networks with billions of parameters feasible.

---

## 2. Computation Graphs: Functions as DAGs

### The Key Abstraction

Any mathematical function can be decomposed into a **directed acyclic graph (DAG)** where:
- **Nodes** are operations (add, multiply, exp, etc.) or input variables
- **Edges** carry intermediate values
- The graph has no cycles (each value is computed exactly once)

**Example:** $f(x, y, z) = (x + y) \cdot z$

```
    x ──┐
        ├── [+] ── a ──┐
    y ──┘               ├── [*] ── f
                   z ──┘
```

Where $a = x + y$ and $f = a \cdot z$.

### Why Graphs?

1. **Modularity:** Each node only needs to know its own local operation and its local derivative. No node needs to understand the global function.

2. **Efficiency:** The forward pass computes the function. The backward pass computes all gradients. Both traverse the graph exactly once.

3. **Generality:** Any differentiable program (including neural networks, loss functions, and regularization) can be represented as a computation graph and differentiated automatically.

---

## 3. The Forward Pass

The forward pass evaluates the function by traversing the graph from inputs to outputs, computing each node's value from its parents.

**Example:** 2-layer neural network on a single input $x$.

$$z_1 = w_1 x + b_1 \quad \text{(linear)}$$
$$h_1 = \sigma(z_1) \quad \text{(activation)}$$
$$z_2 = w_2 h_1 + b_2 \quad \text{(linear)}$$
$$\hat{y} = \sigma(z_2) \quad \text{(output)}$$
$$\mathcal{L} = -[y \log(\hat{y}) + (1-y) \log(1 - \hat{y})] \quad \text{(loss)}$$

The computation graph:

```
x ──→ [*w1] ──→ [+b1] ──→ [sigma] ──→ [*w2] ──→ [+b2] ──→ [sigma] ──→ [loss]
       w1         b1        h1          w2         b2        y_hat       L
                                                               ↑
                                                               y (target)
```

We compute left to right, storing every intermediate value. These stored values are essential --- the backward pass needs them.

---

## 4. The Backward Pass: Chain Rule Through the Graph

### The Chain Rule (Single Variable)

If $y = f(g(x))$, then:

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx} = f'(g(x)) \cdot g'(x)$$

### The Chain Rule (Multivariable)

If $z$ depends on $x$ through multiple paths $z = f(u_1(x), u_2(x), \ldots, u_k(x))$, then:

$$\frac{\partial z}{\partial x} = \sum_{i=1}^{k} \frac{\partial z}{\partial u_i} \cdot \frac{\partial u_i}{\partial x}$$

Gradients **sum** across paths. This is the multivariate chain rule, and it is the mathematical foundation of backpropagation.

### The Backward Pass Algorithm

Traverse the computation graph from output to inputs. At each node:

1. **Receive** the upstream gradient $\frac{\partial \mathcal{L}}{\partial \text{output}}$ (how much the loss changes when this node's output changes)
2. **Compute** the local gradient $\frac{\partial \text{output}}{\partial \text{input}}$ (how much this node's output changes when its input changes)
3. **Multiply** to get the downstream gradient: $\frac{\partial \mathcal{L}}{\partial \text{input}} = \frac{\partial \mathcal{L}}{\partial \text{output}} \cdot \frac{\partial \text{output}}{\partial \text{input}}$
4. **Pass** the downstream gradient to parent nodes
5. If a node has multiple children, **sum** the gradients received from all children

### Local Derivatives for Common Operations

Each operation only needs to know its own derivative:

| Operation | Forward | Local gradient(s) |
|-----------|---------|-------------------|
| Add: $c = a + b$ | $c = a + b$ | $\frac{\partial c}{\partial a} = 1, \quad \frac{\partial c}{\partial b} = 1$ |
| Multiply: $c = a \cdot b$ | $c = ab$ | $\frac{\partial c}{\partial a} = b, \quad \frac{\partial c}{\partial b} = a$ |
| Power: $c = a^n$ | $c = a^n$ | $\frac{\partial c}{\partial a} = n \cdot a^{n-1}$ |
| Sigmoid: $c = \sigma(a)$ | $c = \frac{1}{1+e^{-a}}$ | $\frac{\partial c}{\partial a} = c(1-c)$ |
| ReLU: $c = \max(0, a)$ | $c = \max(0, a)$ | $\frac{\partial c}{\partial a} = \mathbf{1}[a > 0]$ |
| Log: $c = \log(a)$ | $c = \log(a)$ | $\frac{\partial c}{\partial a} = \frac{1}{a}$ |
| Exp: $c = e^a$ | $c = e^a$ | $\frac{\partial c}{\partial a} = e^a = c$ |
| Tanh: $c = \tanh(a)$ | $c = \tanh(a)$ | $\frac{\partial c}{\partial a} = 1 - c^2$ |

---

## 5. Full Derivation: Backprop for a 2-Layer Network

Let us derive every gradient for a concrete 2-layer network with sigmoid activations and binary cross-entropy loss. This is the derivation you must be able to do on paper.

### Setup

- Input: $x$ (scalar for clarity; vectors generalize straightforwardly)
- Hidden layer: $z_1 = w_1 x + b_1$, $\quad h = \sigma(z_1)$
- Output layer: $z_2 = w_2 h + b_2$, $\quad \hat{y} = \sigma(z_2)$
- Loss: $\mathcal{L} = -[y \log(\hat{y}) + (1-y) \log(1 - \hat{y})]$

We need: $\frac{\partial \mathcal{L}}{\partial w_1}, \frac{\partial \mathcal{L}}{\partial b_1}, \frac{\partial \mathcal{L}}{\partial w_2}, \frac{\partial \mathcal{L}}{\partial b_2}$

### Step 1: Gradient of loss w.r.t. output

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}$$

### Step 2: Through the output sigmoid

Using $\sigma'(z) = \sigma(z)(1-\sigma(z)) = \hat{y}(1-\hat{y})$:

$$\frac{\partial \mathcal{L}}{\partial z_2} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_2} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})} \cdot \hat{y}(1-\hat{y}) = \hat{y} - y$$

This beautiful simplification --- the gradient of cross-entropy + sigmoid is just $(\hat{y} - y)$ --- is not a coincidence. It arises because cross-entropy is the natural loss for sigmoid outputs (they form an exponential family).

### Step 3: Gradients of output layer parameters

$$\frac{\partial \mathcal{L}}{\partial w_2} = \frac{\partial \mathcal{L}}{\partial z_2} \cdot \frac{\partial z_2}{\partial w_2} = (\hat{y} - y) \cdot h$$

$$\frac{\partial \mathcal{L}}{\partial b_2} = \frac{\partial \mathcal{L}}{\partial z_2} \cdot \frac{\partial z_2}{\partial b_2} = (\hat{y} - y) \cdot 1 = \hat{y} - y$$

### Step 4: Backpropagate through the hidden layer

$$\frac{\partial \mathcal{L}}{\partial h} = \frac{\partial \mathcal{L}}{\partial z_2} \cdot \frac{\partial z_2}{\partial h} = (\hat{y} - y) \cdot w_2$$

$$\frac{\partial \mathcal{L}}{\partial z_1} = \frac{\partial \mathcal{L}}{\partial h} \cdot \frac{\partial h}{\partial z_1} = (\hat{y} - y) \cdot w_2 \cdot h(1-h)$$

### Step 5: Gradients of hidden layer parameters

$$\frac{\partial \mathcal{L}}{\partial w_1} = \frac{\partial \mathcal{L}}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_1} = (\hat{y} - y) \cdot w_2 \cdot h(1-h) \cdot x$$

$$\frac{\partial \mathcal{L}}{\partial b_1} = \frac{\partial \mathcal{L}}{\partial z_1} \cdot \frac{\partial z_1}{\partial b_1} = (\hat{y} - y) \cdot w_2 \cdot h(1-h)$$

### The Pattern

Notice the structure. Each gradient is a product of:
1. The **error signal** at the output: $(\hat{y} - y)$
2. A chain of **local derivatives** through each layer back to the parameter
3. The **input to the parameter's layer** (for weights) or 1 (for biases)

This telescoping product is why it is called the "chain" rule. For a network with $L$ layers, the gradient of the first layer involves a product of $L$ local derivatives --- and this is precisely where problems arise.

---

## 6. Gradient Flow: Vanishing and Exploding Gradients

### The Core Problem

For a deep network with $L$ layers, the gradient of the loss with respect to layer $l$'s parameters involves:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_L} \cdot \prod_{k=l+1}^{L} \frac{\partial \mathbf{z}_k}{\partial \mathbf{z}_{k-1}} \cdot \frac{\partial \mathbf{z}_l}{\partial \mathbf{W}_l}$$

The product $\prod_{k=l+1}^{L} \frac{\partial \mathbf{z}_k}{\partial \mathbf{z}_{k-1}}$ is a chain of $(L - l)$ matrix multiplications. The behavior of this product determines whether learning is possible.

### Vanishing Gradients

If each factor $\left\|\frac{\partial \mathbf{z}_k}{\partial \mathbf{z}_{k-1}}\right\| < 1$, the product **shrinks exponentially**:

$$\left\|\prod_{k=l+1}^{L}\right\| \approx c^{L-l} \to 0 \quad \text{as } L - l \to \infty$$

**Sigmoid is a primary culprit.** Its derivative $\sigma'(z) = \sigma(z)(1-\sigma(z))$ has a maximum value of 0.25 (at $z = 0$). For each sigmoid layer, the gradient is multiplied by at most 0.25. After 10 layers: $0.25^{10} \approx 10^{-6}$. The gradient effectively disappears.

**Consequence:** Early layers learn extremely slowly or not at all. The network's first layers remain near their random initialization --- they cannot extract useful features from the input.

### Exploding Gradients

If each factor $\left\|\frac{\partial \mathbf{z}_k}{\partial \mathbf{z}_{k-1}}\right\| > 1$, the product **grows exponentially**:

$$\left\|\prod_{k=l+1}^{L}\right\| \approx c^{L-l} \to \infty$$

**Consequence:** Gradient updates become enormous, weights diverge, loss becomes NaN. Training is unstable and fails.

### Solutions (Previews of Later Modules)

| Problem | Solution | Mechanism |
|---------|----------|-----------|
| Vanishing | ReLU activation | Gradient is 1 for positive inputs (no shrinkage) |
| Vanishing | Residual connections | Gradient flows through skip connections (additive, not multiplicative) |
| Vanishing | LSTM gates | Gating mechanism maintains gradient flow over long sequences |
| Vanishing | Careful initialization | Xavier/He initialization keeps variance stable across layers |
| Exploding | Gradient clipping | Cap gradient norm to a maximum value |
| Exploding | Batch normalization | Normalize activations to prevent scale explosion |
| Both | Layer normalization | Stabilize the distribution of layer inputs |

The vanishing gradient problem delayed deep learning by decades. Its solution --- primarily ReLU activations and residual connections --- is what makes networks with hundreds of layers trainable today.

---

## 7. Connection to Automatic Differentiation

Backpropagation is a specific instance of **reverse-mode automatic differentiation (AD)**, a general technique from numerical computing that predates neural networks.

### Forward-Mode vs. Reverse-Mode AD

**Forward-mode AD:** Computes $\frac{\partial f}{\partial x_i}$ for one input $x_i$ at a time. Cost: $O(n)$ passes for $n$ inputs. Efficient when there are few inputs and many outputs.

**Reverse-mode AD:** Computes $\frac{\partial f}{\partial x_i}$ for *all* inputs in one pass. Cost: $O(m)$ passes for $m$ outputs. Efficient when there are many inputs and few outputs.

Neural networks have millions of inputs (parameters) and one output (loss). Reverse-mode AD (backpropagation) is the clear choice:

| Property | Forward-Mode | Reverse-Mode (Backprop) |
|----------|-------------|------------------------|
| Computes | One column of Jacobian per pass | One row of Jacobian per pass |
| Cost for $n$ params, 1 loss | $O(n)$ passes | $O(1)$ passes |
| Memory | Low (no graph needed) | High (must store forward pass values) |
| Use case | Few inputs, many outputs | Many inputs, few outputs (= neural nets) |

### The Memory-Compute Tradeoff

Backpropagation requires storing all intermediate values from the forward pass (for use in computing local gradients during the backward pass). For a network with $L$ layers of width $d$, this is $O(L \cdot d)$ memory.

This is the fundamental tradeoff: we trade memory for compute. Techniques like **gradient checkpointing** reduce memory by recomputing some intermediate values during the backward pass, trading $O(\sqrt{L})$ memory for $O(L \sqrt{L})$ compute.

### What Modern Frameworks Do

PyTorch, TensorFlow, and JAX all implement reverse-mode AD:

1. **Build a computation graph** during the forward pass (PyTorch does this dynamically; TensorFlow 1.x did it statically)
2. **Store intermediate values** ("activations") at each node
3. **Traverse the graph in reverse** during `.backward()`, applying the chain rule at each node
4. **Accumulate gradients** in `.grad` attributes of parameter tensors

The implementation in this module builds this machinery from scratch: a `Value` class that records its computation graph and can backpropagate through it. This is the same core idea behind PyTorch's autograd engine, stripped to its essence.

---

## Key Takeaways

1. **Backpropagation solves the credit assignment problem** by computing the gradient of the loss with respect to every parameter in a single backward pass through the computation graph.

2. **It is the chain rule, applied systematically.** Each node multiplies the upstream gradient by its local derivative and passes the result downstream. Gradients sum when paths merge.

3. **The forward pass computes the function; the backward pass computes all gradients.** Both traverse the graph once. The cost of the backward pass is approximately 2-3x the forward pass.

4. **Gradient flow through deep networks is fragile.** Products of many small derivatives vanish; products of many large derivatives explode. This is not a bug in backpropagation --- it is a fundamental property of composing many functions. Solving it requires architectural innovation (ReLU, residual connections, normalization).

5. **Backpropagation is reverse-mode automatic differentiation** --- a general technique that applies to any differentiable computation, not just neural networks.

---

## References

- Rumelhart, D., Hinton, G., & Williams, R. (1986). *Learning Representations by Back-propagating Errors.* Nature, 323, 533-536.
- Griewank, A. & Walther, A. (2008). *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation.* 2nd edition. SIAM.
- Hochreiter, S. (1991). *Untersuchungen zu dynamischen neuronalen Netzen.* Diploma thesis (identified the vanishing gradient problem).
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* Chapter 6.5. MIT Press.
- Karpathy, A. (2022). *micrograd.* A tiny autograd engine. github.com/karpathy/micrograd.

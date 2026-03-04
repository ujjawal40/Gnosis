# Activation Functions

**The non-linear soul of neural networks.**

A neural network without activation functions is a linear map. No matter how many layers you stack, the composition of linear functions is linear. Activation functions are what give networks the capacity to approximate arbitrary functions. Every choice here has consequences for gradient flow, training dynamics, and representational capacity.

---

## Why Non-Linearity Is Non-Negotiable

Consider a two-layer network with no activation functions:

$$y = W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2) = W'x + b'$$

The composition collapses to a single affine transformation. You can stack 1000 layers and it's still linear regression. The proof is immediate: the space of affine maps is closed under composition.

**What we need:** A function $\sigma$ applied element-wise between layers such that the composition $W_n \sigma(\ldots \sigma(W_1 x + b_1) \ldots) + b_n$ can approximate any continuous function on a compact set (Universal Approximation Theorem, Cybenko 1989, Hornik 1991).

**The requirement is surprisingly mild.** Almost any non-constant, bounded, continuous function works as a universal approximator in the single-hidden-layer case. But "can approximate" and "can be efficiently learned" are very different things. The choice of activation function profoundly affects:

1. **Gradient flow** — Can gradients propagate through 100 layers without vanishing or exploding?
2. **Sparsity** — Does the activation encourage sparse representations?
3. **Computation** — How expensive is the forward/backward pass?
4. **Output distribution** — Does the activation preserve the mean/variance of its input?

---

## Sigmoid: The Founding Function

### Definition

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

### Derivative

The sigmoid has an elegant self-referential derivative:

$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

**Derivation:**

$$\sigma'(x) = \frac{e^{-x}}{(1 + e^{-x})^2} = \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}} = \sigma(x) \cdot (1 - \sigma(x))$$

### Properties

- **Range:** $(0, 1)$ — interpretable as a probability
- **Maximum gradient:** $\sigma'(0) = 0.25$ — already less than 1
- **Monotonic, smooth, differentiable everywhere**

### Historical Importance

The sigmoid was the default activation for decades. Biologically motivated (a neuron either fires or doesn't), it maps any real number to a probability-like output. Logistic regression is literally a single sigmoid neuron.

### The Saturation Problem

This is what killed the sigmoid for deep networks. When $|x|$ is large:

- $\sigma(x) \approx 0$ or $\sigma(x) \approx 1$
- $\sigma'(x) \approx 0$

The gradient vanishes. During backpropagation, gradients are multiplied through layers. If each layer's local gradient is $\leq 0.25$, then after $n$ layers:

$$\frac{\partial L}{\partial W_1} \propto (0.25)^n$$

For $n = 10$ layers: $(0.25)^{10} \approx 10^{-6}$. The first layer barely learns.

**Additional problems:**
- **Non-zero-centered output:** $\sigma(x) \in (0, 1)$, so the output is always positive. This means gradients with respect to weights in the next layer are always the same sign, causing zig-zagging dynamics during optimization.
- **Expensive exponential:** `exp()` is slower than comparison operations.

---

## Tanh: Centering the Sigmoid

### Definition

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$

### Derivative

$$\tanh'(x) = 1 - \tanh^2(x)$$

### Why It Helps

- **Range:** $(-1, 1)$ — zero-centered output
- **Maximum gradient:** $\tanh'(0) = 1$ — double the sigmoid's maximum
- **Zero-centered:** Inputs to the next layer have mean roughly zero, enabling gradient updates in all directions

### Why It Doesn't Help Enough

Tanh still saturates. For $|x| > 3$, the gradient is effectively zero. Deep networks still suffer vanishing gradients. The maximum gradient of 1 means at best, gradients are preserved — they never grow, so even the "ideal" case leads to slow gradient decay over many layers.

LeCun et al. (1998) recommended tanh over sigmoid and suggested normalizing inputs so that the linear region of tanh is well-utilized. This was the state of the art for a decade.

---

## ReLU: The Breakthrough

### Definition

$$\text{ReLU}(x) = \max(0, x)$$

### Derivative

$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \\ \text{undefined} & \text{if } x = 0 \end{cases}$$

In practice, we use a subgradient: $\text{ReLU}'(0) = 0$ (or sometimes 1). It doesn't matter — the probability of landing exactly on 0 is measure-zero.

### Why ReLU Changed Everything (Glorot et al., 2011; Krizhevsky et al., 2012)

1. **No saturation for positive inputs.** The gradient is exactly 1 for $x > 0$. Gradients can flow through arbitrarily many layers without vanishing.

2. **Sparsity.** For a randomly initialized network, roughly 50% of neurons output zero. Sparse representations are computationally efficient and have been linked to better generalization.

3. **Computation.** A single comparison (`x > 0`) versus exponentials. This is 6x faster than sigmoid/tanh in practice.

4. **Biological plausibility.** Neurons in the brain have low firing rates (sparse activation). ReLU captures this better than sigmoid.

### The Dying ReLU Problem

If a neuron's input is negative for all training examples, its gradient is zero and it can never recover. This neuron is "dead" — permanently outputting zero.

**When does this happen?**
- Large learning rates push weights so that $Wx + b < 0$ for all $x$ in the dataset
- Poor initialization
- During training, neurons that were alive can die and never come back

**How bad is it?** In practice, with a well-tuned learning rate, 10-20% of neurons may die. Networks are overparameterized enough that this is usually acceptable. But it's a waste of capacity.

### Non-Differentiability at Zero

Theoretically, ReLU isn't differentiable at $x = 0$. In practice, this doesn't matter because:

1. The probability of any input being exactly 0 is zero (continuous distributions)
2. Gradient-based optimization works with subgradients
3. Empirically, it works — and that's what matters

The deeper reason: optimization of neural networks is already non-convex. We're not looking for exact gradients; we're looking for "good enough" descent directions.

---

## Leaky ReLU, ELU, SELU: Fixing Dying ReLU

### Leaky ReLU (Maas et al., 2013)

$$\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}$$

Typically $\alpha = 0.01$. The small negative slope means gradients can flow even for negative inputs, preventing dead neurons. The derivative is:

$$\text{LeakyReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha & \text{if } x \leq 0 \end{cases}$$

**Parametric ReLU (PReLU):** Let $\alpha$ be a learned parameter. He et al. (2015) showed PReLU improved ImageNet accuracy at negligible extra cost.

### ELU — Exponential Linear Unit (Clevert et al., 2016)

$$\text{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}$$

$$\text{ELU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \text{ELU}(x) + \alpha & \text{if } x \leq 0 \end{cases}$$

**Key property:** ELU has negative values, which pushes the mean activation closer to zero. This acts as a form of implicit batch normalization, reducing the need for explicit normalization layers.

The smooth curve for $x < 0$ also means ELU is differentiable everywhere, unlike ReLU and Leaky ReLU.

### SELU — Scaled Exponential Linear Unit (Klambauer et al., 2017)

$$\text{SELU}(x) = \lambda \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}$$

Where $\lambda \approx 1.0507$ and $\alpha \approx 1.6733$ are derived analytically (not tuned!).

**The remarkable property:** SELU is the only activation function that is self-normalizing. Under specific conditions (dense layers, LeCun normal initialization), the mean and variance of activations converge to (0, 1) as they propagate through layers. This is proven using the Banach fixed-point theorem applied to the map of mean/variance through layers.

**The catch:** Self-normalization only holds for fully-connected networks with proper initialization. It doesn't extend to CNNs, RNNs, or networks with skip connections.

---

## GELU: The Gaussian Gate

### Definition (Hendrycks & Gimpel, 2016)

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the CDF of the standard normal distribution:

$$\Phi(x) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

### Derivative

$$\text{GELU}'(x) = \Phi(x) + x \cdot \phi(x)$$

where $\phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$ is the standard normal PDF.

### Approximation

The exact GELU requires the error function. Common approximations:

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{2/\pi}(x + 0.044715x^3)\right]\right)$$

$$\text{GELU}(x) \approx x \cdot \sigma(1.702x)$$

### The Intuition

GELU is a **stochastic regularizer made deterministic.** Imagine multiplying each input by a Bernoulli random variable whose probability of being 1 depends on how large the input is:

- Large positive inputs are almost certainly kept (gate $\approx 1$)
- Large negative inputs are almost certainly zeroed (gate $\approx 0$)
- Inputs near zero are randomly kept or dropped

GELU is the expected value of this stochastic process. It smoothly interpolates between "pass through" and "zero out" based on input magnitude, with the transition region governed by a Gaussian.

### Why Transformers Use GELU

GELU has become the default for transformer architectures (BERT, GPT, ViT). The reasons:

1. **Smooth, non-monotonic curve** near zero provides richer gradient signal than ReLU
2. **Probabilistic interpretation** connects to dropout and stochastic regularization
3. **Empirically superior** in large-scale language modeling tasks
4. **Better gradient flow** than ReLU for very deep networks with residual connections

---

## Swish / SiLU: Self-Gated Activation

### Definition (Ramachandran et al., 2017; Elfwing et al., 2018)

$$\text{Swish}(x) = x \cdot \sigma(\beta x)$$

where $\sigma$ is the sigmoid function. When $\beta = 1$ (SiLU — Sigmoid Linear Unit):

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

### Derivative

$$\text{SiLU}'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x)) = \sigma(x)(1 + x(1 - \sigma(x)))$$

### Properties

- **Self-gated:** The input gates itself through a sigmoid — no additional parameters needed
- **Smooth and non-monotonic:** Like GELU, it dips slightly below zero
- **Unbounded above, bounded below:** Similar asymptotic behavior to ReLU for large positive inputs
- **Found by automated search:** Ramachandran et al. used reinforcement learning to search over activation function space and Swish emerged as the best

### Relationship to GELU

Swish and GELU are remarkably similar. The sigmoid approximation of GELU is $x \cdot \sigma(1.702x)$, which is Swish with $\beta = 1.702$. They differ by less than 0.1 everywhere and perform similarly in practice.

---

## Softmax: From Logits to Probabilities

### Definition

For a vector $\mathbf{z} \in \mathbb{R}^K$:

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

### Properties

- **Output is a valid probability distribution:** $\sum_i \text{softmax}(\mathbf{z})_i = 1$ and $\text{softmax}(\mathbf{z})_i > 0$
- **Translation invariant:** $\text{softmax}(\mathbf{z} + c) = \text{softmax}(\mathbf{z})$ for any constant $c$
- **Amplifies differences:** The exponential makes larger values relatively much larger
- **Differentiable:** Unlike argmax, softmax is a smooth approximation of "pick the largest"

### Jacobian

The Jacobian of softmax is:

$$\frac{\partial \text{softmax}(\mathbf{z})_i}{\partial z_j} = \text{softmax}(\mathbf{z})_i (\delta_{ij} - \text{softmax}(\mathbf{z})_j)$$

In matrix form: $J = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$ where $\mathbf{p} = \text{softmax}(\mathbf{z})$.

### Temperature Scaling

$$\text{softmax}(\mathbf{z}/T)_i = \frac{e^{z_i/T}}{\sum_{j=1}^{K} e^{z_j/T}}$$

- **$T \to 0$:** Approaches argmax (one-hot). The model becomes maximally confident.
- **$T = 1$:** Standard softmax.
- **$T \to \infty$:** Approaches uniform distribution. All classes equally likely.

Temperature scaling is used in:
- **Knowledge distillation** (Hinton et al., 2015): Soften teacher outputs to transfer "dark knowledge"
- **Language model sampling:** Control creativity/randomness of text generation
- **Calibration:** Post-hoc calibration of model confidence

### Numerical Stability

The naive implementation overflows for large inputs. The fix uses translation invariance:

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_{j} e^{z_j - \max(\mathbf{z})}}$$

Subtracting the maximum prevents overflow while giving identical results.

---

## How to Choose: Rules of Thumb

### For Feed-Forward Networks (MLPs)
- **Default:** ReLU. It's fast, well-understood, and works.
- **If you see dead neurons:** Switch to Leaky ReLU or ELU.
- **If you want self-normalization (no batch norm):** SELU with LeCun normal init.

### For Convolutional Networks
- **Default:** ReLU. It works. Don't overthink it.
- **Modern architectures (ConvNeXt):** GELU, borrowing from transformer conventions.

### For Transformers
- **Default:** GELU. It's what BERT, GPT, ViT use.
- **Alternative:** SiLU/Swish. Used in PaLM, LLaMA, and other large models.
- **GLU variants** (SwiGLU, GeGLU): The current state of the art for transformer FFN blocks. These use the activation as a gating mechanism within the feed-forward layer.

### For Output Layers
- **Binary classification:** Sigmoid (outputs probability).
- **Multi-class classification:** Softmax (outputs probability distribution).
- **Regression:** No activation (linear output).
- **Bounded regression:** Sigmoid or tanh scaled to the output range.

### For Recurrent Networks (LSTM/GRU)
- **Gates:** Sigmoid (they need to be in $[0, 1]$ for gating).
- **State update:** Tanh (traditional, centered).

### General Principles

1. **Start simple.** ReLU for MLPs/CNNs, GELU for transformers.
2. **Match the architecture.** Use what the original paper used — they likely tuned for it.
3. **Check for dead neurons.** If training loss plateaus early, dying ReLU might be the cause.
4. **Don't mix activation functions** between layers without good reason.
5. **The activation function is rarely the bottleneck.** If your model isn't working, look at learning rate, initialization, and architecture first.

---

## Summary Table

| Activation | Formula | Range | Derivative | Key Property |
|-----------|---------|-------|-----------|-------------|
| Sigmoid | $\frac{1}{1+e^{-x}}$ | $(0,1)$ | $\sigma(1-\sigma)$ | Saturates, non-centered |
| Tanh | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $(-1,1)$ | $1 - \tanh^2$ | Centered, still saturates |
| ReLU | $\max(0,x)$ | $[0,\infty)$ | $\mathbb{1}_{x>0}$ | Fast, sparse, can die |
| Leaky ReLU | $\max(\alpha x, x)$ | $(-\infty,\infty)$ | $1$ or $\alpha$ | No dead neurons |
| ELU | See above | $(-\alpha, \infty)$ | Smooth for $x<0$ | Near-zero mean output |
| SELU | $\lambda \cdot \text{ELU}$ | Scaled | Self-normalizing | Requires specific init |
| GELU | $x \cdot \Phi(x)$ | $\approx [-0.17, \infty)$ | $\Phi(x) + x\phi(x)$ | Transformer default |
| SiLU/Swish | $x \cdot \sigma(x)$ | $\approx [-0.28, \infty)$ | See above | Self-gated, smooth |
| Softmax | $\frac{e^{z_i}}{\sum e^{z_j}}$ | $(0,1)^K$ | $\text{diag}(p) - pp^T$ | Output layer for classification |

---

## Key Takeaways

1. **Non-linearity is what makes deep learning work.** Without it, depth is meaningless.
2. **Saturation kills gradients.** Sigmoid and tanh suffer from this in deep networks.
3. **ReLU was the breakthrough** that enabled training of truly deep networks, despite its simplicity.
4. **Modern activations (GELU, SiLU)** trade computational simplicity for smoother gradient landscapes.
5. **The choice of activation function matters less than you think** for most problems — but matters enormously at scale.
6. **Softmax is special** — it's not a per-element activation but a normalizing function across a vector.

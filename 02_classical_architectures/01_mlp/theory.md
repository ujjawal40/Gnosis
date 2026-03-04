# Multi-Layer Perceptrons: From Single Neurons to Deep Networks

## Table of Contents
1. [From Single Neuron to Networks](#1-from-single-neuron-to-networks)
2. [Forward Propagation: Matrix Formulation](#2-forward-propagation-matrix-formulation)
3. [The Universal Approximation Theorem](#3-the-universal-approximation-theorem)
4. [Depth vs Width: Why Depth Matters More](#4-depth-vs-width-why-depth-matters-more)
5. [Weight Initialization](#5-weight-initialization)
6. [Batch Normalization](#6-batch-normalization)
7. [Dropout](#7-dropout)
8. [Feature Learning: What Hidden Layers Actually Learn](#8-feature-learning-what-hidden-layers-actually-learn)

---

## 1. From Single Neuron to Networks

### The Single Neuron

A single neuron computes:

$$y = \sigma(w^\top x + b)$$

where $x \in \mathbb{R}^d$ is the input, $w \in \mathbb{R}^d$ are weights, $b \in \mathbb{R}$ is the bias,
and $\sigma$ is a nonlinear activation function.

This is a **linear classifier** followed by a nonlinearity. It can separate linearly separable classes,
but it fails on XOR and any problem requiring a nonlinear decision boundary. This was Minsky and
Papert's devastating critique (1969) that nearly killed neural network research for a decade.

### The Key Insight: Stacking Layers

The solution is **composition**. If one neuron computes a single linear boundary, a *layer* of neurons
computes many boundaries simultaneously, and a *second layer* can combine those boundaries into
regions of arbitrary shape.

**Layer 1:** Each neuron in the first hidden layer computes a different linear function of the input.
With $h_1$ neurons, we get $h_1$ different half-space indicators. Together, they carve the input space
into up to $2^{h_1}$ regions.

**Layer 2:** Each neuron in the second layer takes the first layer's outputs as its inputs. It can now
compute functions of *functions* — combining the half-spaces into convex regions, and combining
convex regions into arbitrary unions.

Formally, a 2-layer MLP computes:

$$f(x) = W_2 \, \sigma(W_1 x + b_1) + b_2$$

A general $L$-layer MLP computes the composition:

$$f(x) = f_L \circ f_{L-1} \circ \cdots \circ f_1(x)$$

where each $f_\ell(z) = \sigma(W_\ell z + b_\ell)$.

### Why This Works: Compositionality

The power comes from **composition**, not from having many parameters. Consider:
- A polynomial of degree $d$ in $n$ variables has $\binom{n+d}{d}$ terms.
- A network of depth $L$ can represent functions whose "effective degree" grows *exponentially* with $L$.

Each layer transforms the representation space. Layer 1 might learn "is there an edge here?" Layer 2
combines edges into "is there a corner?" Layer 3 combines corners into "is there a shape?" This
hierarchical composition is why deep networks work so well on structured data.

---

## 2. Forward Propagation: Matrix Formulation

### Single Sample

For a single input $x \in \mathbb{R}^{d_0}$ through an $L$-layer network:

$$z^{(1)} = W^{(1)} x + b^{(1)}$$
$$a^{(1)} = \sigma(z^{(1)})$$
$$z^{(2)} = W^{(2)} a^{(1)} + b^{(2)}$$
$$a^{(2)} = \sigma(z^{(2)})$$
$$\vdots$$
$$z^{(L)} = W^{(L)} a^{(L-1)} + b^{(L)}$$
$$\hat{y} = a^{(L)} = \sigma_{\text{out}}(z^{(L)})$$

where:
- $W^{(\ell)} \in \mathbb{R}^{d_\ell \times d_{\ell-1}}$ is the weight matrix for layer $\ell$
- $b^{(\ell)} \in \mathbb{R}^{d_\ell}$ is the bias vector
- $z^{(\ell)}$ is the pre-activation (linear combination)
- $a^{(\ell)}$ is the post-activation
- $\sigma_{\text{out}}$ is the output activation (softmax for classification, identity for regression)

### Batch Formulation

For a mini-batch $X \in \mathbb{R}^{N \times d_0}$ (N samples, each of dimension $d_0$):

$$Z^{(1)} = X W^{(1)\top} + \mathbf{1}_N b^{(1)\top}$$
$$A^{(1)} = \sigma(Z^{(1)})$$
$$Z^{(\ell)} = A^{(\ell-1)} W^{(\ell)\top} + \mathbf{1}_N b^{(\ell)\top}$$
$$A^{(\ell)} = \sigma(Z^{(\ell)})$$

where $Z^{(\ell)}, A^{(\ell)} \in \mathbb{R}^{N \times d_\ell}$. Each row is one sample's activations.

In NumPy, this is literally one line per layer:

```python
Z = A_prev @ W.T + b  # broadcasting handles the bias
A = activation(Z)
```

The beauty of the matrix formulation: **everything is a matrix multiply followed by a pointwise
nonlinearity**. This maps perfectly onto GPU hardware, which is why neural networks became practical
when GPUs became programmable.

### Backpropagation Through Layers

Given loss $\mathcal{L}$, we need $\frac{\partial \mathcal{L}}{\partial W^{(\ell)}}$ and
$\frac{\partial \mathcal{L}}{\partial b^{(\ell)}}$ for every layer $\ell$.

Start from the output. Let $\delta^{(\ell)} = \frac{\partial \mathcal{L}}{\partial z^{(\ell)}}$ be the
error signal at layer $\ell$.

**Output layer** (with softmax + cross-entropy, which simplifies beautifully):
$$\delta^{(L)} = \hat{y} - y$$

**Hidden layers** (chain rule, propagating backward):
$$\delta^{(\ell)} = (W^{(\ell+1)\top} \delta^{(\ell+1)}) \odot \sigma'(z^{(\ell)})$$

**Gradients:**
$$\frac{\partial \mathcal{L}}{\partial W^{(\ell)}} = \delta^{(\ell)} a^{(\ell-1)\top}$$
$$\frac{\partial \mathcal{L}}{\partial b^{(\ell)}} = \delta^{(\ell)}$$

In batch form:
$$\frac{\partial \mathcal{L}}{\partial W^{(\ell)}} = \frac{1}{N} \delta^{(\ell)\top} A^{(\ell-1)}$$
$$\frac{\partial \mathcal{L}}{\partial b^{(\ell)}} = \frac{1}{N} \sum_{i=1}^{N} \delta_i^{(\ell)}$$

This is the **backpropagation algorithm** applied to MLPs — it is just the chain rule applied
systematically, layer by layer, from output to input.

---

## 3. The Universal Approximation Theorem

### What It Says

**Theorem (Cybenko 1989, Hornik 1991):** Let $\sigma$ be any continuous, non-constant, bounded,
monotonically increasing function (e.g., sigmoid). Then for any continuous function
$f: [0,1]^d \to \mathbb{R}$ and any $\varepsilon > 0$, there exists a single-hidden-layer network

$$g(x) = \sum_{i=1}^{N} v_i \, \sigma(w_i^\top x + b_i)$$

such that $\|g - f\|_\infty < \varepsilon$. That is, the network can approximate $f$ uniformly to
any desired precision.

The result was later extended to ReLU and essentially all non-polynomial activations (Leshno et al.,
1993).

### What It Actually Means

The theorem says that a sufficiently wide single-hidden-layer network can approximate *any* continuous
function on a compact domain. This is a powerful **existence** result: the function you want to learn
is somewhere in the hypothesis class of neural networks.

Intuitively, think of it this way:
- Each hidden neuron $\sigma(w_i^\top x + b_i)$ creates a "bump" or "step" in a particular direction.
- With enough bumps, you can sculpt any continuous surface — like building a landscape out of
  sand piles.
- The output layer $\sum v_i \cdot (\text{bump}_i)$ is a weighted combination of these bumps.

### What It Does NOT Say

This is where most people stop, and it is exactly where the real understanding begins.

**1. It says nothing about how many neurons you need.**
The width $N$ required may be astronomically large — potentially exponential in the input dimension.
For a function on $[0,1]^{100}$, you might need more neurons than atoms in the universe. The theorem
is non-constructive about the architecture.

**2. It says nothing about learnability.**
Existence of good weights $\neq$ ability to find them. Gradient descent might never converge to the
approximating network. The loss landscape of a wide shallow network can be highly non-convex with
terrible saddle points.

**3. It says nothing about generalization.**
A network with $N$ parameters that perfectly fits $N$ training points has learned nothing — it has
memorized. The theorem guarantees approximation on the training domain, not prediction on unseen
data. A network could approximate any function on training data while being wildly wrong everywhere
else.

**4. It says nothing about efficiency.**
This is the deepest limitation. A shallow network might need exponentially many neurons to represent
what a deep network represents with polynomially many. The theorem tells you "it's possible" but not
"it's practical."

**5. It applies only to continuous functions on compact sets.**
Real-world data lives on complicated manifolds, not neat compact boxes. Discontinuous functions,
functions on unbounded domains, and functions on non-Euclidean spaces are not covered.

### The Real Lesson

The Universal Approximation Theorem is like saying "any book can be written using the English
alphabet." True, but useless for actually writing a good book. The real questions are:
- What architecture makes learning *efficient*?
- What makes *generalization* possible?
- Why does *depth* help?

---

## 4. Depth vs Width: Why Depth Matters More

### The Intuitive Argument

Consider computing the parity function (XOR of $n$ bits). A depth-2 network requires $2^{n-1}$
hidden neurons. A network of depth $O(n)$ needs only $O(n)$ neurons total. The deep network is
exponentially more efficient.

Why? Because depth enables **re-use of computation**. Each layer can build on the features computed
by the previous layer, creating a hierarchy of increasingly abstract representations.

### Compositional Functions

Most real-world functions are *compositional* — they are built by composing simpler functions:

$$f(x) = f_L \circ f_{L-1} \circ \cdots \circ f_1(x)$$

A sentence's meaning is composed from word meanings. An image's content is composed from edges, textures,
parts, and objects. A physical system's behavior is composed from local interactions.

**Theorem (informal, Telgarsky 2016):** There exist functions computable by networks of depth $k$
with $O(k)$ neurons that cannot be approximated by networks of depth $O(k^{1/3})$ unless they have
exponentially many neurons.

This is a *depth separation* result: there are functions that deep networks compute efficiently but
shallow networks cannot — not just that shallow networks need more neurons, but that they need
*exponentially* more.

### Exponential Expressiveness of Depth

Consider a network of depth $L$ and constant width $w$. The number of linear regions it can create
in input space is:

$$\text{Shallow (depth 1):} \quad O(w^d)$$
$$\text{Deep (depth } L \text{):} \quad O\left(\left(\frac{w}{d}\right)^{(L-1)d} \cdot w^d\right)$$

The deep network creates *exponentially more* linear regions (with ReLU activations), meaning it can
represent exponentially more complex decision boundaries with the same total number of parameters.

**Montufar et al. (2014)** showed this precisely: a ReLU network with $L$ layers of width $w$ can
compute functions with $O((w/d)^{d(L-1)} \cdot w^d)$ linear regions, while a single-layer network
with the same number of neurons computes only $O(w^d)$ regions.

### The Efficiency of Hierarchical Representations

Think about how you would describe a chessboard pattern:
- **Shallow:** "Black at (0,0), white at (0,1), black at (0,2), ..." — listing every cell. Cost: $O(n^2)$.
- **Deep:** "Alternate black and white along each row, then alternate the starting color each row." Cost: $O(1)$.

The deep description exploits *structure* — the pattern is compositional. Neural networks do the same thing:
they learn hierarchical features where each level builds on the previous, exploiting the compositional
structure inherent in natural data.

### Why Not Infinitely Deep?

Depth helps, but there are diminishing (and eventually negative) returns:
1. **Vanishing/exploding gradients:** Gradients shrink or grow exponentially with depth, making training unstable. (Mitigated by residual connections, careful initialization, batch normalization.)
2. **Optimization difficulty:** Deeper loss landscapes have more saddle points and worse conditioning.
3. **Diminishing returns:** For a given problem, there's an effective "compositional depth" beyond which more layers don't help.

The practical answer is: use the depth warranted by the problem's compositional structure, and use
architectural tricks (skip connections, normalization) to make that depth trainable.

---

## 5. Weight Initialization: Why It Matters

### The Problem

Consider a network with $L$ layers. During forward propagation, if weights are too large, activations
explode exponentially: $\|a^{(L)}\| \sim c^L$ for some $c > 1$. If weights are too small,
activations vanish: $\|a^{(L)}\| \sim c^L$ for $c < 1$.

The same happens in reverse during backpropagation: gradients explode or vanish as they propagate
through layers. After just 10-20 layers, values can overflow or underflow floating-point
representation.

**The goal:** Initialize weights so that the *variance of activations* remains stable across layers
during both forward and backward passes.

### Derivation: Xavier/Glorot Initialization

**Setting:** Layer $\ell$ computes $z_j^{(\ell)} = \sum_{i=1}^{n_{\text{in}}} w_{ji} a_i^{(\ell-1)}$.

**Assumptions:**
1. Weights $w_{ji}$ are i.i.d. with zero mean: $\mathbb{E}[w_{ji}] = 0$
2. Inputs $a_i^{(\ell-1)}$ are i.i.d. with zero mean: $\mathbb{E}[a_i] = 0$
3. Weights and inputs are independent

**Forward pass variance:**

$$\text{Var}(z_j) = \text{Var}\left(\sum_{i=1}^{n_\text{in}} w_{ji} a_i\right)$$

Since terms are independent and zero-mean:

$$= \sum_{i=1}^{n_\text{in}} \text{Var}(w_{ji} a_i)$$

$$= \sum_{i=1}^{n_\text{in}} \left[\mathbb{E}[w_{ji}^2]\mathbb{E}[a_i^2] - \underbrace{(\mathbb{E}[w_{ji}])^2}_{=0}\underbrace{(\mathbb{E}[a_i])^2}_{=0}\right]$$

$$= n_\text{in} \cdot \text{Var}(w) \cdot \text{Var}(a)$$

For stable forward propagation, we want $\text{Var}(z_j) = \text{Var}(a_i)$, which requires:

$$n_\text{in} \cdot \text{Var}(w) = 1 \quad \Longrightarrow \quad \text{Var}(w) = \frac{1}{n_\text{in}}$$

**Backward pass variance:**

By symmetric analysis on the backward pass (where gradients flow through $W^\top$), stability requires:

$$\text{Var}(w) = \frac{1}{n_\text{out}}$$

**Compromise (Glorot & Bengio, 2010):** Average the two constraints:

$$\boxed{\text{Var}(w) = \frac{2}{n_\text{in} + n_\text{out}}}$$

**Implementation:**
- Uniform: $w \sim U\left[-\sqrt{\frac{6}{n_\text{in} + n_\text{out}}},\ +\sqrt{\frac{6}{n_\text{in} + n_\text{out}}}\right]$
  (since $\text{Var}(U[-a,a]) = a^2/3$)
- Normal: $w \sim \mathcal{N}\left(0,\ \frac{2}{n_\text{in} + n_\text{out}}\right)$

Xavier initialization is designed for **linear or tanh/sigmoid activations**, where the derivative near
zero is approximately 1.

### Derivation: He Initialization

**Problem with Xavier for ReLU:** ReLU zeros out half of the activations on average ($\text{ReLU}(z) = 0$
for $z < 0$). This means the effective variance is halved:

$$\text{Var}(a_i) = \text{Var}(\text{ReLU}(z_i)) = \frac{1}{2}\text{Var}(z_i)$$

This is because for $z \sim \mathcal{N}(0, \sigma^2)$:
$$\mathbb{E}[\text{ReLU}(z)^2] = \int_0^\infty z^2 \cdot \frac{1}{\sqrt{2\pi}\sigma}e^{-z^2/2\sigma^2}\,dz = \frac{\sigma^2}{2}$$

So the forward pass variance equation becomes:

$$\text{Var}(z^{(\ell)}) = n_\text{in} \cdot \text{Var}(w) \cdot \frac{1}{2}\text{Var}(z^{(\ell-1)})$$

For stability $\text{Var}(z^{(\ell)}) = \text{Var}(z^{(\ell-1)})$:

$$\boxed{\text{Var}(w) = \frac{2}{n_\text{in}}}$$

**He et al. (2015)** — also called "Kaiming initialization":
- Normal: $w \sim \mathcal{N}\left(0,\ \frac{2}{n_\text{in}}\right)$
- Uniform: $w \sim U\left[-\sqrt{\frac{6}{n_\text{in}}},\ +\sqrt{\frac{6}{n_\text{in}}}\right]$

### Empirical Impact

With bad initialization (e.g., $w \sim \mathcal{N}(0, 1)$), a 10-layer network's activations will
have variance $\sim n^{10}$ at the last layer — astronomical. Gradients will be correspondingly
extreme. Training either diverges immediately or gets stuck at near-zero gradients.

With proper initialization, variance stays $\approx 1$ at every layer, and training proceeds smoothly
from the first iteration.

---

## 6. Batch Normalization

### The Problem: Internal Covariate Shift

During training, as weights update, the distribution of inputs to each layer *changes*. Layer 3 is
trying to learn a mapping, but its input distribution (the output of layer 2) keeps shifting.

**Internal covariate shift** (Ioffe & Szegedy, 2015): each layer must continuously adapt to a
changing input distribution, which slows down training and requires careful learning rate tuning.

Think of it this way: you're trying to learn to hit a target, but someone keeps moving the target.
You would learn much faster if the target stayed still.

### The Solution: Normalize Each Layer's Inputs

For a mini-batch $\{z_1, z_2, \ldots, z_N\}$ at some layer, compute:

**Step 1: Compute batch statistics**
$$\mu_B = \frac{1}{N}\sum_{i=1}^N z_i$$
$$\sigma_B^2 = \frac{1}{N}\sum_{i=1}^N (z_i - \mu_B)^2$$

**Step 2: Normalize**
$$\hat{z}_i = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}$$

where $\varepsilon \sim 10^{-5}$ prevents division by zero.

**Step 3: Scale and shift (learnable parameters)**
$$y_i = \gamma \hat{z}_i + \beta$$

The parameters $\gamma$ and $\beta$ are *learned* during training. This is crucial: if the network
needs the un-normalized distribution, it can learn $\gamma = \sigma_B$ and $\beta = \mu_B$ to
recover the original values. Batch normalization does not reduce the network's representational power.

### Why It Works (Multiple Hypotheses)

**Original explanation (Ioffe & Szegedy):** Reduces internal covariate shift, allowing higher
learning rates and faster convergence.

**Smoothing the loss landscape (Santurkar et al., 2018):** BatchNorm makes the loss surface
significantly smoother (reducing the Lipschitz constant of the loss and its gradients), which makes
optimization easier regardless of internal covariate shift. This is now considered the more accurate
explanation.

**Implicit regularization:** By using batch statistics (which are noisy estimates of the true mean
and variance), BatchNorm injects noise into the computation, acting as a regularizer similar to
dropout.

### Training vs Inference

During **training**: use mini-batch statistics $\mu_B$, $\sigma_B^2$.

During **inference**: use running averages accumulated during training:
$$\mu_{\text{running}} \leftarrow (1 - \alpha)\mu_{\text{running}} + \alpha \mu_B$$
$$\sigma^2_{\text{running}} \leftarrow (1 - \alpha)\sigma^2_{\text{running}} + \alpha \sigma_B^2$$

where $\alpha$ (momentum) is typically 0.1. At test time, normalization uses these fixed statistics
so that the output is deterministic and independent of batch composition.

### Backward Pass

The backward pass through BatchNorm requires computing gradients through the normalization operations.
Let $\frac{\partial \mathcal{L}}{\partial y_i}$ be the incoming gradient.

$$\frac{\partial \mathcal{L}}{\partial \gamma} = \sum_{i=1}^N \frac{\partial \mathcal{L}}{\partial y_i} \cdot \hat{z}_i$$
$$\frac{\partial \mathcal{L}}{\partial \beta} = \sum_{i=1}^N \frac{\partial \mathcal{L}}{\partial y_i}$$

For the input gradients, the chain rule through the mean and variance computations gives:

$$\frac{\partial \mathcal{L}}{\partial \hat{z}_i} = \frac{\partial \mathcal{L}}{\partial y_i} \cdot \gamma$$

$$\frac{\partial \mathcal{L}}{\partial \sigma_B^2} = \sum_{i=1}^N \frac{\partial \mathcal{L}}{\partial \hat{z}_i} \cdot (z_i - \mu_B) \cdot \left(-\frac{1}{2}\right)(\sigma_B^2 + \varepsilon)^{-3/2}$$

$$\frac{\partial \mathcal{L}}{\partial \mu_B} = \sum_{i=1}^N \frac{\partial \mathcal{L}}{\partial \hat{z}_i} \cdot \frac{-1}{\sqrt{\sigma_B^2 + \varepsilon}}$$

$$\frac{\partial \mathcal{L}}{\partial z_i} = \frac{\partial \mathcal{L}}{\partial \hat{z}_i} \cdot \frac{1}{\sqrt{\sigma_B^2 + \varepsilon}} + \frac{\partial \mathcal{L}}{\partial \sigma_B^2} \cdot \frac{2(z_i - \mu_B)}{N} + \frac{\partial \mathcal{L}}{\partial \mu_B} \cdot \frac{1}{N}$$

---

## 7. Dropout

### The Problem: Overfitting Through Co-adaptation

In a neural network, neurons can learn to rely on specific other neurons ("co-adapt"). Neuron A
might only work correctly when neuron B provides a specific output. This creates fragile, specialized
pathways that memorize training data but fail to generalize.

### The Solution: Random Masking

**During training:** For each training sample, independently set each neuron's output to zero with
probability $p$ (typically $p = 0.5$ for hidden layers, $p = 0.2$ for input layers):

$$\tilde{a}_i = \frac{m_i \cdot a_i}{1 - p}$$

where $m_i \sim \text{Bernoulli}(1 - p)$ is a random mask. The division by $(1 - p)$ is called
**inverted dropout** — it rescales activations so that the expected value remains unchanged:

$$\mathbb{E}[\tilde{a}_i] = \frac{(1-p) \cdot a_i}{1-p} = a_i$$

**During inference:** Use all neurons with no masking, no scaling. Because we used inverted dropout
during training, the expected activations are already calibrated.

### Interpretation as Ensemble of Sub-Networks

A network with $n$ neurons and dropout generates $2^n$ possible sub-networks (each neuron is either
present or absent). Training with dropout is *approximately* training all $2^n$ sub-networks
simultaneously, sharing weights. At inference, using all neurons with rescaled weights is
*approximately* averaging the predictions of all $2^n$ sub-networks.

This is equivalent to a **geometric ensemble** — an exponentially large ensemble trained with shared
parameters. Srivastava et al. (2014) showed this connection formally.

### Why It Works

1. **Breaks co-adaptation:** Each neuron cannot rely on any specific other neuron being present, so
   it must learn robust, independent features.

2. **Implicit regularization:** Dropout effectively adds noise to the hidden representations,
   preventing the network from memorizing training examples.

3. **Approximate Bayesian inference:** Gal & Ghahramani (2016) showed that a network trained with
   dropout approximates a Gaussian process. Dropout at test time (Monte Carlo dropout) provides
   uncertainty estimates.

4. **Weight scaling effect:** Dropout with rate $p$ has a similar effect to $L^2$ regularization with
   a coefficient proportional to $p$, but adapts to the scale of the activations.

### The Backward Pass

During training, the backward pass simply applies the same mask:

$$\frac{\partial \mathcal{L}}{\partial a_i} = \frac{m_i}{1-p} \cdot \frac{\partial \mathcal{L}}{\partial \tilde{a}_i}$$

Neurons that were dropped contribute zero gradient — they don't learn on that training step. This is
correct: they didn't participate in the forward pass, so they should not receive gradients.

---

## 8. Feature Learning: What Hidden Layers Actually Learn

### Representations, Not Rules

The most important insight about MLPs (and deep networks generally) is that **hidden layers learn
representations**. They do not learn explicit rules; they learn to transform raw input into a
representation that makes the target task easy.

Consider classifying handwritten digits:
- **Raw pixels:** Two 7s might differ in every pixel. The classes are tangled in pixel space.
- **After layer 1:** The network might detect strokes, edges, curves. Now digits with similar strokes
  are nearby in representation space.
- **After layer 2:** The network combines strokes into structural features — loops, crossings,
  endpoints. Now digits with similar structure are nearby.
- **After layer 3:** The representation is nearly linearly separable. A simple linear classifier can
  read off the digit.

Each layer **disentangles** the factors of variation, gradually untangling the data manifold until the
target task becomes trivially solvable by a linear layer.

### What Representations Look Like

For an MLP with hidden layers of width $h$, each sample is mapped to a point in $\mathbb{R}^h$.
The hidden representation $a^{(\ell)} \in \mathbb{R}^h$ is a **learned coordinate system** for the
data.

Good representations have these properties:
- **Separation:** Different classes map to different regions.
- **Invariance:** Irrelevant variations (style, noise) are discarded.
- **Smoothness:** Similar inputs map to nearby representations.
- **Disentanglement:** Different factors of variation align with different dimensions.

### The Manifold Hypothesis

Real-world high-dimensional data (images, text, audio) typically lies on or near a low-dimensional
manifold embedded in the high-dimensional space. An image of a face has millions of pixels, but the
meaningful variations (pose, expression, lighting) span a manifold of perhaps a few hundred
dimensions.

Each hidden layer of an MLP progressively **unfolds** this manifold, mapping the tangled, curved
data surface into a flatter, more linearly separable representation. By the final hidden layer, what
was a complex nonlinear problem has been transformed into a nearly linear one.

### Probing Hidden Representations

You can observe feature learning experimentally:
1. **Visualize activations:** Plot the hidden layer outputs for different inputs. Similar inputs
   should cluster.
2. **Linear probing:** Train a linear classifier on hidden representations. If it achieves high
   accuracy, the representation is good (linearly separable).
3. **Dimensionality reduction:** Apply t-SNE or PCA to hidden representations. Good representations
   show clear class separation.
4. **Weight visualization:** For the first layer, each row of $W^{(1)}$ is a "template" — the
   pattern that maximally activates that neuron. These often look like edges, gabor filters, or
   other interpretable features.

### Feature Learning vs Feature Engineering

Before deep learning, practitioners manually designed features (SIFT, HOG, MFCC) tailored to each
domain. The revolution of deep learning is that the network *learns its own features* from data.

The MLP's hidden layers are a **learned feature extractor**. The output layer is a simple classifier
(or regressor) applied to these learned features. This is why the same architecture (MLP) can work on
images, audio, tabular data, and many other domains — it learns the appropriate features for each.

This is also why **transfer learning** works: the features learned for one task often transfer to
related tasks, because the early layers learn general representations (edges, frequencies, patterns)
that are useful across many problems.

---

## Summary

The MLP is the foundational architecture of deep learning. Its key lessons:

| Concept | Lesson |
|---------|--------|
| Layer stacking | Composition of simple functions yields complex functions |
| Matrix formulation | Neural networks are sequences of linear transforms + nonlinearities |
| Universal approximation | MLPs *can* represent anything, but that alone is not useful |
| Depth > Width | Depth gives exponential expressiveness; hierarchy matches data structure |
| Initialization | Variance control across layers is essential for trainability |
| Batch normalization | Stabilizing intermediate distributions smooths the loss landscape |
| Dropout | Random masking prevents co-adaptation and acts as implicit ensemble |
| Feature learning | Hidden layers learn representations that make the task linearly separable |

Every architecture that follows — CNNs, RNNs, Transformers — builds on these ideas. The MLP
is where you learn them.

---

## References

- Cybenko, G. (1989). *Approximation by superpositions of a sigmoidal function.*
- Hornik, K. (1991). *Approximation capabilities of multilayer feedforward networks.*
- Glorot, X. & Bengio, Y. (2010). *Understanding the difficulty of training deep feedforward neural networks.*
- He, K. et al. (2015). *Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification.*
- Ioffe, S. & Szegedy, C. (2015). *Batch normalization: Accelerating deep network training by reducing internal covariate shift.*
- Srivastava, N. et al. (2014). *Dropout: A simple way to prevent neural networks from overfitting.*
- Montufar, G. et al. (2014). *On the number of linear regions of deep neural networks.*
- Telgarsky, M. (2016). *Benefits of depth in neural networks.*
- Santurkar, S. et al. (2018). *How does batch normalization help optimization?*
- Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian approximation: Representing model uncertainty in deep learning.*

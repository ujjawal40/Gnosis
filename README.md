# Gnosis

**A first-principles journey through the mathematics and science of intelligence — from linear algebra to research frontiers.**

The goal: understand AI deeply enough to find what's broken and fix it. Not incremental improvements — fundamental breakthroughs.

---

## The Chronological Path

Each module builds on the previous. No skipping. Every concept is implemented from scratch before using a library.

### Prerequisites Map

```
Linear Algebra ──→ Calculus ──→ Probability ──→ Information Theory ──→ Optimization
      │                │             │                  │                    │
      └────────────────┴─────────────┴──────────────────┴────────────────────┘
                                     │
                              Single Neuron (Perceptron)
                                     │
                              Backpropagation
                                     │
                         ┌───────────┼───────────┐
                         │           │           │
                        MLP         CNN      RNN/LSTM
                         │           │           │
                         └───────────┼───────────┘
                                     │
                            Embeddings & Word2Vec
                                     │
                              Attention Mechanism
                                     │
                               Transformers
                              ┌──────┼──────┐
                              │      │      │
                           Encoder Decoder  ViT
                              │      │      │
                              └──────┼──────┘
                                     │
                         ┌───────────┼───────────┐
                         │           │           │
                        VAE         GAN      Diffusion
                         │           │           │
                         └───────────┼───────────┘
                                     │
                          ┌──────────┼──────────┐
                          │          │          │
                     Neural ODE  Continual   Geometric
                                 Learning    Deep Learning
                                     │
                              Research Frontier
```

---

## Module 00: Mathematics Foundations

**Why:** Every neural network is a mathematical function. You cannot innovate on what you don't understand.

| # | Topic | Key Concepts | Status |
|---|-------|-------------|--------|
| 01 | Linear Algebra | Vectors, matrices, eigendecomposition, SVD, projections | ✅ |
| 02 | Calculus | Derivatives, chain rule, gradients, Jacobians, Hessians | ✅ |
| 03 | Probability & Statistics | Bayes theorem, distributions, MLE, MAP, expectation | ✅ |
| 04 | Information Theory | Entropy, KL divergence, mutual information, cross-entropy | ✅ |
| 05 | Optimization | Gradient descent, convexity, Lagrange multipliers, duality | ✅ |

## Module 01: Neural Foundations

**Why:** Before architectures, understand the atomic unit — a single neuron — and how learning actually works.

| # | Topic | Key Concepts | Status |
|---|-------|-------------|--------|
| 01 | Perceptron | Single neuron, linear decision boundary, convergence theorem | ✅ |
| 02 | Backpropagation | Computational graphs, chain rule, gradient flow | ✅ |
| 03 | Activation Functions | Sigmoid, tanh, ReLU, GELU — why each exists, gradients | ✅ |
| 04 | Loss Functions | MSE, cross-entropy, hinge — information-theoretic view | ✅ |

## Module 02: Classical Architectures

**Why:** Each architecture solves a specific inductive bias problem. Understanding *why* they work matters more than *how*.

| # | Topic | Key Concepts | Status |
|---|-------|-------------|--------|
| 01 | MLP | Universal approximation, depth vs width, feature learning | ✅ |
| 02 | CNN | Convolution, translation equivariance, hierarchical features | 🔲 |
| 03 | RNN/LSTM | Sequential processing, gating, vanishing gradients | 🔲 |
| 04 | Autoencoders | Compression, latent space, reconstruction | 🔲 |

## Module 03: Representation Learning

**Why:** How knowledge is encoded determines everything — generalization, efficiency, compositionality.

| # | Topic | Key Concepts | Status |
|---|-------|-------------|--------|
| 01 | Word2Vec | Distributional hypothesis, skip-gram, CBOW, negative sampling | 🔲 |
| 02 | Embedding Spaces | Geometry of embeddings, analogies, subspaces | 🔲 |
| 03 | Manifold Learning | t-SNE, UMAP, intrinsic dimensionality | 🔲 |
| 04 | Information Bottleneck | Tishby's theory, compression-prediction tradeoff | 🔲 |

## Module 04: Modern Architectures

**Why:** Transformers dominate. Understanding their mechanics deeply is prerequisite to improving them.

| # | Topic | Key Concepts | Status |
|---|-------|-------------|--------|
| 01 | Attention | Dot-product attention, multi-head, self-attention, complexity | 🔲 |
| 02 | Transformer | Full architecture, positional encoding, layer norm | 🔲 |
| 03 | Encoder Models | BERT, masked language modeling, bidirectional context | 🔲 |
| 04 | Decoder Models | GPT, autoregressive generation, causal masking | 🔲 |
| 05 | Vision Transformer | Patch embeddings, ViT, image as sequence | 🔲 |

## Module 05: Learning Dynamics

**Why:** Architectures are static. Understanding *how* they learn reveals what limits them.

| # | Topic | Key Concepts | Status |
|---|-------|-------------|--------|
| 01 | Loss Landscapes | Geometry, saddle points, mode connectivity, sharpness | 🔲 |
| 02 | Optimization Theory | Adam, learning rate schedules, natural gradient | 🔲 |
| 03 | Generalization | Double descent, bias-variance, PAC learning, NTK | 🔲 |
| 04 | Regularization | Dropout, weight decay, batch norm — why they work | 🔲 |

## Module 06: Generative Models

**Why:** Generation requires deep understanding — you can't create what you don't model.

| # | Topic | Key Concepts | Status |
|---|-------|-------------|--------|
| 01 | VAE | Variational inference, ELBO, reparameterization trick | 🔲 |
| 02 | GAN | Adversarial training, mode collapse, Wasserstein distance | 🔲 |
| 03 | Diffusion | Score matching, denoising, DDPM, classifier-free guidance | 🔲 |
| 04 | Video Generation | Temporal consistency, 3D convolutions, temporal attention | 🔲 |
| 05 | 3D Models | NeRF, point clouds, mesh generation, implicit representations | 🔲 |

## Module 07: Advanced Topics

**Why:** These are where current approaches break down — and where breakthroughs hide.

| # | Topic | Key Concepts | Status |
|---|-------|-------------|--------|
| 01 | Continual Learning | Catastrophic forgetting, EWC, progressive nets | 🔲 |
| 02 | Neural ODE | Continuous-depth networks, adjoint method | 🔲 |
| 03 | Reinforcement Learning | MDPs, policy gradient, RLHF | 🔲 |
| 04 | Geometric Deep Learning | Equivariance, gauge theory, message passing | 🔲 |

## Module 08: Research Frontier

**Why:** This is where the breakthrough lives.

| # | Topic | Key Concepts | Status |
|---|-------|-------------|--------|
| 01 | Scaling Laws | Chinchilla, power laws, compute-optimal training | 🔲 |
| 02 | Mixture of Experts | Sparse activation, routing, conditional computation | 🔲 |
| 03 | State Space Models | Mamba, S4, linear recurrences | 🔲 |
| 04 | Novel Architectures | Geometric attention, compositional nets, dynamic manifolds | 🔲 |
| 05 | Breakthrough Ideas | Your original contributions | 🔲 |

---

## Principles

1. **Implement before you import.** Build it from scratch in NumPy first. Use PyTorch only after you understand what it's doing.
2. **Math first, code second.** Derive the gradients by hand before coding them.
3. **Question everything.** "Why does this work?" is more important than "how do I use this?"
4. **Small experiments prove big ideas.** MNIST before ImageNet. Toy problems before benchmarks.
5. **Write what you learn.** Each module includes theory notes. If you can't explain it, you don't understand it.

## Structure

Each module contains:
- `theory.md` — Mathematical foundations, derivations, intuitions
- `implementation.py` — From-scratch implementation (NumPy only where possible)
- `experiments.py` — Experiments that reveal key properties
- `references.md` — Papers, books, and resources

## Setup

```bash
pip install -r requirements.txt
```

Uses minimal dependencies by design: NumPy for math, Matplotlib for visualization, PyTorch only for later modules.

---

*"The purpose of computing is insight, not numbers."* — Richard Hamming

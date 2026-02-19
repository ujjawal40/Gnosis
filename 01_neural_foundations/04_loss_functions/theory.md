# Loss Functions — The Objective of Learning

## The Central Idea

A loss function measures **how wrong** the model is. Training is the process of minimizing this function. But loss functions are not arbitrary choices — they arise from **probabilistic assumptions** about your data.

**The key insight:** choosing a loss function = choosing a probability distribution.

---

## 1. Loss as Negative Log-Likelihood

The deepest way to understand loss functions is through probability theory.

**Setup:** Given data (x, y), a model predicts p(y|x; θ). We want parameters θ that make the data most likely.

**Maximum Likelihood Estimation:**
```
θ* = argmax_θ  Π p(yi | xi; θ)
    = argmax_θ  Σ log p(yi | xi; θ)    (take log, sum is easier)
    = argmin_θ  -Σ log p(yi | xi; θ)   (negate to minimize)
```

That last line **is** the loss function. Every standard loss is a negative log-likelihood under some distribution.

---

## 2. Mean Squared Error (MSE)

**Formula:**
```
L = (1/n) Σ (yi - ŷi)²
```

**Probabilistic Derivation:**

Assume y = f(x) + ε where ε ~ N(0, σ²). Then:
```
p(y | x; θ) = (1/√(2πσ²)) exp(-(y - f(x;θ))² / (2σ²))

-log p(y|x;θ) = (y - f(x;θ))² / (2σ²) + const
```

MSE is negative log-likelihood under a **Gaussian assumption**. The constant σ² just scales the learning rate.

**Gradient:**
```
∂L/∂ŷi = -2(yi - ŷi) / n
```

This gradient is proportional to the error — large errors get large updates.

**When to use:** Regression tasks where errors are roughly Gaussian.

**When NOT to use:** Classification — MSE has bad gradient properties for probabilities (see below).

---

## 3. Binary Cross-Entropy (BCE)

**Formula:**
```
L = -(1/n) Σ [yi log(ŷi) + (1 - yi) log(1 - ŷi)]
```

where yi ∈ {0, 1} and ŷi ∈ (0, 1) is a predicted probability.

**Probabilistic Derivation:**

Assume y ~ Bernoulli(p), where p = σ(f(x; θ)):
```
p(y|x;θ) = p^y · (1-p)^(1-y)

-log p(y|x;θ) = -[y log(p) + (1-y) log(1-p)]
```

That's exactly binary cross-entropy.

**Gradient:**
```
∂L/∂ŷi = -(yi/ŷi - (1-yi)/(1-ŷi)) / n
```

**Why better than MSE for classification:**

With MSE + sigmoid output: if the model is very wrong (ŷ ≈ 0 but y = 1), the sigmoid is saturated, so the gradient is tiny. The model is stuck.

With BCE + sigmoid: the gradient is `(ŷ - y)`, which is large when wrong. No saturation problem.

This is why we pair sigmoid with BCE and softmax with categorical CE.

---

## 4. Categorical Cross-Entropy

**Formula:**
```
L = -(1/n) Σᵢ Σⱼ yij log(ŷij)
```

where yij is a one-hot vector and ŷij is the softmax output.

**Probabilistic Derivation:**

Assume y ~ Categorical(p₁, ..., pK) where p = softmax(f(x; θ)):
```
p(y=k | x;θ) = softmax(zk)

-log p(y=k | x;θ) = -zk + log(Σ exp(zj))
```

**Gradient (with softmax):**
```
∂L/∂zi = ŷi - yi  (softmax output - one-hot target)
```

This incredibly clean gradient is why softmax + cross-entropy are always used together.

---

## 5. Hinge Loss

**Formula:**
```
L = max(0, 1 - yi · ŷi)
```

where yi ∈ {-1, +1}.

**Origin:** Support Vector Machines (SVMs). The loss is zero when the prediction is correct with margin ≥ 1.

**Key property:** Once the model is confident enough (margin > 1), it stops learning. This creates a "max-margin" classifier.

**Gradient:**
```
∂L/∂ŷi = -yi  if  yi·ŷi < 1
          0    otherwise
```

---

## 6. Huber Loss

**Formula:**
```
L = 0.5 (y - ŷ)²         if |y - ŷ| ≤ δ
    δ|y - ŷ| - 0.5δ²     otherwise
```

**Intuition:** MSE for small errors, MAE for large errors. This makes it robust to outliers while still being smooth near zero.

---

## 7. KL Divergence as Loss

**Formula:**
```
D_KL(p || q) = Σ p(x) log(p(x)/q(x))
```

**Relationship to cross-entropy:**
```
D_KL(p || q) = H(p, q) - H(p)
```

Since H(p) is constant w.r.t. model parameters, minimizing KL divergence = minimizing cross-entropy.

**Usage:** VAE training, knowledge distillation, policy optimization (RL).

---

## 8. Gradient Properties Comparison

| Loss | Gradient when very wrong | Gradient when close | Robustness |
|------|------------------------|--------------------|----|
| MSE | Large (proportional to error) | Small (proportional) | Sensitive to outliers |
| BCE | Large (no saturation) | Small | Good for classification |
| Hinge | Constant (±1) or 0 | Zero (margin satisfied) | Robust |
| Huber | Constant (±δ) | Small (proportional) | Robust to outliers |

---

## 9. The Deep Connection

Loss functions, probability distributions, and information theory are three views of the same thing:

- **Choosing MSE** = assuming Gaussian noise = minimizing variance
- **Choosing cross-entropy** = assuming Bernoulli/Categorical = minimizing surprise (bits)
- **Choosing KL divergence** = measuring information gap between model and truth

Understanding this triangle (loss ↔ probability ↔ information) is one of the most important insights in machine learning.

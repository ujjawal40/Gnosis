# Probability & Statistics: The Language of Uncertainty

**Why this matters:** Every neural network outputs a probability distribution. Training is maximum likelihood estimation. Regularization is a prior. Loss functions are negative log-likelihoods. Generative models *are* probability distributions. If you don't understand probability theory, you're pattern-matching code without understanding what it's doing.

---

## 1. The Foundations: What Is Probability?

### 1.1 Sample Space, Events, and the Kolmogorov Axioms

Probability theory starts with three objects:

**Sample space** Ω: The set of all possible outcomes of an experiment.
- Coin flip: Ω = {H, T}
- Die roll: Ω = {1, 2, 3, 4, 5, 6}
- Neural network output: Ω = ℝ^n (continuous)

**Events**: Subsets of the sample space. An event A ⊆ Ω is a collection of outcomes.
- "Rolling an even number": A = {2, 4, 6}
- Events form a σ-algebra F (closed under complement, countable union, contains Ω)

**Probability measure** P: A function P: F → [0, 1] satisfying the Kolmogorov axioms.

### The Three Axioms (Kolmogorov, 1933)

These three axioms are the entire foundation. Everything else is derived from them.

**Axiom 1 (Non-negativity):** For any event A:
```
P(A) ≥ 0
```

**Axiom 2 (Normalization):** The total probability is 1:
```
P(Ω) = 1
```

**Axiom 3 (Countable Additivity):** For any countable collection of mutually exclusive events A₁, A₂, ...:
```
P(A₁ ∪ A₂ ∪ ...) = P(A₁) + P(A₂) + ...
```

**Key consequences derived from these axioms:**

```
P(∅) = 0                           (impossible event)
P(Aᶜ) = 1 - P(A)                  (complement rule)
P(A ∪ B) = P(A) + P(B) - P(A ∩ B) (inclusion-exclusion)
P(A) ≤ 1                           (bounded above)
If A ⊆ B, then P(A) ≤ P(B)        (monotonicity)
```

**Proof of complement rule from axioms:**
- A and Aᶜ are mutually exclusive, and A ∪ Aᶜ = Ω
- By Axiom 3: P(A ∪ Aᶜ) = P(A) + P(Aᶜ)
- By Axiom 2: P(Ω) = 1
- Therefore: P(A) + P(Aᶜ) = 1, so P(Aᶜ) = 1 - P(A) ∎

### 1.2 Interpretations of Probability

The axioms don't tell you what probability *means*. There are two main views:

**Frequentist:** Probability is the long-run frequency of an event. P(heads) = 0.5 means if you flip a coin infinitely many times, half will be heads. Problem: What does P("it rains tomorrow") = 0.3 mean? You can't repeat tomorrow.

**Bayesian:** Probability is a degree of belief. P(heads) = 0.5 means you're equally uncertain about heads vs tails. P("it rains tomorrow") = 0.3 means you assign 30% credence to rain. This is the view that makes probability useful for machine learning — we're quantifying uncertainty about model parameters, predictions, and the world.

---

## 2. Conditional Probability and Independence

### 2.1 Conditional Probability

**Definition:** The probability of A given that B has occurred:

```
P(A|B) = P(A ∩ B) / P(B),    provided P(B) > 0
```

**Intuition:** We're restricting the universe to outcomes where B happened, then asking how likely A is in that restricted universe.

**Example:** Roll a fair die. A = {6}, B = {even} = {2, 4, 6}.
```
P(A|B) = P({6}) / P({2,4,6}) = (1/6) / (3/6) = 1/3
```
Knowing the die is even makes 6 more likely (1/3 vs 1/6).

### The Chain Rule

From the definition of conditional probability, we can rearrange to get the **product rule**:

```
P(A ∩ B) = P(A|B) · P(B)
```

This generalizes to the **chain rule**:

```
P(A₁ ∩ A₂ ∩ ... ∩ Aₙ) = P(A₁) · P(A₂|A₁) · P(A₃|A₁,A₂) · ... · P(Aₙ|A₁,...,Aₙ₋₁)
```

**Why this matters for ML:** Autoregressive language models decompose the probability of a sequence using exactly this chain rule:

```
P(x₁, x₂, ..., xₙ) = P(x₁) · P(x₂|x₁) · P(x₃|x₁,x₂) · ... · P(xₙ|x₁,...,xₙ₋₁)
```

This is GPT in one equation.

### 2.2 The Law of Total Probability

If B₁, B₂, ..., Bₙ partition the sample space (mutually exclusive, cover everything):

```
P(A) = Σᵢ P(A|Bᵢ) · P(Bᵢ)
```

**Intuition:** To compute P(A), break it into cases. For each possible scenario Bᵢ, compute how likely A is under that scenario, weighted by how likely the scenario is.

**Example:** A disease has 1% prevalence. A test has 99% sensitivity (true positive rate) and 95% specificity (true negative rate). What's the probability of testing positive?

```
P(+) = P(+|disease) · P(disease) + P(+|healthy) · P(healthy)
     = 0.99 · 0.01 + 0.05 · 0.99
     = 0.0099 + 0.0495
     = 0.0594
```

About 6% of people test positive, even though only 1% have the disease. Most positives are false positives.

### 2.3 Independence

**Definition:** Events A and B are independent if:

```
P(A ∩ B) = P(A) · P(B)
```

Equivalently: P(A|B) = P(A). Knowing B tells you nothing about A.

**Conditional independence:** A and B are conditionally independent given C if:

```
P(A ∩ B | C) = P(A|C) · P(B|C)
```

**Why this matters for ML:** Naive Bayes assumes features are conditionally independent given the class. This is almost always wrong, but it works surprisingly well — one of the enduring mysteries of machine learning.

---

## 3. Bayes' Theorem

### 3.1 Derivation

Starting from the definition of conditional probability applied both ways:

```
P(A|B) = P(A ∩ B) / P(B)
P(B|A) = P(A ∩ B) / P(A)
```

From the second equation: P(A ∩ B) = P(B|A) · P(A)

Substituting into the first:

```
P(A|B) = P(B|A) · P(A) / P(B)
```

This is **Bayes' Theorem**. That's the entire derivation — it's just the definition of conditional probability applied twice.

### 3.2 The Named Components

Rename to make the roles clear. We have a **hypothesis** H and **data** D:

```
P(H|D) = P(D|H) · P(H) / P(D)
```

| Term | Name | Role |
|------|------|------|
| P(H\|D) | **Posterior** | What we believe after seeing data |
| P(D\|H) | **Likelihood** | How probable the data is under this hypothesis |
| P(H) | **Prior** | What we believed before seeing data |
| P(D) | **Evidence** (marginal likelihood) | Normalization constant; how probable the data is under any hypothesis |

The denominator P(D) is computed via the law of total probability:

```
P(D) = Σₕ P(D|H=h) · P(H=h)
```

Since P(D) doesn't depend on H, we often write:

```
P(H|D) ∝ P(D|H) · P(H)
posterior ∝ likelihood × prior
```

### 3.3 Intuition: Bayes as a Learning Rule

Bayes' theorem is a recipe for updating beliefs:

1. Start with a **prior** P(H) — your initial belief about the hypothesis.
2. Observe **data** D.
3. Compute the **likelihood** P(D|H) — how well the hypothesis explains the data.
4. Update to the **posterior** P(H|D) — your refined belief.

Each new observation updates the posterior, which becomes the prior for the next observation. This is **sequential Bayesian updating** — the mathematical formalization of learning.

### 3.4 Example: Disease Testing (Revisited)

Using the disease example from above: You tested positive. What's the probability you have the disease?

```
P(disease|+) = P(+|disease) · P(disease) / P(+)
             = (0.99 · 0.01) / 0.0594
             = 0.0099 / 0.0594
             ≈ 0.167
```

Only 16.7% chance of having the disease despite a positive test with 99% sensitivity. The low prior (1% prevalence) dominates. This is the **base rate fallacy** — ignoring the prior leads to dramatically wrong conclusions.

### 3.5 Example: Coin Flipping with Beta Prior

Suppose we have a coin with unknown bias θ = P(heads). We want to infer θ from observed flips.

**Prior:** We start with a Beta distribution: P(θ) = Beta(α, β). This encodes our prior belief. Beta(1,1) = Uniform (no prior knowledge). Beta(10,10) = strong belief the coin is fair.

**Likelihood:** After observing h heads and t tails:
```
P(data|θ) = θʰ · (1-θ)ᵗ
```

**Posterior:** By Bayes' theorem (and the magic of conjugate priors):
```
P(θ|data) = Beta(α + h, β + t)
```

The posterior is also a Beta distribution. We just add our observed counts to the prior parameters. This is "pseudo-counts" intuition: α and β represent "imaginary" prior observations.

After 7 heads and 3 tails with a Beta(1,1) prior:
```
P(θ|data) = Beta(8, 4)
Expected value: 8/12 ≈ 0.667
```

The posterior concentrates around the observed frequency, pulled slightly toward 0.5 by the prior.

---

## 4. Random Variables

### 4.1 Definition

A **random variable** X is a function from the sample space to the real numbers: X: Ω → ℝ.

It maps outcomes to numbers. "The number shown on a die" is a random variable. "The height of a randomly selected person" is a random variable. "The output of a neural network" is a random variable.

### 4.2 Discrete Random Variables

X takes values from a countable set (integers, categories, etc.).

**Probability Mass Function (PMF):**
```
p(x) = P(X = x)
```

Properties:
- p(x) ≥ 0 for all x
- Σₓ p(x) = 1

**Example (fair die):** p(x) = 1/6 for x ∈ {1, 2, 3, 4, 5, 6}, 0 otherwise.

### 4.3 Continuous Random Variables

X takes values from an uncountable set (real numbers, intervals, etc.).

**Probability Density Function (PDF):**
```
f(x) such that P(a ≤ X ≤ b) = ∫ₐᵇ f(x) dx
```

Critical subtlety: **f(x) is NOT a probability.** It's a density. For continuous variables, P(X = x) = 0 for any specific x. Only intervals have nonzero probability.

Properties:
- f(x) ≥ 0 for all x
- ∫₋∞^∞ f(x) dx = 1
- f(x) can be greater than 1 (it's a density, not a probability)

### 4.4 Cumulative Distribution Function (CDF)

Works for both discrete and continuous:

```
F(x) = P(X ≤ x)
```

For continuous: F(x) = ∫₋∞ˣ f(t) dt, and f(x) = dF/dx.

Properties:
- F is non-decreasing
- lim_{x→-∞} F(x) = 0
- lim_{x→+∞} F(x) = 1
- P(a < X ≤ b) = F(b) - F(a)

---

## 5. Expectation, Variance, and Covariance

### 5.1 Expectation (Mean)

**Discrete:**
```
E[X] = Σₓ x · p(x)
```

**Continuous:**
```
E[X] = ∫₋∞^∞ x · f(x) dx
```

**Intuition:** The "center of mass" of the distribution. If you sampled X infinitely many times, the average would converge to E[X] (law of large numbers).

**Key properties (all follow from linearity of sums/integrals):**
```
E[aX + b] = aE[X] + b                  (linearity)
E[X + Y] = E[X] + E[Y]                 (always true, even if dependent)
E[XY] = E[X]·E[Y]                      (only if independent)
E[g(X)] = Σₓ g(x)·p(x)                (law of the unconscious statistician)
```

**Linearity of expectation** is the single most useful property in all of probability. It holds regardless of dependence between variables.

### 5.2 Variance

```
Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²
```

**The second form** is derived by expanding:
```
Var(X) = E[(X - μ)²]
       = E[X² - 2μX + μ²]
       = E[X²] - 2μE[X] + μ²
       = E[X²] - 2μ² + μ²
       = E[X²] - μ²
```

**Intuition:** How spread out the distribution is. Variance is the expected squared deviation from the mean.

**Standard deviation:** σ = √Var(X). Same units as X.

**Key properties:**
```
Var(aX + b) = a²Var(X)              (constants scale quadratically, shifts don't matter)
Var(X + Y) = Var(X) + Var(Y)        (only if independent)
Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)  (general case)
```

### 5.3 Covariance and Correlation

**Covariance:**
```
Cov(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]
```

Measures how X and Y move together. Positive = both increase together. Negative = one increases while other decreases. Zero = no linear relationship (but they could still be dependent!).

**Correlation (Pearson):**
```
ρ(X, Y) = Cov(X, Y) / (σ_X · σ_Y)
```

Normalized covariance: ρ ∈ [-1, 1]. Unit-free measure of linear relationship.

**Critical warning:** Zero correlation does NOT imply independence. Example: X ~ Uniform(-1,1), Y = X². Then Cov(X,Y) = 0 but Y is completely determined by X.

---

## 6. Key Distributions

### 6.1 Bernoulli Distribution

The simplest distribution: a single coin flip.

```
X ~ Bernoulli(θ),     θ ∈ [0, 1]

P(X = 1) = θ
P(X = 0) = 1 - θ

Compact form: p(x) = θˣ(1-θ)¹⁻ˣ,    x ∈ {0, 1}

E[X] = θ
Var(X) = θ(1 - θ)
```

**Why it matters for ML:** Binary classification. The output of a sigmoid is the θ parameter of a Bernoulli distribution.

### 6.2 Binomial Distribution

The number of successes in n independent Bernoulli trials.

```
X ~ Binomial(n, θ)

P(X = k) = C(n,k) · θᵏ · (1-θ)ⁿ⁻ᵏ,    k = 0, 1, ..., n

where C(n,k) = n! / (k!(n-k)!)

E[X] = nθ
Var(X) = nθ(1-θ)
```

### 6.3 Categorical Distribution

Generalization of Bernoulli to K categories.

```
X ~ Categorical(θ₁, θ₂, ..., θ_K),    Σᵢ θᵢ = 1

P(X = k) = θ_k

E[1_{X=k}] = θ_k
```

**Why it matters for ML:** The softmax output of a classifier defines a Categorical distribution over classes. Cross-entropy loss is the negative log-probability of this distribution.

### 6.4 Multinomial Distribution

Generalization of Binomial to K categories. The number of times each category appears in n independent Categorical trials.

```
(X₁, ..., X_K) ~ Multinomial(n, θ₁, ..., θ_K)

P(X₁=x₁, ..., X_K=x_K) = (n! / (x₁!...x_K!)) · θ₁^x₁ · ... · θ_K^x_K

where Σᵢ xᵢ = n

E[Xᵢ] = nθᵢ
Var(Xᵢ) = nθᵢ(1-θᵢ)
Cov(Xᵢ, Xⱼ) = -nθᵢθⱼ    (for i ≠ j)
```

### 6.5 The Gaussian (Normal) Distribution

The most important distribution in all of statistics.

```
X ~ N(μ, σ²)

f(x) = (1 / √(2πσ²)) · exp(-(x-μ)² / (2σ²))

E[X] = μ
Var(X) = σ²
```

**Standard normal:** μ = 0, σ² = 1. Written Z ~ N(0, 1).

**Multivariate Gaussian:**
```
X ~ N(μ, Σ)

f(x) = (1 / √((2π)^d |Σ|)) · exp(-½(x-μ)ᵀΣ⁻¹(x-μ))
```

where μ is a d-dimensional mean vector and Σ is a d×d covariance matrix.

---

## 7. The Gaussian Distribution: Why It's Everywhere

### 7.1 The Central Limit Theorem

**Theorem (CLT):** Let X₁, X₂, ..., Xₙ be i.i.d. random variables with mean μ and variance σ². Then as n → ∞:

```
(X̄ₙ - μ) / (σ/√n) →ᵈ N(0, 1)

where X̄ₙ = (1/n) Σᵢ Xᵢ
```

**In words:** The average of many independent random variables is approximately Gaussian, *regardless of the original distribution.* This is why the Gaussian appears everywhere:

- Measurement error = sum of many small independent errors → Gaussian
- Heights = sum of many genetic and environmental factors → Gaussian
- Financial returns (short-term) ≈ sum of many trades → approximately Gaussian
- Noise in neural network gradients → approximately Gaussian (sum of per-example gradients)

### 7.2 Deriving the Gaussian PDF

Why does the Gaussian have the specific form (1/√(2πσ²)) · exp(-(x-μ)²/(2σ²))? Here's one approach using maximum entropy.

**Setup:** Among all distributions with a given mean μ and variance σ², which has the *maximum entropy* (least additional assumptions)?

We want to maximize:
```
H[f] = -∫ f(x) ln f(x) dx
```
subject to:
```
∫ f(x) dx = 1
∫ x·f(x) dx = μ
∫ (x-μ)²·f(x) dx = σ²
```

Using Lagrange multipliers λ₀, λ₁, λ₂:
```
L = -∫ f(x) ln f(x) dx - λ₀(∫f(x)dx - 1) - λ₁(∫xf(x)dx - μ) - λ₂(∫(x-μ)²f(x)dx - σ²)
```

Taking the functional derivative δL/δf = 0:
```
-ln f(x) - 1 - λ₀ - λ₁x - λ₂(x-μ)² = 0
ln f(x) = -1 - λ₀ - λ₁x - λ₂(x-μ)²
f(x) = exp(-1 - λ₀ - λ₁x - λ₂(x-μ)²)
```

Solving for the Lagrange multipliers using the constraints gives:
```
λ₁ = 0 (symmetry around μ handles the mean constraint)
λ₂ = 1/(2σ²)
exp(-1 - λ₀) = 1/√(2πσ²) (from normalization)
```

Result:
```
f(x) = (1/√(2πσ²)) · exp(-(x-μ)²/(2σ²))
```

**The Gaussian is the maximum entropy distribution for a given mean and variance.** It makes the fewest assumptions beyond what you've specified. This is why assuming Gaussian noise is "principled" — you're being maximally uncertain given the constraints.

---

## 8. Maximum Likelihood Estimation (MLE)

### 8.1 The Core Idea

Given observed data D = {x₁, x₂, ..., xₙ} and a parametric model p(x|θ), find the parameters θ that make the observed data most likely:

```
θ_MLE = argmax_θ P(D|θ)
       = argmax_θ Πᵢ p(xᵢ|θ)       (assuming i.i.d. data)
       = argmax_θ Σᵢ ln p(xᵢ|θ)     (log turns product into sum)
```

The **log-likelihood** is more convenient because:
1. Products become sums (easier to differentiate)
2. Avoids numerical underflow (products of small numbers → 0)
3. log is monotonically increasing, so maximizing log-likelihood = maximizing likelihood

### 8.2 MLE for Gaussian Parameters

**Data:** x₁, ..., xₙ ~ N(μ, σ²) i.i.d.

**Log-likelihood:**
```
ℓ(μ, σ²) = Σᵢ ln N(xᵢ|μ, σ²)
          = Σᵢ [-½ln(2π) - ½ln(σ²) - (xᵢ-μ)²/(2σ²)]
          = -n/2·ln(2π) - n/2·ln(σ²) - 1/(2σ²)·Σᵢ(xᵢ-μ)²
```

**Optimize for μ:** Take ∂ℓ/∂μ and set to 0:
```
∂ℓ/∂μ = 1/σ² · Σᵢ(xᵢ - μ) = 0
Σᵢ(xᵢ - μ) = 0
Σᵢ xᵢ - nμ = 0

μ_MLE = (1/n) Σᵢ xᵢ = x̄    ← the sample mean!
```

**Optimize for σ²:** Take ∂ℓ/∂(σ²) and set to 0:
```
∂ℓ/∂(σ²) = -n/(2σ²) + 1/(2σ⁴)·Σᵢ(xᵢ-μ)² = 0
n/(2σ²) = 1/(2σ⁴)·Σᵢ(xᵢ-μ)²
nσ² = Σᵢ(xᵢ-μ)²

σ²_MLE = (1/n) Σᵢ(xᵢ - x̄)²    ← the sample variance!
```

Note: This is the *biased* estimator. The unbiased version divides by (n-1). The bias vanishes as n grows.

**MLE recovers the statistics we already use intuitively.** The sample mean and sample variance are not arbitrary — they are the maximum likelihood estimates under a Gaussian model.

### 8.3 MLE for Bernoulli Parameter

**Data:** x₁, ..., xₙ ~ Bernoulli(θ) i.i.d. Each xᵢ ∈ {0, 1}.

**Log-likelihood:**
```
ℓ(θ) = Σᵢ [xᵢ ln θ + (1-xᵢ) ln(1-θ)]
      = k·ln θ + (n-k)·ln(1-θ)

where k = Σᵢ xᵢ (number of successes)
```

**Optimize:**
```
∂ℓ/∂θ = k/θ - (n-k)/(1-θ) = 0
k(1-θ) = (n-k)θ
k - kθ = nθ - kθ
k = nθ

θ_MLE = k/n    ← the sample proportion!
```

Again, MLE gives us what intuition already suggests.

### 8.4 MLE and Loss Functions

Here's the punchline that connects probability to neural network training:

**For classification** (Categorical output with softmax):
```
Negative log-likelihood = -Σᵢ ln p(yᵢ|xᵢ, θ) = cross-entropy loss
```

**For regression** (Gaussian output):
```
Negative log-likelihood = -Σᵢ ln N(yᵢ|f_θ(xᵢ), σ²)
                        = (1/2σ²)·Σᵢ(yᵢ - f_θ(xᵢ))² + const
                        ∝ MSE loss
```

**Training a neural network by minimizing a loss function IS performing maximum likelihood estimation.** Cross-entropy loss assumes a Categorical model. MSE loss assumes a Gaussian model with constant variance. Every loss function implies a probabilistic model.

---

## 9. Maximum A Posteriori (MAP) Estimation

### 9.1 From MLE to MAP

MLE finds θ that maximizes P(D|θ). But what if we have prior knowledge about θ?

**MAP** uses Bayes' theorem:
```
θ_MAP = argmax_θ P(θ|D)
      = argmax_θ P(D|θ) · P(θ) / P(D)
      = argmax_θ P(D|θ) · P(θ)          (P(D) doesn't depend on θ)
      = argmax_θ [ln P(D|θ) + ln P(θ)]
      = argmax_θ [log-likelihood + log-prior]
```

MAP = MLE + a prior term. The prior biases the estimate toward values we believe are more likely *a priori*.

### 9.2 MAP Is Regularization

This connection is profound. Consider two common priors on neural network weights:

**Gaussian prior:** P(θ) = N(0, τ²I)
```
ln P(θ) = -1/(2τ²) · ||θ||² + const

θ_MAP = argmax_θ [log-likelihood - λ||θ||²]
```
where λ = 1/(2τ²). This is **L2 regularization** (weight decay)!

**Laplace prior:** P(θ) = Laplace(0, b)
```
ln P(θ) = -(1/b) · ||θ||₁ + const

θ_MAP = argmax_θ [log-likelihood - λ||θ||₁]
```
This is **L1 regularization** (LASSO), which promotes sparsity!

**Summary:**
| What you're doing | Probabilistic interpretation |
|---|---|
| Training with MSE loss | MLE with Gaussian noise model |
| Training with cross-entropy | MLE with Categorical model |
| Adding L2 regularization | MAP with Gaussian prior on weights |
| Adding L1 regularization | MAP with Laplace prior on weights |
| Dropout | Approximate Bayesian inference |

**Regularization is not a hack.** It's a principled Bayesian prior expressing the belief that simpler models (smaller weights) are more likely.

---

## 10. Sampling Methods

### 10.1 Why Sampling?

Many quantities in probabilistic ML involve intractable integrals:
```
E[f(x)] = ∫ f(x) p(x) dx
```

When p(x) is complex (like the posterior in a Bayesian neural network), we can't compute this analytically. Instead, we **approximate** it using samples.

### 10.2 Monte Carlo Estimation

**Core idea:** Approximate expectations with sample averages.

If x₁, ..., xₙ ~ p(x), then:
```
E[f(x)] ≈ (1/n) Σᵢ f(xᵢ)
```

By the law of large numbers, this converges to the true expectation as n → ∞.

**Error:** The approximation error decreases as O(1/√n) regardless of dimensionality. This is why Monte Carlo is powerful in high dimensions — unlike grid-based methods, its convergence rate doesn't suffer from the curse of dimensionality.

**Example: Estimating π**
```
Sample (x, y) uniformly from [-1, 1] × [-1, 1]
If x² + y² ≤ 1, point is inside the unit circle
π ≈ 4 · (points inside) / (total points)
```

### 10.3 Importance Sampling

**Problem:** We want E_p[f(x)] but can't sample from p(x) easily.

**Solution:** Sample from a different distribution q(x) that we *can* sample from:

```
E_p[f(x)] = ∫ f(x) p(x) dx
           = ∫ f(x) · (p(x)/q(x)) · q(x) dx
           = E_q[f(x) · p(x)/q(x)]
           ≈ (1/n) Σᵢ f(xᵢ) · w(xᵢ)

where xᵢ ~ q(x) and w(xᵢ) = p(xᵢ)/q(xᵢ) are importance weights
```

**Key insight:** This is an unbiased estimator for any q with support wherever p·f is nonzero. But variance depends heavily on how well q matches p·f.

**Why this matters for ML:**
- Policy gradient methods in RL use importance sampling to reuse old experience
- Variational inference uses importance sampling to estimate the evidence lower bound (ELBO)
- Off-policy RL is essentially importance sampling from a behavior policy

---

## 11. Why All of This Matters for Machine Learning

Let's now explicitly map every concept in this module to machine learning practice.

### 11.1 Loss Functions Are Negative Log-Likelihoods

| Loss Function | Probabilistic Model |
|---|---|
| Mean Squared Error | Gaussian with fixed variance |
| Cross-Entropy | Categorical (via softmax) |
| Binary Cross-Entropy | Bernoulli (via sigmoid) |
| Huber Loss | Gaussian-Laplace hybrid |

When someone proposes a new loss function, ask: "What distribution does this correspond to?"

### 11.2 Training Is MLE (or MAP)

Minimizing the loss function over training data = maximizing the likelihood of the data under the model:

```
argmin_θ L(θ) = argmin_θ [-Σᵢ ln p(yᵢ|xᵢ, θ)]
              = argmax_θ [Σᵢ ln p(yᵢ|xᵢ, θ)]
              = MLE
```

Add regularization → MAP.

### 11.3 Generative Models Are Probability Distributions

Every generative model defines (or approximates) a probability distribution:

| Model | What it learns |
|---|---|
| VAE | q(z\|x) ≈ p(z\|x), generates via p(x\|z) |
| GAN | Implicit distribution via generator |
| Diffusion | p(x_{t-1}\|x_t) — learned denoising distribution |
| Autoregressive (GPT) | p(x_t\|x_{<t}) — conditional distributions via chain rule |
| Normalizing Flow | Explicit invertible transformation of a simple distribution |

### 11.4 The Bayesian View of Deep Learning

```
Prior:      P(θ) — our belief about weights before training (e.g., Gaussian → weight decay)
Likelihood: P(D|θ) — how well the network fits the data
Posterior:  P(θ|D) — what we should believe about weights after training
Prediction: P(y*|x*, D) = ∫ P(y*|x*, θ) P(θ|D) dθ — average over all plausible weights
```

Full Bayesian inference averages over the posterior instead of committing to a single point estimate. This gives:
- Better calibrated uncertainty estimates
- Automatic Occam's razor (complex models are penalized)
- No need for separate regularization

The integral is intractable for neural networks, which is why we use approximations: Monte Carlo dropout, variational inference, Laplace approximation, or ensembles.

### 11.5 Information Theory Connection

Cross-entropy loss = H(p, q) = H(p) + D_KL(p || q)

Minimizing cross-entropy loss = minimizing KL divergence between the true distribution p and the model distribution q (since H(p) is constant). This connects directly to Module 04 (Information Theory).

---

## Summary: The Probability Toolkit for ML

```
Probability Axioms ─── foundation of reasoning under uncertainty
       │
Conditional Probability ─── chain rule → autoregressive models
       │
Bayes' Theorem ─── posterior ∝ likelihood × prior → learning
       │
Random Variables ─── discrete (classification) + continuous (regression)
       │
├── Bernoulli/Categorical → sigmoid/softmax outputs
├── Gaussian → noise models, latent spaces, weight initialization
└── Mixtures → complex multi-modal distributions
       │
Expectation/Variance ─── loss = expected negative log-likelihood
       │
MLE ─── training neural networks by minimizing loss
       │
MAP ─── MLE + regularization (L1, L2, dropout)
       │
Sampling ─── Monte Carlo, importance sampling → Bayesian inference, RL
       │
Central Limit Theorem ─── why Gaussian assumptions are often reasonable
```

Every equation in this module will reappear, in disguise, throughout the rest of this project.

---

## References

- Kolmogorov, A.N. *Foundations of the Theory of Probability* (1933)
- Jaynes, E.T. *Probability Theory: The Logic of Science* (2003) — the Bayesian bible
- Bishop, C.M. *Pattern Recognition and Machine Learning* (2006) — Ch. 1-2
- Murphy, K.P. *Machine Learning: A Probabilistic Perspective* (2012) — Ch. 2-5
- Blei, D.M., Kucukelbir, A., McAuliffe, J.D. "Variational Inference: A Review for Statisticians" (2017)

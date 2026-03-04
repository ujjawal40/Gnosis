# Information Theory

*"The fundamental problem of communication is that of reproducing at one point either exactly or approximately a message selected at another point."* -- Claude Shannon, 1948

Shannon didn't just solve communication. He created a mathematical framework for **surprise**, **uncertainty**, and **knowledge** -- the same framework that underpins every loss function, every generative model, and every representation we train.

---

## 1. The Core Insight: Information Is Surprise

Before Shannon, "information" was a vague word. Shannon made it precise:

**Information is the reduction of uncertainty.**

If I tell you "the sun rose today," you learn almost nothing -- you already expected it. If I tell you "a meteor struck downtown," you learn a lot. The first message carries low information; the second carries high information.

Shannon formalized this: the **information content** (or **self-information**) of an event with probability p(x) is:

```
I(x) = -log_2(p(x))    [measured in bits]
```

Why logarithm? Three axioms force this choice:

1. **Monotonicity**: Less probable events carry more information. If p(x) < p(y), then I(x) > I(y).
2. **Zero information for certainty**: If p(x) = 1, then I(x) = 0. A guaranteed event tells you nothing.
3. **Additivity**: Information from independent events adds. I(x, y) = I(x) + I(y) for independent x, y.

The ONLY function satisfying all three is I(x) = -log p(x). This isn't a convention -- it's a theorem.

### Why bits?

With log base 2, information is measured in **bits** -- the number of yes/no questions needed to identify the outcome.

- Fair coin flip: I(heads) = -log_2(1/2) = 1 bit (one yes/no question)
- Fair die roll: I(3) = -log_2(1/6) ~= 2.585 bits
- Rare event (p = 1/1024): I(x) = -log_2(1/1024) = 10 bits

---

## 2. Entropy: The Expected Surprise

### Definition

**Entropy** is the expected information content -- the average surprise:

```
H(X) = -sum_x p(x) log_2 p(x) = E_p[I(X)]
```

This is the most important quantity in information theory.

### Derivation from Axioms

Shannon derived entropy from three requirements for any measure of uncertainty H(p_1, ..., p_n):

**Axiom 1 -- Continuity**: H is a continuous function of the probabilities.

**Axiom 2 -- Maximum at uniformity**: For a uniform distribution over n outcomes, H(1/n, ..., 1/n) is maximized and increases with n. More equally-likely outcomes means more uncertainty.

**Axiom 3 -- Composition (Chain Rule)**: If a choice is broken into successive sub-choices, H decomposes consistently. Specifically, if we group outcomes and choose in stages:

```
H(p_1, ..., p_n) = H(q_1, ..., q_m) + sum_j q_j * H(p_{j,1}/q_j, ..., p_{j,k_j}/q_j)
```

where q_j = sum of probabilities in group j.

**Theorem (Shannon, 1948)**: The ONLY function satisfying all three axioms is:

```
H(X) = -C * sum_x p(x) log p(x)
```

where C > 0 is a constant that determines the unit (C = 1/ln(2) for bits).

### Properties of Entropy

1. **Non-negativity**: H(X) >= 0, with equality iff X is deterministic.
2. **Maximum**: H(X) <= log_2(|X|), with equality iff X is uniform.
3. **Concavity**: H is a concave function of the probability distribution.
4. **Chain rule**: H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y).

### Meaning

Entropy measures:
- **Uncertainty**: How unpredictable is X?
- **Information content**: How much do we learn on average by observing X?
- **Compression limit**: The minimum average bits per symbol to encode X (Shannon's source coding theorem).
- **Randomness**: How "spread out" is the distribution?

### Examples

| Distribution | Entropy | Intuition |
|---|---|---|
| Deterministic: p = (1, 0, 0) | 0 bits | No uncertainty at all |
| Fair coin: p = (0.5, 0.5) | 1 bit | One yes/no question |
| Biased coin: p = (0.9, 0.1) | 0.469 bits | Mostly predictable |
| Fair die: p = (1/6, ..., 1/6) | 2.585 bits | More outcomes, more uncertainty |
| Fair byte: p = (1/256, ..., 1/256) | 8 bits | Exactly one byte of randomness |

### Conditional Entropy

The entropy of X given knowledge of Y:

```
H(X|Y) = -sum_{x,y} p(x,y) log p(x|y) = H(X,Y) - H(Y)
```

Key property: **conditioning reduces entropy** (on average):

```
H(X|Y) <= H(X)
```

with equality iff X and Y are independent. Knowing something about Y can never increase your uncertainty about X on average.

---

## 3. Cross-Entropy: Measuring Model Quality

### Definition

The **cross-entropy** between a true distribution p and a model distribution q is:

```
H(p, q) = -sum_x p(x) log q(x) = E_p[-log q(x)]
```

### Meaning

Imagine you need to encode messages drawn from distribution p, but you designed your code based on distribution q. Cross-entropy H(p, q) is the **average number of bits you'll actually need** per message.

- If q = p (perfect model): H(p, q) = H(p). You use the optimal number of bits.
- If q != p (imperfect model): H(p, q) > H(p). You waste bits because your model is wrong.

The "wasted bits" are precisely the KL divergence (next section):

```
H(p, q) = H(p) + D_KL(p || q)
```

### Why Cross-Entropy Is THE Loss Function for Classification

In classification, we have:
- True distribution p: one-hot vector (all mass on correct class)
- Model distribution q: softmax output (predicted probabilities)

For a one-hot p with the true class being k:

```
H(p, q) = -sum_i p(i) log q(i) = -log q(k)
```

This is just **negative log-likelihood** of the correct class! Minimizing cross-entropy loss = maximizing the likelihood the model assigns to the correct answers.

This isn't a coincidence or a design choice. Cross-entropy loss comes directly from asking: "How many bits does our model waste?"

---

## 4. KL Divergence: The Distance Between Distributions

### Definition

The **Kullback-Leibler divergence** from distribution q to distribution p is:

```
D_KL(p || q) = sum_x p(x) log(p(x) / q(x)) = E_p[log(p(x)/q(x))]
```

Equivalently:

```
D_KL(p || q) = H(p, q) - H(p)
```

KL divergence measures the **extra bits** needed when using q instead of p.

### Properties

1. **Non-negativity** (Gibbs' inequality): D_KL(p || q) >= 0, with equality iff p = q.
2. **NOT a distance**: It violates symmetry and the triangle inequality.
3. **Asymmetry**: D_KL(p || q) != D_KL(q || p) in general. This matters enormously.
4. **Undefined**: When q(x) = 0 but p(x) > 0, D_KL = infinity. The model must assign nonzero probability everywhere the truth has nonzero probability.

### The Asymmetry Matters

**Forward KL: D_KL(p || q)** -- "moment matching" / "mean-seeking"
- Penalizes q for assigning low probability where p is high.
- Forces q to cover all modes of p.
- Result: q tends to be **overdispersed** (too broad).
- Used in: Variational inference (ELBO maximization uses reverse KL, but forward KL appears in expectation propagation).

**Reverse KL: D_KL(q || p)** -- "mode-seeking"
- Penalizes q for assigning high probability where p is low.
- Allows q to ignore modes of p.
- Result: q tends to be **underdispersed** (too narrow, locks onto one mode).
- Used in: Variational autoencoders, policy optimization.

This asymmetry explains why VAEs produce blurry images (they use reverse KL, which mode-seeks) while GANs can produce sharp but sometimes mode-collapsed images.

### Proof of Non-negativity (Gibbs' Inequality)

Using Jensen's inequality with the convex function -log:

```
D_KL(p || q) = -sum_x p(x) log(q(x)/p(x))
             >= -log(sum_x p(x) * q(x)/p(x))    [Jensen's inequality]
             = -log(sum_x q(x))
             = -log(1) = 0
```

Equality holds iff q(x)/p(x) is constant, i.e., p = q.

---

## 5. Mutual Information: Shared Knowledge

### Definition

The **mutual information** between X and Y measures how much knowing one tells you about the other:

```
I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)
```

Equivalently:

```
I(X; Y) = D_KL(p(x,y) || p(x)p(y))
```

Mutual information is the KL divergence between the joint distribution and the product of marginals. It measures how far X and Y are from being independent.

### Properties

1. **Non-negativity**: I(X; Y) >= 0, with equality iff X and Y are independent.
2. **Symmetry**: I(X; Y) = I(Y; X). Unlike KL divergence, mutual information IS symmetric.
3. **Bounded**: I(X; Y) <= min(H(X), H(Y)). You can't share more information than either variable contains.
4. **Self-information**: I(X; X) = H(X). A variable shares all its information with itself.
5. **Chain rule**: I(X; Y, Z) = I(X; Y) + I(X; Z|Y).

### The Information Venn Diagram

```
    ┌──────────────────────────────────────┐
    │             H(X, Y)                  │
    │  ┌──────────────┬───────────────┐    │
    │  │    H(X|Y)    │    H(Y|X)    │    │
    │  │              │              │    │
    │  │  Unique to X │ Unique to Y  │    │
    │  │              │              │    │
    │  │       ┌──────┴──────┐       │    │
    │  │       │   I(X;Y)   │       │    │
    │  │       │   Shared   │       │    │
    │  │       └──────┬──────┘       │    │
    │  │              │              │    │
    │  └──────────────┴───────────────┘    │
    └──────────────────────────────────────┘

    H(X) = H(X|Y) + I(X;Y)
    H(Y) = H(Y|X) + I(X;Y)
    H(X,Y) = H(X|Y) + H(Y|X) + I(X;Y)
```

### Why Mutual Information Matters in ML

- **Feature selection**: Features with high I(feature; target) are informative.
- **Representation learning**: Good representations maximize I(representation; task).
- **Independence testing**: I(X; Y) = 0 iff X and Y are independent.
- **Information bottleneck**: Compress X while preserving I(compressed; Y).

---

## 6. The Information Bottleneck

### Tishby's Principle

Naftali Tishby (2015) proposed that deep learning works by finding an optimal **information bottleneck**: each layer compresses the input while preserving information about the output.

Given input X and target Y, find a compressed representation T that solves:

```
min_{p(t|x)} I(X; T) - beta * I(T; Y)
```

- **I(X; T)**: How much of X does T remember? (compression term)
- **I(T; Y)**: How much does T know about Y? (prediction term)
- **beta**: Controls the tradeoff. Large beta favors prediction; small beta favors compression.

### The Two Phases of Training

Tishby's controversial (and debated) hypothesis about deep learning:

**Phase 1 -- Fitting**: The network increases I(T; Y) rapidly. It learns to predict.

**Phase 2 -- Compression**: The network decreases I(X; T) while maintaining I(T; Y). It forgets irrelevant details.

The claim is that generalization comes from this compression phase: by discarding irrelevant information about X, the network becomes robust to noise and overfitting.

### The Information Plane

Plot each layer's representation as a point (I(X; T_i), I(T_i; Y)):

```
I(T; Y)
  ^
  |        * Layer 5 (near target)
  |      *  Layer 4
  |    *   Layer 3
  |   *   Layer 2
  |  *   Layer 1 (near input)
  |
  └─────────────────> I(X; T)
```

Good training should move points **up and left**: more information about Y, less about X.

### Caveats

The information bottleneck theory is not universally accepted:
- The compression phase may depend on activation functions (observed with tanh, less clear with ReLU).
- Estimating mutual information in high dimensions is extremely difficult.
- The connection to generalization is still debated.

But the core idea -- that good representations compress irrelevant information -- is powerful and widely influential.

---

## 7. Connections to Machine Learning

### 7.1 Cross-Entropy Loss IS Information Theory

The standard classification loss:

```
L = -sum_{i=1}^{N} log q(y_i | x_i)
```

This is exactly the cross-entropy H(p_data, q_model) estimated over the training set. Minimizing this loss:

1. Minimizes the KL divergence D_KL(p_data || q_model) (since H(p_data) is constant).
2. Maximizes the likelihood of the data under the model.
3. Finds the model that wastes the fewest bits encoding the true labels.

All three perspectives are equivalent. Information theory unifies them.

### 7.2 KL Divergence in VAEs

The VAE loss (ELBO) has two terms:

```
L_VAE = E_q[-log p(x|z)] + D_KL(q(z|x) || p(z))
```

- **Reconstruction term**: Cross-entropy between data and reconstruction.
- **KL term**: Forces the encoder distribution q(z|x) to stay close to the prior p(z).

The KL term is pure information theory: it measures how many extra bits the encoder uses compared to the prior. It acts as a regularizer, preventing the latent space from becoming too complex.

### 7.3 Mutual Information in Representation Learning

**InfoNCE loss** (used in contrastive learning like SimCLR, CPC):

```
L_InfoNCE = -E[log(exp(f(x,y)) / sum_j exp(f(x,y_j)))]
```

This is a lower bound on mutual information I(X; Y). Minimizing InfoNCE maximizes a bound on the mutual information between different views of the same data.

**Deep InfoMax (DIM)**: Directly maximizes mutual information between input and representation.

**MINE (Mutual Information Neural Estimation)**: Uses neural networks to estimate mutual information through the Donsker-Varadhan representation.

### 7.4 Entropy as Uncertainty Quantification

- **Predictive entropy**: H(Y|x) = -sum_y p(y|x) log p(y|x) measures model uncertainty.
- High entropy = model is unsure (useful for active learning, out-of-distribution detection).
- **Maximum entropy principle**: When you have incomplete information, the distribution with maximum entropy (subject to known constraints) is the least biased choice. This justifies:
  - Gaussian distributions when you only know mean and variance.
  - Uniform distributions when you know nothing.
  - Softmax as the maximum entropy classifier.

---

## 8. Data Processing Inequality

### Statement

If X -> Y -> Z forms a Markov chain (Z depends on X only through Y), then:

```
I(X; Z) <= I(X; Y)
```

**You cannot create information by processing.** Any function of Y can only preserve or lose information about X, never gain it.

### Implications

1. **Feature engineering can only lose information**: Any transformation of your raw data can only reduce the information available about the target. Choose transformations that lose as little as possible.

2. **Deep networks form Markov chains**: X -> h_1 -> h_2 -> ... -> h_L -> Y. Each layer can only preserve or lose information from the input. The network must learn which information to keep and which to discard.

3. **Compression is irreversible**: Once information is lost, no amount of processing can recover it. This is why lossy compression has fundamental limits.

4. **Sufficient statistics**: A statistic T(X) is sufficient for parameter theta if I(T(X); theta) = I(X; theta). Sufficient statistics lose NO information about the parameter.

### Why This Matters for ML

The data processing inequality tells us that the best possible classifier from features can never beat the best possible classifier from raw data. If your feature extraction pipeline loses information about the target, no model can recover it.

This doesn't mean raw data is always better in practice (features reduce complexity, overfitting risk, computation), but it sets a fundamental limit.

---

## 9. Rate-Distortion Theory

### The Problem

How much do you have to compress to achieve a given level of accuracy?

Rate-distortion theory formalizes the tradeoff between:
- **Rate R**: Bits per symbol used for encoding (compression level).
- **Distortion D**: Average error in reconstruction (accuracy loss).

### The Rate-Distortion Function

For a source X and distortion measure d(x, x_hat):

```
R(D) = min_{p(x_hat|x): E[d(x,x_hat)] <= D} I(X; X_hat)
```

This gives the **minimum rate** (bits) needed to achieve distortion at most D. It's a fundamental limit -- no compression scheme can do better.

### Properties

1. R(D) is a convex, non-increasing function of D.
2. R(0) = H(X) for discrete sources with Hamming distortion. Lossless compression requires at least H(X) bits.
3. R(D_max) = 0. If you tolerate maximum distortion, you need zero bits (just output a constant).

### Connection to ML

Rate-distortion theory is essentially the information bottleneck:
- **Rate = I(X; T)**: How much the representation remembers about the input.
- **Distortion = expected loss**: How well the representation predicts the target.

Modern connections:
- **Neural compression**: Learned image/video codecs approach rate-distortion bounds.
- **Model compression**: Pruning and quantization trade model size (rate) for accuracy (distortion).
- **Variational inference**: The ELBO involves a rate-distortion tradeoff -- the KL term is the rate, and the reconstruction term is the distortion.
- **Bits-back coding**: VAEs can be used for lossless compression, achieving rates near the ELBO.

### The Rate-Distortion Curve

```
Rate R (bits)
  ^
  |
  |*
  | *
  |  *
  |   *
  |     *
  |        *
  |            *
  |                  *
  └──────────────────────> Distortion D
  0                      D_max

  - Steep region: Small distortion reduction costs many bits
  - Flat region: Large distortion is cheap to encode
  - The curve is the fundamental limit of compression
```

---

## 10. Summary: The Information-Theoretic View of ML

| ML Concept | Information Theory View |
|---|---|
| Training | Minimizing cross-entropy H(p_data, q_model) |
| Cross-entropy loss | Extra bits from using q instead of p, plus entropy of p |
| Overfitting | Model memorizes noise: I(model; noise) > 0 |
| Generalization | Model captures signal: high I(model; signal), low I(model; noise) |
| Good representations | High I(repr; target), low I(repr; input) |
| Data augmentation | Reduces I(repr; nuisance variables) |
| Dropout / regularization | Limits mutual information I(weights; training data) |
| Model compression | Rate-distortion tradeoff on model parameters |
| Feature extraction | Data processing inequality sets limits |
| VAE latent space | Rate-distortion tradeoff via KL term |

The deepest insight: **learning is compression**. A model that truly understands its data has found the shortest description of the patterns in that data. This is Occam's razor, formalized in bits.

---

## Key Equations Reference

```
Self-information:       I(x) = -log p(x)
Entropy:                H(X) = -sum_x p(x) log p(x)
Joint entropy:          H(X,Y) = -sum_{x,y} p(x,y) log p(x,y)
Conditional entropy:    H(X|Y) = H(X,Y) - H(Y)
Cross-entropy:          H(p,q) = -sum_x p(x) log q(x)
KL divergence:          D_KL(p||q) = sum_x p(x) log(p(x)/q(x))
Mutual information:     I(X;Y) = H(X) + H(Y) - H(X,Y)
                               = D_KL(p(x,y) || p(x)p(y))
Chain rule (entropy):   H(X_1,...,X_n) = sum_i H(X_i | X_1,...,X_{i-1})
Chain rule (MI):        I(X; Y,Z) = I(X;Y) + I(X;Z|Y)
Data processing:        X -> Y -> Z  =>  I(X;Z) <= I(X;Y)
Rate-distortion:        R(D) = min I(X; X_hat) s.t. E[d(X,X_hat)] <= D
```

---

## References

- Shannon, C. E. (1948). *A Mathematical Theory of Communication*. Bell System Technical Journal.
- Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley.
- MacKay, D. J. C. (2003). *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press.
- Tishby, N., & Zaslavsky, N. (2015). *Deep Learning and the Information Bottleneck Principle*. IEEE Information Theory Workshop.
- Shwartz-Ziv, R., & Tishby, N. (2017). *Opening the Black Box of Deep Neural Networks via Information*. arXiv:1703.00810.
- Poole, B., Ozair, S., Van Den Oord, A., Alemi, A., & Tucker, G. (2019). *On Variational Bounds of Mutual Information*. ICML.

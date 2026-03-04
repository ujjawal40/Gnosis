# Word2Vec — Learning Meaning from Context

## The Distributional Hypothesis

*"You shall know a word by the company it keeps."* — J.R. Firth (1957)

Words that appear in similar contexts have similar meanings. Word2Vec turns this linguistic insight into a learning algorithm.

---

## 1. The Problem

How do you represent a word for a neural network?

**One-hot encoding:** "cat" = [0, 0, 1, 0, ..., 0] (vocabulary-length vector)
- No notion of similarity: cos("cat", "dog") = 0
- Dimensionality = vocabulary size (huge)

**Word2Vec solution:** Learn a dense vector where similar words are close.
- "cat" = [0.2, -0.5, 0.8, ...] (300 dimensions)
- cos("cat", "dog") ≈ 0.8

---

## 2. Two Architectures

### Skip-Gram
Given a center word, predict context words.
```
"The [cat] sat on the mat"
center: "cat"
context: {"The", "sat"}  (window size = 1)

Objective: maximize P("The"|"cat") · P("sat"|"cat")
```

### CBOW (Continuous Bag of Words)
Given context words, predict the center word.
```
context: {"The", "sat"} → predict "cat"
```

Skip-gram works better for rare words, CBOW is faster. We implement skip-gram.

---

## 3. The Mathematics

For center word c and context word o:

```
P(o|c) = exp(v_o · v_c) / Σ_w exp(v_w · v_c)
```

This is softmax over the entire vocabulary — expensive! (O(|V|) per word pair)

### Negative Sampling

Instead of computing over all words, sample k "negative" (random) words:

```
L = log σ(v_o · v_c) + Σ_{i=1}^{k} E[log σ(-v_ni · v_c)]
```

where σ is sigmoid, and n_i are random "negative" words.

Intuition: push the center word close to its real context, far from random words.

---

## 4. Why It Works: The Geometry of Meaning

The learned vectors capture semantic relationships as **linear directions**:
```
v("king") - v("man") + v("woman") ≈ v("queen")
v("Paris") - v("France") + v("Germany") ≈ v("Berlin")
```

This emerges automatically from the training objective. The reason:
- "king" and "queen" appear in similar contexts (royalty, power, reign)
- "king" and "man" share "male" contexts but differ on "royalty"
- The difference vector captures the "royalty" concept

---

## 5. Connection to Matrix Factorization

Levy & Goldberg (2014) proved: Word2Vec with skip-gram + negative sampling is implicitly factorizing the word-context PMI (Pointwise Mutual Information) matrix.

```
W^T · C ≈ PMI(w, c) - log(k)
```

where PMI(w,c) = log(P(w,c) / P(w)P(c))

This connects neural embeddings to classical distributional semantics.

---

## References
- Mikolov et al. (2013) "Efficient Estimation of Word Representations in Vector Space"
- Mikolov et al. (2013) "Distributed Representations of Words and Phrases"
- Levy & Goldberg (2014) "Neural Word Embedding as Implicit Matrix Factorization"

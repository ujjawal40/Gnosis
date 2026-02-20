# Attention — The Mechanism That Changed Everything

## The Problem Attention Solves

RNNs compress an entire sequence into a single fixed-size vector:
```
"The cat sat on the mat" → h_final = [0.2, -0.5, ...] → decoder
```

This bottleneck means long sequences lose early information. Attention lets the decoder **look back at every position** and decide what's relevant.

---

## 1. Bahdanau Attention (2015)

The original attention for machine translation:

```
score(h_dec, h_enc_j) = v^T · tanh(W_1 · h_dec + W_2 · h_enc_j)
α_j = softmax(score_j)                    # Attention weights
context = Σ_j α_j · h_enc_j              # Weighted sum of encoder states
```

**Intuition:** At each decoder step, compute a "relevance score" for each encoder position, then take a weighted average. The model learns what to attend to.

---

## 2. Scaled Dot-Product Attention (Transformer)

Simplified and made more efficient:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

Where:
- **Q** (Query): "What am I looking for?"
- **K** (Key): "What do I contain?"
- **V** (Value): "What information do I provide?"

The dot product QK^T measures compatibility. Division by √d_k prevents softmax saturation for large d.

### Why Q, K, V?

Think of a dictionary lookup:
- Query: your search term
- Keys: index entries
- Values: the content

Attention computes a "soft lookup" — instead of exact match, it returns a weighted combination of all values based on query-key similarity.

---

## 3. Multi-Head Attention

Instead of one attention function, use multiple in parallel:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Why multiple heads?** Each head can attend to different things:
- Head 1: syntactic relationships ("what is the subject?")
- Head 2: semantic similarity ("what words mean similar things?")
- Head 3: positional patterns ("what's nearby?")

---

## 4. Self-Attention

When Q, K, V all come from the same sequence, it's **self-attention**:

```
Each token attends to every other token in the same sequence.
```

This is the core of the Transformer. Every token can directly interact with every other token — no sequential bottleneck.

### Complexity

Self-attention is O(n²·d) where n = sequence length, d = dimension.
- This is its strength (global interactions) and weakness (quadratic scaling)
- For n = 4096 tokens: 16M attention scores per layer per head

---

## 5. Causal (Masked) Attention

For autoregressive generation (GPT), tokens can only attend to **previous** tokens:

```
Mask = upper triangular matrix of -∞
Attention(Q, K, V) = softmax(QK^T / √d_k + Mask) · V
```

The -∞ values become 0 after softmax, preventing information leakage from the future.

---

## The Paradigm Shift

Before attention: information flows through a bottleneck (hidden state).
After attention: any position can directly access any other position.

This eliminates the fundamental limitation of RNNs and enables parallelization across positions (no sequential dependency in the forward pass).

---

## References
- Bahdanau et al. (2015) "Neural Machine Translation by Jointly Learning to Align and Translate"
- Vaswani et al. (2017) "Attention Is All You Need"

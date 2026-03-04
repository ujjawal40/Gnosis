# The Transformer — Attention Is All You Need

## Why The Transformer Matters

The Transformer (Vaswani et al., 2017) replaced RNNs with pure attention. It powers GPT, BERT, LLaMA, and virtually every modern language model. Understanding it deeply is non-negotiable.

---

## Architecture

```
Input Tokens
     │
[Token Embedding + Positional Encoding]
     │
     ▼
┌─────────────────────────┐
│  Multi-Head Attention    │◀── residual + layer norm
│         ↓                │
│  Feed-Forward Network    │◀── residual + layer norm
└─────────────────────────┘
     │ × N layers
     ▼
Output Logits
```

### Key Components:

1. **Token Embedding:** Maps discrete tokens to dense vectors
2. **Positional Encoding:** Injects position information (attention has no notion of order)
3. **Multi-Head Self-Attention:** Every token attends to every other token
4. **Feed-Forward Network (FFN):** Two-layer MLP applied to each position independently
5. **Residual Connections:** x + sublayer(x) — enables gradient flow in deep networks
6. **Layer Normalization:** Stabilizes activations

---

## Positional Encoding

Attention treats input as a set (order-invariant). We must inject position:

**Sinusoidal (original):**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Why sinusoids?**
- Different frequencies capture different scales of position
- Can extrapolate to longer sequences than seen during training
- PE(pos+k) is a linear function of PE(pos) — relative positions are encodable

**Learned (modern):** Just make PE a trainable parameter matrix. Simpler, often works as well.

---

## The Feed-Forward Network

```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

Applied identically to each position. The inner dimension is typically 4× the model dimension.

**What it does:** The FFN acts as a "key-value memory" that stores factual knowledge. Each neuron in W_1 matches a pattern, and W_2 produces the output.

---

## Residual Connections + Layer Norm

```
output = LayerNorm(x + Sublayer(x))
```

**Residual connections:** Without them, gradients vanish in deep transformers. The addition creates a direct path for gradients.

**Layer normalization:** Normalizes across features (not batch). More stable than batch norm for variable-length sequences.

**Pre-norm vs Post-norm:**
- Original (post-norm): LayerNorm(x + Sublayer(x))
- Modern (pre-norm): x + Sublayer(LayerNorm(x)) — more stable for very deep models

---

## Encoder vs Decoder

**Encoder (BERT-style):** Bidirectional self-attention. Each token sees all other tokens.
**Decoder (GPT-style):** Causal self-attention. Each token sees only previous tokens.
**Encoder-Decoder (Original):** Encoder processes input, decoder attends to encoder output via cross-attention.

---

## Why Transformers Work

1. **Parallelizable:** No sequential dependency in forward pass (unlike RNNs)
2. **Direct long-range connections:** Token 1 can directly attend to token 1000
3. **Expressive:** Multi-head attention + FFN is a universal approximator
4. **Scalable:** Performance improves predictably with more compute/data/parameters

---

## Limitations

1. **O(n²) complexity:** Quadratic in sequence length
2. **No built-in inductive bias:** Needs massive data (unlike CNNs with translation equivariance)
3. **Position information is added, not intrinsic:** Positional encoding is a workaround
4. **Interpretability:** Despite attention weights, reasoning is opaque

---

## References
- Vaswani et al. (2017) "Attention Is All You Need"
- Devlin et al. (2019) "BERT"
- Radford et al. (2018, 2019) "GPT, GPT-2"

# Recurrent Neural Networks & LSTM

## The Problem: Sequences Have Memory

MLPs and CNNs process fixed-size inputs. But language, music, time series, and video are **sequences** — the meaning of each element depends on what came before.

"The dog that chased the cat sat on the mat" — to know what "sat" refers to, you need to remember "dog" from 8 words ago.

---

## 1. The Vanilla RNN

### Architecture

At each timestep t, the RNN takes an input x_t and the previous hidden state h_{t-1}:

```
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)
y_t = W_hy · h_t + b_y
```

The hidden state h_t is the "memory" — it carries information from all previous timesteps.

**Parameter sharing:** The same weights (W_hh, W_xh, W_hy) are used at every timestep. This is the temporal equivalent of CNN's spatial parameter sharing.

### Unrolling Through Time

```
x_0    x_1    x_2    x_3
 │      │      │      │
 ▼      ▼      ▼      ▼
[RNN]→[RNN]→[RNN]→[RNN]→ h_3
 │      │      │      │
 ▼      ▼      ▼      ▼
y_0    y_1    y_2    y_3
```

When unrolled, an RNN is just a very deep feedforward network with shared weights.

### Backpropagation Through Time (BPTT)

Backprop through the unrolled network. The gradient flows backward through each timestep:

```
dL/dW_hh = Σ_t dL/dh_t · ∂h_t/∂W_hh
```

But ∂h_t/∂h_{t-k} involves multiplying W_hh k times. This is where the problem starts.

---

## 2. The Vanishing Gradient Problem

When backpropagating through T timesteps:

```
∂h_T/∂h_0 = Π_{t=1}^{T} ∂h_t/∂h_{t-1} = Π_{t=1}^{T} W_hh · diag(tanh'(z_t))
```

Since |tanh'(x)| ≤ 1, this product shrinks exponentially with T.

**Result:** The network can't learn long-range dependencies. It "forgets" anything more than ~10-20 steps back.

This is not just a practical nuisance — it's a fundamental limitation of vanilla RNNs.

---

## 3. LSTM: The Gating Solution

### Key Insight

Instead of one hidden state updated each step, use **gates** to control information flow:
- **Forget gate:** What to forget from memory
- **Input gate:** What new information to store
- **Output gate:** What to output from memory

### The LSTM Equations

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)        # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)        # Input gate
g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)     # Candidate memory
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t           # Cell state update
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)        # Output gate
h_t = o_t ⊙ tanh(c_t)                       # Hidden state
```

where σ is sigmoid, ⊙ is element-wise multiply.

### Why This Solves Vanishing Gradients

The cell state c_t has a **direct additive path** through time:
```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
```

If f_t ≈ 1 (forget gate open), then c_t ≈ c_{t-1} + stuff. The gradient flows through the addition without multiplication by W. This is the same principle as ResNet's skip connections.

### GRU: Simplified LSTM

The Gated Recurrent Unit combines forget and input gates:
```
z_t = σ(W_z · [h_{t-1}, x_t])              # Update gate
r_t = σ(W_r · [h_{t-1}, x_t])              # Reset gate
h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])     # Candidate
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t   # Update
```

Fewer parameters, often comparable performance to LSTM.

---

## 4. Limitations of RNNs/LSTMs

Even with gating, RNNs have fundamental issues:
1. **Sequential processing:** Can't parallelize across timesteps (GPUs are underutilized)
2. **Fixed-size bottleneck:** All information must flow through the hidden state vector
3. **Still struggle with very long sequences** (>1000 tokens)

These limitations motivated the **attention mechanism** and **transformers** (Module 04).

---

## References

- Elman (1990) "Finding Structure in Time"
- Hochreiter & Schmidhuber (1997) "Long Short-Term Memory"
- Cho et al. (2014) "Learning Phrase Representations using RNN Encoder-Decoder"
- Pascanu et al. (2013) "On the Difficulty of Training Recurrent Neural Networks"

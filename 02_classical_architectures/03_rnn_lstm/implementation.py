"""
RNN and LSTM from Scratch
==========================

Complete implementation of vanilla RNN and LSTM using only NumPy.
Includes training on sequence tasks to demonstrate the vanishing gradient
problem and how LSTM solves it.
"""

import numpy as np


# ==============================================================================
# Part 1: Vanilla RNN
# ==============================================================================

class VanillaRNN:
    """
    Vanilla RNN: h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)

    Simple but suffers from vanishing gradients on long sequences.
    """

    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        scale_h = np.sqrt(1.0 / hidden_size)
        scale_x = np.sqrt(1.0 / input_size)

        self.W_xh = np.random.randn(input_size, hidden_size) * scale_x
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale_h
        self.b_h = np.zeros((1, hidden_size))
        self.W_hy = np.random.randn(hidden_size, output_size) * scale_h
        self.b_y = np.zeros((1, output_size))

    def forward(self, inputs, h0=None):
        """
        Forward pass through sequence.
        inputs: list of (batch, input_size) arrays, one per timestep
        Returns: outputs (list), hidden states (list)
        """
        batch_size = inputs[0].shape[0]
        if h0 is None:
            h = np.zeros((batch_size, self.hidden_size))
        else:
            h = h0

        self.inputs = inputs
        self.hiddens = [h]
        outputs = []

        for t in range(len(inputs)):
            z = inputs[t] @ self.W_xh + h @ self.W_hh + self.b_h
            h = np.tanh(z)
            self.hiddens.append(h)
            y = h @ self.W_hy + self.b_y
            outputs.append(y)

        return outputs, self.hiddens

    def backward(self, d_outputs, lr=0.01):
        """
        Backpropagation Through Time (BPTT).
        d_outputs: list of gradients from loss for each timestep.
        """
        T = len(d_outputs)
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)
        dW_hy = np.zeros_like(self.W_hy)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros_like(self.hiddens[0])

        for t in reversed(range(T)):
            dy = d_outputs[t]
            h = self.hiddens[t + 1]
            h_prev = self.hiddens[t]

            # Output layer gradients
            dW_hy += h.T @ dy
            db_y += np.sum(dy, axis=0, keepdims=True)

            # Hidden state gradient (from output + from future timestep)
            dh = dy @ self.W_hy.T + dh_next

            # Through tanh: d/dz tanh(z) = 1 - tanh²(z)
            dz = dh * (1 - h ** 2)

            # Parameter gradients
            dW_xh += self.inputs[t].T @ dz
            dW_hh += h_prev.T @ dz
            db_h += np.sum(dz, axis=0, keepdims=True)

            # Gradient to pass to previous timestep
            dh_next = dz @ self.W_hh.T

        # Gradient clipping to prevent exploding gradients
        for grad in [dW_xh, dW_hh, db_h, dW_hy, db_y]:
            np.clip(grad, -5, 5, out=grad)

        n = d_outputs[0].shape[0]
        self.W_xh -= lr * dW_xh / n
        self.W_hh -= lr * dW_hh / n
        self.b_h -= lr * db_h / n
        self.W_hy -= lr * dW_hy / n
        self.b_y -= lr * db_y / n


# ==============================================================================
# Part 2: LSTM
# ==============================================================================

class LSTM:
    """
    Long Short-Term Memory network.

    Four gates control information flow:
    - Forget gate (f): what to erase from cell state
    - Input gate (i): what new info to write
    - Gate gate (g): candidate values to write
    - Output gate (o): what to read from cell state
    """

    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        combined = input_size + hidden_size
        scale = np.sqrt(1.0 / combined)

        # Combined weight matrices for efficiency: [x_t, h_{t-1}] -> gates
        self.W_f = np.random.randn(combined, hidden_size) * scale
        self.b_f = np.ones((1, hidden_size))  # Bias forget gate to 1 (remember by default)
        self.W_i = np.random.randn(combined, hidden_size) * scale
        self.b_i = np.zeros((1, hidden_size))
        self.W_g = np.random.randn(combined, hidden_size) * scale
        self.b_g = np.zeros((1, hidden_size))
        self.W_o = np.random.randn(combined, hidden_size) * scale
        self.b_o = np.zeros((1, hidden_size))

        # Output projection
        self.W_y = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        self.b_y = np.zeros((1, output_size))

    def forward(self, inputs, h0=None, c0=None):
        batch_size = inputs[0].shape[0]
        if h0 is None:
            h = np.zeros((batch_size, self.hidden_size))
        else:
            h = h0
        if c0 is None:
            c = np.zeros((batch_size, self.hidden_size))
        else:
            c = c0

        self.inputs = inputs
        self.cache = []
        outputs = []

        for t in range(len(inputs)):
            # Concatenate input and hidden state
            combined = np.hstack([inputs[t], h])

            # Gates
            f = self._sigmoid(combined @ self.W_f + self.b_f)  # Forget
            i = self._sigmoid(combined @ self.W_i + self.b_i)  # Input
            g = np.tanh(combined @ self.W_g + self.b_g)         # Candidate
            o = self._sigmoid(combined @ self.W_o + self.b_o)  # Output

            # Cell state and hidden state
            c = f * c + i * g          # The key: additive update preserves gradients
            h = o * np.tanh(c)

            # Output
            y = h @ self.W_y + self.b_y
            outputs.append(y)

            self.cache.append((combined, f, i, g, o, c, h, np.tanh(c)))

        self.final_h = h
        self.final_c = c
        return outputs

    def backward(self, d_outputs, lr=0.01):
        """BPTT for LSTM."""
        T = len(d_outputs)

        # Initialize gradients
        dW_f = np.zeros_like(self.W_f)
        dW_i = np.zeros_like(self.W_i)
        dW_g = np.zeros_like(self.W_g)
        dW_o = np.zeros_like(self.W_o)
        db_f = np.zeros_like(self.b_f)
        db_i = np.zeros_like(self.b_i)
        db_g = np.zeros_like(self.b_g)
        db_o = np.zeros_like(self.b_o)
        dW_y = np.zeros_like(self.W_y)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros((d_outputs[0].shape[0], self.hidden_size))
        dc_next = np.zeros_like(dh_next)

        for t in reversed(range(T)):
            dy = d_outputs[t]
            combined, f, i, g, o, c, h, tanh_c = self.cache[t]
            c_prev = self.cache[t - 1][5] if t > 0 else np.zeros_like(c)

            # Output layer
            dW_y += h.T @ dy
            db_y += np.sum(dy, axis=0, keepdims=True)

            dh = dy @ self.W_y.T + dh_next

            # Output gate
            do = dh * tanh_c
            do_raw = do * o * (1 - o)  # sigmoid derivative

            # Cell state
            dc = dh * o * (1 - tanh_c ** 2) + dc_next

            # Forget gate
            df = dc * c_prev
            df_raw = df * f * (1 - f)

            # Input gate
            di = dc * g
            di_raw = di * i * (1 - i)

            # Gate gate
            dg = dc * i
            dg_raw = dg * (1 - g ** 2)  # tanh derivative

            # Cell state gradient for previous timestep
            dc_next = dc * f

            # Weight gradients
            dW_f += combined.T @ df_raw
            dW_i += combined.T @ di_raw
            dW_g += combined.T @ dg_raw
            dW_o += combined.T @ do_raw
            db_f += np.sum(df_raw, axis=0, keepdims=True)
            db_i += np.sum(di_raw, axis=0, keepdims=True)
            db_g += np.sum(dg_raw, axis=0, keepdims=True)
            db_o += np.sum(do_raw, axis=0, keepdims=True)

            # Hidden state gradient for previous timestep
            d_combined = (df_raw @ self.W_f.T + di_raw @ self.W_i.T +
                          dg_raw @ self.W_g.T + do_raw @ self.W_o.T)
            dh_next = d_combined[:, self.inputs[0].shape[1]:]

        # Clip and update
        n = d_outputs[0].shape[0]
        for param, grad in [(self.W_f, dW_f), (self.W_i, dW_i),
                            (self.W_g, dW_g), (self.W_o, dW_o),
                            (self.b_f, db_f), (self.b_i, db_i),
                            (self.b_g, db_g), (self.b_o, db_o),
                            (self.W_y, dW_y), (self.b_y, db_y)]:
            np.clip(grad, -5, 5, out=grad)
            param -= lr * grad / n

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ==============================================================================
# Part 3: Experiments
# ==============================================================================

def softmax(z):
    e = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def experiment_vanishing_gradient():
    """Show that vanilla RNN fails on long-range dependencies."""
    print("=" * 60)
    print("EXPERIMENT 1: Vanishing Gradient Problem")
    print("=" * 60)
    print("\nTask: Remember the first element of a sequence and output it at the end.")
    print("RNN must carry information across the entire sequence length.\n")

    np.random.seed(42)

    for seq_len in [5, 10, 20, 50]:
        # Task: first input is the "signal" (0 or 1), rest are noise
        # Output at last step should match the first input
        n_samples = 200
        X_data = np.random.randn(n_samples, seq_len, 1) * 0.1
        y_data = np.random.randint(0, 2, (n_samples, 1)).astype(float)
        X_data[:, 0, 0] = y_data[:, 0]  # Signal at position 0

        rnn = VanillaRNN(1, 16, 1)

        # Train
        for epoch in range(100):
            inputs = [X_data[:, t, :] for t in range(seq_len)]
            outputs, _ = rnn.forward(inputs)

            # Loss only on last timestep
            pred = 1.0 / (1.0 + np.exp(-outputs[-1]))
            d_outputs = [np.zeros_like(outputs[t]) for t in range(seq_len)]
            d_outputs[-1] = (pred - y_data) / n_samples
            rnn.backward(d_outputs, lr=0.1)

        # Evaluate
        inputs = [X_data[:, t, :] for t in range(seq_len)]
        outputs, _ = rnn.forward(inputs)
        pred = (1.0 / (1.0 + np.exp(-outputs[-1])) > 0.5).astype(float)
        acc = np.mean(pred == y_data) * 100

        print(f"  Seq length {seq_len:3d}: RNN accuracy = {acc:.1f}%", end="")
        print("  (random = 50%)" if acc < 60 else "")

    print("\n  RNN accuracy degrades with sequence length — vanishing gradients!")


def experiment_lstm_remembers():
    """Show LSTM handles long-range dependencies."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: LSTM Remembers")
    print("=" * 60)
    print("\nSame task: remember first element, output at end.\n")

    np.random.seed(42)

    for seq_len in [5, 10, 20, 50]:
        n_samples = 200
        X_data = np.random.randn(n_samples, seq_len, 1) * 0.1
        y_data = np.random.randint(0, 2, (n_samples, 1)).astype(float)
        X_data[:, 0, 0] = y_data[:, 0]

        lstm = LSTM(1, 16, 1)

        for epoch in range(100):
            inputs = [X_data[:, t, :] for t in range(seq_len)]
            outputs = lstm.forward(inputs)

            pred = 1.0 / (1.0 + np.exp(-outputs[-1]))
            d_outputs = [np.zeros_like(outputs[t]) for t in range(seq_len)]
            d_outputs[-1] = (pred - y_data) / n_samples
            lstm.backward(d_outputs, lr=0.1)

        inputs = [X_data[:, t, :] for t in range(seq_len)]
        outputs = lstm.forward(inputs)
        pred = (1.0 / (1.0 + np.exp(-outputs[-1])) > 0.5).astype(float)
        acc = np.mean(pred == y_data) * 100

        print(f"  Seq length {seq_len:3d}: LSTM accuracy = {acc:.1f}%")

    print("\n  LSTM maintains accuracy — the cell state preserves information!")
    print("  Key: additive cell update c_t = f*c_{t-1} + i*g (gradient flows through +)")


def experiment_sequence_prediction():
    """Train LSTM on a simple sequence prediction task."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Sequence Prediction")
    print("=" * 60)
    print("\nTask: Predict next value in a sine wave.\n")

    np.random.seed(42)

    # Generate sine wave data
    t = np.linspace(0, 8 * np.pi, 500)
    data = np.sin(t)

    seq_len = 10
    X_seqs, y_seqs = [], []
    for i in range(len(data) - seq_len):
        X_seqs.append(data[i:i + seq_len])
        y_seqs.append(data[i + seq_len])

    X = np.array(X_seqs)[:400]
    y = np.array(y_seqs)[:400].reshape(-1, 1)

    lstm = LSTM(1, 32, 1)

    for epoch in range(201):
        # Mini-batch
        idx = np.random.choice(len(X), 64, replace=False)
        xb = X[idx]
        yb = y[idx]

        inputs = [xb[:, t:t + 1] for t in range(seq_len)]
        outputs = lstm.forward(inputs)

        # MSE loss on last output
        pred = outputs[-1]
        loss = np.mean((pred - yb) ** 2)

        d_outputs = [np.zeros_like(outputs[t]) for t in range(seq_len)]
        d_outputs[-1] = 2 * (pred - yb) / len(yb)

        lstm.backward(d_outputs, lr=0.005)

        if epoch % 50 == 0:
            # Full evaluation
            all_inputs = [X[:, t:t + 1] for t in range(seq_len)]
            all_outputs = lstm.forward(all_inputs)
            all_pred = all_outputs[-1]
            full_loss = np.mean((all_pred - y) ** 2)
            print(f"  Epoch {epoch:3d} | MSE: {full_loss:.6f}")

    # Show some predictions
    print("\n  Sample predictions:")
    all_inputs = [X[:10, t:t + 1] for t in range(seq_len)]
    all_outputs = lstm.forward(all_inputs)
    preds = all_outputs[-1].flatten()
    actual = y[:10].flatten()
    for i in range(10):
        print(f"    Predicted: {preds[i]:7.4f} | Actual: {actual[i]:7.4f} | Error: {abs(preds[i]-actual[i]):.4f}")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("RNN & LSTM FROM SCRATCH")
    print("Sequential processing with memory\n")

    experiment_vanishing_gradient()
    experiment_lstm_remembers()
    experiment_sequence_prediction()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. RNNs process sequences by maintaining a hidden state across timesteps.
2. Vanilla RNNs suffer from vanishing gradients: can't learn long-range dependencies.
3. LSTM solves this with gates and an additive cell state update.
4. The forget gate bias (initialized to 1) = "remember by default".
5. BPTT is just backprop through the unrolled computation graph.
6. Limitation: sequential processing can't be parallelized (motivates transformers).

Next: Autoencoders learn compressed representations.
Then: Attention mechanism removes the sequential bottleneck.
""")

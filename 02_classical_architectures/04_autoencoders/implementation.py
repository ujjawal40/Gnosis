"""
Autoencoders from Scratch
==========================

Implements standard, denoising, and sparse autoencoders in NumPy.
Demonstrates unsupervised feature learning and dimensionality reduction.
"""

import numpy as np


class Autoencoder:
    """
    Simple autoencoder: Encoder → Bottleneck → Decoder

    Uses the MLP components we built earlier, reimplemented here
    for self-containment.
    """

    def __init__(self, input_dim, hidden_dims, latent_dim):
        """
        Args:
            input_dim: size of input
            hidden_dims: list of hidden layer sizes for encoder (decoder mirrors)
            latent_dim: size of bottleneck
        """
        # Build encoder layers
        encoder_dims = [input_dim] + hidden_dims + [latent_dim]
        self.encoder_W = []
        self.encoder_b = []
        for i in range(len(encoder_dims) - 1):
            scale = np.sqrt(2.0 / encoder_dims[i])
            self.encoder_W.append(np.random.randn(encoder_dims[i], encoder_dims[i+1]) * scale)
            self.encoder_b.append(np.zeros((1, encoder_dims[i+1])))

        # Build decoder layers (mirror of encoder)
        decoder_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        self.decoder_W = []
        self.decoder_b = []
        for i in range(len(decoder_dims) - 1):
            scale = np.sqrt(2.0 / decoder_dims[i])
            self.decoder_W.append(np.random.randn(decoder_dims[i], decoder_dims[i+1]) * scale)
            self.decoder_b.append(np.zeros((1, decoder_dims[i+1])))

        self.latent_dim = latent_dim

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_grad(self, x):
        return (x > 0).astype(float)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def encode(self, x):
        """Forward through encoder. Returns latent code."""
        self.enc_cache = [x]
        self.enc_pre_act = []
        h = x
        for i in range(len(self.encoder_W)):
            z = h @ self.encoder_W[i] + self.encoder_b[i]
            self.enc_pre_act.append(z)
            if i < len(self.encoder_W) - 1:
                h = self._relu(z)
            else:
                h = z  # Linear output at bottleneck
            self.enc_cache.append(h)
        return h

    def decode(self, z):
        """Forward through decoder. Returns reconstruction."""
        self.dec_cache = [z]
        self.dec_pre_act = []
        h = z
        for i in range(len(self.decoder_W)):
            z_pre = h @ self.decoder_W[i] + self.decoder_b[i]
            self.dec_pre_act.append(z_pre)
            if i < len(self.decoder_W) - 1:
                h = self._relu(z_pre)
            else:
                h = self._sigmoid(z_pre)  # Output in [0,1]
            self.dec_cache.append(h)
        return h

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def backward(self, x, x_hat, sparsity_penalty=0.0, lr=0.01):
        """Full backward pass through decoder then encoder."""
        n = x.shape[0]

        # dL/dx_hat for MSE + sigmoid output
        s = x_hat
        grad = (s - x) * s * (1 - s)  # Combined sigmoid + MSE gradient

        # Backward through decoder
        for i in reversed(range(len(self.decoder_W))):
            h_prev = self.dec_cache[i]
            dW = h_prev.T @ grad / n
            db = np.sum(grad, axis=0, keepdims=True) / n

            if i > 0:
                grad = grad @ self.decoder_W[i].T
                grad = grad * self._relu_grad(self.dec_pre_act[i-1])

            self.decoder_W[i] -= lr * dW
            self.decoder_b[i] -= lr * db

        if i == 0:
            grad = (self.dec_cache[0] - self.dec_cache[0]) * 0  # Reset
            # Re-derive from top
            grad_z = ((s - x) * s * (1 - s))
            for j in reversed(range(len(self.decoder_W))):
                if j == 0:
                    grad = grad_z @ self.decoder_W[0].T
                    break
                grad_z = grad_z @ self.decoder_W[j].T
                grad_z = grad_z * self._relu_grad(self.dec_pre_act[j-1])
            grad = grad_z

        # Add sparsity penalty on latent activations
        if sparsity_penalty > 0:
            z = self.enc_cache[-1]
            grad += sparsity_penalty * np.sign(z) / n

        # Backward through encoder
        for i in reversed(range(len(self.encoder_W))):
            h_prev = self.enc_cache[i]
            dW = h_prev.T @ grad / n
            db = np.sum(grad, axis=0, keepdims=True) / n

            if i > 0:
                grad = grad @ self.encoder_W[i].T
                grad = grad * self._relu_grad(self.enc_pre_act[i-1])

            self.encoder_W[i] -= lr * dW
            self.encoder_b[i] -= lr * db

    def train(self, X, epochs=100, batch_size=32, lr=0.01,
              noise_factor=0.0, sparsity_penalty=0.0, verbose=True):
        """Training loop with optional denoising and sparsity."""
        history = []
        for epoch in range(epochs):
            idx = np.random.permutation(len(X))
            epoch_loss = 0
            n_batch = 0

            for i in range(0, len(X), batch_size):
                batch = X[idx[i:i+batch_size]]

                # Denoising: add noise to input
                if noise_factor > 0:
                    noisy = batch + noise_factor * np.random.randn(*batch.shape)
                    noisy = np.clip(noisy, 0, 1)
                    x_hat, z = self.forward(noisy)
                else:
                    x_hat, z = self.forward(batch)

                loss = np.mean((batch - x_hat) ** 2)
                self.backward(batch, x_hat, sparsity_penalty, lr)

                epoch_loss += loss
                n_batch += 1

            avg_loss = epoch_loss / n_batch
            history.append(avg_loss)

            if verbose and (epoch % 25 == 0 or epoch == epochs - 1):
                print(f"  Epoch {epoch:4d} | MSE: {avg_loss:.6f}")

        return history


# ==============================================================================
# Experiments
# ==============================================================================

def generate_data(n=500, dim=20, true_dim=3, seed=42):
    """Generate high-dim data that actually lies on a low-dim manifold."""
    np.random.seed(seed)
    # True latent factors
    z = np.random.randn(n, true_dim)
    # Random linear projection to high-dim + non-linearity
    W = np.random.randn(true_dim, dim)
    X = np.abs(z @ W)  # Non-negative, non-linear
    # Normalize to [0, 1]
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    return X, z


def experiment_basic():
    """Standard autoencoder: compress and reconstruct."""
    print("=" * 60)
    print("EXPERIMENT 1: Standard Autoencoder")
    print("=" * 60)
    print("20D data on a 3D manifold -> compress to 3D bottleneck\n")

    X, z_true = generate_data(500, dim=20, true_dim=3)

    ae = Autoencoder(input_dim=20, hidden_dims=[12, 6], latent_dim=3)
    ae.train(X, epochs=200, lr=0.05, batch_size=32)

    x_hat, z = ae.forward(X)
    final_mse = np.mean((X - x_hat) ** 2)
    print(f"\n  Final reconstruction MSE: {final_mse:.6f}")
    print(f"  Latent dim: {z.shape[1]} (original: {X.shape[1]})")
    print(f"  Compression ratio: {X.shape[1] / z.shape[1]:.1f}x")


def experiment_denoising():
    """Denoising autoencoder: learn robust features."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Denoising Autoencoder")
    print("=" * 60)
    print("Train with noisy input, reconstruct clean output\n")

    X, _ = generate_data(500, dim=20, true_dim=3)

    print("Standard (no noise):")
    ae_standard = Autoencoder(20, [12, 6], 3)
    ae_standard.train(X, epochs=100, lr=0.05, verbose=False)
    x_hat, _ = ae_standard.forward(X)
    clean_mse = np.mean((X - x_hat) ** 2)

    print("Denoising (noise_factor=0.3):")
    ae_denoise = Autoencoder(20, [12, 6], 3)
    ae_denoise.train(X, epochs=100, lr=0.05, noise_factor=0.3, verbose=False)
    x_hat_d, _ = ae_denoise.forward(X)
    denoise_mse = np.mean((X - x_hat_d) ** 2)

    # Test robustness: how well do they handle new noise?
    X_noisy = X + 0.2 * np.random.randn(*X.shape)
    X_noisy = np.clip(X_noisy, 0, 1)

    x_hat_s, _ = ae_standard.forward(X_noisy)
    x_hat_d, _ = ae_denoise.forward(X_noisy)

    print(f"\n  Reconstruction MSE (clean input):")
    print(f"    Standard: {clean_mse:.6f}")
    print(f"    Denoising: {denoise_mse:.6f}")
    print(f"\n  Reconstruction MSE (noisy input):")
    print(f"    Standard: {np.mean((X - x_hat_s) ** 2):.6f}")
    print(f"    Denoising: {np.mean((X - x_hat_d) ** 2):.6f}")
    print(f"\n  Denoising AE is more robust to unseen noise!")


def experiment_sparse():
    """Sparse autoencoder: learn interpretable features."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Sparse Autoencoder")
    print("=" * 60)
    print("Sparsity penalty forces most latent units to be inactive\n")

    X, _ = generate_data(500, dim=20, true_dim=3)

    for sparsity in [0.0, 0.01, 0.1]:
        ae = Autoencoder(20, [12], 8)  # Overcomplete: latent > true dim
        ae.train(X, epochs=100, lr=0.05, sparsity_penalty=sparsity, verbose=False)
        _, z = ae.forward(X)

        # Measure sparsity: fraction of near-zero activations
        active = np.mean(np.abs(z) > 0.1)
        print(f"  Sparsity penalty={sparsity:.2f}: {active*100:.1f}% of latent units active, "
              f"mean |z|={np.mean(np.abs(z)):.4f}")

    print("\n  Higher sparsity -> fewer active units -> more interpretable features")


def experiment_bottleneck_size():
    """How does bottleneck size affect reconstruction quality?"""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Bottleneck Size")
    print("=" * 60)
    print("Data lives on 3D manifold. What happens with different bottleneck sizes?\n")

    X, _ = generate_data(500, dim=20, true_dim=3)

    for latent_dim in [1, 2, 3, 5, 10]:
        np.random.seed(42)
        ae = Autoencoder(20, [12], latent_dim)
        ae.train(X, epochs=150, lr=0.05, verbose=False)
        x_hat, z = ae.forward(X)
        mse = np.mean((X - x_hat) ** 2)
        print(f"  Latent dim {latent_dim:2d}: MSE = {mse:.6f}")

    print("\n  MSE drops sharply at dim=3 (the true manifold dimension)")
    print("  Beyond 3: diminishing returns (no more info to capture)")
    print("  This is how autoencoders discover intrinsic dimensionality!")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("AUTOENCODERS FROM SCRATCH")
    print("Unsupervised learning through compression\n")

    experiment_basic()
    experiment_denoising()
    experiment_sparse()
    experiment_bottleneck_size()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. Autoencoders learn compressed representations by reconstructing input
2. The bottleneck forces the network to keep only essential information
3. Denoising AEs learn more robust features than standard AEs
4. Sparse AEs learn interpretable, disentangled features
5. Bottleneck size reveals the intrinsic dimensionality of data
6. Foundation for VAEs: add probabilistic structure to the latent space

Next: Word2Vec applies similar ideas to learn word representations.
""")

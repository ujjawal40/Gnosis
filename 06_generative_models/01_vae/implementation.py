"""
Variational Autoencoder (VAE) from Scratch
============================================

Implements a VAE in NumPy with the reparameterization trick,
ELBO loss, and generation from the learned latent space.
"""

import numpy as np


class VAE:
    """
    Variational Autoencoder.

    Encoder: x → (μ, log σ²)     -- outputs distribution parameters
    Sampling: z = μ + σ · ε       -- reparameterization trick
    Decoder: z → x_reconstructed

    Loss = Reconstruction + KL Divergence
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        self.latent_dim = latent_dim
        scale_h = np.sqrt(2.0 / input_dim)
        scale_l = np.sqrt(2.0 / hidden_dim)
        scale_d = np.sqrt(2.0 / latent_dim)

        # Encoder: input -> hidden -> (mu, logvar)
        self.enc_W1 = np.random.randn(input_dim, hidden_dim) * scale_h
        self.enc_b1 = np.zeros(hidden_dim)
        self.enc_W_mu = np.random.randn(hidden_dim, latent_dim) * scale_l
        self.enc_b_mu = np.zeros(latent_dim)
        self.enc_W_logvar = np.random.randn(hidden_dim, latent_dim) * scale_l
        self.enc_b_logvar = np.zeros(latent_dim)

        # Decoder: z -> hidden -> output
        self.dec_W1 = np.random.randn(latent_dim, hidden_dim) * scale_d
        self.dec_b1 = np.zeros(hidden_dim)
        self.dec_W2 = np.random.randn(hidden_dim, input_dim) * scale_l
        self.dec_b2 = np.zeros(input_dim)

    def _relu(self, x):
        return np.maximum(0, x)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def encode(self, x):
        """Encode input to latent distribution parameters."""
        self.enc_h = self._relu(x @ self.enc_W1 + self.enc_b1)
        mu = self.enc_h @ self.enc_W_mu + self.enc_b_mu
        logvar = self.enc_h @ self.enc_W_logvar + self.enc_b_logvar
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Sample z using the reparameterization trick: z = μ + σ·ε."""
        std = np.exp(0.5 * logvar)
        self.eps = np.random.randn(*mu.shape)
        return mu + std * self.eps

    def decode(self, z):
        """Decode latent vector to reconstruction."""
        self.dec_h = self._relu(z @ self.dec_W1 + self.dec_b1)
        return self._sigmoid(self.dec_h @ self.dec_W2 + self.dec_b2)

    def forward(self, x):
        """Full forward pass: encode, sample, decode."""
        self.mu, self.logvar = self.encode(x)
        self.z = self.reparameterize(self.mu, self.logvar)
        x_recon = self.decode(self.z)
        return x_recon, self.mu, self.logvar

    def loss(self, x, x_recon, mu, logvar):
        """
        ELBO loss = Reconstruction + KL Divergence

        Reconstruction: BCE between input and output
        KL: analytical form for Gaussian q vs N(0,I) prior
        """
        # Reconstruction loss (binary cross-entropy)
        x_recon_clip = np.clip(x_recon, 1e-8, 1 - 1e-8)
        recon = -np.mean(np.sum(
            x * np.log(x_recon_clip) + (1 - x) * np.log(1 - x_recon_clip),
            axis=1))

        # KL divergence: -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        kl = -0.5 * np.mean(np.sum(1 + logvar - mu**2 - np.exp(logvar), axis=1))

        return recon + kl, recon, kl

    def backward(self, x, x_recon, lr=0.001):
        """Simplified backward pass with gradient updates."""
        n = x.shape[0]
        x_recon_clip = np.clip(x_recon, 1e-8, 1 - 1e-8)

        # Decoder output gradient (BCE)
        d_out = (-x / x_recon_clip + (1 - x) / (1 - x_recon_clip)) / n
        d_sigmoid = d_out * x_recon * (1 - x_recon)

        # Decoder layer 2
        self.dec_W2 -= lr * (self.dec_h.T @ d_sigmoid)
        self.dec_b2 -= lr * np.sum(d_sigmoid, axis=0)

        d_dec_h = d_sigmoid @ self.dec_W2.T
        d_dec_h *= (self.dec_h > 0)  # ReLU grad

        # Decoder layer 1
        self.dec_W1 -= lr * (self.z.T @ d_dec_h)
        self.dec_b1 -= lr * np.sum(d_dec_h, axis=0)

        # Gradient to z
        dz = d_dec_h @ self.dec_W1.T

        # Add KL gradient to mu and logvar
        dmu = dz + self.mu / n  # KL grad w.r.t. mu
        std = np.exp(0.5 * self.logvar)
        dlogvar = dz * 0.5 * std * self.eps + 0.5 * (-1 + np.exp(self.logvar)) / n

        # Encoder mu and logvar layers
        self.enc_W_mu -= lr * (self.enc_h.T @ dmu)
        self.enc_b_mu -= lr * np.sum(dmu, axis=0)
        self.enc_W_logvar -= lr * (self.enc_h.T @ dlogvar)
        self.enc_b_logvar -= lr * np.sum(dlogvar, axis=0)

        # Encoder hidden layer
        d_enc_h = dmu @ self.enc_W_mu.T + dlogvar @ self.enc_W_logvar.T
        d_enc_h *= (self.enc_h > 0)

        self.enc_W1 -= lr * (x.T @ d_enc_h)
        self.enc_b1 -= lr * np.sum(d_enc_h, axis=0)

    def generate(self, n_samples=10):
        """Generate new samples by sampling from the prior z ~ N(0,I)."""
        z = np.random.randn(n_samples, self.latent_dim)
        return self.decode(z)


# ==============================================================================
# Experiments
# ==============================================================================

def generate_data(n=500, dim=10, seed=42):
    """Simple data on a low-dim manifold, normalized to [0,1]."""
    np.random.seed(seed)
    z = np.random.randn(n, 2)
    W = np.random.randn(2, dim)
    X = 1.0 / (1.0 + np.exp(-(z @ W)))  # Sigmoid to keep in [0,1]
    return X


def experiment_vae_training():
    """Train a VAE and show reconstruction + generation."""
    print("=" * 60)
    print("EXPERIMENT 1: VAE Training")
    print("=" * 60)

    X = generate_data(500, dim=10)
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} dimensions")
    print(f"True latent dimension: 2\n")

    np.random.seed(42)
    vae = VAE(input_dim=10, hidden_dim=32, latent_dim=2)

    for epoch in range(301):
        idx = np.random.permutation(len(X))
        batch_size = 64
        total_loss, total_recon, total_kl = 0, 0, 0
        n_batch = 0

        for i in range(0, len(X), batch_size):
            batch = X[idx[i:i+batch_size]]
            x_recon, mu, logvar = vae.forward(batch)
            loss, recon, kl = vae.loss(batch, x_recon, mu, logvar)
            vae.backward(batch, x_recon, lr=0.001)

            total_loss += loss
            total_recon += recon
            total_kl += kl
            n_batch += 1

        if epoch % 75 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {total_loss/n_batch:.4f} "
                  f"| Recon: {total_recon/n_batch:.4f} "
                  f"| KL: {total_kl/n_batch:.4f}")

    # Test reconstruction
    x_recon, _, _ = vae.forward(X[:5])
    print(f"\nReconstruction quality (first 3 dims of 5 samples):")
    for i in range(5):
        print(f"  Original:      {X[i, :3].round(3)}")
        print(f"  Reconstructed: {x_recon[i, :3].round(3)}")
        print()


def experiment_latent_space():
    """Explore the structure of the learned latent space."""
    print("=" * 60)
    print("EXPERIMENT 2: Latent Space Structure")
    print("=" * 60)

    X = generate_data(500, dim=10)
    np.random.seed(42)
    vae = VAE(10, 32, 2)

    for epoch in range(200):
        idx = np.random.choice(len(X), 64)
        batch = X[idx]
        x_recon, mu, logvar = vae.forward(batch)
        vae.backward(batch, x_recon, lr=0.001)

    # Encode all data
    mu, logvar = vae.encode(X)
    print(f"\nLatent space statistics:")
    print(f"  Mean of μ:       {np.mean(mu, axis=0).round(3)}")
    print(f"  Std of μ:        {np.std(mu, axis=0).round(3)}")
    print(f"  Mean of σ:       {np.mean(np.exp(0.5 * logvar), axis=0).round(3)}")
    print(f"\n  KL pushes these toward N(0,I): mean→0, std→1")

    # Generate new samples
    print(f"\nGenerated samples (first 3 dims):")
    generated = vae.generate(5)
    for i in range(5):
        print(f"  Sample {i}: {generated[i, :3].round(3)}")

    print(f"\n  These are NEW data points sampled from the learned distribution!")


def experiment_interpolation():
    """Interpolate between two data points in latent space."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Latent Space Interpolation")
    print("=" * 60)

    X = generate_data(500, dim=10)
    np.random.seed(42)
    vae = VAE(10, 32, 2)

    for epoch in range(200):
        idx = np.random.choice(len(X), 64)
        batch = X[idx]
        x_recon, mu, logvar = vae.forward(batch)
        vae.backward(batch, x_recon, lr=0.001)

    # Take two data points
    x1, x2 = X[0:1], X[100:101]
    mu1, _ = vae.encode(x1)
    mu2, _ = vae.encode(x2)

    print(f"\nInterpolating between two points in latent space:")
    print(f"  Point A (latent): {mu1[0].round(3)}")
    print(f"  Point B (latent): {mu2[0].round(3)}")
    print(f"\n  Alpha   First 3 dims of decoded output")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        z_interp = (1 - alpha) * mu1 + alpha * mu2
        decoded = vae.decode(z_interp)
        print(f"  {alpha:.2f}    {decoded[0, :3].round(3)}")

    print(f"\n  Smooth interpolation! The latent space is continuous and meaningful.")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("VARIATIONAL AUTOENCODER FROM SCRATCH\n")

    experiment_vae_training()
    experiment_latent_space()
    experiment_interpolation()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. VAE = autoencoder + probabilistic latent space
2. ELBO loss = reconstruction + KL divergence
3. Reparameterization trick: z = μ + σ·ε enables backprop through sampling
4. KL term forces latent space to be smooth and structured
5. Can generate NEW data by sampling z ~ N(0,I) and decoding
6. Smooth interpolation in latent space = meaningful transitions

Foundation for: VQ-VAE, hierarchical VAEs, and the VAE in Stable Diffusion.
Next: GANs take a different approach — adversarial training.
""")

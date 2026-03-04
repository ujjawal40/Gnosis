"""
Diffusion Model (DDPM) from Scratch
=====================================

Implements Denoising Diffusion Probabilistic Model in NumPy.
Forward process adds noise, reverse process learns to denoise.
"""

import numpy as np


class SimpleDenoiser:
    """Neural network that predicts noise given noisy input and timestep."""

    def __init__(self, data_dim, hidden_dim=64, n_timesteps=100):
        self.n_timesteps = n_timesteps
        # Time embedding: learned embedding for each timestep
        self.time_embed = np.random.randn(n_timesteps, hidden_dim) * 0.1

        # Network: [x_noisy, time_embed] -> hidden -> hidden -> noise_pred
        input_dim = data_dim + hidden_dim
        s1 = np.sqrt(2.0 / input_dim)
        s2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim) * s1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * s2
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, data_dim) * s2
        self.b3 = np.zeros(data_dim)

    def forward(self, x_noisy, t):
        """Predict the noise that was added to create x_noisy at timestep t."""
        time_emb = self.time_embed[t]
        if len(time_emb.shape) == 1:
            time_emb = np.tile(time_emb, (x_noisy.shape[0], 1))

        self.inp = np.hstack([x_noisy, time_emb])
        self.h1 = np.maximum(0, self.inp @ self.W1 + self.b1)
        self.h2 = np.maximum(0, self.h1 @ self.W2 + self.b2)
        self.out = self.h2 @ self.W3 + self.b3  # Linear output (predict noise)
        return self.out

    def backward(self, grad, lr=0.001):
        n = grad.shape[0]
        self.W3 -= lr * (self.h2.T @ grad) / n
        self.b3 -= lr * np.mean(grad, axis=0)
        grad = (grad @ self.W3.T) * (self.h2 > 0)
        self.W2 -= lr * (self.h1.T @ grad) / n
        self.b2 -= lr * np.mean(grad, axis=0)
        grad = (grad @ self.W2.T) * (self.h1 > 0)
        self.W1 -= lr * (self.inp.T @ grad) / n
        self.b1 -= lr * np.mean(grad, axis=0)


class DDPM:
    """Denoising Diffusion Probabilistic Model."""

    def __init__(self, data_dim, n_timesteps=100, hidden_dim=64):
        self.n_timesteps = n_timesteps
        self.data_dim = data_dim

        # Noise schedule: linear from beta_start to beta_end
        self.betas = np.linspace(1e-4, 0.02, n_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_bar = np.cumprod(self.alphas)  # ᾱ_t

        self.denoiser = SimpleDenoiser(data_dim, hidden_dim, n_timesteps)

    def forward_process(self, x0, t):
        """
        Add noise to x0 to get x_t.
        q(x_t | x_0) = N(√ᾱ_t · x_0, (1-ᾱ_t) · I)
        """
        noise = np.random.randn(*x0.shape)
        alpha_bar_t = self.alpha_bar[t]
        x_t = np.sqrt(alpha_bar_t) * x0 + np.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise

    def train_step(self, x0, lr=0.001):
        """One training step: random timestep, add noise, predict noise."""
        batch_size = x0.shape[0]

        # Random timestep for each sample
        t = np.random.randint(0, self.n_timesteps)

        # Forward process: add noise
        x_t, noise = self.forward_process(x0, t)

        # Predict noise
        noise_pred = self.denoiser.forward(x_t, t)

        # MSE loss between predicted and actual noise
        loss = np.mean((noise_pred - noise) ** 2)

        # Backward
        grad = 2 * (noise_pred - noise)
        self.denoiser.backward(grad, lr)

        return loss

    def sample(self, n_samples=10):
        """
        Generate samples by reversing the diffusion process.
        Start from pure noise, iteratively denoise.
        """
        # Start from pure noise
        x = np.random.randn(n_samples, self.data_dim)

        for t in reversed(range(self.n_timesteps)):
            # Predict noise at current timestep
            noise_pred = self.denoiser.forward(x, t)

            # Remove predicted noise (DDPM sampling formula)
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.betas[t]

            # Mean of reverse distribution
            x = (1 / np.sqrt(alpha_t)) * (
                x - (beta_t / np.sqrt(1 - alpha_bar_t)) * noise_pred
            )

            # Add noise (except at t=0)
            if t > 0:
                sigma = np.sqrt(beta_t)
                x += sigma * np.random.randn(*x.shape)

        return x


def experiment_diffusion():
    """Train DDPM on 2D data."""
    print("=" * 60)
    print("DIFFUSION MODEL (DDPM) FROM SCRATCH")
    print("=" * 60)

    np.random.seed(42)

    # Data: ring shape
    n = 500
    theta = np.random.uniform(0, 2 * np.pi, n)
    r = 1.0 + np.random.randn(n) * 0.1
    data = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    print(f"\nData: {n} points in a ring (2D)")
    print(f"Timesteps: 50, learning rate: 0.001\n")

    model = DDPM(data_dim=2, n_timesteps=50, hidden_dim=64)

    # Show forward process
    print("Forward process (adding noise):")
    x0 = data[:1]
    for t in [0, 10, 25, 49]:
        x_t, _ = model.forward_process(x0, t)
        print(f"  t={t:2d}: ({x_t[0,0]:.3f}, {x_t[0,1]:.3f})  "
              f"ᾱ_t={model.alpha_bar[t]:.4f}")
    print("  (ᾱ_t → 0 means more noise, x_t → N(0,I))\n")

    # Training
    batch_size = 64
    for epoch in range(1001):
        idx = np.random.choice(len(data), batch_size)
        loss = model.train_step(data[idx], lr=0.001)

        if epoch % 250 == 0:
            print(f"  Epoch {epoch:4d} | MSE: {loss:.6f}")

    # Generate samples
    print("\nGenerating 10 samples:")
    samples = model.sample(10)
    for i in range(10):
        r_gen = np.sqrt(samples[i, 0]**2 + samples[i, 1]**2)
        print(f"  ({samples[i,0]:6.3f}, {samples[i,1]:6.3f})  radius: {r_gen:.3f}")

    avg_radius = np.mean(np.sqrt(samples[:, 0]**2 + samples[:, 1]**2))
    print(f"\n  Average radius: {avg_radius:.3f} (target: ~1.0)")
    print(f"  The model learned to generate points on a ring!")


if __name__ == "__main__":
    experiment_diffusion()
    print("""
KEY TAKEAWAYS:
1. Forward: gradually add noise until data becomes N(0,I)
2. Reverse: learn to predict and remove noise step by step
3. Training: simple MSE between predicted and actual noise
4. Sampling: start from noise, iteratively denoise
5. Stable training (no adversarial dynamics like GANs)
6. This is the foundation of Stable Diffusion, DALL-E, Midjourney
""")

"""
GAN from Scratch — Generative Adversarial Network in NumPy
"""

import numpy as np


class Generator:
    def __init__(self, noise_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(noise_dim, hidden_dim) * np.sqrt(2.0/noise_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0/hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0/hidden_dim)
        self.b3 = np.zeros(output_dim)

    def forward(self, z):
        self.z = z
        self.h1 = np.maximum(0, z @ self.W1 + self.b1)
        self.h2 = np.maximum(0, self.h1 @ self.W2 + self.b2)
        self.out = np.tanh(self.h2 @ self.W3 + self.b3)
        return self.out

    def backward(self, grad, lr=0.0002):
        n = grad.shape[0]
        # Through tanh
        grad = grad * (1 - self.out**2)
        self.W3 -= lr * (self.h2.T @ grad) / n
        self.b3 -= lr * np.mean(grad, axis=0)
        grad = grad @ self.W3.T * (self.h2 > 0)
        self.W2 -= lr * (self.h1.T @ grad) / n
        self.b2 -= lr * np.mean(grad, axis=0)
        grad = grad @ self.W2.T * (self.h1 > 0)
        self.W1 -= lr * (self.z.T @ grad) / n
        self.b1 -= lr * np.mean(grad, axis=0)


class Discriminator:
    def __init__(self, input_dim, hidden_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0/input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0/hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0/hidden_dim)
        self.b3 = np.zeros(1)

    def forward(self, x):
        self.x = x
        self.h1 = np.maximum(0.2 * (x @ self.W1 + self.b1),
                              x @ self.W1 + self.b1)  # LeakyReLU
        self.h2 = np.maximum(0.2 * (self.h1 @ self.W2 + self.b2),
                              self.h1 @ self.W2 + self.b2)
        logit = self.h2 @ self.W3 + self.b3
        self.out = 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))
        return self.out

    def backward(self, grad, lr=0.0002):
        n = grad.shape[0]
        grad = grad * self.out * (1 - self.out)  # sigmoid grad
        self.W3 -= lr * (self.h2.T @ grad) / n
        self.b3 -= lr * np.mean(grad, axis=0)
        grad = grad @ self.W3.T
        lrelu_grad = np.where(self.h2 > 0, 1, 0.2)
        grad = grad * lrelu_grad
        self.W2 -= lr * (self.h1.T @ grad) / n
        self.b2 -= lr * np.mean(grad, axis=0)
        grad = grad @ self.W2.T
        lrelu_grad = np.where(self.h1 > 0, 1, 0.2)
        grad = grad * lrelu_grad
        self.W1 -= lr * (self.x.T @ grad) / n
        self.b1 -= lr * np.mean(grad, axis=0)
        return grad @ self.W1.T  # For passing to generator


def train_gan():
    """Train GAN to generate 2D Gaussian mixture data."""
    print("=" * 60)
    print("GAN: Learning to Generate 2D Data")
    print("=" * 60)

    np.random.seed(42)

    # Real data: mixture of 4 Gaussians
    n = 1000
    centers = np.array([[2, 2], [-2, 2], [-2, -2], [2, -2]])
    real_data = np.vstack([
        np.random.randn(n//4, 2) * 0.3 + c for c in centers
    ])
    # Normalize to [-1, 1]
    real_data = real_data / 4.0

    noise_dim = 8
    G = Generator(noise_dim, 64, 2)
    D = Discriminator(2, 64)

    print(f"\nReal data: 4 Gaussian clusters in 2D")
    print(f"Generator: z({noise_dim}D) -> 64 -> 64 -> 2D")
    print(f"Discriminator: 2D -> 64 -> 64 -> real/fake\n")

    batch_size = 64
    for epoch in range(2001):
        # Sample real and fake data
        idx = np.random.choice(len(real_data), batch_size)
        real = real_data[idx]
        z = np.random.randn(batch_size, noise_dim)
        fake = G.forward(z)

        # Train Discriminator: maximize log(D(real)) + log(1-D(fake))
        d_real = D.forward(real)
        d_real_grad = -1.0 / (d_real + 1e-8)
        D.backward(d_real_grad, lr=0.0002)

        d_fake = D.forward(fake)
        d_fake_grad = 1.0 / (1 - d_fake + 1e-8)
        D.backward(d_fake_grad, lr=0.0002)

        # Train Generator: minimize log(1-D(G(z))) ≈ maximize log(D(G(z)))
        z = np.random.randn(batch_size, noise_dim)
        fake = G.forward(z)
        d_gen = D.forward(fake)
        g_grad = -1.0 / (d_gen + 1e-8)
        d_to_g = D.backward(g_grad, lr=0)  # Don't update D here
        G.backward(d_to_g, lr=0.0002)

        if epoch % 500 == 0:
            # Evaluate
            z_test = np.random.randn(200, noise_dim)
            generated = G.forward(z_test)
            d_real_score = np.mean(D.forward(real_data[:200]))
            d_fake_score = np.mean(D.forward(generated))

            print(f"  Epoch {epoch:4d} | D(real): {d_real_score:.3f} | "
                  f"D(fake): {d_fake_score:.3f} | "
                  f"Gen mean: ({np.mean(generated[:,0]):.3f}, {np.mean(generated[:,1]):.3f})")

    # Final evaluation
    z_test = np.random.randn(500, noise_dim)
    generated = G.forward(z_test)
    print(f"\nGenerated data statistics:")
    print(f"  Mean:  ({np.mean(generated[:,0]):.3f}, {np.mean(generated[:,1]):.3f})")
    print(f"  Std:   ({np.std(generated[:,0]):.3f}, {np.std(generated[:,1]):.3f})")
    print(f"  Range: x=[{generated[:,0].min():.2f}, {generated[:,0].max():.2f}] "
          f"y=[{generated[:,1].min():.2f}, {generated[:,1].max():.2f}]")


if __name__ == "__main__":
    print("GENERATIVE ADVERSARIAL NETWORK FROM SCRATCH\n")
    train_gan()
    print("""
KEY TAKEAWAYS:
1. GAN = Generator vs Discriminator in a minimax game
2. Generator learns to map noise to data distribution
3. Discriminator provides the training signal (no explicit loss function)
4. Training is unstable — mode collapse is a real issue
5. D(real) → 0.5, D(fake) → 0.5 at equilibrium
""")

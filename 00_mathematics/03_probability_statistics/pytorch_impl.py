"""
Probability & Statistics with PyTorch
=======================================

PyTorch re-implementation of probability and statistics from implementation.py.
Uses torch.distributions for sampling, inference, and mixture models.

Comparison:
    NumPy (implementation.py)         →  PyTorch (this file)
    ───────────────────────────────────────────────────────────
    Custom Gaussian class             →  torch.distributions.Normal
    Custom Bernoulli class            →  torch.distributions.Bernoulli
    Custom Categorical class          →  torch.distributions.Categorical
    Custom Beta distribution          →  torch.distributions.Beta
    Manual Box-Muller sampling        →  dist.sample()
    Manual EM algorithm               →  Gradient-based GMM with autograd
    Manual MLE                        →  MLE via autograd optimization
"""

import torch
import torch.distributions as dist
import numpy as np


# =============================================================================
# PART 1: DISTRIBUTIONS
# =============================================================================

def demo_distributions():
    """Demonstrate torch.distributions API."""
    print("=" * 60)
    print("TORCH.DISTRIBUTIONS")
    print("=" * 60)

    # Gaussian
    normal = dist.Normal(loc=0.0, scale=1.0)
    samples = normal.sample((1000,))
    print(f"Normal(0,1): mean={samples.mean():.4f}, std={samples.std():.4f}")
    print(f"  log_prob(0) = {normal.log_prob(torch.tensor(0.0)):.4f}")
    print(f"  entropy = {normal.entropy():.4f}")

    # Bernoulli
    bern = dist.Bernoulli(probs=0.7)
    samples = bern.sample((1000,))
    print(f"\nBernoulli(0.7): empirical p = {samples.mean():.4f}")

    # Categorical
    cat = dist.Categorical(probs=torch.tensor([0.1, 0.3, 0.6]))
    samples = cat.sample((1000,))
    counts = torch.bincount(samples.long(), minlength=3).float() / 1000
    print(f"\nCategorical([0.1, 0.3, 0.6]): empirical = {counts.tolist()}")

    # Beta (conjugate prior for Bernoulli)
    beta = dist.Beta(2.0, 5.0)
    samples = beta.sample((1000,))
    print(f"\nBeta(2,5): mean={samples.mean():.4f} "
          f"(theoretical: {2/(2+5):.4f})")
    print()


# =============================================================================
# PART 2: BAYESIAN INFERENCE
# =============================================================================

def demo_bayesian_update():
    """Bayesian coin-flip inference using Beta-Bernoulli conjugacy."""
    print("=" * 60)
    print("BAYESIAN INFERENCE (Beta-Bernoulli)")
    print("=" * 60)

    true_p = 0.7
    torch.manual_seed(42)

    # Prior: Beta(1, 1) = Uniform
    alpha, beta_param = 1.0, 1.0

    data = dist.Bernoulli(probs=true_p).sample((100,))

    # Sequential updates
    for n in [1, 5, 10, 25, 50, 100]:
        observed = data[:n]
        heads = observed.sum().item()
        tails = n - heads

        post_alpha = alpha + heads
        post_beta = beta_param + tails
        posterior = dist.Beta(post_alpha, post_beta)

        mean = post_alpha / (post_alpha + post_beta)
        std = torch.sqrt(posterior.variance)
        print(f"  n={n:3d}: posterior=Beta({post_alpha:.0f},{post_beta:.0f}), "
              f"mean={mean:.4f}, std={std.item():.4f}")

    print(f"  True p = {true_p}")
    print()


# =============================================================================
# PART 3: MAXIMUM LIKELIHOOD ESTIMATION WITH AUTOGRAD
# =============================================================================

def demo_mle():
    """MLE using torch.autograd instead of closed-form solutions."""
    print("=" * 60)
    print("MLE WITH AUTOGRAD")
    print("=" * 60)

    torch.manual_seed(42)

    # Generate data from N(3.0, 2.0)
    true_mu, true_sigma = 3.0, 2.0
    data = dist.Normal(true_mu, true_sigma).sample((500,))

    # Learn mu and sigma via gradient descent on NLL
    mu = torch.tensor(0.0, requires_grad=True)
    log_sigma = torch.tensor(0.0, requires_grad=True)  # log for positivity

    optimizer = torch.optim.Adam([mu, log_sigma], lr=0.05)

    for step in range(501):
        sigma = torch.exp(log_sigma)
        nll = -dist.Normal(mu, sigma).log_prob(data).mean()

        optimizer.zero_grad()
        nll.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"  Step {step:3d}: mu={mu.item():.4f}, "
                  f"sigma={torch.exp(log_sigma).item():.4f}, NLL={nll.item():.4f}")

    print(f"  True: mu={true_mu}, sigma={true_sigma}")
    print()


# =============================================================================
# PART 4: GAUSSIAN MIXTURE MODEL (EM with autograd)
# =============================================================================

class GaussianMixtureModel:
    """
    GMM using EM algorithm with PyTorch tensors.
    Matches the NumPy implementation but uses torch for computations.
    """

    def __init__(self, n_components: int, n_dims: int):
        self.k = n_components
        self.d = n_dims
        self.means = None
        self.covs = None
        self.weights = None

    def fit(self, X: torch.Tensor, n_iter: int = 100, tol: float = 1e-6):
        """Fit GMM using Expectation-Maximization."""
        n = X.shape[0]

        # Initialize
        indices = torch.randperm(n)[:self.k]
        self.means = X[indices].clone()
        self.covs = torch.stack([torch.eye(self.d)] * self.k)
        self.weights = torch.ones(self.k) / self.k

        prev_ll = float("-inf")

        for iteration in range(n_iter):
            # E-step: compute responsibilities
            resp = self._e_step(X)

            # M-step: update parameters
            self._m_step(X, resp)

            # Check convergence
            ll = self.log_likelihood(X)
            if abs(ll - prev_ll) < tol:
                print(f"  Converged at iteration {iteration}")
                break
            prev_ll = ll

        return self

    def _e_step(self, X: torch.Tensor) -> torch.Tensor:
        """Compute responsibilities (posterior of component assignments)."""
        n = X.shape[0]
        log_resp = torch.zeros(n, self.k)

        for j in range(self.k):
            mvn = dist.MultivariateNormal(self.means[j], self.covs[j])
            log_resp[:, j] = torch.log(self.weights[j]) + mvn.log_prob(X)

        # Normalize in log space
        log_resp -= torch.logsumexp(log_resp, dim=1, keepdim=True)
        return torch.exp(log_resp)

    def _m_step(self, X: torch.Tensor, resp: torch.Tensor):
        """Update parameters using responsibilities."""
        n = X.shape[0]
        Nk = resp.sum(dim=0)

        for j in range(self.k):
            self.weights[j] = Nk[j] / n
            self.means[j] = (resp[:, j:j+1] * X).sum(dim=0) / Nk[j]

            diff = X - self.means[j]
            self.covs[j] = (resp[:, j:j+1] * diff).T @ diff / Nk[j]
            # Add regularization for stability
            self.covs[j] += 1e-6 * torch.eye(self.d)

    def log_likelihood(self, X: torch.Tensor) -> float:
        """Compute log-likelihood of data under the model."""
        n = X.shape[0]
        log_probs = torch.zeros(n, self.k)

        for j in range(self.k):
            mvn = dist.MultivariateNormal(self.means[j], self.covs[j])
            log_probs[:, j] = torch.log(self.weights[j]) + mvn.log_prob(X)

        return torch.logsumexp(log_probs, dim=1).sum().item()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Assign each sample to the most likely component."""
        resp = self._e_step(X)
        return resp.argmax(dim=1)


def demo_gmm():
    """Fit a GMM on synthetic data."""
    print("=" * 60)
    print("GAUSSIAN MIXTURE MODEL (EM)")
    print("=" * 60)

    torch.manual_seed(42)

    # Generate 3-component mixture
    n_per = 200
    X1 = dist.MultivariateNormal(torch.tensor([0., 0.]),
                                  torch.eye(2) * 0.5).sample((n_per,))
    X2 = dist.MultivariateNormal(torch.tensor([4., 4.]),
                                  torch.eye(2) * 0.8).sample((n_per,))
    X3 = dist.MultivariateNormal(torch.tensor([0., 5.]),
                                  torch.eye(2) * 0.3).sample((n_per,))
    X = torch.cat([X1, X2, X3], dim=0)

    gmm = GaussianMixtureModel(n_components=3, n_dims=2)
    gmm.fit(X, n_iter=100)

    labels = gmm.predict(X)
    print(f"  Cluster sizes: {torch.bincount(labels).tolist()}")
    print(f"  Learned means:")
    for j in range(3):
        print(f"    Component {j}: {gmm.means[j].tolist()}")
    print(f"  Log-likelihood: {gmm.log_likelihood(X):.2f}")
    print()


# =============================================================================
# PART 5: MONTE CARLO METHODS
# =============================================================================

def demo_monte_carlo():
    """Estimate pi using Monte Carlo with PyTorch."""
    print("=" * 60)
    print("MONTE CARLO (Pi Estimation)")
    print("=" * 60)

    torch.manual_seed(42)

    for n in [100, 1_000, 10_000, 100_000, 1_000_000]:
        points = torch.rand(n, 2) * 2 - 1  # Uniform in [-1, 1]²
        inside = (points.norm(dim=1) <= 1.0).float()
        pi_est = 4.0 * inside.mean().item()
        error = abs(pi_est - np.pi)
        print(f"  n={n:>9d}: π ≈ {pi_est:.6f}, error={error:.6f}")
    print()


if __name__ == "__main__":
    demo_distributions()
    demo_bayesian_update()
    demo_mle()
    demo_gmm()
    demo_monte_carlo()

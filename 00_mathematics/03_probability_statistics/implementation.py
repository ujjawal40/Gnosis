"""
Probability & Statistics: From-Scratch Implementation
======================================================

Everything built with NumPy only. No scipy.stats, no sklearn.
Each section implements a concept from theory.md and demonstrates it.

Sections:
    1. Probability Distributions (Gaussian, Bernoulli, Categorical)
    2. Bayes' Theorem: Coin Flip with Prior Update
    3. Maximum Likelihood Estimation (Gaussian, Bernoulli)
    4. Monte Carlo Estimation of Pi
    5. Central Limit Theorem Demonstration
    6. Gaussian Mixture Model via EM Algorithm
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import os

# Create output directory for plots
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

np.random.seed(42)


# ===========================================================================
# Section 1: Probability Distributions from Scratch
# ===========================================================================

class Gaussian:
    """
    Univariate Gaussian (Normal) distribution.

    PDF: f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

    We implement everything from the formula — no calls to np.random.normal
    for the PDF/log-prob computations.
    """

    def __init__(self, mu=0.0, sigma=1.0):
        assert sigma > 0, "Standard deviation must be positive"
        self.mu = mu
        self.sigma = sigma
        self.variance = sigma ** 2

    def pdf(self, x):
        """Probability density function."""
        x = np.asarray(x, dtype=np.float64)
        normalization = 1.0 / np.sqrt(2.0 * np.pi * self.variance)
        exponent = -0.5 * ((x - self.mu) ** 2) / self.variance
        return normalization * np.exp(exponent)

    def log_prob(self, x):
        """Log probability density — more numerically stable than log(pdf(x))."""
        x = np.asarray(x, dtype=np.float64)
        return (-0.5 * np.log(2.0 * np.pi * self.variance)
                - 0.5 * ((x - self.mu) ** 2) / self.variance)

    def cdf(self, x):
        """
        Cumulative distribution function using the error function.
        CDF(x) = 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))

        We implement erf via its Taylor series for educational purposes,
        but use a rational approximation for accuracy.
        """
        x = np.asarray(x, dtype=np.float64)
        z = (x - self.mu) / (self.sigma * np.sqrt(2.0))
        return 0.5 * (1.0 + self._erf(z))

    def sample(self, n=1):
        """
        Sample using the Box-Muller transform.

        If U1, U2 ~ Uniform(0,1), then:
            Z = sqrt(-2*ln(U1)) * cos(2*pi*U2) ~ N(0,1)

        This is how random Gaussian samples are actually generated.
        """
        u1 = np.random.uniform(0, 1, size=n)
        u2 = np.random.uniform(0, 1, size=n)
        # Box-Muller transform
        z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
        return self.mu + self.sigma * z

    @staticmethod
    def _erf(z):
        """Approximate error function using Abramowitz and Stegun formula 7.1.26."""
        z = np.asarray(z, dtype=np.float64)
        sign = np.sign(z)
        z = np.abs(z)
        # Constants for the approximation
        p = 0.3275911
        a1, a2, a3, a4, a5 = (0.254829592, -0.284496736, 1.421413741,
                                -1.453152027, 1.061405429)
        t = 1.0 / (1.0 + p * z)
        poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
        result = 1.0 - poly * np.exp(-z * z)
        return sign * result

    def __repr__(self):
        return f"Gaussian(mu={self.mu:.4f}, sigma={self.sigma:.4f})"


class Bernoulli:
    """
    Bernoulli distribution: single coin flip.

    PMF: p(x) = theta^x * (1-theta)^(1-x),  x in {0, 1}
    """

    def __init__(self, theta=0.5):
        assert 0.0 <= theta <= 1.0, "theta must be in [0, 1]"
        self.theta = theta

    def pmf(self, x):
        """Probability mass function."""
        x = np.asarray(x, dtype=np.float64)
        return np.where(x == 1, self.theta, np.where(x == 0, 1.0 - self.theta, 0.0))

    def log_prob(self, x):
        """Log probability mass function."""
        x = np.asarray(x, dtype=np.float64)
        # Clip to avoid log(0)
        theta_safe = np.clip(self.theta, 1e-10, 1.0 - 1e-10)
        return x * np.log(theta_safe) + (1.0 - x) * np.log(1.0 - theta_safe)

    def sample(self, n=1):
        """Sample by thresholding uniform random variables."""
        u = np.random.uniform(0, 1, size=n)
        return (u < self.theta).astype(np.float64)

    def __repr__(self):
        return f"Bernoulli(theta={self.theta:.4f})"


class Categorical:
    """
    Categorical distribution over K categories.

    PMF: p(x=k) = theta_k,  k in {0, 1, ..., K-1}

    This is what a softmax output represents.
    """

    def __init__(self, probs):
        probs = np.asarray(probs, dtype=np.float64)
        assert np.all(probs >= 0), "Probabilities must be non-negative"
        assert np.abs(probs.sum() - 1.0) < 1e-8, f"Probabilities must sum to 1, got {probs.sum()}"
        self.probs = probs
        self.K = len(probs)

    def pmf(self, x):
        """Probability mass function. x is a category index (or array of indices)."""
        x = np.asarray(x, dtype=np.int64)
        return self.probs[x]

    def log_prob(self, x):
        """Log probability mass function."""
        x = np.asarray(x, dtype=np.int64)
        probs_safe = np.clip(self.probs, 1e-10, 1.0)
        return np.log(probs_safe[x])

    def sample(self, n=1):
        """
        Sample using the inverse CDF method (also called the inverse transform method).

        1. Compute cumulative probabilities
        2. Draw u ~ Uniform(0,1)
        3. Return the smallest k such that CDF(k) > u
        """
        cumulative = np.cumsum(self.probs)
        u = np.random.uniform(0, 1, size=n)
        # For each u, find the first cumulative bin it falls into
        samples = np.searchsorted(cumulative, u)
        return samples

    def __repr__(self):
        return f"Categorical(K={self.K}, probs={np.round(self.probs, 4)})"


def demo_distributions():
    """Demonstrate all three distributions."""
    print("=" * 70)
    print("SECTION 1: Probability Distributions from Scratch")
    print("=" * 70)

    # --- Gaussian ---
    print("\n--- Gaussian Distribution ---")
    g = Gaussian(mu=2.0, sigma=1.5)
    print(f"Distribution: {g}")

    samples = g.sample(10000)
    print(f"Sample mean:     {samples.mean():.4f}  (theoretical: {g.mu:.4f})")
    print(f"Sample std:      {samples.std():.4f}  (theoretical: {g.sigma:.4f})")

    # Verify PDF integrates to ~1 (numerical integration via Riemann sum)
    x_range = np.linspace(-10, 14, 10000)
    dx = x_range[1] - x_range[0]
    integral = np.sum(g.pdf(x_range) * dx)
    print(f"PDF integral:    {integral:.6f}  (should be 1.0)")

    # Verify CDF
    print(f"CDF(mu):         {g.cdf(g.mu):.6f}  (should be 0.5)")
    print(f"CDF(mu+sigma):   {g.cdf(g.mu + g.sigma):.6f}  (should be ~0.8413)")
    print(f"CDF(mu+2*sigma): {g.cdf(g.mu + 2 * g.sigma):.6f}  (should be ~0.9772)")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    x = np.linspace(-4, 8, 500)
    axes[0].plot(x, g.pdf(x), 'b-', linewidth=2, label='PDF')
    axes[0].hist(samples, bins=80, density=True, alpha=0.5, color='steelblue', label='Samples')
    axes[0].set_title(f'Gaussian PDF (mu={g.mu}, sigma={g.sigma})')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Density')
    axes[0].legend()

    axes[1].plot(x, g.cdf(x), 'r-', linewidth=2)
    axes[1].set_title('Gaussian CDF')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('P(X <= x)')
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(x=g.mu, color='gray', linestyle='--', alpha=0.5)

    # Show multiple Gaussians
    for mu, sigma, color in [(0, 0.5, 'red'), (0, 1, 'blue'), (0, 2, 'green'), (2, 1, 'purple')]:
        dist = Gaussian(mu, sigma)
        axes[2].plot(x, dist.pdf(x), color=color, linewidth=2,
                     label=f'N({mu}, {sigma}^2)')
    axes[2].set_title('Gaussian Family')
    axes[2].set_xlabel('x')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "01_gaussian_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {os.path.join(PLOT_DIR, '01_gaussian_distribution.png')}")

    # --- Bernoulli ---
    print("\n--- Bernoulli Distribution ---")
    b = Bernoulli(theta=0.7)
    print(f"Distribution: {b}")

    samples = b.sample(10000)
    print(f"Sample mean:     {samples.mean():.4f}  (theoretical: {b.theta:.4f})")
    print(f"Sample variance: {samples.var():.4f}  (theoretical: {b.theta * (1 - b.theta):.4f})")
    print(f"P(X=1):          {b.pmf(1):.4f}")
    print(f"P(X=0):          {b.pmf(0):.4f}")
    print(f"log P(X=1):      {b.log_prob(1):.4f}")
    print(f"log P(X=0):      {b.log_prob(0):.4f}")

    # --- Categorical ---
    print("\n--- Categorical Distribution ---")
    c = Categorical(probs=np.array([0.1, 0.3, 0.4, 0.15, 0.05]))
    print(f"Distribution: {c}")

    samples = c.sample(10000)
    empirical_probs = np.bincount(samples, minlength=c.K) / len(samples)
    print(f"Theoretical probs: {c.probs}")
    print(f"Empirical probs:   {empirical_probs}")

    fig, ax = plt.subplots(figsize=(8, 4))
    bar_width = 0.35
    indices = np.arange(c.K)
    ax.bar(indices - bar_width / 2, c.probs, bar_width, label='Theoretical', color='steelblue')
    ax.bar(indices + bar_width / 2, empirical_probs, bar_width, label='Empirical (n=10000)', color='coral')
    ax.set_xlabel('Category')
    ax.set_ylabel('Probability')
    ax.set_title('Categorical Distribution: Theoretical vs Empirical')
    ax.set_xticks(indices)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "02_categorical_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {os.path.join(PLOT_DIR, '02_categorical_distribution.png')}")


# ===========================================================================
# Section 2: Bayes' Theorem — Coin Flip with Prior Update
# ===========================================================================

class BetaDistribution:
    """
    Beta distribution: the conjugate prior for Bernoulli/Binomial.

    PDF: f(theta; alpha, beta) = [theta^(alpha-1) * (1-theta)^(beta-1)] / B(alpha, beta)

    where B(alpha, beta) = Gamma(alpha)*Gamma(beta) / Gamma(alpha+beta)

    We implement B using the log-gamma function for numerical stability.
    """

    def __init__(self, alpha=1.0, beta=1.0):
        assert alpha > 0 and beta > 0, "Parameters must be positive"
        self.alpha = alpha
        self.beta = beta

    def pdf(self, theta):
        """Probability density at theta."""
        theta = np.asarray(theta, dtype=np.float64)
        # Use log to avoid overflow, then exponentiate
        log_pdf = self._log_pdf(theta)
        return np.exp(log_pdf)

    def _log_pdf(self, theta):
        """Log PDF for numerical stability."""
        theta = np.asarray(theta, dtype=np.float64)
        # Handle boundary values
        log_pdf = np.full_like(theta, -np.inf)
        valid = (theta > 0) & (theta < 1)
        t = theta[valid]
        log_pdf[valid] = ((self.alpha - 1) * np.log(t) +
                          (self.beta - 1) * np.log(1 - t) -
                          self._log_beta(self.alpha, self.beta))
        return log_pdf

    @staticmethod
    def _log_beta(a, b):
        """Log of the Beta function using the log-gamma function."""
        return _log_gamma(a) + _log_gamma(b) - _log_gamma(a + b)

    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    def mode(self):
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
        return None  # Mode undefined or at boundary

    def variance(self):
        ab = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab ** 2 * (ab + 1))

    def update(self, heads, tails):
        """
        Bayesian update after observing data.

        The magic of conjugate priors:
            Prior:     Beta(alpha, beta)
            Data:      h heads, t tails
            Posterior: Beta(alpha + h, beta + t)
        """
        return BetaDistribution(self.alpha + heads, self.beta + tails)

    def __repr__(self):
        return f"Beta(alpha={self.alpha:.2f}, beta={self.beta:.2f})"


def _log_gamma(x):
    """
    Log-gamma function using Stirling's approximation with Lanczos correction.
    Good enough for our purposes; production code uses more precise methods.
    """
    # Use the Lanczos approximation (same approach as most numerical libraries)
    x = float(x)
    if x <= 0:
        return float('inf')

    # Coefficients for Lanczos approximation (g=7, n=9)
    coefs = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]

    if x < 0.5:
        # Use reflection formula: Gamma(x)*Gamma(1-x) = pi / sin(pi*x)
        return (np.log(np.pi) - np.log(np.abs(np.sin(np.pi * x)))
                - _log_gamma(1 - x))

    x -= 1
    g = 7
    a = coefs[0]
    t = x + g + 0.5
    for i in range(1, len(coefs)):
        a += coefs[i] / (x + i)

    return 0.5 * np.log(2 * np.pi) + (x + 0.5) * np.log(t) - t + np.log(a)


def demo_bayes_coin_flip():
    """
    Demonstrate Bayesian updating: infer coin bias from sequential observations.

    True bias: 0.7 (unfair coin)
    Prior: Beta(1,1) = Uniform (we start knowing nothing)

    Watch how the posterior concentrates around the true value as we see more data.
    """
    print("\n" + "=" * 70)
    print("SECTION 2: Bayes' Theorem — Coin Flip with Prior Update")
    print("=" * 70)

    true_theta = 0.7

    # Generate coin flip data
    n_total = 500
    flips = (np.random.uniform(size=n_total) < true_theta).astype(int)

    # Observation checkpoints for plotting
    checkpoints = [0, 1, 5, 10, 25, 50, 100, 250, 500]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()
    theta_range = np.linspace(0.001, 0.999, 500)

    prior = BetaDistribution(1.0, 1.0)  # Uniform prior — we know nothing

    print(f"\nTrue coin bias: {true_theta}")
    print(f"Prior: {prior} (uniform — no prior knowledge)")
    print(f"\nSequential Bayesian updates:")
    print(f"{'Flips seen':>12} {'Heads':>6} {'Tails':>6} {'Posterior Mean':>16} {'Posterior Std':>14}")
    print("-" * 60)

    for idx, n in enumerate(checkpoints):
        if n == 0:
            posterior = prior
            heads, tails = 0, 0
        else:
            observed = flips[:n]
            heads = observed.sum()
            tails = n - heads
            posterior = prior.update(heads, tails)

        post_mean = posterior.mean()
        post_std = np.sqrt(posterior.variance())
        print(f"{n:>12} {heads:>6} {tails:>6} {post_mean:>16.4f} {post_std:>14.4f}")

        # Plot
        pdf_vals = posterior.pdf(theta_range)
        axes[idx].plot(theta_range, pdf_vals, 'b-', linewidth=2)
        axes[idx].fill_between(theta_range, pdf_vals, alpha=0.3, color='steelblue')
        axes[idx].axvline(x=true_theta, color='red', linestyle='--', linewidth=1.5, label=f'True = {true_theta}')
        axes[idx].axvline(x=post_mean, color='green', linestyle=':', linewidth=1.5, label=f'Mean = {post_mean:.3f}')
        axes[idx].set_title(f'n = {n} flips\n{posterior}', fontsize=10)
        axes[idx].set_xlim(0, 1)
        if idx >= 6:
            axes[idx].set_xlabel('theta')
        if idx % 3 == 0:
            axes[idx].set_ylabel('Density')
        axes[idx].legend(fontsize=7)

    plt.suptitle('Bayesian Coin Flip: Posterior Concentrates on True Value', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "03_bayesian_coin_flip.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {os.path.join(PLOT_DIR, '03_bayesian_coin_flip.png')}")

    # Demonstrate effect of different priors
    print("\n--- Effect of Prior Strength ---")
    heads, tails = 7, 3  # Same data: 7 heads out of 10
    priors = [
        ("Uniform (no opinion)", BetaDistribution(1, 1)),
        ("Weak fair prior", BetaDistribution(5, 5)),
        ("Strong fair prior", BetaDistribution(50, 50)),
        ("Very strong fair prior", BetaDistribution(200, 200)),
    ]

    print(f"\nData: {heads} heads, {tails} tails (MLE = {heads/(heads+tails):.2f})")
    print(f"{'Prior':>30} {'Prior Mean':>12} {'Posterior Mean':>16}")
    print("-" * 62)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['blue', 'green', 'orange', 'red']

    for (name, prior_dist), color in zip(priors, colors):
        posterior = prior_dist.update(heads, tails)
        print(f"{name:>30} {prior_dist.mean():>12.4f} {posterior.mean():>16.4f}")

        pdf_vals = posterior.pdf(theta_range)
        ax.plot(theta_range, pdf_vals, color=color, linewidth=2,
                label=f'{name}: post mean = {posterior.mean():.3f}')

    ax.axvline(x=true_theta, color='black', linestyle='--', linewidth=1.5, label=f'True theta = {true_theta}')
    ax.axvline(x=heads / (heads + tails), color='gray', linestyle=':', linewidth=1.5, label=f'MLE = {heads/(heads+tails):.2f}')
    ax.set_xlabel('theta')
    ax.set_ylabel('Posterior Density')
    ax.set_title('Effect of Prior Strength on Posterior (10 observations: 7H, 3T)')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "04_prior_strength_effect.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {os.path.join(PLOT_DIR, '04_prior_strength_effect.png')}")
    print("\nKey insight: With little data, the prior dominates. With much data, the prior washes out.")


# ===========================================================================
# Section 3: Maximum Likelihood Estimation
# ===========================================================================

def demo_mle():
    """
    Demonstrate MLE for Gaussian and Bernoulli, showing convergence with more data.
    """
    print("\n" + "=" * 70)
    print("SECTION 3: Maximum Likelihood Estimation from Scratch")
    print("=" * 70)

    # --- MLE for Gaussian ---
    print("\n--- MLE for Gaussian Parameters ---")
    true_mu = 3.0
    true_sigma = 2.0
    print(f"True parameters: mu = {true_mu}, sigma = {true_sigma}")

    # Generate data from the true distribution
    true_dist = Gaussian(true_mu, true_sigma)
    all_data = true_dist.sample(10000)

    # MLE formulas (derived in theory.md):
    #   mu_hat = (1/n) * sum(x_i)           — sample mean
    #   sigma2_hat = (1/n) * sum((x_i - mu_hat)^2) — sample variance (biased)

    sample_sizes = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
    mu_estimates = []
    sigma_estimates = []
    log_likelihoods = []

    print(f"\n{'n':>8} {'mu_MLE':>10} {'sigma_MLE':>12} {'Log-Likelihood':>16}")
    print("-" * 50)

    for n in sample_sizes:
        data = all_data[:n]

        # MLE estimates (closed-form)
        mu_hat = data.mean()
        sigma2_hat = ((data - mu_hat) ** 2).mean()  # Biased MLE estimator
        sigma_hat = np.sqrt(sigma2_hat)

        # Compute log-likelihood at MLE
        mle_dist = Gaussian(mu_hat, sigma_hat)
        ll = mle_dist.log_prob(data).sum()

        mu_estimates.append(mu_hat)
        sigma_estimates.append(sigma_hat)
        log_likelihoods.append(ll / n)  # Normalize by n for comparison

        print(f"{n:>8} {mu_hat:>10.4f} {sigma_hat:>12.4f} {ll / n:>16.4f}")

    # Plot convergence
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].semilogx(sample_sizes, mu_estimates, 'bo-', linewidth=2, markersize=5)
    axes[0].axhline(y=true_mu, color='red', linestyle='--', linewidth=2, label=f'True mu = {true_mu}')
    axes[0].set_xlabel('Sample Size (n)')
    axes[0].set_ylabel('Estimated mu')
    axes[0].set_title('MLE for Mean: Convergence')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(sample_sizes, sigma_estimates, 'go-', linewidth=2, markersize=5)
    axes[1].axhline(y=true_sigma, color='red', linestyle='--', linewidth=2, label=f'True sigma = {true_sigma}')
    axes[1].set_xlabel('Sample Size (n)')
    axes[1].set_ylabel('Estimated sigma')
    axes[1].set_title('MLE for Std Dev: Convergence')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogx(sample_sizes, log_likelihoods, 'mo-', linewidth=2, markersize=5)
    axes[2].set_xlabel('Sample Size (n)')
    axes[2].set_ylabel('Avg Log-Likelihood')
    axes[2].set_title('Average Log-Likelihood per Sample')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('MLE for Gaussian: Estimates Converge to True Parameters', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "05_mle_gaussian_convergence.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {os.path.join(PLOT_DIR, '05_mle_gaussian_convergence.png')}")

    # --- MLE for Bernoulli ---
    print("\n--- MLE for Bernoulli Parameter ---")
    true_theta = 0.73
    print(f"True parameter: theta = {true_theta}")

    all_flips = (np.random.uniform(size=10000) < true_theta).astype(float)

    theta_estimates = []

    print(f"\n{'n':>8} {'theta_MLE':>12} {'|Error|':>10} {'Log-Likelihood/n':>18}")
    print("-" * 52)

    for n in sample_sizes:
        data = all_flips[:n]

        # MLE: theta_hat = k/n (sample proportion)
        theta_hat = data.mean()
        theta_estimates.append(theta_hat)

        # Log-likelihood
        b = Bernoulli(theta_hat)
        ll = b.log_prob(data).sum()

        print(f"{n:>8} {theta_hat:>12.4f} {abs(theta_hat - true_theta):>10.4f} {ll / n:>18.4f}")

    # Visualization: likelihood function for different sample sizes
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    theta_range = np.linspace(0.01, 0.99, 500)

    for n, color in [(5, 'blue'), (20, 'green'), (100, 'orange'), (1000, 'red')]:
        data = all_flips[:n]
        k = data.sum()
        # Log-likelihood as function of theta
        ll_func = k * np.log(theta_range) + (n - k) * np.log(1 - theta_range)
        # Normalize for visualization
        ll_func_norm = ll_func - ll_func.max()
        axes[0].plot(theta_range, np.exp(ll_func_norm), linewidth=2,
                     color=color, label=f'n={n} (MLE={k / n:.3f})')

    axes[0].axvline(x=true_theta, color='black', linestyle='--', label=f'True = {true_theta}')
    axes[0].set_xlabel('theta')
    axes[0].set_ylabel('Normalized Likelihood')
    axes[0].set_title('Bernoulli Likelihood Function')
    axes[0].legend(fontsize=9)

    axes[1].semilogx(sample_sizes, theta_estimates, 'ro-', linewidth=2, markersize=5)
    axes[1].axhline(y=true_theta, color='black', linestyle='--', linewidth=2, label=f'True = {true_theta}')
    axes[1].set_xlabel('Sample Size (n)')
    axes[1].set_ylabel('Estimated theta')
    axes[1].set_title('MLE for Bernoulli: Convergence')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "06_mle_bernoulli.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {os.path.join(PLOT_DIR, '06_mle_bernoulli.png')}")

    # --- Show MLE connection to cross-entropy loss ---
    print("\n--- MLE = Minimizing Cross-Entropy ---")
    print("For a Bernoulli model with sigmoid output:")
    print("  Negative log-likelihood = -[y*log(p) + (1-y)*log(1-p)]")
    print("  This is EXACTLY binary cross-entropy loss.")
    print()
    print("For a Categorical model with softmax output:")
    print("  Negative log-likelihood = -sum_k [y_k * log(p_k)]")
    print("  This is EXACTLY categorical cross-entropy loss.")
    print()
    print("CONCLUSION: Training by minimizing cross-entropy IS doing MLE.")


# ===========================================================================
# Section 4: Monte Carlo Estimation of Pi
# ===========================================================================

def demo_monte_carlo_pi():
    """
    Estimate pi using Monte Carlo sampling.

    Method: Sample (x, y) uniformly from [-1, 1]^2.
    The ratio of points inside the unit circle to total points equals pi/4.
    Therefore pi = 4 * (points inside) / (total points).

    This demonstrates the core Monte Carlo idea: approximate integrals with samples.
    """
    print("\n" + "=" * 70)
    print("SECTION 4: Monte Carlo Estimation of Pi")
    print("=" * 70)

    sample_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    estimates = []
    errors = []

    # Pre-generate all samples at once for efficiency
    max_n = max(sample_sizes)
    x_all = np.random.uniform(-1, 1, max_n)
    y_all = np.random.uniform(-1, 1, max_n)
    inside_all = (x_all ** 2 + y_all ** 2) <= 1.0

    print(f"\n{'Samples':>12} {'Estimate':>12} {'Error':>12} {'|Error|':>12}")
    print("-" * 52)

    for n in sample_sizes:
        inside_count = inside_all[:n].sum()
        pi_estimate = 4.0 * inside_count / n
        error = pi_estimate - np.pi
        estimates.append(pi_estimate)
        errors.append(abs(error))
        print(f"{n:>12,} {pi_estimate:>12.6f} {error:>+12.6f} {abs(error):>12.6f}")

    print(f"\n{'True pi':>12} {np.pi:>12.6f}")
    print(f"\nMonte Carlo error decreases as O(1/sqrt(n)):")
    print(f"  100 samples -> ~{1/np.sqrt(100):.3f} expected error")
    print(f"  10000 samples -> ~{1/np.sqrt(10000):.3f} expected error")
    print(f"  1000000 samples -> ~{1/np.sqrt(1000000):.4f} expected error")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Scatter plot showing the method
    n_show = 5000
    inside = inside_all[:n_show]
    axes[0].scatter(x_all[:n_show][inside], y_all[:n_show][inside],
                    s=0.5, alpha=0.5, color='steelblue', label='Inside')
    axes[0].scatter(x_all[:n_show][~inside], y_all[:n_show][~inside],
                    s=0.5, alpha=0.5, color='coral', label='Outside')
    theta = np.linspace(0, 2 * np.pi, 200)
    axes[0].plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1.5)
    axes[0].set_aspect('equal')
    axes[0].set_title(f'Monte Carlo Pi (n={n_show})')
    axes[0].legend(markerscale=10)

    # Convergence plot
    axes[1].semilogx(sample_sizes, estimates, 'bo-', linewidth=2, markersize=5)
    axes[1].axhline(y=np.pi, color='red', linestyle='--', linewidth=2, label=f'True pi = {np.pi:.6f}')
    axes[1].set_xlabel('Number of Samples')
    axes[1].set_ylabel('Estimated Pi')
    axes[1].set_title('Convergence of Estimate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Error vs 1/sqrt(n) scaling
    axes[2].loglog(sample_sizes, errors, 'ro-', linewidth=2, markersize=5, label='Actual |error|')
    # Theoretical O(1/sqrt(n)) reference line
    ref_sizes = np.array(sample_sizes, dtype=float)
    ref_line = 2.0 / np.sqrt(ref_sizes)  # scaled reference
    axes[2].loglog(sample_sizes, ref_line, 'k--', linewidth=1.5, label=r'O(1/$\sqrt{n}$) reference')
    axes[2].set_xlabel('Number of Samples')
    axes[2].set_ylabel('Absolute Error')
    axes[2].set_title('Error Scaling (should follow 1/sqrt(n))')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "07_monte_carlo_pi.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {os.path.join(PLOT_DIR, '07_monte_carlo_pi.png')}")


# ===========================================================================
# Section 5: Central Limit Theorem Demonstration
# ===========================================================================

def demo_central_limit_theorem():
    """
    The CLT in action: show that the sum of ANY distribution converges to Gaussian.

    We demonstrate with several very non-Gaussian distributions:
    1. Uniform (flat)
    2. Exponential (heavily skewed)
    3. Bernoulli (discrete, two-point)
    4. A bizarre custom distribution (bimodal spikes)

    For each, we draw n samples, sum them, repeat many times, and show
    the resulting distribution approaches Gaussian.
    """
    print("\n" + "=" * 70)
    print("SECTION 5: Central Limit Theorem Demonstration")
    print("=" * 70)

    n_experiments = 50000  # Number of times we compute the sum
    sum_sizes = [1, 2, 5, 10, 30, 100]  # How many terms in each sum

    # Define our source distributions (as sampling functions + descriptions)
    distributions = [
        ("Uniform(0,1)", lambda n: np.random.uniform(0, 1, n), 0.5, 1.0 / 12),
        ("Exponential(1)", lambda n: np.random.exponential(1.0, n), 1.0, 1.0),
        ("Bernoulli(0.3)", lambda n: (np.random.uniform(0, 1, n) < 0.3).astype(float), 0.3, 0.21),
        ("Bimodal spikes", lambda n: np.where(np.random.uniform(0, 1, n) < 0.5,
                                               np.random.normal(-3, 0.3, n),
                                               np.random.normal(3, 0.3, n)),
         0.0, 9.09),  # Approximate mean and variance
    ]

    fig, axes = plt.subplots(len(distributions), len(sum_sizes), figsize=(20, 12))

    for row, (name, sampler, true_mean, true_var) in enumerate(distributions):
        print(f"\n--- {name} ---")
        print(f"True mean: {true_mean:.3f}, True variance: {true_var:.3f}")

        for col, n in enumerate(sum_sizes):
            # Draw n samples, n_experiments times; compute standardized mean each time
            all_samples = sampler(n * n_experiments).reshape(n_experiments, n)
            sample_means = all_samples.mean(axis=1)

            # Standardize: Z = (X_bar - mu) / (sigma / sqrt(n))
            if true_var > 0 and n > 0:
                standardized = (sample_means - true_mean) / np.sqrt(true_var / n)
            else:
                standardized = sample_means

            # Compare to standard Gaussian
            ax = axes[row, col]
            ax.hist(standardized, bins=80, density=True, alpha=0.6, color='steelblue')

            # Overlay Gaussian PDF
            x_range = np.linspace(-4, 4, 300)
            gaussian_pdf = Gaussian(0, 1).pdf(x_range)
            ax.plot(x_range, gaussian_pdf, 'r-', linewidth=2)

            if row == 0:
                ax.set_title(f'n = {n}', fontsize=11)
            if col == 0:
                ax.set_ylabel(name, fontsize=10)
            ax.set_xlim(-4, 4)
            ax.set_ylim(0, 0.55)

            # Report how Gaussian-like the result is (using Kolmogorov-Smirnov-like metric)
            # Simple metric: compare empirical CDF to Gaussian CDF at several points
            if col == len(sum_sizes) - 1 and row == 0:
                print(f"  n={n:>3}: sample std of standardized = {standardized.std():.4f} (should be ~1.0)")

    plt.suptitle('Central Limit Theorem: Any Distribution -> Gaussian\n(Red = N(0,1) reference)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "08_central_limit_theorem.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Summary statistics
    print(f"\nSummary: Standardized mean of n samples from each distribution")
    print(f"{'Distribution':>20} {'n=1 std':>10} {'n=5 std':>10} {'n=30 std':>10} {'n=100 std':>10}")
    print("-" * 64)
    for name, sampler, true_mean, true_var in distributions:
        stds = []
        for n in [1, 5, 30, 100]:
            all_samples = sampler(n * n_experiments).reshape(n_experiments, n)
            sample_means = all_samples.mean(axis=1)
            if true_var > 0 and n > 0:
                standardized = (sample_means - true_mean) / np.sqrt(true_var / n)
            else:
                standardized = sample_means
            stds.append(standardized.std())
        print(f"{name:>20} {stds[0]:>10.4f} {stds[1]:>10.4f} {stds[2]:>10.4f} {stds[3]:>10.4f}")
    print("(All values should approach 1.0 as n increases — that's the CLT)")

    print(f"\nPlot saved: {os.path.join(PLOT_DIR, '08_central_limit_theorem.png')}")


# ===========================================================================
# Section 6: Gaussian Mixture Model via EM Algorithm
# ===========================================================================

class GaussianMixtureModel:
    """
    Gaussian Mixture Model trained with the Expectation-Maximization algorithm.

    The model:
        p(x) = sum_k  pi_k * N(x | mu_k, sigma_k^2)

    where pi_k are mixing coefficients (sum to 1), mu_k are means, sigma_k are stds.

    EM alternates between:
        E-step: Compute responsibilities (which component likely generated each point)
        M-step: Update parameters using weighted statistics

    This is the fundamental algorithm behind many clustering and density estimation
    methods. It generalizes k-means (which is the limit as variances -> 0).
    """

    def __init__(self, K, max_iter=200, tol=1e-6):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol

        # Parameters (initialized in fit)
        self.pi = None      # Mixing coefficients: (K,)
        self.mu = None      # Means: (K,)
        self.sigma = None   # Standard deviations: (K,)
        self.log_likelihoods = []

    def _initialize(self, X):
        """
        Initialize parameters using a simple heuristic:
        - Means: random data points
        - Variances: overall data variance
        - Weights: uniform
        """
        n = len(X)

        # Pick K random data points as initial means
        indices = np.random.choice(n, self.K, replace=False)
        self.mu = X[indices].copy()

        # Initialize all components with the data's overall std
        overall_std = X.std()
        self.sigma = np.full(self.K, overall_std)

        # Uniform mixing weights
        self.pi = np.full(self.K, 1.0 / self.K)

    def _e_step(self, X):
        """
        E-step: Compute responsibilities.

        r_{nk} = pi_k * N(x_n | mu_k, sigma_k^2) / sum_j [pi_j * N(x_n | mu_j, sigma_j^2)]

        r_{nk} = "probability that point n was generated by component k"

        We work in log-space for numerical stability.
        """
        n = len(X)
        log_resp = np.zeros((n, self.K))

        for k in range(self.K):
            g = Gaussian(self.mu[k], self.sigma[k])
            log_resp[:, k] = np.log(self.pi[k] + 1e-300) + g.log_prob(X)

        # Log-sum-exp trick for numerical stability
        log_resp_max = log_resp.max(axis=1, keepdims=True)
        log_resp_shifted = log_resp - log_resp_max
        log_normalizer = log_resp_max.squeeze() + np.log(np.exp(log_resp_shifted).sum(axis=1))

        # Normalize to get responsibilities
        resp = np.exp(log_resp - log_normalizer[:, np.newaxis])

        return resp, log_normalizer

    def _m_step(self, X, resp):
        """
        M-step: Update parameters using weighted statistics.

        N_k = sum_n r_{nk}                           (effective number of points in cluster k)
        mu_k = (1/N_k) * sum_n r_{nk} * x_n          (weighted mean)
        sigma_k^2 = (1/N_k) * sum_n r_{nk} * (x_n - mu_k)^2  (weighted variance)
        pi_k = N_k / N                               (mixing coefficient)
        """
        n = len(X)

        for k in range(self.K):
            N_k = resp[:, k].sum()

            if N_k < 1e-10:
                # Empty cluster — reinitialize from a random data point
                self.mu[k] = X[np.random.randint(n)]
                self.sigma[k] = X.std()
                self.pi[k] = 1.0 / self.K
                continue

            # Update mean
            self.mu[k] = (resp[:, k] * X).sum() / N_k

            # Update variance (with floor to prevent collapse)
            variance = (resp[:, k] * (X - self.mu[k]) ** 2).sum() / N_k
            self.sigma[k] = np.sqrt(max(variance, 1e-6))

            # Update mixing coefficient
            self.pi[k] = N_k / n

        # Renormalize pi (should already sum to 1, but just in case)
        self.pi /= self.pi.sum()

    def fit(self, X):
        """Run EM algorithm."""
        X = np.asarray(X, dtype=np.float64)
        self._initialize(X)
        self.log_likelihoods = []

        for iteration in range(self.max_iter):
            # E-step
            resp, log_normalizer = self._e_step(X)

            # Compute log-likelihood
            ll = log_normalizer.sum()
            self.log_likelihoods.append(ll)

            # Check convergence
            if iteration > 0:
                delta = abs(ll - self.log_likelihoods[-2])
                if delta < self.tol:
                    break

            # M-step
            self._m_step(X, resp)

        return self

    def pdf(self, x):
        """Evaluate the mixture PDF at x."""
        x = np.asarray(x, dtype=np.float64)
        density = np.zeros_like(x)
        for k in range(self.K):
            g = Gaussian(self.mu[k], self.sigma[k])
            density += self.pi[k] * g.pdf(x)
        return density

    def predict(self, X):
        """Assign each point to its most likely component."""
        resp, _ = self._e_step(X)
        return resp.argmax(axis=1)

    def sample(self, n=1):
        """Sample from the mixture model."""
        # First, choose which component to sample from
        components = np.random.choice(self.K, size=n, p=self.pi)
        samples = np.zeros(n)
        for k in range(self.K):
            mask = components == k
            count = mask.sum()
            if count > 0:
                g = Gaussian(self.mu[k], self.sigma[k])
                samples[mask] = g.sample(count)
        return samples


def demo_gmm():
    """
    Demonstrate the EM algorithm for Gaussian Mixture Models.

    1. Generate data from a known mixture
    2. Fit a GMM using our EM implementation
    3. Visualize the results and convergence
    """
    print("\n" + "=" * 70)
    print("SECTION 6: Gaussian Mixture Model via EM Algorithm")
    print("=" * 70)

    # --- Generate synthetic data from a 3-component mixture ---
    true_params = {
        'pi': [0.3, 0.5, 0.2],
        'mu': [-3.0, 1.0, 5.0],
        'sigma': [0.8, 1.2, 0.6],
    }

    n_total = 2000

    print("\nTrue mixture parameters:")
    for k in range(3):
        print(f"  Component {k}: pi={true_params['pi'][k]:.1f}, "
              f"mu={true_params['mu'][k]:.1f}, sigma={true_params['sigma'][k]:.1f}")

    # Generate data
    component_assignments = np.random.choice(3, size=n_total, p=true_params['pi'])
    data = np.zeros(n_total)
    for k in range(3):
        mask = component_assignments == k
        count = mask.sum()
        data[mask] = Gaussian(true_params['mu'][k], true_params['sigma'][k]).sample(count)

    # Shuffle (EM shouldn't depend on order, but let's be proper)
    np.random.shuffle(data)

    print(f"\nGenerated {n_total} data points from 3-component mixture.")
    print(f"Data range: [{data.min():.2f}, {data.max():.2f}], mean: {data.mean():.3f}, std: {data.std():.3f}")

    # --- Fit GMM ---
    print("\nFitting GMM with K=3 components...")
    gmm = GaussianMixtureModel(K=3, max_iter=200, tol=1e-8)
    gmm.fit(data)

    print(f"EM converged after {len(gmm.log_likelihoods)} iterations.")

    # Sort components by mean for easy comparison
    order = np.argsort(gmm.mu)

    print("\nRecovered parameters (sorted by mean):")
    print(f"{'':>12} {'pi':>8} {'mu':>10} {'sigma':>10}")
    print("-" * 42)
    for i, k in enumerate(order):
        print(f"  Component {i}: {gmm.pi[k]:>7.3f} {gmm.mu[k]:>10.3f} {gmm.sigma[k]:>10.3f}")
    print(f"  True      0: {true_params['pi'][0]:>7.3f} {true_params['mu'][0]:>10.3f} {true_params['sigma'][0]:>10.3f}")
    print(f"  True      1: {true_params['pi'][1]:>7.3f} {true_params['mu'][1]:>10.3f} {true_params['sigma'][1]:>10.3f}")
    print(f"  True      2: {true_params['pi'][2]:>7.3f} {true_params['mu'][2]:>10.3f} {true_params['sigma'][2]:>10.3f}")

    # --- Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Data histogram with fitted density
    x_range = np.linspace(data.min() - 1, data.max() + 1, 1000)

    axes[0, 0].hist(data, bins=80, density=True, alpha=0.5, color='gray', label='Data')

    # Plot individual components
    colors = ['blue', 'green', 'red']
    for i, k in enumerate(order):
        g = Gaussian(gmm.mu[k], gmm.sigma[k])
        component_pdf = gmm.pi[k] * g.pdf(x_range)
        axes[0, 0].plot(x_range, component_pdf, color=colors[i], linewidth=2,
                        linestyle='--', label=f'Component {i} (pi={gmm.pi[k]:.2f})')

    # Plot mixture density
    mixture_pdf = gmm.pdf(x_range)
    axes[0, 0].plot(x_range, mixture_pdf, 'k-', linewidth=2.5, label='Mixture PDF')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Fitted Gaussian Mixture Model')
    axes[0, 0].legend(fontsize=9)

    # 2. True density overlay for comparison
    axes[0, 1].hist(data, bins=80, density=True, alpha=0.5, color='gray', label='Data')

    # True density
    true_density = np.zeros_like(x_range)
    for k in range(3):
        g = Gaussian(true_params['mu'][k], true_params['sigma'][k])
        true_density += true_params['pi'][k] * g.pdf(x_range)

    axes[0, 1].plot(x_range, true_density, 'r-', linewidth=2.5, label='True density')
    axes[0, 1].plot(x_range, mixture_pdf, 'b--', linewidth=2.5, label='Fitted density')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('True vs Fitted Density')
    axes[0, 1].legend()

    # 3. Log-likelihood convergence
    axes[1, 0].plot(gmm.log_likelihoods, 'b-', linewidth=2)
    axes[1, 0].set_xlabel('EM Iteration')
    axes[1, 0].set_ylabel('Log-Likelihood')
    axes[1, 0].set_title('EM Convergence (Log-Likelihood Must Increase)')
    axes[1, 0].grid(True, alpha=0.3)

    # Verify monotonic increase
    ll_array = np.array(gmm.log_likelihoods)
    diffs = np.diff(ll_array)
    is_monotonic = np.all(diffs >= -1e-6)  # Allow tiny numerical errors
    axes[1, 0].annotate(f'Monotonically increasing: {is_monotonic}',
                        xy=(0.5, 0.05), xycoords='axes fraction',
                        fontsize=10, ha='center',
                        bbox=dict(boxstyle='round', facecolor='lightgreen' if is_monotonic else 'lightsalmon'))

    # 4. Cluster assignments
    assignments = gmm.predict(data)
    for i, k in enumerate(order):
        mask = assignments == k
        axes[1, 1].scatter(data[mask], np.random.normal(0, 0.02, mask.sum()),
                          s=3, alpha=0.3, color=colors[i], label=f'Cluster {i} (n={mask.sum()})')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('(jittered for visibility)')
    axes[1, 1].set_title('Data Points Colored by Cluster Assignment')
    axes[1, 1].legend(fontsize=9)

    plt.suptitle('Gaussian Mixture Model: EM Algorithm from Scratch', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "09_gaussian_mixture_model.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {os.path.join(PLOT_DIR, '09_gaussian_mixture_model.png')}")

    # --- Show GMM with wrong K ---
    print("\n--- What Happens with Wrong Number of Components? ---")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, K_test in enumerate([2, 3, 5]):
        gmm_test = GaussianMixtureModel(K=K_test, max_iter=200, tol=1e-8)
        gmm_test.fit(data)

        ax = axes[idx]
        ax.hist(data, bins=80, density=True, alpha=0.4, color='gray')
        ax.plot(x_range, gmm_test.pdf(x_range), 'b-', linewidth=2.5, label='Fitted')
        ax.plot(x_range, true_density, 'r--', linewidth=2, label='True')
        ax.set_title(f'K = {K_test} components\nFinal LL = {gmm_test.log_likelihoods[-1]:.1f}')
        ax.set_xlabel('x')
        ax.legend(fontsize=9)

        print(f"K={K_test}: Final log-likelihood = {gmm_test.log_likelihoods[-1]:.2f}, "
              f"iterations = {len(gmm_test.log_likelihoods)}")

    plt.suptitle('GMM with Different K: Underfitting vs Overfitting', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "10_gmm_model_selection.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {os.path.join(PLOT_DIR, '10_gmm_model_selection.png')}")

    # --- Sample from the fitted model ---
    print("\n--- Sampling from Fitted GMM ---")
    gmm_samples = gmm.sample(5000)
    print(f"Generated 5000 samples from fitted GMM")
    print(f"Sample mean: {gmm_samples.mean():.3f}, Sample std: {gmm_samples.std():.3f}")
    print(f"Data mean:   {data.mean():.3f}, Data std:   {data.std():.3f}")


# ===========================================================================
# Summary and ML Connections
# ===========================================================================

def print_ml_connections():
    """Print the connections between probability theory and ML practice."""
    print("\n" + "=" * 70)
    print("SUMMARY: Probability Theory -> Machine Learning")
    print("=" * 70)
    print("""
    Concept                  | ML Manifestation
    -------------------------+------------------------------------------
    Bernoulli distribution   | Sigmoid output for binary classification
    Categorical distribution | Softmax output for multi-class classification
    Gaussian distribution    | Noise model for regression, weight init,
                             |   latent spaces in VAEs
    Log-likelihood           | -1 * loss function
    MLE                      | Training by minimizing cross-entropy/MSE
    MAP                      | Training with L1/L2 regularization
    Bayesian posterior       | Full uncertainty over weights (BNNs)
    Bayes' theorem           | Every update rule is posterior proportional
                             |   to likelihood times prior
    CLT                      | Why Gaussian assumptions are reasonable
    Monte Carlo              | Estimating gradients, ELBO, policy gradients
    GMM / EM                 | Clustering, density estimation, latent
                             |   variable models (VAEs generalize this)

    The key insight: there is no separation between "probability theory"
    and "machine learning". ML *is* applied probability theory. Every
    architecture, loss function, and training procedure has a precise
    probabilistic interpretation. Understanding the probability gives
    you the power to invent new methods, not just use existing ones.
    """)


# ===========================================================================
# Main: Run All Demonstrations
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  PROBABILITY & STATISTICS: FROM-SCRATCH IMPLEMENTATION")
    print("  All code uses NumPy only. No scipy, no sklearn.")
    print("=" * 70)

    demo_distributions()
    demo_bayes_coin_flip()
    demo_mle()
    demo_monte_carlo_pi()
    demo_central_limit_theorem()
    demo_gmm()
    print_ml_connections()

    print("\n" + "=" * 70)
    print(f"All plots saved to: {PLOT_DIR}/")
    print("=" * 70)

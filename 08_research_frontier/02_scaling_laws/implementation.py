"""
Scaling Laws: From Scratch Implementation
============================================

Neural scaling laws describe how model performance improves with scale.

Key relationships (Kaplan et al., 2020; Chinchilla, 2022):
    L(N) = a * N^(-α)     loss as function of parameters
    L(D) = b * D^(-β)     loss as function of data
    L(C) = c * C^(-γ)     loss as function of compute

Concepts:
    1. Power law fitting
    2. Compute-optimal scaling (Chinchilla)
    3. Emergent abilities analysis
    4. Scaling prediction extrapolation

All code uses only NumPy. No frameworks.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SAVE_DIR = Path(__file__).parent / "plots"
SAVE_DIR.mkdir(exist_ok=True)

np.random.seed(42)


# =============================================================================
# PART 1: POWER LAW FITTING
# =============================================================================

def fit_power_law(x: np.ndarray, y: np.ndarray):
    """
    Fit y = a * x^b using log-linear regression.

    In log space: log(y) = log(a) + b * log(x)
    """
    log_x = np.log(x)
    log_y = np.log(y)

    # Linear regression in log space
    n = len(x)
    b = (n * np.sum(log_x * log_y) - np.sum(log_x) * np.sum(log_y)) / \
        (n * np.sum(log_x ** 2) - np.sum(log_x) ** 2)
    log_a = (np.sum(log_y) - b * np.sum(log_x)) / n
    a = np.exp(log_a)

    # R² in log space
    y_pred = a * x ** b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot

    return a, b, r_squared


def predict_power_law(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Predict y = a * x^b."""
    return a * x ** b


# =============================================================================
# PART 2: SIMULATE SCALING EXPERIMENTS
# =============================================================================

def simulate_model_scaling(param_counts: np.ndarray, data_size: int = 100000,
                           noise_level: float = 0.02) -> np.ndarray:
    """
    Simulate loss vs model size (parameters).

    Based on empirical observation: L(N) ≈ a * N^(-0.076)
    (Kaplan et al., 2020 for language models)
    """
    a, alpha = 3.0, -0.076
    losses = a * param_counts.astype(float) ** alpha
    losses += np.random.randn(len(losses)) * noise_level
    return np.clip(losses, 0.1, 10.0)


def simulate_data_scaling(data_sizes: np.ndarray, model_params: int = 1000000,
                          noise_level: float = 0.02) -> np.ndarray:
    """
    Simulate loss vs data size.

    L(D) ≈ b * D^(-0.095)
    """
    b, beta = 5.0, -0.095
    losses = b * data_sizes.astype(float) ** beta
    losses += np.random.randn(len(losses)) * noise_level
    return np.clip(losses, 0.1, 10.0)


def simulate_compute_scaling(flops: np.ndarray,
                              noise_level: float = 0.02) -> np.ndarray:
    """
    Simulate loss vs compute (FLOPs).

    L(C) ≈ c * C^(-0.050)
    """
    c, gamma = 10.0, -0.050
    losses = c * flops.astype(float) ** gamma
    losses += np.random.randn(len(losses)) * noise_level
    return np.clip(losses, 0.1, 10.0)


# =============================================================================
# PART 3: CHINCHILLA OPTIMAL SCALING
# =============================================================================

def chinchilla_optimal(compute_budget: float, a: float = 406.4,
                       b: float = 410.7, alpha: float = 0.34,
                       beta: float = 0.28):
    """
    Compute-optimal model and data size (Hoffmann et al., 2022).

    Given a compute budget C (in FLOPs), find optimal N (params) and D (data):
        C ≈ 6 * N * D  (approximate compute formula)
        N_opt ∝ C^(beta / (alpha + beta))
        D_opt ∝ C^(alpha / (alpha + beta))

    Key insight: Chinchilla found most LLMs are undertrained.
    GPT-3 (175B params, 300B tokens) should have been ~70B params, 1.4T tokens.
    """
    # Compute-optimal allocation
    N_opt = a * compute_budget ** (beta / (alpha + beta))
    D_opt = b * compute_budget ** (alpha / (alpha + beta))
    return N_opt, D_opt


# =============================================================================
# PART 4: EMERGENT ABILITIES
# =============================================================================

def simulate_emergent_ability(model_sizes: np.ndarray,
                               threshold: float = 1e8) -> np.ndarray:
    """
    Simulate emergent abilities that appear suddenly at scale.

    Below threshold: near-random performance
    Above threshold: sharp improvement (phase transition)
    """
    performance = np.zeros(len(model_sizes))
    for i, size in enumerate(model_sizes):
        if size < threshold:
            performance[i] = 0.1 + np.random.rand() * 0.1
        else:
            # Sigmoid-like emergence
            x = np.log10(size / threshold)
            performance[i] = 0.1 + 0.8 / (1 + np.exp(-5 * x))
            performance[i] += np.random.rand() * 0.05
    return np.clip(performance, 0, 1)


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate scaling law analysis."""
    print("=" * 70)
    print("NEURAL SCALING LAWS")
    print("=" * 70)

    # 1. Loss vs Parameters
    params = np.logspace(3, 9, 20)  # 1K to 1B parameters
    losses_params = simulate_model_scaling(params)
    a_n, b_n, r2_n = fit_power_law(params, losses_params)
    print(f"\nLoss vs Parameters: L = {a_n:.3f} * N^({b_n:.4f}), R² = {r2_n:.4f}")

    # 2. Loss vs Data
    data_sizes = np.logspace(3, 8, 20)
    losses_data = simulate_data_scaling(data_sizes)
    a_d, b_d, r2_d = fit_power_law(data_sizes, losses_data)
    print(f"Loss vs Data:       L = {a_d:.3f} * D^({b_d:.4f}), R² = {r2_d:.4f}")

    # 3. Loss vs Compute
    flops = np.logspace(12, 24, 20)
    losses_compute = simulate_compute_scaling(flops)
    a_c, b_c, r2_c = fit_power_law(flops, losses_compute)
    print(f"Loss vs Compute:    L = {a_c:.3f} * C^({b_c:.4f}), R² = {r2_c:.4f}")

    # 4. Chinchilla optimal
    print(f"\nChinchilla Optimal Scaling:")
    for compute in [1e18, 1e20, 1e22, 1e24]:
        N_opt, D_opt = chinchilla_optimal(compute)
        print(f"  C = {compute:.0e} FLOPs → N = {N_opt:.2e} params, "
              f"D = {D_opt:.2e} tokens, ratio = {D_opt/N_opt:.1f}")

    # 5. Emergent abilities
    model_sizes_em = np.logspace(6, 11, 30)
    emergence = simulate_emergent_ability(model_sizes_em, threshold=1e8)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss vs Params
    axes[0, 0].scatter(params, losses_params, c='blue', alpha=0.6)
    p_fit = np.logspace(3, 10, 100)
    axes[0, 0].plot(p_fit, predict_power_law(p_fit, a_n, b_n), 'r--',
                     label=f'L = {a_n:.2f}·N^{b_n:.3f}')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xlabel('Parameters')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss vs Model Size')
    axes[0, 0].legend()

    # Loss vs Data
    axes[0, 1].scatter(data_sizes, losses_data, c='green', alpha=0.6)
    d_fit = np.logspace(3, 9, 100)
    axes[0, 1].plot(d_fit, predict_power_law(d_fit, a_d, b_d), 'r--',
                     label=f'L = {a_d:.2f}·D^{b_d:.3f}')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_xlabel('Data Size')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Loss vs Data Size')
    axes[0, 1].legend()

    # Loss vs Compute
    axes[1, 0].scatter(flops, losses_compute, c='orange', alpha=0.6)
    c_fit = np.logspace(12, 25, 100)
    axes[1, 0].plot(c_fit, predict_power_law(c_fit, a_c, b_c), 'r--',
                     label=f'L = {a_c:.2f}·C^{b_c:.3f}')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xlabel('Compute (FLOPs)')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Loss vs Compute')
    axes[1, 0].legend()

    # Emergent abilities
    axes[1, 1].plot(model_sizes_em, emergence, 'bo-', markersize=4)
    axes[1, 1].axvline(1e8, color='r', linestyle='--', alpha=0.5, label='Threshold')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_xlabel('Model Size')
    axes[1, 1].set_ylabel('Task Performance')
    axes[1, 1].set_title('Emergent Abilities')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(SAVE_DIR / "scaling_laws.png", dpi=100)
    plt.close()
    print(f"\nPlot saved to {SAVE_DIR / 'scaling_laws.png'}")


if __name__ == "__main__":
    demo()

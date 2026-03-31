"""
Exponential Moving Average & Model Averaging: From Scratch
=============================================================

Techniques that improve generalization by averaging model weights.

Methods:
    1. Exponential Moving Average (EMA)
    2. Stochastic Weight Averaging (SWA)
    3. Polyak Averaging
    4. Model Soup (greedy interpolation)

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
# PART 1: EXPONENTIAL MOVING AVERAGE
# =============================================================================

class EMA:
    """
    Exponential Moving Average of model parameters.

    θ_ema = decay * θ_ema + (1 - decay) * θ_model

    Used in: Diffusion models, semi-supervised learning, BYOL.
    Typical decay: 0.999 to 0.9999.
    """

    def __init__(self, decay: float = 0.999):
        self.decay = decay
        self.shadow = None
        self.step_count = 0

    def register(self, params: list):
        """Initialize shadow params."""
        self.shadow = [p.copy() for p in params]

    def update(self, params: list):
        """Update EMA with current params."""
        self.step_count += 1
        # Bias correction for early steps
        decay = min(self.decay, (1 + self.step_count) / (10 + self.step_count))

        for i, p in enumerate(params):
            self.shadow[i] = decay * self.shadow[i] + (1 - decay) * p

    def get_params(self) -> list:
        """Get EMA parameters."""
        return [s.copy() for s in self.shadow]


# =============================================================================
# PART 2: STOCHASTIC WEIGHT AVERAGING
# =============================================================================

class SWA:
    """
    Stochastic Weight Averaging (Izmailov et al., 2018).

    Average model weights from multiple points along the SGD
    trajectory. Works best with cyclical or high constant LR.

    θ_swa = (θ_swa * n + θ_model) / (n + 1)
    """

    def __init__(self):
        self.averaged = None
        self.n_models = 0

    def update(self, params: list):
        """Add current model to running average."""
        if self.averaged is None:
            self.averaged = [p.copy() for p in params]
        else:
            for i, p in enumerate(params):
                self.averaged[i] = (self.averaged[i] * self.n_models + p) / \
                                   (self.n_models + 1)
        self.n_models += 1

    def get_params(self) -> list:
        return [a.copy() for a in self.averaged]


# =============================================================================
# PART 3: POLYAK AVERAGING
# =============================================================================

class PolyakAveraging:
    """
    Polyak-Ruppert Averaging.

    Simply average all iterates from step t0 onwards:
    θ_avg = (1/T) Σ_{t=t0}^{T} θ_t

    Theoretical guarantee: converges at optimal rate for convex problems.
    """

    def __init__(self, start_step: int = 0):
        self.start_step = start_step
        self.accumulated = None
        self.count = 0
        self.step = 0

    def update(self, params: list):
        self.step += 1
        if self.step < self.start_step:
            return

        if self.accumulated is None:
            self.accumulated = [p.copy() for p in params]
        else:
            for i, p in enumerate(params):
                self.accumulated[i] += p
        self.count += 1

    def get_params(self) -> list:
        if self.count == 0:
            return self.accumulated
        return [a / self.count for a in self.accumulated]


# =============================================================================
# PART 4: MODEL SOUP
# =============================================================================

def model_soup_uniform(model_params_list: list) -> list:
    """
    Uniform Model Soup (Wortsman et al., 2022).

    Simply average weights of multiple independently trained models.
    """
    n = len(model_params_list)
    result = [np.zeros_like(p) for p in model_params_list[0]]
    for params in model_params_list:
        for i, p in enumerate(params):
            result[i] += p / n
    return result


def model_soup_greedy(model_params_list: list, eval_fn,
                       base_data) -> list:
    """
    Greedy Model Soup.

    Start with best model, greedily add models that improve val accuracy.
    """
    # Evaluate all models
    scores = [eval_fn(p, base_data) for p in model_params_list]
    order = np.argsort(scores)[::-1]

    # Start with best
    current = [p.copy() for p in model_params_list[order[0]]]
    current_score = scores[order[0]]
    n_in_soup = 1

    for idx in order[1:]:
        # Try adding this model
        candidate = [(c * n_in_soup + p) / (n_in_soup + 1)
                      for c, p in zip(current, model_params_list[idx])]
        cand_score = eval_fn(candidate, base_data)

        if cand_score > current_score:
            current = candidate
            current_score = cand_score
            n_in_soup += 1

    return current


# =============================================================================
# DEMO
# =============================================================================

def demo():
    print("=" * 70)
    print("EXPONENTIAL MOVING AVERAGE & MODEL AVERAGING")
    print("=" * 70)

    # Simulate training on a noisy quadratic
    dim = 10
    optimal = np.ones(dim) * 3.0

    def loss(params):
        return np.sum((params - optimal) ** 2)

    # Training with SGD (noisy gradients)
    params = np.zeros(dim)
    lr = 0.01

    ema = EMA(decay=0.99)
    ema.register([params])

    swa = SWA()
    polyak = PolyakAveraging(start_step=50)

    losses_raw = []
    losses_ema = []
    losses_swa = []

    for step in range(200):
        grad = 2 * (params - optimal) + np.random.randn(dim) * 2.0  # Noisy gradient
        params = params - lr * grad

        ema.update([params])
        if step >= 100 and step % 10 == 0:
            swa.update([params])
        polyak.update([params])

        losses_raw.append(loss(params))
        losses_ema.append(loss(ema.get_params()[0]))
        if swa.averaged is not None:
            losses_swa.append(loss(swa.get_params()[0]))

    print(f"\nFinal losses:")
    print(f"  Raw SGD:  {losses_raw[-1]:.6f}")
    print(f"  EMA:      {losses_ema[-1]:.6f}")
    print(f"  SWA:      {losses_swa[-1]:.6f}")
    print(f"  Polyak:   {loss(polyak.get_params()[0]):.6f}")

    # Model Soup demo
    print(f"\n--- Model Soup ---")
    models = []
    for i in range(5):
        p = optimal + np.random.randn(dim) * 0.5
        models.append([p])
        print(f"  Model {i+1} loss: {loss(p):.4f}")

    soup = model_soup_uniform(models)
    print(f"  Uniform soup loss: {loss(soup[0]):.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(losses_raw, alpha=0.5, label="Raw SGD")
    ax.plot(losses_ema, label="EMA (0.99)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("EMA vs Raw SGD")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(SAVE_DIR / "ema.png", dpi=100)
    plt.close()
    print("\nPlots saved.")


if __name__ == "__main__":
    demo()

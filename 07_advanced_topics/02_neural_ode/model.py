"""
Neural ODE in PyTorch
=======================

Chen et al. (2018): Neural Ordinary Differential Equations.
Instead of discrete layers, defines a continuous dynamics: dh/dt = f(h, t, θ).

Uses Euler and RK4 ODE solvers (from scratch, not torchdiffeq).

Architecture:
    h(0) = encoder(x) → ODE solve dh/dt = f_θ(h, t) → h(T) → decoder(h(T)) → output
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Callable


@dataclass
class NeuralODEConfig:
    input_dim: int = 2
    hidden_dim: int = 64
    output_dim: int = 2
    n_steps: int = 20      # ODE integration steps
    t_span: float = 1.0    # Integration time
    solver: str = "rk4"    # "euler" or "rk4"


# =============================================================================
# ODE SOLVERS (from scratch)
# =============================================================================

def euler_step(f: Callable, h: torch.Tensor, t: float, dt: float) -> torch.Tensor:
    """Euler method: h(t+dt) = h(t) + dt * f(h, t)."""
    return h + dt * f(h, t)


def rk4_step(f: Callable, h: torch.Tensor, t: float, dt: float) -> torch.Tensor:
    """
    4th-order Runge-Kutta: much more accurate than Euler.
    Uses 4 evaluations of f per step for O(dt^4) error.
    """
    k1 = f(h, t)
    k2 = f(h + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(h + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(h + dt * k3, t + dt)
    return h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def ode_solve(f: Callable, h0: torch.Tensor, t_span: float,
              n_steps: int, solver: str = "rk4") -> torch.Tensor:
    """Integrate ODE from t=0 to t=t_span."""
    dt = t_span / n_steps
    step_fn = rk4_step if solver == "rk4" else euler_step

    h = h0
    t = 0.0
    for _ in range(n_steps):
        h = step_fn(f, h, t, dt)
        t += dt
    return h


# =============================================================================
# ODE FUNCTION (the learnable dynamics)
# =============================================================================

class ODEFunc(nn.Module):
    """
    The learned dynamics: dh/dt = f(h, t).
    A simple MLP that takes the current state and outputs the derivative.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, h: torch.Tensor, t: float) -> torch.Tensor:
        return self.net(h)


# =============================================================================
# NEURAL ODE MODEL
# =============================================================================

class NeuralODE(nn.Module):
    """
    Neural ODE: continuous-depth neural network.

    Instead of h_{l+1} = h_l + f(h_l) (ResNet),
    we solve dh/dt = f(h, t) continuously from t=0 to t=T.

    Benefits:
        - Constant memory (adjoint method)
        - Adaptive computation
        - Continuous normalizing flows
    """

    def __init__(self, config: NeuralODEConfig):
        super().__init__()
        self.config = config

        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.Tanh(),
        )

        self.ode_func = ODEFunc(config.hidden_dim)

        self.decoder = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.encoder(x)

        # Solve ODE
        h_T = ode_solve(
            self.ode_func, h0,
            t_span=self.config.t_span,
            n_steps=self.config.n_steps,
            solver=self.config.solver,
        )

        return self.decoder(h_T)

    def trajectory(self, x: torch.Tensor, n_points: int = 50) -> torch.Tensor:
        """Get full trajectory for visualization."""
        h = self.encoder(x)
        dt = self.config.t_span / n_points
        step_fn = rk4_step if self.config.solver == "rk4" else euler_step

        trajectory = [h.detach()]
        t = 0.0
        for _ in range(n_points):
            h = step_fn(self.ode_func, h, t, dt)
            trajectory.append(h.detach())
            t += dt

        return torch.stack(trajectory, dim=0)  # (n_points+1, batch, hidden_dim)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

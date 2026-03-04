"""
Optimization with PyTorch
==========================

PyTorch re-implementation of all optimization algorithms from implementation.py.
Uses torch.optim for standard optimizers and shows custom optimizer implementation.

Comparison:
    NumPy (implementation.py)     →  PyTorch (this file)
    ─────────────────────────────────────────────────────
    Manual GD                     →  torch.optim.SGD (lr only)
    Manual Momentum               →  torch.optim.SGD(momentum=0.9)
    Manual Nesterov               →  torch.optim.SGD(nesterov=True)
    Manual AdaGrad                →  torch.optim.Adagrad
    Manual RMSProp                →  torch.optim.RMSprop
    Manual Adam                   →  torch.optim.Adam
    Manual LR schedules           →  torch.optim.lr_scheduler
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


# =============================================================================
# TEST FUNCTIONS (same as NumPy version, but with autograd)
# =============================================================================

def rosenbrock(xy: torch.Tensor) -> torch.Tensor:
    """Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)². Minimum at (1,1)."""
    x, y = xy[0], xy[1]
    return (1 - x)**2 + 100 * (y - x**2)**2


def beale(xy: torch.Tensor) -> torch.Tensor:
    """Beale: challenging with sharp curved valley. Minimum at (3, 0.5)."""
    x, y = xy[0], xy[1]
    return ((1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2
            + (2.625 - x + x*y**3)**2)


# =============================================================================
# OPTIMIZER COMPARISON ON TEST FUNCTIONS
# =============================================================================

@dataclass
class OptimResult:
    name: str
    trajectory: List[Tuple[float, float]]
    losses: List[float]
    final_point: Tuple[float, float]
    final_loss: float


def run_optimizer(opt_class, opt_kwargs: dict, fn, start: torch.Tensor,
                  n_steps: int = 5000) -> OptimResult:
    """Run an optimizer on a test function and record trajectory."""
    xy = start.clone().detach().requires_grad_(True)
    optimizer = opt_class([xy], **opt_kwargs)

    trajectory = [(xy[0].item(), xy[1].item())]
    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()
        loss = fn(xy)
        loss.backward()
        optimizer.step()

        trajectory.append((xy[0].item(), xy[1].item()))
        losses.append(loss.item())

    return OptimResult(
        name=f"{opt_class.__name__}({opt_kwargs})",
        trajectory=trajectory,
        losses=losses,
        final_point=(xy[0].item(), xy[1].item()),
        final_loss=losses[-1],
    )


def demo_optimizer_comparison():
    """Compare all torch.optim optimizers on Rosenbrock."""
    print("=" * 60)
    print("OPTIMIZER COMPARISON ON ROSENBROCK")
    print("=" * 60)

    start = torch.tensor([-1.0, 1.0])
    n_steps = 5000

    configs = [
        ("SGD",      torch.optim.SGD,     {"lr": 0.001}),
        ("Momentum", torch.optim.SGD,     {"lr": 0.001, "momentum": 0.9}),
        ("Nesterov", torch.optim.SGD,     {"lr": 0.001, "momentum": 0.9, "nesterov": True}),
        ("Adagrad",  torch.optim.Adagrad, {"lr": 0.1}),
        ("RMSprop",  torch.optim.RMSprop, {"lr": 0.001}),
        ("Adam",     torch.optim.Adam,    {"lr": 0.01}),
        ("AdamW",    torch.optim.AdamW,   {"lr": 0.01, "weight_decay": 0.01}),
    ]

    results = []
    for name, opt_class, kwargs in configs:
        result = run_optimizer(opt_class, kwargs, rosenbrock, start, n_steps)
        results.append((name, result))
        print(f"  {name:12s}: final=({result.final_point[0]:.4f}, {result.final_point[1]:.4f}), "
              f"loss={result.final_loss:.6f}")

    print(f"\n  Target: (1.0, 1.0), loss=0.0")
    print()
    return results


# =============================================================================
# LEARNING RATE SCHEDULES
# =============================================================================

def demo_lr_schedules():
    """Demonstrate PyTorch learning rate schedulers."""
    print("=" * 60)
    print("LEARNING RATE SCHEDULES")
    print("=" * 60)

    n_epochs = 100
    dummy_param = torch.tensor([1.0], requires_grad=True)

    schedules = {
        "StepLR(30, γ=0.1)": lambda opt: torch.optim.lr_scheduler.StepLR(
            opt, step_size=30, gamma=0.1),
        "CosineAnnealing(100)": lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=100),
        "ExponentialLR(γ=0.95)": lambda opt: torch.optim.lr_scheduler.ExponentialLR(
            opt, gamma=0.95),
        "OneCycleLR(0.1)": lambda opt: torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=0.1, total_steps=100),
    }

    for name, sched_fn in schedules.items():
        optimizer = torch.optim.SGD([dummy_param], lr=0.1)
        scheduler = sched_fn(optimizer)

        lrs = []
        for epoch in range(n_epochs):
            lrs.append(optimizer.param_groups[0]["lr"])
            # Fake step
            optimizer.zero_grad()
            loss = dummy_param.sum()
            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f"  {name:30s}: "
              f"start={lrs[0]:.4f}, mid={lrs[50]:.4f}, end={lrs[-1]:.6f}")
    print()


# =============================================================================
# WARMUP + COSINE DECAY (Modern Default)
# =============================================================================

class WarmupCosineScheduler:
    """
    Linear warmup followed by cosine decay.
    This is the standard schedule used in modern training (GPT, ViT, etc.).
    """

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            scale = self.step_count / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / \
                       (self.total_steps - self.warmup_steps)
            scale = 0.5 * (1 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * scale)

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


def demo_warmup_cosine():
    """Demonstrate warmup + cosine schedule."""
    print("=" * 60)
    print("WARMUP + COSINE DECAY")
    print("=" * 60)

    dummy = torch.tensor([1.0], requires_grad=True)
    optimizer = torch.optim.Adam([dummy], lr=0.001)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=10, total_steps=100)

    lrs = []
    for step in range(100):
        lrs.append(scheduler.get_lr())
        scheduler.step()

    print(f"  Step  0: lr={lrs[0]:.6f}")
    print(f"  Step  5: lr={lrs[5]:.6f} (warmup)")
    print(f"  Step 10: lr={lrs[10]:.6f} (peak)")
    print(f"  Step 50: lr={lrs[50]:.6f} (decay)")
    print(f"  Step 99: lr={lrs[99]:.6f} (near min)")
    print()


# =============================================================================
# CUSTOM OPTIMIZER EXAMPLE
# =============================================================================

class CustomAdam(torch.optim.Optimizer):
    """
    Adam optimizer implemented from scratch as a torch.optim.Optimizer subclass.
    Shows how to write custom optimizers that plug into the PyTorch ecosystem.
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1
                m, v = state["m"], state["v"]

                # Update biased moments
                m.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

                # Bias correction
                m_hat = m / (1 - beta1 ** state["step"])
                v_hat = v / (1 - beta2 ** state["step"])

                # Parameter update
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

        return loss


def demo_custom_optimizer():
    """Compare custom Adam vs torch.optim.Adam on Rosenbrock."""
    print("=" * 60)
    print("CUSTOM vs BUILT-IN ADAM")
    print("=" * 60)

    for name, opt_class in [("torch.optim.Adam", torch.optim.Adam),
                             ("CustomAdam", CustomAdam)]:
        xy = torch.tensor([-1.0, 1.0], requires_grad=True)
        optimizer = opt_class([xy], lr=0.01)

        for step in range(3001):
            optimizer.zero_grad()
            loss = rosenbrock(xy)
            loss.backward()
            optimizer.step()

        print(f"  {name:20s}: ({xy[0].item():.6f}, {xy[1].item():.6f}), "
              f"loss={rosenbrock(xy).item():.8f}")
    print()


if __name__ == "__main__":
    demo_optimizer_comparison()
    demo_lr_schedules()
    demo_warmup_cosine()
    demo_custom_optimizer()

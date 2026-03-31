"""
Mixture of Experts in PyTorch
================================

PyTorch MoE implementation with top-K routing, load balancing,
and expert capacity management.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class MoEConfig:
    in_features: int = 784
    hidden_dim: int = 128
    n_classes: int = 10
    n_experts: int = 8
    top_k: int = 2
    expert_dim: int = 64
    balance_coef: float = 0.01      # Load balancing loss weight
    dropout: float = 0.1


# =============================================================================
# EXPERT
# =============================================================================

class Expert(nn.Module):
    """Single expert: Linear → ReLU → Linear."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# TOP-K ROUTER
# =============================================================================

class TopKRouter(nn.Module):
    """Learned router that selects top-K experts per input."""

    def __init__(self, in_dim: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.gate = nn.Linear(in_dim, n_experts, bias=False)
        self.top_k = top_k
        self.n_experts = n_experts

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: (top_k_indices, top_k_weights, router_probs)"""
        logits = self.gate(x)  # (B, n_experts)
        router_probs = F.softmax(logits, dim=-1)

        top_k_weights, top_k_idx = torch.topk(router_probs, self.top_k, dim=-1)
        # Renormalize weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        return top_k_idx, top_k_weights, router_probs


# =============================================================================
# MOE LAYER
# =============================================================================

class MoELayer(nn.Module):
    """
    Mixture of Experts layer with top-K routing.

    Only activates K out of N experts per input token.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 n_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(in_dim, hidden_dim, out_dim) for _ in range(n_experts)
        ])
        self.router = TopKRouter(in_dim, n_experts, top_k)
        self.n_experts = n_experts
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (output, router_probs for load balancing)."""
        B = x.size(0)
        top_k_idx, top_k_weights, router_probs = self.router(x)

        # Compute all expert outputs (simple approach)
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, E, D)

        # Gather selected experts
        idx_expanded = top_k_idx.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1))
        selected = torch.gather(expert_outputs, 1, idx_expanded)  # (B, K, D)

        # Weighted sum
        output = (selected * top_k_weights.unsqueeze(-1)).sum(dim=1)  # (B, D)

        return output, router_probs


def load_balancing_loss(router_probs: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Auxiliary loss for balanced expert utilization."""
    # Fraction of tokens per expert
    expert_usage = router_probs.mean(dim=0)  # (n_experts,)
    # Penalize deviation from uniform
    return n_experts * (expert_usage * expert_usage).sum()


# =============================================================================
# FULL MODEL
# =============================================================================

class MoEClassifier(nn.Module):
    """MoE-based classifier."""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.in_features, config.hidden_dim)
        self.moe = MoELayer(config.hidden_dim, config.expert_dim,
                            config.hidden_dim, config.n_experts, config.top_k)
        self.classifier = nn.Linear(config.hidden_dim, config.n_classes)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        h = F.relu(self.input_proj(x))
        h, router_probs = self.moe(h)
        h = self.dropout(F.relu(h))
        logits = self.classifier(h)
        return logits, router_probs

"""
Knowledge Distillation in PyTorch
====================================

Teacher-student training framework with soft target matching,
feature distillation, and temperature scaling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class KDConfig:
    """Knowledge distillation configuration."""
    teacher_dims: List[int] = None     # [784, 512, 256, 10]
    student_dims: List[int] = None     # [784, 128, 10]
    temperature: float = 4.0
    alpha: float = 0.7                 # weight for soft targets
    n_classes: int = 10

    def __post_init__(self):
        if self.teacher_dims is None:
            self.teacher_dims = [784, 512, 256, 128, 10]
        if self.student_dims is None:
            self.student_dims = [784, 128, 64, 10]


# =============================================================================
# MODELS
# =============================================================================

class TeacherModel(nn.Module):
    """Large teacher model."""

    def __init__(self, dims: List[int]):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)


class StudentModel(nn.Module):
    """Small student model."""

    def __init__(self, dims: List[int]):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)


# =============================================================================
# DISTILLATION LOSS
# =============================================================================

class DistillationLoss(nn.Module):
    """
    Combined distillation loss.

    L = α * T² * KL(soft_teacher || soft_student) + (1-α) * CE(hard_labels, student)
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean',
        ) * (self.temperature ** 2)

        hard_loss = F.cross_entropy(student_logits, labels)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

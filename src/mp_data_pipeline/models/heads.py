"""Grouped and hierarchical multitask heads with physics constraints."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ml.tasks import (
    ELASTIC_TASKS,
    ELECTRONIC_CLASSIFICATION_TASKS,
    ELECTRONIC_REGRESSION_TASKS,
    STABILITY_TASKS,
    STRUCTURE_TASKS,
    THERMO_TASKS,
)


class MLPHead(nn.Module):
    """Two-layer MLP prediction head."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TaskHead(nn.Module):
    """Per-task MLP head that emits a scalar."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class HierarchicalGroup(nn.Module):
    """Shared group trunk with task-specific second-stage heads."""

    def __init__(self, in_dim: int, hidden_dim: int, task_names: list[str], dropout: float = 0.1):
        super().__init__()
        self.task_names = task_names
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        task_hidden = max(hidden_dim // 2, 64)
        self.task_heads = nn.ModuleDict(
            {task_name: TaskHead(hidden_dim, task_hidden, dropout) for task_name in task_names}
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        latent = self.trunk(x)
        return {task_name: head(latent) for task_name, head in self.task_heads.items()}


def _apply_constraints(task_name: str, raw_value: torch.Tensor) -> torch.Tensor:
    """Apply task-specific physics constraints to raw outputs."""
    if task_name == "band_gap":
        return F.softplus(raw_value)
    if task_name == "volume":
        return F.softplus(raw_value) + 1.0
    if task_name == "density":
        return F.softplus(raw_value) + 0.1
    if task_name in {"bulk_modulus_vrh", "shear_modulus_vrh", "universal_anisotropy"}:
        return F.softplus(raw_value)
    if task_name == "homogeneous_poisson":
        return torch.sigmoid(raw_value) * 1.5 - 1.0
    return raw_value


class GroupedTaskHeads(nn.Module):
    """Task heads split by property groups."""

    def __init__(self, hidden_dim: int = 256, head_hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.thermo = MLPHead(hidden_dim, head_hidden_dim, len(THERMO_TASKS), dropout)
        self.electronic = MLPHead(
            hidden_dim,
            head_hidden_dim,
            len(ELECTRONIC_REGRESSION_TASKS) + len(ELECTRONIC_CLASSIFICATION_TASKS),
            dropout,
        )
        # Unified head_hidden_dim for all heads (previously stability/structure used head_hidden_dim // 2)
        self.stability = MLPHead(hidden_dim, head_hidden_dim, len(STABILITY_TASKS), dropout)
        self.structure = MLPHead(hidden_dim, head_hidden_dim, len(STRUCTURE_TASKS), dropout)
        self.elastic = MLPHead(hidden_dim, head_hidden_dim, len(ELASTIC_TASKS), dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}

        # Thermo head - no constraints
        thermo_out = self.thermo(x)
        for i, task_name in enumerate(THERMO_TASKS):
            out[task_name] = thermo_out[:, i]

        # Electronic head - with physics constraints
        electronic_out = self.electronic(x)
        offset = 0
        for task_name in ELECTRONIC_REGRESSION_TASKS:
            out[task_name] = _apply_constraints(task_name, electronic_out[:, offset])
            offset += 1
        for task_name in ELECTRONIC_CLASSIFICATION_TASKS:
            out[task_name] = electronic_out[:, offset]
            offset += 1

        # Stability head - no constraints (logits)
        stability_out = self.stability(x)
        for i, task_name in enumerate(STABILITY_TASKS):
            out[task_name] = stability_out[:, i]

        # Structure head - with physics constraints
        structure_out = self.structure(x)
        for i, task_name in enumerate(STRUCTURE_TASKS):
            out[task_name] = _apply_constraints(task_name, structure_out[:, i])

        # Elastic head - with physics constraints
        elastic_out = self.elastic(x)
        for i, task_name in enumerate(ELASTIC_TASKS):
            out[task_name] = _apply_constraints(task_name, elastic_out[:, i])

        return out


class ElectronicHierarchicalHeads(GroupedTaskHeads):
    """Stage C h1: hierarchical electronic head, parallel others."""

    def __init__(self, hidden_dim: int = 256, head_hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__(hidden_dim=hidden_dim, head_hidden_dim=head_hidden_dim, dropout=dropout)
        electronic_tasks = ELECTRONIC_REGRESSION_TASKS + ELECTRONIC_CLASSIFICATION_TASKS
        self.electronic_hier = HierarchicalGroup(hidden_dim, head_hidden_dim, electronic_tasks, dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = super().forward(x)
        electronic_out = self.electronic_hier(x)
        for task_name, raw_value in electronic_out.items():
            out[task_name] = _apply_constraints(task_name, raw_value)
        return out


class ElasticDerivedHeads(GroupedTaskHeads):
    """Stage C h2: elastic head predicts base moduli then derived quantities."""

    def __init__(self, hidden_dim: int = 256, head_hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__(hidden_dim=hidden_dim, head_hidden_dim=head_hidden_dim, dropout=dropout)
        self.elastic_trunk = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, head_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.elastic_primary = MLPHead(head_hidden_dim, head_hidden_dim, 2, dropout)
        derived_in_dim = head_hidden_dim + 2
        derived_hidden = max(head_hidden_dim // 2, 64)
        self.elastic_derived = nn.ModuleDict(
            {
                "homogeneous_poisson": TaskHead(derived_in_dim, derived_hidden, dropout),
                "universal_anisotropy": TaskHead(derived_in_dim, derived_hidden, dropout),
            }
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = super().forward(x)
        latent = self.elastic_trunk(x)
        primary_raw = self.elastic_primary(latent)
        out["bulk_modulus_vrh"] = _apply_constraints("bulk_modulus_vrh", primary_raw[:, 0])
        out["shear_modulus_vrh"] = _apply_constraints("shear_modulus_vrh", primary_raw[:, 1])
        derived_features = torch.cat([latent, primary_raw], dim=-1)
        for task_name, head in self.elastic_derived.items():
            out[task_name] = _apply_constraints(task_name, head(derived_features))
        return out


class HybridHierarchicalHeads(GroupedTaskHeads):
    """Stage C hybrid: hierarchical electronic head + elastic derived head."""

    def __init__(self, hidden_dim: int = 256, head_hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__(hidden_dim=hidden_dim, head_hidden_dim=head_hidden_dim, dropout=dropout)
        electronic_tasks = ELECTRONIC_REGRESSION_TASKS + ELECTRONIC_CLASSIFICATION_TASKS
        self.electronic_hier = HierarchicalGroup(hidden_dim, head_hidden_dim, electronic_tasks, dropout)
        self.elastic_trunk = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, head_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.elastic_primary = MLPHead(head_hidden_dim, head_hidden_dim, 2, dropout)
        derived_in_dim = head_hidden_dim + 2
        derived_hidden = max(head_hidden_dim // 2, 64)
        self.elastic_derived = nn.ModuleDict(
            {
                "homogeneous_poisson": TaskHead(derived_in_dim, derived_hidden, dropout),
                "universal_anisotropy": TaskHead(derived_in_dim, derived_hidden, dropout),
            }
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = super().forward(x)
        electronic_out = self.electronic_hier(x)
        for task_name, raw_value in electronic_out.items():
            out[task_name] = _apply_constraints(task_name, raw_value)

        latent = self.elastic_trunk(x)
        primary_raw = self.elastic_primary(latent)
        out["bulk_modulus_vrh"] = _apply_constraints("bulk_modulus_vrh", primary_raw[:, 0])
        out["shear_modulus_vrh"] = _apply_constraints("shear_modulus_vrh", primary_raw[:, 1])
        derived_features = torch.cat([latent, primary_raw], dim=-1)
        for task_name, head in self.elastic_derived.items():
            out[task_name] = _apply_constraints(task_name, head(derived_features))
        return out

"""Loss functions for multitask property training."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn as nn

from ..ml.tasks import ALL_TASKS, CLASSIFICATION_TASKS, TASK_INDEX


def build_static_task_weights(mask_matrix: np.ndarray, task_names: Iterable[str]) -> Dict[str, float]:
    """Build static weights as 1/sqrt(coverage)."""
    weights: Dict[str, float] = {}
    n = mask_matrix.shape[0]
    for task_name in task_names:
        idx = TASK_INDEX[task_name]
        coverage = float(mask_matrix[:, idx].sum()) / max(1.0, float(n))
        coverage = max(coverage, 1e-6)
        weights[task_name] = 1.0 / float(np.sqrt(coverage))

    total = sum(weights.values())
    if total > 0:
        norm = len(weights) / total
        for key in weights:
            weights[key] *= norm
    return weights


class MultiTaskLoss(nn.Module):
    """Masked multitask loss with optional consistency penalties."""

    def __init__(
        self,
        enabled_tasks: List[str],
        task_weights: Dict[str, float],
        lambda_cbm_vbm: float = 0.05,
        lambda_stability_consistency: float = 0.02,
    ):
        super().__init__()
        self.enabled_tasks = enabled_tasks
        self.task_weights = task_weights
        self.lambda_cbm_vbm = lambda_cbm_vbm
        self.lambda_stability_consistency = lambda_stability_consistency
        self.huber = nn.SmoothL1Loss(reduction="none", beta=1.0)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
        # Keep a gradient path even if a batch has no valid labels for enabled tasks.
        loss = preds.sum() * 0.0
        metrics: Dict[str, float] = {}

        for task in self.enabled_tasks:
            idx = TASK_INDEX[task]
            mask = masks[:, idx] > 0.5
            if not torch.any(mask):
                continue

            y_hat = preds[mask, idx]
            y = targets[mask, idx]
            weight = float(self.task_weights.get(task, 1.0))

            if task in CLASSIFICATION_TASKS:
                task_loss = self.bce(y_hat, y).mean()
            else:
                task_loss = self.huber(y_hat, y).mean()

            weighted = weight * task_loss
            loss = loss + weighted
            metrics[f"loss_{task}"] = float(task_loss.detach().cpu().item())

        # Soft consistency: cbm >= vbm + margin.
        if {"cbm", "vbm"}.issubset(self.enabled_tasks):
            idx_cbm = TASK_INDEX["cbm"]
            idx_vbm = TASK_INDEX["vbm"]
            mask = (masks[:, idx_cbm] > 0.5) & (masks[:, idx_vbm] > 0.5)
            if torch.any(mask):
                margin = 0.01
                penalty = torch.relu(preds[mask, idx_vbm] - preds[mask, idx_cbm] + margin).mean()
                loss = loss + self.lambda_cbm_vbm * penalty
                metrics["penalty_cbm_vbm"] = float(penalty.detach().cpu().item())

        # Soft consistency: stable probability should drop as hull energy rises.
        if {"is_stable", "energy_above_hull"}.issubset(self.enabled_tasks):
            idx_stable = TASK_INDEX["is_stable"]
            idx_hull = TASK_INDEX["energy_above_hull"]
            mask = (masks[:, idx_stable] > 0.5) & (masks[:, idx_hull] > 0.5)
            if torch.any(mask):
                p_stable = torch.sigmoid(preds[mask, idx_stable])
                hull = torch.relu(preds[mask, idx_hull])
                penalty = (p_stable * hull).mean()
                loss = loss + self.lambda_stability_consistency * penalty
                metrics["penalty_stability_hull"] = float(penalty.detach().cpu().item())

        metrics["loss_total"] = float(loss.detach().cpu().item())
        return loss, metrics


def regression_metric_names() -> List[str]:
    """Regression tasks used for metric reporting."""
    return [task.name for task in ALL_TASKS if task.task_type == "regression"]

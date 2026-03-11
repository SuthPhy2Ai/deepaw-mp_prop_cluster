"""Training and evaluation utilities for multitask models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..ml.tasks import CLASSIFICATION_TASKS, TASK_INDEX
from .losses import MultiTaskLoss


@dataclass
class TrainerConfig:
    """Trainer hyperparameters."""

    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    device: str = "cuda"
    amp: bool = False  # Disabled due to NaN issues with GraphBackbone
    accumulation_steps: int = 1  # Gradient accumulation steps


@dataclass
class EpochResult:
    """Aggregated epoch metrics."""

    loss: float
    task_metrics: Dict[str, float]


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _collect_predictions(
    preds: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    task_names: List[str],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for task in task_names:
        idx = TASK_INDEX[task]
        task_mask = masks[:, idx] > 0.5
        if not np.any(task_mask):
            continue

        y_pred = preds[task_mask, idx]
        y_true = targets[task_mask, idx]

        if task in CLASSIFICATION_TASKS:
            probs = 1.0 / (1.0 + np.exp(-y_pred))
            if len(np.unique(y_true)) < 2:
                auc = float("nan")
            else:
                auc = float(roc_auc_score(y_true, probs))
            acc = float(((probs >= 0.5) == (y_true >= 0.5)).mean())
            metrics[f"{task}_auroc"] = auc
            metrics[f"{task}_acc"] = acc
        else:
            mae = float(np.mean(np.abs(y_pred - y_true)))
            rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
            metrics[f"{task}_mae"] = mae
            metrics[f"{task}_rmse"] = rmse
    return metrics


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: MultiTaskLoss,
    config: TrainerConfig,
    writer: Optional[SummaryWriter] = None,
    epoch: int = 0,
) -> EpochResult:
    """Run one training epoch with gradient accumulation support."""
    device = torch.device(config.device)
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp and device.type == "cuda")

    running_loss = 0.0
    n_batches = 0
    metric_accumulator: Dict[str, float] = {}

    # Track per-task loss for debugging
    task_loss_accumulator: Dict[str, float] = {}

    optimizer.zero_grad(set_to_none=True)

    # Print interval for iter-level monitoring
    print_interval = max(1, len(loader) // 10)  # Print 10 times per epoch

    for batch_idx, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        batch = _to_device(batch, device)

        with torch.cuda.amp.autocast(enabled=config.amp and device.type == "cuda"):
            preds = model(batch)
            loss, metrics = loss_fn(preds, batch["targets"], batch["masks"])

            # Check for NaN in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠ WARNING: NaN/Inf detected at batch {batch_idx}/{len(loader)}")
                print(f"  Loss: {loss.item()}")
                print(f"  Task losses:")
                for key, value in metrics.items():
                    if key.startswith("loss_"):
                        print(f"    {key}: {value:.4f}")
                print(f"  Predictions stats: min={preds.min().item():.4f}, max={preds.max().item():.4f}")
                print(f"  Targets stats: min={batch['targets'].min().item():.4f}, max={batch['targets'].max().item():.4f}")
                raise ValueError(f"NaN/Inf loss detected at batch {batch_idx}")

            # Scale loss by accumulation steps
            loss = loss / config.accumulation_steps

        scaler.scale(loss).backward()

        # Only step optimizer every accumulation_steps
        if (batch_idx + 1) % config.accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Log gradient norm to tensorboard
            if writer is not None:
                global_step = epoch * len(loader) + batch_idx
                writer.add_scalar("train/grad_norm", float(grad_norm), global_step)

        running_loss += float(loss.detach().cpu().item()) * config.accumulation_steps
        n_batches += 1
        for key, value in metrics.items():
            metric_accumulator[key] = metric_accumulator.get(key, 0.0) + value
            # Track per-task losses separately
            if key.startswith("loss_"):
                task_loss_accumulator[key] = task_loss_accumulator.get(key, 0.0) + value

        # Log to TensorBoard every batch
        if writer is not None:
            global_step = epoch * len(loader) + batch_idx
            current_loss = float(loss.detach().cpu().item()) * config.accumulation_steps
            writer.add_scalar("train_iter/loss_batch", current_loss, global_step)
            # Log per-task losses
            for key, value in metrics.items():
                if key.startswith("loss_"):
                    writer.add_scalar(f"train_iter/{key}", value, global_step)

        # Print iter-level monitoring
        if (batch_idx + 1) % print_interval == 0:
            current_avg_loss = running_loss / n_batches
            print(f"\n[Epoch {epoch} Iter {batch_idx+1}/{len(loader)}] Current avg loss: {current_avg_loss:.4f}")
            # Show all task losses
            if task_loss_accumulator:
                print("  Task losses:")
                for task_key in sorted(task_loss_accumulator.keys()):
                    task_name = task_key.replace("loss_", "")
                    avg_task_loss = task_loss_accumulator[task_key] / n_batches
                    print(f"    {task_name:30s}: {avg_task_loss:.4f}")


    mean_loss = running_loss / max(1, n_batches)

    # Print per-task loss breakdown for debugging
    if n_batches > 0:
        print(f"\n[Epoch {epoch}] Task Loss Breakdown:")
        for task_key in sorted(task_loss_accumulator.keys()):
            task_name = task_key.replace("loss_", "")
            avg_task_loss = task_loss_accumulator[task_key] / n_batches
            print(f"  {task_name:30s}: {avg_task_loss:.4f}")
        print(f"  {'TOTAL':30s}: {mean_loss:.4f}\n")
    out_metrics = {k: v / max(1, n_batches) for k, v in metric_accumulator.items()}

    # Log per-task losses to tensorboard
    if writer is not None:
        writer.add_scalar("train/loss_total", mean_loss, epoch)
        for key, value in out_metrics.items():
            if key.startswith("loss_"):
                writer.add_scalar(f"train/{key}", value, epoch)

    return EpochResult(loss=mean_loss, task_metrics=out_metrics)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: MultiTaskLoss,
    config: TrainerConfig,
    enabled_tasks: List[str],
    writer: Optional[SummaryWriter] = None,
    epoch: int = 0,
    split_name: str = "val",
) -> EpochResult:
    """Evaluate model on a dataloader."""
    device = torch.device(config.device)
    model.eval()

    running_loss = 0.0
    n_batches = 0
    metric_accumulator: Dict[str, float] = {}

    # Track per-task loss for debugging
    task_loss_accumulator: Dict[str, float] = {}

    all_preds = []
    all_targets = []
    all_masks = []

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = _to_device(batch, device)
        preds = model(batch)
        loss, metrics = loss_fn(preds, batch["targets"], batch["masks"])

        running_loss += float(loss.detach().cpu().item())
        n_batches += 1
        for key, value in metrics.items():
            metric_accumulator[key] = metric_accumulator.get(key, 0.0) + value
            # Track per-task losses separately
            if key.startswith("loss_"):
                task_loss_accumulator[key] = task_loss_accumulator.get(key, 0.0) + value

        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(batch["targets"].detach().cpu().numpy())
        all_masks.append(batch["masks"].detach().cpu().numpy())

    mean_loss = running_loss / max(1, n_batches)
    out_metrics = {k: v / max(1, n_batches) for k, v in metric_accumulator.items()}

    # Print per-task loss breakdown for debugging
    if n_batches > 0:
        print(f"\n[Epoch {epoch} {split_name.upper()}] Task Loss Breakdown:")
        for task_key in sorted(task_loss_accumulator.keys()):
            task_name = task_key.replace("loss_", "")
            avg_task_loss = task_loss_accumulator[task_key] / n_batches
            print(f"  {task_name:30s}: {avg_task_loss:.4f}")
        print(f"  {'TOTAL':30s}: {mean_loss:.4f}\n")

    if all_preds:
        preds_np = np.concatenate(all_preds, axis=0)
        targets_np = np.concatenate(all_targets, axis=0)
        masks_np = np.concatenate(all_masks, axis=0)
        out_metrics.update(_collect_predictions(preds_np, targets_np, masks_np, enabled_tasks))

    # Log validation metrics to tensorboard
    if writer is not None:
        writer.add_scalar(f"{split_name}/loss_total", mean_loss, epoch)
        for key, value in out_metrics.items():
            if not key.startswith("loss_"):  # Log MAE, RMSE, AUROC, ACC
                writer.add_scalar(f"{split_name}/{key}", value, epoch)

    return EpochResult(loss=mean_loss, task_metrics=out_metrics)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: TrainerConfig,
    best_metric: float,
    enabled_tasks: List[str],
) -> None:
    """Save training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "config": asdict(config),
        "best_metric": best_metric,
        "enabled_tasks": enabled_tasks,
    }
    torch.save(payload, path)


def save_metrics(path: Path, metrics: Dict[str, float]) -> None:
    """Save metrics dict as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))

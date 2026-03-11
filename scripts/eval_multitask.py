#!/usr/bin/env python3
"""Evaluate a trained multitask model checkpoint."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mp_data_pipeline.config import DB_PATH
from mp_data_pipeline.ml.dataset import AseGraphMultitaskDataset, collate_graph_samples
from mp_data_pipeline.ml.pyg_dataset import PyGMaterialsDataset, collate_pyg_batch
from mp_data_pipeline.ml.splits import load_split
from mp_data_pipeline.models.multitask_model import MultitaskPropertyModel
from mp_data_pipeline.training.losses import MultiTaskLoss, build_static_task_weights
from mp_data_pipeline.training.trainer import TrainerConfig, evaluate, save_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multitask checkpoint")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument(
        "--split",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "split_iid_seed42.json",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", type=str, default="graph")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--max-neighbors", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-pyg", dest="use_pyg", action="store_true", help="Use PyG InMemoryDataset backend")
    parser.add_argument("--no-pyg", dest="use_pyg", action="store_false", help="Disable PyG backend")
    parser.set_defaults(use_pyg=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts" / "eval_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = load_split(args.split)
    split_tag = args.split.stem

    if args.use_pyg:
        pyg_root = PROJECT_ROOT / "data" / "pyg_cache" / split_tag
        val_ds = PyGMaterialsDataset(
            root=str(pyg_root / "val"),
            db_path=str(args.db),
            mp_ids=split["val"],
            cutoff=args.cutoff,
            max_neighbors=args.max_neighbors,
        )
        test_ds = PyGMaterialsDataset(
            root=str(pyg_root / "test"),
            db_path=str(args.db),
            mp_ids=split["test"],
            cutoff=args.cutoff,
            max_neighbors=args.max_neighbors,
        )
        collate_fn = collate_pyg_batch
    else:
        val_ds = AseGraphMultitaskDataset(
            db_path=args.db,
            mp_ids=split["val"],
            cutoff=args.cutoff,
            max_neighbors=args.max_neighbors,
        )
        test_ds = AseGraphMultitaskDataset(
            db_path=args.db,
            mp_ids=split["test"],
            cutoff=args.cutoff,
            max_neighbors=args.max_neighbors,
        )
        collate_fn = collate_graph_samples

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    enabled_tasks = ckpt.get("enabled_tasks")
    if not enabled_tasks:
        raise ValueError("Checkpoint missing enabled_tasks")

    # Build task weights from val masks just to satisfy MultiTaskLoss constructor.
    import numpy as np

    # Compute masks from backend-specific samples
    val_masks = []
    for i in range(len(val_ds)):
        sample = val_ds[i]
        if args.use_pyg:
            val_masks.append(sample.mask.numpy())
        else:
            val_masks.append(sample.masks)
    val_masks = np.stack(val_masks, axis=0)
    task_weights = build_static_task_weights(val_masks, enabled_tasks)

    model = MultitaskPropertyModel(
        backbone_name=args.backbone,
        hidden_dim=args.hidden_dim,
        n_layers=args.layers,
        cutoff=args.cutoff,
        head_variant=getattr(args, "head_variant", "grouped"),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(torch.device(args.device))

    config = TrainerConfig(device=args.device)
    loss_fn = MultiTaskLoss(enabled_tasks=enabled_tasks, task_weights=task_weights)

    val_metrics = evaluate(model, val_loader, loss_fn, config, enabled_tasks)
    test_metrics = evaluate(model, test_loader, loss_fn, config, enabled_tasks)

    payload = {
        "checkpoint": str(args.checkpoint),
        "split": str(args.split),
        "enabled_tasks": enabled_tasks,
        "val_metrics": {"loss": val_metrics.loss, **val_metrics.task_metrics},
        "test_metrics": {"loss": test_metrics.loss, **test_metrics.task_metrics},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train multitask crystal-property models."""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mp_data_pipeline.config import DB_PATH
from mp_data_pipeline.ml.dataset import AseGraphMultitaskDataset, collate_graph_samples
from mp_data_pipeline.ml.enhanced_dataset import EnhancedGraphDataset, collate_enhanced_graph_samples
from mp_data_pipeline.ml.pyg_dataset import PyGMaterialsDataset, collate_pyg_batch
from mp_data_pipeline.ml.splits import load_split
from mp_data_pipeline.ml.tasks import TASK_NAME_LIST, stage_task_names
from mp_data_pipeline.models.multitask_model import MultitaskPropertyModel
from mp_data_pipeline.training.losses import MultiTaskLoss, build_static_task_weights
from mp_data_pipeline.training.sampler import build_stage_sampler
from mp_data_pipeline.training.trainer import (
    TrainerConfig,
    evaluate,
    save_checkpoint,
    save_metrics,
    train_one_epoch,
)
# Phase 2A: Training stability imports
from mp_data_pipeline.training.ema import EMAModel
from mp_data_pipeline.training.warmup import WarmupScheduler
from mp_data_pipeline.training.checkpoint import BestKCheckpoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multitask model")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="ASE DB path")
    parser.add_argument(
        "--split",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "split_iid_seed42.json",
        help="Split JSON path",
    )
    parser.add_argument("--stage", type=str, default="a", help="Training stage: a|b|c")
    parser.add_argument("--backbone", type=str, default="graph", help="graph|composition|enhanced_graph|xpainn")
    parser.add_argument(
        "--head-variant",
        type=str,
        default="grouped",
        help="grouped|stagec_h1|stagec_h2|stagec_hybrid",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--max-neighbors", type=int, default=24)
    parser.add_argument("--n-rbf", type=int, default=64, help="Number of RBF basis functions")
    parser.add_argument("--use-angles", action="store_true", help="Use angle features (for enhanced_graph)")
    parser.add_argument("--use-edge-update", action="store_true", help="Use edge update (for enhanced_graph)")
    # DeePAW integration
    parser.add_argument("--use-deepaw-features", action="store_true", help="Use DeePAW pretrained atom features")
    parser.add_argument(
        "--deepaw-checkpoint",
        type=str,
        default="/home/sutianhao/data/deepaw_test/DeePAW-main/checkpoints/f_nonlocal_escn_best.pth",
        help="Path to DeePAW checkpoint",
    )
    parser.add_argument(
        "--deepaw-fusion",
        type=str,
        default="add",
        choices=["add", "concat"],
        help="How to fuse DeePAW features with atom embeddings",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--oversample-elastic", type=float, default=4.0)
    parser.add_argument("--use-pyg", dest="use_pyg", action="store_true", help="Use PyG InMemoryDataset backend")
    parser.add_argument("--no-pyg", dest="use_pyg", action="store_false", help="Disable PyG backend")
    parser.set_defaults(use_pyg=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Phase 2A: Training stability
    parser.add_argument("--ema-decay", type=float, default=0.0, help="EMA decay rate (0 to disable)")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Number of warmup epochs")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Gradient clipping threshold (0 to disable)")
    parser.add_argument("--best-k", type=int, default=1, help="Keep top-K checkpoints")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "runs",
        help="Directory for run artifacts",
    )
    parser.add_argument(
        "--exclude-tasks",
        type=str,
        nargs="+",
        default=None,
        help="Tasks to exclude from training (e.g., volume density is_stable)",
    )
    parser.add_argument(
        "--only-task",
        type=str,
        default=None,
        help="Train only one task (overrides stage task set)",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Initialize model weights from a checkpoint (backbone+heads).",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze backbone parameters (typically with --init-checkpoint).",
    )
    return parser.parse_args()


def _get_mask_worker_wrapper(args):
    """Worker function for parallel mask computation."""
    db_path, mp_id, cutoff, max_neighbors, strict_elastic_filter = args
    # Create a temporary dataset with single mp_id
    from mp_data_pipeline.ml.dataset import AseGraphMultitaskDataset
    ds = AseGraphMultitaskDataset(
        db_path=db_path,
        mp_ids=[mp_id],
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        strict_elastic_filter=strict_elastic_filter,
    )
    return ds.get_masks(0)


def dataset_masks(dataset: AseGraphMultitaskDataset) -> np.ndarray:
    """Extract masks from dataset efficiently using multiprocessing with caching."""
    from multiprocessing import Pool, cpu_count
    import time
    import hashlib
    import pickle

    # Create cache key based on dataset configuration
    cache_key = hashlib.md5(
        f"{dataset.db_path}_{len(dataset)}_{dataset.cutoff}_{dataset.max_neighbors}_{dataset.strict_elastic_filter}".encode()
    ).hexdigest()
    cache_file = Path(f"data/cache/masks_{cache_key}.pkl")
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    # Try to load from cache
    if cache_file.exists():
        print(f"Loading masks from cache: {cache_file}", flush=True)
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            # Verify cache is valid (same mp_ids)
            if cached_data['mp_ids'] == dataset.mp_ids:
                print(f"✅ Loaded {len(cached_data['masks'])} masks from cache", flush=True)
                return cached_data['masks']
            else:
                print("⚠️ Cache invalid (different mp_ids), recomputing...", flush=True)

    n_workers = min(cpu_count(), 16)  # Use up to 16 cores
    print(f"Computing masks for {len(dataset)} samples using {n_workers} workers...", flush=True)

    start = time.time()

    # Prepare arguments for workers
    args_list = [
        (dataset.db_path, dataset.mp_ids[i], dataset.cutoff, dataset.max_neighbors, dataset.strict_elastic_filter)
        for i in range(len(dataset))
    ]

    with Pool(n_workers) as pool:
        masks = pool.map(_get_mask_worker_wrapper, args_list, chunksize=100)

    masks_array = np.stack(masks, axis=0)
    elapsed = time.time() - start
    print(f"✅ Masks computed in {elapsed:.1f}s ({len(dataset)/elapsed:.1f} samples/sec)", flush=True)

    # Save to cache
    print(f"Saving masks to cache: {cache_file}", flush=True)
    with open(cache_file, 'wb') as f:
        pickle.dump({'mp_ids': dataset.mp_ids, 'masks': masks_array}, f)
    print(f"✅ Cache saved", flush=True)

    return masks_array


def main() -> None:
    args = parse_args()
    split = load_split(args.split)
    split_tag = args.split.stem

    # Choose dataset and collate function based on backend
    if args.use_pyg:
        # Use PyG InMemoryDataset for maximum speed
        print("Using PyG InMemoryDataset (all data in memory)")
        dataset_cls = PyGMaterialsDataset
        collate_fn = collate_pyg_batch

        # PyG dataset needs root directory for caching
        pyg_root = PROJECT_ROOT / "data" / "pyg_cache" / split_tag

        train_ds = dataset_cls(
            root=str(pyg_root / "train"),
            db_path=str(args.db),
            mp_ids=split["train"],
            cutoff=args.cutoff,
            max_neighbors=args.max_neighbors,
        )
        val_ds = dataset_cls(
            root=str(pyg_root / "val"),
            db_path=str(args.db),
            mp_ids=split["val"],
            cutoff=args.cutoff,
            max_neighbors=args.max_neighbors,
        )

        # PyG dataset doesn't need sampler - data is already in memory
        sampler = None

    elif args.backbone == "enhanced_graph" and (args.use_angles or args.use_deepaw_features):
        # Use EnhancedGraphDataset for angles or DeePAW features
        dataset_cls = EnhancedGraphDataset
        collate_fn = collate_enhanced_graph_samples
        dataset_kwargs = {
            "cutoff": args.cutoff,
            "max_neighbors": args.max_neighbors,
            "compute_angles": args.use_angles,
        }

        train_ds = dataset_cls(
            db_path=args.db,
            mp_ids=split["train"],
            **dataset_kwargs,
        )
        val_ds = dataset_cls(
            db_path=args.db,
            mp_ids=split["val"],
            **dataset_kwargs,
        )

        sampler = build_stage_sampler(train_ds, stage=args.stage, oversample_elastic=args.oversample_elastic, split_file=args.split)

    else:
        # Use original ASE dataset
        dataset_cls = AseGraphMultitaskDataset
        collate_fn = collate_graph_samples
        dataset_kwargs = {
            "cutoff": args.cutoff,
            "max_neighbors": args.max_neighbors,
        }

        train_ds = dataset_cls(
            db_path=args.db,
            mp_ids=split["train"],
            **dataset_kwargs,
        )
        val_ds = dataset_cls(
            db_path=args.db,
            mp_ids=split["val"],
            **dataset_kwargs,
        )

        sampler = build_stage_sampler(train_ds, stage=args.stage, oversample_elastic=args.oversample_elastic, split_file=args.split)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    enabled_tasks = stage_task_names(args.stage, exclude_tasks=args.exclude_tasks)
    if args.only_task:
        if args.only_task not in TASK_NAME_LIST:
            raise ValueError(f"Unknown task for --only-task: {args.only_task}")
        if args.only_task not in enabled_tasks:
            raise ValueError(
                f"Task '{args.only_task}' is not enabled for stage '{args.stage}'. "
                f"Current enabled tasks: {enabled_tasks}"
            )
        enabled_tasks = [args.only_task]

    # Extract masks differently for PyG vs ASE datasets
    if args.use_pyg:
        # PyG dataset: masks are already in the data
        print("Extracting masks from PyG dataset...")
        masks_list = []
        for i in range(len(train_ds)):
            data = train_ds[i]
            masks_list.append(data.mask.numpy())
        train_masks = np.stack(masks_list, axis=0)
        print(f"✅ Extracted masks for {len(train_masks)} samples")

        # Re-filter masks based on enabled_tasks
        # PyG dataset contains masks for ALL tasks, but we only want enabled ones
        from mp_data_pipeline.ml.tasks import TASK_INDEX
        enabled_indices = [TASK_INDEX[task] for task in enabled_tasks]
        disabled_indices = [i for i in range(len(TASK_NAME_LIST)) if i not in enabled_indices]

        # Zero out masks for disabled tasks
        if disabled_indices:
            print(f"Re-filtering masks: disabling {len(disabled_indices)} tasks not in enabled_tasks")
            train_masks[:, disabled_indices] = 0.0
            disabled_names = [TASK_NAME_LIST[i] for i in disabled_indices]
            print(f"Disabled tasks: {disabled_names}")
    else:
        # ASE dataset: compute masks with multiprocessing
        train_masks = dataset_masks(train_ds)

    task_weights = build_static_task_weights(train_masks, enabled_tasks)

    model = MultitaskPropertyModel(
        backbone_name=args.backbone,
        hidden_dim=args.hidden_dim,
        n_layers=args.layers,
        cutoff=args.cutoff,
        n_rbf=args.n_rbf,
        use_angles=args.use_angles,
        use_edge_update=args.use_edge_update,
        use_deepaw_features=args.use_deepaw_features,
        deepaw_checkpoint=args.deepaw_checkpoint if args.use_deepaw_features else None,
        deepaw_fusion=args.deepaw_fusion,
        head_variant=args.head_variant,
    )

    if args.init_checkpoint is not None:
        init_payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(init_payload["model_state"], strict=False)
        print(f"✅ Loaded init checkpoint: {args.init_checkpoint}")

    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("✅ Backbone frozen (only heads will be updated)")

    config = TrainerConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        amp=not args.no_amp,
    )

    model.to(torch.device(config.device))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = MultiTaskLoss(enabled_tasks=enabled_tasks, task_weights=task_weights)

    # Phase 2A: Initialize EMA
    ema = None
    if args.ema_decay > 0:
        ema = EMAModel(model, decay=args.ema_decay)
        print(f"✅ EMA enabled with decay={args.ema_decay}")

    # Phase 2A: Initialize warmup scheduler
    scheduler = None
    if args.warmup_epochs > 0:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        base_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
        scheduler = WarmupScheduler(optimizer, warmup_epochs=args.warmup_epochs, base_scheduler=base_scheduler)
        print(f"✅ Warmup enabled for {args.warmup_epochs} epochs")

    # Update config with new parameters
    config.grad_clip = args.grad_clip if args.grad_clip > 0 else 1.0  # Default to 1.0 if not specified

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / run_id
    ckpt_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "metrics"
    tb_dir = run_dir / "tensorboard"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Phase 2A: Initialize Best-K checkpoint manager (after ckpt_dir is defined)
    best_k_ckpts = None
    if args.best_k > 1:
        best_k_ckpts = BestKCheckpoints(save_dir=str(ckpt_dir), k=args.best_k, mode='min', prefix='best')
        print(f"✅ Best-K checkpointing enabled (K={args.best_k})")

    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=str(tb_dir))

    config_payload = {
        "db": str(args.db),
        "split": str(args.split),
        "stage": args.stage,
        "backbone": args.backbone,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "cutoff": args.cutoff,
        "max_neighbors": args.max_neighbors,
        "n_rbf": args.n_rbf,
        "use_angles": args.use_angles,
        "use_edge_update": args.use_edge_update,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "use_pyg": args.use_pyg,
        "device": args.device,
        "enabled_tasks": enabled_tasks,
        "head_variant": args.head_variant,
        "only_task": args.only_task,
        "task_weights": task_weights,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        # Phase 2A parameters
        "ema_decay": args.ema_decay,
        "warmup_epochs": args.warmup_epochs,
        "grad_clip": args.grad_clip,
        "best_k": args.best_k,
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "freeze_backbone": args.freeze_backbone,
        "n_trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "n_total_params": int(sum(p.numel() for p in model.parameters())),
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2))

    best_val = float("inf")
    history = []

    for epoch in range(1, config.epochs + 1):
        train_res = train_one_epoch(model, train_loader, optimizer, loss_fn, config, writer=writer, epoch=epoch)

        # Phase 2A: Update EMA after training
        if ema is not None:
            ema.update()

        # Phase 2A: Evaluate with EMA model if enabled
        if ema is not None:
            ema.apply_shadow()
            val_res = evaluate(model, val_loader, loss_fn, config, enabled_tasks, writer=writer, epoch=epoch, split_name="val")
            ema.restore()
        else:
            val_res = evaluate(model, val_loader, loss_fn, config, enabled_tasks, writer=writer, epoch=epoch, split_name="val")

        # Phase 2A: Step warmup scheduler
        if scheduler is not None:
            scheduler.step()

        # Log learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("train/lr", current_lr, epoch)

        log_row = {
            "epoch": epoch,
            "train_loss": train_res.loss,
            "val_loss": val_res.loss,
            **{f"train_{k}": v for k, v in train_res.task_metrics.items()},
            **{f"val_{k}": v for k, v in val_res.task_metrics.items()},
        }
        history.append(log_row)

        print(
            f"epoch={epoch:03d} train_loss={train_res.loss:.4f} "
            f"val_loss={val_res.loss:.4f}"
        )

        # Save best checkpoint (original method)
        if val_res.loss < best_val:
            best_val = val_res.loss
            checkpoint_dict = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "config": asdict(config),
                "best_metric": best_val,
                "enabled_tasks": enabled_tasks,
            }
            if ema is not None:
                checkpoint_dict["ema_state"] = ema.state_dict()
            if scheduler is not None:
                checkpoint_dict["scheduler_state"] = scheduler.state_dict()

            save_checkpoint(
                ckpt_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                best_metric=best_val,
                enabled_tasks=enabled_tasks,
            )

        # Phase 2A: Save to Best-K checkpoints
        if best_k_ckpts is not None:
            checkpoint_dict = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "config": asdict(config),
                "best_metric": val_res.loss,
                "enabled_tasks": enabled_tasks,
            }
            if ema is not None:
                checkpoint_dict["ema_state"] = ema.state_dict()
            if scheduler is not None:
                checkpoint_dict["scheduler_state"] = scheduler.state_dict()

            saved_path = best_k_ckpts.save(val_res.loss, checkpoint_dict, epoch)
            if saved_path:
                print(f"  💾 Saved to Best-K: {saved_path}")

    save_metrics(metrics_dir / "history.json", {"history": history})

    # Close tensorboard writer
    writer.close()

    # Evaluate best checkpoint on train/val splits.
    best_payload = torch.load(ckpt_dir / "best.pt", map_location=config.device, weights_only=False)
    model.load_state_dict(best_payload["model_state"])

    train_best = evaluate(model, train_loader, loss_fn, config, enabled_tasks)
    val_best = evaluate(model, val_loader, loss_fn, config, enabled_tasks)

    summary = {
        "best_epoch": best_payload["epoch"],
        "best_val_loss": best_payload["best_metric"],
        "train_metrics": {"loss": train_best.loss, **train_best.task_metrics},
        "val_metrics": {"loss": val_best.loss, **val_best.task_metrics},
    }
    save_metrics(metrics_dir / "best_summary.json", summary)

    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run EXP-208: DeePAW Zero Single-Task Family with per-task capacity sizing."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mp_data_pipeline.ml.tasks import TASK_NAME_LIST


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EXP-208: DeePAW zero-version single-task orchestration")
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "data" / "splits" / "split_iid_seed42.json")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "db" / "mp_materials.db")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--deepaw-checkpoint",
        type=Path,
        default=Path("/home/sutianhao/data/deepaw_test/DeePAW-main/checkpoints/f_nonlocal_escn_best.pth"),
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "stage_a" / "phase2_deepaw" / "exp208_deepaw_zero_single_task",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "runs_exp208",
    )
    return parser.parse_args()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def extract_run_dir(log_text: str) -> str:
    hits = re.findall(r"run_dir=(.+)", log_text)
    return hits[-1].strip() if hits else ""


def load_train_counts() -> Dict[str, int]:
    counts = {}
    root = PROJECT_ROOT / "reports" / "gpt_eval_v4_single_task"
    for task_dir in root.iterdir():
        if not task_dir.is_dir():
            continue
        payload = json.loads((task_dir / "results.json").read_text())
        counts[task_dir.name] = int(payload["train"]["metrics"][task_dir.name]["n_samples"])
    # exp208 also includes volume/density, estimate by direct PyG cache scan from current split
    if "volume" not in counts or "density" not in counts:
        from mp_data_pipeline.ml.pyg_dataset import PyGMaterialsDataset
        from mp_data_pipeline.ml.splits import load_split
        split = load_split(PROJECT_ROOT / "data" / "splits" / "split_iid_seed42.json")
        ds = PyGMaterialsDataset(
            root=str(PROJECT_ROOT / "data" / "pyg_cache" / "split_iid_seed42" / "train"),
            db_path=str(PROJECT_ROOT / "data" / "db" / "mp_materials.db"),
            mp_ids=split["train"],
            cutoff=6.0,
            max_neighbors=24,
        )
        task_index = {name: idx for idx, name in enumerate(TASK_NAME_LIST)}
        for task in ["volume", "density"]:
            idx = task_index[task]
            counts[task] = int(sum(float(ds[i].mask[idx]) > 0.5 for i in range(len(ds))))
    return counts


def capacity_policy(task: str, train_samples: int) -> Dict[str, object]:
    """Same capacity policy as exp106 for fair comparison."""
    if task in {"homogeneous_poisson", "universal_anisotropy"}:
        return {
            "hidden_dim": 160,
            "layers": 4,
            "batch_size": 32,
            "epochs": 80,
            "lr": 2e-4,
            "weight_decay": 1e-5,
            "grad_clip": 0.8,
            "no_amp": True,
        }
    if task in {"volume", "density"}:
        return {
            "hidden_dim": 320,
            "layers": 7,
            "batch_size": 32,
            "epochs": 70,
            "lr": 8e-5,
            "weight_decay": 1e-5,
            "grad_clip": 0.5,
            "no_amp": False,
        }
    if train_samples >= 100000:
        return {
            "hidden_dim": 256,
            "layers": 6,
            "batch_size": 64,
            "epochs": 50,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "grad_clip": 1.0,
            "no_amp": False,
        }
    if train_samples >= 50000:
        return {
            "hidden_dim": 224,
            "layers": 6,
            "batch_size": 64,
            "epochs": 55,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "grad_clip": 1.0,
            "no_amp": False,
        }
    return {
        "hidden_dim": 160,
        "layers": 4,
        "batch_size": 32,
        "epochs": 80,
        "lr": 2e-4,
        "weight_decay": 1e-5,
        "grad_clip": 0.8,
        "no_amp": False,
    }


def parse_best_summary(run_dir: Path, task: str) -> Dict[str, str]:
    best_path = run_dir / "metrics" / "best_summary.json"
    if not best_path.exists():
        return {"best_epoch": "", "best_val_loss": "", "val_primary_metric": ""}
    payload = json.loads(best_path.read_text())
    val_metrics = payload.get("val_metrics", {})
    primary_key = f"{task}_auroc" if task in {"is_metal", "is_stable"} else f"{task}_mae"
    return {
        "best_epoch": str(payload.get("best_epoch", "")),
        "best_val_loss": str(payload.get("best_val_loss", "")),
        "val_primary_metric": str(val_metrics.get(primary_key, "")),
    }


def main() -> None:
    args = parse_args()
    counts = load_train_counts()
    tasks = args.tasks if args.tasks else TASK_NAME_LIST
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]

    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    (args.experiment_dir / "logs").mkdir(parents=True, exist_ok=True)
    (args.experiment_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (args.experiment_dir / "tasks").mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "EXP-208: DeePAW Zero Single-Task Family - Enhanced Graph + DeePAW Replace mode",
        "split": str(args.split),
        "runs_root": str(args.runs_root),
        "deepaw_checkpoint": str(args.deepaw_checkpoint),
        "deepaw_fusion": "replace",
        "backbone": "enhanced_graph",
        "dry_run": args.dry_run,
        "tasks": tasks,
        "train_counts": {task: counts.get(task) for task in tasks},
        "capacity_policy": "volume/density->320x7; >=100k->256x6; >=50k->224x6; else->160x4",
    }
    write_text(args.experiment_dir / "manifest.json", json.dumps(manifest, indent=2))

    rows: List[Dict[str, str]] = []
    for task in tasks:
        train_samples = counts.get(task, 0)
        policy = capacity_policy(task, train_samples)
        task_dir = args.experiment_dir / "tasks" / task
        task_dir.mkdir(parents=True, exist_ok=True)
        task_runs_root = args.runs_root / task
        task_runs_root.mkdir(parents=True, exist_ok=True)

        # Build training command with DeePAW parameters
        cmd = [
            "python",
            "scripts/train_multitask.py",
            "--db", str(args.db),
            "--split", str(args.split),
            "--stage", "full",
            "--only-task", task,
            "--backbone", "enhanced_graph",
            "--use-deepaw-features",
            "--deepaw-checkpoint", str(args.deepaw_checkpoint),
            "--deepaw-fusion", "replace",
            "--hidden-dim", str(policy["hidden_dim"]),
            "--layers", str(policy["layers"]),
            "--cutoff", "6.0",
            "--max-neighbors", "24",
            "--n-rbf", "128",
            "--batch-size", str(policy["batch_size"]),
            "--epochs", str(policy["epochs"]),
            "--lr", str(policy["lr"]),
            "--weight-decay", str(policy["weight_decay"]),
            "--num-workers", "4",
            "--warmup-epochs", "5",
            "--grad-clip", str(policy["grad_clip"]),
            "--device", args.device,
            "--out-dir", str(task_runs_root),
            "--use-pyg",
        ]
        if policy.get("no_amp"):
            cmd.append("--no-amp")

        # Generate bash script for this task
        bash_script = f"""#!/bin/bash
# EXP-208: DeePAW Zero Single-Task - {task}
# Capacity: {policy['hidden_dim']}×{policy['layers']}, Samples: {train_samples}

set -e
export PYTHONNOUSERSITE=1

{' '.join(cmd)}
"""
        write_text(task_dir / "training_cmd.sh", bash_script)
        (task_dir / "training_cmd.sh").chmod(0o755)

        print(f"[{task}] capacity={policy['hidden_dim']}×{policy['layers']}, samples={train_samples}")

        if args.dry_run:
            print(f"  [DRY-RUN] Would execute: {' '.join(cmd)}")
            rows.append({
                "task": task,
                "train_samples": str(train_samples),
                "hidden_dim": str(policy["hidden_dim"]),
                "layers": str(policy["layers"]),
                "epochs": str(policy["epochs"]),
                "status": "dry_run",
            })
            continue

        # Execute training
        log_path = args.experiment_dir / "logs" / f"{task}.log"
        print(f"  Training... (log: {log_path})")
        with open(log_path, "w") as log_file:
            proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)

        if proc.returncode != 0:
            print(f"  ❌ Training failed (exit code {proc.returncode})")
            rows.append({
                "task": task,
                "train_samples": str(train_samples),
                "hidden_dim": str(policy["hidden_dim"]),
                "layers": str(policy["layers"]),
                "epochs": str(policy["epochs"]),
                "status": "failed",
                "exit_code": str(proc.returncode),
            })
            continue

        # Parse results
        log_text = log_path.read_text()
        run_dir_str = extract_run_dir(log_text)
        run_dir = Path(run_dir_str) if run_dir_str else None

        if run_dir and run_dir.exists():
            summary = parse_best_summary(run_dir, task)
            print(f"  ✅ Completed: epoch={summary['best_epoch']}, val_loss={summary['best_val_loss']}, metric={summary['val_primary_metric']}")
            rows.append({
                "task": task,
                "train_samples": str(train_samples),
                "hidden_dim": str(policy["hidden_dim"]),
                "layers": str(policy["layers"]),
                "epochs": str(policy["epochs"]),
                "status": "completed",
                "run_dir": str(run_dir),
                **summary,
            })
        else:
            print(f"  ⚠️  Completed but no run_dir found")
            rows.append({
                "task": task,
                "train_samples": str(train_samples),
                "hidden_dim": str(policy["hidden_dim"]),
                "layers": str(policy["layers"]),
                "epochs": str(policy["epochs"]),
                "status": "completed_no_dir",
            })

    # Write results CSV
    csv_path = args.experiment_dir / "metrics" / "exp208_results.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n✅ Results saved to {csv_path}")

    print(f"\n✅ EXP-208 orchestration complete!")
    print(f"   Experiment dir: {args.experiment_dir}")
    print(f"   Runs root: {args.runs_root}")


if __name__ == "__main__":
    main()

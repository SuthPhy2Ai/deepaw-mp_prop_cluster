#!/usr/bin/env python3
"""Run isolated zero-version single-task trainings with per-task capacity sizing."""

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
    parser = argparse.ArgumentParser(description="Zero-version single-task orchestration")
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "data" / "splits" / "split_iid_seed42.json")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "db" / "mp_materials.db")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-pyg", action="store_true")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "zero_version" / "exp106_zero_single_task_family",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "runs_zero",
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
    # zero version also includes volume/density, estimate by direct PyG cache scan from current split
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
    if task in {"homogeneous_poisson", "universal_anisotropy"}:
        return {
            "hidden_dim": 128,
            "layers": 4,
            "batch_size": 32,
            "epochs": 90,
            "lr": 8e-5,
            "weight_decay": 1e-5,
            "grad_clip": 0.5,
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
        "description": "Zero version: one model per property, no shared backbone, task-sized capacity",
        "split": str(args.split),
        "runs_root": str(args.runs_root),
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
        cmd = [
            "python",
            "scripts/train_multitask.py",
            "--db", str(args.db),
            "--split", str(args.split),
            "--stage", "full",
            "--only-task", task,
            "--backbone", "graph",
            "--hidden-dim", str(policy["hidden_dim"]),
            "--layers", str(policy["layers"]),
            "--cutoff", "6.0",
            "--max-neighbors", "24",
            "--n-rbf", "64",
            "--batch-size", str(policy["batch_size"]),
            "--epochs", str(policy["epochs"]),
            "--lr", str(policy["lr"]),
            "--weight-decay", str(policy["weight_decay"]),
            "--num-workers", "4",
            "--warmup-epochs", "5",
            "--grad-clip", str(policy["grad_clip"]),
            "--device", args.device,
            "--out-dir", str(task_runs_root),
        ]
        cmd.append("--no-pyg" if args.no_pyg else "--use-pyg")
        if policy["no_amp"]:
            cmd.append("--no-amp")
        write_text(task_dir / "training_cmd.sh", " ".join(cmd) + "\n")

        row = {
            "task": task,
            "train_samples": str(train_samples),
            "hidden_dim": str(policy["hidden_dim"]),
            "layers": str(policy["layers"]),
            "batch_size": str(policy["batch_size"]),
            "epochs": str(policy["epochs"]),
            "lr": str(policy["lr"]),
            "no_amp": str(policy["no_amp"]),
            "status": "planned" if args.dry_run else "running",
            "run_dir": "",
            "best_epoch": "",
            "best_val_loss": "",
            "val_primary_metric": "",
        }
        rows.append(row)
        with (args.experiment_dir / "metrics" / "zero_runs.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        if not args.dry_run:
            log_path = args.experiment_dir / "logs" / f"{task}.log"
            with log_path.open("w") as log_handle:
                log_handle.write(f"$ {' '.join(cmd)}\n\n")
                log_handle.flush()
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    text=True,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                )
                ret = proc.wait()
            if ret != 0:
                row["status"] = f"failed({ret})"
            else:
                row["status"] = "completed"
                row["run_dir"] = extract_run_dir(log_path.read_text())
                if row["run_dir"]:
                    row.update(parse_best_summary(Path(row["run_dir"]), task))

        with (args.experiment_dir / "metrics" / "zero_runs.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    completed = sum(1 for row in rows if row["status"] == "completed")
    failed = sum(1 for row in rows if row["status"].startswith("failed"))
    write_text(
        args.experiment_dir / "TRAINING_STATUS.md",
        "\n".join(
            [
                "# Zero 版本训练状态",
                "",
                f"- total_tasks: {len(rows)}",
                f"- completed: {completed}",
                f"- failed: {failed}",
                f"- dry_run: {args.dry_run}",
                "",
                "详情见: `metrics/zero_runs.csv` 与 `logs/*.log`",
            ]
        )
        + "\n",
    )
    print(f"saved_manifest={args.experiment_dir / 'manifest.json'}")
    print(f"saved_csv={args.experiment_dir / 'metrics' / 'zero_runs.csv'}")


if __name__ == "__main__":
    main()

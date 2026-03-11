#!/usr/bin/env python3
"""Run isolated Stage B v4 single-task fine-tuning jobs."""

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

from mp_data_pipeline.ml.tasks import stage_task_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage B v4 single-task orchestration")
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        required=True,
        help="Checkpoint used to initialize shared backbone and heads.",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "split_iid_seed42.json",
    )
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "db" / "mp_materials.db")
    parser.add_argument("--backbone", type=str, default="graph")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--max-neighbors", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tasks", nargs="+", default=None, help="Explicit task list to run.")
    parser.add_argument("--max-tasks", type=int, default=0, help="If >0, run first N tasks only.")
    parser.add_argument("--no-pyg", action="store_true", help="Disable PyG backend.")
    parser.add_argument("--dry-run", action="store_true", help="Only write plan and commands, do not train.")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "stage_b"
        / "phase4_single_task"
        / "exp104_stageb_v4_single_task_heads",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "runs_stageb_v4",
    )
    return parser.parse_args()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def extract_run_dir(stdout: str) -> str:
    m = re.findall(r"run_dir=(.+)", stdout)
    if not m:
        return ""
    return m[-1].strip()


def parse_best_summary(run_dir: Path, task: str) -> Dict[str, str]:
    best_path = run_dir / "metrics" / "best_summary.json"
    if not best_path.exists():
        return {
            "best_epoch": "",
            "best_val_loss": "",
            "val_primary_metric": "",
            "train_primary_metric": "",
        }
    payload = json.loads(best_path.read_text())
    val_metrics = payload.get("val_metrics", {})
    train_metrics = payload.get("train_metrics", {})
    primary_key = f"{task}_auroc" if task in {"is_metal", "is_stable"} else f"{task}_mae"
    return {
        "best_epoch": str(payload.get("best_epoch", "")),
        "best_val_loss": str(payload.get("best_val_loss", "")),
        "val_primary_metric": str(val_metrics.get(primary_key, "")),
        "train_primary_metric": str(train_metrics.get(primary_key, "")),
    }


def main() -> None:
    args = parse_args()
    if not args.base_checkpoint.exists():
        raise FileNotFoundError(f"base checkpoint not found: {args.base_checkpoint}")

    task_list: List[str] = args.tasks if args.tasks else stage_task_names("b")
    if args.max_tasks > 0:
        task_list = task_list[: args.max_tasks]

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    (args.experiment_dir / "tasks").mkdir(parents=True, exist_ok=True)
    (args.experiment_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (args.experiment_dir / "logs").mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": ts,
        "base_checkpoint": str(args.base_checkpoint),
        "split": str(args.split),
        "runs_root": str(args.runs_root),
        "dry_run": args.dry_run,
        "tasks": task_list,
    }
    write_text(args.experiment_dir / "manifest.json", json.dumps(manifest, indent=2))

    rows: List[Dict[str, str]] = [
        {
            "task": task,
            "status": "planned" if args.dry_run else "queued",
            "run_dir": "",
            "best_epoch": "",
            "best_val_loss": "",
            "val_primary_metric": "",
            "train_primary_metric": "",
        }
        for task in task_list
    ]
    with (args.experiment_dir / "metrics" / "single_task_runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    write_text(
        args.experiment_dir / "TRAINING_STATUS.md",
        "\n".join(
            [
                "# Stage B v4 单任务微调状态",
                "",
                f"- created_at: {ts}",
                f"- total_tasks: {len(rows)}",
                "- completed: 0",
                "- failed: 0",
                "- running: True",
                f"- dry_run: {args.dry_run}",
                "",
                "详情见: `metrics/single_task_runs.csv` 与 `logs/*.log`",
            ]
        )
        + "\n",
    )

    for i, task in enumerate(task_list):
        task_dir = args.experiment_dir / "tasks" / task
        task_dir.mkdir(parents=True, exist_ok=True)
        task_runs_root = args.runs_root / task
        task_runs_root.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python",
            "scripts/train_multitask.py",
            "--db",
            str(args.db),
            "--split",
            str(args.split),
            "--stage",
            "b",
            "--backbone",
            args.backbone,
            "--hidden-dim",
            str(args.hidden_dim),
            "--layers",
            str(args.layers),
            "--cutoff",
            str(args.cutoff),
            "--max-neighbors",
            str(args.max_neighbors),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
            "--only-task",
            task,
            "--init-checkpoint",
            str(args.base_checkpoint),
            "--freeze-backbone",
            "--out-dir",
            str(task_runs_root),
        ]
        if args.no_pyg:
            cmd.append("--no-pyg")
        else:
            cmd.append("--use-pyg")

        write_text(task_dir / "training_cmd.sh", " ".join(cmd) + "\n")

        row = rows[i]
        row["status"] = "planned" if args.dry_run else "running"
        print(f"[{i+1}/{len(task_list)}] task={task} status={row['status']}", flush=True)

        if not args.dry_run:
            log_path = args.experiment_dir / "logs" / f"{task}.log"
            with log_path.open("w") as log_f:
                log_f.write(f"$ {' '.join(cmd)}\n\n")
                log_f.flush()
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    text=True,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                )
                ret = proc.wait()
            if proc.returncode != 0:
                row["status"] = f"failed({ret})"
            else:
                run_dir_str = extract_run_dir(log_path.read_text())
                row["status"] = "completed"
                row["run_dir"] = run_dir_str
                if run_dir_str:
                    summary = parse_best_summary(Path(run_dir_str), task)
                    row.update(summary)
            print(
                f"[{i+1}/{len(task_list)}] task={task} done status={row['status']} run_dir={row['run_dir']}",
                flush=True,
            )

        with (args.experiment_dir / "metrics" / "single_task_runs.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    done = sum(1 for r in rows if r["status"] == "completed")
    failed = sum(1 for r in rows if r["status"].startswith("failed"))
    write_text(
        args.experiment_dir / "TRAINING_STATUS.md",
        "\n".join(
            [
                "# Stage B v4 单任务微调状态",
                "",
                f"- created_at: {ts}",
                f"- total_tasks: {len(rows)}",
                f"- completed: {done}",
                f"- failed: {failed}",
                f"- running: {not args.dry_run and (done + failed) < len(rows)}",
                f"- dry_run: {args.dry_run}",
                "",
                "详情见: `metrics/single_task_runs.csv` 与 `logs/*.log`",
            ]
        )
        + "\n",
    )

    print(f"saved_manifest={args.experiment_dir / 'manifest.json'}")
    print(f"saved_status={args.experiment_dir / 'TRAINING_STATUS.md'}")
    print(f"saved_csv={args.experiment_dir / 'metrics' / 'single_task_runs.csv'}")


if __name__ == "__main__":
    main()

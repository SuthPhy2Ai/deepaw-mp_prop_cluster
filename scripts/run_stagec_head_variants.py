#!/usr/bin/env python3
"""Run isolated Stage C head-architecture experiments sequentially."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_SPECS = [
    {
        "id": "exp107",
        "name": "stagec_h1_electronic_hier",
        "label": "Stage C h1",
        "head_variant": "stagec_h1",
        "description": "Electronic hierarchical head: group head -> task heads for band tasks",
        "config": PROJECT_ROOT / "configs" / "exp107_stagec_h1_electronic_hier.json",
        "experiment_dir": PROJECT_ROOT / "experiments" / "stage_c" / "phase1_head_variants" / "exp107_stagec_h1_electronic_hier",
        "runs_root": PROJECT_ROOT / "artifacts" / "runs_stagec_h1",
    },
    {
        "id": "exp108",
        "name": "stagec_h2_elastic_derived",
        "label": "Stage C h2",
        "head_variant": "stagec_h2",
        "description": "Elastic derived head: base moduli head -> derived Poisson/anisotropy heads",
        "config": PROJECT_ROOT / "configs" / "exp108_stagec_h2_elastic_derived.json",
        "experiment_dir": PROJECT_ROOT / "experiments" / "stage_c" / "phase1_head_variants" / "exp108_stagec_h2_elastic_derived",
        "runs_root": PROJECT_ROOT / "artifacts" / "runs_stagec_h2",
    },
    {
        "id": "exp109",
        "name": "stagec_hybrid_hier_combo",
        "label": "Stage C hybrid",
        "head_variant": "stagec_hybrid",
        "description": "Hybrid hierarchical head: electronic hierarchy + elastic derived hierarchy",
        "config": PROJECT_ROOT / "configs" / "exp109_stagec_hybrid_hier_combo.json",
        "experiment_dir": PROJECT_ROOT / "experiments" / "stage_c" / "phase1_head_variants" / "exp109_stagec_hybrid_hier_combo",
        "runs_root": PROJECT_ROOT / "artifacts" / "runs_stagec_hybrid",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage C head-variant orchestration")
    parser.add_argument("--base-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts" / "runs_stageb_v3" / "20260308_070539" / "checkpoints" / "best.pt")
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "data" / "splits" / "split_iid_seed42.json")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "db" / "mp_materials.db")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backbone", type=str, default="graph")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--max-neighbors", type=int, default=24)
    parser.add_argument("--n-rbf", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--no-pyg", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--experiments", nargs="+", default=None, help="Subset: exp107 exp108 exp109")
    parser.add_argument(
        "--family-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "stage_c" / "phase1_head_variants" / "family_stagec_head_variants",
    )
    return parser.parse_args()


def selected_specs(exp_ids: list[str] | None) -> list[dict]:
    if not exp_ids:
        return EXPERIMENT_SPECS
    wanted = set(exp_ids)
    return [spec for spec in EXPERIMENT_SPECS if spec["id"] in wanted]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def extract_run_dir(log_text: str) -> str:
    hits = re.findall(r"run_dir=(.+)", log_text)
    return hits[-1].strip() if hits else ""


def parse_best_summary(run_dir: Path, variant: str) -> Dict[str, str]:
    best_path = run_dir / "metrics" / "best_summary.json"
    if not best_path.exists():
        return {"best_epoch": "", "best_val_loss": ""}
    payload = json.loads(best_path.read_text())
    return {
        "best_epoch": str(payload.get("best_epoch", "")),
        "best_val_loss": str(payload.get("best_val_loss", "")),
    }


def write_experiment_config(spec: dict, args: argparse.Namespace) -> None:
    payload = {
        "experiment_id": spec["id"],
        "experiment_name": spec["name"],
        "description": spec["description"],
        "phase": "stage_c",
        "base_checkpoint": str(args.base_checkpoint.relative_to(PROJECT_ROOT)),
        "model": {
            "backbone": args.backbone,
            "head_variant": spec["head_variant"],
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "cutoff": args.cutoff,
            "max_neighbors": args.max_neighbors,
            "n_rbf": args.n_rbf,
            "use_pyg": not args.no_pyg,
            "freeze_backbone": args.freeze_backbone,
        },
        "training": {
            "split": str(args.split.relative_to(PROJECT_ROOT)),
            "stage": "b",
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "grad_clip": args.grad_clip,
            "device": args.device,
        },
        "isolation": {
            "experiment_dir": str(spec["experiment_dir"].relative_to(PROJECT_ROOT)),
            "runs_root": str(spec["runs_root"].relative_to(PROJECT_ROOT)),
            "log_file": str((spec["experiment_dir"] / "logs" / "train.log").relative_to(PROJECT_ROOT)),
        },
    }
    write_text(spec["config"], json.dumps(payload, indent=2) + "\n")


def training_cmd(spec: dict, args: argparse.Namespace) -> list[str]:
    cmd = [
        "python",
        "scripts/train_multitask.py",
        "--db", str(args.db),
        "--split", str(args.split),
        "--stage", "b",
        "--backbone", args.backbone,
        "--head-variant", spec["head_variant"],
        "--hidden-dim", str(args.hidden_dim),
        "--layers", str(args.layers),
        "--cutoff", str(args.cutoff),
        "--max-neighbors", str(args.max_neighbors),
        "--n-rbf", str(args.n_rbf),
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--num-workers", str(args.num_workers),
        "--warmup-epochs", str(args.warmup_epochs),
        "--grad-clip", str(args.grad_clip),
        "--device", args.device,
        "--init-checkpoint", str(args.base_checkpoint),
        "--out-dir", str(spec["runs_root"]),
    ]
    if args.freeze_backbone:
        cmd.append("--freeze-backbone")
    cmd.append("--no-pyg" if args.no_pyg else "--use-pyg")
    return cmd


def write_plan_docs(spec: dict, args: argparse.Namespace) -> None:
    exp_dir = spec["experiment_dir"]
    lines = [
        f"# {spec['label']} Plan",
        "",
        f"- experiment_id: `{spec['id']}`",
        f"- head_variant: `{spec['head_variant']}`",
        f"- description: {spec['description']}",
        f"- base_checkpoint: `{args.base_checkpoint.relative_to(PROJECT_ROOT)}`",
        f"- stage: `b`",
        f"- freeze_backbone: `{args.freeze_backbone}`",
        f"- use_pyg: `{not args.no_pyg}`",
        f"- epochs: `{args.epochs}`",
        f"- lr: `{args.lr}`",
        f"- runs_root: `{spec['runs_root'].relative_to(PROJECT_ROOT)}`",
        "",
        "## Intent",
        "",
        "- h1: test group-head to task-head hierarchy for electronic tasks.",
        "- h2: test base-elastic to derived-elastic hierarchy for Poisson/anisotropy.",
        "- hybrid: combine h1 and h2 in one shared-backbone experiment.",
    ]
    write_text(exp_dir / "PLAN.md", "\n".join(lines) + "\n")
    write_text(exp_dir / "README.md", f"# {spec['label']}\n\n{spec['description']}\n")


def main() -> None:
    args = parse_args()
    specs = selected_specs(args.experiments)
    if not args.base_checkpoint.exists():
        raise FileNotFoundError(f"base checkpoint not found: {args.base_checkpoint}")
    args.family_dir.mkdir(parents=True, exist_ok=True)
    (args.family_dir / "logs").mkdir(parents=True, exist_ok=True)

    family_rows: List[Dict[str, str]] = []
    for spec in specs:
        spec["experiment_dir"].mkdir(parents=True, exist_ok=True)
        (spec["experiment_dir"] / "logs").mkdir(parents=True, exist_ok=True)
        (spec["experiment_dir"] / "metrics").mkdir(parents=True, exist_ok=True)
        spec["runs_root"].mkdir(parents=True, exist_ok=True)
        write_experiment_config(spec, args)
        write_plan_docs(spec, args)
        cmd = training_cmd(spec, args)
        write_text(spec["experiment_dir"] / "training_cmd.sh", " ".join(cmd) + "\n")
        write_text(
            spec["experiment_dir"] / "manifest.json",
            json.dumps(
                {
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "experiment_id": spec["id"],
                    "head_variant": spec["head_variant"],
                    "command": cmd,
                    "dry_run": args.dry_run,
                },
                indent=2,
            ) + "\n",
        )

        row = {
            "experiment_id": spec["id"],
            "label": spec["label"],
            "head_variant": spec["head_variant"],
            "status": "planned" if args.dry_run else "running",
            "run_dir": "",
            "best_epoch": "",
            "best_val_loss": "",
            "experiment_dir": str(spec["experiment_dir"].relative_to(PROJECT_ROOT)),
        }
        family_rows.append(row)

        if not args.dry_run:
            log_path = spec["experiment_dir"] / "logs" / "train.log"
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
            if ret != 0:
                row["status"] = f"failed({ret})"
            else:
                row["status"] = "completed"
                run_dir = extract_run_dir(log_path.read_text())
                row["run_dir"] = run_dir
                if run_dir:
                    row.update(parse_best_summary(Path(run_dir), spec["head_variant"]))

        write_text(
            spec["experiment_dir"] / "TRAINING_STATUS.md",
            "\n".join(
                [
                    f"# {spec['label']} 训练状态",
                    "",
                    f"- status: {row['status']}",
                    f"- head_variant: `{spec['head_variant']}`",
                    f"- run_dir: `{row['run_dir'] or 'N/A'}`",
                    f"- best_epoch: `{row['best_epoch'] or 'N/A'}`",
                    f"- best_val_loss: `{row['best_val_loss'] or 'N/A'}`",
                    f"- train_log: `{(spec['experiment_dir'] / 'logs' / 'train.log').relative_to(PROJECT_ROOT)}`",
                ]
            ) + "\n",
        )
        with (args.family_dir / "stagec_family_status.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(family_rows[0].keys()))
            writer.writeheader()
            writer.writerows(family_rows)

    write_text(
        args.family_dir / "QUEUE_STATUS.md",
        "\n".join(
            [
                "# Stage C Head Variants Queue Status",
                "",
                f"- created_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"- dry_run: {args.dry_run}",
                "",
                "详情见: `stagec_family_status.csv` 与各实验目录下的 `TRAINING_STATUS.md`。",
            ]
        ) + "\n",
    )
    print(f"saved_family_status={args.family_dir / 'stagec_family_status.csv'}")
    print(f"saved_queue_status={args.family_dir / 'QUEUE_STATUS.md'}")


if __name__ == "__main__":
    main()

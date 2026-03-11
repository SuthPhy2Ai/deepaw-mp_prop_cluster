#!/usr/bin/env python
"""Experiment management tool for Phase 2 training."""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def create_experiment(exp_id: str, exp_name: str, stage: str = "a", phase: int = 2):
    """Create a new experiment directory structure."""
    base_dir = Path("/home/sutianhao/data/mp-data-pipeline")

    # Determine stage directory based on experiment ID
    if exp_id.startswith("exp0") or (exp_id.startswith("exp") and int(exp_id[3:6]) < 100):
        stage_dir = "stage_a"
        phase_subdir = f"phase{phase}_baseline" if phase == 1 else f"phase{phase}_enhancements"
    else:
        stage_dir = "stage_b"
        phase_subdir = f"phase{phase}_baseline" if phase == 3 else f"phase{phase}_enhancements"

    exp_dir = base_dir / "experiments" / stage_dir / phase_subdir / f"{exp_id}_{exp_name}"

    if exp_dir.exists():
        print(f"❌ Experiment {exp_id}_{exp_name} already exists!")
        return False

    # Create directory structure
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "analysis").mkdir(exist_ok=True)
    (exp_dir / "tensorboard").mkdir(exist_ok=True)
    (exp_dir / "metrics").mkdir(exist_ok=True)

    # Create README template
    readme_content = f"""# Experiment {exp_id}: {exp_name.replace('_', ' ').title()}

**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Status**: 🔄 Running
**Phase**: {phase}

---

## Overview

[Brief description of this experiment]

## Configuration

### Changes from Baseline

[List key changes]

### Training Setup

[Training hyperparameters]

## Results

[To be filled after training]

## Key Findings

[To be filled after analysis]

## Files

- `model_checkpoint.pt` - Best model weights
- `config.json` - Full training configuration
- `training_log.txt` - Console output
- `tensorboard/` - Training curves
- `analysis/` - Post-training analysis

## Next Steps

[Future improvements or follow-up experiments]
"""

    (exp_dir / "README.md").write_text(readme_content)

    print(f"✅ Created experiment directory: {exp_dir}")
    return True


def run_training(exp_id: str, exp_name: str, config: dict, stage: str = "a", phase: int = 2):
    """Run training for an experiment."""
    base_dir = Path("/home/sutianhao/data/mp-data-pipeline")

    # Determine stage directory based on experiment ID
    if exp_id.startswith("exp0") or (exp_id.startswith("exp") and int(exp_id[3:6]) < 100):
        stage_dir = "stage_a"
        phase_subdir = f"phase{phase}_baseline" if phase == 1 else f"phase{phase}_enhancements"
    else:
        stage_dir = "stage_b"
        phase_subdir = f"phase{phase}_baseline" if phase == 3 else f"phase{phase}_enhancements"

    exp_dir = base_dir / "experiments" / stage_dir / phase_subdir / f"{exp_id}_{exp_name}"

    if not exp_dir.exists():
        print(f"❌ Experiment directory does not exist. Create it first.")
        return False

    # Build training command
    cmd = [
        "python", "scripts/train_multitask.py",
        "--split", config.get("split", "data/splits/split_iid_seed42.json"),
        "--stage", config.get("stage", "a"),
        "--backbone", config.get("backbone", "graph"),
        "--batch-size", str(config.get("batch_size", 64)),
        "--num-workers", str(config.get("num_workers", 8)),
        "--exclude-tasks", "volume", "density", "is_stable",
        "--epochs", str(config.get("epochs", 100)),
        "--lr", str(config.get("lr", 1e-4)),
        "--weight-decay", str(config.get("weight_decay", 1e-5)),
        "--grad-clip", str(config.get("grad_clip", 1.0)),
        "--warmup-epochs", str(config.get("warmup_epochs", 5)),
    ]

    # Add optional parameters
    if config.get("cutoff"):
        cmd.extend(["--cutoff", str(config["cutoff"])])
    if config.get("max_neighbors"):
        cmd.extend(["--max-neighbors", str(config["max_neighbors"])])
    if config.get("n_rbf"):
        cmd.extend(["--n-rbf", str(config["n_rbf"])])
    if config.get("use_angles"):
        cmd.append("--use-angles")
    if config.get("use_edge_update"):
        cmd.append("--use-edge-update")
    if config.get("dropout"):
        cmd.extend(["--dropout", str(config["dropout"])])
    if config.get("ema_decay"):
        cmd.extend(["--ema-decay", str(config["ema_decay"])])

    # Save config
    config_path = exp_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Run training
    log_file = exp_dir / "training_log.txt"
    print(f"🚀 Starting training for {exp_id}_{exp_name}")
    print(f"📝 Logging to: {log_file}")
    print(f"💻 Command: {' '.join(cmd)}")

    with open(log_file, 'w') as f:
        f.write(f"Training started: {datetime.now()}\n")
        f.write(f"Command: {' '.join(cmd)}\n\n")

    # Run in background with nohup
    nohup_cmd = f"nohup {' '.join(cmd)} >> {log_file} 2>&1 &"
    subprocess.run(nohup_cmd, shell=True, cwd=base_dir)

    print(f"✅ Training started in background")
    print(f"📊 Monitor with: tail -f {log_file}")

    return True


def run_analysis(exp_id: str, exp_name: str, stage: str = "a", phase: int = 2):
    """Run post-training analysis for an experiment."""
    base_dir = Path("/home/sutianhao/data/mp-data-pipeline")

    # Determine stage directory based on experiment ID
    if exp_id.startswith("exp0") or (exp_id.startswith("exp") and int(exp_id[3:6]) < 100):
        stage_dir = "stage_a"
        phase_subdir = f"phase{phase}_baseline" if phase == 1 else f"phase{phase}_enhancements"
    else:
        stage_dir = "stage_b"
        phase_subdir = f"phase{phase}_baseline" if phase == 3 else f"phase{phase}_enhancements"

    exp_dir = base_dir / "experiments" / stage_dir / phase_subdir / f"{exp_id}_{exp_name}"

    checkpoint = exp_dir / "model_checkpoint.pt"
    if not checkpoint.exists():
        print(f"❌ Checkpoint not found: {checkpoint}")
        return False

    analysis_dir = exp_dir / "analysis"

    cmd = [
        "python", "scripts/analyze_best_model.py",
        "--checkpoint", str(checkpoint),
        "--split", "data/splits/split_iid_seed42.json",
        "--output-dir", str(analysis_dir),
        "--batch-size", "64",
        "--num-workers", "8",
        "--device", "cuda"
    ]

    print(f"🔍 Running analysis for {exp_id}_{exp_name}")
    print(f"💻 Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=base_dir)

    if result.returncode == 0:
        print(f"✅ Analysis complete: {analysis_dir}")
        return True
    else:
        print(f"❌ Analysis failed")
        return False


def list_experiments(stage: str = None):
    """List all experiments."""
    base_dir = Path("/home/sutianhao/data/mp-data-pipeline/experiments")

    stages = [f"stage_{stage}"] if stage else ["stage_a", "stage_b"]

    for stage_dir in stages:
        stage_path = base_dir / stage_dir
        if not stage_path.exists():
            continue

        print(f"\n{'='*60}")
        print(f"  {stage_dir.upper().replace('_', ' ')}")
        print(f"{'='*60}")

        # List all phase subdirectories
        phase_dirs = sorted([d for d in stage_path.iterdir() if d.is_dir() and d.name.startswith("phase")])

        if not phase_dirs:
            print("  No experiments found")
            continue

        for phase_dir in phase_dirs:
            experiments = sorted([d for d in phase_dir.iterdir() if d.is_dir() and d.name.startswith("exp")])

            if experiments:
                print(f"\n  {phase_dir.name}:")
                for exp_dir in experiments:
                    readme = exp_dir / "README.md"
                    checkpoint = exp_dir / "model_checkpoint.pt"

                    status = "✅" if checkpoint.exists() else "🔄"

                    print(f"\n    {status} {exp_dir.name}")
                    print(f"       Path: {exp_dir}")

                    if readme.exists():
                        # Extract status from README
                        content = readme.read_text()
                        for line in content.split('\n'):
                            if line.startswith('**Status**:'):
                                print(f"       {line}")
                                break


def main():
    parser = argparse.ArgumentParser(description="Experiment management tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Create experiment
    create_parser = subparsers.add_parser("create", help="Create new experiment")
    create_parser.add_argument("exp_id", help="Experiment ID (e.g., exp002)")
    create_parser.add_argument("exp_name", help="Experiment name (e.g., regularization)")
    create_parser.add_argument("--phase", type=int, default=2, help="Phase number")

    # Run training
    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument("exp_id", help="Experiment ID")
    train_parser.add_argument("exp_name", help="Experiment name")
    train_parser.add_argument("--config", required=True, help="Path to config JSON file")
    train_parser.add_argument("--phase", type=int, default=2, help="Phase number")

    # Run analysis
    analyze_parser = subparsers.add_parser("analyze", help="Run analysis")
    analyze_parser.add_argument("exp_id", help="Experiment ID")
    analyze_parser.add_argument("exp_name", help="Experiment name")
    analyze_parser.add_argument("--phase", type=int, default=2, help="Phase number")

    # List experiments
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--phase", type=int, help="Filter by phase")

    args = parser.parse_args()

    if args.command == "create":
        create_experiment(args.exp_id, args.exp_name, args.phase)
    elif args.command == "train":
        with open(args.config) as f:
            config = json.load(f)
        run_training(args.exp_id, args.exp_name, config, args.phase)
    elif args.command == "analyze":
        run_analysis(args.exp_id, args.exp_name, args.phase)
    elif args.command == "list":
        list_experiments(args.phase)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

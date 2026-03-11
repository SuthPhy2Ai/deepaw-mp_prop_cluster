#!/usr/bin/env python3
"""Monitor training progress in real-time."""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_latest_run():
    """Get the most recent run directory."""
    runs_dir = PROJECT_ROOT / "artifacts" / "runs"
    if not runs_dir.exists():
        return None
    runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
    return runs[-1] if runs else None


def monitor_run(run_dir):
    """Monitor a training run."""
    print(f"Monitoring run: {run_dir.name}")
    print("=" * 60)

    history_file = run_dir / "metrics" / "history.json"
    last_epoch = 0

    while True:
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                    history = data.get("history", [])

                    if len(history) > last_epoch:
                        for entry in history[last_epoch:]:
                            epoch = entry["epoch"]
                            train_loss = entry["train_loss"]
                            val_loss = entry["val_loss"]

                            # Get key metrics
                            is_metal_auroc = entry.get("val_is_metal_auroc", 0)
                            band_gap_mae = entry.get("val_band_gap_mae", 0)

                            print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                                  f"is_metal_auroc={is_metal_auroc:.4f}, band_gap_mae={band_gap_mae:.4f}")

                        last_epoch = len(history)

                        # Check if training is complete
                        config_file = run_dir / "config.json"
                        if config_file.exists():
                            with open(config_file) as f:
                                config = json.load(f)
                                total_epochs = config.get("epochs", 0)
                                if last_epoch >= total_epochs:
                                    print("\n" + "=" * 60)
                                    print("Training complete!")

                                    # Show final summary
                                    summary_file = run_dir / "metrics" / "best_summary.json"
                                    if summary_file.exists():
                                        with open(summary_file) as f:
                                            summary = json.load(f)
                                            print(f"\nBest epoch: {summary['best_epoch']}")
                                            print(f"Best val loss: {summary['best_val_loss']:.4f}")
                                            print("\nValidation metrics:")
                                            for k, v in summary['val_metrics'].items():
                                                print(f"  {k}: {v:.4f}")
                                    return
            except (json.JSONDecodeError, KeyError):
                pass

        time.sleep(10)  # Check every 10 seconds


def main():
    """Main monitoring loop."""
    print("Waiting for training to start...")

    # Wait for a run to appear
    while True:
        run_dir = get_latest_run()
        if run_dir:
            # Wait a bit to ensure the run has started writing
            time.sleep(5)
            monitor_run(run_dir)
            break
        time.sleep(5)


if __name__ == "__main__":
    main()

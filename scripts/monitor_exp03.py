#!/usr/bin/env python3
"""Monitor EXP-03 training progress."""

import time
import json
from pathlib import Path
from datetime import datetime

def get_latest_run():
    """Get the most recent run directory."""
    runs_dir = Path("artifacts/runs")
    if not runs_dir.exists():
        return None

    runs = sorted(runs_dir.iterdir(), key=lambda x: x.name, reverse=True)
    for run in runs:
        config_file = run / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
                if config.get("backbone") == "enhanced_graph" and config.get("epochs") == 50:
                    return run
    return None

def monitor():
    """Monitor training progress."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Monitoring EXP-03 training...")
    print("=" * 80)

    last_epoch = -1
    start_time = time.time()

    while True:
        run_dir = get_latest_run()

        if run_dir is None:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for training to start...")
            time.sleep(30)
            continue

        # Check config
        config_file = run_dir / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)

            if last_epoch == -1:
                print(f"\n✅ Training started!")
                print(f"Run directory: {run_dir.name}")
                print(f"Configuration:")
                print(f"  - Backbone: {config['backbone']}")
                print(f"  - Cutoff: {config['cutoff']} Å")
                print(f"  - Max neighbors: {config['max_neighbors']}")
                print(f"  - N_RBF: {config['n_rbf']}")
                print(f"  - Hidden dim: {config['hidden_dim']}")
                print(f"  - Layers: {config['layers']}")
                print(f"  - Learning rate: {config['lr']}")
                print(f"  - Batch size: {config['batch_size']}")
                print(f"  - Epochs: {config['epochs']}")
                print("=" * 80)

        # Check history
        history_file = run_dir / "metrics" / "history.json"
        if history_file.exists():
            with open(history_file) as f:
                data = json.load(f)
                history = data.get("history", [])

            if history:
                latest = history[-1]
                epoch = latest["epoch"]

                if epoch > last_epoch:
                    elapsed = time.time() - start_time
                    elapsed_min = elapsed / 60

                    train_loss = latest["train_loss"]
                    val_loss = latest["val_loss"]

                    # Extract key metrics
                    is_metal_auroc = latest.get("val_is_metal_auroc", 0)
                    band_gap_mae = latest.get("val_band_gap_mae", 0)
                    volume_mae = latest.get("val_volume_mae", 0)

                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch:02d}/50 | "
                          f"Train: {train_loss:.2f} | Val: {val_loss:.2f} | "
                          f"AUROC: {is_metal_auroc:.4f} | BG: {band_gap_mae:.3f} | "
                          f"Vol: {volume_mae:.1f} | Time: {elapsed_min:.1f}m")

                    last_epoch = epoch

                    if epoch >= 50:
                        print("\n" + "=" * 80)
                        print("✅ Training completed!")
                        print(f"Total time: {elapsed_min:.1f} minutes ({elapsed_min/60:.1f} hours)")

                        # Show best results
                        best_summary = run_dir / "metrics" / "best_summary.json"
                        if best_summary.exists():
                            with open(best_summary) as f:
                                best = json.load(f)

                            print(f"\nBest results (epoch {best['best_epoch']}):")
                            val_metrics = best['val_metrics']
                            print(f"  - is_metal AUROC: {val_metrics.get('is_metal_auroc', 0):.4f}")
                            print(f"  - is_stable AUROC: {val_metrics.get('is_stable_auroc', 0):.4f}")
                            print(f"  - band_gap MAE: {val_metrics.get('band_gap_mae', 0):.3f} eV")
                            print(f"  - volume MAE: {val_metrics.get('volume_mae', 0):.1f} Å³")
                            print(f"  - energy_above_hull MAE: {val_metrics.get('energy_above_hull_mae', 0):.3f} eV")

                        return

        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")

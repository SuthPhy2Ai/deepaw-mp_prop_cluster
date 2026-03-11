#!/usr/bin/env python3
"""Real-time monitor for training progress with data loading detection."""

import time
import subprocess
from pathlib import Path
from datetime import datetime

def check_process(pid):
    """Check if process is running."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pid,pcpu,pmem,etime,cmd"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0, result.stdout
    except:
        return False, ""

def check_gpu():
    """Check GPU usage."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            gpu_util, mem_used = result.stdout.strip().split(", ")
            return int(gpu_util), int(mem_used)
    except:
        pass
    return 0, 0

def get_latest_run():
    """Get the most recent run directory."""
    runs_dir = Path("artifacts/runs")
    if not runs_dir.exists():
        return None

    runs = sorted(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
    return runs[0] if runs else None

def main():
    pid = 2882213
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Monitoring PID {pid}")
    print("=" * 80)

    start_time = time.time()
    last_status = None

    while True:
        running, ps_output = check_process(pid)

        if not running:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Process stopped!")
            break

        elapsed = time.time() - start_time
        elapsed_min = elapsed / 60

        # Check GPU
        gpu_util, gpu_mem = check_gpu()

        # Check for new run directory
        latest_run = get_latest_run()

        # Determine status
        if latest_run and (latest_run.name.startswith("202603") or latest_run.name.startswith("202604")):
            # Check if it's from today
            run_time = latest_run.stat().st_mtime
            if time.time() - run_time < 3600:  # Within last hour
                status = "TRAINING"
            else:
                status = "DATA_LOADING"
        else:
            status = "DATA_LOADING"

        if status != last_status:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status changed: {status}")
            last_status = status

        # Print status line
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"Status: {status:12s} | "
              f"Time: {elapsed_min:5.1f}m | "
              f"GPU: {gpu_util:2d}% / {gpu_mem:5d}MB",
              end="\r", flush=True)

        time.sleep(10)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

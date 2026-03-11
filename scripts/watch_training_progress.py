#!/usr/bin/env python3
"""
Lightweight training monitor that appends periodic status lines to a log.

Why this exists:
- Some phases (e.g. PyG cache conversion / preloading) are CPU-heavy and can look "stuck".
- This script provides a continuous heartbeat with progress/GPU/CPU snapshots.

Example:
  python scripts/watch_training_progress.py \
    --pattern "python scripts/train_multitask.py.*runs_exp201" \
    --train-log logs/exp201_add_full.log \
    --watch-log logs/exp201_watch.log \
    --interval 30
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()


def _find_pid(pattern: str) -> Optional[int]:
    try:
        out = _run(["bash", "-lc", f"pgrep -af '{pattern}' || true"])
    except Exception:
        return None
    if not out:
        return None

    self_pid = os.getpid()
    for line in out.splitlines():
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmdline = parts[1] if len(parts) > 1 else ""

        # Avoid matching the watcher itself (its argv contains the pattern string).
        if pid == self_pid:
            continue
        if "watch_training_progress.py" in cmdline:
            continue
        return pid

    return None


def _tail_progress(train_log: Path) -> Tuple[Optional[str], Optional[float]]:
    """
    Return (progress_text, progress_ratio) if found.
    """
    if not train_log.exists():
        return None, None

    # Read last ~400KB for speed.
    try:
        data = train_log.read_bytes()
    except Exception:
        return None, None
    chunk = data[-400_000:]
    text = chunk.decode("utf-8", errors="ignore")

    # Phase 0: target/mask preloading (ASE dataset).
    m = None
    # tqdm format example:
    # "Preloading:   7%|▋         | 8665/123903 [03:10<42:12, 45.53it/s]"
    for mm in re.finditer(r"Preloading:.*?\|\s*(\d+)/(\d+)\s+\[", text):
        m = mm
    if m:
        cur = int(m.group(1))
        tot = int(m.group(2))
        ratio = (cur / tot) if tot else None
        return f"preload {cur}/{tot} ({ratio*100:.2f}%)", ratio

    # Phase 1: PyG conversion.
    m = None
    for mm in re.finditer(r"Converting to PyG:.*?\|\s+(\d+)/(\d+)\s+\[", text):
        m = mm
    if m:
        cur = int(m.group(1))
        tot = int(m.group(2))
        ratio = (cur / tot) if tot else None
        return f"pyg_convert {cur}/{tot} ({ratio*100:.2f}%)", ratio

    # Phase 2: training epoch iterations (printed by tqdm).
    m = None
    for mm in re.finditer(r"\[Epoch\s+(\d+)\s+Iter\s+(\d+)/(\d+)\]", text):
        m = mm
    if m:
        epoch = int(m.group(1))
        it = int(m.group(2))
        tot = int(m.group(3))
        ratio = (it / tot) if tot else None
        return f"train epoch={epoch} iter={it}/{tot} ({ratio*100:.2f}%)", ratio

    # Phase 3: eval loops.
    m = None
    for mm in re.finditer(r"eval:\s+(\d+)%\|", text):
        m = mm
    if m:
        return "eval running", None

    return None, None


def _ps_snapshot(pid: int) -> str:
    try:
        out = _run(["bash", "-lc", f"ps -p {pid} -o pid,etime,pcpu,pmem,rss,cmd --no-headers || true"])
        return out if out else "ps: <missing>"
    except Exception as e:
        return f"ps: <error {type(e).__name__}>"


def _gpu_snapshot() -> str:
    try:
        out = _run(
            [
                "bash",
                "-lc",
                "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total "
                "--format=csv,noheader,nounits 2>/dev/null | head -n 1 || true",
            ]
        )
        if not out:
            return "gpu: n/a"
        gpu, memu, used, total = [x.strip() for x in out.split(",")]
        return f"gpu_util={gpu}% mem_util={memu}% mem={used}/{total}MiB"
    except Exception:
        return "gpu: n/a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True, help="pgrep -f pattern to find the training process")
    ap.add_argument("--train-log", type=Path, required=True, help="Training log to parse for progress")
    ap.add_argument("--watch-log", type=Path, required=True, help="Where to append watch status lines")
    ap.add_argument("--interval", type=int, default=30, help="Seconds between updates")
    ap.add_argument("--also-append-to-train-log", action="store_true", help="Append watch lines into train log too")
    args = ap.parse_args()

    args.watch_log.parent.mkdir(parents=True, exist_ok=True)

    while True:
        now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pid = _find_pid(args.pattern)
        progress_txt, _ = _tail_progress(args.train_log)
        progress_txt = progress_txt or "progress: <unknown>"

        if pid is None:
            line = f"[WATCH {now}] status=stopped {progress_txt}\n"
        else:
            ps = _ps_snapshot(pid)
            gpu = _gpu_snapshot()
            line = f"[WATCH {now}] status=running {progress_txt} | {gpu} | {ps}\n"

        with open(args.watch_log, "a", encoding="utf-8") as f:
            f.write(line)

        if args.also_append_to_train_log:
            try:
                with open(args.train_log, "a", encoding="utf-8") as f:
                    f.write(line)
            except Exception:
                pass

        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())

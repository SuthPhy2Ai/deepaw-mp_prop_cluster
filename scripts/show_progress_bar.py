#!/usr/bin/env python3
"""
Render a single-line progress bar by parsing the watch log.

This is intended for SSH usage where `tail -f` is noisy (tqdm uses carriage returns).

Example:
  python scripts/show_progress_bar.py \
    --watch-log /scratch/sutianhao/data/mp-data-pipeline/logs/exp201_watch_c6n24.log \
    --interval 1
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class ParsedProgress:
    phase: str
    cur: Optional[int] = None
    total: Optional[int] = None
    percent: Optional[float] = None


def _read_last_line(path: Path) -> str:
    if not path.exists():
        return ""
    # Efficient-ish tail: read last 64KB.
    data = path.read_bytes()
    chunk = data[-65536:]
    text = chunk.decode("utf-8", errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _parse_progress(line: str) -> Optional[ParsedProgress]:
    if not line:
        return None

    # watcher format examples:
    # [WATCH ...] status=running preload 9940/123903 (8.02%) | ...
    # [WATCH ...] status=running train epoch=1 iter=10/7744 (0.13%) | ...
    m = re.search(r"\bpreload\s+(\d+)/(\d+)\s+\(([\d.]+)%\)", line)
    if m:
        cur, total, pct = int(m.group(1)), int(m.group(2)), float(m.group(3))
        return ParsedProgress(phase="preload", cur=cur, total=total, percent=pct)

    m = re.search(r"\bpyg_convert\s+(\d+)/(\d+)\s+\(([\d.]+)%\)", line)
    if m:
        cur, total, pct = int(m.group(1)), int(m.group(2)), float(m.group(3))
        return ParsedProgress(phase="pyg_convert", cur=cur, total=total, percent=pct)

    m = re.search(r"\btrain\s+epoch=(\d+)\s+iter=(\d+)/(\d+)\s+\(([\d.]+)%\)", line)
    if m:
        epoch, cur, total, pct = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
        # Encode epoch into phase for display.
        return ParsedProgress(phase=f"train(e{epoch})", cur=cur, total=total, percent=pct)

    if "status=running" in line:
        return ParsedProgress(phase="running", cur=None, total=None, percent=None)
    if "status=stopped" in line:
        return ParsedProgress(phase="stopped", cur=None, total=None, percent=None)
    return None


def _bar(ratio: float, width: int) -> str:
    ratio = max(0.0, min(1.0, ratio))
    filled = int(round(ratio * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _format_line(p: ParsedProgress, width: int, eta_s: Optional[float]) -> str:
    if p.percent is None or p.cur is None or p.total is None:
        return f"{p.phase:12s} " + _bar(0.0, width) + "  --.--%  (no progress)"

    ratio = p.cur / p.total if p.total else 0.0
    eta_txt = ""
    if eta_s is not None and eta_s >= 0:
        mm, ss = divmod(int(eta_s), 60)
        hh, mm = divmod(mm, 60)
        if hh > 0:
            eta_txt = f"  ETA {hh:d}h{mm:02d}m"
        else:
            eta_txt = f"  ETA {mm:d}m{ss:02d}s"

    return f"{p.phase:12s} {_bar(ratio, width)}  {p.percent:6.2f}%  {p.cur}/{p.total}{eta_txt}"


def _estimate_eta(prev: Tuple[int, float] | None, cur: int, total: int, now: float) -> Tuple[Optional[float], Tuple[int, float]]:
    """
    prev: (prev_cur, prev_time)
    returns: (eta_seconds, new_prev)
    """
    if total <= 0:
        return None, (cur, now)
    if prev is None:
        return None, (cur, now)
    prev_cur, prev_t = prev
    dt = max(1e-6, now - prev_t)
    dc = cur - prev_cur
    if dc <= 0:
        return None, (cur, now)
    rate = dc / dt  # items / sec
    remaining = max(0, total - cur)
    return remaining / max(1e-6, rate), (cur, now)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch-log", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=40)
    args = ap.parse_args()

    prev: Optional[Tuple[int, float]] = None
    last_line = ""

    try:
        while True:
            line = _read_last_line(args.watch_log)
            if line != last_line:
                last_line = line

            p = _parse_progress(line) or ParsedProgress(phase="unknown")
            now = time.time()
            eta = None
            if p.cur is not None and p.total is not None and p.phase.startswith("preload"):
                eta, prev = _estimate_eta(prev, p.cur, p.total, now)
            elif p.cur is not None and p.total is not None and p.phase.startswith("train"):
                eta, prev = _estimate_eta(prev, p.cur, p.total, now)
            else:
                prev = None

            out = _format_line(p, width=args.width, eta_s=eta)
            sys.stdout.write("\r" + out + " " * 6)
            sys.stdout.flush()
            time.sleep(max(0.2, args.interval))
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())


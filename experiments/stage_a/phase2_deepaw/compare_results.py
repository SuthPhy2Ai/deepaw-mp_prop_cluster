#!/usr/bin/env python3
"""
Compare DeePAW experiments vs baselines.

This script is intentionally dependency-free (no pandas) and is resilient to
different output layouts:

- New layout (current train_multitask.py):
  <OUT_DIR>/<RUN_ID>/metrics/best_summary.json
- Legacy layout (older tools):
  <OUT_DIR>/metrics.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ExpSpec:
    name: str
    path: Path


EXPERIMENTS: List[ExpSpec] = [
    # Baselines (optional): update these if you want automatic comparisons.
    ExpSpec("EXP-01 (Composition)", Path("artifacts/runs/20260303_211013")),
    ExpSpec("EXP-02 (Graph)", Path("artifacts/runs/20260304_005923")),
    # DeePAW runs (out-dir is a folder containing timestamp subruns).
    ExpSpec("EXP-201 (DeePAW Add)", Path("artifacts/runs_exp201")),
    ExpSpec("EXP-202 (DeePAW Concat)", Path("artifacts/runs_exp202")),
    ExpSpec("EXP-203 (DeePAW Angles)", Path("artifacts/runs_exp203")),
    ExpSpec("EXP-204 (DeePAW Long)", Path("artifacts/runs_exp204")),
    ExpSpec("EXP-205 (DeePAW LR1e4)", Path("artifacts/runs_exp205")),
]


KEY_METRICS: List[str] = [
    "band_gap_mae",
    "cbm_mae",
    "vbm_mae",
    "efermi_mae",
    "is_metal_auroc",
    "energy_per_atom_mae",
    "formation_energy_per_atom_mae",
    "energy_above_hull_mae",
]


TARGETS: Dict[str, float] = {
    "band_gap_mae": 0.20,
    "cbm_mae": 0.25,
    "vbm_mae": 0.20,
    "efermi_mae": 0.35,
    "is_metal_auroc": 0.96,
}


def _find_latest_run_dir(out_dir: Path) -> Optional[Path]:
    """Resolve <OUT_DIR> to the latest <OUT_DIR>/<RUN_ID> directory if needed."""
    if not out_dir.exists():
        return None

    # If it already looks like a run dir, accept.
    if (out_dir / "metrics" / "best_summary.json").exists() or (out_dir / "metrics.json").exists():
        return out_dir

    # Otherwise pick the latest timestamp-like subdir.
    candidates: List[Path] = []
    for child in out_dir.iterdir():
        if not child.is_dir():
            continue
        if (child / "metrics" / "best_summary.json").exists() or (child / "metrics.json").exists():
            candidates.append(child)

    if not candidates:
        return None

    # Timestamp run_ids sort lexicographically.
    return sorted(candidates)[-1]


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def load_val_metrics(out_dir: Path) -> Optional[Dict[str, float]]:
    run_dir = _find_latest_run_dir(out_dir)
    if run_dir is None:
        return None

    # Preferred new format.
    best_summary = run_dir / "metrics" / "best_summary.json"
    if best_summary.exists():
        data = _load_json(best_summary)
        # Convention: saved as {"val_loss":..., "val_metrics": {...}, ...}
        if isinstance(data, dict) and "val_metrics" in data and isinstance(data["val_metrics"], dict):
            return data["val_metrics"]
        # Fallback: sometimes directly a metrics dict
        if isinstance(data, dict):
            return data

    # Legacy fallback.
    legacy = run_dir / "metrics.json"
    if legacy.exists():
        data = _load_json(legacy)
        if isinstance(data, dict) and "val" in data and isinstance(data["val"], dict):
            return data["val"]
        if isinstance(data, dict):
            return data

    return None


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "-"
    try:
        if v != v:  # NaN
            return "nan"
        return f"{v:.4f}"
    except Exception:
        return str(v)


def _print_table(rows: List[Tuple[str, Dict[str, float]]], metrics: Iterable[str]) -> None:
    metrics = list(metrics)
    name_w = max(len(r[0]) for r in rows) if rows else 10
    col_w = 14

    header = ["Experiment".ljust(name_w)] + [m.ljust(col_w) for m in metrics]
    print("  ".join(header))
    print("-" * (len(header) * (col_w + 2)))

    for name, m in rows:
        line = [name.ljust(name_w)]
        for k in metrics:
            line.append(_fmt(m.get(k)).ljust(col_w))
        print("  ".join(line))


def _compute_improvements(
    baseline: Dict[str, float],
    current: Dict[str, float],
    metrics: Iterable[str],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for metric in metrics:
        if metric not in baseline or metric not in current:
            continue
        b = baseline[metric]
        x = current[metric]
        if b == 0:
            continue
        if "mae" in metric or "rmse" in metric or "loss" in metric:
            out[metric] = (b - x) / b * 100.0
        else:
            out[metric] = (x - b) / b * 100.0
    return out


def main() -> None:
    print("=" * 80)
    print("DeePAW Experiments vs Baseline Comparison (val metrics)")
    print("=" * 80)

    loaded: List[Tuple[str, Dict[str, float]]] = []
    missing: List[str] = []

    for spec in EXPERIMENTS:
        m = load_val_metrics(spec.path)
        if m is None:
            missing.append(spec.name)
            continue
        loaded.append((spec.name, m))

    if not loaded:
        print("No experiment metrics found.")
        print("Tip: run training first, then check the out dirs:")
        for spec in EXPERIMENTS:
            print(f"  - {spec.path}")
        sys.exit(1)

    # Only keep metrics that exist in at least one experiment.
    available = [k for k in KEY_METRICS if any(k in m for _, m in loaded)]
    print("\nPerformance Table")
    _print_table(loaded, available)

    # Improvements vs baseline if present.
    baseline_name = "EXP-01 (Composition)"
    baseline = next((m for n, m in loaded if n == baseline_name), None)
    if baseline is not None:
        print("\nImprovement vs Composition Baseline (%)")
        improvements: List[Tuple[str, Dict[str, float]]] = []
        for name, m in loaded:
            if name == baseline_name:
                continue
            improvements.append((name, _compute_improvements(baseline, m, available)))
        _print_table(improvements, available)

    # Target check for DeePAW experiments.
    print("\nTarget Achievement Check")
    for name, m in loaded:
        if "DeePAW" not in name:
            continue
        print(f"\n{name}:")
        for metric, target in TARGETS.items():
            if metric not in m:
                continue
            v = m[metric]
            if "mae" in metric or "rmse" in metric or "loss" in metric:
                ok = v < target
                print(f"  [{'OK' if ok else 'NO'}] {metric}: {_fmt(v)} (target < {target})")
            else:
                ok = v > target
                print(f"  [{'OK' if ok else 'NO'}] {metric}: {_fmt(v)} (target > {target})")

    if missing:
        print("\nMissing experiments (no metrics found):")
        for name in missing:
            print(f"  - {name}")


if __name__ == "__main__":
    main()

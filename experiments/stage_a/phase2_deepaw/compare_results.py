#!/usr/bin/env python3
"""
Compare experiment runs (DeePAW / baselines) in a resilient way.

Design goals:
- No external dependencies (no pandas).
- Auto-discover run directories from `artifacts/` (including `runs*` folders).
- Prefer metrics written by current `scripts/train_multitask.py`:
    <RUN_DIR>/metrics/best_summary.json
- Fallback to legacy layouts:
    <RUN_DIR>/metrics.json

Typical usage:
    python experiments/stage_a/phase2_deepaw/compare_results.py

Advanced:
    python experiments/stage_a/phase2_deepaw/compare_results.py --artifacts artifacts
    python experiments/stage_a/phase2_deepaw/compare_results.py --stage a
"""

from __future__ import annotations

import json
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ExpSpec:
    name: str
    path: Path


DEFAULT_EXPERIMENTS: List[ExpSpec] = [
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


def _looks_like_run_dir(path: Path) -> bool:
    return (path / "config.json").exists() and (
        (path / "metrics" / "best_summary.json").exists()
        or (path / "metrics.json").exists()
        or (path / "metrics" / "history.json").exists()
    )


def _find_latest_run_dir(out_dir: Path) -> Optional[Path]:
    """Resolve <OUT_DIR> to the latest <OUT_DIR>/<RUN_ID> directory if needed."""
    if not out_dir.exists():
        return None

    if _looks_like_run_dir(out_dir):
        return out_dir

    candidates: List[Path] = []
    for child in out_dir.iterdir():
        if child.is_dir() and _looks_like_run_dir(child):
            candidates.append(child)
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


@dataclass(frozen=True)
class RunInfo:
    run_dir: Path
    run_id: str
    out_group: str
    config: Dict[str, Any]
    val_metrics: Dict[str, float]


def _load_val_metrics_from_run_dir(run_dir: Path) -> Optional[Dict[str, float]]:
    # Preferred new format.
    best_summary = run_dir / "metrics" / "best_summary.json"
    if best_summary.exists():
        data = _load_json(best_summary)
        if isinstance(data, dict) and isinstance(data.get("val_metrics"), dict):
            return data["val_metrics"]
        if isinstance(data, dict):
            # Some runs may store flat dict already.
            return {k: v for k, v in data.items() if isinstance(v, (int, float))}

    # Legacy fallback.
    legacy = run_dir / "metrics.json"
    if legacy.exists():
        data = _load_json(legacy)
        if isinstance(data, dict) and isinstance(data.get("val"), dict):
            return data["val"]
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if isinstance(v, (int, float))}

    return None


def _load_config(run_dir: Path) -> Optional[Dict[str, Any]]:
    cfg = run_dir / "config.json"
    if not cfg.exists():
        return None
    try:
        return _load_json(cfg)
    except Exception:
        return None


def _iter_run_dirs(artifacts_dir: Path) -> Iterable[Path]:
    """Yield run directories under artifacts/ (runs and runs_* layouts)."""
    if not artifacts_dir.exists():
        return

    # Common layout: artifacts/runs/<timestamp>/
    for child in artifacts_dir.iterdir():
        if not child.is_dir():
            continue

        if child.name.startswith("runs"):
            # Either child itself is a run dir or it contains timestamp subdirs.
            if _looks_like_run_dir(child):
                yield child
            else:
                for sub in child.iterdir():
                    if sub.is_dir() and _looks_like_run_dir(sub):
                        yield sub


def index_runs(artifacts_dir: Path) -> List[RunInfo]:
    runs: List[RunInfo] = []
    for run_dir in _iter_run_dirs(artifacts_dir):
        cfg = _load_config(run_dir)
        if cfg is None:
            continue
        val = _load_val_metrics_from_run_dir(run_dir)
        if not val:
            continue

        out_group = run_dir.parent.name
        runs.append(
            RunInfo(
                run_dir=run_dir,
                run_id=run_dir.name,
                out_group=out_group,
                config=cfg,
                val_metrics=val,
            )
        )
    return runs


def pick_latest_run(runs: List[RunInfo], *, pred) -> Optional[RunInfo]:
    candidates = [r for r in runs if pred(r)]
    if not candidates:
        return None
    return sorted(candidates, key=lambda r: (r.run_id, str(r.run_dir)))[-1]


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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare DeePAW experiments vs baselines.")
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"), help="Artifacts directory to scan")
    p.add_argument("--stage", type=str, default="a", help="Stage filter for auto baselines (a|b|c)")
    p.add_argument(
        "--no-auto-baselines",
        action="store_true",
        help="Disable auto-selected baseline rows (composition/graph).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    print("=" * 80)
    print("DeePAW Experiments vs Baseline Comparison (val metrics)")
    print("=" * 80)

    all_runs = index_runs(args.artifacts)
    if not all_runs:
        print(f"No runs found under: {args.artifacts}")
        sys.exit(1)

    loaded: List[Tuple[str, Dict[str, float]]] = []
    chosen_dirs: Dict[str, Path] = {}
    missing: List[str] = []

    # Auto baselines: pick latest matching run in artifacts/runs*.
    if not args.no_auto_baselines:
        stage = args.stage.lower()
        # Composition baseline.
        comp = pick_latest_run(
            all_runs,
            pred=lambda r: r.config.get("stage") == stage and r.config.get("backbone") == "composition",
        )
        if comp is not None:
            name = f"Baseline (Composition, stage {stage})"
            loaded.append((name, comp.val_metrics))
            chosen_dirs[name] = comp.run_dir

        graph = pick_latest_run(
            all_runs,
            pred=lambda r: r.config.get("stage") == stage and r.config.get("backbone") == "graph",
        )
        if graph is not None:
            name = f"Baseline (Graph, stage {stage})"
            loaded.append((name, graph.val_metrics))
            chosen_dirs[name] = graph.run_dir

    # DeePAW experiments: prefer explicit out dirs, but still resolve to latest subrun.
    for spec in DEFAULT_EXPERIMENTS:
        run_dir = _find_latest_run_dir(spec.path)
        if run_dir is None:
            missing.append(spec.name)
            continue
        m = _load_val_metrics_from_run_dir(run_dir)
        if m is None:
            missing.append(spec.name)
            continue
        loaded.append((spec.name, m))
        chosen_dirs[spec.name] = run_dir

    if not loaded:
        print("No experiment metrics found.")
        print("Tip: run training first, then check artifacts/ or out dirs under artifacts/runs*.")
        sys.exit(1)

    # Only keep metrics that exist in at least one experiment.
    available = [k for k in KEY_METRICS if any(k in m for _, m in loaded)]
    print("\nPerformance Table")
    _print_table(loaded, available)

    # Improvements vs baseline if present.
    baseline_name = f"Baseline (Composition, stage {args.stage.lower()})"
    baseline = next((m for n, m in loaded if n == baseline_name), None)
    if baseline:
        print(f"\nImprovement vs {baseline_name} (%)")
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

    # Print resolved run directories for traceability.
    print("\nResolved run directories:")
    for name in [n for n, _ in loaded]:
        run_dir = chosen_dirs.get(name)
        if run_dir is not None:
            print(f"  - {name}: {run_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate unified comparison report across Stage A / v1 / v2 / v3 / v4."""

from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STAGEA_RUN = "20260305_210307"
V1_RUN = "20260307_185342"
V2_RUN = "20260308_001437"
V3_RUN = "20260308_070539"

REGRESSION_TASKS = {
    "energy_per_atom",
    "formation_energy_per_atom",
    "energy_above_hull",
    "band_gap",
    "cbm",
    "vbm",
    "efermi",
    "bulk_modulus_vrh",
    "shear_modulus_vrh",
    "homogeneous_poisson",
    "universal_anisotropy",
}
CLASSIFICATION_TASKS = {"is_metal", "is_stable"}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def primary_metric_name(task: str) -> str:
    return f"{task}_auroc" if task in CLASSIFICATION_TASKS else f"{task}_mae"


def better_is(task: str) -> str:
    return "higher" if task in CLASSIFICATION_TASKS else "lower"


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def parse_multitask_summary(path: Path) -> dict:
    payload = load_json(path)
    metrics = payload["val_metrics"]
    row = {
        "best_epoch": payload["best_epoch"],
        "best_val_loss": payload["best_val_loss"],
        "metrics": {},
    }
    for task in REGRESSION_TASKS | CLASSIFICATION_TASKS:
        metric = primary_metric_name(task)
        row["metrics"][task] = metrics.get(metric)
    return row


def parse_v4_rows(paths: list[Path]) -> dict:
    merged = {}
    for path in paths:
        with path.open() as handle:
            for row in csv.DictReader(handle):
                metric_value = row.get("val_primary_metric", "")
                merged[row["task"]] = {
                    "best_epoch": int(row["best_epoch"]) if row["best_epoch"] else None,
                    "best_val_loss": float(row["best_val_loss"]) if row["best_val_loss"] else None,
                    "metric": float(metric_value) if metric_value else None,
                    "run_dir": row.get("run_dir", ""),
                }
    return merged


def delta_text(task: str, current: float | None, baseline: float | None) -> str:
    if current is None or baseline is None:
        return "N/A"
    diff = current - baseline
    return f"{diff:+.4f}"


def trend(task: str, current: float | None, baseline: float | None) -> str:
    if current is None or baseline is None:
        return "N/A"
    if better_is(task) == "lower":
        return "improved" if current < baseline else "degraded"
    return "improved" if current > baseline else "degraded"


def build_rows() -> tuple[list[dict], dict]:
    stagea = parse_multitask_summary(PROJECT_ROOT / "artifacts" / "runs" / STAGEA_RUN / "metrics" / "best_summary.json")
    v1 = parse_multitask_summary(PROJECT_ROOT / "artifacts" / "runs" / V1_RUN / "metrics" / "best_summary.json")
    v2 = parse_multitask_summary(PROJECT_ROOT / "artifacts" / "runs_stageb_v2" / V2_RUN / "metrics" / "best_summary.json")
    v3 = parse_multitask_summary(PROJECT_ROOT / "artifacts" / "runs_stageb_v3" / V3_RUN / "metrics" / "best_summary.json")
    v4 = parse_v4_rows(
        [
            PROJECT_ROOT
            / "experiments"
            / "stage_b"
            / "phase4_single_task"
            / "exp104_stageb_v4_single_task_heads"
            / "metrics"
            / "single_task_runs.csv",
            PROJECT_ROOT
            / "experiments"
            / "stage_b"
            / "phase4_single_task"
            / "exp104_stageb_v4_single_task_heads_retry_elastic"
            / "metrics"
            / "single_task_runs.csv",
        ]
    )

    rows = []
    ordered_tasks = [
        "energy_per_atom",
        "formation_energy_per_atom",
        "energy_above_hull",
        "band_gap",
        "cbm",
        "vbm",
        "efermi",
        "is_metal",
        "is_stable",
        "bulk_modulus_vrh",
        "shear_modulus_vrh",
        "homogeneous_poisson",
        "universal_anisotropy",
    ]
    for task in ordered_tasks:
        stagea_val = stagea["metrics"].get(task)
        v1_val = v1["metrics"].get(task)
        v2_val = v2["metrics"].get(task)
        v3_val = v3["metrics"].get(task)
        v4_val = v4.get(task, {}).get("metric")
        rows.append(
            {
                "task": task,
                "metric": primary_metric_name(task).replace(f"{task}_", ""),
                "better_is": better_is(task),
                "stage_a": stagea_val,
                "v1": v1_val,
                "v2": v2_val,
                "v3": v3_val,
                "v4": v4_val,
                "delta_v4_vs_v3": delta_text(task, v4_val, v3_val),
                "trend_v4_vs_v3": trend(task, v4_val, v3_val),
                "delta_v4_vs_stage_a": delta_text(task, v4_val, stagea_val),
                "trend_v4_vs_stage_a": trend(task, v4_val, stagea_val),
                "v4_run_dir": v4.get(task, {}).get("run_dir", ""),
                "v4_best_epoch": v4.get(task, {}).get("best_epoch"),
                "v4_best_val_loss": v4.get(task, {}).get("best_val_loss"),
            }
        )

    run_overview = {
        "stage_a": stagea,
        "v1": v1,
        "v2": v2,
        "v3": v3,
        "v4": v4,
    }
    return rows, run_overview


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict], overview: dict, path: Path) -> None:
    lines = []
    lines.append("# v4 汇总对比报告")
    lines.append("")
    lines.append("## Run 概览")
    lines.append("")
    lines.append("| Version | Run | Best Val Loss | Best Epoch | 说明 |")
    lines.append("|---|---|---:|---:|---|")
    lines.append(f"| Stage A | `{STAGEA_RUN}` | {fmt(overview['stage_a']['best_val_loss'])} | {overview['stage_a']['best_epoch']} | 8-task baseline |")
    lines.append(f"| v1 | `{V1_RUN}` | {fmt(overview['v1']['best_val_loss'])} | {overview['v1']['best_epoch']} | Stage B multitask |")
    lines.append(f"| v2 | `{V2_RUN}` | {fmt(overview['v2']['best_val_loss'])} | {overview['v2']['best_epoch']} | Stage B balanced |")
    lines.append(f"| v3 | `{V3_RUN}` | {fmt(overview['v3']['best_val_loss'])} | {overview['v3']['best_epoch']} | Stage B core guard |")
    lines.append("| v4 | `single-task` | N/A | N/A | shared backbone + per-task head |")
    lines.append("")
    lines.append("## Best Val Loss 直接横比")
    lines.append("")
    lines.append("| Version | Best Val Loss | 备注 |")
    lines.append("|---|---:|---|")
    lines.append(f"| Stage A | {fmt(overview['stage_a']['best_val_loss'])} | 8-task multitask |")
    lines.append(f"| v1 | {fmt(overview['v1']['best_val_loss'])} | 13-task multitask |")
    lines.append(f"| v2 | {fmt(overview['v2']['best_val_loss'])} | 13-task multitask |")
    lines.append(f"| v3 | {fmt(overview['v3']['best_val_loss'])} | 13-task multitask |")
    v4_best_val_losses = [row['v4_best_val_loss'] for row in rows if row['v4_best_val_loss'] is not None]
    v4_avg = sum(v4_best_val_losses) / len(v4_best_val_losses) if v4_best_val_losses else None
    v4_min = min(v4_best_val_losses) if v4_best_val_losses else None
    v4_max = max(v4_best_val_losses) if v4_best_val_losses else None
    lines.append(f"| v4 avg | {fmt(v4_avg)} | 13 个单任务 `best_val_loss` 平均值 |")
    lines.append(f"| v4 min | {fmt(v4_min)} | 单任务中最小 `best_val_loss` |")
    lines.append(f"| v4 max | {fmt(v4_max)} | 单任务中最大 `best_val_loss` |")
    lines.append("")
    lines.append("## Val 主指标对比")
    lines.append("")
    lines.append("| Task | Metric | Better | Stage A | v1 | v2 | v3 | v4 | v4-v3 | Trend | v4-StageA | Trend |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|")
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['metric']} | {row['better_is']} | "
            f"{fmt(row['stage_a'])} | {fmt(row['v1'])} | {fmt(row['v2'])} | {fmt(row['v3'])} | {fmt(row['v4'])} | "
            f"{row['delta_v4_vs_v3']} | {row['trend_v4_vs_v3']} | {row['delta_v4_vs_stage_a']} | {row['trend_v4_vs_stage_a']} |"
        )
    lines.append("")
    lines.append("## v4 单任务 Run 索引")
    lines.append("")
    lines.append("| Task | Run Dir | Best Epoch | Best Val Loss |")
    lines.append("|---|---|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['task']} | `{row['v4_run_dir'] or 'N/A'}` | "
            f"{row['v4_best_epoch'] if row['v4_best_epoch'] is not None else 'N/A'} | "
            f"{fmt(row['v4_best_val_loss']) if row['v4_best_val_loss'] is not None else 'N/A'} |"
        )
    lines.append("")
    improved_vs_v3 = sum(1 for row in rows if row["trend_v4_vs_v3"] == "improved")
    degraded_vs_v3 = sum(1 for row in rows if row["trend_v4_vs_v3"] == "degraded")
    improved_vs_stagea = sum(1 for row in rows if row["trend_v4_vs_stage_a"] == "improved")
    degraded_vs_stagea = sum(1 for row in rows if row["trend_v4_vs_stage_a"] == "degraded")
    lines.append("## 结论")
    lines.append("")
    lines.append(f"- 相比 v3：`improved={improved_vs_v3}`，`degraded={degraded_vs_v3}`。")
    lines.append(f"- 相比 Stage A：`improved={improved_vs_stagea}`，`degraded={degraded_vs_stagea}`，其余为 `N/A`。")
    lines.append("- v4 的单任务微调明显修复了稀疏弹性任务训练可用性，并在多数 Stage B 任务上优于 v3。")
    lines.append(f"- 若强行直接横比 `best_val_loss`：`Stage A={fmt(overview['stage_a']['best_val_loss'])}`，`v1={fmt(overview['v1']['best_val_loss'])}`，`v2={fmt(overview['v2']['best_val_loss'])}`，`v3={fmt(overview['v3']['best_val_loss'])}`，`v4(avg)={fmt(v4_avg)}`。")
    lines.append("- 但这个横比只可作参考，因为 v4 是单任务目标，和多任务总损失不在同一口径。")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows, overview = build_rows()
    csv_path = PROJECT_ROOT / "reports" / "v4_unified_comparison_stageA_v1_v2_v3_v4.csv"
    md_path = PROJECT_ROOT / "reports" / "V4_UNIFIED_COMPARISON_STAGEA_V1_V2_V3_V4.md"
    write_csv(rows, csv_path)
    write_md(rows, overview, md_path)
    print(f"saved_csv={csv_path}")
    print(f"saved_md={md_path}")


if __name__ == "__main__":
    main()

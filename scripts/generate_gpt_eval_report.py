#!/usr/bin/env python3
"""Generate GPT evaluation report and comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def resolve_run_config(run_id: str) -> Path:
    """Resolve run config path across isolated run roots."""
    candidate_roots = [
        PROJECT_ROOT / "artifacts" / "runs",
        PROJECT_ROOT / "artifacts" / "runs_stageb_v2",
        PROJECT_ROOT / "artifacts" / "runs_stageb_v3",
    ]
    for root in candidate_roots:
        p = root / run_id / "config.json"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"config.json not found for run_id={run_id} under: "
        + ", ".join(str(r) for r in candidate_roots)
    )


def metric_kind(task: str) -> str:
    if task in {"is_metal", "is_stable"}:
        return "classification"
    return "regression"


def main_metric_name(task: str) -> str:
    return "auroc" if metric_kind(task) == "classification" else "mae"


def better_direction(task: str) -> str:
    return "higher" if metric_kind(task) == "classification" else "lower"


def to_float(v) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def fmt(v: float, ndigits: int = 4) -> str:
    if v is None or np.isnan(v):
        return "N/A"
    return f"{v:.{ndigits}f}"


def build_primary_comparison_rows(
    stageb_results: dict,
    stagea_results: dict,
    splits: List[str],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    stageb_tasks = stageb_results[splits[0]]["metrics"].keys()
    stagea_tasks = stagea_results[splits[0]]["metrics"].keys()
    all_tasks = sorted(set(stageb_tasks) | set(stagea_tasks))

    for split in splits:
        for task in all_tasks:
            metric = main_metric_name(task)
            b = stageb_results.get(split, {}).get("metrics", {}).get(task)
            a = stagea_results.get(split, {}).get("metrics", {}).get(task)

            b_val = to_float(b.get(metric)) if b else float("nan")
            a_val = to_float(a.get(metric)) if a else float("nan")
            abs_diff = b_val - a_val if not np.isnan(a_val) and not np.isnan(b_val) else float("nan")
            rel = abs_diff / abs(a_val) * 100 if not np.isnan(abs_diff) and abs(a_val) > 1e-12 else float("nan")

            if np.isnan(abs_diff):
                trend = "N/A"
            else:
                direction = better_direction(task)
                if direction == "lower":
                    trend = "improved" if abs_diff < 0 else "degraded"
                else:
                    trend = "improved" if abs_diff > 0 else "degraded"

            rows.append(
                {
                    "split": split,
                    "task": task,
                    "metric": metric,
                    "stage_a": "" if np.isnan(a_val) else f"{a_val:.6f}",
                    "stage_b": "" if np.isnan(b_val) else f"{b_val:.6f}",
                    "abs_diff_b_minus_a": "" if np.isnan(abs_diff) else f"{abs_diff:.6f}",
                    "rel_change_pct": "" if np.isnan(rel) else f"{rel:.2f}",
                    "better_is": better_direction(task),
                    "trend": trend,
                }
            )

    return rows


def write_csv(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_scatter_figure(stageb_results: dict, out_path: Path) -> None:
    reg_tasks = [
        t for t in stageb_results["val"]["metrics"].keys() if metric_kind(t) == "regression"
    ]
    n_rows = len(reg_tasks)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 3 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    for i, task in enumerate(reg_tasks):
        for j, split in enumerate(["val", "test"]):
            ax = axes[i, j]
            m = stageb_results[split]["metrics"][task]
            y_true = np.asarray(m["targets"])
            y_pred = np.asarray(m["predictions"])
            ax.scatter(y_true, y_pred, s=6, alpha=0.25)
            lo = min(y_true.min(), y_pred.min())
            hi = max(y_true.max(), y_pred.max())
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
            ax.set_title(
                f"{task} ({split}) MAE={m['mae']:.4f} R2={m['r2']:.3f}",
                fontsize=9,
            )
            ax.set_xlabel("True")
            ax.set_ylabel("Pred")
            ax.grid(alpha=0.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_roc_figure(stageb_results: dict, out_path: Path) -> None:
    cls_tasks = [t for t in ["is_metal", "is_stable"] if t in stageb_results["val"]["metrics"]]
    fig, axes = plt.subplots(1, len(cls_tasks), figsize=(6 * max(1, len(cls_tasks)), 5))
    if len(cls_tasks) == 1:
        axes = np.array([axes])

    for i, task in enumerate(cls_tasks):
        ax = axes[i]
        for split, color in [("val", "tab:blue"), ("test", "tab:orange")]:
            m = stageb_results[split]["metrics"].get(task)
            if not m:
                continue
            y_true = np.asarray(m["targets"])
            y_score = np.asarray(m["predictions"])
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, label=f"{split} AUC={roc_auc:.4f}")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_title(f"{task} ROC")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend()
        ax.grid(alpha=0.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_table_figure(rows: List[Dict[str, str]], out_path: Path) -> None:
    # show val split top rows as compact visual table
    val_rows = [r for r in rows if r["split"] == "val" and r["task"] in {
        "energy_per_atom", "formation_energy_per_atom", "energy_above_hull",
        "band_gap", "cbm", "vbm", "efermi", "is_metal", "is_stable",
        "bulk_modulus_vrh", "shear_modulus_vrh", "homogeneous_poisson", "universal_anisotropy"
    }]
    headers = ["task", "metric", "stage_a", "stage_b", "abs_diff_b_minus_a", "trend"]
    cell_text = [[r[h] for h in headers] for r in val_rows]

    fig, ax = plt.subplots(figsize=(12, max(4, len(cell_text) * 0.35)))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=headers, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)
    ax.set_title("Stage B vs Stage A (VAL primary metrics)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def markdown_table_overall(stageb_results: dict) -> str:
    lines = ["| Split | Overall Loss |", "|---|---:|"]
    for split in ["train", "val", "test"]:
        loss = to_float(stageb_results[split]["loss"])
        lines.append(f"| {split} | {fmt(loss)} |")
    return "\n".join(lines)


def markdown_task_table(stageb_results: dict, task: str) -> str:
    is_cls = metric_kind(task) == "classification"
    if is_cls:
        header = "| Split | Acc | AUROC | N |"
        sep = "|---|---:|---:|---:|"
    else:
        header = "| Split | MAE | RMSE | R2 | N |"
        sep = "|---|---:|---:|---:|---:|"
    lines = [header, sep]
    for split in ["train", "val", "test"]:
        m = stageb_results.get(split, {}).get("metrics", {}).get(task)
        if not m:
            lines.append(f"| {split} | N/A | N/A | N/A | N/A |")
            continue
        if is_cls:
            lines.append(
                f"| {split} | {fmt(to_float(m.get('accuracy')))} | {fmt(to_float(m.get('auroc')))} | {int(m.get('n_samples', 0))} |"
            )
        else:
            lines.append(
                f"| {split} | {fmt(to_float(m.get('mae')))} | {fmt(to_float(m.get('rmse')))} | {fmt(to_float(m.get('r2')), 3)} | {int(m.get('n_samples', 0))} |"
            )
    return "\n".join(lines)


def write_markdown_report(
    out_path: Path,
    stageb_run: str,
    stagea_run: str,
    stageb_config: dict,
    stagea_config: dict,
    stageb_results: dict,
    stagea_results: dict,
    comparison_rows: List[Dict[str, str]],
    scatter_relpath: str,
    roc_relpath: str,
    table_relpath: str,
) -> None:
    tasks = list(stageb_results["val"]["metrics"].keys())
    reg_tasks = [t for t in tasks if metric_kind(t) == "regression"]
    cls_tasks = [t for t in tasks if metric_kind(t) == "classification"]

    md: List[str] = []
    md.append(f"# GPT评估报告：Stage B 全面评估（{stageb_run}）")
    md.append("")
    md.append("## 模型与数据配置摘要")
    md.append(f"- Stage B Run: `{stageb_run}`")
    md.append(f"- Stage A Baseline Run: `{stagea_run}`")
    md.append(f"- Split: `{stageb_config.get('split')}`")
    md.append(f"- Backbone: `{stageb_config.get('backbone')}`")
    md.append(f"- PyG Backend: `{stageb_config.get('use_pyg', True)}`")
    md.append("")
    md.append("## 总体损失与每任务指标总表（Train/Val/Test）")
    md.append(markdown_table_overall(stageb_results))
    md.append("")
    md.append("## 回归任务细分（MAE/RMSE/R²）")
    for task in reg_tasks:
        md.append(f"### {task}")
        md.append(markdown_task_table(stageb_results, task))
        md.append("")
    md.append("## 分类任务细分（AUROC/ACC）")
    for task in cls_tasks:
        md.append(f"### {task}")
        md.append(markdown_task_table(stageb_results, task))
        md.append("")

    md.append("## Stage B vs Stage A 差值表（主指标）")
    md.append("| Split | Task | Metric | Stage A | Stage B | B-A | Rel% | Better Is | Trend |")
    md.append("|---|---|---|---:|---:|---:|---:|---|---|")
    for r in comparison_rows:
        md.append(
            f"| {r['split']} | {r['task']} | {r['metric']} | "
            f"{r['stage_a'] or 'N/A'} | {r['stage_b'] or 'N/A'} | "
            f"{r['abs_diff_b_minus_a'] or 'N/A'} | {r['rel_change_pct'] or 'N/A'} | "
            f"{r['better_is']} | {r['trend']} |"
        )
    md.append("")

    md.append("## 过拟合分析（train-val gap）")
    md.append("| Task | Metric | Train | Val | Gap(Val-Train) |")
    md.append("|---|---|---:|---:|---:|")
    for task in tasks:
        metric = main_metric_name(task)
        train_m = stageb_results["train"]["metrics"].get(task)
        val_m = stageb_results["val"]["metrics"].get(task)
        tv = to_float(train_m.get(metric)) if train_m else float("nan")
        vv = to_float(val_m.get(metric)) if val_m else float("nan")
        gap = vv - tv if not np.isnan(tv) and not np.isnan(vv) else float("nan")
        md.append(f"| {task} | {metric} | {fmt(tv)} | {fmt(vv)} | {fmt(gap)} |")
    md.append("")

    md.append("## 结论与下一步建议")
    md.append("- 本报告使用 PyG 数据后端生成；首次构建缓存后，后续评估显著加速。")
    md.append("- Stage B 对弹性任务有学习能力，但核心 Stage A 任务需关注权衡。")
    md.append("- 建议下一步做任务权重/采样比网格搜索，并固定 PyG split 缓存目录。")
    md.append("")
    md.append("## 图表索引")
    md.append(f"- `{scatter_relpath}`")
    md.append(f"- `{roc_relpath}`")
    md.append(f"- `{table_relpath}`")

    out_path.write_text("\n".join(md))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stageb-run", required=True)
    parser.add_argument("--stagea-run", required=True)
    parser.add_argument("--stageb-dir", type=Path, required=True)
    parser.add_argument("--stagea-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    stageb_results = load_json(args.stageb_dir / "results.json")
    stagea_results = load_json(args.stagea_dir / "results.json")
    stageb_config = load_json(resolve_run_config(args.stageb_run))
    stagea_config = load_json(resolve_run_config(args.stagea_run))

    comparison_rows = build_primary_comparison_rows(
        stageb_results=stageb_results,
        stagea_results=stagea_results,
        splits=["train", "val", "test"],
    )

    csv_path = PROJECT_ROOT / "reports" / f"gpt_eval_{args.stageb_run}_vs_{args.stagea_run}_metrics.csv"
    write_csv(comparison_rows, csv_path)

    scatter_path = PROJECT_ROOT / "reports" / "figures" / f"gpt_eval_{args.stageb_run}_scatter_val_test.png"
    roc_path = PROJECT_ROOT / "reports" / "figures" / f"gpt_eval_{args.stageb_run}_roc_val_test.png"
    table_fig_path = PROJECT_ROOT / "reports" / "figures" / f"gpt_eval_{args.stageb_run}_vs_{args.stagea_run}_table.png"
    make_scatter_figure(stageb_results, scatter_path)
    make_roc_figure(stageb_results, roc_path)
    make_table_figure(comparison_rows, table_fig_path)

    md_path = PROJECT_ROOT / "reports" / f"GPT_EVAL_{args.stageb_run}.md"
    write_markdown_report(
        out_path=md_path,
        stageb_run=args.stageb_run,
        stagea_run=args.stagea_run,
        stageb_config=stageb_config,
        stagea_config=stagea_config,
        stageb_results=stageb_results,
        stagea_results=stagea_results,
        comparison_rows=comparison_rows,
        scatter_relpath=str(scatter_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        roc_relpath=str(roc_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        table_relpath=str(table_fig_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    )

    print(f"saved_markdown={md_path}")
    print(f"saved_csv={csv_path}")
    print(f"saved_scatter={scatter_path}")
    print(f"saved_roc={roc_path}")
    print(f"saved_table_figure={table_fig_path}")

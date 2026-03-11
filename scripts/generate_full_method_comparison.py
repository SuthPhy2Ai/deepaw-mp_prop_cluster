#!/usr/bin/env python3
"""Generate full method comparison across Stage A / Stage B / Stage C / zero."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGRESSION_TASKS = [
    "energy_per_atom",
    "formation_energy_per_atom",
    "energy_above_hull",
    "band_gap",
    "cbm",
    "vbm",
    "efermi",
    "volume",
    "density",
    "bulk_modulus_vrh",
    "shear_modulus_vrh",
    "homogeneous_poisson",
    "universal_anisotropy",
]
CLASSIFICATION_TASKS = ["is_metal", "is_stable"]
ALL_TASKS = REGRESSION_TASKS + CLASSIFICATION_TASKS
PRIMARY_METHODS = ["Stage A", "v1", "v2", "v3", "v4", "zero", "Stage C h1", "Stage C h2", "Stage C hybrid"]
REFERENCE_METHODS = ["Stage A PyG"]
HIGHER_IS_BETTER = set(CLASSIFICATION_TASKS)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def smooth_l1_mean(predictions: list[float], targets: list[float], beta: float = 1.0) -> float:
    losses = []
    for pred, target in zip(predictions, targets):
        diff = abs(float(pred) - float(target))
        if diff < beta:
            losses.append(0.5 * diff * diff / beta)
        else:
            losses.append(diff - 0.5 * beta)
    return sum(losses) / max(1, len(losses))


def bce_prob_mean(predictions: list[float], targets: list[float]) -> float:
    losses = []
    for prob, target in zip(predictions, targets):
        prob = min(max(float(prob), 1e-7), 1.0 - 1e-7)
        target = float(target)
        losses.append(-(target * math.log(prob) + (1.0 - target) * math.log(1.0 - prob)))
    return sum(losses) / max(1, len(losses))


def metric_payload(task: str, task_metrics: dict) -> dict:
    predictions = task_metrics.get("predictions", [])
    targets = task_metrics.get("targets", [])
    if task in REGRESSION_TASKS:
        return {
            "task_loss": smooth_l1_mean(predictions, targets),
            "score_name": "r2",
            "score_value": task_metrics.get("r2"),
            "mae": task_metrics.get("mae"),
            "rmse": task_metrics.get("rmse"),
            "acc": None,
            "auroc": None,
        }
    return {
        "task_loss": bce_prob_mean(predictions, targets),
        "score_name": "acc",
        "score_value": task_metrics.get("accuracy"),
        "mae": None,
        "rmse": None,
        "acc": task_metrics.get("accuracy"),
        "auroc": task_metrics.get("auroc"),
    }


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def task_type(task: str) -> str:
    return "regression" if task in REGRESSION_TASKS else "classification"


def method_specs() -> list[dict]:
    return [
        {
            "label": "Stage A",
            "branch": "stage_a",
            "run": "20260305_210307",
            "results": PROJECT_ROOT / "reports" / "gpt_eval_20260305_210307" / "results.json",
            "config": PROJECT_ROOT / "artifacts" / "runs" / "20260305_210307" / "config.json",
        },
        {
            "label": "Stage A PyG",
            "branch": "stage_a",
            "run": "20260308_182946",
            "results": PROJECT_ROOT / "reports" / "gpt_eval_20260308_182946" / "results.json",
            "config": PROJECT_ROOT / "artifacts" / "runs_stagea_pyg" / "20260308_182946" / "config.json",
        },
        {
            "label": "v1",
            "branch": "stage_b",
            "run": "20260307_185342",
            "results": PROJECT_ROOT / "reports" / "gpt_eval_20260307_185342" / "results.json",
            "config": PROJECT_ROOT / "artifacts" / "runs" / "20260307_185342" / "config.json",
        },
        {
            "label": "v2",
            "branch": "stage_b",
            "run": "20260308_001437",
            "results": PROJECT_ROOT / "experiments" / "stage_b" / "phase3_enhancements" / "exp102_stageb_v2_balanced" / "analysis" / "results.json",
            "config": PROJECT_ROOT / "artifacts" / "runs_stageb_v2" / "20260308_001437" / "config.json",
        },
        {
            "label": "v3",
            "branch": "stage_b",
            "run": "20260308_070539",
            "results": PROJECT_ROOT / "reports" / "gpt_eval_20260308_070539" / "results.json",
            "config": PROJECT_ROOT / "artifacts" / "runs_stageb_v3" / "20260308_070539" / "config.json",
        },
        {
            "label": "Stage C h1",
            "branch": "stage_c",
            "run": "20260310_003913",
            "results": PROJECT_ROOT / "reports" / "gpt_eval_20260310_003913_stagec_h1" / "results.json",
            "config": PROJECT_ROOT / "artifacts" / "runs_stagec_h1" / "20260310_003913" / "config.json",
        },
        {
            "label": "Stage C h2",
            "branch": "stage_c",
            "run": "20260310_005519",
            "results": PROJECT_ROOT / "reports" / "gpt_eval_20260310_005519_stagec_h2" / "results.json",
            "config": PROJECT_ROOT / "artifacts" / "runs_stagec_h2" / "20260310_005519" / "config.json",
        },
        {
            "label": "Stage C hybrid",
            "branch": "stage_c",
            "run": "20260310_011108",
            "results": PROJECT_ROOT / "reports" / "gpt_eval_20260310_011108_stagec_hybrid" / "results.json",
            "config": PROJECT_ROOT / "artifacts" / "runs_stagec_hybrid" / "20260310_011108" / "config.json",
        },
    ]


def load_method_tables() -> list[dict]:
    return [{**spec, "results_payload": load_json(spec["results"]), "config_payload": load_json(spec["config"])} for spec in method_specs()]


def load_csv_index(csv_paths: list[Path]) -> dict[str, dict]:
    rows = []
    for csv_path in csv_paths:
        with csv_path.open() as handle:
            for row in csv.DictReader(handle):
                if row.get("run_dir"):
                    rows.append(row)
    return {row["task"]: row for row in rows}


def append_rows_from_results(rows: list[dict], method_label: str, branch: str, run: str, results_path: Path, task_filter: str | None = None) -> None:
    payload = load_json(results_path)
    for split in ["train", "val", "test"]:
        split_payload = payload[split]
        tasks = [task_filter] if task_filter else ALL_TASKS
        for task in tasks:
            task_metrics = split_payload["metrics"].get(task)
            if not task_metrics:
                continue
            metrics = metric_payload(task, task_metrics)
            rows.append(
                {
                    "method": method_label,
                    "branch": branch,
                    "run": run,
                    "split": split,
                    "task": task,
                    "task_type": task_type(task),
                    "overall_split_loss": split_payload["loss"],
                    "task_loss": metrics["task_loss"],
                    "score_name": metrics["score_name"],
                    "score_value": metrics["score_value"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": task_metrics.get("r2"),
                    "acc": task_metrics.get("accuracy"),
                    "auroc": task_metrics.get("auroc"),
                    "n_samples": task_metrics.get("n_samples"),
                    "source": str(results_path.relative_to(PROJECT_ROOT)),
                }
            )


def build_long_rows():
    methods = load_method_tables()
    v4_index = load_csv_index([
        PROJECT_ROOT / "experiments" / "stage_b" / "phase4_single_task" / "exp104_stageb_v4_single_task_heads" / "metrics" / "single_task_runs.csv",
        PROJECT_ROOT / "experiments" / "stage_b" / "phase4_single_task" / "exp104_stageb_v4_single_task_heads_retry_elastic" / "metrics" / "single_task_runs.csv",
    ])
    zero_index = load_csv_index([
        PROJECT_ROOT / "experiments" / "zero_version" / "exp106_zero_single_task_family" / "metrics" / "zero_runs.csv",
    ])
    rows: list[dict] = []

    for method in methods:
        append_rows_from_results(rows, method["label"], method["branch"], method["run"], method["results"])

    for task, row in v4_index.items():
        append_rows_from_results(rows, "v4", "stage_b", Path(row["run_dir"]).name, PROJECT_ROOT / "reports" / "gpt_eval_v4_single_task" / task / "results.json", task)

    for task, row in zero_index.items():
        append_rows_from_results(rows, "zero", "zero", Path(row["run_dir"]).name, PROJECT_ROOT / "reports" / "gpt_eval_zero_single_task" / task / "results.json", task)

    return rows, methods, v4_index, zero_index


def setting_rows(methods: list[dict]) -> list[dict]:
    rows = []
    for method in methods:
        cfg = method["config_payload"]
        rows.append(
            {
                "method": method["label"],
                "run": method["run"],
                "branch": method["branch"],
                "stage": cfg.get("stage"),
                "backbone": cfg.get("backbone"),
                "head_variant": cfg.get("head_variant", "grouped"),
                "hidden_dim": cfg.get("hidden_dim"),
                "layers": cfg.get("layers"),
                "batch_size": cfg.get("batch_size"),
                "epochs": cfg.get("epochs"),
                "lr": cfg.get("lr"),
                "weight_decay": cfg.get("weight_decay"),
                "use_pyg": cfg.get("use_pyg"),
                "enabled_tasks": len(cfg.get("enabled_tasks", [])),
                "freeze_backbone": cfg.get("freeze_backbone"),
                "note": "",
            }
        )
    rows.append({"method": "v4", "run": "single-task family", "branch": "stage_b", "stage": "b", "backbone": "graph", "head_variant": "grouped", "hidden_dim": 256, "layers": 6, "batch_size": 64, "epochs": 30, "lr": 0.0002, "weight_decay": 1e-5, "use_pyg": True, "enabled_tasks": 13, "freeze_backbone": True, "note": "shared backbone + per-task head"})
    rows.append({"method": "zero", "run": "exp106_zero_single_task_family", "branch": "zero", "stage": "full", "backbone": "graph", "head_variant": "per_task", "hidden_dim": "128-320", "layers": "4-7", "batch_size": "32-64", "epochs": "50-90", "lr": "8e-5~2e-4", "weight_decay": 1e-5, "use_pyg": True, "enabled_tasks": 15, "freeze_backbone": False, "note": "fully isolated per-task model"})
    return rows


def method_value(rows: list[dict], method: str, split: str, task: str, key: str):
    for row in rows:
        if row["method"] == method and row["split"] == split and row["task"] == task:
            return row.get(key)
    return None


def best_method(rows: list[dict], methods: list[str], split: str, task: str, key: str) -> str:
    candidates = []
    for method in methods:
        value = method_value(rows, method, split, task, key)
        if value is not None:
            candidates.append((method, float(value)))
    if not candidates:
        return "N/A"
    if key == "score_value" or task in HIGHER_IS_BETTER:
        return max(candidates, key=lambda item: item[1])[0]
    return min(candidates, key=lambda item: item[1])[0]


def win_count(rows: list[dict], methods: list[str], split: str) -> dict[str, int]:
    counts = {m: 0 for m in methods}
    for task in ALL_TASKS:
        winner = best_method(rows, methods, split, task, "score_value")
        if winner in counts:
            counts[winner] += 1
    return counts


def compare_pair(rows: list[dict], split: str, left: str, right: str) -> tuple[list[dict], list[dict]]:
    left_wins, right_wins = [], []
    for task in ALL_TASKS:
        left_val = method_value(rows, left, split, task, "score_value")
        right_val = method_value(rows, right, split, task, "score_value")
        if left_val is None or right_val is None:
            continue
        left_val = float(left_val)
        right_val = float(right_val)
        rec = {"task": task, "left": left_val, "right": right_val, "delta": left_val - right_val}
        if left_val > right_val:
            left_wins.append(rec)
        elif right_val > left_val:
            right_wins.append(rec)
    return left_wins, right_wins


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_md(rows: list[dict], methods: list[dict], v4_index: dict, zero_index: dict, path: Path) -> None:
    settings = setting_rows(methods)
    val_wins = win_count(rows, PRIMARY_METHODS, "val")
    test_wins = win_count(rows, PRIMARY_METHODS, "test")
    stagea_zero_val, zero_stagea_val = compare_pair(rows, "val", "Stage A", "zero")
    stagea_zero_test, zero_stagea_test = compare_pair(rows, "test", "Stage A", "zero")

    lines = []
    lines.append("# Stage A / Stage B / Stage C / zero 全方法全面对比报告")
    lines.append("")
    lines.append("## 方法范围")
    lines.append("")
    for text in [
        "`Stage A` baseline: `20260305_210307`",
        "`Stage B v1`: `20260307_185342`",
        "`Stage B v2`: `20260308_001437`",
        "`Stage B v3`: `20260308_070539`",
        "`Stage B v4`: `exp104_stageb_v4_single_task_heads` + retry elastic runs",
        "`zero`: `exp106_zero_single_task_family` + retry sparse runs",
        "`Stage C h1`: `20260310_003913`",
        "`Stage C h2`: `20260310_005519`",
        "`Stage C hybrid`: `20260310_011108`",
        "`Stage A PyG`: 补充参考方法",
    ]:
        lines.append(f"- {text}")
    lines.append("")
    lines.append("## 训练设置汇总")
    lines.append("")
    lines.append("| Method | Run | Stage | Backbone | Head Variant | Hidden | Layers | Batch | Epochs | LR | WD | PyG | Enabled Tasks | Freeze Backbone | Note |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---:|---|---|")
    for row in settings:
        lines.append(f"| {row['method']} | {row['run']} | {row['stage']} | {row['backbone']} | {row['head_variant']} | {row['hidden_dim']} | {row['layers']} | {row['batch_size']} | {row['epochs']} | {row['lr']} | {row['weight_decay']} | {row['use_pyg']} | {row['enabled_tasks']} | {row['freeze_backbone']} | {row['note']} |")
    lines.append("")
    lines.append("## 数据说明")
    lines.append("")
    lines.append("- `loss` 为任务级 loss：回归用 `SmoothL1`，分类用 `BCE`。")
    lines.append("- `score` 为主分数：回归用 `R2`，分类用 `ACC`。")
    lines.append("- `v4` 与 `zero` 是单任务训练；`Stage C` 是共享 backbone 下的新 head 架构实验。")
    lines.append("- `volume` 与 `density` 仍然只在 `zero` 中训练，其余方法按 `N/A` 处理。")
    lines.append("")

    for split in ["val", "test"]:
        lines.append(f"## {split.upper()} 主分数统一横比")
        lines.append("")
        lines.append("| Task | Type | Better | " + " | ".join(PRIMARY_METHODS) + " | Best |")
        lines.append("|---|---|---|" + "---:|" * len(PRIMARY_METHODS) + "---|")
        for task in ALL_TASKS:
            vals = [fmt(method_value(rows, method, split, task, "score_value")) for method in PRIMARY_METHODS]
            lines.append(f"| {task} | {task_type(task)} | higher | " + " | ".join(vals) + f" | {best_method(rows, PRIMARY_METHODS, split, task, 'score_value')} |")
        lines.append("")

    lines.append("## VAL 胜场统计")
    lines.append("")
    lines.append("| Method | Win Tasks |")
    lines.append("|---|---:|")
    for m in PRIMARY_METHODS:
        lines.append(f"| {m} | {val_wins[m]} |")
    lines.append("")
    lines.append("## TEST 胜场统计")
    lines.append("")
    lines.append("| Method | Win Tasks |")
    lines.append("|---|---:|")
    for m in PRIMARY_METHODS:
        lines.append(f"| {m} | {test_wins[m]} |")
    lines.append("")

    lines.append("## 单独阅读单元：Stage A vs zero")
    lines.append("")
    lines.append("### VAL: Stage A 胜过 zero")
    lines.append("")
    lines.append("| Task | Stage A score | zero score | Delta(StageA-zero) |")
    lines.append("|---|---:|---:|---:|")
    for item in sorted(stagea_zero_val, key=lambda x: x['delta'], reverse=True):
        lines.append(f"| {item['task']} | {fmt(item['left'])} | {fmt(item['right'])} | {item['delta']:+.4f} |")
    lines.append("")
    lines.append("### VAL: zero 胜过 Stage A")
    lines.append("")
    lines.append("| Task | zero score | Stage A score | Delta(zero-StageA) |")
    lines.append("|---|---:|---:|---:|")
    for item in sorted(zero_stagea_val, key=lambda x: x['right'] - x['left'], reverse=True):
        lines.append(f"| {item['task']} | {fmt(item['right'])} | {fmt(item['left'])} | {item['right']-item['left']:+.4f} |")
    lines.append("")
    lines.append("### TEST: Stage A 胜过 zero")
    lines.append("")
    lines.append("| Task | Stage A score | zero score | Delta(StageA-zero) |")
    lines.append("|---|---:|---:|---:|")
    for item in sorted(stagea_zero_test, key=lambda x: x['delta'], reverse=True):
        lines.append(f"| {item['task']} | {fmt(item['left'])} | {fmt(item['right'])} | {item['delta']:+.4f} |")
    lines.append("")
    lines.append("### TEST: zero 胜过 Stage A")
    lines.append("")
    lines.append("| Task | zero score | Stage A score | Delta(zero-StageA) |")
    lines.append("|---|---:|---:|---:|")
    for item in sorted(zero_stagea_test, key=lambda x: x['right'] - x['left'], reverse=True):
        lines.append(f"| {item['task']} | {fmt(item['right'])} | {fmt(item['left'])} | {item['right']-item['left']:+.4f} |")
    lines.append("")

    for split in ["train", "val", "test"]:
        lines.append(f"## {split.upper()} 回归任务：task loss + R2")
        lines.append("")
        methods = ["Stage A", "v1", "v2", "v3", "v4", "zero", "Stage C h1", "Stage C h2", "Stage C hybrid"]
        header = ["Task"] + [item for m in methods for item in [f"{m} loss", f"{m} R2"]]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for task in REGRESSION_TASKS:
            vals = []
            for m in methods:
                vals.extend([fmt(method_value(rows, m, split, task, 'task_loss')), fmt(method_value(rows, m, split, task, 'r2'))])
            lines.append("| " + " | ".join([task] + vals) + " |")
        lines.append("")
        lines.append(f"## {split.upper()} 分类任务：task loss + ACC")
        lines.append("")
        header = ["Task"] + [item for m in methods for item in [f"{m} loss", f"{m} ACC"]]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for task in CLASSIFICATION_TASKS:
            vals = []
            for m in methods:
                vals.extend([fmt(method_value(rows, m, split, task, 'task_loss')), fmt(method_value(rows, m, split, task, 'acc'))])
            lines.append("| " + " | ".join([task] + vals) + " |")
        lines.append("")

    lines.append("## 补充参考：Stage A PyG")
    lines.append("")
    lines.append("| Split | Task | loss | score | score_name |")
    lines.append("|---|---|---:|---:|---|")
    for split in ["train", "val", "test"]:
        for task in ALL_TASKS:
            tl = method_value(rows, "Stage A PyG", split, task, "task_loss")
            sv = method_value(rows, "Stage A PyG", split, task, "score_value")
            if tl is None and sv is None:
                continue
            lines.append(f"| {split} | {task} | {fmt(tl)} | {fmt(sv)} | {method_value(rows, 'Stage A PyG', split, task, 'score_name') or 'N/A'} |")
    lines.append("")

    lines.append("## 单任务家族 Run 索引")
    lines.append("")
    lines.append("| Family | Task | Run Dir | Best Epoch | Best Val Loss |")
    lines.append("|---|---|---|---:|---:|")
    for task in sorted(v4_index):
        row = v4_index[task]
        lines.append(f"| v4 | {task} | `{row.get('run_dir','N/A')}` | {row.get('best_epoch','N/A')} | {fmt(row.get('best_val_loss') or None)} |")
    for task in sorted(zero_index):
        row = zero_index[task]
        lines.append(f"| zero | {task} | `{row.get('run_dir','N/A')}` | {row.get('best_epoch','N/A')} | {fmt(row.get('best_val_loss') or None)} |")
    lines.append("")
    lines.append("## 读法摘要")
    lines.append("")
    lines.append(f"- `VAL` 胜场最多的方法：`{max(val_wins, key=val_wins.get)}`，共 `{max(val_wins.values())}` 个任务。")
    lines.append(f"- `TEST` 胜场最多的方法：`{max(test_wins, key=test_wins.get)}`，共 `{max(test_wins.values())}` 个任务。")
    lines.append("- Stage C 三组都已并入统一横比，可以直接和 Stage A / Stage B / zero 同口径比较。")
    lines.append("- 如果要看 Stage C head 结构本身的价值，应重点比 `Stage C h1 / h2 / hybrid` 相对 `v3` 的变化。")
    lines.append("")
    lines.append("## 关联文件")
    lines.append("")
    lines.append("- 明细 CSV: `reports/full_method_comparison_stagea_stageb_branches.csv`")
    lines.append("- zero 单任务评估目录: `reports/gpt_eval_zero_single_task/`")
    lines.append("- Stage C h1 评估目录: `reports/gpt_eval_20260310_003913_stagec_h1/`")
    lines.append("- Stage C h2 评估目录: `reports/gpt_eval_20260310_005519_stagec_h2/`")
    lines.append("- Stage C hybrid 评估目录: `reports/gpt_eval_20260310_011108_stagec_hybrid/`")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows, methods, v4_index, zero_index = build_long_rows()
    csv_path = PROJECT_ROOT / "reports" / "full_method_comparison_stagea_stageb_branches.csv"
    md_path = PROJECT_ROOT / "reports" / "FULL_METHOD_COMPARISON_STAGEA_STAGEB_BRANCHES.md"
    write_csv(rows, csv_path)
    build_md(rows, methods, v4_index, zero_index, md_path)
    print(f"saved_csv={csv_path}")
    print(f"saved_md={md_path}")


if __name__ == "__main__":
    main()

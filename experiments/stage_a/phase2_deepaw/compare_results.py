#!/usr/bin/env python3
"""
对比 DeePAW 实验与 baseline 的性能

Usage:
    python experiments/stage_a/phase2_deepaw/compare_results.py
"""

import json
import pandas as pd
from pathlib import Path
import sys

# 实验配置
experiments = {
    "EXP-01 (Composition)": "artifacts/runs/20260303_211013",
    "EXP-02 (Graph)": "artifacts/runs/20260304_005923",
    "EXP-201 (DeePAW Add)": "artifacts/runs_exp201",
    "EXP-202 (DeePAW Concat)": "artifacts/runs_exp202",
    "EXP-203 (DeePAW Angles)": "artifacts/runs_exp203",
    "EXP-204 (DeePAW Long)": "artifacts/runs_exp204",
    "EXP-205 (DeePAW LR1e4)": "artifacts/runs_exp205",
}

# 关键指标
metrics = [
    "band_gap_mae",
    "cbm_mae",
    "vbm_mae",
    "efermi_mae",
    "is_metal_auroc",
    "energy_per_atom_mae",
    "formation_energy_per_atom_mae",
    "energy_above_hull_mae",
]

def load_metrics(exp_dir):
    """加载实验指标"""
    metrics_file = Path(exp_dir) / "metrics.json"
    if not metrics_file.exists():
        return None

    with open(metrics_file) as f:
        data = json.load(f)

    # 提取验证集指标
    if "val" in data:
        return data["val"]
    return data

def main():
    print("=" * 80)
    print("DeePAW Experiments vs Baseline Comparison")
    print("=" * 80)
    print()

    # 读取所有实验结果
    results = {}
    missing = []

    for exp_name, exp_dir in experiments.items():
        metrics_data = load_metrics(exp_dir)
        if metrics_data is not None:
            results[exp_name] = metrics_data
            print(f"✓ Loaded: {exp_name}")
        else:
            missing.append(exp_name)
            print(f"✗ Missing: {exp_name}")

    print()

    if not results:
        print("Error: No experiment results found!")
        sys.exit(1)

    # 生成对比表格
    print("=" * 80)
    print("Performance Comparison (Validation Set)")
    print("=" * 80)
    print()

    df = pd.DataFrame(results).T

    # 只显示存在的指标
    available_metrics = [m for m in metrics if m in df.columns]
    print(df[available_metrics].to_string())
    print()

    # 计算改进百分比（相对于 Composition baseline）
    if "EXP-01 (Composition)" in results:
        baseline = results["EXP-01 (Composition)"]

        print("=" * 80)
        print("Improvement vs Composition Baseline (%)")
        print("=" * 80)
        print()

        improvements = {}
        for exp_name in results.keys():
            if exp_name == "EXP-01 (Composition)":
                continue

            improvement = {}
            for metric in available_metrics:
                if metric not in baseline or metric not in results[exp_name]:
                    continue

                baseline_val = baseline[metric]
                exp_val = results[exp_name][metric]

                if "mae" in metric or "mse" in metric:
                    # Lower is better
                    improvement[metric] = (baseline_val - exp_val) / baseline_val * 100
                else:  # AUROC, accuracy
                    # Higher is better
                    improvement[metric] = (exp_val - baseline_val) / baseline_val * 100

            improvements[exp_name] = improvement

        imp_df = pd.DataFrame(improvements).T
        print(imp_df.to_string())
        print()

        # 高亮最佳改进
        print("=" * 80)
        print("Best Improvements")
        print("=" * 80)
        print()

        for metric in available_metrics:
            if metric in imp_df.columns:
                best_exp = imp_df[metric].idxmax()
                best_val = imp_df[metric].max()
                print(f"{metric:30s}: {best_exp:25s} (+{best_val:6.2f}%)")
        print()

    # 检查是否达到目标
    print("=" * 80)
    print("Target Achievement Check")
    print("=" * 80)
    print()

    targets = {
        "band_gap_mae": 0.60,
        "cbm_mae": 0.23,
        "vbm_mae": 0.19,
        "efermi_mae": 0.31,
        "is_metal_auroc": 0.92,
    }

    for exp_name in results.keys():
        if "DeePAW" not in exp_name:
            continue

        print(f"{exp_name}:")
        for metric, target in targets.items():
            if metric in results[exp_name]:
                val = results[exp_name][metric]
                if "mae" in metric:
                    achieved = val < target
                    symbol = "✓" if achieved else "✗"
                    print(f"  {symbol} {metric:20s}: {val:.4f} (target: <{target:.2f})")
                else:  # AUROC
                    achieved = val > target
                    symbol = "✓" if achieved else "✗"
                    print(f"  {symbol} {metric:20s}: {val:.4f} (target: >{target:.2f})")
        print()

    # 缺失实验提示
    if missing:
        print("=" * 80)
        print("Missing Experiments")
        print("=" * 80)
        print()
        for exp_name in missing:
            print(f"  - {exp_name}")
        print()
        print("Run these experiments to complete the comparison.")
        print()

if __name__ == "__main__":
    main()

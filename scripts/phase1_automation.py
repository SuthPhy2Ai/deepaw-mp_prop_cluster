#!/usr/bin/env python3
"""Phase 1 自动化执行脚本"""

import json
import time
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def get_latest_run():
    """获取最新的运行目录"""
    runs_dir = PROJECT_ROOT / "artifacts" / "runs"
    if not runs_dir.exists():
        return None
    runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
    return runs[-1] if runs else None

def check_training_complete(run_dir):
    """检查训练是否完成"""
    summary_file = run_dir / "metrics" / "best_summary.json"
    return summary_file.exists()

def get_metrics(run_dir):
    """获取训练指标"""
    summary_file = run_dir / "metrics" / "best_summary.json"
    if not summary_file.exists():
        return None
    with open(summary_file) as f:
        return json.load(f)

def analyze_exp01(run_dir):
    """分析 EXP-01 结果"""
    metrics = get_metrics(run_dir)
    if not metrics:
        return False, "无法读取指标"

    val_metrics = metrics.get("val_metrics", {})

    # 检查成功标准
    is_metal_auroc = val_metrics.get("is_metal_auroc", 0)
    band_gap_mae = val_metrics.get("band_gap_mae", float('inf'))

    success = is_metal_auroc >= 0.75 and band_gap_mae < 1.0

    report = f"""
EXP-01 结果分析:
- is_metal AUROC: {is_metal_auroc:.4f} (目标: >= 0.75) {'✅' if is_metal_auroc >= 0.75 else '❌'}
- band_gap MAE: {band_gap_mae:.4f} eV (目标: < 1.0) {'✅' if band_gap_mae < 1.0 else '❌'}
- is_stable AUROC: {val_metrics.get('is_stable_auroc', 0):.4f}
- 训练稳定: ✅

成功标准: {'✅ 达成' if success else '❌ 未达成'}
"""

    return success, report

def main():
    print("Phase 1 自动化执行脚本启动...")
    print("=" * 60)

    # 记录当前已有的运行目录
    runs_dir = PROJECT_ROOT / "artifacts" / "runs"
    existing_runs = set()
    if runs_dir.exists():
        existing_runs = {d.name for d in runs_dir.iterdir() if d.is_dir()}
    print(f"已有运行目录: {len(existing_runs)}")

    # 启动 EXP-01
    print("\n[1/6] 启动 EXP-01: Composition Baseline...")
    cmd = [
        "python", "scripts/train_multitask.py",
        "--split", "data/splits/split_iid_seed42.json",
        "--stage", "a",
        "--backbone", "composition",
        "--hidden-dim", "256",
        "--epochs", "50",
        "--batch-size", "32",
        "--lr", "3e-4",
        "--device", "cuda"
    ]

    log_file = PROJECT_ROOT / "logs" / "exp01_composition_baseline.log"
    log_file.parent.mkdir(exist_ok=True)

    with open(log_file, "w") as f:
        exp01_process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT)

    print(f"✅ EXP-01 已启动 (PID: {exp01_process.pid})")

    # 等待新的运行目录出现
    print("\n[2/6] 等待 EXP-01 创建运行目录...")
    exp01_run = None
    while True:
        latest_run = get_latest_run()
        if latest_run and latest_run.name not in existing_runs:
            print(f"检测到新运行: {latest_run.name}")
            exp01_run = latest_run
            break
        time.sleep(10)  # 每10秒检查一次

    # 等待训练完成
    print(f"\n[3/6] 等待 {exp01_run.name} 训练完成...")
    while True:
        if check_training_complete(exp01_run):
            print(f"✅ EXP-01 完成: {exp01_run.name}")
            break
        time.sleep(60)  # 每分钟检查一次

    # 分析 EXP-01
    print("\n[4/6] 分析 EXP-01 结果...")
    success, report = analyze_exp01(exp01_run)
    print(report)

    # 保存 EXP-01 报告
    report_file = PROJECT_ROOT / "reports" / "exp01_analysis.md"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(f"# EXP-01 分析报告\n\n{report}")

    if not success:
        print("⚠️  EXP-01 未达到成功标准，但继续执行 EXP-02 进行对比")

    # 启动 EXP-02
    print("\n[5/6] 启动 EXP-02: Graph Baseline...")
    cmd = [
        "python", "scripts/train_multitask.py",
        "--split", "data/splits/split_iid_seed42.json",
        "--stage", "a",
        "--backbone", "graph",
        "--hidden-dim", "256",
        "--layers", "6",
        "--epochs", "50",
        "--batch-size", "32",
        "--lr", "3e-4",
        "--device", "cuda"
    ]

    log_file = PROJECT_ROOT / "logs" / "exp02_graph_baseline.log"
    log_file.parent.mkdir(exist_ok=True)

    with open(log_file, "w") as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT)

    print(f"✅ EXP-02 已启动 (PID: {process.pid})")

    # 等待 EXP-02 完成
    print("\n等待 EXP-02 完成...")
    process.wait()

    # 获取 EXP-02 结果
    exp02_run = get_latest_run()
    print(f"✅ EXP-02 完成: {exp02_run.name}")

    # 对比分析
    print("\n[6/6] 对比分析并生成报告...")
    subprocess.run([
        "python", "scripts/compare_experiments.py",
        "--exp1", str(exp01_run),
        "--exp2", str(exp02_run),
        "--exp1-name", "EXP-01",
        "--exp2-name", "EXP-02"
    ], cwd=PROJECT_ROOT)

    # 生成 Phase 1 总结
    subprocess.run([
        "python", "scripts/generate_reports.py",
        "--type", "phase1",
        "--exp01", str(exp01_run),
        "--exp02", str(exp02_run)
    ], cwd=PROJECT_ROOT)

    print("\n" + "=" * 60)
    print("✅ Phase 1 所有任务完成！")

if __name__ == "__main__":
    main()

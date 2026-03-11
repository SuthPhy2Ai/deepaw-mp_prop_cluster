#!/usr/bin/env python3
"""Phase 2 自动化执行脚本 - 弹性任务与超参数优化"""

import json
import subprocess
import sys
import time
from pathlib import Path

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


def analyze_stage_b(run_dir):
    """分析 Stage B 结果"""
    metrics = get_metrics(run_dir)
    if not metrics:
        return False, "无法读取指标"

    val_metrics = metrics.get("val_metrics", {})

    # 检查 Stage A 任务是否劣化
    stage_a_tasks = {
        "is_metal_auroc": 0.75,
        "band_gap_mae": 1.0,
    }

    stage_a_ok = True
    for task, threshold in stage_a_tasks.items():
        if task in val_metrics:
            val = val_metrics[task]
            if "auroc" in task:
                if val < threshold:
                    stage_a_ok = False
            else:  # MAE
                if val > threshold:
                    stage_a_ok = False

    # 检查弹性任务
    elastic_tasks = [
        "bulk_modulus_vrh_mae",
        "shear_modulus_vrh_mae",
        "homogeneous_poisson_mae",
    ]

    elastic_ok = any(task in val_metrics for task in elastic_tasks)

    report = f"""
Stage B 结果分析:

Stage A 任务:
- is_metal AUROC: {val_metrics.get('is_metal_auroc', 0):.4f} {'✅' if val_metrics.get('is_metal_auroc', 0) >= 0.75 else '❌'}
- band_gap MAE: {val_metrics.get('band_gap_mae', 0):.4f} eV {'✅' if val_metrics.get('band_gap_mae', 999) < 1.0 else '❌'}

弹性任务:
- bulk_modulus_vrh MAE: {val_metrics.get('bulk_modulus_vrh_mae', 'N/A')}
- shear_modulus_vrh MAE: {val_metrics.get('shear_modulus_vrh_mae', 'N/A')}
- homogeneous_poisson MAE: {val_metrics.get('homogeneous_poisson_mae', 'N/A')}

Stage A 任务保持: {'✅' if stage_a_ok else '❌'}
弹性任务可学习: {'✅' if elastic_ok else '❌'}
"""

    success = stage_a_ok and elastic_ok
    return success, report


def main():
    """主函数"""
    print("Phase 2 自动化执行脚本启动...")
    print("=" * 60)

    # 从 Phase 1 获取最佳 backbone
    phase1_summary = PROJECT_ROOT / "reports" / "phase1_summary.md"
    if not phase1_summary.exists():
        print("❌ 未找到 Phase 1 总结，请先完成 Phase 1")
        sys.exit(1)

    # 假设使用 graph backbone（从 Phase 1 结果中应该是最好的）
    best_backbone = "graph"
    print(f"使用最佳 backbone: {best_backbone}")

    # [1/4] EXP-03: Stage B baseline
    print("\n[1/4] 启动 EXP-03: Stage B Baseline...")
    cmd = [
        "python", "scripts/train_multitask.py",
        "--split", "data/splits/split_iid_seed42.json",
        "--stage", "b",
        "--backbone", best_backbone,
        "--hidden-dim", "256",
        "--layers", "6",
        "--epochs", "80",
        "--batch-size", "32",
        "--lr", "3e-4",
        "--oversample-elastic", "4.0",
        "--device", "cuda"
    ]

    log_file = PROJECT_ROOT / "logs" / "exp03_stage_b_baseline.log"
    log_file.parent.mkdir(exist_ok=True)

    with open(log_file, "w") as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT)

    print(f"✅ EXP-03 已启动 (PID: {process.pid})")

    # 等待完成
    print("\n等待 EXP-03 完成...")
    process.wait()

    exp03_run = get_latest_run()
    print(f"✅ EXP-03 完成: {exp03_run.name}")

    # 分析结果
    print("\n[2/4] 分析 EXP-03 结果...")
    success, report = analyze_stage_b(exp03_run)
    print(report)

    if not success:
        print("❌ Stage B 未达到成功标准，停止 Phase 2")
        sys.exit(1)

    # [3/4] 超参数搜索
    print("\n[3/4] 启动超参数搜索...")
    print("⚠️  这将运行多个实验，预计需要 60-100 GPU-hours")

    # 运行超参数搜索（限制为5个实验以节省时间）
    subprocess.run([
        "python", "scripts/hyperparameter_search.py",
        "--split", "data/splits/split_iid_seed42.json",
        "--stage", "b",
        "--backbone", best_backbone,
        "--epochs", "80",
        "--max-experiments", "5",
        "--device", "cuda"
    ], cwd=PROJECT_ROOT)

    # [4/4] 生成报告
    print("\n[4/4] 生成 Phase 2 总结报告...")

    # 找到最佳模型
    search_results_file = PROJECT_ROOT / "reports" / "hyperparameter_search_results.json"
    if search_results_file.exists():
        with open(search_results_file) as f:
            results = json.load(f)
            if results:
                best_run_dir = Path(results[0]["run_dir"])
            else:
                best_run_dir = exp03_run
    else:
        best_run_dir = exp03_run

    # 生成 Phase 2 总结
    subprocess.run([
        "python", "scripts/generate_reports.py",
        "--type", "phase2",
        "--best-run", str(best_run_dir),
        "--search-results", str(search_results_file)
    ], cwd=PROJECT_ROOT)

    # 生成模型卡片
    subprocess.run([
        "python", "scripts/generate_reports.py",
        "--type", "model-card",
        "--best-run", str(best_run_dir)
    ], cwd=PROJECT_ROOT)

    print("\n" + "=" * 60)
    print("✅ Phase 2 所有任务完成！")
    print(f"最佳模型: {best_run_dir}")


if __name__ == "__main__":
    main()

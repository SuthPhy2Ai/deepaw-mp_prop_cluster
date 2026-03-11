#!/usr/bin/env python3
"""Phase 2 超参数搜索脚本"""

import argparse
import itertools
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_experiment(config, run_name):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"启动实验: {run_name}")
    print(f"配置: {config}")
    print(f"{'='*60}\n")

    cmd = [
        "python", "scripts/train_multitask.py",
        "--split", str(config["split"]),
        "--stage", config["stage"],
        "--backbone", config["backbone"],
        "--hidden-dim", str(config["hidden_dim"]),
        "--layers", str(config["layers"]),
        "--epochs", str(config["epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--oversample-elastic", str(config.get("oversample_elastic", 4.0)),
        "--device", config.get("device", "cuda"),
    ]

    log_file = PROJECT_ROOT / "logs" / f"{run_name}.log"
    log_file.parent.mkdir(exist_ok=True)

    start_time = time.time()

    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT
        )

    print(f"进程 PID: {process.pid}")
    print(f"日志文件: {log_file}")

    # 等待完成
    process.wait()

    elapsed = time.time() - start_time
    print(f"\n实验完成！耗时: {elapsed/3600:.2f} 小时")

    if process.returncode != 0:
        print(f"⚠️ 实验失败，退出码: {process.returncode}")
        return None

    # 获取最新的run目录
    runs_dir = PROJECT_ROOT / "artifacts" / "runs"
    runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
    if runs:
        return runs[-1]
    return None


def load_results(run_dir):
    """加载实验结果"""
    summary_file = run_dir / "metrics" / "best_summary.json"
    if not summary_file.exists():
        return None

    with open(summary_file) as f:
        return json.load(f)


def grid_search(search_space, base_config, max_experiments=None):
    """网格搜索"""
    # 生成所有组合
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    combinations = list(itertools.product(*values))

    if max_experiments and len(combinations) > max_experiments:
        print(f"⚠️ 组合数 ({len(combinations)}) 超过最大实验数 ({max_experiments})")
        print(f"将随机采样 {max_experiments} 个组合")
        import random
        random.shuffle(combinations)
        combinations = combinations[:max_experiments]

    print(f"\n总共 {len(combinations)} 个实验配置")

    results = []

    for idx, combo in enumerate(combinations, 1):
        config = base_config.copy()
        for key, value in zip(keys, combo):
            config[key] = value

        run_name = f"hp_search_{idx:02d}_" + "_".join(f"{k}{v}" for k, v in zip(keys, combo))

        print(f"\n[{idx}/{len(combinations)}] {run_name}")

        run_dir = run_experiment(config, run_name)

        if run_dir:
            metrics = load_results(run_dir)
            if metrics:
                results.append({
                    "config": config,
                    "run_dir": str(run_dir),
                    "run_name": run_name,
                    "metrics": metrics,
                })

    return results


def analyze_results(results, output_dir):
    """分析超参数搜索结果"""
    if not results:
        print("没有有效结果")
        return

    print(f"\n{'='*60}")
    print("超参数搜索结果分析")
    print(f"{'='*60}\n")

    # 按验证损失排序
    results_sorted = sorted(results, key=lambda x: x["metrics"]["best_val_loss"])

    print("Top 5 配置:\n")
    for idx, r in enumerate(results_sorted[:5], 1):
        print(f"{idx}. {r['run_name']}")
        print(f"   Val Loss: {r['metrics']['best_val_loss']:.4f}")
        print(f"   Config: {r['config']}")
        print()

    # 保存完整结果
    output_file = output_dir / "hyperparameter_search_results.json"
    with open(output_file, 'w') as f:
        json.dump(results_sorted, f, indent=2)
    print(f"完整结果已保存: {output_file}")

    # 生成报告
    report_file = output_dir / "hyperparameter_search_report.md"
    with open(report_file, 'w') as f:
        f.write("# 超参数搜索报告\n\n")
        f.write(f"**日期**: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**实验数**: {len(results)}\n\n")
        f.write("---\n\n")
        f.write("## Top 5 配置\n\n")

        for idx, r in enumerate(results_sorted[:5], 1):
            f.write(f"### {idx}. {r['run_name']}\n\n")
            f.write(f"- **Val Loss**: {r['metrics']['best_val_loss']:.4f}\n")
            f.write(f"- **Best Epoch**: {r['metrics']['best_epoch']}\n")
            f.write(f"- **Run Dir**: {r['run_dir']}\n\n")
            f.write("**配置**:\n")
            for k, v in r['config'].items():
                f.write(f"- {k}: {v}\n")
            f.write("\n**关键指标**:\n")
            for k, v in r['metrics']['val_metrics'].items():
                if k != "loss":
                    f.write(f"- {k}: {v:.4f}\n")
            f.write("\n---\n\n")

    print(f"报告已保存: {report_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="超参数搜索")
    parser.add_argument("--split", type=Path, required=True, help="Split文件")
    parser.add_argument("--stage", type=str, default="b", help="训练阶段")
    parser.add_argument("--backbone", type=str, default="graph", help="Backbone类型")
    parser.add_argument("--epochs", type=int, default=80, help="训练轮数")
    parser.add_argument("--max-experiments", type=int, help="最大实验数")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    args = parser.parse_args()

    # 基础配置
    base_config = {
        "split": args.split,
        "stage": args.stage,
        "backbone": args.backbone,
        "epochs": args.epochs,
        "device": args.device,
    }

    # 搜索空间
    search_space = {
        "lr": [1e-4, 3e-4, 5e-4],
        "hidden_dim": [128, 256, 384],
        "layers": [4, 6, 8],
        "batch_size": [16, 32, 64],
    }

    print("超参数搜索配置:")
    print(f"  基础配置: {base_config}")
    print(f"  搜索空间: {search_space}")

    # 执行搜索
    results = grid_search(search_space, base_config, args.max_experiments)

    # 分析结果
    output_dir = PROJECT_ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    analyze_results(results, output_dir)


if __name__ == "__main__":
    main()

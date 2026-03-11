#!/usr/bin/env python3
"""对比两个实验的结果"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_metrics(run_dir):
    """加载实验指标"""
    summary_file = run_dir / "metrics" / "best_summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary not found: {summary_file}")

    with open(summary_file) as f:
        return json.load(f)


def compare_metrics(exp1_metrics, exp2_metrics, exp1_name, exp2_name):
    """对比两个实验的指标"""
    val1 = exp1_metrics["val_metrics"]
    val2 = exp2_metrics["val_metrics"]

    # 所有任务
    all_tasks = sorted(set(val1.keys()) | set(val2.keys()))

    results = []
    better_count = 0
    total_count = 0

    for task in all_tasks:
        if task == "loss":
            continue

        if task not in val1 or task not in val2:
            continue

        v1 = val1[task]
        v2 = val2[task]

        # 判断方向（AUROC越高越好，MAE越低越好）
        if "auroc" in task or "auc" in task:
            better = v2 > v1
            change = v2 - v1
            direction = "higher"
        else:  # MAE, MSE, loss
            better = v2 < v1
            change = v1 - v2
            direction = "lower"

        if better:
            better_count += 1
        total_count += 1

        results.append({
            "task": task,
            "exp1": v1,
            "exp2": v2,
            "change": change,
            "better": better,
            "direction": direction,
        })

    return results, better_count, total_count


def print_comparison(results, better_count, total_count, exp1_name, exp2_name):
    """打印对比结果"""
    print("\n" + "=" * 80)
    print(f"实验对比: {exp1_name} vs {exp2_name}")
    print("=" * 80)

    print(f"\n{'Task':<40} {'EXP1':>10} {'EXP2':>10} {'Change':>10} {'Better':>8}")
    print("-" * 80)

    for r in results:
        symbol = "✅" if r["better"] else "❌"
        print(f"{r['task']:<40} {r['exp1']:>10.4f} {r['exp2']:>10.4f} {r['change']:>+10.4f} {symbol:>8}")

    print("-" * 80)
    print(f"{exp2_name} 优于 {exp1_name}: {better_count}/{total_count} ({better_count/total_count*100:.1f}%)")
    print("=" * 80)


def plot_comparison(results, exp1_name, exp2_name, output_path):
    """绘制对比图"""
    # 分类任务和回归任务
    classification_tasks = [r for r in results if "auroc" in r["task"] or "auc" in r["task"]]
    regression_tasks = [r for r in results if r not in classification_tasks]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 分类任务 (AUROC)
    if classification_tasks:
        tasks = [r["task"].replace("_auroc", "").replace("_", " ").title() for r in classification_tasks]
        exp1_vals = [r["exp1"] for r in classification_tasks]
        exp2_vals = [r["exp2"] for r in classification_tasks]

        x = np.arange(len(tasks))
        width = 0.35

        axes[0].bar(x - width/2, exp1_vals, width, label=exp1_name, alpha=0.8)
        axes[0].bar(x + width/2, exp2_vals, width, label=exp2_name, alpha=0.8)
        axes[0].set_xlabel('Task')
        axes[0].set_ylabel('AUROC')
        axes[0].set_title('Classification Tasks (Higher is Better)')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(tasks, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_ylim([0, 1])

    # 回归任务 (MAE)
    if regression_tasks:
        tasks = [r["task"].replace("_mae", "").replace("_", " ").title() for r in regression_tasks]
        exp1_vals = [r["exp1"] for r in regression_tasks]
        exp2_vals = [r["exp2"] for r in regression_tasks]

        x = np.arange(len(tasks))
        width = 0.35

        axes[1].bar(x - width/2, exp1_vals, width, label=exp1_name, alpha=0.8)
        axes[1].bar(x + width/2, exp2_vals, width, label=exp2_name, alpha=0.8)
        axes[1].set_xlabel('Task')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Regression Tasks (Lower is Better)')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(tasks, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n对比图已保存: {output_path}")


def generate_markdown_report(results, better_count, total_count, exp1_name, exp2_name,
                             exp1_dir, exp2_dir, output_path):
    """生成Markdown报告"""
    report = f"""# 实验对比报告

**日期**: {Path(exp1_dir).name} vs {Path(exp2_dir).name}

---

## 实验配置

### {exp1_name}
- Run ID: {Path(exp1_dir).name}
- 配置: {exp1_dir}/config.json

### {exp2_name}
- Run ID: {Path(exp2_dir).name}
- 配置: {exp2_dir}/config.json

---

## 指标对比

| Task | {exp1_name} | {exp2_name} | Change | Better |
|------|-------------|-------------|--------|--------|
"""

    for r in results:
        symbol = "✅" if r["better"] else "❌"
        report += f"| {r['task']} | {r['exp1']:.4f} | {r['exp2']:.4f} | {r['change']:+.4f} | {symbol} |\n"

    report += f"""
---

## 总结

**{exp2_name} 优于 {exp1_name}**: {better_count}/{total_count} ({better_count/total_count*100:.1f}%)

"""

    if better_count / total_count > 0.7:
        report += f"✅ **结论**: {exp2_name} 显著优于 {exp1_name}\n"
    elif better_count / total_count > 0.5:
        report += f"⚠️ **结论**: {exp2_name} 略优于 {exp1_name}\n"
    else:
        report += f"❌ **结论**: {exp2_name} 未优于 {exp1_name}\n"

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"报告已保存: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="对比两个实验")
    parser.add_argument("--exp1", type=Path, required=True, help="实验1目录")
    parser.add_argument("--exp2", type=Path, required=True, help="实验2目录")
    parser.add_argument("--exp1-name", type=str, default="EXP-01", help="实验1名称")
    parser.add_argument("--exp2-name", type=str, default="EXP-02", help="实验2名称")
    parser.add_argument("--output", type=Path, help="输出报告路径")
    args = parser.parse_args()

    # 加载指标
    print(f"加载 {args.exp1_name}: {args.exp1}")
    exp1_metrics = load_metrics(args.exp1)

    print(f"加载 {args.exp2_name}: {args.exp2}")
    exp2_metrics = load_metrics(args.exp2)

    # 对比
    results, better_count, total_count = compare_metrics(
        exp1_metrics, exp2_metrics, args.exp1_name, args.exp2_name
    )

    # 打印结果
    print_comparison(results, better_count, total_count, args.exp1_name, args.exp2_name)

    # 生成图表
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_path = figures_dir / f"comparison_{args.exp1_name}_vs_{args.exp2_name}.png"
    plot_comparison(results, args.exp1_name, args.exp2_name, plot_path)

    # 生成报告
    if args.output:
        output_path = args.output
    else:
        reports_dir = PROJECT_ROOT / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = reports_dir / f"comparison_{args.exp1_name}_vs_{args.exp2_name}.md"

    generate_markdown_report(
        results, better_count, total_count,
        args.exp1_name, args.exp2_name,
        args.exp1, args.exp2,
        output_path
    )


if __name__ == "__main__":
    main()

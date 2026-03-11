#!/usr/bin/env python3
"""完整流程自动化脚本 - Phase 0 到 Phase 2"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_phase0():
    """运行 Phase 0: 基础设施验证"""
    print("\n" + "=" * 60)
    print("Phase 0: 基础设施验证")
    print("=" * 60)

    # 数据质量分析
    print("\n[1/3] 数据质量分析...")
    subprocess.run(["python", "scripts/analyze_data_quality.py"], cwd=PROJECT_ROOT, check=True)

    # 归一化测试
    print("\n[2/3] 归一化测试...")
    subprocess.run(["python", "-m", "pytest", "tests/test_normalization.py", "-v"], cwd=PROJECT_ROOT, check=True)

    # 归一化可视化
    print("\n[3/3] 归一化可视化...")
    subprocess.run(["python", "scripts/visualize_normalization.py"], cwd=PROJECT_ROOT, check=True)

    print("\n✅ Phase 0 完成！")


def run_phase1():
    """运行 Phase 1: Baseline 建立"""
    print("\n" + "=" * 60)
    print("Phase 1: Baseline 建立")
    print("=" * 60)

    # 直接运行自动化脚本（它会启动 EXP-01，等待完成，然后启动 EXP-02）
    print("\n启动 Phase 1 自动化脚本...")
    print("⚠️  这将需要 20-30 GPU-hours，请耐心等待...")
    subprocess.run(["python", "scripts/phase1_automation.py"], cwd=PROJECT_ROOT, check=True)

    print("\n✅ Phase 1 完成！")


def run_phase2():
    """运行 Phase 2: 弹性任务与优化"""
    print("\n" + "=" * 60)
    print("Phase 2: 弹性任务与优化")
    print("=" * 60)

    subprocess.run(["python", "scripts/phase2_automation.py"], cwd=PROJECT_ROOT, check=True)

    print("\n✅ Phase 2 完成！")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="完整流程自动化")
    parser.add_argument("--phase", type=str, choices=["0", "1", "2", "all"],
                       default="all", help="运行哪个阶段")
    parser.add_argument("--skip-phase0", action="store_true",
                       help="跳过 Phase 0（如果已完成）")
    args = parser.parse_args()

    print("=" * 60)
    print("多任务材料性质预测 - 完整流程自动化")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  运行阶段: {args.phase}")
    print(f"  跳过 Phase 0: {args.skip_phase0}")

    try:
        if args.phase == "all":
            if not args.skip_phase0:
                run_phase0()
            run_phase1()
            run_phase2()
        elif args.phase == "0":
            run_phase0()
        elif args.phase == "1":
            run_phase1()
        elif args.phase == "2":
            run_phase2()

        print("\n" + "=" * 60)
        print("✅ 所有任务完成！")
        print("=" * 60)

        # 显示最终结果位置
        print("\n最终输出:")
        print(f"  - Phase 1 总结: reports/phase1_summary.md")
        print(f"  - Phase 2 总结: reports/phase2_summary.md")
        print(f"  - 最佳模型卡片: reports/best_model_card.md")
        print(f"  - 所有运行: artifacts/runs/")
        print(f"  - 所有日志: logs/")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ 执行失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(1)


if __name__ == "__main__":
    main()

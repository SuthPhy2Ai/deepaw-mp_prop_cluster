#!/usr/bin/env python3
"""可视化归一化前后的数据分布"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.db import connect

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mp_data_pipeline.config import DB_PATH
from mp_data_pipeline.ml.normalization import NORMALIZATION_STATS


def plot_distribution(values, title, xlabel, output_path, normalized_values=None):
    """绘制分布图"""
    fig, axes = plt.subplots(1, 2 if normalized_values is not None else 1, figsize=(12, 4))

    if normalized_values is None:
        axes = [axes]

    # 原始分布
    axes[0].hist(values, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{title} - Original')
    axes[0].grid(True, alpha=0.3)

    # 添加统计信息
    mean_val = np.mean(values)
    std_val = np.std(values)
    axes[0].axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.3f}')
    axes[0].axvline(mean_val + std_val, color='g', linestyle='--', alpha=0.5, label=f'±1σ: {std_val:.3f}')
    axes[0].axvline(mean_val - std_val, color='g', linestyle='--', alpha=0.5)
    axes[0].legend()

    # 归一化后分布
    if normalized_values is not None:
        axes[1].hist(normalized_values, bins=50, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel(f'{xlabel} (normalized)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{title} - Normalized')
        axes[1].grid(True, alpha=0.3)

        norm_mean = np.mean(normalized_values)
        norm_std = np.std(normalized_values)
        axes[1].axvline(norm_mean, color='r', linestyle='--', label=f'Mean: {norm_mean:.3f}')
        axes[1].axvline(norm_mean + norm_std, color='g', linestyle='--', alpha=0.5, label=f'±1σ: {norm_std:.3f}')
        axes[1].axvline(norm_mean - norm_std, color='g', linestyle='--', alpha=0.5)
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("归一化可视化")
    print("=" * 60)

    db = connect(str(DB_PATH))

    # 创建输出目录
    output_dir = PROJECT_ROOT / "reports" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 需要可视化的属性
    properties = [
        ("energy_per_atom", "Energy per Atom (eV)"),
        ("formation_energy_per_atom", "Formation Energy per Atom (eV)"),
        ("energy_above_hull", "Energy Above Hull (eV)"),
        ("band_gap", "Band Gap (eV)"),
        ("volume", "Volume (Å³)"),
        ("density", "Density (g/cm³)"),
    ]

    for prop, label in properties:
        print(f"\n处理: {prop}")

        # 收集数据
        values = []
        for row in db.select():
            val = row.get(prop)
            if val is not None:
                values.append(float(val))

        if not values:
            print(f"  跳过 {prop}: 无数据")
            continue

        values = np.array(values)
        print(f"  样本数: {len(values)}")
        print(f"  范围: [{values.min():.3f}, {values.max():.3f}]")
        print(f"  均值: {values.mean():.3f}, 标准差: {values.std():.3f}")

        # 归一化
        if prop in NORMALIZATION_STATS:
            stats = NORMALIZATION_STATS[prop]
            normalized = (values - stats["mean"]) / stats["std"]

            print(f"  归一化后均值: {normalized.mean():.6f}, 标准差: {normalized.std():.6f}")

            output_path = output_dir / f"normalization_{prop}.png"
            plot_distribution(values, prop, label, output_path, normalized)
        else:
            output_path = output_dir / f"distribution_{prop}.png"
            plot_distribution(values, prop, label, output_path)

    print("\n" + "=" * 60)
    print("可视化完成！")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()

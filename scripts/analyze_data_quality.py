#!/usr/bin/env python3
"""数据质量分析脚本 - Phase 0 验证"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase.db import connect

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mp_data_pipeline.config import DB_PATH
from mp_data_pipeline.ml.splits import load_split
from mp_data_pipeline.ml.tasks import ELASTIC_TASKS

# 弹性属性的物理范围
ELASTIC_RANGES = {
    "bulk_modulus_vrh": (0, 1000),  # GPa
    "shear_modulus_vrh": (0, 500),  # GPa
    "homogeneous_poisson": (-1.0, 0.5),
    "universal_anisotropy": (0, 100),
}


def analyze_elastic_data(db_path: Path):
    """分析弹性数据覆盖率和分布"""
    print("\n" + "=" * 60)
    print("1. 弹性数据统计")
    print("=" * 60)

    db = connect(str(db_path))
    total_count = db.count()

    elastic_stats = {task: {"count": 0, "values": [], "outliers": []} for task in ELASTIC_TASKS}

    for row in db.select():
        for task in ELASTIC_TASKS:
            val = row.get(task)
            if val is not None:
                elastic_stats[task]["count"] += 1
                elastic_stats[task]["values"].append(float(val))

                # 检查物理范围
                min_val, max_val = ELASTIC_RANGES[task]
                if not (min_val <= val <= max_val):
                    elastic_stats[task]["outliers"].append({
                        "mp_id": row.get("mp_id"),
                        "value": float(val),
                    })

    print(f"\n总样本数: {total_count:,}")
    print(f"\n弹性属性覆盖率:")

    for task in ELASTIC_TASKS:
        count = elastic_stats[task]["count"]
        coverage = count / total_count * 100
        print(f"  {task:30s}: {count:6,} ({coverage:5.2f}%)")

        if elastic_stats[task]["values"]:
            values = np.array(elastic_stats[task]["values"])
            print(f"    范围: [{values.min():.3f}, {values.max():.3f}]")
            print(f"    均值: {values.mean():.3f}, 标准差: {values.std():.3f}")
            print(f"    中位数: {np.median(values):.3f}")

            # 异常值统计
            n_outliers = len(elastic_stats[task]["outliers"])
            if n_outliers > 0:
                outlier_pct = n_outliers / count * 100
                print(f"    ⚠️  异常值: {n_outliers} ({outlier_pct:.2f}%)")
                if n_outliers <= 5:
                    for outlier in elastic_stats[task]["outliers"]:
                        print(f"      - {outlier['mp_id']}: {outlier['value']:.3f}")

    # 计算有任意弹性数据的样本数
    elastic_mp_ids = set()
    for row in db.select():
        if any(row.get(task) is not None for task in ELASTIC_TASKS):
            elastic_mp_ids.add(row.get("mp_id"))

    print(f"\n有任意弹性数据的样本: {len(elastic_mp_ids):,} ({len(elastic_mp_ids)/total_count*100:.2f}%)")

    return elastic_stats, len(elastic_mp_ids)


def analyze_split_balance(split_path: Path, db_path: Path):
    """分析 split 平衡性"""
    print("\n" + "=" * 60)
    print("2. Split 平衡性分析")
    print("=" * 60)

    split = load_split(split_path)
    db = connect(str(db_path))

    # 构建 mp_id 到 row 的映射
    rows_by_id = {}
    for row in db.select():
        rows_by_id[row.get("mp_id")] = row

    for split_name in ["train", "val", "test"]:
        mp_ids = split[split_name]
        print(f"\n{split_name.upper()} Split:")
        print(f"  总样本数: {len(mp_ids):,}")

        # 统计金属/非金属
        metal_count = 0
        stable_count = 0
        elastic_count = 0
        element_counts = []

        for mp_id in mp_ids:
            if mp_id not in rows_by_id:
                continue
            row = rows_by_id[mp_id]

            if row.get("is_metal"):
                metal_count += 1
            if row.get("is_stable"):
                stable_count += 1
            if any(row.get(task) is not None for task in ELASTIC_TASKS):
                elastic_count += 1

            # 元素数量
            atoms = row.toatoms()
            element_counts.append(len(set(atoms.numbers)))

        metal_pct = metal_count / len(mp_ids) * 100
        stable_pct = stable_count / len(mp_ids) * 100
        elastic_pct = elastic_count / len(mp_ids) * 100

        print(f"  金属比例: {metal_count:,} ({metal_pct:.2f}%)")
        print(f"  稳定比例: {stable_count:,} ({stable_pct:.2f}%)")
        print(f"  弹性覆盖: {elastic_count:,} ({elastic_pct:.2f}%)")
        print(f"  平均元素数: {np.mean(element_counts):.2f} ± {np.std(element_counts):.2f}")


def check_chemsys_leakage(split_path: Path, db_path: Path):
    """检查 ChemSys-OOD 信息泄漏"""
    print("\n" + "=" * 60)
    print("3. ChemSys-OOD 信息泄漏检查")
    print("=" * 60)

    split = load_split(split_path)
    db = connect(str(db_path))

    # 收集每个 split 的元素集合
    train_elements = set()
    test_elements = set()

    rows_by_id = {}
    for row in db.select():
        rows_by_id[row.get("mp_id")] = row

    for mp_id in split["train"]:
        if mp_id in rows_by_id:
            atoms = rows_by_id[mp_id].toatoms()
            train_elements.update(atoms.numbers)

    for mp_id in split["test"]:
        if mp_id in rows_by_id:
            atoms = rows_by_id[mp_id].toatoms()
            test_elements.update(atoms.numbers)

    overlap = train_elements & test_elements
    train_only = train_elements - test_elements
    test_only = test_elements - train_elements

    print(f"\n训练集元素数: {len(train_elements)}")
    print(f"测试集元素数: {len(test_elements)}")
    print(f"重叠元素数: {len(overlap)} ({len(overlap)/len(test_elements)*100:.1f}% of test)")

    if test_only:
        print(f"\n⚠️  测试集独有元素 (原子序数): {sorted(test_only)}")
    else:
        print(f"\n✅ 测试集没有独有元素（完全重叠）")

    # 检查化学系统级别的泄漏
    train_chemsys = set()
    test_chemsys = set()

    for mp_id in split["train"][:1000]:  # 采样检查
        if mp_id in rows_by_id:
            atoms = rows_by_id[mp_id].toatoms()
            chemsys = tuple(sorted(set(atoms.numbers)))
            train_chemsys.add(chemsys)

    for mp_id in split["test"][:1000]:
        if mp_id in rows_by_id:
            atoms = rows_by_id[mp_id].toatoms()
            chemsys = tuple(sorted(set(atoms.numbers)))
            test_chemsys.add(chemsys)

    chemsys_overlap = train_chemsys & test_chemsys
    print(f"\n化学系统重叠 (采样 1000):")
    print(f"  训练集化学系统数: {len(train_chemsys)}")
    print(f"  测试集化学系统数: {len(test_chemsys)}")
    print(f"  重叠化学系统数: {len(chemsys_overlap)} ({len(chemsys_overlap)/len(test_chemsys)*100:.1f}%)")


def detect_outliers(db_path: Path):
    """检测异常值"""
    print("\n" + "=" * 60)
    print("4. 异常值检测 (3-sigma 规则)")
    print("=" * 60)

    db = connect(str(db_path))

    # 收集所有数值属性
    properties = [
        "energy_per_atom",
        "formation_energy_per_atom",
        "energy_above_hull",
        "band_gap",
        "volume",
        "density",
    ] + ELASTIC_TASKS

    prop_values = {prop: [] for prop in properties}

    for row in db.select():
        for prop in properties:
            val = row.get(prop)
            if val is not None:
                prop_values[prop].append(float(val))

    print("\n异常值统计 (超出 mean ± 3*std):")
    total_outliers = 0

    for prop in properties:
        if not prop_values[prop]:
            continue

        values = np.array(prop_values[prop])
        mean = values.mean()
        std = values.std()

        lower = mean - 3 * std
        upper = mean + 3 * std

        outliers = np.sum((values < lower) | (values > upper))
        outlier_pct = outliers / len(values) * 100

        if outlier_pct > 1.0:  # 只显示 > 1% 的
            print(f"  {prop:30s}: {outliers:5,} / {len(values):6,} ({outlier_pct:5.2f}%)")
            total_outliers += outliers

    print(f"\n总异常值数: {total_outliers:,}")


def main():
    """主函数"""
    print("=" * 60)
    print("数据质量分析报告")
    print("=" * 60)
    print(f"数据库: {DB_PATH}")

    # 1. 弹性数据分析
    elastic_stats, elastic_count = analyze_elastic_data(DB_PATH)

    # 2. Split 平衡性 (IID)
    iid_split = PROJECT_ROOT / "data" / "splits" / "split_iid_seed42.json"
    if iid_split.exists():
        analyze_split_balance(iid_split, DB_PATH)

    # 3. ChemSys-OOD 泄漏检查
    chemsys_split = PROJECT_ROOT / "data" / "splits" / "split_chemsys_ood_seed42.json"
    if chemsys_split.exists():
        check_chemsys_leakage(chemsys_split, DB_PATH)

    # 4. 异常值检测
    detect_outliers(DB_PATH)

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)

    if elastic_count >= 10000:
        print(f"✅ 弹性数据充足: {elastic_count:,} >= 10,000")
    elif elastic_count >= 5000:
        print(f"⚠️  弹性数据偏少: {elastic_count:,} (5,000-10,000)")
    else:
        print(f"❌ 弹性数据不足: {elastic_count:,} < 5,000")
        print("   建议: 放弃 Stage B，只做 Stage A")

    print("\n报告生成完成！")


if __name__ == "__main__":
    main()

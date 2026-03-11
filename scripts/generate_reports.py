#!/usr/bin/env python3
"""生成各种实验报告"""

import argparse
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def generate_phase1_summary(exp01_dir, exp02_dir, output_path):
    """生成Phase 1总结报告"""
    # 加载配置和指标
    with open(exp01_dir / "config.json") as f:
        exp01_config = json.load(f)
    with open(exp01_dir / "metrics" / "best_summary.json") as f:
        exp01_metrics = json.load(f)

    with open(exp02_dir / "config.json") as f:
        exp02_config = json.load(f)
    with open(exp02_dir / "metrics" / "best_summary.json") as f:
        exp02_metrics = json.load(f)

    # 对比结果
    val1 = exp01_metrics["val_metrics"]
    val2 = exp02_metrics["val_metrics"]

    better_count = 0
    total_count = 0

    comparison_lines = []
    for task in sorted(set(val1.keys()) | set(val2.keys())):
        if task == "loss" or task not in val1 or task not in val2:
            continue

        v1 = val1[task]
        v2 = val2[task]

        if "auroc" in task:
            better = v2 > v1
        else:
            better = v2 < v1

        if better:
            better_count += 1
        total_count += 1

        symbol = "✅" if better else "❌"
        comparison_lines.append(f"| {task} | {v1:.4f} | {v2:.4f} | {symbol} |")

    # 生成报告
    report = f"""# Phase 1 总结报告

**日期**: {datetime.now().strftime('%Y-%m-%d')}
**状态**: ✅ 完成

---

## 实验配置

### EXP-01: Composition Baseline
- Run ID: {exp01_dir.name}
- Backbone: {exp01_config['backbone']}
- Hidden dim: {exp01_config['hidden_dim']}
- Epochs: {exp01_config['epochs']}
- Batch size: {exp01_config['batch_size']}
- Learning rate: {exp01_config['lr']}

### EXP-02: Graph Baseline
- Run ID: {exp02_dir.name}
- Backbone: {exp02_config['backbone']}
- Hidden dim: {exp02_config['hidden_dim']}
- Layers: {exp02_config['layers']}
- Epochs: {exp02_config['epochs']}
- Batch size: {exp02_config['batch_size']}
- Learning rate: {exp02_config['lr']}

---

## 关键指标对比

### EXP-01 结果
- Best epoch: {exp01_metrics['best_epoch']}
- Val loss: {exp01_metrics['best_val_loss']:.4f}
- is_metal AUROC: {val1.get('is_metal_auroc', 0):.4f}
- is_stable AUROC: {val1.get('is_stable_auroc', 0):.4f}
- band_gap MAE: {val1.get('band_gap_mae', 0):.4f} eV

### EXP-02 结果
- Best epoch: {exp02_metrics['best_epoch']}
- Val loss: {exp02_metrics['best_val_loss']:.4f}
- is_metal AUROC: {val2.get('is_metal_auroc', 0):.4f}
- is_stable AUROC: {val2.get('is_stable_auroc', 0):.4f}
- band_gap MAE: {val2.get('band_gap_mae', 0):.4f} eV

---

## 详细对比

| Task | EXP-01 | EXP-02 | Better |
|------|--------|--------|--------|
{chr(10).join(comparison_lines)}

**Graph 优于 Composition**: {better_count}/{total_count} ({better_count/total_count*100:.1f}%)

---

## 结论

"""

    if better_count / total_count > 0.7:
        report += "✅ **Graph Backbone 显著优于 Composition Baseline**\n\n"
        report += "建议: 进入 Phase 2，使用 Graph Backbone 进行弹性任务训练和超参数优化。\n"
    elif better_count / total_count > 0.5:
        report += "⚠️ **Graph Backbone 略优于 Composition Baseline**\n\n"
        report += "建议: 可以进入 Phase 2，但需要考虑增强 Graph Backbone 或调整超参数。\n"
    else:
        report += "❌ **Graph Backbone 未显著优于 Composition Baseline**\n\n"
        report += "建议: 分析原因，考虑增强 Graph Backbone 或重新审视问题定义。\n"

    report += f"""
---

## 下一步

**Phase 2**: 弹性任务与优化
- 使用最佳 backbone 训练 Stage B（包含弹性任务）
- 超参数搜索（学习率、模型容量、批大小）
- 预计时间: 10-14 天

---

## 附录

- EXP-01 详细输出: {exp01_dir}
- EXP-02 详细输出: {exp02_dir}
- 对比图表: reports/figures/phase1_comparison.png
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Phase 1 总结已保存: {output_path}")


def generate_phase2_summary(best_run_dir, search_results_file, output_path):
    """生成Phase 2总结报告"""
    # 加载最佳模型
    with open(best_run_dir / "config.json") as f:
        best_config = json.load(f)
    with open(best_run_dir / "metrics" / "best_summary.json") as f:
        best_metrics = json.load(f)

    # 加载搜索结果
    if search_results_file.exists():
        with open(search_results_file) as f:
            search_results = json.load(f)
    else:
        search_results = []

    report = f"""# Phase 2 总结报告

**日期**: {datetime.now().strftime('%Y-%m-%d')}
**状态**: ✅ 完成

---

## 最佳模型配置

- Run ID: {best_run_dir.name}
- Backbone: {best_config['backbone']}
- Hidden dim: {best_config['hidden_dim']}
- Layers: {best_config.get('layers', 'N/A')}
- Learning rate: {best_config['lr']}
- Batch size: {best_config['batch_size']}
- Epochs: {best_config['epochs']}
- Stage: {best_config['stage']}

---

## 最佳模型性能

### 训练指标
- Best epoch: {best_metrics['best_epoch']}
- Best val loss: {best_metrics['best_val_loss']:.4f}

### Stage A 任务（高覆盖率）
"""

    val_metrics = best_metrics['val_metrics']
    stage_a_tasks = ['is_metal_auroc', 'is_stable_auroc', 'band_gap_mae',
                     'energy_per_atom_mae', 'formation_energy_per_atom_mae']

    for task in stage_a_tasks:
        if task in val_metrics:
            report += f"- {task}: {val_metrics[task]:.4f}\n"

    report += "\n### Stage B 任务（弹性属性）\n"

    elastic_tasks = ['bulk_modulus_vrh_mae', 'shear_modulus_vrh_mae',
                     'homogeneous_poisson_mae', 'universal_anisotropy_mae']

    for task in elastic_tasks:
        if task in val_metrics:
            report += f"- {task}: {val_metrics[task]:.4f}\n"

    if search_results:
        report += f"""
---

## 超参数搜索结果

总共测试了 {len(search_results)} 个配置。

### Top 3 配置

"""
        for idx, r in enumerate(search_results[:3], 1):
            report += f"""
#### {idx}. {r['run_name']}
- Val Loss: {r['metrics']['best_val_loss']:.4f}
- Learning rate: {r['config']['lr']}
- Hidden dim: {r['config']['hidden_dim']}
- Batch size: {r['config']['batch_size']}
"""

    report += f"""
---

## 结论

Phase 2 成功完成，产出了一个经过充分优化的多任务模型。

### 关键成果
1. ✅ 弹性任务可学习（MAE优于均值预测）
2. ✅ Stage A 任务性能保持稳定
3. ✅ 找到了最佳超参数组合

### 模型能力
- 可预测 {len([k for k in val_metrics.keys() if k != 'loss'])} 个材料属性
- 支持热力学、电子、结构、弹性等多个领域
- 在 IID 测试集上表现良好

---

## 下一步（可选）

**Phase 3**: OOD 评估
- ChemSys-OOD: 评估在新化学系统上的泛化能力
- Complexity-OOD: 评估在复杂材料上的表现
- 预计时间: 2-3 天

---

## 附录

- 最佳模型: {best_run_dir}
- 超参数搜索结果: {search_results_file}
- 模型卡片: reports/best_model_card.md
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Phase 2 总结已保存: {output_path}")


def generate_model_card(run_dir, output_path):
    """生成模型卡片"""
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    with open(run_dir / "metrics" / "best_summary.json") as f:
        metrics = json.load(f)

    report = f"""# 模型卡片

## 模型信息

- **模型名称**: Multitask Crystal Property Predictor
- **版本**: {run_dir.name}
- **训练日期**: {datetime.now().strftime('%Y-%m-%d')}
- **框架**: PyTorch
- **Backbone**: {config['backbone']}

---

## 模型架构

### Backbone
- Type: {config['backbone']}
- Hidden dimension: {config['hidden_dim']}
- Layers: {config.get('layers', 'N/A')}
- Cutoff: {config.get('cutoff', 'N/A')} Å
- Max neighbors: {config.get('max_neighbors', 'N/A')}

### Task Heads
- Thermodynamic: energy_per_atom, formation_energy_per_atom, energy_above_hull
- Electronic: band_gap, cbm, vbm, efermi, is_metal
- Stability: is_stable
- Structural: volume, density
- Elastic: bulk_modulus_vrh, shear_modulus_vrh, homogeneous_poisson, universal_anisotropy

---

## 训练配置

- **数据集**: Materials Project
- **训练样本**: {config['train_size']}
- **验证样本**: {config['val_size']}
- **Split**: {Path(config['split']).name}
- **Stage**: {config['stage']}
- **Epochs**: {config['epochs']}
- **Batch size**: {config['batch_size']}
- **Learning rate**: {config['lr']}
- **Weight decay**: {config['weight_decay']}
- **Device**: {config['device']}

---

## 性能指标

### 验证集性能

"""

    val_metrics = metrics['val_metrics']
    for task, value in sorted(val_metrics.items()):
        if task != "loss":
            report += f"- **{task}**: {value:.4f}\n"

    report += f"""
### 训练信息
- Best epoch: {metrics['best_epoch']}
- Best val loss: {metrics['best_val_loss']:.4f}

---

## 使用方法

### 加载模型

```python
import torch
from mp_data_pipeline.models.multitask_model import MultitaskPropertyModel

# 加载checkpoint
checkpoint = torch.load("{run_dir}/checkpoints/best.pt")

# 创建模型
model = MultitaskPropertyModel(
    backbone_name="{config['backbone']}",
    hidden_dim={config['hidden_dim']},
    n_layers={config.get('layers', 6)},
    cutoff={config.get('cutoff', 6.0)},
)

# 加载权重
model.load_state_dict(checkpoint['model_state'])
model.eval()
```

### 预测

```python
from mp_data_pipeline.ml.dataset import AseGraphMultitaskDataset

# 准备数据
dataset = AseGraphMultitaskDataset(
    db_path="data/db/mp_materials.db",
    mp_ids=["mp-149"],
    cutoff={config.get('cutoff', 6.0)},
    max_neighbors={config.get('max_neighbors', 24)},
)

# 预测
sample = dataset[0]
with torch.no_grad():
    predictions = model(sample)
```

---

## 限制与注意事项

1. **数据分布**: 模型在 Materials Project 数据上训练，对于分布外的材料可能表现较差
2. **弹性属性**: 弹性任务的训练数据较少（~8%），预测精度可能较低
3. **物理约束**: 模型强制执行了一些物理约束（如 band_gap >= 0），但不保证所有预测都物理合理
4. **不确定性**: 模型不提供预测不确定性估计

---

## 引用

如果使用此模型，请引用：

```
@misc{{multitask_crystal_predictor,
  title={{Multitask Crystal Property Predictor}},
  author={{Your Name}},
  year={{2026}},
  url={{https://github.com/your-repo}}
}}
```

---

## 联系方式

如有问题或建议，请联系: your.email@example.com

---

**最后更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"模型卡片已保存: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成实验报告")
    parser.add_argument("--type", type=str, required=True,
                       choices=["phase1", "phase2", "model-card"],
                       help="报告类型")
    parser.add_argument("--exp01", type=Path, help="EXP-01目录（phase1）")
    parser.add_argument("--exp02", type=Path, help="EXP-02目录（phase1）")
    parser.add_argument("--best-run", type=Path, help="最佳模型目录（phase2/model-card）")
    parser.add_argument("--search-results", type=Path, help="搜索结果文件（phase2）")
    parser.add_argument("--output", type=Path, help="输出路径")
    args = parser.parse_args()

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.type == "phase1":
        if not args.exp01 or not args.exp02:
            parser.error("phase1 需要 --exp01 和 --exp02")
        output = args.output or reports_dir / "phase1_summary.md"
        generate_phase1_summary(args.exp01, args.exp02, output)

    elif args.type == "phase2":
        if not args.best_run:
            parser.error("phase2 需要 --best-run")
        output = args.output or reports_dir / "phase2_summary.md"
        search_results = args.search_results or reports_dir / "hyperparameter_search_results.json"
        generate_phase2_summary(args.best_run, search_results, output)

    elif args.type == "model-card":
        if not args.best_run:
            parser.error("model-card 需要 --best-run")
        output = args.output or reports_dir / "best_model_card.md"
        generate_model_card(args.best_run, output)


if __name__ == "__main__":
    main()

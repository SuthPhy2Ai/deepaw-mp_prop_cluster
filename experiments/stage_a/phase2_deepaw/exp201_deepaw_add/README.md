# EXP-201: DeePAW + Enhanced Graph (Add Fusion)

## 实验目标

验证 DeePAW 预训练原子特征对材料性质预测的改进效果，特别是电子性质。

## 配置

### 模型架构
- **Backbone**: enhanced_graph
- **DeePAW Features**: 启用
- **Fusion Method**: add (加法融合)
- **Hidden Dim**: 256
- **Cutoff**: 8.0 Å
- **Max Neighbors**: 48
- **RBF Basis**: 128

### 训练参数
- **Epochs**: 50
- **Batch Size**: 64
- **Learning Rate**: 2e-4
- **Weight Decay**: 1e-5
- **Gradient Clip**: 1.0
- **Warmup Epochs**: 5
- **AMP**: 禁��

### 任务配置
- **Stage**: A (8 个核心任务)
- **Excluded Tasks**: volume, density, is_stable
- **Enabled Tasks**:
  - Thermodynamic: energy_per_atom, formation_energy_per_atom, energy_above_hull
  - Electronic: band_gap, cbm, vbm, efermi, is_metal

## 运行方法

```bash
cd /scratch/sutianhao/data/mp-data-pipeline
bash experiments/stage_a/phase2_deepaw/exp201_deepaw_add/train.sh
```

## 预期结果

### 电子性质改进
| 任务 | Baseline (EXP-01) | 目标 | 改进 |
|------|-------------------|------|------|
| band_gap | 0.715 eV | <0.60 eV | >16% |
| cbm | 0.292 eV | <0.23 eV | >21% |
| vbm | 0.235 eV | <0.19 eV | >19% |
| efermi | 0.383 eV | <0.31 eV | >19% |
| is_metal | 0.9098 AUROC | >0.92 | >1% |

### 训练时间
- 预计: 40-50 分钟
- 相比 baseline: +20-30%

## 与 Baseline 对比

### EXP-01 (Composition Baseline)
- Backbone: composition (无图结构)
- 赢得 9/11 任务 (82%)
- 电子性质表现优秀

### EXP-02 (Graph Baseline)
- Backbone: graph (SchNet-style)
- 仅赢得 2/11 任务 (18%)
- 电子性质表现较差

### EXP-201 (DeePAW + Enhanced Graph)
- 结合图结构和预训练特征
- 预期在电子性质上显著超越 baseline
- 保持结构性质的优势

## 评估

训练完成后运行评估：

```bash
python scripts/eval_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --checkpoint artifacts/runs_exp201/checkpoints/best.pt
```

## 输出文件

- **Checkpoint**: `artifacts/runs_exp201/checkpoints/best.pt`
- **Config**: `artifacts/runs_exp201/config.json`
- **Metrics**: `artifacts/runs_exp201/metrics.json`
- **Logs**: `artifacts/runs_exp201/train.log`

## 注意事项

1. **GPU 内存**: 约需 6-8GB
2. **训练稳定性**: 使用 `--no-amp` 避免混合精度问题
3. **梯度裁剪**: `--grad-clip 1.0` 防止梯度爆炸
4. **学习率预热**: `--warmup-epochs 5` 稳定初期训练

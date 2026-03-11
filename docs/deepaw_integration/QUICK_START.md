# DeePAW 集成快速开始指南

## 5 分钟快速上手

### 前置条件

确保已安装依赖并下载了 DeePAW checkpoint：

```bash
# 检查 DeePAW checkpoint 是否存在
ls /home/sutianhao/data/deepaw_test/DeePAW-main/checkpoints/f_nonlocal_escn_best.pth

# 检查 MP 数据集
ls /scratch/sutianhao/data/mp-data-pipeline/data/db/mp_materials.db
```

---

## 方式 1: 使用提供的脚本 (推荐)

### 步骤 1: 进入项目目录

```bash
cd /scratch/sutianhao/data/mp-data-pipeline
```

### 步骤 2: 运行训练脚本

```bash
bash experiments/stage_a/phase2_deepaw/train_deepaw.sh
```

就这么简单！脚本会自动：
- 使用 IID split (seed=42)
- 启用 DeePAW 特征
- 训练 50 epochs
- 保存最佳模型到 `artifacts/runs/<timestamp>/`

### 步骤 3: 查看结果

训练完成后，检查输出目录：

```bash
# 查看最新的运行
ls -lt artifacts/runs/ | head -5

# 查看训练日志
cat artifacts/runs/<run_id>/train.log

# 查看最佳指标
cat artifacts/runs/<run_id>/metrics.json
```

---

## 方式 2: 手动命令 (自定义配置)

### 基础训练

```bash
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone enhanced_graph \
  --use-deepaw-features \
  --epochs 50
```

### 完整配置

```bash
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone enhanced_graph \
  --cutoff 8.0 \
  --max-neighbors 48 \
  --n-rbf 128 \
  --use-deepaw-features \
  --deepaw-checkpoint /home/sutianhao/data/deepaw_test/DeePAW-main/checkpoints/f_nonlocal_escn_best.pth \
  --deepaw-fusion add \
  --batch-size 64 \
  --num-workers 8 \
  --exclude-tasks volume density is_stable \
  --epochs 50 \
  --lr 2e-4 \
  --no-amp
```

---

## 方式 3: 快速测试 (1 epoch)

验证集成是否正常工作：

```bash
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone enhanced_graph \
  --use-deepaw-features \
  --epochs 1 \
  --batch-size 8
```

预期输出：
```
Using PyG InMemoryDataset (all data in memory)
Extracting masks from PyG dataset...
✅ Extracted masks for 123902 samples
...
[Epoch 1 Iter 1548/15488] Current avg loss: 9.6406
...
[Epoch 1 Iter 10836/15488] Current avg loss: 8.4146
```

如果看到 loss 正常下降，说明集成成功！

---

## 评估模型

训练完成后评估性能：

```bash
python scripts/eval_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --checkpoint artifacts/runs/<run_id>/checkpoints/best.pt
```

---

## 常见参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--split` | 数据划分文件 | `data/splits/split_iid_seed42.json` |
| `--stage` | 训练阶段 (a/b/c) | `a` |
| `--backbone` | 骨干网络 | `enhanced_graph` |

### DeePAW 相关

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use-deepaw-features` | 启用 DeePAW | False |
| `--deepaw-checkpoint` | 模型路径 | f_nonlocal_escn_best.pth |
| `--deepaw-fusion` | 融合方式 | `add` |

### 训练配置

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--epochs` | 训练轮数 | 50 |
| `--batch-size` | 批大小 | 64 |
| `--lr` | 学习率 | 2e-4 |
| `--num-workers` | 数据加载线程 | 8 |

### 图配置

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--cutoff` | 截断半径 (Å) | 8.0 |
| `--max-neighbors` | 最大邻居数 | 48 |
| `--n-rbf` | RBF 基函数数量 | 128 |

---

## 对比实验

### Baseline (无 DeePAW)

```bash
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone enhanced_graph \
  --epochs 50
```

### + DeePAW

```bash
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone enhanced_graph \
  --use-deepaw-features \
  --epochs 50
```

### 对比结果

运行两个实验后，使用评估脚本对比：

```bash
# Baseline
python scripts/eval_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --checkpoint artifacts/runs/<baseline_run>/checkpoints/best.pt

# + DeePAW
python scripts/eval_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --checkpoint artifacts/runs/<deepaw_run>/checkpoints/best.pt
```

---

## 预期性能

### 电子性质改进

| 任务 | Baseline | + DeePAW | 改进 |
|------|----------|----------|------|
| band_gap | 0.231 eV | <0.18 eV | >20% |
| cbm | 0.292 eV | <0.23 eV | >21% |
| vbm | 0.235 eV | <0.19 eV | >19% |
| efermi | 0.383 eV | <0.31 eV | >19% |

### 训练时间

- Baseline: ~33 分钟 (50 epochs)
- + DeePAW: ~40-50 分钟 (50 epochs)
- 增加: +20-30%

---

## 故障排除

### 问题 1: CUDA 内存不足

**错误**：
```
RuntimeError: CUDA out of memory
```

**解决**：
```bash
# 减小 batch size
--batch-size 32  # 或 16
```

### 问题 2: 训练速度慢

**现象**：
- 训练速度 <10 it/s

**解决**：
```bash
# 增加数据加载线程
--num-workers 8

# 预计算图缓存
python scripts/precompute_graphs.py
```

### 问题 3: 找不到 checkpoint

**错误**：
```
FileNotFoundError: f_nonlocal_escn_best.pth
```

**解决**：
```bash
# 检查路径
ls /home/sutianhao/data/deepaw_test/DeePAW-main/checkpoints/

# 或手动指定
--deepaw-checkpoint /path/to/your/checkpoint.pth
```

---

## 下一步

1. **运行完整训练**：使用 50 epochs
2. **评估性能**：对比 baseline
3. **分析结果**：查看哪些任务改进最大
4. **调优参数**：尝试不同的融合策略

---

## 获取帮助

- 查看完整文档: `docs/deepaw_integration/README.md`
- 实现细节: `docs/deepaw_integration/IMPLEMENTATION_NOTES.md`
- 实验配置: `experiments/stage_a/phase2_deepaw/README.md`

# DeePAW 系统训练计划总结

## 概述

已为 DeePAW 集成模型创建完整的系统训练计划，包含 6 个核心实验，覆盖不同融合策略、训练配置和任务范围。

## 实验列表

### 第一优先级（必须执行）

#### EXP-201: DeePAW Add Fusion
- **目录**: `experiments/stage_a/phase2_deepaw/exp201_deepaw_add/`
- **配置**: Add 融合，50 epochs，lr=2e-4
- **目标**: 验证 DeePAW 基础效果
- **预计时间**: 40-50 分钟

#### EXP-202: DeePAW Concat Fusion
- **目录**: `experiments/stage_a/phase2_deepaw/exp202_deepaw_concat/`
- **配置**: Concat 融合，50 epochs，lr=2e-4
- **目标**: 对比融合策略
- **预计时间**: 45-55 分钟

### 第二优先级（推荐执行）

#### EXP-203: DeePAW + Angles
- **目录**: `experiments/stage_a/phase2_deepaw/exp203_deepaw_angles/`
- **配置**: Add 融合 + 角度特征，50 epochs
- **目标**: 验证特征协同效果
- **预计时间**: 50-60 分钟

#### EXP-206: DeePAW Stage B
- **目录**: `experiments/stage_b/phase2_deepaw/exp206_deepaw_stageb/`
- **配置**: Stage B (16 tasks)，100 epochs
- **目标**: 完整任务训练
- **预计时间**: 80-100 分钟

### 第三优先级（可选执行）

#### EXP-204: Long Training
- **目录**: `experiments/stage_a/phase2_deepaw/exp204_deepaw_long/`
- **配置**: 100 epochs
- **目标**: 验证收敛性
- **预计时间**: 80-100 分钟

#### EXP-205: Lower Learning Rate
- **目录**: `experiments/stage_a/phase2_deepaw/exp205_deepaw_lr1e4/`
- **配置**: lr=1e-4，50 epochs
- **目标**: 稳定性验证
- **预计时间**: 40-50 分钟

## 快速开始

### 运行单个实验

```bash
cd /scratch/sutianhao/data/mp-data-pipeline

# 运行 EXP-201 (推荐首先运行)
bash experiments/stage_a/phase2_deepaw/exp201_deepaw_add/train.sh
```

### 批量运行所有实验

```bash
cd /scratch/sutianhao/data/mp-data-pipeline

# 按优先级顺序运行所有 6 个实验
bash experiments/stage_a/phase2_deepaw/run_all_experiments.sh
```

**注意**: 批量运行预计需要约 9 小时完成。

### 对比结果

```bash
cd /scratch/sutianhao/data/mp-data-pipeline

# 生成对比报告
python experiments/stage_a/phase2_deepaw/compare_results.py
```

## 预期改进目标

| 任务 | Baseline (EXP-01) | 目标 (DeePAW) | 改进 |
|------|-------------------|---------------|------|
| band_gap | 0.715 eV | <0.60 eV | >16% |
| cbm | 0.292 eV | <0.23 eV | >21% |
| vbm | 0.235 eV | <0.19 eV | >19% |
| efermi | 0.383 eV | <0.31 eV | >19% |
| is_metal | 0.9098 AUROC | >0.92 | >1% |

## 文件结构

```
experiments/stage_a/phase2_deepaw/
├── exp201_deepaw_add/
│   ├── train.sh          # 训练脚本
│   └── README.md         # 实验说明
├── exp202_deepaw_concat/
│   ├── train.sh
│   └── README.md
├── exp203_deepaw_angles/
│   ├── train.sh
│   └── README.md
├── exp204_deepaw_long/
│   ├── train.sh
│   └── README.md
├── exp205_deepaw_lr1e4/
│   ├── train.sh
│   └── README.md
├── run_all_experiments.sh  # 批量运行脚本
├── compare_results.py      # 结果对比脚本
└── TRAINING_PLAN.md        # 本文档

experiments/stage_b/phase2_deepaw/
└── exp206_deepaw_stageb/
    ├── train.sh
    └── README.md
```

## 输出目录

训练结果将保存到：

```
artifacts/
├── runs_exp201/          # EXP-201 结果
│   ├── checkpoints/
│   │   └── best.pt
│   ├── config.json
│   ├── metrics.json
│   └── train.log
├── runs_exp202/          # EXP-202 结果
├── runs_exp203/          # EXP-203 结果
├── runs_exp204/          # EXP-204 结果
├── runs_exp205/          # EXP-205 结果
└── runs_exp206/          # EXP-206 结果
```

## 评估

每个实验完成后，可以运行评估脚本：

```bash
python scripts/eval_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --checkpoint artifacts/runs_expXXX/checkpoints/best.pt
```

## 注意事项

### 资源需求
- **GPU 内存**: 每个实验约 6-8GB
- **训练时间**: 单个实验 40-100 分钟
- **总时间**: 全部实验约 9 小时

### 训练稳定性
- 使用 `--no-amp` 避免混合精度问题
- 使用 `--grad-clip 1.0` 防止梯度爆炸
- 使用 `--warmup-epochs 5` 稳定初期训练

### 建议执行顺序
1. 先运行 EXP-201 (Add Fusion) 验证基础效果
2. 如果效果良好，运行 EXP-202 (Concat Fusion) 对比
3. 根据前两个实验结果决定是否运行其他实验

## 与历史 Baseline 对比

### EXP-01 (Composition Baseline)
- 赢得 9/11 任务 (82%)
- 电子性质表现优秀
- 无图结构信息

### EXP-02 (Graph Baseline)
- 仅赢得 2/11 任务 (18%)
- 电子性质表现较差
- 有图结构但无预训练

### EXP-201+ (DeePAW + Enhanced Graph)
- 结合图结构和预训练特征
- 预期在电子性质上显著超越 baseline
- 保持结构性质的优势

## 下一步

1. **运行第一优先级实验** (EXP-201, EXP-202)
2. **分析结果** 使用 `compare_results.py`
3. **根据结果决定** 是否运行其他实验
4. **撰写实验报告** 总结 DeePAW 集成效果

## 参考文档

- **实现文档**: `docs/deepaw_integration/README.md`
- **快速开始**: `docs/deepaw_integration/QUICK_START.md`
- **实现笔记**: `docs/deepaw_integration/IMPLEMENTATION_NOTES.md`
- **训练计划**: `/home/sutianhao/.claude/plans/twinkling-humming-petal.md`

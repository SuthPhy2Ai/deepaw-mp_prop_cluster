# MP Data Pipeline

Materials Project (MP) 数据管道与多任务机器学习训练系统。

## 项目概述

本项目包含两个主要功能：

1. **数据管道**：从 Materials Project API 下载或从预下载的 JSONL.gz 文件加载材料数据，存储到 ASE SQLite 数据库
2. **多任务机器学习训练**：使用图神经网络模型预测晶体性质（热力学、电子、结构、弹性），支持多种数据划分策略

**预下载数据集**：`data/raw/summary_all_merged.jsonl.gz`（280MB 压缩）包含约 155k 材料的完整结构、能量、电子和弹性性质数据。使用此文件可避免 API 调用。

**当前状态**：
- ✅ Phase 1 基线实验完成（见下方实验结果）
- 🚀 Phase 2 增强图骨干网络实现就绪，准备开始实验

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 构建数据库（推荐：使用预下载数据）

```bash
# 从 JSONL.gz 构建 ASE 数据库（快速，无需 API key）
python scripts/load_from_jsonl.py --overwrite

# 校验数据库
python scripts/validate.py
```

### 3. 多任务模型训练

```bash
# 生成数据划分（IID / ChemSys-OOD / Complexity-OOD）
python scripts/export_splits.py

# Stage A 训练（8 个核心任务，不含弹性性质）
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone graph \
  --batch-size 64 \
  --num-workers 8 \
  --exclude-tasks volume density is_stable \
  --epochs 50

# 评估 checkpoint
python scripts/eval_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --checkpoint artifacts/runs/<run_id>/checkpoints/best.pt
```

---

## 目录结构

```text
.
├── src/mp_data_pipeline/              # 核心模块
│   ├── ml/                            # 数据集、划分、任务定义
│   │   ├── dataset.py                 # 原始图数据集
│   │   ├── enhanced_dataset.py        # 增强数据集（角度特征）
│   │   ├── splits.py                  # 数据划分逻辑
│   │   └── tasks.py                   # 任务定义和分组
│   ├── models/                        # 骨干网络、头部、多任务模型
│   │   ├── backbones.py               # 原始图和组成骨干网络
│   │   ├── enhanced_backbones.py      # 增强图骨干网络（Phase 2）
│   │   ├── graph_features.py          # 角度计算和特征扩展
│   │   ├── heads.py                   # 任务特定预测头
│   │   └── multitask_model.py         # 主多任务模型
│   ├── training/                      # 训练器、损失、采样器
│   ├── fetch_mp_data.py               # API 下载逻辑
│   ├── load_from_jsonl.py             # JSONL.gz 加载器
│   ├── store_to_ase.py                # Checkpoint 到 ASE 转换器
│   └── validate.py                    # 数据库验证
│
├── scripts/                           # CLI 入口点
│   ├── load_from_jsonl.py             # 推荐：加载预下载数据
│   ├── export_splits.py               # 生成 train/val/test 划分
│   ├── train_multitask.py             # 训练多任务模型
│   ├── eval_multitask.py              # 评估 checkpoints
│   ├── analyze_best_model.py          # 模型分析工具
│   ├── experiment_manager.py          # 实验管理工具
│   └── legacy/                        # 历史脚本
│
├── experiments/                       # 实验组织（新结构）
│   ├── stage_a/                       # Stage A: 8 核心任务
│   │   ├── phase1_baseline/
│   │   │   └── exp001_baseline_graph/ # Phase 1 基线模型
│   │   ├── phase2_enhancements/       # Phase 2 增强模型（计划中）
│   │   └── summary.md
│   ├── stage_b/                       # Stage B: 18 任务（含弹性）
│   │   ├── phase3_baseline/
│   │   ├── phase3_enhancements/
│   │   └── summary.md
│   ├── comparison/                    # 跨阶段对比
│   ├── EXPERIMENTS.md                 # 实验追踪表
│   └── README.md
│
├── data/
│   ├── raw/                           # summary_all_merged.jsonl.gz (280MB)
│   ├── db/                            # mp_materials.db (ASE SQLite)
│   ├── splits/                        # Train/val/test 划分 JSON 文件
│   └── checkpoints/                   # API 下载 checkpoints
│
├── artifacts/runs/                    # 训练输出（config, checkpoints, metrics）
├── reports/                           # 分析报告和可视化
│   ├── PHASE1_COMPLETE_REPORT.md      # Phase 1 基线对比
│   ├── PHASE1_FINAL_SUMMARY.md        # Phase 1 最终总结
│   ├── PHASE2_TRAINING_PLAN.md        # Phase 2 详细计划
│   ├── figures/                       # 可视化图表
│   └── plans/                         # 研究计划和实验日志
│
├── configs/                           # 实验配置文件
│   ├── exp002_regularization.json
│   ├── exp003_enhanced_graph.json
│   ├── exp004_angle_features.json
│   └── exp005_full_stack.json
│
├── logs/                              # 应用日志
├── docs/project_status/PHASE1_PHASE2_CHECKLIST.md  # Phase 1→2 过渡清单
├── docs/guides/PHASE2_QUICKSTART.md                # Phase 2 快速启动指南
├── CLAUDE.md                          # Claude Code 项目指南
├── README.md                          # 本文件
└── requirements.txt
```

---

## 实验结果

### Stage A: 8 核心任务（不含弹性性质）

#### Phase 1: 基线模型（exp001_baseline_graph）

**状态**: ✅ 完成
**最佳 Epoch**: 38
**验证损失**: 0.2226
**测试损失**: 0.2223
**训练时间**: 50 epochs (~2 天)

**任务列表**（8 个）：
- 热力学（3）：energy_per_atom, formation_energy_per_atom, energy_above_hull
- 电子（5）：band_gap, cbm, vbm, efermi, is_metal

**关键指标（验证集）**：

| 任务 | MAE / AUROC | R² | 状态 |
|------|-------------|-----|------|
| formation_energy_per_atom | 0.0800 eV | 0.988 | ⭐ 优秀 |
| energy_above_hull | 0.0644 eV | 0.924 | ⭐ 优秀 |
| is_metal | AUROC 0.9575 | - | ⭐ 优秀 |
| band_gap | 0.2308 eV | 0.916 | ✅ 良好 |
| efermi | 0.3834 eV | - | ⚠️ 需改进 |
| energy_per_atom | 0.3606 eV | 0.997 | ⚠️ 需改进 |
| cbm | 0.2921 eV | - | ⚠️ 需改进 |
| vbm | 0.2346 eV | - | ⚠️ 需改进 |

**过拟合分析**：
- 整体 train-val gap: 53% ⚠️
- energy_per_atom gap: 95% ⚠️
- cbm gap: 57% ⚠️
- band_gap gap: 61% ⚠️

**主要发现**：
1. ✅ 热力学性质预测表现优秀（formation energy R²=0.988）
2. ✅ 金属分类接近完美（AUROC=0.9575）
3. ⚠️ 电子性质存在显著过拟合
4. ⚠️ 需要更强的正则化和更好的图架构

**文件位置**：`experiments/stage_a/phase1_baseline/exp001_baseline_graph/`

---

### Phase 2: 增强模型（计划中）

**目标**：解决 Phase 1 发现的过拟合和电子性质预测问题

#### 计划实验

| ID | 名称 | 目标 | 关键改动 | 预期改进 |
|----|------|------|----------|----------|
| exp002 | regularization | 减少过拟合 | dropout=0.1, weight_decay=1e-4, lr=5e-5 | Train-val gap < 5% |
| exp003 | enhanced_graph | 更好的图表示 | cutoff=8.0Å, neighbors=48, rbf=128 | efermi MAE < 0.35 eV |
| exp004 | angle_features | 三体相互作用 | 基于 exp003 + 角度特征 | band_gap MAE < 0.20 eV |
| exp005 | full_stack | 生产就绪 | 组合所有增强 | Val loss < 0.19 |

**配置文件**：`configs/exp002-005_*.json`
**详细计划**：`reports/PHASE2_TRAINING_PLAN.md`
**快速启动**：`docs/guides/PHASE2_QUICKSTART.md`

---

### Stage B: 18 任务（含弹性性质）

**状态**: 📋 计划中（Phase 3）

**额外任务**（10 个）：
- 结构（2）：volume, density
- 稳定性（1）：is_stable
- 弹性/力学（7）：bulk_modulus_vrh, shear_modulus_vrh, youngs_modulus, homogeneous_poisson, universal_anisotropy 等

**数据覆盖挑战**：
- 总材料数：~155k
- 含弹性数据：~11k（7.1%）
- 解决方案：使用 `--oversample-elastic 4.0` 加权采样

**前置条件**：
1. ✅ 完成 Stage A Phase 2 实验
2. ✅ 确定最佳 Stage A 架构
3. ✅ 验证加权采样策略

---

## 数据库使用

### 从 ASE DB 读取数据

主数据库路径：`data/db/mp_materials.db`

#### 基本查询

```python
from ase.db import connect

db = connect("data/db/mp_materials.db")
print("总材料数:", db.count())

# 条件查询（注意：使用 formula_pretty，不是 formula）
for row in db.select('band_gap>2.0, is_stable=True', limit=10):
    print(row.mp_id, row.formula_pretty, row.band_gap)

# 按 mp_id 精确查询
row = db.get(mp_id='mp-149')
atoms = row.toatoms()
```

#### 复杂字段（row.data）

```python
row = db.get(mp_id='mp-149')
print(row.data.get('elastic_tensor_voigt'))  # 6×6 弹性张量
print(row.data.get('bulk_modulus'))          # dict with vrh/voigt/reuss
```

### 数据库 Schema

**Key-Value Pairs（可搜索）**：
- `mp_id`, `formula_pretty`, `crystal_system`, `point_group`（前缀 `pg_`）
- `spacegroup`, `nsites`, `is_stable`, `is_metal`
- `energy_per_atom`, `formation_energy_per_atom`, `energy_above_hull`
- `band_gap`, `cbm`, `vbm`, `efermi`
- `volume`, `density`
- `bulk_modulus_vrh`, `shear_modulus_vrh`, `youngs_modulus`, `homogeneous_poisson`, `universal_anisotropy`

**Data Dict（非可搜索）**：
- `elastic_tensor_voigt`（6×6 列表）
- `compliance_tensor_voigt`（6×6 列表）
- `bulk_modulus`, `shear_modulus`（dict with vrh/voigt/reuss）

---

## 数据集统计

预下载数据集（`data/raw/summary_all_merged.jsonl.gz`）：
- **154,879 材料** 总计
- **10,994 材料（7.1%）** 含弹性数据
- **33,973 材料（21.9%）** 热力学稳定
- **72,640 材料（46.9%）** 金属

---

## 实验管理

### 使用实验管理工具

```bash
# 列出所有实验
python scripts/experiment_manager.py list

# 创建新实验
python scripts/experiment_manager.py create exp002 regularization

# 运行分析
python scripts/experiment_manager.py analyze exp002 regularization
```

### 实验命名约定

- **Stage A**（8 任务）：exp001-099
- **Stage B**（18 任务）：exp101-199

---

## 重要说明

### 数据管道
- **推荐路径**：使用 `load_from_jsonl.py` 加载预下载数据（无需 API key）
- **Schema 注意事项**：
  - 使用 `formula_pretty` 字段，不是 `formula`
  - Point groups 存储为 `pg_-1`, `pg_1` 等，避免 ASE 字符串/整数歧义
  - 弹性性质仅约 7% 材料可用
- **Young's Modulus**：从 bulk/shear moduli 计算：E = 9KG/(3K+G)

### 多任务训练
- **Stage A vs B**：Stage A 排除弹性任务（高覆盖），Stage B 包含所有任务
- **加权采样**：Stage B 使用 `WeightedRandomSampler` 过采样弹性数据（默认 4×）
- **任务权重**：基于训练集数据覆盖自动计算
- **Band Gap 约束**：模型通过 softplus 激活强制非负 band gap
- **骨干网络选项**：
  - `graph`：原始 SchNet 风格消息传递（6.0Å cutoff, 24 neighbors, 64 RBF）
  - `composition`：仅元素嵌入（无结构）
  - `enhanced_graph`：增强架构，可配置特征（Phase 2）

### Stage A 训练关键配置

**必须排除的任务**：`--exclude-tasks volume density is_stable`

原因：
- `volume` 值范围 5.61 到 10,887.91（过大，导致梯度爆炸和 NaN loss）
- `density`, `is_stable` 为稳定性排除

---

## API 下载流程（可选）

仅在需要从 API 下载时使用（预下载数据已包含所有内容）：

```bash
# 1) 设置 API key
export MP_API_KEY="your_materials_project_api_key"

# 2) 下载数据（支持断点续传）
python scripts/fetch_mp_data.py

# 3) 从 checkpoint 写入 ASE 数据库
python scripts/store_to_ase.py
```

---

## 配置

所有配置在 `src/mp_data_pipeline/config.py`：
- `MP_API_KEY`：从环境变量读取（仅 API 下载需要）
- `CHUNK_SIZE`：每次 API 请求的材料数（默认：500）
- `MAX_RETRIES`：失败 API 调用的重试次数（默认：3）
- `DB_PATH`：输出数据库（`data/db/mp_materials.db`）
- `RAW_JSONL_PATH`：预下载数据（`data/raw/summary_all_merged.jsonl.gz`）
- `CHECKPOINT_DIR`：API 下载 checkpoints（`data/checkpoints/`）

---

## 文档索引

- **实验追踪**：`experiments/EXPERIMENTS.md`
- **Stage A 总结**：`experiments/stage_a/summary.md`
- **Stage B 总结**：`experiments/stage_b/summary.md`
- **Phase 1 完整报告**：`reports/PHASE1_COMPLETE_REPORT.md`
- **Phase 1 最终总结**：`reports/PHASE1_FINAL_SUMMARY.md`
- **Phase 2 训练计划**：`reports/PHASE2_TRAINING_PLAN.md`
- **Phase 2 快速启动**：`docs/guides/PHASE2_QUICKSTART.md`
- **Phase 1→2 清单**：`docs/project_status/PHASE1_PHASE2_CHECKLIST.md`
- **Claude Code 指南**：`CLAUDE.md`

---

## 下一步

1. ✅ Phase 1 基线完成
2. ✅ 实验结构重组完成
3. 🚀 启动 Phase 2 实验（exp002-005）
4. 📋 Phase 3 Stage B 训练（未来）

**立即开始 Phase 2**：参见 `docs/guides/PHASE2_QUICKSTART.md`

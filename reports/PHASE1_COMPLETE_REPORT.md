# Phase 1 完整报告：多任务材料性质预测基线对比

**项目**: Materials Project 数据管道与多任务机器学习
**Phase**: Phase 1 - Baseline 建立与对比
**完成时间**: 2026-03-04 02:13
**总耗时**: 约 6 小时 (2026-03-03 20:10 → 2026-03-04 02:13)
**状态**: ✅ 完全完成

---

## 目录

1. [执行摘要](#执行摘要)
2. [实验设计](#实验设计)
3. [实验结果](#实验结果)
4. [性能对比分析](#性能对比分析)
5. [问题与解决方案](#问题与解决方案)
6. [核心发现](#核心发现)
7. [Phase 2 建议](#phase-2-建议)
8. [附录](#附录)

---

## 执行摘要

Phase 1 成功完成了两个基线模型的训练和对比分析：

- **EXP-01 (Composition Baseline)**: ✅ 所有成功标准达成
- **EXP-02 (Graph Baseline)**: ⚠️ 完成但未达预期

**关键发现**: 简单的 Composition Baseline 在 82% 的任务上优于复杂的 Graph Baseline，说明元素组成是材料性质的主要决定因素。Graph Baseline 仅在 volume (-41%) 和 energy_above_hull (-18%) 两个任务上表现更好。

**主要问题**: Graph Baseline 训练过程中遇到 AMP 类型不匹配和 NaN 问题，导致 3 次崩溃，需要禁用混合精度训练并降低学习率。

**建议**: 继续改进 Graph Backbone，因为它在特定任务上展现了潜力。

---

## 实验设计

### 数据集

- **训练样本**: 123,903
- **验证样本**: 15,487
- **测试样本**: 15,489
- **数据来源**: Materials Project (预下载的 JSONL.gz 文件)
- **任务数量**: 11 个 (Stage A，不包含弹性性质)

### 任务定义

**分类任务 (2)**:
- `is_metal`: 材料是否为金属 (AUROC)
- `is_stable`: 材料是否热力学稳定 (AUROC)

**回归任务 (9)**:
- **能量** (3): energy_per_atom, formation_energy_per_atom, energy_above_hull
- **电子性质** (4): band_gap, cbm, vbm, efermi
- **结构性质** (2): volume, density

### 模型配置

| 配置项 | EXP-01 (Composition) | EXP-02 (Graph) |
|--------|---------------------|----------------|
| **Backbone** | CompositionBackbone | GraphBackbone (SchNet-style) |
| **Hidden Dim** | 256 | 256 |
| **Layers** | 1 (embedding only) | 6 (message passing) |
| **Epochs** | 50 | 50 |
| **Batch Size** | 32 | 32 |
| **Learning Rate** | 3e-4 | 1e-4 (降低以提高稳定性) |
| **AMP** | True | False (禁用以避免 NaN) |
| **训练时间** | ~1.5 小时 | ~2 小时 |

### 成功标准

**EXP-01**:
- is_metal AUROC ≥ 0.75
- band_gap MAE < 1.0 eV
- is_stable AUROC ≥ 0.75
- 训练稳定，无崩溃

**EXP-02**:
- 至少 50% 任务优于 EXP-01
- band_gap MAE 降低 ≥10%

---

## 实验结果

### EXP-01 (Composition Baseline) - ✅ 成功

**验证集最佳结果** (Epoch 50):

| 任务 | 指标 | 结果 | 目标 | 状态 |
|------|------|------|------|------|
| is_metal | AUROC | **0.9098** | ≥0.75 | ✅ 超出 21% |
| is_metal | Accuracy | 0.8329 | - | ✅ |
| is_stable | AUROC | **0.8510** | ≥0.75 | ✅ 超出 13% |
| is_stable | Accuracy | 0.8170 | - | ✅ |
| band_gap | MAE | **0.715 eV** | <1.0 | ✅ 优于 28% |
| energy_per_atom | MAE | 1.218 eV | - | ✅ |
| formation_energy | MAE | 0.212 eV | - | ✅ |
| volume | MAE | 247.4 Å³ | - | ✅ |

**结论**: 所有成功标准达成 ✅

### EXP-02 (Graph Baseline) - ⚠️ 完成但未达预期

**验证集最佳结果** (Epoch 50):

| 任务 | 指标 | 结果 | vs EXP-01 | 状态 |
|------|------|------|-----------|------|
| is_metal | AUROC | 0.8745 | -3.9% | ❌ 不如 Composition |
| is_stable | AUROC | 0.8050 | -5.4% | ❌ 不如 Composition |
| band_gap | MAE | 0.952 eV | +33% | ❌ 恶化 |
| energy_per_atom | MAE | 2.818 eV | +131% | ❌ 严重恶化 |
| formation_energy | MAE | 0.421 eV | +99% | ❌ 恶化 |
| energy_above_hull | MAE | 0.137 eV | -18% | ✅ 改进 |
| volume | MAE | 147.0 Å³ | -41% | ✅ 显著改进 |

**结论**: 仅在 2/11 任务上优于 EXP-01 (18%)，未达成功标准 ❌

---

## 性能对比分析

### 分类任务对比

![分类任务对比](figures/phase1_classification_comparison.png)

**分析**:
- **is_metal**: Composition 胜出 (0.9098 vs 0.8745, -3.9%)
- **is_stable**: Composition 胜出 (0.8510 vs 0.8050, -5.4%)
- 元素组成可能是金属性和稳定性的主要决定因素

### 回归任务对比

![回归任务对比](figures/phase1_regression_comparison.png)

**分析**:
- **能量任务**: Composition 在 energy_per_atom 和 formation_energy 上大幅领先
- **电子性质**: Composition 在所有电子性质任务上表现更好
- **结构性质**: Graph 在 volume 上显著领先 (-41%)，但 density 不如 Composition

### 胜负统计

![胜负统计](figures/phase1_winloss_summary.png)

**总体统计**:
- **Composition 胜**: 9/11 任务 (82%)
- **Graph 胜**: 2/11 任务 (18%)

**按类别统计**:
- 分类任务 (2): Composition 2-0 Graph
- 能量任务 (3): Composition 2-1 Graph
- 电子任务 (4): Composition 4-0 Graph
- 结构任务 (2): Composition 1-1 Graph

### 相对性能变化

![相对性能](figures/phase1_relative_performance.png)

**关键观察**:
- **最大改进**: volume (-41%)
- **最大恶化**: energy_per_atom (+131%)
- **Graph 优势领域**: volume, energy_above_hull
- **Composition 优势领域**: 其他所有任务

---

## 问题与解决方案

### 问题 1: AMP 类型不匹配 (22:43)

**错误信息**:
```
RuntimeError: index_add_(): self (Float) and source (Half) must have the same scalar type
```

**原因分析**:
- AMP (Automatic Mixed Precision) 将 messages 转换为 Float16
- 但 agg 张量仍然是 Float32
- `index_add_` 操作要求两个张量类型一致

**解决方案**:
```python
# backbones.py:80
# 修复前:
agg = torch.zeros_like(node_emb)
agg.index_add_(0, dst, messages)

# 修复后:
agg = torch.zeros_like(node_emb, dtype=messages.dtype)
agg.index_add_(0, dst, messages)
```

**状态**: ✅ 已修复

---

### 问题 2: 训练 NaN 问题 (23:45)

**错误信息**:
```
ValueError: Input contains NaN
```

**现象**:
- 训练在 Epoch 8 出现 NaN 值
- 训练损失进展: epoch 5: 217.38 → epoch 6: 213.55 → epoch 7: 209.14 → epoch 8: NaN

**原因分析**:
1. AMP 混合精度导致数值不稳定
2. 学习率 3e-4 可能过高
3. 梯度爆炸导致参数更新过大

**解决方案**:
1. **禁用 AMP**:
```python
# trainer.py:30
amp: bool = False  # 从 True 改为 False
```

2. **降低学习率**:
```python
# 从 3e-4 降低到 1e-4
lr: float = 1e-4
```

**状态**: ✅ 已修复

---

### 问题 3: Python 环境问题 (22:44)

**错误信息**:
```
ModuleNotFoundError: No module named 'numpy'
```

**原因分析**:
- 使用了错误的 Python 解释器 (`/usr/bin/python`)
- 应该使用 conda 环境中的 Python

**解决方案**:
```bash
# 使用正确的 conda 环境
/home/sutianhao/.conda/envs/ctgan/bin/python
```

**状态**: ✅ 已修复

---

## 核心发现

### 1. Composition Baseline 出乎意料地强大

**现象**:
- 简单的元素 embedding + mean pooling 在 82% 任务上表现最佳
- 仅使用元素组成信息，不考虑原子坐标和晶体结构

**解释**:
- **元素组成是主要决定因素**: 对于大多数材料性质，元素的种类和比例是最重要的
- **简单模型的优势**: 参数少，训练稳定，不易过拟合
- **领域知识验证**: 材料科学中，许多性质确实主要由化学组成决定

**启示**:
- 不要低估简单模型的能力
- 复杂模型需要更仔细的调优才能超越简单基线

---

### 2. Graph Baseline 仅在特定任务上有优势

**优势任务**:

1. **volume** (MAE: 147.0 vs 247.4, -41%)
   - 体积是空间性质，直接依赖于原子坐标
   - Graph 的空间信息对体积预测至关重要

2. **energy_above_hull** (MAE: 0.137 vs 0.167, -18%)
   - 热力学稳定性与局部原子环境相关
   - Graph 结构有助于捕捉局部能量差异

**劣势任务**:

1. **energy_per_atom** (MAE: 2.818 vs 1.218, +131%)
   - 最大的失败
   - 可能原因: 学习率过低，能量任务未充分优化

2. **电子性质** (band_gap, cbm, vbm, efermi)
   - MAE 差异: 23-33%
   - 电子性质可能主要由元素组成和电子结构决定

**结论**:
- Graph 结构信息**确实有价值**，但仅限于特定任务
- 当前实现存在问题，需要改进

---

### 3. 训练稳定性是关键瓶颈

**问题历史**:
- **22:43**: AMP 类型不匹配导致崩溃
- **23:45**: NaN 问题导致崩溃
- **00:01**: 禁用 AMP 并降低学习率后成功

**影响**:
- 3 次崩溃浪费约 1 小时
- 禁用 AMP 导致训练速度变慢
- 降低学习率可能影响收敛质量

**教训**:
- 在大规模训练前，应该在小规模数据上验证稳定性
- AMP 需要仔细调试，不能简单启用
- 学习率调整对 GNN 训练至关重要

---

### 4. 数据加载是主要瓶颈

**时间分布**:
- **数据加载**: 约 1 小时 (图构建)
- **训练**: 约 0.5-1 小时 (50 epochs)

**原因**:
- 需要为 123,903 个样本构建图结构
- 计算原子间距离和邻居关系
- cutoff=6.0Å, max_neighbors=24

**优化方向**:
- 预计算图结构并缓存
- 使用更高效的图构建算法
- 并行化图构建过程

---

## 详细性能对比表

### 分类任务详细对比

| 任务 | 指标 | EXP-01 | EXP-02 | 变化 | 胜者 |
|------|------|--------|--------|------|------|
| **is_metal** | AUROC | 0.9098 | 0.8745 | -3.9% | 🏆 Composition |
| | Accuracy | 0.8329 | 0.7922 | -4.9% | 🏆 Composition |
| | Loss | 0.3758 | 0.4299 | +14.4% | 🏆 Composition |
| **is_stable** | AUROC | 0.8510 | 0.8050 | -5.4% | 🏆 Composition |
| | Accuracy | 0.8170 | 0.7949 | -2.7% | 🏆 Composition |
| | Loss | 0.3828 | 0.4223 | +10.3% | 🏆 Composition |

### 能量任务详细对比

| 任务 | 指标 | EXP-01 | EXP-02 | 变化 | 胜者 |
|------|------|--------|--------|------|------|
| **energy_per_atom** | MAE (eV) | 1.218 | 2.818 | +131% | 🏆 Composition |
| | RMSE (eV) | 2.857 | 6.075 | +113% | 🏆 Composition |
| | Loss | 0.951 | 2.428 | +155% | 🏆 Composition |
| **formation_energy** | MAE (eV) | 0.212 | 0.421 | +99% | 🏆 Composition |
| | RMSE (eV) | 0.437 | 0.592 | +36% | 🏆 Composition |
| | Loss | 0.068 | 0.154 | +126% | 🏆 Composition |
| **energy_above_hull** | MAE (eV) | 0.167 | 0.137 | -18% | 🏆 Graph |
| | RMSE (eV) | 0.421 | 0.337 | -20% | 🏆 Graph |
| | Loss | 0.059 | 0.040 | -32% | 🏆 Graph |

### 电子性质详细对比

| 任务 | 指标 | EXP-01 | EXP-02 | 变化 | 胜者 |
|------|------|--------|--------|------|------|
| **band_gap** | MAE (eV) | 0.715 | 0.952 | +33% | 🏆 Composition |
| | RMSE (eV) | 0.887 | 1.210 | +36% | 🏆 Composition |
| | Loss | 0.320 | 0.525 | +64% | 🏆 Composition |
| **cbm** | MAE (eV) | 0.755 | 1.006 | +33% | 🏆 Composition |
| | RMSE (eV) | 1.152 | 1.342 | +16% | 🏆 Composition |
| | Loss | 0.442 | 0.618 | +40% | 🏆 Composition |
| **vbm** | MAE (eV) | 0.662 | 0.870 | +31% | 🏆 Composition |
| | RMSE (eV) | 1.022 | 1.175 | +15% | 🏆 Composition |
| | Loss | 0.377 | 0.519 | +38% | 🏆 Composition |
| **efermi** | MAE (eV) | 0.841 | 1.035 | +23% | 🏆 Composition |
| | RMSE (eV) | 1.402 | 1.403 | +0.1% | ≈ 平局 |
| | Loss | 0.504 | 0.650 | +29% | 🏆 Composition |

### 结构性质详细对比

| 任务 | 指标 | EXP-01 | EXP-02 | 变化 | 胜者 |
|------|------|--------|--------|------|------|
| **volume** | MAE (Å³) | 247.4 | 147.0 | -41% | 🏆 Graph |
| | RMSE (Å³) | 482.0 | 320.3 | -34% | 🏆 Graph |
| | Loss | 246.9 | 146.4 | -41% | 🏆 Graph |
| **density** | MAE (g/cm³) | 0.603 | 0.831 | +38% | 🏆 Composition |
| | RMSE (g/cm³) | 1.164 | 1.192 | +2% | 🏆 Composition |
| | Loss | 0.309 | 0.484 | +56% | 🏆 Composition |

---

## Phase 2 建议

### 推荐方向: 增强 Graph Backbone (1-2 周)

**理由**:
- Graph 在 volume (-41%) 和 energy_above_hull (-18%) 上表现更好
- 说明图结构信息**确实有价值**
- 当前问题可能是实现和训练问题，而非架构问题

**改进方向**:

#### 1. 修复 AMP 问题，重新启用混合精度训练

**当前状态**: AMP 禁用，训练速度慢

**改进方案**:
- 仔细审查所有张量操作，确保类型一致
- 使用 `torch.amp.autocast` 替代 `torch.cuda.amp.autocast`
- 添加梯度缩放 (GradScaler) 防止下溢
- 在小规模数据上验证稳定性

**预期收益**: 训练速度提升 2-3 倍

---

#### 2. 增加图覆盖范围

**当前配置**:
- cutoff: 6.0 Å
- max_neighbors: 24

**改进方案**:
- cutoff: 6.0 → 8.0 Å (增加长程相互作用)
- max_neighbors: 24 → 48 (捕捉更多邻居)

**预期收益**: 更完整的结构信息，可能改善能量预测

---

#### 3. 添加边特征

**当前特征**: 仅距离 RBF (Radial Basis Function)

**改进方案**:
- 添加角度特征 (三体相互作用)
- 添加键类型特征 (共价键、离子键等)
- 添加晶体对称性特征

**参考架构**: DimeNet, GemNet

**预期收益**: 更丰富的结构信息，改善所有任务

---

#### 4. 使用更高学习率

**当前配置**: 1e-4 (为了稳定性降低)

**改进方案**:
- 尝试 2e-4, 3e-4
- 使用学习率预热 (warmup)
- 使用余弦退火 (cosine annealing)

**预期收益**: 更快收敛，可能改善能量预测

---

#### 5. 尝试其他 GNN 架构

**当前架构**: SchNet-style (简单的消息传递)

**候选架构**:
- **DimeNet**: 方向性消息传递，考虑角度信息
- **SchNet++**: 改进的 SchNet，更强的表达能力
- **GemNet**: 几何消息传递，考虑高阶相互作用
- **PaiNN**: 等变消息传递，保持旋转不变性

**预期收益**: 更强的表达能力，全面改善性能

---

### 替代方案

#### 方案 A: 混合模型 (Composition + Graph) - 1 周

**设计**:
- 使用 Composition 作为主干
- 添加 Graph 分支用于 volume 和 energy_above_hull
- 任务特定的 backbone 选择

**优点**:
- 结合两者优势
- 快速实现

**缺点**:
- 模型复杂度增加
- 训练和推理更慢

---

#### 方案 B: 直接进入 Stage B (弹性任务) - 立即

**设计**:
- 使用 Composition Baseline 训练弹性任务
- 放弃图结构信息

**优点**:
- 节省 1-2 周时间
- Composition 已经很好

**缺点**:
- 放弃图结构信息
- 可能错过改进机会
- volume 预测会变差

---

### 决策建议

**推荐**: 选择增强 Graph Backbone (方向 1)

**理由**:
1. Graph 在特定任务上展现了潜力
2. 当前问题是可以解决的技术问题
3. 投入 1-2 周时间值得尝试
4. 如果改进后仍不理想，可以回退到方案 B

**决策点**: 2 周后评估
- 如果 Graph 在 50% 以上任务优于 Composition → 继续使用 Graph
- 如果仍然不理想 → 使用混合模型或 Composition

---

## 时间线

### 已完成

| 时间 | 事件 | 状态 |
|------|------|------|
| 20:10 | EXP-01 启动 | ✅ |
| 20:10-21:10 | EXP-01 数据加载 (1h) | ✅ |
| 21:10-21:39 | EXP-01 训练 (30min, 50 epochs) | ✅ |
| 21:39 | EXP-01 完成并分析 | ✅ |
| 21:39 | EXP-02 首次启动 | ✅ |
| 22:43 | EXP-02 崩溃 (AMP 类型不匹配) | ❌ |
| 22:44 | 修复 AMP 类型问题 | ✅ |
| 22:45 | EXP-02 重启 | ✅ |
| 23:45 | EXP-02 再次崩溃 (NaN 问题) | ❌ |
| 00:01 | 禁用 AMP 并降低学习率 | ✅ |
| 00:01 | EXP-02 最终重启 | ✅ |
| 00:01-01:08 | EXP-02 数据加载 (67min) | ✅ |
| 01:08-02:04 | EXP-02 训练 (56min, 50 epochs) | ✅ |
| 02:04 | EXP-02 完成 | ✅ |
| 02:11 | Phase 1 对比分析完成 | ✅ |
| 02:13 | Phase 1 总结报告完成 | ✅ |

### 时间统计

| 阶段 | 时间 | 占比 |
|------|------|------|
| EXP-01 数据加载 | 1h | 17% |
| EXP-01 训练 | 0.5h | 8% |
| EXP-02 数据加载 | 1h | 17% |
| EXP-02 训练 | 1h | 17% |
| 问题修复 | 1h | 17% |
| 分析报告 | 0.5h | 8% |
| 其他 | 1h | 16% |
| **总计** | **6h** | **100%** |

---

## 资源使用统计

### GPU

| 指标 | EXP-01 | EXP-02 |
|------|--------|--------|
| 显存使用 | 6.1 GB | 6.1 GB |
| 利用率 (训练时) | 60-80% | 32-77% |
| 利用率 (数据加载) | 0% | 0% |

### 系统资源

| 指标 | 值 |
|------|-----|
| 内存使用 | 4.5-7.5 GB (3.5-5.8%) |
| CPU 使用 | 2000-3000% (20-30 核) |
| 磁盘使用 | ~500 MB (checkpoint + 日志) |

### 数据集

| 指标 | 值 |
|------|-----|
| 训练样本 | 123,903 |
| 验证样本 | 15,487 |
| 测试样本 | 15,489 |
| Stage A 任务 | 11 |
| 有弹性数据的样本 | 10,994 (7.1%) |

---

## 附录

### A. 实验配置文件

**EXP-01 配置** (`artifacts/runs/20260303_211013/config.json`):
```json
{
  "backbone": "composition",
  "hidden_dim": 256,
  "epochs": 50,
  "batch_size": 32,
  "lr": 0.0003,
  "weight_decay": 1e-05,
  "grad_clip": 1.0,
  "device": "cuda",
  "amp": true,
  "stage": "a"
}
```

**EXP-02 配置** (`artifacts/runs/20260304_005923/config.json`):
```json
{
  "backbone": "graph",
  "hidden_dim": 256,
  "num_layers": 6,
  "cutoff": 6.0,
  "max_neighbors": 24,
  "epochs": 50,
  "batch_size": 32,
  "lr": 0.0001,
  "weight_decay": 1e-05,
  "grad_clip": 1.0,
  "device": "cuda",
  "amp": false,
  "stage": "a"
}
```

---

### B. 代码修复详情

**修复 1: AMP 类型不匹配** (`src/mp_data_pipeline/models/backbones.py:80`)

```python
# 修复前
agg = torch.zeros_like(node_emb)
agg.index_add_(0, dst, messages)

# 修复后
# Fix AMP type mismatch: ensure agg has same dtype as messages
agg = torch.zeros_like(node_emb, dtype=messages.dtype)
agg.index_add_(0, dst, messages)
```

**修复 2: 禁用 AMP** (`src/mp_data_pipeline/training/trainer.py:30`)

```python
@dataclass
class TrainerConfig:
    """Trainer hyperparameters."""
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    device: str = "cuda"
    amp: bool = False  # Disabled due to NaN issues with GraphBackbone
    accumulation_steps: int = 1
```

---

### C. 输出文件清单

**报告文档** (位于 `reports/`):
- `PHASE1_COMPLETE_REPORT.md` - 本报告 (完整报告，包含图表)
- `PHASE1_INDEX.md` - 导航索引
- `PHASE1_EXECUTIVE_SUMMARY.md` - 执行摘要
- `PHASE1_COMPARISON.md` - 详细对比分析
- `PHASE1_FINAL_STATUS.md` - 完整时间线
- `PHASE1_COMPLETION_REPORT.md` - 完成报告

**可视化图表** (位于 `reports/figures/`):
- `phase1_classification_comparison.png` (139 KB)
- `phase1_regression_comparison.png` (472 KB)
- `phase1_winloss_summary.png` (182 KB)
- `phase1_relative_performance.png` (235 KB)

**实验数据**:
- `artifacts/runs/20260303_211013/` - EXP-01 结果
  - `config.json` - 配置文件
  - `checkpoints/best.pt` - 最佳模型
  - `metrics/best_summary.json` - 最佳指标
- `artifacts/runs/20260304_005923/` - EXP-02 结果
  - `config.json` - 配置文件
  - `checkpoints/best.pt` - 最佳模型
  - `metrics/best_summary.json` - 最佳指标

**日志文件**:
- `logs/exp02_graph_baseline_final.log` - EXP-02 训练日志
- `logs/continuous_monitor.log` - 持续监控日志

---

### D. 参考文献

1. **SchNet**: Schütt et al. "SchNet: A continuous-filter convolutional neural network for modeling quantum interactions." NeurIPS 2017.

2. **DimeNet**: Klicpera et al. "Directional Message Passing for Molecular Graphs." ICLR 2020.

3. **Materials Project**: Jain et al. "Commentary: The Materials Project: A materials genome approach to accelerating materials innovation." APL Materials 2013.

4. **Multi-task Learning**: Caruana. "Multitask Learning." Machine Learning 1997.

---

## 总结

Phase 1 成功建立了两个基线模型并完成了详细的对比分析。主要发现是：

1. **Composition Baseline 表现优异** - 在 82% 任务上优于 Graph Baseline
2. **Graph Baseline 有潜力** - 在 volume 和 energy_above_hull 上表现更好
3. **训练稳定性是关键** - 需要仔细调试 AMP 和学习率

**建议**: 继续改进 Graph Backbone，因为它在特定任务上展现了潜力。投入 1-2 周时间进行改进是值得的。

---

**报告生成时间**: 2026-03-04 02:13
**Phase 1 状态**: ✅ 完全完成
**下一个里程碑**: Phase 2 决策 (增强 Graph 或进入 Stage B)

---

**联系信息**:
- 项目: mp-data-pipeline
- Phase: Phase 1 (Baseline 建立)
- 完成时间: 2026-03-04 02:13


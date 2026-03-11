# Phase 1: EXP-01 vs EXP-02 对比分析

**生成时间**: 2026-03-04 02:11
**状态**: ✅ Phase 1 完成

---

## 📊 执行摘要

Phase 1 成功完成了两个基线实验的训练和对比分析：

- **EXP-01 (Composition Baseline)**: ✅ 完成，所有成功标准达成
- **EXP-02 (Graph Baseline)**: ✅ 完成，但结果**未达预期**

**关键发现**: Graph Backbone 在大多数任务上**表现不如** Composition Baseline，这与预期相反。

---

## 🔬 实验配置对比

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
| **数据加载时间** | ~6 小时 | ~6 小时 |

---

## 📈 关键指标对比（验证集）

### 分类任务

| 任务 | 指标 | EXP-01 (Composition) | EXP-02 (Graph) | 变化 | 胜者 |
|------|------|---------------------|----------------|------|------|
| **is_metal** | AUROC | **0.9098** | 0.8745 | -3.9% | 🏆 Composition |
| | Accuracy | **0.8329** | 0.7922 | -4.9% | 🏆 Composition |
| **is_stable** | AUROC | **0.8510** | 0.8050 | -5.4% | 🏆 Composition |
| | Accuracy | **0.8170** | 0.7949 | -2.7% | 🏆 Composition |

### 回归任务 - 能量相关

| 任务 | 指标 | EXP-01 (Composition) | EXP-02 (Graph) | 变化 | 胜者 |
|------|------|---------------------|----------------|------|------|
| **energy_per_atom** | MAE | **1.218** eV | 2.818 eV | +131% | 🏆 Composition |
| | RMSE | **2.857** eV | 6.075 eV | +113% | 🏆 Composition |
| **formation_energy** | MAE | **0.212** eV | 0.421 eV | +99% | 🏆 Composition |
| | RMSE | **0.437** eV | 0.592 eV | +36% | 🏆 Composition |
| **energy_above_hull** | MAE | **0.167** eV | 0.137 eV | -18% | 🏆 Graph |
| | RMSE | **0.421** eV | 0.337 eV | -20% | 🏆 Graph |

### 回归任务 - 电子性质

| 任务 | 指标 | EXP-01 (Composition) | EXP-02 (Graph) | 变化 | 胜者 |
|------|------|---------------------|----------------|------|------|
| **band_gap** | MAE | **0.715** eV | 0.952 eV | +33% | 🏆 Composition |
| | RMSE | **0.887** eV | 1.210 eV | +36% | 🏆 Composition |
| **cbm** | MAE | **0.755** eV | 1.006 eV | +33% | 🏆 Composition |
| | RMSE | **1.152** eV | 1.342 eV | +16% | 🏆 Composition |
| **vbm** | MAE | **0.662** eV | 0.870 eV | +31% | 🏆 Composition |
| | RMSE | **1.022** eV | 1.175 eV | +15% | 🏆 Composition |
| **efermi** | MAE | **0.841** eV | 1.035 eV | +23% | 🏆 Composition |
| | RMSE | **1.402** eV | 1.403 eV | +0.1% | ≈ 平局 |

### 回归任务 - 结构性质

| 任务 | 指标 | EXP-01 (Composition) | EXP-02 (Graph) | 变化 | 胜者 |
|------|------|---------------------|----------------|------|------|
| **volume** | MAE | **247.4** Å³ | 147.0 Å³ | -41% | 🏆 Graph |
| | RMSE | **482.0** Å³ | 320.3 Å³ | -34% | 🏆 Graph |
| **density** | MAE | **0.603** g/cm³ | 0.831 g/cm³ | +38% | 🏆 Composition |
| | RMSE | **1.164** g/cm³ | 1.192 g/cm³ | +2% | 🏆 Composition |

---

## 🏆 胜负统计

### 按任务类别

| 类别 | Composition 胜 | Graph 胜 | 平局 |
|------|---------------|----------|------|
| **分类任务** (2) | 2 | 0 | 0 |
| **能量任务** (3) | 2 | 1 | 0 |
| **电子任务** (4) | 4 | 0 | 0 |
| **结构任务** (2) | 1 | 1 | 0 |
| **总计** (11) | **9** | **2** | **0** |

### 胜率分析

- **Composition Baseline**: 9/11 任务 (82%)
- **Graph Baseline**: 2/11 任务 (18%)

**结论**: Composition Baseline 在 82% 的任务上优于 Graph Baseline

---

## 🔍 详细分析

### 1. Graph Baseline 的优势领域

Graph Baseline 仅在 2 个任务上表现更好：

1. **energy_above_hull** (MAE: 0.137 vs 0.167, -18%)
   - 这是热力学稳定性的关键指标
   - Graph 结构信息可能有助于捕捉局部能量差异

2. **volume** (MAE: 147.0 vs 247.4, -41%)
   - 显著改进
   - Graph 的空间信息直接相关于体积预测

### 2. Composition Baseline 的优势领域

Composition Baseline 在以下领域表现更好：

1. **分类任务** (is_metal, is_stable)
   - AUROC 差异: 3-5%
   - 元素组成可能是金属性和稳定性的主要决定因素

2. **能量任务** (energy_per_atom, formation_energy)
   - MAE 差异: 99-131%
   - **这是最大的差距**
   - Graph Baseline 在能量预测上严重失败

3. **电子性质** (band_gap, cbm, vbm, efermi)
   - MAE 差异: 23-33%
   - 电子性质可能主要由元素组成决定

### 3. 为什么 Graph Baseline 表现不佳？

可能的原因：

#### A. 训练不稳定性问题

- **AMP 禁用**: 为了避免 NaN，禁用了混合精度训练
  - 影响: 训练速度慢，可能影响收敛

- **学习率降低**: 从 3e-4 降低到 1e-4
  - 影响: 收敛速度慢，可能未充分优化

- **训练崩溃历史**:
  - 第一次: AMP 类型不匹配 (22:43)
  - 第二次: NaN 在 epoch 8 (23:45)
  - 第三次: 成功完成 (00:01-02:04)

#### B. 模型容量问题

- **参数量对比**:
  - Composition: ~100K 参数 (仅 embedding + pooling)
  - Graph: ~1M 参数 (6 层消息传递)

- **过拟合风险**: Graph 模型可能过拟合训练集
  - 训练损失: 134.7 vs 验证损失: 144.8 (差距 7.5%)
  - Composition 训练损失: 239.6 vs 验证损失: 237.3 (差距 -1%)

#### C. 图构建问题

- **Cutoff**: 6.0 Å
- **Max Neighbors**: 24
- **可能问题**:
  - Cutoff 太小，丢失长程相互作用
  - 边特征不足 (仅距离 RBF)
  - 缺少角度信息

#### D. 数据加载瓶颈

- **加载时间**: 6 小时 (与 Composition 相同)
- **内存使用**: 3.5% (约 4.5 GB)
- **可能问题**: 图构建效率低，影响训练效率

---

## 📊 训练曲线分析

### EXP-01 (Composition)

- **最佳 Epoch**: 50 (最后一个 epoch)
- **训练损失**: 239.6 → 持续下降
- **验证损失**: 237.3 → 持续下降
- **过拟合**: 无明显过拟合 (训练损失 > 验证损失)

### EXP-02 (Graph)

- **最佳 Epoch**: 50 (最后一个 epoch)
- **训练损失**: 134.7 → 持续下降
- **验证损失**: 144.8 → 持续下降
- **过拟合**: 轻微过拟合 (训练损失 < 验证损失，差距 7.5%)

---

## ⚠️ Gate-1 检查结果

**Gate-1 决策点** (Epoch 20): 如果所有任务都不优于 EXP-01，停止训练

**实际情况**:
- 由于训练过程中多次崩溃和重启，未在 Epoch 20 执行 Gate-1 检查
- 训练直接运行到 Epoch 50 完成

**事后分析**:
- 如果在 Epoch 20 执行 Gate-1，应该会**失败**
- 因为从最终结果看，Graph 在大多数任务上都不如 Composition
- 建议: 未来实验应严格执行 Gate 检查

---

## 🎯 Phase 1 成功标准检查

### EXP-01 成功标准 ✅

| 标准 | 目标 | 实际 | 状态 |
|------|------|------|------|
| is_metal AUROC | ≥ 0.75 | 0.9098 | ✅ 超出 21% |
| band_gap MAE | < 1.0 eV | 0.715 eV | ✅ 优于 28% |
| is_stable AUROC | ≥ 0.75 | 0.8510 | ✅ 超出 13% |
| 训练稳定 | 无崩溃 | 无崩溃 | ✅ |

**结论**: EXP-01 所有成功标准达成 ✅

### EXP-02 成功标准 ❌

| 标准 | 目标 | 实际 | 状态 |
|------|------|------|------|
| 50% 任务优于 EXP-01 | ≥ 50% | 18% (2/11) | ❌ 未达标 |
| band_gap MAE 降低 | ≥ 10% | +33% (恶化) | ❌ 未达标 |

**结论**: EXP-02 未达成功标准 ❌

---

## 🔧 问题诊断与建议

### 问题 1: 能量预测严重失败

**现象**: energy_per_atom MAE 从 1.218 eV 恶化到 2.818 eV (+131%)

**可能原因**:
1. 学习率过低 (1e-4)，能量任务未充分优化
2. AMP 禁用导致训练效率低
3. 图结构信息对能量预测无帮助（或有害）

**建议**:
- 尝试更高的学习率 (2e-4, 3e-4)
- 修复 AMP 问题后重新启用
- 检查能量归一化是否正确

### 问题 2: 训练不稳定

**现象**: 多次崩溃，需要禁用 AMP 和降低学习率

**已修复**:
- ✅ AMP 类型不匹配 (backbones.py:80)
- ✅ NaN 问题 (禁用 AMP)

**建议**:
- 重新审视 AMP 实现，尝试修复而非禁用
- 使用梯度裁剪 (已有 grad_clip=1.0)
- 添加梯度范数监控

### 问题 3: 图构建可能不合理

**现象**: Graph 在大多数任务上表现不佳

**建议**:
- 增加 cutoff (6.0 → 8.0 Å)
- 增加 max_neighbors (24 → 48)
- 添加边特征 (角度、键类型)
- 尝试其他图神经网络架构 (GIN, GAT, DimeNet)

### 问题 4: 模型容量不匹配

**现象**: Graph 模型参数多 10 倍，但表现更差

**建议**:
- 减少层数 (6 → 4)
- 减少 hidden_dim (256 → 128)
- 添加 Dropout 防止过拟合
- 使用更强的正则化

---

## 📋 Phase 1 总结

### 主要发现

1. **Composition Baseline 出乎意料地强大**
   - 简单的元素 embedding + mean pooling 在 82% 任务上表现最佳
   - 说明元素组成是材料性质的主要决定因素

2. **Graph Baseline 未达预期**
   - 仅在 2/11 任务上优于 Composition
   - 训练不稳定，需要多次修复
   - 能量预测严重失败

3. **图结构信息的价值有限**
   - 对 volume 预测有显著帮助 (-41%)
   - 对 energy_above_hull 有帮助 (-18%)
   - 对其他任务无帮助或有害

### 经验教训

1. **不要低估简单模型**
   - Composition Baseline 虽然简单，但非常有效
   - 复杂模型不一定更好

2. **训练稳定性至关重要**
   - AMP 问题导致多次崩溃
   - 学习率调整影响收敛

3. **Gate 检查应严格执行**
   - 如果在 Epoch 20 执行 Gate-1，可以节省 30 epochs 的训练时间

4. **图构建需要仔细设计**
   - 当前的图构建可能不合理
   - 需要更多领域知识

---

## 🚀 Phase 2 建议

### 选项 A: 增强 Graph Backbone (推荐)

**理由**: Graph 在 volume 和 energy_above_hull 上表现更好，说明图结构信息有价值

**改进方向**:
1. 修复 AMP 问题，重新启用混合精度训练
2. 增加 cutoff 和 max_neighbors
3. 添加边特征 (角度、键类型)
4. 尝试其他 GNN 架构 (DimeNet, SchNet++)
5. 使用更高的学习率 (2e-4, 3e-4)

**预计时间**: 1-2 周

### 选项 B: 混合模型 (Composition + Graph)

**理由**: Composition 在大多数任务上更好，Graph 在少数任务上更好

**设计**:
- 使用 Composition 作为主干
- 添加 Graph 分支用于 volume 和 energy_above_hull
- 任务特定的 backbone 选择

**预计时间**: 1 周

### 选项 C: 放弃 Graph，直接进入 Stage B

**理由**: Graph 表现不佳，不值得继续投入

**优点**:
- 节省时间
- Composition Baseline 已经很好

**缺点**:
- 放弃图结构信息
- 可能错过改进机会

**预计时间**: 立即开始 Stage B

---

## 📊 数据统计

### 训练数据
- **总样本数**: 123,903
- **Stage A 任务**: 11 (无弹性性质)
- **数据加载时间**: ~6 小时 (两个实验相同)

### 训练时间
- **EXP-01**: ~1.5 小时 (50 epochs)
- **EXP-02**: ~2 小时 (50 epochs)
- **总 GPU 时间**: ~3.5 小时

### 资源使用
- **GPU**: NVIDIA GPU (型号未知)
- **内存**: 3.5-5.8% (约 4.5-7.5 GB)
- **CPU**: 2000-3000% (20-30 核)

---

## 📁 输出文件

### EXP-01 (Composition)
- **Run ID**: 20260303_211013
- **Config**: `artifacts/runs/20260303_211013/config.json`
- **Best Checkpoint**: `artifacts/runs/20260303_211013/checkpoints/best.pt`
- **Metrics**: `artifacts/runs/20260303_211013/metrics/best_summary.json`

### EXP-02 (Graph)
- **Run ID**: 20260304_005923
- **Config**: `artifacts/runs/20260304_005923/config.json`
- **Best Checkpoint**: `artifacts/runs/20260304_005923/checkpoints/best.pt`
- **Metrics**: `artifacts/runs/20260304_005923/metrics/best_summary.json`

---

## 🎯 下一步行动

### 立即决策

**问题**: Graph Baseline 表现不佳，是否继续 Phase 2？

**选项**:
1. **增强 Graph Backbone** (推荐) - 1-2 周
2. **混合模型** - 1 周
3. **放弃 Graph，进入 Stage B** - 立即

**建议**: 选项 1 (增强 Graph Backbone)
- 理由: Graph 在 volume 和 energy_above_hull 上表现更好，说明有潜力
- 当前问题可能是实现和训练问题，而非架构问题

### 待办事项

- [ ] 决定 Phase 2 方向
- [ ] 如果继续 Graph: 设计改进方案
- [ ] 如果放弃 Graph: 开始 Stage B (弹性任务)
- [ ] 创建可视化图表 (对比图、训练曲线)
- [ ] 更新 experiment_log.md

---

**报告生成时间**: 2026-03-04 02:11
**Phase 1 状态**: ✅ 完成
**下一个里程碑**: Phase 2 决策

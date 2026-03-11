# 多任务材料性质预测：务实执行计划（v4）

- 文档日期：2026-03-03
- 文档版本：v4.0
- 状态：Active
- 上一版本：`2026-03-03_multitask_model_architecture_execution_plan_v3.md`
- 审查报告：`plan_review_synthesis.md`
- 目标：基于 v3 深度审查，提供可立即执行的务实计划

---

## 0. 执行摘要

本版本是 v3 的务实修订版，基于 6 位专家的深度审查。核心变化：

**简化架构**：3 套 → 2 套（B0 composition + B1 graph）
**压缩实验**：12 个 → 5 个
**降低成本**：450-750 GPU-hours → 100-150 GPU-hours
**快速反馈**：首个 baseline 1 天内完成

**主推路线**：渐进式验证
- Phase 0: 流程验证（1 天）
- Phase 1: Baseline 建立（3-5 天）
- Phase 2: 优化与弹性任务（1-2 周）

---

## 1. 研究目标（量化门槛）

### 1.1 Phase 1 目标（必须达成）

在 test-IID 上：

**必须达成**（否则停止并调试）:
- `is_metal` AUROC >= 0.75
- `band_gap` MAE < 1.0 eV
- 训练不崩溃，loss 下降

**期望达成**（验证方向正确）:
- `is_metal` AUROC >= 0.85
- `is_stable` AUROC >= 0.80
- `band_gap` MAE < 0.5 eV
- Graph (B1) 在 50% 任务上优于 Composition (B0)

**理想达成**（超出预期）:
- `band_gap` MAE 相比 composition 降低 >= 25%
- `energy_above_hull` MAE 降低 >= 20%

### 1.2 Phase 2 目标（弹性任务）

在弹性子集 test 上，相比均值预测：

- `bulk_modulus_vrh` MAE 降低 >= 15%
- `shear_modulus_vrh` MAE 降低 >= 15%
- `homogeneous_poisson` MAE 降低 >= 10%

### 1.3 Fail-fast 规则

**Gate-1**（Phase 1 后）:
- 如果 EXP-02 (Graph) 在 20 epochs 时所有任务都不优于 EXP-01 (Composition)
- 则停止并分析原因（代码 bug、数据问题、架构问题）

**Gate-2**（Phase 2 后）:
- 如果 EXP-03 弹性任务完全无法学习（MAE 不降）
- 则检查数据过滤和标注质量

---

## 2. 数据协议

### 2.1 数据来源

- 数据库：`data/db/mp_materials.db`
- 总样本：154,879
- 高覆盖任务（100%）：11 个
- 弹性任务（8.42%）：4 个

### 2.2 标签清洗（必须实现）

**弹性数据过滤**（P0 优先级）:
```python
def is_valid_elastic(row):
    if row.bulk_modulus_vrh is None:
        return False
    return (
        row.bulk_modulus_vrh > 0 and
        row.shear_modulus_vrh > 0 and
        -1 <= row.homogeneous_poisson <= 0.5 and
        row.universal_anisotropy >= 0
    )
```

**异常值处理**:
- 3-sigma 裁剪应用于 `energy_above_hull` 和弹性模量
- 记录过滤掉的样本数量

### 2.3 目标变换

**Z-score 标准化**（所有回归任务）:
```python
mean = train_targets.mean(dim=0)  # 只在训练集计算
std = train_targets.std(dim=0)
normalized = (targets - mean) / (std + 1e-8)
```

**Log1p 变换**（可选，长尾任务）:
- `energy_above_hull`
- `bulk_modulus_vrh`
- `shear_modulus_vrh`

**验证清单**:
- [ ] mean/std 只在训练集计算
- [ ] 验证/测试集用训练集的 mean/std
- [ ] 推理时正确反归一化
- [ ] log1p 只用于正值任务

---

## 3. 数据切分（已完成）

### 3.1 Split-S1: IID
- 路径：`data/splits/split_iid_seed42.json`
- 规则：按 `mp_id` 随机 80/10/10
- Seed：42

### 3.2 Split-S2: ChemSys-OOD
- 路径：`data/splits/split_chemsys_ood_seed42.json`
- 规则：按 `chemsys` 分组后切分
- 比例：约 70/15/15

### 3.3 Split-S3: Complexity-OOD
- 路径：`data/splits/split_complexity_ood.json`
- 规则：train ≤4 元素，val =5，test ≥6
- 用途：挑战性评估，非主要指标

---

## 4. 模型架构（简化版）

### 4.1 输入图构建（统一）

- 节点：原子序数 embedding（dim=128）
- 边：半径图 cutoff=6.0 Å，max_neighbors=24
- 边特征：RBF（n_rbf=64）
- 周期边界：PBC 展开

### 4.2 Backbone 选择（2 套）

#### B0: Composition Baseline
- 类型：纯组分编码器
- 实现：`CompositionBackbone`
- 参数：~1M
- 用途：建立最简单的 baseline

#### B1: Graph Network (Simple)
- 类型：消息传递图网络
- 实现：`MessagePassingBackbone`
- 配置：
  - layers=6
  - hidden_dim=256
  - edge_dim=64 (RBF)
  - dropout=0.1
- 参数：~5-10M
- 用途：验证图结构价值

#### B2: Graph Network (Enhanced) - 可选
- 在 B1 基础上添加：
  - Edge update
  - Multi-head attention
  - Residual connections
- 仅在 B1 效果不足时实现

#### B3: Equivariant Network - 延后
- EquiformerV2 风格
- 需要 e3nn 库
- 实现复杂度高（2-3 周）
- 仅在 B1/B2 都不够时考虑

### 4.3 多任务头（分组）

共享池化向量 `h_global` 后，接 5 个 group head：

**1. Thermo Head**:
- 输出：`energy_per_atom`, `formation_energy_per_atom`, `energy_above_hull`
- MLP: 256 → 256 → 3

**2. Electronic Head**:
- 输出：`band_gap`, `cbm`, `vbm`, `efermi`, `is_metal`
- MLP: 256 → 256 → 5
- **物理约束**：`band_gap = softplus(raw_output)`

**3. Stability Head**:
- 输出：`is_stable`
- MLP: 256 → 128 → 1

**4. Structure Head**:
- 输出：`volume`, `density`
- MLP: 256 → 128 → 2
- **物理约束**：`volume = softplus(raw) + 1.0`, `density = softplus(raw) + 0.1`

**5. Elastic Head**:
- 输出：`bulk_modulus_vrh`, `shear_modulus_vrh`, `homogeneous_poisson`, `universal_anisotropy`
- MLP: 256 → 256 → 4
- **物理约束**：模量用 softplus，泊松比用 sigmoid 映射到 [-1, 0.5]

### 4.4 物理约束（必须实现）

**P0 约束**（立即实现）:
1. `band_gap >= 0`: 用 softplus
2. `cbm >= vbm`: 损失函数中添加约束项

**P1 约束**（短期实现）:
3. `volume > 0`, `density > 0`: 用 softplus + offset
4. 弹性模量 > 0: 用 softplus

**P2 约束**（可选）:
5. 稳定性一致性：`is_stable` 与 `energy_above_hull` 负相关

---

## 5. 损失函数

### 5.1 基础损失

**回归任务**:
```python
loss_reg = F.huber_loss(pred, target, delta=1.0, reduction='none')
loss_reg = (loss_reg * mask).sum() / mask.sum()
```

**分类任务**:
```python
loss_cls = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
loss_cls = (loss_cls * mask).sum() / mask.sum()
```

### 5.2 物理约束损失

**CBM/VBM 约束**:
```python
cbm = pred[:, task_idx['cbm']]
vbm = pred[:, task_idx['vbm']]
mask_both = mask[:, task_idx['cbm']] * mask[:, task_idx['vbm']]
violation = F.relu(vbm - cbm + 0.01)
loss_cbm_vbm = (violation * mask_both).sum() / mask_both.sum()
```

### 5.3 总损失

**Phase 1**（静态权重）:
```python
loss = sum(w_t * loss_t for t in tasks) + 0.01 * loss_cbm_vbm
```

权重计算：
```python
w_t = 1.0 / sqrt(coverage_t)  # 低覆盖任务权重更高
```

**Phase 2**（可选：不确定度加权）:
```python
# 每个任务学习 log_sigma
loss_t = exp(-log_sigma_t) * base_loss_t + log_sigma_t
```

### 5.4 采样策略

**Stage A**：普通随机采样

**Stage B**：Task-balanced 采样
- Batch=64 时：
  - 32 条从全量池采样
  - 32 条优先从弹性标签可用池采样

---

## 6. 训练配置

### 6.1 优化器

- Optimizer: AdamW
- lr: 3e-4
- weight_decay: 1e-5
- betas: (0.9, 0.999)

### 6.2 学习率调度

- Warmup: 5 epochs (线性)
- Scheduler: Cosine decay
- Min lr: 1e-6

### 6.3 正则化

- Dropout: 0.1
- Gradient clip: 1.0
- EMA: 0.999（P1 优先级，可延后）

### 6.4 训练技巧

**P0**（必须实现）:
- AMP (自动混合精度): fp16 或 bf16
- 梯度累积：支持小 batch 模拟大 batch

**P1**（短期实现）:
- EMA：指数移动平均
- Per-task loss 监控

**P2**（可选）:
- 分布式训练（多 GPU）

---

## 7. 实验路线图（5 步）

### Phase 0: 流程验证

**EXP-00: Sanity Check**
- 配置：B0, Stage A, 5 epochs, tiny split
- Batch size: 16
- 目标：验证训练流程可运行
- 成功标准：不崩溃，loss 下降
- 时间：1-2 小时

### Phase 1: Baseline 建立

**EXP-01: Composition Baseline**
- 配置：B0, Stage A, 50 epochs, IID split
- Batch size: 32
- 目标：建立最简单的 baseline
- 记录：所有任务的 MAE/AUROC
- 成功标准：
  - `is_metal` AUROC > 0.75
  - `band_gap` MAE < 训练集标准差
- 时间：8-12 GPU-hours

**EXP-02: Graph Baseline**
- 配置：B1, Stage A, 50 epochs, IID split
- Batch size: 32
- 目标：验证图结构是否有帮助
- 成功标准：
  - 至少 50% 任务优于 EXP-01
  - `band_gap` MAE 相比 EXP-01 降低 > 10%
- 时间：12-18 GPU-hours
- **Gate-1**：如果不优于 EXP-01，停止并分析

### Phase 2: 优化与弹性

**EXP-03: Best Backbone + Stage B**
- 配置：Phase 1 最佳 + Stage B, 80 epochs
- Batch size: 64（或 32 + 梯度累积）
- 目标：验证弹性任务可学习
- 成功标准：
  - Stage A 任务不劣化 > 5%
  - 弹性任务 MAE 优于均值预测
- 时间：15-24 GPU-hours
- **Gate-2**：如果弹性任务无改善，检查数据

**EXP-04: Hyperparameter Tuning**
- 配置：最佳 backbone + Stage B
- 调整：
  - lr: [1e-4, 3e-4, 5e-4]
  - hidden_dim: [128, 256, 384]
  - layers: [4, 6, 8]
- 时间：3-5 次实验，每次 15-20 GPU-hours

### Phase 3: OOD 评估（可选）

**EXP-05: OOD Evaluation**
- 配置：最佳模型在 ChemSys-OOD 和 Complexity-OOD 上评估
- 不训练新模型，只评估现有 checkpoint
- 时间：2-4 小时

---

## 8. 硬件要求与时间预算

### 8.1 最小配置

**方案 A: 单 GPU（最低）**
- GPU: 1x RTX 3090/4090 (24GB)
- Batch size: 16-32
- 梯度累积: 2-4 步
- Hidden dim: 256
- Phase 1-2 时间：7-10 天

**方案 B: 单 A100（推荐）**
- GPU: 1x A100 40GB/80GB
- Batch size: 64
- Hidden dim: 256-384
- Phase 1-2 时间：3-5 天

**方案 C: 多 GPU（理想）**
- GPU: 4x A100 80GB
- 需实现 DDP
- Phase 1-2 时间：1-2 天

### 8.2 时间预算（单 GPU）

- Phase 0: 0.5 天
- Phase 1: 2-3 天
- Phase 2: 5-7 天
- **总计**: 约 10 天（单 GPU）

### 8.3 成本对比

| 维度 | v3 计划 | v4 计划 |
|---|---|---|
| 实验数量 | 12 | 5 |
| 总 GPU 时间 | 450-750h | 100-150h |
| 首次结果 | 需完成 4 个实验 | 1 个实验即可 |
| 最低硬件 | 4x A100 | 1x 3090 |

---

## 9. 评估与报告

### 9.1 必须输出

每个实验必须生成：

1. `config.json`：完整配置
2. `metrics_iid.json`：IID 测试集指标
3. `per_task_metrics.csv`：每个任务的详细指标
4. `training_log.txt`：训练日志
5. `best.pt`：最佳 checkpoint

### 9.2 可选输出

- `metrics_chemsys_ood.json`
- `metrics_complexity_ood.json`
- `error_analysis.csv`：按晶系、元素数、金属性分析

### 9.3 记录到实验日志

每次实验后更新 `reports/plans/experiment_log.md`：

```markdown
| 日期 | 实验ID | 配置摘要 | 主要指标 | 结论 |
|---|---|---|---|---|
| 2026-03-03 | EXP-01 | B0, Stage A, 50 epochs | is_metal AUROC=0.82, band_gap MAE=0.65 | Baseline 建立 |
```

### 9.4 更新指标汇总

最佳结果记录到 `reports/plans/metrics_summary.md`。

---

## 10. 实施清单

### 10.1 本周（P0 阻塞问题）

**代码修复**（~4 小时）:
- [ ] 添加 `band_gap >= 0` 约束（heads.py）
- [ ] 实现 `cbm >= vbm` 约束损失（losses.py）
- [ ] 添加弹性数据过滤（dataset.py）
- [ ] 验证归一化正确性（tests/test_normalization.py）

**基础设施**（~3 小时）:
- [ ] 实现 AMP 支持（trainer.py）
- [ ] 添加梯度累积（trainer.py）

**实验**（~12 GPU-hours）:
- [ ] 运行 EXP-00 验证流程
- [ ] 启动 EXP-01 建立 baseline

**工具**（~1 小时）:
- [ ] 创建 `scripts/analyze_splits.py`

### 10.2 下周（P1 重要功能）

**代码增强**（~5 小时）:
- [ ] 添加 EMA
- [ ] 增强监控（per-task loss 曲线）
- [ ] 添加体积/密度非负约束

**实验**（~18 GPU-hours）:
- [ ] 运行 EXP-02 (Graph)
- [ ] 对比 EXP-01 vs EXP-02
- [ ] 决定是否通过 Gate-1

**文档**:
- [ ] 更新 experiment_log.md
- [ ] 更新 metrics_summary.md

### 10.3 第三周（Phase 2）

**实验**（~40 GPU-hours）:
- [ ] 运行 EXP-03 (Stage B)
- [ ] 开始 EXP-04 (超参数调优)

**可选增强**:
- [ ] 实现不确定度加权
- [ ] 增强 B1 架构（edge update, attention）

---

## 11. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| B1 不优于 B0 | 中 | 高 | Gate-1 提前停止，分析原因 |
| 弹性任务无法学习 | 中 | 中 | Gate-2 检查数据质量 |
| 显存不足 | 低 | 中 | 梯度累积 + 小 batch |
| 数据加载慢 | 中 | 低 | 增加 workers，预加载 |
| 训练不收敛 | 低 | 高 | 从简单 baseline 开始 |

---

## 12. 与 v3 的主要差异

| 维度 | v3 | v4 |
|---|---|---|
| 架构数量 | 3 (B1/B2/B3) | 2 (B0/B1) |
| 实验数量 | 12 | 5 |
| 总 GPU 时间 | 450-750h | 100-150h |
| 首次结果 | R01-R04 后 | EXP-01 后 |
| 决策点 | 3 gates | 2 gates |
| 物理约束 | 描述 | 明确实现 |
| 不确定度加权 | 必须 | 可选 |
| GradNorm | 提及 | 删除 |
| Stage C | 必须 | 可选 |
| 最低硬件 | 4x A100 | 1x 3090 |

---

## 13. 参考文献

保留 v3 的参考文献列表。

---

## 14. 变更记录

- v4.0（2026-03-03）：基于 plan_review_synthesis.md 的深度审查，大幅简化为务实可执行版本
  - 架构从 3 套简化到 2 套
  - 实验从 12 个压缩到 5 个
  - 明确 P0/P1/P2 优先级
  - 添加详细的实施清单
  - 降低硬件要求和时间预算

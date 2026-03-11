# 多任务材料性质预测：模型结构深度计划（v2）

- 文档日期：2026-03-03
- 文档版本：v2.0
- 状态：Active
- 依赖数据：`data/db/mp_materials.db`（MP 风格标签）
- 本文重点：**模型结构选型与结构设计**（而非仅训练流程）

---

## 1. 研究目标（结构导向版）

构建一个“共享结构编码器 + 分组任务头”的多任务模型，满足：

1. 在高覆盖任务（`band_gap`, `energy_above_hull`, `is_metal`, `is_stable` 等）上稳定优于组成基线；
2. 在低覆盖弹性任务（`bulk_modulus_vrh`, `shear_modulus_vrh`, `homogeneous_poisson`, `universal_anisotropy`）上避免负迁移；
3. 具备 OOD（按 `chemsys`）可解释退化行为；
4. 架构可扩展到后续张量任务（若补齐 `elastic_tensor` 标签）。

---

## 2. 最新工作扫描（用于选型）

> 以下作为“2026-03 快照”，用于指导架构取舍。结论是推断，不是逐字原文。

### 2.1 基础/主干模型候选

1. **MACE family**
- MACE 架构（NeurIPS 2022）强调高阶等变消息传递，兼顾精度和效率。
- MACE-MP-0（2024）提出材料化学 foundation model，可直接用于多体系模拟与迁移。

2. **EquiformerV2 / FAIR-Chem 系列**
- EquiformerV2（ICLR 2024）在等变 Transformer 可扩展性上有改进。
- OMat24（2024）提供大规模数据与预训练模型；官方文档在 2025 建议优先 UMA。

3. **CHGNet / M3GNet**
- CHGNet（Nature Machine Intelligence 2023）引入电荷相关信息（通过磁矩代理）。
- M3GNet（Nature Computational Science 2023）是材料图网络通用势的重要基线。

4. **MatterSim / Orb-v3**
- MatterSim（2024）强调跨元素、温压条件下通用模拟能力和微调效率。
- Orb-v3（2025）强调精度-速度-内存帕累托。

### 2.2 基准与实践信号

1. **Matbench Discovery（Nature Machine Intelligence 2025）**显示模型排名与任务定义更贴近“发现效率”，而非仅回归 MAE。
2. 大量 2024-2025 研究共同结论：foundation/uMLIP 在零样本有价值，但**微调与任务对齐**仍关键，尤其是弹性和复杂 OOD 场景。

---

## 3. 架构决策：先做什么，不做什么

## 3.1 一期推荐主线（建议直接执行）

**主线架构：`Equivariant Shared Encoder + Task-Grouped Heads + Masked Multi-Task Optimization`**

核心原因：

1. 你是结构输入、性质多输出，等变编码器对晶体几何最直接；
2. 标签覆盖高度不均（154k vs 13k），必须分组 head 与掩码损失；
3. 需要后续扩展，shared backbone 最利于增量加任务。

## 3.2 一期不建议重投入方向

1. 直接上纯 LLM 文本路线作为主模型（可做辅助分支，不做主干）；
2. 直接做端到端全量张量回归（当前张量标签为空）；
3. 仅单任务训练后拼装（工程可行但泛化和维护成本高）。

---

## 4. 结构设计（可实现蓝图）

## 4.1 输入层（Input Block）

1. 原子节点：`Z` 的可学习 embedding。
2. 边特征：距离 RBF 展开 + cutoff mask。
3. 可选几何增强：角度/三体信息（可通过 line-graph 或等变模块隐式建模）。
4. 全局辅助特征（可选拼接到 readout）：
- `nsites`, `spacegroup`, `crystal_system`（one-hot/embedding）
- 组成统计特征（元素分数、平均原子序数等）

## 4.2 主干编码器（Backbone）

推荐做三档并行试验：

1. **B1（稳健 baseline）**：ALIGNN / Matformer 类（较轻、快速迭代）
2. **B2（主推）**：EquiformerV2 小/中参数量（31M/86M 级）
3. **B3（对照）**：MACE-MP 风格特征提取 + 任务头

> 说明：B2 作为主推，是因为在近期开源生态与 OOD 结果中更活跃；B1 作为强工程基线；B3 用于验证“foundation 特征迁移”收益。

## 4.3 共享表示到多任务头（Head Design）

采用“任务分组 + 专用头”，而不是每个任务独立完全割裂：

1. **Thermo 组**：`energy_per_atom`, `formation_energy_per_atom`, `energy_above_hull`
2. **Electronic 组**：`band_gap`, `cbm`, `vbm`, `efermi`, `is_metal`
3. **Stability 组**：`is_stable`（可与 `energy_above_hull` 做一致性约束）
4. **Structure 组**：`volume`, `density`
5. **Elastic 组（低覆盖）**：`bulk_modulus_vrh`, `shear_modulus_vrh`, `homogeneous_poisson`, `universal_anisotropy`

每组 head 采用：`MLP + LayerNorm + Dropout`，输出均值（可选方差）

## 4.4 物理一致性约束（Soft Constraints）

1. `band_gap >= 0`：使用 `softplus` 输出层
2. `cbm >= vbm`：加入 margin penalty
3. `is_stable` 与 `energy_above_hull` 一致性：
- 设 `p_stable` 与 `e_hull` 的 soft monotonic penalty

---

## 5. 多任务优化设计（重点）

## 5.1 掩码损失（必须）

每任务按标签可用性做 mask，缺失标签不计入损失。

## 5.2 权重策略（分阶段）

1. Warmup：静态权重（按 `1/sqrt(coverage)`）
2. 稳定后：切换到 **Uncertainty Weighting** 或 **GradNorm**
3. 若冲突明显：启用 **PCGrad**（仅在需要时）

## 5.3 两阶段训练（缓解负迁移）

1. **Stage A**：高覆盖任务（不含弹性）
2. **Stage B**：加载 A 权重，加入弹性组，使用 task-balanced sampler
3. **Stage C（可选）**：冻结 backbone 下层，仅微调 Elastic head + 上层 adapter

---

## 6. 具体实验矩阵（Architecture-Focused）

## 6.1 第一轮（确定主干）

1. B1 + grouped heads
2. B2 + grouped heads
3. B3 + grouped heads

对比指标：
- 主任务平均 rank（回归 MAE + 分类 AUROC）
- `is_stable` 与 `energy_above_hull` 一致性
- 训练吞吐与显存占用

## 6.2 第二轮（确定优化器与损失）

固定最佳主干后比较：

1. Static weights
2. Uncertainty weighting
3. GradNorm
4. PCGrad（只在前 3 显示冲突时）

## 6.3 第三轮（低覆盖专项）

1. 单独 Elastic head 微调
2. 两阶段 vs 单阶段
3. Oversampling ratio sweep（2x/4x/8x）

---

## 7. 评估协议（防止“看起来好”）

1. **三套 split 固化**：IID / `chemsys` OOD / 高元复杂度 OOD
2. **每任务单独报表**：
- 回归：MAE/RMSE/R2
- 分类：AUROC/PR-AUC/F1
3. **分群报告**：按晶系、元素数、金属/非金属
4. **校准评估**（若做不确定度）：ECE/可靠性曲线

---

## 8. 工程落地（你项目可直接对应）

建议新增目录：

1. `src/mp_data_pipeline/models/`：`backbones.py`, `heads.py`, `multitask_model.py`
2. `src/mp_data_pipeline/training/`：`losses.py`, `optimizers.py`, `trainer.py`
3. `scripts/`：`train_multitask.py`, `eval_multitask.py`, `export_splits.py`
4. `reports/plans/`：持续维护实验日志与指标表

---

## 9. 30 天执行排期（结构优先）

1. D1-D4：数据导出 + split 固化 + label mask pipeline
2. D5-D10：B1/B2/B3 首轮比较
3. D11-D16：多任务权重策略比较
4. D17-D22：弹性任务专项（两阶段 + 过采样）
5. D23-D26：OOD + 分群误差分析
6. D27-D30：定版模型与推理脚本

---

## 10. 风险清单（结构相关）

1. **负迁移**：弹性任务拖累主任务
- 对策：两阶段 + 弹性分组 head + PCGrad 兜底

2. **主干过重导致吞吐过低**
- 对策：先 31M 级模型验证方法，再扩到 86M

3. **OOD 表现不稳定**
- 对策：`chemsys` group split + 分群校准

4. **foundation checkpoint 与 MP 标签体系不完全对齐**
- 对策：统一 label protocol，必要时做小规模领域微调

---

## 11. 本版本关键决策

1. 采用“共享等变主干 + 分组任务头”作为主架构；
2. 训练采用“高覆盖先训 -> 全任务微调”两阶段；
3. 权重策略采用“静态 warmup -> 自适应”而非一次性复杂优化；
4. 把 OOD 和任务一致性约束纳入一等公民，不只追求平均 MAE。

---

## 12. 参考来源（最新工作与主文献）

1. MACE (NeurIPS 2022): https://papers.nips.cc/paper_files/paper/2022/hash/4a36c3c51af11ed9f34615b81edb5bbc-Abstract-Conference.html
2. MACE-MP-0 foundation model (2024): https://arxiv.org/abs/2401.00096
3. CHGNet (Nature Machine Intelligence 2023): https://www.nature.com/articles/s42256-023-00716-3
4. MatterSim (2024): https://arxiv.org/abs/2405.04967
5. EquiformerV2 (ICLR 2024): https://arxiv.org/abs/2306.12059
6. OMat24 dataset & models (2024): https://arxiv.org/abs/2410.12771
7. FAIR-Chem pretrained models docs (2025 UMA recommendation): https://fair-chem.github.io/inorganic_materials/models.html
8. Matbench Discovery repo + Nature 2025 benchmark context: https://github.com/janosh/matbench-discovery
9. Matbench Discovery preprint: https://arxiv.org/abs/2308.14920
10. Matformer (Periodic Graph Transformers, 2022): https://arxiv.org/abs/2209.11807
11. ALIGNN (npj Computational Materials, 2021): https://www.nature.com/articles/s41524-021-00650-1
12. Uncertainty weighting for MTL: https://arxiv.org/abs/1705.07115
13. GradNorm: https://arxiv.org/abs/1711.02257
14. PCGrad (gradient surgery): https://paperswithcode.com/paper/gradient-surgery-for-multi-task-learning-1
15. Nash-MTL: https://proceedings.mlr.press/v162/navon22a.html

---

## 13. 变更记录

- v2.0（2026-03-03）：新增“模型结构深度计划”，引入最新工作对比、主干选型、任务头设计与分阶段优化策略。

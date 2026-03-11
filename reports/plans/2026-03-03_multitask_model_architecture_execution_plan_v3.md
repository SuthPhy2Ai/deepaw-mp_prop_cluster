# 多任务材料性质预测：模型结构执行级计划（v3）

- 文档日期：2026-03-03
- 文档版本：v3.0
- 状态：Active
- 上一版本：`reports/plans/2026-03-03_multitask_model_architecture_deep_plan_v2.md`
- 目标：把“架构想法”变成可执行实验协议，避免假大空

---

## 0. 执行摘要（先看这个）

本版给出可直接实现的 3 套 backbone 方案（B1/B2/B3），统一用“共享编码器 + 分组 head + 掩码损失 + 两阶段训练”。

**主推路线**：B2（等变 Transformer）

- Stage A（高覆盖任务）120 epochs
- Stage B（全任务）80 epochs
- Stage C（弹性专项）40 epochs（可选）

以 IID + ChemSys-OOD + Complexity-OOD 三协议评估。达到门槛后才进入下一阶段。

---

## 1. 研究目标（量化门槛）

## 1.1 Primary KPI（必须达成）

在 test-IID 上：

1. `is_metal` AUROC >= 0.90
2. `is_stable` AUROC >= 0.85
3. `band_gap` MAE 相比 composition baseline 降低 >= 25%
4. `energy_above_hull` MAE 相比 composition baseline 降低 >= 20%

## 1.2 Secondary KPI（阶段目标）

在弹性子集 test（有标签）上，相比均值预测：

1. `bulk_modulus_vrh` MAE 降低 >= 20%
2. `shear_modulus_vrh` MAE 降低 >= 20%
3. `homogeneous_poisson` MAE 降低 >= 15%

## 1.3 Fail-fast 规则（节省时间）

任一架构若在 30% 训练进度时同时满足：

1. `band_gap` MAE 不优于 B0（composition baseline）
2. `is_metal` AUROC < 0.82

则提前停止该路线，不再追加算力。

---

## 2. 数据协议（冻结）

数据来自：`data/db/mp_materials.db`

当前关键覆盖（固定快照）：

- 总样本：154,879
- 高覆盖任务（100%）：`energy_per_atom`, `formation_energy_per_atom`, `energy_above_hull`, `band_gap`, `is_metal`, `is_stable`, `volume`, `density`
- 中覆盖任务：`cbm`, `vbm`（89,339; 57.68%），`efermi`（154,825; 99.97%）
- 低覆盖任务（弹性）：13,045（8.42%）

## 2.1 标签清洗（写入数据管线）

1. 去除 NaN/Inf（已在 loader 有处理）
2. 弹性过滤规则（仅用于训练，不删除原数据）
- `bulk_modulus_vrh > 0`
- `shear_modulus_vrh > 0`
- `-1 <= homogeneous_poisson <= 0.5`
- `universal_anisotropy >= 0`
3. 记录每条样本 mask，不做硬删除（除非全任务都无效）

## 2.2 目标变换（固定）

1. 回归目标：训练集统计量 z-score 标准化
2. 强长尾任务可选 `log1p`（仅正值）：`energy_above_hull`, `bulk_modulus_vrh`, `shear_modulus_vrh`
3. 推理阶段逆变换回原单位

---

## 3. 三套数据切分（必须全部产出）

## 3.1 Split-S1: IID

- 按 `mp_id` 随机 80/10/10
- seed 固定：42

## 3.2 Split-S2: ChemSys-OOD

- 按 `chemsys` 分组后切分，组内不泄漏
- 目标比例约 70/15/15（按样本量）

## 3.3 Split-S3: Complexity-OOD

- 训练：元素数 <= 4
- 验证：元素数 = 5
- 测试：元素数 >= 6

输出清单必须写入：`reports/plans/splits_manifest.md`

---

## 4. 模型结构蓝图（可直接编码）

## 4.1 输入图构建（统一）

1. 节点：原子序数 embedding（dim=128）
2. 边：半径图 cutoff=6.0 Å，max_neighbors=24
3. 边特征：RBF（n_rbf=64）+ 可选方向编码
4. 晶胞周期边界按 PBC 展开

## 4.2 Backbone 候选

### B1（工程稳健）

- 类型：ALIGNN/Matformer 类图网络
- 参数预算：10M-25M
- 作用：快速基线、低算力复现

### B2（主推）

- 类型：等变 Transformer（EquiformerV2 风格）
- 建议配置：
  - layers=8
  - hidden_dim=192
  - heads=8
  - dropout=0.1
- 参数预算：30M-60M

### B3（迁移对照）

- 类型：MACE/基础模型特征 + 任务头
- 策略：backbone 冻结与半冻结各跑一版

## 4.3 多任务头（分组，非完全共享）

共享池化向量 `h_global` 之后，接 5 个 group head：

1. Thermo head：
- 输出：`energy_per_atom`, `formation_energy_per_atom`, `energy_above_hull`
- MLP: 256 -> 256 -> 3

2. Electronic head：
- 输出：`band_gap`, `cbm`, `vbm`, `efermi`, `is_metal`
- MLP: 256 -> 256 -> 4 回归 + 1 分类

3. Stability head：
- 输出：`is_stable`
- MLP: 256 -> 128 -> 1

4. Structure head：
- 输出：`volume`, `density`
- MLP: 256 -> 128 -> 2

5. Elastic head：
- 输出：`bulk_modulus_vrh`, `shear_modulus_vrh`, `homogeneous_poisson`, `universal_anisotropy`
- MLP: 256 -> 256 -> 4

## 4.4 物理一致性层（软约束）

1. `band_gap` 输出用 `softplus` 保证非负
2. 加约束项：`cbm >= vbm`
- `L_cbm_vbm = mean(relu(vbm - cbm + m))`, `m=0.01`
3. 稳定性一致性（软）：
- 让 `is_stable` logit 与 `energy_above_hull` 负相关

---

## 5. 损失函数与优化（执行细节）

## 5.1 任务损失

设任务集合 `T`，样本 i 的任务 t 标签掩码为 `m_{i,t}`。

### 回归任务

- 基础损失：Huber（delta=1.0）
- 不确定度加权形式：

`L_reg_t = mean_i[ m_{i,t} * ( exp(-s_t) * Huber(yhat_{i,t}, y_{i,t}) + s_t ) ]`

其中 `s_t` 为可学习标量。

### 分类任务

`L_cls_t = mean_i[ m_{i,t} * BCEWithLogits(logit_{i,t}, y_{i,t}) ]`

### 总损失

`L = sum_t L_t + lambda1 * L_cbm_vbm + lambda2 * L_stability_consistency`

建议：`lambda1=0.05`, `lambda2=0.02` 起步。

## 5.2 采样器（关键）

- Stage A：普通随机采样
- Stage B/C：task-balanced 采样
  - batch=64 时：
    - 32 条来自全量池
    - 32 条优先从弹性标签可用池采样

## 5.3 优化器与训练配置

1. Optimizer: AdamW
2. lr: 3e-4
3. weight_decay: 1e-5
4. warmup: 5 epochs
5. scheduler: cosine decay
6. amp: fp16/bf16
7. grad_clip: 1.0
8. EMA: 0.999（建议开启）

---

## 6. 训练阶段设计（必须按阶段跑）

## Stage A（高覆盖预训练）

- 任务：不含弹性组
- epoch: 120
- 早停：patience=20
- 输出：`ckpt_stageA_best.pt`

## Stage B（全任务联合）

- 初始化：加载 Stage A
- 任务：全部任务
- epoch: 80
- 弹性权重提高（采样 + loss）
- 输出：`ckpt_stageB_best.pt`

## Stage C（弹性专项，可选）

- 初始化：Stage B
- 冻结 backbone 前 50% 层
- 重点训练 elastic head
- epoch: 40
- 输出：`ckpt_stageC_elastic_best.pt`

---

## 7. 实验矩阵（run list，照表执行）

## 7.1 架构筛选（第一轮）

1. R01: B1 + StageA
2. R02: B2 + StageA
3. R03: B3(frozen) + StageA
4. R04: B3(half-frozen) + StageA

Gate-G1：按 S1-IID 综合排名选前 2。

## 7.2 联合训练策略（第二轮）

1. R05: Top1 + StageB + static weights
2. R06: Top1 + StageB + uncertainty weights
3. R07: Top1 + StageB + GradNorm
4. R08: Top2 + StageB + uncertainty weights

Gate-G2：若 R06/R07 相对 R05 无收益，则回退 static。

## 7.3 弹性专项（第三轮）

1. R09: Top1 + StageC + oversample 2x
2. R10: Top1 + StageC + oversample 4x
3. R11: Top1 + StageC + oversample 8x
4. R12: Top1 + StageB only（无 StageC，对照）

Gate-G3：弹性 MAE 改善显著才保留 StageC。

---

## 8. 评估与报告模板（固定）

每个 run 必须输出：

1. `metrics_iid.json`
2. `metrics_chemsys_ood.json`
3. `metrics_complexity_ood.json`
4. `per_task_table.csv`
5. `error_slices.csv`（晶系、元素数、金属性）

在 `reports/plans/metrics_summary.md` 记录最佳结果（均值±方差，3 seeds）。

---

## 9. 计算资源与时间预算

## 9.1 推荐算力

1. 最佳：4x A100 80GB
2. 可用：1x A100 80GB（时间延长约 3-4 倍）
3. 最低：1x 3090/4090（先跑 B1，小 batch 梯度累积）

## 9.2 预算估计（单 seed）

1. Stage A：6-12 GPU-hours
2. Stage B：4-8 GPU-hours
3. Stage C：2-4 GPU-hours

完整矩阵（12 runs，1 seed）约 150-250 GPU-hours；
3 seeds 完整复现约 450-750 GPU-hours。

---

## 10. 工程落地文件（必须创建）

建议新增：

1. `src/mp_data_pipeline/models/backbones.py`
2. `src/mp_data_pipeline/models/heads.py`
3. `src/mp_data_pipeline/models/multitask_model.py`
4. `src/mp_data_pipeline/training/losses.py`
5. `src/mp_data_pipeline/training/sampler.py`
6. `src/mp_data_pipeline/training/trainer.py`
7. `scripts/train_multitask.py`
8. `scripts/eval_multitask.py`
9. `scripts/export_splits.py`

输出目录规范：

- `artifacts/runs/{run_id}/configs/`
- `artifacts/runs/{run_id}/checkpoints/`
- `artifacts/runs/{run_id}/metrics/`
- `artifacts/runs/{run_id}/logs/`

---

## 11. 决策闸门（避免无限试错）

1. **Gate-G1（架构）**：R01-R04 后只保留前 2
2. **Gate-G2（优化策略）**：R05-R08 后只保留前 1
3. **Gate-G3（弹性专项）**：R09-R12 只有当弹性收益明显才保留 StageC

“明显收益”定义：

- 至少 2/4 弹性任务 MAE 相比 R12 降低 >= 8%，且主任务不劣化超过 2%。

---

## 12. 参考来源（用于选型依据）

1. MACE (NeurIPS 2022): https://papers.nips.cc/paper_files/paper/2022/hash/4a36c3c51af11ed9f34615b81edb5bbc-Abstract-Conference.html
2. MACE-MP-0 (2024): https://arxiv.org/abs/2401.00096
3. CHGNet (Nature Machine Intelligence 2023): https://www.nature.com/articles/s42256-023-00716-3
4. EquiformerV2 (ICLR 2024): https://arxiv.org/abs/2306.12059
5. OMat24 dataset/models (2024): https://arxiv.org/abs/2410.12771
6. FAIR-Chem models docs (2025): https://fair-chem.github.io/inorganic_materials/models.html
7. Matbench Discovery benchmark context: https://github.com/janosh/matbench-discovery
8. Matbench Discovery preprint: https://arxiv.org/abs/2308.14920
9. ALIGNN: https://www.nature.com/articles/s41524-021-00650-1
10. Matformer: https://arxiv.org/abs/2209.11807
11. Uncertainty weighting: https://arxiv.org/abs/1705.07115
12. GradNorm: https://arxiv.org/abs/1711.02257
13. PCGrad: https://paperswithcode.com/paper/gradient-surgery-for-multi-task-learning-1

---

## 13. 变更记录

- v3.0（2026-03-03）：将 v2 拆解为执行级协议，补充了具体层级配置、损失定义、采样器、阶段训练、run matrix、算力预算与决策闸门。

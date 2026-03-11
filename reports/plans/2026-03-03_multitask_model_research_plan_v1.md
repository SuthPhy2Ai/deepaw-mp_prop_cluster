# 多任务材料性质预测研究计划（v1）

- 文档日期：2026-03-03
- 文档版本：v1.0
- 状态：Active
- 数据来源：`data/db/mp_materials.db`（154,879 条，2026-03-03 验证）

## 1. 研究目标（必须明确）

构建一个基于晶体结构输入的多任务模型，在**同一个模型**中同时预测关键材料性质，并在 IID 与 OOD 场景下保持可用精度。

### 1.1 主目标（Primary Objective）

在测试集上，相比组成基线模型（composition-only），多任务结构模型在高覆盖任务上达到显著提升，并满足以下最低门槛：

1. `is_metal`：AUROC >= 0.90
2. `is_stable`：AUROC >= 0.85
3. `band_gap`：MAE 显著低于基线（目标 >= 25% 相对下降）
4. `energy_above_hull`：MAE 显著低于基线（目标 >= 20% 相对下降）

### 1.2 次目标（Secondary Objective）

在弹性子集（约 13k）上，模型对以下任务优于“均值预测”基线：

1. `bulk_modulus_vrh`
2. `shear_modulus_vrh`
3. `homogeneous_poisson`
4. `universal_anisotropy`

## 2. 研究问题与假设

### RQ1
共享编码器的多任务学习是否优于单任务模型？

- H1：在高覆盖任务上，多任务模型总体优于单任务平均表现。

### RQ2
弹性低覆盖任务是否会拖累主任务（负迁移）？

- H2：两阶段训练（先高覆盖后全任务）可缓解负迁移。

### RQ3
模型在化学体系 OOD 上的性能退化是否可控？

- H3：按 `chemsys` 分组划分并训练后，OOD 指标下降幅度可控且可解释。

## 3. 任务范围

### 3.1 本期纳入任务

- 回归：`energy_per_atom`、`formation_energy_per_atom`、`energy_above_hull`、`band_gap`、`cbm`、`vbm`、`efermi`、`volume`、`density`
- 分类：`is_metal`、`is_stable`
- 低覆盖回归：`bulk_modulus_vrh`、`shear_modulus_vrh`、`homogeneous_poisson`、`universal_anisotropy`

### 3.2 本期排除任务

- `elastic_tensor`、`compliance_tensor`（当前源数据覆盖为 0）

## 4. 成功判据（验收标准）

## 4.1 必达

1. 完整可复现训练流程（固定 split、固定 seed、固定配置）
2. 主目标四项全部达标
3. 生成可复用推理接口（结构输入 -> 多任务输出）

## 4.2 加分

1. 弹性任务优于单任务模型
2. OOD 指标下降小于预设阈值（例如 MAE 上升 < 30%）
3. 提供校准后不确定度估计（ensemble/MC dropout）

## 5. 执行计划（里程碑）

### M1：数据与协议冻结（D1-D5）

1. 导出训练表（结构、标签、mask）
2. 完成标签清洗规则（NaN/Inf、物理约束过滤）
3. 生成三套固定 split：IID、`chemsys` OOD、复杂度 OOD
4. 将 split 与清洗脚本持久化

### M2：基线建立（D6-D10）

1. Baseline-A：composition-only
2. Baseline-B：单任务结构模型
3. 固化 baseline 指标表

### M3：多任务主实验（D11-D20）

1. 共享编码器 + 多头 + mask loss
2. 权重策略：coverage 权重 -> uncertainty weighting
3. 两阶段训练（高覆盖 -> 全任务）

### M4：评估与误差分析（D21-D25）

1. IID/OOD 分别评估
2. 子群体切片（晶系/元素数/金属性）
3. 失败样本回溯

### M5：封装与交付（D26-D30）

1. 最佳 checkpoint 固化
2. 推理脚本与文档
3. 研究总结报告

## 6. 持久化策略（防丢）

以下文件/目录作为“长期记录”，每次更新必须落盘：

1. 研究主计划：`reports/plans/2026-03-03_multitask_model_research_plan_v1.md`
2. 最新入口：`reports/plans/LATEST.md`
3. 数据划分：`reports/plans/splits_manifest.md`
4. 实验台账：`reports/plans/experiment_log.md`
5. 指标汇总：`reports/plans/metrics_summary.md`

## 7. 风险与应对

1. 低覆盖任务噪声大：两阶段训练 + task-balanced sampling
2. 标签异常值影响回归：稳健损失（Huber）+ 分位裁剪
3. 多任务冲突：GradNorm / PCGrad 作为备选
4. OOD 崩溃：`chemsys` 分组验证 + 子群体回归分析

## 8. 决策记录模板（每次关键决策都要写）

| 日期 | 决策 | 原因 | 影响 |
|---|---|---|---|
| 2026-03-03 | 采用两阶段训练 | 降低低覆盖任务负迁移 | 提升主任务稳定性 |

## 9. 变更记录

- v1.0（2026-03-03）：首次创建，明确主/次目标、验收标准、里程碑与持久化规范。

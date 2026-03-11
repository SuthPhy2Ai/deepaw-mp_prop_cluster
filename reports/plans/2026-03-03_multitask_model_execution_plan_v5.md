# 多任务材料性质预测：现实执行计划（v5）

**文档日期**: 2026-03-03
**文档版本**: v5.0
**状态**: Draft
**上一版本**: `2026-03-03_multitask_model_execution_plan_v4.md`
**变更理由**: 基于严格审查，修正 v4 的过度乐观估算，补充缺失的失败场景分析

---

## 0. 执行摘要

v5 是对 v4 的现实主义修订，核心变化：

**时间估算修正**: 10 天 → 3-4 周（单 GPU）
**增加缓冲**: 预留 50% 调试时间
**补充失败场景**: 每个 Phase 都有明确的停止条件和回退方案
**强化验证**: 在 EXP-00 之前完成数据质量分析和基础设施验证

**核心原则**:
- 悲观估算，超额交付
- 每步验证，快速失败
- 记录一切，便于复盘

---

## 1. 背景与动机

### 1.1 为什么需要 v5？

v4 计划虽然相比 v3 更务实，但仍存在以下问题：

1. **时间估算过于乐观**: 假设一切顺利，未考虑调试时间
2. **失败场景分析不足**: 只有 2 个 Gate，缺少详细的失败处理
3. **数据质量问题被低估**: 弹性数据只有 8.42%，质量未知
4. **监控基础设施缺失**: 无法诊断训练问题
5. **CompositionBackbone 可能太弱**: 可能导致 baseline 过低

### 1.2 v5 的改进

- 更现实的时间估算（考虑调试和重试）
- 每个 Phase 都有详细的失败场景和回退方案
- 在实验前完成数据质量分析和基础设施验证
- 增强监控和可视化（提升到 P0）
- 可选增强 CompositionBackbone 或直接跳过

---

## 2. 当前实现状态（已验证）

### 2.1 已实现功能 ✅

**核心架构**:
- ✅ CompositionBackbone: 简单的 atom embedding + mean pooling
- ✅ GraphBackbone: 6 层消息传递网络
- ✅ GroupedTaskHeads: 5 个分组任务头
- ✅ 物理约束: band_gap, volume, density, elastic moduli (softplus/sigmoid)

**训练基础设施**:
- ✅ AMP 支持: torch.cuda.amp.autocast + GradScaler
- ✅ 梯度累积: accumulation_steps 参数
- ✅ 梯度裁剪: grad_clip=1.0
- ✅ CBM/VBM 约束损失: lambda=0.05
- ✅ 弹性数据过滤: strict_elastic_filter=True

**数据处理**:
- ✅ 三种 split: IID, ChemSys-OOD, Complexity-OOD
- ✅ WeightedRandomSampler: Stage B/C 过采样弹性数据
- ✅ 图构建: cutoff=6.0Å, max_neighbors=24

### 2.2 未实现/未验证功能 ❌

**测试验证**:
- ❌ 归一化测试无法运行（pytest 未安装）
- ❌ 没有运行过任何实验（EXP-00 未执行）
- ❌ 训练流程未端到端验证

**监控基础设施**:
- ❌ 无 per-task loss 曲线
- ❌ 无梯度范数监控
- ❌ 无学习率曲线
- ❌ 无 tensorboard/wandb 集成

**数据质量分析**:
- ❌ 无弹性数据覆盖率统计
- ❌ 无异常值分析
- ❌ 无 split 平衡性检查
- ❌ 无 ChemSys-OOD 信息泄漏检查

**高级功能**:
- ❌ EMA (指数移动平均)
- ❌ 学习率 warmup scheduler
- ❌ 分布式训练

---

## 3. Phase 0: 基础设施验证（2-3 天）

**目标**: 在运行实验前，确保所有基础设施就绪

### 3.1 环境准备（0.5 天）

**任务清单**:
1. 安装缺失依赖
   ```bash
   pip install pytest tensorboard
   ```

2. 验证 Python 环境
   ```bash
   python -c "import torch; print(torch.__version__)"
   python -c "import ase; print(ase.__version__)"
   ```

3. 检查 GPU 可用性
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   nvidia-smi
   ```

**成功标准**: 所有依赖安装成功，GPU 可用

### 3.2 数据质量分析（1 天）

**创建分析脚本**: `scripts/analyze_data_quality.py`

**分析内容**:
1. **弹性数据统计**
   - 总样本数 vs 弹性样本数
   - 每个 split (train/val/test) 的弹性覆盖率
   - 弹性属性的分布（直方图）

2. **异常值检测**
   - 3-sigma 规则检测异常值
   - 物理不合理值（负模量、泊松比超范围）
   - 记录异常样本的 mp_id

3. **Split 平衡性**
   - 每个 split 的金属/非金属比例
   - 每个 split 的稳定/不稳定比例
   - 每个 split 的平均元素数

4. **ChemSys-OOD 信息泄漏检查**
   - 统计训练集和测试集的元素重叠度
   - 检查是否存在 "Fe-O" 在训练，"Fe-O-H" 在测试的情况
   - 如果泄漏严重，考虑重新生成 split

**输出文件**:
- `reports/data_quality_report.md`: 文字报告
- `reports/figures/elastic_distribution.png`: 弹性数据分布图
- `reports/figures/split_balance.png`: Split 平衡性图

**成功标准**:
- 弹性数据过滤后 >= 10,000 样本
- 异常值比例 < 5%
- Split 平衡性合理（金属比例差异 < 10%）

**失败场景**:
- 如果弹性数据 < 5,000: 放弃 Stage B，只做 Stage A
- 如果异常值 > 10%: 需要数据清洗，延后 1-2 天
- 如果 ChemSys-OOD 严重泄漏: 重新生成 split

### 3.3 归一化验证（0.5 天）

**任务**:
1. 运行归一化测试
   ```bash
   python -m pytest tests/test_normalization.py -v
   ```

2. 如果测试失败，修复归一化代码

3. 添加可视化：绘制归一化前后的分布图
   ```python
   # 在 scripts/visualize_normalization.py 中
   import matplotlib.pyplot as plt
   # 绘制 band_gap, energy_above_hull 等的归一化前后分布
   ```

**成功标准**: 所有测试通过，分布图显示归一化合理

### 3.4 监控基础设施（0.5 天）

**任务**: 增强训练监控（提升到 P0）

**修改文件**: `src/mp_data_pipeline/training/trainer.py`

**添加功能**:
1. Tensorboard 集成
   ```python
   from torch.utils.tensorboard import SummaryWriter
   writer = SummaryWriter(log_dir=run_dir / "tensorboard")
   ```

2. Per-task loss 记录
   ```python
   for task in enabled_tasks:
       writer.add_scalar(f"train/{task}_loss", loss_t, epoch)
   ```

3. 梯度范数监控
   ```python
   total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e10)
   writer.add_scalar("train/grad_norm", total_norm, step)
   ```

4. 学习率曲线
   ```python
   writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
   ```

**成功标准**: 运行 1 个 epoch 后，tensorboard 显示所有曲线

### 3.5 Sanity Check（0.5 天）

**任务**: 运行最小化实验验证流程

**命令**:
```bash
python scripts/train_multitask.py \
  --split data/splits/split_iid_tiny.json \
  --stage a \
  --backbone composition \
  --epochs 2 \
  --batch-size 8 \
  --device cuda
```

**检查项**:
- [ ] 训练不崩溃（无 OOM、无 NaN loss）
- [ ] Loss 下降
- [ ] Checkpoint 保存成功
- [ ] Tensorboard 记录正常
- [ ] 评估脚本可运行

**失败场景**:
- OOM: 减小 batch_size 或 hidden_dim
- NaN loss: 检查学习率、归一化、数据异常值
- 其他错误: 调试并修复

**预计时间**: 0.5 天（包括调试）

---

## 4. Phase 1: Baseline 建立（5-7 天）

**目标**: 建立可靠的 baseline，验证图结构价值

### 4.1 决策点：是否增强 CompositionBackbone？

**当前 CompositionBackbone 的问题**:
- 只是简单的 atom embedding + mean pooling
- 没有考虑元素比例（stoichiometry）
- 没有考虑元素周期表位置

**选项 A**: 直接使用当前 CompositionBackbone
- 优点: 节省时间
- 缺点: baseline 可能过低，导致 B1 提升被高估

**选项 B**: 增强 CompositionBackbone
- 添加元素比例特征
- 添加周期表位置编码（族、周期）
- 预计时间: +1 天

**选项 C**: 跳过 CompositionBackbone，直接从 GraphBackbone 开始
- 优点: 节省 2-3 天
- 缺点: 缺少对比，不知道图结构的真实价值

**建议**: 选项 A（先用简单版本），如果结果太差（AUROC < 0.7）再考虑选项 B

### 4.2 EXP-01: Composition Baseline（2-3 天）

**配置**:
```bash
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone composition \
  --hidden-dim 256 \
  --epochs 50 \
  --batch-size 32 \
  --lr 3e-4 \
  --device cuda
```

**预期时间**: 8-12 GPU-hours（实际可能 12-18 小时，考虑调试）

**成功标准**（必须达成）:
- `is_metal` AUROC >= 0.75
- `band_gap` MAE < 1.0 eV
- 训练不崩溃，loss 稳定下降

**期望标准**:
- `is_metal` AUROC >= 0.80
- `is_stable` AUROC >= 0.75
- `band_gap` MAE < 0.7 eV

**失败场景**:
- 如果 `is_metal` AUROC < 0.70: CompositionBackbone 太弱，执行选项 B（增强）
- 如果训练不收敛: 检查学习率、归一化、数据质量
- 如果 loss 出现 NaN: 检查梯度爆炸、数据异常值

**调试预留**: 1-2 天

### 4.3 EXP-02: Graph Baseline（3-4 天）

**配置**:
```bash
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone graph \
  --hidden-dim 256 \
  --layers 6 \
  --epochs 50 \
  --batch-size 32 \
  --lr 3e-4 \
  --device cuda
```

**预期时间**: 12-18 GPU-hours（实际可能 18-24 小时）

**成功标准**:
- 至少 50% 任务优于 EXP-01
- `band_gap` MAE 相比 EXP-01 降低 >= 10%

**期望标准**:
- 至少 70% 任务优于 EXP-01
- `band_gap` MAE 降低 >= 20%
- `is_metal` AUROC >= 0.85

**Gate-1 决策**（20 epochs 时）:
- 如果所有任务都不优于 EXP-01: **停止训练**，分析原因
  - 可能原因: 图构建错误、学习率不合适、模型容量不足
  - 回退方案: 检查代码、调整超参数、增加模型容量

**失败场景**:
- 如果 Gate-1 失败: 预留 2-3 天调试
- 如果最终结果不优于 EXP-01: 考虑增强 GraphBackbone（添加 edge update, attention）

**调试预留**: 1-2 天

### 4.4 Phase 1 总结与决策

**输出文件**:
- `reports/phase1_summary.md`: 对比 EXP-01 vs EXP-02
- `reports/figures/phase1_comparison.png`: 可视化对比

**决策点**:
- 如果 GraphBackbone 显著优于 Composition: 进入 Phase 2
- 如果差异不大: 考虑增强 GraphBackbone 或调整超参数
- 如果 GraphBackbone 更差: 深入分析原因，可能需要重构

**预计总时间**: 5-7 天（包括调试）

---

## 5. Phase 2: 弹性任务与优化（10-14 天）

**目标**: 验证弹性任务可学习，找到最佳超参数

### 5.1 EXP-03: Best Backbone + Stage B（4-5 天）

**配置**:
```bash
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage b \
  --backbone <best_from_phase1> \
  --hidden-dim 256 \
  --layers 6 \
  --epochs 80 \
  --batch-size 32 \
  --oversample-elastic 4.0 \
  --lr 3e-4 \
  --device cuda
```

**预期时间**: 15-24 GPU-hours（实际可能 24-36 小时）

**成功标准**:
- Stage A 任务不劣化 > 5%
- 弹性任务 MAE 优于均值预测（baseline）

**期望标准**:
- `bulk_modulus_vrh` MAE 相比均值预测降低 >= 15%
- `shear_modulus_vrh` MAE 降低 >= 15%
- `homogeneous_poisson` MAE 降低 >= 10%

**Gate-2 决策**（40 epochs 时）:
- 如果弹性任务完全无法学习（MAE 不降）: **停止训练**
  - 可能原因: 数据质量差、过采样不足、任务权重不合理
  - 回退方案: 检查数据过滤、增加过采样倍数（8x-16x）、调整任务权重

**失败场景**:
- 如果 Stage A 任务劣化 > 10%: 过采样太激进，减少到 2x
- 如果弹性任务无改善: 检查数据质量，可能需要重新清洗
- 如果训练不稳定: 降低学习率或增加 warmup

**调试预留**: 2-3 天

### 5.2 EXP-04: 超参数调优（6-9 天）

**目标**: 找到最佳超参数组合

**搜索空间**:
- Learning rate: [1e-4, 3e-4, 5e-4]
- Hidden dim: [128, 256, 384]
- Layers: [4, 6, 8]
- Batch size: [16, 32, 64]

**策略**: 网格搜索或随机搜索（3-5 次实验）

**单次实验配置**:
```bash
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage b \
  --backbone <best> \
  --hidden-dim <search> \
  --layers <search> \
  --epochs 80 \
  --batch-size <search> \
  --lr <search> \
  --device cuda
```

**预期时间**: 每次 15-20 GPU-hours，共 60-100 GPU-hours

**成功标准**: 找到比 EXP-03 更好的配置

**优化建议**:
- 先搜索学习率（最重要）
- 再搜索模型容量（hidden_dim, layers）
- 最后调整 batch_size

**失败场景**:
- 如果所有配置都不优于 EXP-03: 当前架构已达上限，考虑增强架构
- 如果显存不足: 使用梯度累积或减小 batch_size

**调试预留**: 2-3 天

### 5.3 Phase 2 总结

**输出文件**:
- `reports/phase2_summary.md`: 超参数搜索结果
- `reports/figures/hyperparameter_heatmap.png`: 超参数影响可视化
- `reports/best_model_card.md`: 最佳模型的详细信息

**预计总时间**: 10-14 天（包括调试）

---

## 6. Phase 3: OOD 评估（可选，2-3 天）

**目标**: 评估模型的泛化能力

### 6.1 ChemSys-OOD 评估

**前提**: Phase 0 已检查无严重信息泄漏

**命令**:
```bash
python scripts/eval_multitask.py \
  --split data/splits/split_chemsys_ood_seed42.json \
  --checkpoint artifacts/runs/<best>/checkpoints/best.pt \
  --backbone <best> \
  --out artifacts/eval_chemsys_ood.json
```

**分析**:
- 对比 IID test vs ChemSys-OOD test 的性能下降
- 识别哪些任务泛化能力强/弱
- 分析失败案例（哪些化学系统预测最差）

### 6.2 Complexity-OOD 评估

**注意**: 这是极端的分布偏移，预期性能会显著下降

**命令**:
```bash
python scripts/eval_multitask.py \
  --split data/splits/split_complexity_ood.json \
  --checkpoint artifacts/runs/<best>/checkpoints/best.pt \
  --backbone <best> \
  --out artifacts/eval_complexity_ood.json
```

**分析**:
- 作为挑战性评估，不作为主要指标
- 了解模型在复杂材料上的表现
- 识别模型的局限性

### 6.3 建议

**如果时间紧张**: 跳过 Phase 3，专注于 IID 性能
**如果有时间**: 运行 ChemSys-OOD，跳过 Complexity-OOD

---

## 7. 时间与成本估算

### 7.1 现实时间估算（单 GPU）

| Phase | 理想时间 | 调试时间 | 总时间 |
|---|---|---|---|
| Phase 0: 基础设施验证 | 2 天 | 1 天 | **3 天** |
| Phase 1: Baseline 建立 | 4 天 | 2 天 | **6 天** |
| Phase 2: 弹性任务与优化 | 8 天 | 4 天 | **12 天** |
| Phase 3: OOD 评估（可选） | 1 天 | 1 天 | **2 天** |
| **总计（不含 Phase 3）** | 14 天 | 7 天 | **21 天** |
| **总计（含 Phase 3）** | 15 天 | 8 天 | **23 天** |

**结论**: 预计 **3-4 周**（单 GPU），而非 v4 声称的 10 天

### 7.2 GPU 时间估算

| 实验 | 单次时间 | 重试次数 | 总时间 |
|---|---|---|---|
| EXP-00 (Sanity) | 1h | 2 | 2h |
| EXP-01 (Composition) | 12h | 1.5 | 18h |
| EXP-02 (Graph) | 18h | 1.5 | 27h |
| EXP-03 (Stage B) | 24h | 1.5 | 36h |
| EXP-04 (Tuning, 5 次) | 20h × 5 | 1 | 100h |
| **总计** | - | - | **183 GPU-hours** |

**对比 v4**: v4 估算 100-150h，v5 估算 180h（更现实）

### 7.3 硬件建议

**最低配置**（可行但慢）:
- 1x RTX 3090/4090 (24GB)
- Batch size: 16-32
- 梯度累积: 2-4 步
- 预计时间: 4-5 周

**推荐配置**:
- 1x A100 40GB/80GB
- Batch size: 32-64
- 预计时间: 3-4 周

**理想配置**:
- 4x A100 80GB（需实现 DDP）
- 预计时间: 1-2 周

---

## 8. 失败场景与回退方案

### 8.1 Phase 0 失败场景

| 失败场景 | 概率 | 回退方案 | 额外时间 |
|---|---|---|---|
| 弹性数据 < 5,000 | 中 | 放弃 Stage B，只做 Stage A | -1 周 |
| 异常值 > 10% | 中 | 数据清洗，重新过滤 | +2 天 |
| ChemSys-OOD 严重泄漏 | 低 | 重新生成 split | +1 天 |
| 归一化测试失败 | 低 | 修复归一化代码 | +1 天 |
| Sanity check 崩溃 | 中 | 调试代码、调整配置 | +1-2 天 |

### 8.2 Phase 1 失败场景

| 失败场景 | 概率 | 回退方案 | 额外时间 |
|---|---|---|---|
| Composition AUROC < 0.70 | 中 | 增强 CompositionBackbone | +1 天 |
| Graph 不优于 Composition | 中 | 增强 GraphBackbone | +3-5 天 |
| 训练不收敛 | 低 | 调整学习率、归一化 | +2-3 天 |
| OOM 错误 | 中 | 减小 batch/hidden_dim | +1 天 |

### 8.3 Phase 2 失败场景

| 失败场景 | 概率 | 回退方案 | 额外时间 |
|---|---|---|---|
| 弹性任务无法学习 | 中 | 检查数据质量、增加过采样 | +3-5 天 |
| Stage A 任务劣化 > 10% | 中 | 减少过采样倍数 | +2-3 天 |
| 超参数搜索无改善 | 低 | 当前架构已达上限 | 停止或重构 |

### 8.4 止损点

**Phase 0 止损点**:
- 如果弹性数据质量极差（< 3,000 或异常值 > 20%）: 放弃整个项目或重新采集数据

**Phase 1 止损点**:
- 如果 EXP-02 在调试 1 周后仍不优于 EXP-01: 考虑更换架构（B2/B3）或重新审视问题定义

**Phase 2 止损点**:
- 如果弹性任务在调试 1 周后仍无改善: 放弃弹性任务，只发布 Stage A 模型

---

## 9. 成功标准（量化）

### 9.1 Phase 1 成功标准

**必须达成**（否则停止）:
- ✅ `is_metal` AUROC >= 0.75
- ✅ `band_gap` MAE < 1.0 eV
- ✅ 训练稳定，无崩溃

**期望达成**（验证方向正确）:
- ✅ `is_metal` AUROC >= 0.85
- ✅ `is_stable` AUROC >= 0.80
- ✅ `band_gap` MAE < 0.5 eV
- ✅ Graph 在 50% 任务上优于 Composition

**理想达成**（超出预期）:
- ✅ `band_gap` MAE < 0.3 eV
- ✅ Graph 在 70% 任务上优于 Composition
- ✅ 所有分类任务 AUROC >= 0.85

### 9.2 Phase 2 成功标准

**必须达成**:
- ✅ Stage A 任务不劣化 > 5%
- ✅ 弹性任务 MAE 优于均值预测

**期望达成**:
- ✅ `bulk_modulus_vrh` MAE 降低 >= 15%
- ✅ `shear_modulus_vrh` MAE 降低 >= 15%
- ✅ `homogeneous_poisson` MAE 降低 >= 10%

**理想达成**:
- ✅ 弹性任务 MAE 降低 >= 25%
- ✅ Stage A 任务略有提升（多任务学习的正迁移）

---

## 10. 实施清单

### 10.1 Phase 0 清单（2-3 天）

**环境准备**:
- [ ] 安装 pytest, tensorboard
- [ ] 验证 Python 环境和 GPU

**数据质量分析**:
- [ ] 创建 `scripts/analyze_data_quality.py`
- [ ] 运行分析，生成报告
- [ ] 检查弹性数据覆盖率和异常值
- [ ] 检查 ChemSys-OOD 信息泄漏

**归一化验证**:
- [ ] 运行 `pytest tests/test_normalization.py`
- [ ] 创建 `scripts/visualize_normalization.py`
- [ ] 绘制归一化前后分布图

**监控基础设施**:
- [ ] 修改 `trainer.py` 添加 tensorboard 集成
- [ ] 添加 per-task loss 记录
- [ ] 添加梯度范数监控
- [ ] 添加学习率曲线

**Sanity Check**:
- [ ] 运行 EXP-00（tiny split, 2 epochs）
- [ ] 验证训练流程端到端可用

### 10.2 Phase 1 清单（5-7 天）

**CompositionBackbone 决策**:
- [ ] 决定使用简单版本还是增强版本

**EXP-01**:
- [ ] 运行 Composition baseline（50 epochs）
- [ ] 分析结果，记录到 experiment_log.md
- [ ] 如果 AUROC < 0.70，考虑增强

**EXP-02**:
- [ ] 运行 Graph baseline（50 epochs）
- [ ] 在 20 epochs 时检查 Gate-1
- [ ] 分析结果，对比 EXP-01

**Phase 1 总结**:
- [ ] 创建 `reports/phase1_summary.md`
- [ ] 绘制对比图
- [ ] 决定是否进入 Phase 2

### 10.3 Phase 2 清单（10-14 天）

**EXP-03**:
- [ ] 运行 Stage B（80 epochs）
- [ ] 在 40 epochs 时检查 Gate-2
- [ ] 分析弹性任务学习情况

**EXP-04**:
- [ ] 设计超参数搜索空间
- [ ] 运行 3-5 次实验
- [ ] 分析结果，找到最佳配置

**Phase 2 总结**:
- [ ] 创建 `reports/phase2_summary.md`
- [ ] 创建 `reports/best_model_card.md`
- [ ] 绘制超参数影响图

### 10.4 Phase 3 清单（可选，2-3 天）

**OOD 评估**:
- [ ] 运行 ChemSys-OOD 评估
- [ ] （可选）运行 Complexity-OOD 评估
- [ ] 分析泛化能力

---

## 11. 与 v4 的主要差异

| 维度 | v4 | v5 |
|---|---|---|
| **总时间估算** | 10 天 | 21-23 天（3-4 周） |
| **GPU 时间** | 100-150h | 180h |
| **Phase 0** | 0.5 天 | 3 天（增加数据分析和监控） |
| **调试预留** | 无 | 50% 缓冲时间 |
| **失败场景** | 2 个 Gate | 每个 Phase 都有详细失败场景 |
| **监控** | P1（可选） | P0（必须） |
| **数据质量分析** | 提及但未实施 | P0（必须） |
| **CompositionBackbone** | 直接使用 | 有增强选项 |
| **止损点** | 无 | 每个 Phase 都有明确止损点 |

---

## 12. 关键文件清单

### 12.1 需要创建的文件

**分析脚本**:
- `scripts/analyze_data_quality.py`: 数据质量分析
- `scripts/visualize_normalization.py`: 归一化可视化

**报告文件**:
- `reports/data_quality_report.md`: 数据质量报告
- `reports/phase1_summary.md`: Phase 1 总结
- `reports/phase2_summary.md`: Phase 2 总结
- `reports/best_model_card.md`: 最佳模型卡片

**可视化**:
- `reports/figures/elastic_distribution.png`
- `reports/figures/split_balance.png`
- `reports/figures/phase1_comparison.png`
- `reports/figures/hyperparameter_heatmap.png`

### 12.2 需要修改的文件

**训练基础设施**:
- `src/mp_data_pipeline/training/trainer.py`: 添加 tensorboard 集成

**可选增强**:
- `src/mp_data_pipeline/models/backbones.py`: 增强 CompositionBackbone（如果需要）

---

## 13. 验证计划

### 13.1 Phase 0 验证

**数据质量**:
```bash
python scripts/analyze_data_quality.py
# 检查输出报告，确认弹性数据 >= 10,000，异常值 < 5%
```

**归一化**:
```bash
python -m pytest tests/test_normalization.py -v
python scripts/visualize_normalization.py
# 检查所有测试通过，分布图合理
```

**监控**:
```bash
python scripts/train_multitask.py --split data/splits/split_iid_tiny.json --epochs 2
tensorboard --logdir artifacts/runs/<run_id>/tensorboard
# 检查 tensorboard 显示所有曲线
```

### 13.2 Phase 1 验证

**EXP-01 验证**:
```bash
# 检查 artifacts/runs/<exp01>/metrics/best_summary.json
# 确认 is_metal AUROC >= 0.75, band_gap MAE < 1.0
```

**EXP-02 验证**:
```bash
# 对比 EXP-01 vs EXP-02
python scripts/compare_experiments.py --exp1 <exp01> --exp2 <exp02>
# 确认至少 50% 任务有改善
```

### 13.3 Phase 2 验证

**EXP-03 验证**:
```bash
# 检查弹性任务是否学习
# 对比 Stage A 任务是否劣化
```

**最终模型验证**:
```bash
python scripts/eval_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --checkpoint artifacts/runs/<best>/checkpoints/best.pt
# 确认所有指标符合预期
```

---

## 14. 风险评估

### 14.1 高风险项（概率 > 30%）

1. **弹性任务无法学习**（概率 40%）
   - 影响: Phase 2 失败
   - 缓解: 提前做数据质量分析，准备回退方案

2. **Graph 不优于 Composition**（概率 35%）
   - 影响: Phase 1 延长
   - 缓解: 准备增强 GraphBackbone 的方案

3. **训练时间超出预期**（概率 50%）
   - 影响: 整体延期
   - 缓解: 已预留 50% 缓冲时间

### 14.2 中风险项（概率 10-30%）

4. **OOM 错误**（概率 25%）
   - 影响: 需要调整配置
   - 缓解: 梯度累积、减小 batch_size

5. **数据质量差**（概率 20%）
   - 影响: 需要数据清洗
   - 缓解: Phase 0 提前检查

### 14.3 低风险项（概率 < 10%）

6. **代码 bug**（概率 10%）
   - 影响: 调试时间
   - 缓解: Sanity check 提前发现

---

## 15. 总结

### 15.1 v5 的核心改进

1. **现实主义**: 时间估算从 10 天修正为 3-4 周
2. **风险管理**: 每个 Phase 都有详细的失败场景和回退方案
3. **质量优先**: Phase 0 强制进行数据质量分析和基础设施验证
4. **可观测性**: 监控提升到 P0，确保问题可诊断

### 15.2 关键原则

- **悲观估算，超额交付**: 预留 50% 缓冲时间
- **快速失败**: 每个 Phase 都有 Gate，及时止损
- **记录一切**: 详细记录每次实验，便于复盘
- **渐进式**: 从简单到复杂，逐步验证

### 15.3 预期成果

**Phase 1 结束时**:
- 建立可靠的 baseline
- 验证图结构的价值
- 识别模型的基本能力

**Phase 2 结束时**:
- 验证弹性任务可学习（或确认无法学习）
- 找到最佳超参数
- 产出可发布的模型

**整体完成时**:
- 一个经过充分验证的多任务模型
- 详细的实验报告和分析
- 清晰的后续改进方向

---

## 16. 参考文献

保留 v3/v4 的参考文献列表。

---

## 17. 变更记录

- **v5.0**（2026-03-03）: 基于严格审查，修正 v4 的过度乐观估算
  - 时间估算从 10 天修正为 3-4 周
  - 增加 Phase 0（基础设施验证）
  - 补充详细的失败场景和回退方案
  - 监控和数据质量分析提升到 P0
  - 预留 50% 调试缓冲时间
  - 增加止损点和风险评估

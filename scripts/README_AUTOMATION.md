# 实验自动化脚本使用指南

本目录包含完整的实验自动化脚本，用于执行 v5 计划中的所有阶段。

## 快速开始

### 运行完整流程（Phase 0 → Phase 1 → Phase 2）

```bash
python scripts/run_full_pipeline.py
```

这将自动执行：
1. Phase 0: 基础设施验证（数据质量分析、归一化测试）
2. Phase 1: Baseline 建立（EXP-01 Composition + EXP-02 Graph）
3. Phase 2: 弹性任务与超参数优化

**预计时间**: 3-4 周（单 GPU）
**GPU 时间**: ~180 GPU-hours

### 运行单个阶段

```bash
# 只运行 Phase 0
python scripts/run_full_pipeline.py --phase 0

# 只运行 Phase 1
python scripts/run_full_pipeline.py --phase 1

# 只运行 Phase 2
python scripts/run_full_pipeline.py --phase 2

# 跳过 Phase 0（如果已完成）
python scripts/run_full_pipeline.py --skip-phase0
```

## 脚本说明

### 核心自动化脚本

1. **run_full_pipeline.py** - 主控脚本
   - 协调所有阶段的执行
   - 提供命令行参数控制

2. **phase1_automation.py** - Phase 1 自动化
   - 等待 EXP-01 完成
   - 自动启动 EXP-02
   - 生成对比报告

3. **phase2_automation.py** - Phase 2 自动化
   - 运行 Stage B baseline
   - 执行超参数搜索
   - 生成最终报告

### 工具脚本

4. **analyze_data_quality.py** - 数据质量分析
   - 弹性数据覆盖率统计
   - 异常值检测
   - Split 平衡性检查

5. **visualize_normalization.py** - 归一化可视化
   - 绘制归一化前后的分布图
   - 输出到 `reports/figures/`

6. **compare_experiments.py** - 实验对比
   ```bash
   python scripts/compare_experiments.py \
     --exp1 artifacts/runs/20260303_120000 \
     --exp2 artifacts/runs/20260303_140000 \
     --exp1-name "EXP-01" \
     --exp2-name "EXP-02"
   ```

7. **hyperparameter_search.py** - 超参数搜索
   ```bash
   python scripts/hyperparameter_search.py \
     --split data/splits/split_iid_seed42.json \
     --stage b \
     --backbone graph \
     --epochs 80 \
     --max-experiments 5
   ```

8. **generate_reports.py** - 报告生成
   ```bash
   # Phase 1 总结
   python scripts/generate_reports.py \
     --type phase1 \
     --exp01 artifacts/runs/exp01 \
     --exp02 artifacts/runs/exp02

   # Phase 2 总结
   python scripts/generate_reports.py \
     --type phase2 \
     --best-run artifacts/runs/best

   # 模型卡片
   python scripts/generate_reports.py \
     --type model-card \
     --best-run artifacts/runs/best
   ```

9. **monitor_training.py** - 训练监控
   ```bash
   python scripts/monitor_training.py
   ```
   实时显示训练进度（每10秒更新）

## 输出文件

### 报告文件（reports/）

- `phase1_summary.md` - Phase 1 总结报告
- `phase2_summary.md` - Phase 2 总结报告
- `best_model_card.md` - 最佳模型卡片
- `comparison_EXP-01_vs_EXP-02.md` - 实验对比报告
- `hyperparameter_search_report.md` - 超参数搜索报告
- `hyperparameter_search_results.json` - 超参数搜索详细结果

### 可视化（reports/figures/）

- `normalization_*.png` - 归一化分布图
- `elastic_distribution.png` - 弹性数据分布
- `split_balance.png` - Split 平衡性
- `comparison_*.png` - 实验对比图
- `hyperparameter_heatmap.png` - 超参数影响热图

### 训练输出（artifacts/runs/）

每个训练运行创建一个目录：`artifacts/runs/YYYYMMDD_HHMMSS/`

包含：
- `config.json` - 训练配置
- `checkpoints/best.pt` - 最佳模型权重
- `metrics/history.json` - 训练历史
- `metrics/best_summary.json` - 最佳模型指标
- `tensorboard/` - TensorBoard 日志

### 日志文件（logs/）

- `exp01_composition_baseline.log` - EXP-01 训练日志
- `exp02_graph_baseline.log` - EXP-02 训练日志
- `exp03_stage_b_baseline.log` - EXP-03 训练日志
- `hp_search_*.log` - 超参数搜索日志

## 监控训练

### 查看实时日志

```bash
tail -f logs/exp01_composition_baseline.log
```

### 使用 TensorBoard

```bash
tensorboard --logdir artifacts/runs/
```

然后访问 http://localhost:6006

### 使用监控脚本

```bash
python scripts/monitor_training.py
```

## 故障排除

### 训练卡住不动

**症状**: 进程运行但无输出，GPU 未使用

**原因**: 数据集加载阶段（构建图结构）

**解决**: 耐心等待，数据集加载需要 5-10 分钟（123k 材料）

### OOM 错误

**症状**: CUDA out of memory

**解决方案**:
```bash
# 减小 batch size
python scripts/train_multitask.py ... --batch-size 16

# 或减小 hidden dim
python scripts/train_multitask.py ... --hidden-dim 128
```

### 训练太慢

**症状**: 每个 epoch 需要很长时间

**解决方案**:
- 检查 GPU 是否被使用：`nvidia-smi`
- 减少 epochs：`--epochs 20`
- 使用更小的 split 进行测试

### Phase 1 失败

**症状**: EXP-01 或 EXP-02 未达到成功标准

**解决方案**:
1. 检查日志文件查看错误
2. 查看 TensorBoard 确认训练是否收敛
3. 调整学习率或模型容量
4. 参考 v5 计划中的失败场景和回退方案

## 时间估算

| 阶段 | 任务 | GPU 时间 | 实际时间（单GPU） |
|------|------|----------|-------------------|
| Phase 0 | 基础设施验证 | 2h | 0.5 天 |
| Phase 1 | EXP-01 (Composition) | 12-18h | 1-2 天 |
| Phase 1 | EXP-02 (Graph) | 18-27h | 2-3 天 |
| Phase 2 | EXP-03 (Stage B) | 24-36h | 3-4 天 |
| Phase 2 | 超参数搜索 (5次) | 100h | 10-12 天 |
| **总计** | | **180h** | **3-4 周** |

## 成功标准

### Phase 1

**必须达成**:
- is_metal AUROC >= 0.75
- band_gap MAE < 1.0 eV
- Graph 在 50% 任务上优于 Composition

**期望达成**:
- is_metal AUROC >= 0.85
- band_gap MAE < 0.5 eV
- Graph 在 70% 任务上优于 Composition

### Phase 2

**必须达成**:
- Stage A 任务不劣化 > 5%
- 弹性任务 MAE 优于均值预测

**期望达成**:
- 弹性任务 MAE 降低 >= 15%
- Stage A 任务略有提升

## 下一步

完成所有阶段后：

1. 查看 `reports/phase2_summary.md` 了解最终结果
2. 查看 `reports/best_model_card.md` 了解最佳模型
3. （可选）运行 Phase 3 进行 OOD 评估
4. 使用最佳模型进行预测

## 参考

- 完整计划: `reports/plans/2026-03-03_multitask_model_execution_plan_v5.md`
- 实验日志: `reports/plans/experiment_log.md`
- 项目文档: `CLAUDE.md`

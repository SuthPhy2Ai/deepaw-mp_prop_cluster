# EXP-204: DeePAW Replace Fusion

## 概述

这个实验测试一个新的 DeePAW 集成策略：**直接替换**原子 embedding，而不是融合。

## 动机

之前的 EXP-201/202/203 使用了"融合"策略：
- EXP-201: `node_emb = atom_emb + deepaw_proj` (Add)
- EXP-202: `node_emb = fusion_proj(cat([atom_emb, deepaw_proj]))` (Concat)

**问题**：
1. 两个分支竞争：`atom_emb` 和 `deepaw_proj` 可能互相干扰
2. 参数冗余：两个分支都在学习原子表示
3. 训练不稳定：优化目标不清晰

## 新策略：Replace

```python
# 直接用 DeePAW 特征替换 atom embedding
if use_deepaw:
    node_emb = deepaw_proj(deepaw_extractor(...))  # 只有这一个分支
else:
    node_emb = atom_emb(z)
```

**优势**：
- ✅ 没有竞争：只有一个信息源
- ✅ 参数高效：减少 3-16% 参数
- ✅ 训练稳定：优化目标清晰（学习如何压缩 3200→256）
- ✅ 更像标准的预训练-微调范式

## 配置

- **Backbone**: enhanced_graph
- **DeePAW Fusion**: replace (新模式)
- **Cutoff**: 6.0 Å
- **Max Neighbors**: 24
- **n_rbf**: 64 (和 Baseline 一致)
- **Batch Size**: 64
- **Learning Rate**: 1e-4 (和 Baseline 一致)
- **Epochs**: 50
- **AMP**: Disabled (--no-amp)
- **PyG**: Enabled (--use-pyg)

**注意**：超参数已调整为和 Baseline 一致，以获得最佳性能。

## 预期结果

如果 Replace 策略有效，应该看到：
1. 训练更稳定（loss 曲线更平滑）
2. 性能优于 EXP-201/202（Add/Concat）
3. 可能接近或超过 Baseline（如果 DeePAW 特征确实有用）

## 对比基准

| 实验 | 策略 | Val Loss | Band Gap MAE |
|------|------|----------|--------------|
| Baseline | 无 DeePAW | 0.9781 | 0.2308 eV |
| EXP-201 | Add | 1.0739 | 0.2581 eV |
| EXP-202 | Concat | 1.0430 | 0.2558 eV |
| EXP-204 | **Replace** | ? | ? |

## 运行

```bash
bash experiments/stage_a/phase2_deepaw/exp204_deepaw_replace/train.sh
```

## 输出

结果保存到：`artifacts/runs_exp204/`

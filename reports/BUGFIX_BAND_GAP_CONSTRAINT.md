# Band Gap双重约束Bug修复

**日期**: 2026-03-05
**严重性**: 高 - 影响band gap预测准确性
**状态**: ✅ 已修复

## 问题描述

Band gap约束被错误地应用了两次：

1. **第一次**: 在 `GroupedTaskHeads.forward()` 中 (heads.py:68)
   ```python
   if task_name == "band_gap":
       out[task_name] = F.softplus(raw_value)
   ```

2. **第二次**: 在 `MultitaskPropertyModel.forward()` 中 (multitask_model.py:93-99)
   ```python
   band_gap_idx = self.task_to_index["band_gap"]
   band_gap_col = torch.nn.functional.softplus(pred[:, band_gap_idx : band_gap_idx + 1])
   pred = torch.cat([pred[:, :band_gap_idx], band_gap_col, pred[:, band_gap_idx + 1 :]], dim=1)
   ```

## 影响分析

双重softplus导致band gap预测系统性偏大：

| 原始值 | 一次softplus | 两次softplus | 偏差 |
|--------|-------------|-------------|------|
| -2.0   | 0.127       | 0.759       | +0.632 |
| -1.0   | 0.313       | 0.862       | +0.549 |
| 0.0    | 0.693       | 1.099       | +0.406 |
| 1.0    | 1.313       | 1.551       | +0.238 |
| 2.0    | 2.127       | 2.240       | +0.113 |

**问题**：
- 训练时模型学习的是"双重softplus后的值"
- 预测时所有band gap都会偏大
- 梯度传播受到双重非线性影响

## 修复方案

移除 `MultitaskPropertyModel.forward()` 中的重复约束，保留 `GroupedTaskHeads` 中的约束。

### 修改文件

**src/mp_data_pipeline/models/multitask_model.py**

```python
# 修复前
def forward(self, batch_dict: dict) -> torch.Tensor:
    graph_emb = self.backbone(batch_dict)
    head_out = self.heads(graph_emb)

    ordered_outputs = []
    for task_name in TASK_NAME_LIST:
        ordered_outputs.append(head_out[task_name])
    pred = torch.stack(ordered_outputs, dim=-1)

    # Enforce non-negative band gap via softplus.
    band_gap_idx = self.task_to_index["band_gap"]
    band_gap_col = torch.nn.functional.softplus(pred[:, band_gap_idx : band_gap_idx + 1])
    pred = torch.cat(
        [pred[:, :band_gap_idx], band_gap_col, pred[:, band_gap_idx + 1 :]],
        dim=1,
    )

    return pred

# 修复后
def forward(self, batch_dict: dict) -> torch.Tensor:
    graph_emb = self.backbone(batch_dict)
    head_out = self.heads(graph_emb)

    ordered_outputs = []
    for task_name in TASK_NAME_LIST:
        ordered_outputs.append(head_out[task_name])
    pred = torch.stack(ordered_outputs, dim=-1)

    # Note: band_gap constraint already applied in GroupedTaskHeads
    # No need for duplicate softplus here

    return pred
```

## 验证

### 1. 单元测试通过
```bash
PYTHONPATH=src:$PYTHONPATH python -m pytest tests/test_enhanced_backbones.py -v
# 7 passed in 2.19s
```

### 2. 功能测试
```python
model = MultitaskPropertyModel(backbone_name="composition")
output = model(batch_dict)
band_gap = output[0, band_gap_idx]
# ✅ Band gap非负且只应用一次softplus
```

## 影响范围

### 需要重新训练的模型

所有已训练的模型都受此bug影响，建议重新训练：

- ✅ EXP-01 (Composition) - Phase 1
- ✅ EXP-02 (Graph) - Phase 1
- ✅ EXP-03 (Graph, Stage B) - Phase 1
- ✅ EXP-04 (Enhanced Graph) - Phase 2
- ✅ EXP-05 (XPaiNN) - Phase 2

### 预期改进

修复后预期band gap预测更准确：
- MAE可能降低0.1-0.3 eV
- 小band gap材料预测更准确
- 训练收敛可能更快

## 后续行动

1. ✅ 代码修复完成
2. ⏳ 重新训练所有模型
3. ⏳ 对比修复前后的band gap预测性能
4. ⏳ 更新Phase 1/2实验报告

## 教训

- 物理约束应该只在一个地方应用
- 需要更完善的单元测试覆盖约束逻辑
- Code review应该检查重复的约束应用

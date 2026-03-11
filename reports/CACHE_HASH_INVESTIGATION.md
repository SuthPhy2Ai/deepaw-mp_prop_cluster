# Cache Hash 问题完整调查报告

**日期**: 2026-03-05  
**调查目的**: 为训练做准备，彻底解决图缓存hash不匹配问题

---

## 问题总结

训练和评估脚本**无法使用预计算的图缓存**，导致每次都要on-the-fly计算图结构，严重影响性能。

---

## 根本原因

### Cache Key 计算不一致

**precompute_graphs.py** (生成缓存):
```python
# Line 89-91
cache_key = md5(f"{db_path}_{len(all_mp_ids)}_{cutoff}_{max_neighbors}")
# 包含: db路径 + 样本总数 + cutoff + max_neighbors
```

**dataset.py** (查找缓存):
```python
# Line 67-69
cache_key = md5(f"{db_path}_{cutoff}_{max_neighbors}")
# 包含: db路径 + cutoff + max_neighbors
# 缺少: 样本总数 ❌
```

### 实际Hash值

| 来源 | Hash | 状态 |
|------|------|------|
| precompute_graphs.py 生成 | `a942a9ec54a42f623c0159fa815ca563` | ✅ 文件存在 (265MB) |
| dataset.py 查找 | `cc750d893c4f189a544347615f59bd0b` | ❌ 文件不存在 |

**参数**:
- db_path: `/scratch/sutianhao/data/mp-data-pipeline/data/db/mp_materials.db`
- cutoff: `6.0`
- max_neighbors: `24`
- total_samples: `154879`

---

## 影响分析

### 1. 训练性能影响

**EXP-05 训练日志分析**:
- ✅ Masks缓存: 正常加载 (`masks_f80d2e5eabd37bdce654a1e1feadeeaa.pkl`)
- ❌ 图缓存: 未加载（没有 "Loading graph cache" 消息）
- 结果: 每个batch耗时 2-3秒（包含on-the-fly图计算）

**性能损失估算**:
- 每个epoch: 3872 batches × 2.5s = ~2.7小时
- 如果使用缓存: 预计可减少30-50%时间

### 2. 现有缓存状态

```
data/cache/graphs_a942a9ec54a42f623c0159fa815ca563.pkl
- 大小: 265MB
- 包含: 30,000个图（不是全部154,879个）
- 参数: cutoff=6.0, max_neighbors=24
```

**注意**: 缓存不完整！只有30k/154k个图。

---

## 解决方案对比

### 方案1: 修改 dataset.py (推荐 ⭐)

**改动**:
```python
# dataset.py line 67-69
cache_key = hashlib.md5(
    f"{self.db_path}_{len(self.mp_ids)}_{self.cutoff}_{self.max_neighbors}".encode()
).hexdigest()
```

**优点**:
- ✅ 与precompute_graphs.py逻辑一致
- ✅ 不同split可以有独立缓存
- ✅ 更精确的缓存匹配

**缺点**:
- ⚠️ 需要在`__init__`中先设置`self.mp_ids`再调用`_load_graph_cache()`
- ⚠️ 现有缓存需要重新生成（因为只有30k个图）

### 方案2: 修改 precompute_graphs.py

**改动**:
```python
# precompute_graphs.py line 89-91
cache_key = hashlib.md5(
    f"{args.db}_{args.cutoff}_{args.max_neighbors}".encode()
).hexdigest()
```

**优点**:
- ✅ 简单直接
- ✅ 一个缓存可被所有split共享

**缺点**:
- ⚠️ 如果不同split需要不同的图集合，会有问题
- ⚠️ 现有缓存需要重命名或重新生成

### 方案3: 符号链接（临时方案）

**操作**:
```bash
cd data/cache
ln -s graphs_a942a9ec54a42f623c0159fa815ca563.pkl \
      graphs_cc750d893c4f189a544347615f59bd0b.pkl
```

**优点**:
- ✅ 立即生效，无需改代码
- ✅ 可以快速验证缓存是否有效

**缺点**:
- ❌ 治标不治本
- ❌ 每次参数变化都要手动创建
- ❌ 缓存不完整（只有30k图）

---

## 推荐行动计划

### 阶段1: 立即修复（方案2 + 重新生成完整缓存）

1. **修改 precompute_graphs.py** 移除样本总数
   - 理由: 图结构只依赖于cutoff和max_neighbors，与样本数量无关
   - 好处: 一个缓存文件可以服务所有split

2. **重新生成完整缓存**
   ```bash
   python scripts/precompute_graphs.py \
     --cutoff 6.0 \
     --max-neighbors 24 \
     --workers 16
   ```
   - 预计时间: ~30分钟（154k个结构）
   - 生成hash: `cc750d893c4f189a544347615f59bd0b`

3. **验证缓存加载**
   - 重新运行训练，检查日志中是否有 "Loading graph cache"
   - 观察batch时间是否显著减少

### 阶段2: 长期优化（可选）

- 考虑实现增量缓存更新机制
- 添加缓存版本控制
- 实现缓存完整性检查

---

## 验证清单

- [ ] 修改 precompute_graphs.py 的cache key计算
- [ ] 删除旧的不完整缓存
- [ ] 重新生成完整的图缓存
- [ ] 验证训练脚本能加载缓存
- [ ] 验证评估脚本能加载缓存
- [ ] 测量性能提升（batch时间对比）

---

## 附录: 为什么现有缓存只有30k个图？

需要检查 `precompute_graphs.py` 的历史运行记录，可能原因：
1. 进程被中断（OOM或手动停止）
2. 某些结构计算失败被跳过
3. 使用了子集而非完整数据集

建议: 重新生成完整缓存以确保覆盖所有样本。

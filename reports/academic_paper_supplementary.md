# 补充材料：多任务图神经网络材料性质预测

**Supplementary Materials: Multi-Task Graph Neural Networks for Materials Property Prediction**

---

## S1. 详细性能表格

### S1.1 完整任务性能对比

| 任务 | 类型 | 训练MAE/AUROC | 验证MAE/AUROC | Val/Train | 等级 |
|------|------|---------------|---------------|-----------|------|
| is_metal | 分类 | 0.9056 | 0.9029 | 0.998 | ⭐ |
| is_stable | 分类 | 0.8448 | 0.8389 | 0.993 | ✅ |
| band_gap | 回归 | 0.5431 eV | 0.5402 eV | 0.995 | ✅ |
| vbm | 回归 | 0.4790 eV | 0.4770 eV | 0.996 | ✅ |
| cbm | 回归 | 0.5911 eV | 0.5940 eV | 1.005 | ✅ |
| efermi | 回归 | 0.6196 eV | 0.6186 eV | 0.998 | ✅ |
| formation_energy | 回归 | 0.2108 eV | 0.2150 eV | 1.020 | ✅ |
| energy_above_hull | 回归 | 0.1195 eV | 0.1225 eV | 1.025 | ✅ |
| energy_per_atom | 回归 | 1.6699 eV | 1.7711 eV | 1.061 | ⚠️ |
| bulk_modulus_vrh | 回归 | 3.27 GPa | 8.35 GPa | 2.555 | ⚠️ |
| shear_modulus_vrh | 回归 | 4.93 GPa | 10.38 GPa | 2.106 | ⚠️ |
| homogeneous_poisson | 回归 | 0.0308 | 0.0423 | 1.373 | ✅ |
| universal_anisotropy | 回归 | 1.41 | 1.82 | 1.291 | ✅ |

### S1.2 训练历史（每5个epoch）

| Epoch | 训练损失 | 验证损失 | 学习率 | 时间(s) |
|-------|---------|---------|--------|---------|
| 1 | 224.91 | 117.24 | 2.0e-5 | 40 |
| 5 | 65.27 | 47.62 | 1.0e-4 | 40 |
| 10 | 52.04 | 41.85 | 1.0e-4 | 39 |
| 15 | 46.23 | 37.92 | 1.0e-4 | 39 |
| 20 | 40.66 | 32.85 | 1.0e-4 | 39 |
| 25 | 36.84 | 30.47 | 1.0e-4 | 39 |
| 30 | 32.05 | 29.16 | 1.0e-4 | 39 |
| 35 | 27.18 | 27.83 | 1.0e-4 | 39 |
| 40 | 22.49 | 26.60 | 1.0e-4 | 39 |
| 42 | 21.63 | 26.56 | 1.0e-4 | 39 |
| 45 | 20.87 | 26.62 | 1.0e-4 | 39 |
| 50 | 19.72 | 26.68 | 1.0e-4 | 39 |

---

## S2. 模型架构细节

### S2.1 网络层配置

**原子嵌入层**：
- 输入：原子序数 (0-100)
- 输出：256维向量
- 参数量：25,600

**消息传递层（×6）**：
- 消息MLP：(256×2 + 64) → 256 → 256
- 更新MLP：(256×2) → 256 → 256
- LayerNorm：256维
- 每层参数量：约400K
- 总参数量：约2.4M

**任务头（×5组）**：
- 热力学头：256 → 256 → 3
- 电子头：256 → 256 → 5
- 稳定性头：256 → 256 → 1
- 弹性头：256 → 256 → 4
- 总参数量：约100K

**总参数量**：约2.5M

### S2.2 激活函数选择

所有隐藏层使用SiLU激活函数：
$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

优势：
- 平滑且非单调
- 梯度性质优于ReLU
- 在深度网络中表现稳定

---

## S3. 数据处理细节

### S3.1 图构建算法

```python
def build_graph(atoms, cutoff=6.0, max_neighbors=24):
    # 1. 计算周期性邻居列表
    i, j, d = neighbor_list("ijd", atoms, cutoff)
    
    # 2. 对每个原子，保留最近的max_neighbors个邻居
    edge_index = []
    edge_dist = []
    for atom_idx in range(len(atoms)):
        neighbors = j[i == atom_idx]
        distances = d[i == atom_idx]
        
        # 按距离排序
        sorted_idx = np.argsort(distances)
        top_k = sorted_idx[:max_neighbors]
        
        edge_index.append(np.stack([
            np.full(len(top_k), atom_idx),
            neighbors[top_k]
        ]))
        edge_dist.append(distances[top_k])
    
    return np.concatenate(edge_index, axis=1), np.concatenate(edge_dist)
```

### S3.2 RBF展开实现

```python
def rbf_expansion(distances, n_rbf=64, cutoff=6.0):
    # 高斯中心均匀分布
    centers = np.linspace(0, cutoff, n_rbf)
    gamma = 10.0 / cutoff
    
    # 计算RBF特征
    rbf = np.exp(-gamma * (distances[:, None] - centers[None, :]) ** 2)
    return rbf
```

---


## S4. 任务权重计算

基于数据覆盖率的任务权重：

| 任务 | 覆盖率 | 原始权重 | 归一化权重 |
|------|--------|----------|------------|
| energy_per_atom | 100% | 1.000 | 0.543 |
| formation_energy | 100% | 1.000 | 0.543 |
| energy_above_hull | 100% | 1.000 | 0.543 |
| band_gap | 100% | 1.000 | 0.543 |
| efermi | 100% | 1.000 | 0.543 |
| is_metal | 100% | 1.000 | 0.543 |
| is_stable | 100% | 1.000 | 0.543 |
| cbm | 23.2% | 2.076 | 0.714 |
| vbm | 23.2% | 2.076 | 0.714 |
| bulk_modulus_vrh | 7.1% | 3.752 | 1.889 |
| shear_modulus_vrh | 7.1% | 3.752 | 1.941 |
| homogeneous_poisson | 7.1% | 3.752 | 1.934 |
| universal_anisotropy | 7.1% | 3.752 | 2.007 |

---

## S5. 代码和数据

### S5.1 代码仓库结构

```
mp-data-pipeline/
├── src/mp_data_pipeline/
│   ├── models/
│   │   ├── backbones.py          # 图骨干网络
│   │   ├── heads.py               # 任务头
│   │   └── multitask_model.py    # 多任务模型
│   ├── ml/
│   │   ├── dataset.py            # 数据集
│   │   ├── tasks.py              # 任务定义
│   │   └── splits.py             # 数据划分
│   └── training/
│       ├── trainer.py            # 训练器
│       ├── losses.py             # 损失函数
│       └── sampler.py            # 采样器
├── scripts/
│   ├── train_multitask.py        # 训练脚本
│   └── eval_multitask.py         # 评估脚本
└── data/
    ├── db/mp_materials.db        # ASE数据库
    └── splits/split_iid_seed42.json
```

### S5.2 训练命令

```bash
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage b \
  --backbone graph \
  --batch-size 64 \
  --num-workers 8 \
  --epochs 50 \
  --lr 1e-4 \
  --weight-decay 1e-5 \
  --grad-clip 1.0 \
  --warmup-epochs 5 \
  --oversample-elastic 4.0
```

---

## S6. 图表说明

### 图1：任务性能对比
展示16个任务的训练集和验证集性能对比，使用柱状图表示MAE/AUROC。

### 图2：任务组概览
按5个任务组（热力学、电子、稳定性、弹性）展示平均性能。

### 图3：训练曲线
展示50个epoch的训练损失和验证损失变化曲线。

### 图4：训练-验证对比
散点图展示各任务的训练性能vs验证性能，对角线表示完美泛化。

### 图5：ROC曲线
展示is_metal和is_stable两个分类任务的ROC曲线和AUROC值。

### 图6：预测-真值散点图
展示6个关键回归任务的预测值vs真实值散点图，包含MAE、RMSE和R²统计。

---

**补充材料完成日期**：2026年3月10日


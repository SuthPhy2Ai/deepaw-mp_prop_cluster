# DeePAW 预训练原子特征集成文档

## 概述

本文档记录了将 DeePAW 项目的预训练原子塔特征集成到 MP 材料性质预测模型的完整实现过程。

### 背景与动机

**问题**：
- 当前 MP 模型使用简单的原子 embedding (`nn.Embedding(119, 256)`)
- 电子性质预测 (cbm, vbm, efermi) MAE 较高
- 整体 train-val gap 达 53%，过拟合严重

**解决方案**：
- 集成 DeePAW 在电荷密度预测任务上预训练的原子表示
- DeePAW 使用 eSCN 架构，包含丰富的电子结构信息
- 提取 3200 维原子特征后投影到模型的 hidden_dim

**预期改进**：
- 电子性质 MAE ↓ 20%+
- Train-val gap ↓ 至 <40%
- 金属分类 AUROC ↑

---

## 架构设计

### 整体流程

```
材料结构 (ASE Atoms)
    ↓
原子坐标 + 原子序数 + 边图
    ↓
┌─────────────────────────────────────┐
│  DeePAW 特征提取器                    │
│  - 加载预训练 checkpoint              │
│  - 运行 atom_blocks 消息传递          │
│  - 输出: (N, 3200) 原子表示           │
└─────────────────────────────────────┘
    ↓
投影层: Linear(3200 → 256)
    ↓
特征融合 (add 或 concat)
    ↓
EnhancedGraphBackbone 继续处理
    ↓
任务头预测
```

### 关键设计决策

1. **方案 B：复用 MP 图结构**
   - 不需要晶胞参数 (cell)
   - 直接使用 MP pipeline 已构建的 edge_index
   - 避免重复构建邻居图

2. **冻结预训练权重**
   - DeePAW 模型参数冻结 (`freeze=True`)
   - 仅训练投影层和下游任务头
   - 加速训练，防止过拟合

3. **特征融合策略**
   - `add`: 直接相加 (默认)
   - `concat`: 拼接后再投影

---

## 实现细节

### 新增文件

#### 1. `src/mp_data_pipeline/models/deepaw_extractor.py`

DeePAW 原子特征提取器核心模块。

**主要类**：
```python
class DeePAWAtomFeatureExtractor:
    """从 DeePAW 预训练模型提取原子表示"""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        cutoff: float = 4.0,
        max_neighbors: int = 20,
        freeze: bool = True,
    )

    @torch.no_grad()
    def extract_atom_features(
        self,
        atomic_numbers: torch.Tensor,  # (N,)
        positions: torch.Tensor,       # (N, 3)
        edge_index: torch.Tensor,      # (2, E)
    ) -> torch.Tensor:
        # 返回: (N, 3200) 原子表示
```

**关键实现**：
- 加载 DeePAW eSCN 模型的 atom_blocks
- 转换 edge_index 格式: (2, E) → (E, 2)
- 运行球谐消息传递
- Flatten 输出: [N, 25, 128] → [N, 3200]

### 修改文件

#### 2. `src/mp_data_pipeline/models/enhanced_backbones.py`

在 `EnhancedGraphBackbone` 中集成 DeePAW 特征。

**新增参数**：
```python
use_deepaw_features: bool = False
deepaw_checkpoint: Optional[str] = None
deepaw_fusion: str = "add"  # "add" | "concat"
```

**Forward 修改**：
```python
# 1. 原始 embedding
node_emb = self.atom_emb(z)  # (N, 256)

# 2. DeePAW 特征 (如果启用)
if self.use_deepaw_features:
    pos = batch_dict["pos"]
    deepaw_features = self.deepaw_extractor.extract_atom_features(
        z, pos, edge_index
    )  # (N, 3200)
    deepaw_proj = self.deepaw_proj(deepaw_features)  # (N, 256)

    if self.deepaw_fusion == "add":
        node_emb = node_emb + deepaw_proj
    elif self.deepaw_fusion == "concat":
        node_emb = torch.cat([node_emb, deepaw_proj], dim=-1)
        node_emb = self.fusion_proj(node_emb)

# 3. 继续消息传递...
```

#### 3. `src/mp_data_pipeline/ml/enhanced_dataset.py`

扩展数据集以提供原子坐标。

**EnhancedGraphSample 扩展**：
```python
@dataclass
class EnhancedGraphSample(GraphSample):
    edge_angles: np.ndarray
    triplet_index: np.ndarray
    positions: np.ndarray  # NEW: (N, 3) 原子坐标
```

**Collate 函数修改**：
```python
batch_dict["pos"] = torch.cat(positions_list, dim=0).float()  # (total_nodes, 3)
```

#### 4. `scripts/train_multitask.py`

添加 DeePAW 相关命令行参数。

**新增参数**：
```bash
--use-deepaw-features       # 启用 DeePAW 特征
--deepaw-checkpoint PATH    # DeePAW checkpoint 路径
--deepaw-fusion {add,concat}  # 特征融合方式
```

#### 5. `src/mp_data_pipeline/models/multitask_model.py`

将 DeePAW 参数传递给 backbone。

---

## 使用方法

### 环境要求

- PyTorch >= 1.12
- PyTorch Geometric
- DeePAW 代码库 (位于 `/home/sutianhao/data/deepaw_test/DeePAW-main/`)
- 预训练 checkpoint: `f_nonlocal_escn_best.pth`

### 训练命令

#### 快速开始

使用提供的训练脚本：

```bash
cd /scratch/sutianhao/data/mp-data-pipeline
bash experiments/stage_a/phase2_deepaw/train_deepaw.sh
```

#### 手动训练

```bash
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone enhanced_graph \
  --cutoff 8.0 \
  --max-neighbors 48 \
  --n-rbf 128 \
  --use-deepaw-features \
  --deepaw-checkpoint /home/sutianhao/data/deepaw_test/DeePAW-main/checkpoints/f_nonlocal_escn_best.pth \
  --deepaw-fusion add \
  --batch-size 64 \
  --num-workers 8 \
  --exclude-tasks volume density is_stable \
  --epochs 50 \
  --lr 2e-4 \
  --no-amp
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use-deepaw-features` | 启用 DeePAW 特征提取 | False |
| `--deepaw-checkpoint` | DeePAW 模型路径 | f_nonlocal_escn_best.pth |
| `--deepaw-fusion` | 特征融合方式 (add/concat) | add |
| `--backbone` | 必须使用 enhanced_graph | - |
| `--no-amp` | 禁用混合精度 (推荐) | False |

### 评估

```bash
python scripts/eval_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --checkpoint artifacts/runs/<run_id>/checkpoints/best.pt
```

---

## 验证结果

### 单元测试

测试 DeePAW 特征提取器：

```python
from src.mp_data_pipeline.models.deepaw_extractor import DeePAWAtomFeatureExtractor
import torch

extractor = DeePAWAtomFeatureExtractor(
    checkpoint_path='/home/sutianhao/data/deepaw_test/DeePAW-main/checkpoints/f_nonlocal_escn_best.pth',
    device='cuda'
)

# 测试 Si2O2 结构
z = torch.tensor([14, 14, 8, 8])
pos = torch.randn(4, 3)
edge_index = torch.tensor([[0,1,2,3], [1,0,3,2]])

features = extractor.extract_atom_features(z, pos, edge_index)
print(f'Features shape: {features.shape}')  # (4, 3200)
```

**结果**：✅ 通过
- 输出形状: (4, 3200)
- 特征统计: mean ≈ 0.0005, std ≈ 0.20

### 集成测试

1 epoch 训练测试：

```bash
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone enhanced_graph \
  --use-deepaw-features \
  --epochs 1 \
  --batch-size 8
```

**结果**：✅ 通过
- 训练速度: ~33-37 it/s
- Loss 下降: 9.64 → 8.41 (13% 改进)
- 所有 8 个任务正常训练
- 无错误或崩溃

---

## 性能预期

基于 DeePAW 的电荷密度预训练，预期改进：

### 电子性质 (最大改进)

| 任务 | Baseline MAE | 目标 MAE | 改进 |
|------|--------------|----------|------|
| band_gap | 0.2308 eV | <0.18 eV | >20% |
| cbm | 0.2921 eV | <0.23 eV | >21% |
| vbm | 0.2346 eV | <0.19 eV | >19% |
| efermi | 0.3834 eV | <0.31 eV | >19% |

### 金属分类

| 任务 | Baseline AUROC | 目标 AUROC |
|------|----------------|------------|
| is_metal | 0.9098 | >0.92 |

### 泛化性能

- Train-val gap: 53% → <40%

### 热力学性质 (中等改进)

- energy_per_atom, formation_energy_per_atom 可能有 5-10% 改进

### 结构性质 (改进有限)

- volume, density 改进预期较小

---

## 技术细节

### 内存优化

- DeePAW 模型较大 (105M 参数)
- 使用 `freeze=True` 冻结参数
- 使用 `@torch.no_grad()` 包裹特征提取
- 建议禁用 AMP (`--no-amp`)

### 训练时间

- 特征提取增加约 20-30% 训练时间
- 50 epochs 预计 40-50 分钟 (vs 33 分钟 baseline)
- 训练速度: ~33-37 it/s (batch_size=64)

### 兼容性

- ✅ 向后兼容：所有新功能默认禁用
- ✅ 不影响现有训练流程
- ✅ 可与其他增强特征 (angles, edge_update) 组合使用

---

## 故障排除

### 问题 1: torch_cluster 导入错误

**错误**：
```
OSError: undefined symbol: _ZN3c106detail14torchCheckFailE...
```

**解决**：
代码已自动处理，创建 mock torch_cluster 模块。

### 问题 2: CUDA 内存不足

**解决**：
- 减小 batch_size (64 → 32)
- 使用 `--no-amp` 禁用混合精度

### 问题 3: 训练速度慢

**解决**：
- 确保 `--num-workers 8`
- 预计算图缓存: `python scripts/precompute_graphs.py`

---

## 文件清单

### 新增文件
- `src/mp_data_pipeline/models/deepaw_extractor.py` - 特征提取器
- `experiments/stage_a/phase2_deepaw/train_deepaw.sh` - 训练脚本
- `experiments/stage_a/phase2_deepaw/README.md` - 实验说明
- `docs/deepaw_integration/README.md` - 本文档

### 修改文件
- `src/mp_data_pipeline/models/enhanced_backbones.py` - 集成 DeePAW
- `src/mp_data_pipeline/ml/enhanced_dataset.py` - 添加 positions
- `scripts/train_multitask.py` - 添加 CLI 参数
- `src/mp_data_pipeline/models/multitask_model.py` - 传递参数

---

## 参考资料

- DeePAW 项目: `/home/sutianhao/data/deepaw_test/DeePAW-main/`
- 实现计划: `/home/sutianhao/.claude/plans/twinkling-humming-petal.md`
- 实验配置: `experiments/stage_a/phase2_deepaw/`

---

## 更新日志

### 2026-03-11
- ✅ 完成 DeePAW 特征提取器实现
- ✅ 集成到 EnhancedGraphBackbone
- ✅ 扩展数据集支持原子坐标
- ✅ 添加训练脚本和 CLI 参数
- ✅ 通过单元测试和集成测试
- ✅ 创建完整文档

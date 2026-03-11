# DeePAW 原子塔预训练集成方案

## 背景

DeePAW项目使用双塔架构预测电荷密度：
- **原子塔 (Atom Tower)**: 学习原子表示
- **体素塔 (Probe Tower)**: 预测探针点电荷密度

我们希望使用预训练的原子塔来改进MP材料性质预测模型的初始embedding。

---

## DeePAW 原子塔架构

### e3nn版本 (f_nonlocal.py)

```python
class AtomicConfigurationModel(nn.Module):
    def __init__(
        self,
        num_interactions=3,      # 消息传递层数
        num_neighbors=20,        # 最大邻居数
        mul=500,                 # irreps倍数
        lmax=4,                  # 最大角动量
        cutoff=4.0,              # 截断半径 (Å)
        basis="gaussian",        # 径向基函数
        num_basis=10,            # 基函数数量
    ):
        # 原子embedding: one-hot编码
        self.num_species = 118  # 所有元素
        nodes = torch.nn.functional.one_hot(nodes, num_classes=118)

        # E3等变消息传递
        self.interaction_block = InteractionBlock(...)
```

**参数量**: ~1.9M

**输出**:
- `atom_representation`: List of node embeddings from each layer
- 每层输出维度由irreps决定 (复杂的球谐表示)

### eSCN版本 (f_nonlocal_escn.py)

```python
class F_nonlocal_escn(nn.Module):
    def __init__(
        self,
        num_layers=3,            # 原子消息传递层数
        sphere_channels=128,     # 每个球谐分量的通道数
        lmax=4,                  # 最大球谐度
        mmax=2,                  # SO(2)卷积最大阶
        cutoff=4.0,              # 截断半径
        num_neighbors=20,        # 邻居数
    ):
        # 原子embedding: 原子序数 → 128维
        self.atom_embedding = nn.Embedding(118, sphere_channels)

        # 球谐表示: [N_atoms, (lmax+1)^2, C]
        # (lmax+1)^2 = 25 个球谐基函数
        x = torch.zeros(num_atoms, 25, 128)
        x[:, 0, :] = self.atom_embedding(nodes)  # 仅L=0分量

        # SO(2)卷积消息传递
        for block in self.atom_blocks:
            x = block(x, ...)

        atom_repr = x  # [N_atoms, 25, 128]
```

**参数量**: ~27M (更强大)

**输出**:
- `atom_repr`: [N_atoms, 25, 128] 球谐通道表示
- 可以flatten为 [N_atoms, 3200] 向量

---

## 集成方案

### 方案1: 提取原子表示作为额外特征

**思路**: 使用预训练原子塔生成原子表示，作为额外特征输入到MP模型

```python
# 1. 加载预训练DeePAW模型
from deepaw.models.escn import F_nonlocal_escn
deepaw_model = F_nonlocal_escn()
deepaw_model.load_state_dict(torch.load('checkpoints/f_nonlocal_escn_best.pth'))
deepaw_model.eval()

# 2. 提取原子表示
def extract_atom_features(structure):
    """从DeePAW提取原子特征"""
    # 构建输入字典 (需要适配DeePAW的输入格式)
    input_dict = build_deepaw_input(structure)

    with torch.no_grad():
        # 仅运行原子塔
        atom_xyz = unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        nodes = unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])

        # 初始embedding
        x = torch.zeros(num_atoms, 25, 128)
        x[:, 0, :] = deepaw_model.atom_embedding(nodes)

        # 消息传递
        for block in deepaw_model.atom_blocks:
            x = block(x, ...)

        # Flatten: [N_atoms, 25*128=3200]
        atom_features = x.reshape(num_atoms, -1)

    return atom_features

# 3. 在MP模型中使用
class EnhancedGraphBackbone(nn.Module):
    def __init__(self, hidden_dim=256, use_deepaw_features=True):
        super().__init__()

        # 原始原子embedding
        self.atom_emb = nn.Embedding(119, hidden_dim)

        # DeePAW特征投影
        if use_deepaw_features:
            self.deepaw_proj = nn.Linear(3200, hidden_dim)

    def forward(self, batch_dict):
        z = batch_dict["z"]

        # 原始embedding
        node_emb = self.atom_emb(z)

        # 添加DeePAW特征
        if hasattr(self, 'deepaw_proj'):
            deepaw_feat = extract_atom_features(batch_dict)
            node_emb = node_emb + self.deepaw_proj(deepaw_feat)

        # 继续消息传递...
```

**优点**:
- 利用预训练知识
- 保持MP模型架构不变
- 可以fine-tune或freeze DeePAW特征

**缺点**:
- 需要适配输入格式
- 增加计算开销
- DeePAW训练在电荷密度任务，可能与材料性质预测不完全匹配

---

### 方案2: 迁移原子Embedding层

**思路**: 直接使用DeePAW的原子embedding初始化MP模型

```python
# 1. 加载DeePAW embedding权重
deepaw_model = F_nonlocal_escn()
deepaw_model.load_state_dict(torch.load('checkpoints/f_nonlocal_escn_best.pth'))

# 2. 提取embedding权重
deepaw_atom_emb = deepaw_model.atom_embedding.weight  # [118, 128]

# 3. 初始化MP模型
mp_model = EnhancedGraphBackbone(hidden_dim=256)

# 4. 迁移embedding (需要维度匹配)
if mp_model.atom_emb.weight.shape[1] == 128:
    # 直接复制
    mp_model.atom_emb.weight.data[:118] = deepaw_atom_emb
elif mp_model.atom_emb.weight.shape[1] == 256:
    # 投影或重复
    mp_model.atom_emb.weight.data[:118, :128] = deepaw_atom_emb
    mp_model.atom_emb.weight.data[:118, 128:] = deepaw_atom_emb  # 重复
```

**优点**:
- 简单直接
- 无额外计算开销
- 可以fine-tune

**缺点**:
- 维度可能不匹配 (DeePAW: 128, MP: 256)
- 仅迁移初始embedding，未利用消息传递层

---

### 方案3: 知识蒸馏

**思路**: 使用DeePAW原子表示作为teacher，训练MP模型学习相似表示

```python
# 1. 提取DeePAW原子表示 (teacher)
teacher_features = extract_atom_features(structure)  # [N_atoms, 3200]

# 2. MP模型预测 (student)
student_features = mp_model.get_atom_features(batch_dict)  # [N_atoms, 256]

# 3. 蒸馏损失
distill_loss = F.mse_loss(
    student_proj(student_features),  # 投影到3200维
    teacher_features.detach()
)

# 4. 总损失
total_loss = task_loss + alpha * distill_loss
```

**优点**:
- 灵活，不受维度限制
- 可以选择性蒸馏
- 保持MP模型轻量

**缺点**:
- 需要额外训练
- 超参数调优 (alpha)

---

## 实现建议

### 推荐方案: 方案1 (提取特征)

**原因**:
1. DeePAW在电荷密度任务上预训练，原子表示包含丰富的电子结构信息
2. MP任务 (band gap, formation energy) 与电子结构高度相关
3. 可以freeze DeePAW特征，仅训练投影层

### 实现步骤

1. **适配输入格式**
   - DeePAW需要: `atom_xyz`, `atom_edges`, `atom_edges_displacement`, `cell`
   - MP已有: PyG Data格式
   - 需要转换函数

2. **提取原子特征**
   - 运行DeePAW原子塔 (不运行probe塔)
   - 保存原子表示

3. **集成到MP模型**
   - 添加特征投影层
   - 与原始embedding相加或拼接

4. **训练策略**
   - **阶段1**: Freeze DeePAW，仅训练投影层和下游任务头 (5 epochs)
   - **阶段2**: Fine-tune全部参数 (45 epochs)

---

## 代码示例

见 `scripts/integrate_deepaw_features.py` (待实现)

---

## 预期改进

基于DeePAW的电荷密度预训练，预期在以下任务上有提升：

- ✅ **band_gap**: 直接相关电子结构
- ✅ **cbm/vbm**: 导带/价带位置
- ✅ **efermi**: 费米能级
- ✅ **is_metal**: 金属性分类
- ⚠️ **formation_energy**: 间接相关
- ⚠️ **elastic properties**: 关联较弱

**目标**:
- 电子性质 MAE 降低 20%+
- 减少过拟合 (train-val gap < 40%)

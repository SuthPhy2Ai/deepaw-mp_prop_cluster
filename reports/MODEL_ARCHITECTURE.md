# 多任务材料性质预测模型架构

## 整体架构概览

```
输入数据 (晶体结构)
    ↓
┌─────────────────────────────────────────┐
│         Backbone (特征提取器)            │
│  ┌────────────────────────────────────┐ │
│  │ 选项1: CompositionBackbone         │ │
│  │ 选项2: GraphBackbone               │ │
│  │ 选项3: EnhancedGraphBackbone       │ │
│  │ 选项4: XPaiNNBackbone (E(3)等变)   │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
    ↓
图级别嵌入 (num_graphs, hidden_dim=256)
    ↓
┌─────────────────────────────────────────┐
│      GroupedTaskHeads (任务头)          │
│  ┌────────────────────────────────────┐ │
│  │ Thermo Head    → 3个热力学任务     │ │
│  │ Electronic Head → 5个电子性质任务  │ │
│  │ Stability Head  → 1个稳定性任务    │ │
│  │ Structure Head  → 2个结构任务      │ │
│  │ Elastic Head    → 4个弹性任务      │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
    ↓
18个任务预测输出
```

---

## 1. Backbone 架构详解

### 1.1 CompositionBackbone (组成特征)

**设计理念**: 仅使用元素组成信息，不考虑晶体结构

```
原子序数 (z)
    ↓
Embedding(100+1, hidden_dim=256)
    ↓
节点嵌入 (num_atoms, 256)
    ↓
Mean Pooling (按batch聚合)
    ↓
LayerNorm
    ↓
图级别嵌入 (num_graphs, 256)
```

**特点**:
- 最简单的baseline
- 参数量: ~25K (仅embedding层)
- Phase 1实验中表现出色 (赢得9/11任务)

---

### 1.2 GraphBackbone (图神经网络)

**设计理念**: SchNet风格的消息传递，考虑原子间距离

```
输入: z, edge_index, edge_dist, batch

原子嵌入层
    z → Embedding(100+1, 256) → node_emb

边特征提取
    edge_dist → RBF(n_rbf=64, cutoff=6.0Å) → rbf_feat
    rbf_feat → MLP(64→256) → edge_feat

消息传递 (6层)
    每层:
    ┌─────────────────────────────────────┐
    │ MessagePassingLayer                 │
    │  1. 构造消息:                       │
    │     [h_src, h_dst, edge_feat]       │
    │     → MLP → messages                │
    │  2. 聚合消息:                       │
    │     messages → index_add(dst) → agg │
    │  3. 更新节点:                       │
    │     [h_node, agg] → MLP → update    │
    │     h_node = LayerNorm(h_node + update) │
    └─────────────────────────────────────┘

Mean Pooling
    node_emb → graph_emb (num_graphs, 256)

输出归一化
    graph_emb → LayerNorm → output
```

**参数配置**:
- hidden_dim: 256
- n_layers: 6
- n_rbf: 64
- cutoff: 6.0Å
- max_neighbors: 24

**特点**:
- Phase 1中表现不佳 (仅赢得2/11任务)
- 需要禁用AMP才能稳定训练
- 参数量: ~1.5M

---

### 1.3 EnhancedGraphBackbone (增强图网络)

**设计理念**: 改进GraphBackbone的局限性

```
输入: z, edge_index, edge_dist, batch

原子嵌入
    z → Embedding(119, 256) → node_emb

增强边特征
    edge_dist → EnhancedRBF(n_rbf=128, cutoff=8.0Å) → edge_feat

    EnhancedRBF细节:
    - 更多基函数: 64 → 128
    - 更大截断半径: 6.0Å → 8.0Å
    - 高斯RBF: exp(-γ(d - c_i)²)

可选: 角度特征 (三体相互作用)
    if use_angles:
        triplets → AngleExpansion(n_angle_basis=32)
        → 聚合到边特征

增强消息传递 (6层)
    每层:
    ┌─────────────────────────────────────┐
    │ EnhancedMessagePassingLayer         │
    │  1. 消息计算:                       │
    │     [h_src, h_dst, edge_feat]       │
    │     → MLP → messages                │
    │  2. 可选边更新:                     │
    │     if use_edge_update:             │
    │       [edge_feat, messages]         │
    │       → MLP → edge_feat'            │
    │  3. 消息聚合 (AMP兼容):             │
    │     agg = zeros_like(node_emb,      │
    │                      dtype=messages.dtype) │
    │     agg.index_add_(dst, messages)   │
    │  4. 节点更新:                       │
    │     [h_node, agg] → MLP → update    │
    │     h_node = LayerNorm(h_node + update) │
    └─────────────────────────────────────┘

Mean Pooling + 输出MLP
    node_emb → mean_pool → MLP(256→256) → output
```

**参数配置**:
- hidden_dim: 256
- n_layers: 6
- n_rbf: 128 (↑ from 64)
- cutoff: 8.0Å (↑ from 6.0Å)
- max_neighbors: 48 (↑ from 24)
- use_angles: False/True
- use_edge_update: False/True

**改进点**:
1. **更大图覆盖**: 8.0Å截断 + 48邻居
2. **更丰富边特征**: 128个RBF基函数
3. **三体相互作用**: 可选角度特征
4. **边特征更新**: 可选边更新机制
5. **AMP兼容**: 修复dtype不匹配问题

**参数量**: ~2.5M (取决于配置)

---

### 1.4 XPaiNNBackbone (E(3)等变网络)

**设计理念**: 基于e3nn的E(3)等变消息传递

```
输入: z (或at_no), pos, edge_index, batch

嵌入层 (XEmbedding)
    ┌─────────────────────────────────────┐
    │ 1. 节点嵌入:                        │
    │    at_no → Embedding(120, node_dim) │
    │    → x_scalar                       │
    │                                     │
    │ 2. 边向量和距离:                    │
    │    vec = pos[src] - pos[dst]        │
    │    dist = ||vec||                   │
    │                                     │
    │ 3. 径向基函数:                      │
    │    dist → SphericalBesselj0(num_rbf)│
    │    → rbf                            │
    │                                     │
    │ 4. 截断函数:                        │
    │    dist → CosineCutoff → fcut       │
    │                                     │
    │ 5. 球谐函数:                        │
    │    vec → SphericalHarmonics         │
    │    → rsh (edge_irreps维度)          │
    └─────────────────────────────────────┘

初始化球形特征
    x_spherical = zeros(num_atoms, edge_irreps.dim)

消息传递 (num_interactions=3轮)
    每轮包含两个步骤:

    步骤1: XPaiNNMessage
    ┌─────────────────────────────────────┐
    │ 输入: x_scalar, x_spherical, rbf,   │
    │       fcut, rsh, edge_index         │
    │                                     │
    │ 1. 标量消息:                        │
    │    x_scalar → MLP → scalar_out      │
    │    filter = RBF_MLP(rbf) * fcut     │
    │    filter_out = scalar_out * filter │
    │                                     │
    │ 2. 分割门控和消息:                  │
    │    filter_out → split →             │
    │      [gate_state, gate_edge, msg_s] │
    │                                     │
    │ 3. 球形消息:                        │
    │    msg_v = TensorProduct(           │
    │      x_spherical[dst], gate_state)  │
    │    edge_v = TensorProduct(          │
    │      rsh, gate_edge)                │
    │    msg_v = msg_v + edge_v           │
    │                                     │
    │ 4. 聚合 (AMP兼容):                  │
    │    x_scalar += index_add(msg_s)     │
    │    x_spherical += index_add(msg_v)  │
    └─────────────────────────────────────┘

    步骤2: XPaiNNUpdate
    ┌─────────────────────────────────────┐
    │ 输入: x_scalar, x_spherical         │
    │                                     │
    │ 1. 球形变换:                        │
    │    U = Linear_U(x_spherical)        │
    │    V = Linear_V(x_spherical)        │
    │                                     │
    │ 2. 提取不变量:                      │
    │    V_inv = Norm(V)                  │
    │                                     │
    │ 3. MLP处理:                         │
    │    [x_scalar, V_inv] → MLP →        │
    │      [a_vv, a_sv, a_ss]             │
    │                                     │
    │ 4. 更新球形特征:                    │
    │    d_v = TensorProduct(U, a_vv)     │
    │    x_spherical += d_v               │
    │                                     │
    │ 5. 更新标量特征:                    │
    │    inner = EquivariantDot(U, V)     │
    │    d_s = a_sv * inner + a_ss        │
    │    x_scalar += d_s                  │
    └─────────────────────────────────────┘

全局池化
    x_scalar → scatter(mean, batch) → x_pooled

输出投影
    x_pooled → Linear(node_dim→node_dim) → output
```

**参数配置**:
- node_dim: 128
- edge_irreps: "128x0e + 64x1o + 32x2e"
  - 128个标量 (l=0, 偶宇称)
  - 64个向量 (l=1, 奇宇称)
  - 32个二阶张量 (l=2, 偶宇称)
- num_interactions: 3
- num_rbf: 20
- cutoff: 6.0Å

**特点**:
1. **E(3)等变性**: 对旋转和平移保持等变
2. **球谐函数**: 捕获角度信息
3. **标量+球形特征**: 双通道信息流
4. **物理约束**: 自然满足对称性

**参数量**: ~3-4M (取决于edge_irreps配置)

**Phase 2实验状态**:
- 已完成3个epoch训练
- 遇到并修复了多个bug (输入格式、AMP dtype等)
- 训练稳定，loss正常下降

---

## 2. Task Heads 架构

### 2.1 任务分组

```
18个任务分为5组:

1. Thermodynamic (3个回归任务)
   - energy_per_atom
   - formation_energy_per_atom
   - energy_above_hull

2. Electronic (5个任务: 4回归 + 1分类)
   - band_gap (回归, 物理约束: ≥0)
   - cbm (回归)
   - vbm (回归)
   - efermi (回归)
   - is_metal (分类)

3. Stability (1个分类任务)
   - is_stable

4. Structure (2个回归任务)
   - volume (物理约束: >0)
   - density (物理约束: >0)

5. Elastic (4个回归任务)
   - bulk_modulus_vrh (物理约束: >0)
   - shear_modulus_vrh (物理约束: >0)
   - homogeneous_poisson (物理约束: -1≤ν≤0.5)
   - universal_anisotropy (物理约束: ≥0)
```

### 2.2 GroupedTaskHeads 结构

```
输入: graph_emb (num_graphs, hidden_dim=256)

┌─────────────────────────────────────────────┐
│ Thermo Head                                 │
│   MLPHead(256 → 256 → 3)                    │
│   输出: [energy_per_atom, formation_energy, │
│          energy_above_hull]                 │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Electronic Head                             │
│   MLPHead(256 → 256 → 5)                    │
│   物理约束:                                 │
│   - band_gap: softplus(raw) 确保≥0          │
│   输出: [band_gap, cbm, vbm, efermi,        │
│          is_metal_logit]                    │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Stability Head                              │
│   MLPHead(256 → 128 → 1)                    │
│   输出: [is_stable_logit]                   │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Structure Head                              │
│   MLPHead(256 → 128 → 2)                    │
│   物理约束:                                 │
│   - volume: softplus(raw) + 1.0 (最小1Ų)    │
│   - density: softplus(raw) + 0.1 (最小0.1)  │
│   输出: [volume, density]                   │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Elastic Head                                │
│   MLPHead(256 → 256 → 4)                    │
│   物理约束:                                 │
│   - moduli: softplus(raw) 确保>0            │
│   - poisson: sigmoid(raw)*1.5 - 1.0         │
│     确保-1≤ν≤0.5                            │
│   - anisotropy: softplus(raw) 确保≥0        │
│   输出: [bulk_modulus, shear_modulus,       │
│          poisson, anisotropy]               │
└─────────────────────────────────────────────┘
```

### 2.3 MLPHead 基础结构

```
MLPHead(in_dim, hidden_dim, out_dim, dropout=0.1)

输入 (in_dim)
    ↓
Linear(in_dim → hidden_dim)
    ↓
SiLU激活
    ↓
Dropout(p=0.1)
    ↓
Linear(hidden_dim → out_dim)
    ↓
输出 (out_dim)
```

**参数量统计**:
- Thermo: (256×256 + 256×3) ≈ 66K
- Electronic: (256×256 + 256×5) ≈ 67K
- Stability: (256×128 + 128×1) ≈ 33K
- Structure: (256×128 + 128×2) ≈ 33K
- Elastic: (256×256 + 256×4) ≈ 66K
- **总计**: ~265K

---

## 3. 数据流详解

### 3.1 训练时数据流

```
1. 数据加载
   ASE Database
       ↓
   GraphSample(mp_id, atomic_numbers, positions, ...)
       ↓
   Collate Function
       ↓
   Batch Dict: {
       'z': (num_atoms,),
       'edge_index': (2, num_edges),
       'edge_dist': (num_edges,),
       'batch': (num_atoms,),
       'pos': (num_atoms, 3),  # XPaiNN需要
   }

2. 前向传播
   Batch Dict
       ↓
   Backbone.forward(batch_dict)
       ↓
   graph_emb (num_graphs, 256)
       ↓
   GroupedTaskHeads.forward(graph_emb)
       ↓
   task_dict: {
       'energy_per_atom': (num_graphs,),
       'band_gap': (num_graphs,),
       ...
   }
       ↓
   Stack to tensor (num_graphs, 18)

3. 损失计算
   predictions (num_graphs, 18)
   targets (num_graphs, 18)
   masks (num_graphs, 18)  # 标记哪些任务有标签
       ↓
   Per-task loss (masked)
       ↓
   Weighted sum → total_loss
```

### 3.2 图构建流程

```
晶体结构 (ASE Atoms)
    ↓
ASE neighbor_list(cutoff, max_neighbors)
    ↓
边列表: [(i, j, dist, vec), ...]
    ↓
PyG格式:
    edge_index: [[src...], [dst...]]
    edge_dist: [dist...]
    edge_vec: [[dx, dy, dz], ...]  # XPaiNN需要
```

**图构建参数**:
- GraphBackbone: cutoff=6.0Å, max_neighbors=24
- EnhancedGraphBackbone: cutoff=8.0Å, max_neighbors=48
- XPaiNNBackbone: cutoff=6.0Å (从edge_index推断)

---

## 4. 模型合理性分析

### 4.1 架构优势

✅ **模块化设计**
- Backbone和Heads解耦，易于替换和实验
- 支持4种不同复杂度的backbone

✅ **物理约束**
- Band gap ≥ 0
- Volume, density > 0
- Poisson ratio ∈ [-1, 0.5]
- 弹性模量 > 0

✅ **任务分组**
- 相关任务共享head参数
- 减少参数量，提高泛化能力

✅ **渐进式增强**
- Composition → Graph → Enhanced → XPaiNN
- 可以逐步验证每个改进的效果

### 4.2 潜在问题

✅ **~~重复的band_gap约束~~** - 已修复 (2026-03-05)
- ~~GroupedTaskHeads中已有softplus~~
- ~~MultitaskPropertyModel中又做了一次softplus~~
- **修复**: 已移除MultitaskPropertyModel中的重复约束
- 详见: [BUGFIX_BAND_GAP_CONSTRAINT.md](BUGFIX_BAND_GAP_CONSTRAINT.md)

⚠️ **~~Head维度不一致~~** - 已优化 (2026-03-05)
- ~~Thermo/Electronic/Elastic: hidden_dim=256~~
- ~~Stability/Structure: hidden_dim=128~~
- **优化**: 已统一所有head为256维
- 详见: [OPTIMIZATION_HEAD_AND_ANGLES.md](OPTIMIZATION_HEAD_AND_ANGLES.md)

⚠️ **XPaiNN输出维度不匹配**
- XPaiNN输出: node_dim=128
- 其他backbone输出: hidden_dim=256
- **当前解决**: out_proj将128投影到256
- **建议**: 统一node_dim=256或调整head输入

⚠️ **~~角度特征未实现~~** - 已完成 (2026-03-05)
- ~~EnhancedGraphBackbone支持角度特征~~
- ~~但forward中TODO未实现聚合逻辑~~
- **完成**: 已实现完整的角度特征聚合到边
- 向后兼容，不提供角度数据时自动退化
- 详见: [OPTIMIZATION_HEAD_AND_ANGLES.md](OPTIMIZATION_HEAD_AND_ANGLES.md)

⚠️ **缺少残差连接**
- GraphBackbone有残差: `h + update`
- EnhancedGraphBackbone有残差
- XPaiNN有残差
- 但Heads中没有残差连接
- **建议**: 考虑在Heads中添加skip connection

### 4.3 性能瓶颈

🔍 **图构建开销**
- 每个epoch都需要on-the-fly构建图
- 解决方案: 预计算图并缓存 (已实现)

🔍 **Mask计算开销**
- 训练初始化时需要遍历所有样本
- 解决方案: 并行计算 (已实现)

🔍 **XPaiNN计算复杂度**
- 球谐函数和张量积计算密集
- 建议: 减少num_interactions或edge_irreps维度

---

## 5. 改进建议

### 5.1 短期改进 (Phase 2)

1. **修复重复约束**
   - 移除MultitaskPropertyModel中的band_gap softplus

2. **统一Head维度**
   - 所有head使用hidden_dim=256

3. **完成角度特征**
   - 实现EnhancedGraphBackbone中的角度聚合
   - 或移除未使用的AngleExpansion

4. **统一输出维度**
   - XPaiNN使用node_dim=256

### 5.2 中期改进 (Phase 3)

1. **Attention机制**
   - 在message passing中引入attention权重
   - 学习重要的原子对

2. **残差Heads**
   - 在MLPHead中添加skip connection
   - 提高梯度流动

3. **自适应池化**
   - 除了mean pooling，尝试attention pooling
   - 学习重要原子的权重

4. **多尺度特征**
   - 在不同层提取特征并concat
   - 类似于FPN的思想

### 5.3 长期改进 (Phase 4+)

1. **专用Elastic Head**
   - 直接输出6×6弹性张量
   - 保证对称性和正定性

2. **原子能量基准**
   - 添加per-element reference energy
   - 提高能量预测精度

3. **不确定性估计**
   - 添加evidential或ensemble方法
   - 输出预测置信度

4. **预训练策略**
   - 在大规模数据上预训练backbone
   - Fine-tune on specific tasks

---

## 6. 总结

当前模型架构整体**合理且模块化**，具有以下特点:

**优势**:
- 清晰的模块划分
- 物理约束嵌入
- 渐进式复杂度
- 良好的扩展性

**需要改进**:
- 修复重复约束
- 统一维度设计
- 完成未实现功能
- 优化计算效率

**实验验证**:
- Phase 1: Composition baseline强于Graph
- Phase 2: XPaiNN训练稳定，待评估性能
- 建议: 系统性ablation study验证每个改进

模型架构为后续优化提供了良好的基础。

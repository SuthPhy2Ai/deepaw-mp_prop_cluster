# 环境配置指南

本文档记录了项目的完整环境配置信息。

## 系统信息

- **操作系统**: Linux 5.14.0-284.30.1.el9_2.x86_64
- **用户**: sutianhao
- **项目路径**: `/scratch/sutianhao/data/mp-data-pipeline`
- **备用路径**: `/home/sutianhao/data/mp-data-pipeline`

## Python 环境

### 推荐：使用 Conda 环境

项目推荐使用以下 Conda 环境之一：

1. **deepaw** (推荐): `/home/sutianhao/miniforge3/envs/deepaw`
   - 包含 DeePAW 预训练特征支持
   - 适用于 Phase 2 DeePAW 集成实验

2. **deepaw_test**: `/home/sutianhao/.conda/envs/deepaw_test`
   - 测试环境

### 创建新环境

如果需要从头创建环境：

```bash
# 创建 conda 环境
conda create -n mp-data-pipeline python=3.10

# 激活环境
conda activate mp-data-pipeline

# 安装依赖
pip install -r requirements.txt

# 安装 PyTorch (根据你的 CUDA 版本)
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 PyTorch Geometric
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

## 核心依赖

### 必需依赖 (requirements.txt)

```
mp-api>=0.41          # Materials Project API 客户端
ase>=3.22             # Atomic Simulation Environment (数据库)
pymatgen>=2024.1      # 材料科学核心库
numpy                 # 数值计算
tqdm                  # 进度条
torch>=2.2            # PyTorch 深度学习框架
scikit-learn>=1.3     # 机器学习工具
```

### 额外依赖

```bash
# PyTorch Geometric (图神经网络)
torch-geometric
torch-scatter
torch-sparse
torch-cluster

# 可选：DeePAW 预训练特征
# 需要从 DeePAW 项目安装
```

## GPU 配置

项目需要 GPU 支持以进行高效训练：

```bash
# 检查 GPU 可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0)}')"
```

### 推荐配置

- **GPU 内存**: ≥16GB (用于 batch_size=64)
- **CUDA 版本**: 11.8 或 12.1
- **cuDNN**: 与 CUDA 版本匹配

## 数据目录

### 主要数据路径

```
/scratch/sutianhao/data/mp-data-pipeline/
├── data/
│   ├── raw/
│   │   └── summary_all_merged.jsonl.gz  # 预下载数据集 (280MB)
│   ├── db/
│   │   └── mp_materials.db              # ASE 数据库 (~2GB)
│   ├── splits/
│   │   ├── split_iid_seed42.json        # IID 划分
│   │   ├── split_chemsys_ood.json       # ChemSys OOD 划分
│   │   └── split_complexity_ood.json    # Complexity OOD 划分
│   ├── cache/                           # 图缓存 (自动生成)
│   └── pyg_cache/                       # PyG 图缓存 (自动生成)
│
├── artifacts/
│   ├── runs/                            # 训练输出
│   ├── runs_exp201/                     # exp201 输出
│   ├── runs_exp202/                     # exp202 输出
│   └── ...
│
└── experiments/                         # 实验组织
```

### 存储需求

- **原始数据**: ~280MB (JSONL.gz)
- **ASE 数据库**: ~2GB
- **图缓存**: ~5-10GB (首次生成后可复用)
- **训练输出**: ~100MB-1GB 每个实验
- **总计**: 建议预留 **50GB** 空间

## 环境变量

### 可选配置

```bash
# Materials Project API Key (仅 API 下载需要)
export MP_API_KEY="your_api_key_here"

# PyTorch 设置
export CUDA_VISIBLE_DEVICES=0           # 指定 GPU
export OMP_NUM_THREADS=8                # OpenMP 线程数

# 数据路径 (如果使用非默认路径)
export MP_DATA_ROOT="/scratch/sutianhao/data/mp-data-pipeline"
```

### 添加到 ~/.bashrc

```bash
# 添加到 ~/.bashrc 以持久化
echo 'export MP_DATA_ROOT="/scratch/sutianhao/data/mp-data-pipeline"' >> ~/.bashrc
source ~/.bashrc
```

## 验证安装

运行以下命令验证环境配置：

```bash
# 1. 检查 Python 版本
python --version  # 应该 >= 3.10

# 2. 检查核心依赖
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
python -c "import ase; print(f'ASE: {ase.__version__}')"
python -c "import pymatgen; print(f'Pymatgen: {pymatgen.__version__}')"

# 3. 检查 GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 4. 验证数据库
python scripts/validate.py

# 5. 运行快速测试
python -m pytest tests/ -v
```

## 常见问题

### 1. CUDA 版本不匹配

**问题**: `RuntimeError: CUDA error: no kernel image is available`

**解决**:
```bash
# 重新安装匹配的 PyTorch 版本
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. PyG 安装失败

**问题**: `torch-scatter` 等扩展安装失败

**解决**:
```bash
# 使用预编译的 wheel
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

### 3. 内存不足

**问题**: `CUDA out of memory`

**解决**:
```bash
# 减小 batch size
python scripts/train_multitask.py --batch-size 32  # 默认 64

# 或使用梯度累积
python scripts/train_multitask.py --batch-size 32 --accumulate-grad-batches 2
```

### 4. 数据库锁定

**问题**: `database is locked`

**解决**:
```bash
# 删除锁文件
rm data/db/mp_materials.db.lock
```

## 性能优化

### DataLoader 配置

```bash
# 使用多进程加载数据
python scripts/train_multitask.py --num-workers 8  # 根据 CPU 核心数调整

# 启用 pin_memory (GPU 训练)
# 已在代码中默认启用
```

### 图缓存

```bash
# 预计算图缓存 (首次运行)
python scripts/precompute_graphs.py

# 缓存位置
# - DGL: data/cache/
# - PyG: data/pyg_cache/
```

### AMP (自动混合精度)

```bash
# Stage A 可以使用 AMP
python scripts/train_multitask.py --stage a  # AMP 默认启用

# Stage B 必须禁用 AMP (dtype 问题)
python scripts/train_multitask.py --stage b --no-amp
```

## 更新日志

- **2026-03-14**: 创建环境配置文档
- **2026-03-11**: 添加 DeePAW 集成支持
- **2026-03-06**: Phase 1 基线完成
- **2026-03-03**: 项目初始化

## 相关文档

- [README.md](README.md) - 项目概述和快速开始
- [CLAUDE.md](CLAUDE.md) - Claude Code 项目指南
- [experiments/README.md](experiments/README.md) - 实验文档

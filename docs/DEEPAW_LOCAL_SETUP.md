# DeePAW（本地私有依赖）接入说明

本仓库 **不包含** DeePAW 源码（`DeePAW-main` 为私有/保密项目）。但你可以在本机通过“绝对路径 / 环境变量 / PYTHONPATH / editable install”的方式，让本仓库的代码在运行时导入 `deepaw`。

下文假设 DeePAW 项目目录类似：

```bash
/abs/path/to/DeePAW-main/
  deepaw/
  checkpoints/
  setup.py
  ...
```

## 方案 A（推荐）：editable install

在你用于训练/推理的 Python 环境里执行：

```bash
pip install -e /abs/path/to/DeePAW-main
```

验证：

```bash
python -c "import deepaw; print('deepaw import OK:', deepaw.__file__)"
```

优点：
- 最稳定；无需改 `PYTHONPATH`
- IDE/类型提示友好

注意：
- 需要 DeePAW 的依赖在当前环境里可用（如 `e3nn`、`ase` 等）

## 方案 B：仅通过 `PYTHONPATH` 指向 DeePAW 目录

适用于你不想在环境里安装 DeePAW，只想“运行时可 import”：

```bash
export DEEPAW_ROOT=/abs/path/to/DeePAW-main
export PYTHONPATH="$DEEPAW_ROOT:$PYTHONPATH"
```

验证：

```bash
python -c "import deepaw; print('deepaw import OK:', deepaw.__file__)"
```

你也可以把这两行写进 `~/.bashrc` 或者项目专用的 `env.sh`。

## 方案 C：用符号链接满足硬编码路径（不改代码时的兜底）

当前仓库里有一些脚本/模块可能会直接写死 DeePAW 的路径（例如 `src/mp_data_pipeline/models/deepaw_extractor.py` 里插入了一个固定路径到 `sys.path`）。

如果你暂时不想改这些实现，可以用软链接把“真实 DeePAW 位置”映射到代码期望的位置：

```bash
mkdir -p /home/sutianhao/data/deepaw_test
ln -s /abs/path/to/DeePAW-main /home/sutianhao/data/deepaw_test/DeePAW-main
```

验证：

```bash
python -c "import sys; sys.path.insert(0, '/home/sutianhao/data/deepaw_test/DeePAW-main'); import deepaw; print(deepaw.__file__)"
```

## 使用建议

- **不要**把 DeePAW 目录复制到本仓库里；也不要把 DeePAW 权重文件提交到 Git。
- 如果你需要在训练脚本里使用 DeePAW 的 checkpoint（例如 `checkpoints/f_nonlocal.pth`），建议只在本地通过绝对路径传入。
- 后续如果要做“用 DeePAW 原子塔生成 embedding”的集成，建议把 DeePAW 路径做成可配置项（例如优先读 `DEEPAW_ROOT`），避免硬编码。


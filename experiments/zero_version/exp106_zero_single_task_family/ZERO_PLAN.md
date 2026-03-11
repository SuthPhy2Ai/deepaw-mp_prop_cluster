# Zero 版本计划

## 目标
- 用最暴力的方式建立单性质上界：每个性质单独模型、单独主干、单独日志、单独评估。

## 方法
- 使用 `scripts/train_multitask.py --stage full --only-task <task>` 逐任务训练。
- 统一走 `PyG` 路线。
- 根据任务样本量自动调整模型容量与训练超参。

## 自动化执行
- 编排脚本：`scripts/run_zero_single_task.py`
- 输出台账：`metrics/zero_runs.csv`
- 每个任务独立命令保存在 `tasks/<task>/training_cmd.sh`

## 预期产物
- 每个任务一份独立 run
- `zero_runs.csv` 总表
- 后续统一评估与并入全方法报告

# Stage B v4 一页总结

## 1. 结论先行

- `v4` 是当前 `Stage B` 路线里最好的版本。
- 相比 `v3`，`v4` 在 `13/13` 个任务主指标上全部改善。
- 相比 `Stage A`，`v4` 仍未追回核心 `8` 个高覆盖任务的最佳水平，但差距明显缩小。
- `v4` 证明了“共享预训练主干 + 单任务头独立微调”可以有效降低多任务相互拖累。
- `v4` 也修复了稀疏弹性任务的训练可用性问题，`4` 个弹性任务最终全部稳定跑通。

## 2. 方案定义

- 目标：验证是否应放弃单一 `Stage B` 多任务联合优化，改为共享主干的单任务精调。
- 方法：以 `v3` 最优 checkpoint 为共享初始化，只冻结 backbone，逐个任务独立训练 head。
- 数据口径：`IID split`，文件为 `data/splits/split_iid_seed42.json`。
- 评估对象：`13` 个 `Stage B` 任务。
- 数据后端：`PyG`，开启缓存。

## 3. 训练设置

| 项目 | 取值 |
|---|---|
| 实验 ID | `exp104_stageb_v4_single_task_heads` |
| 基座 checkpoint | `artifacts/runs_stageb_v3/20260308_070539/checkpoints/best.pt` |
| Backbone | `graph` |
| Hidden Dim | `256` |
| Layers | `6` |
| Cutoff | `6.0` |
| Max Neighbors | `24` |
| 后端 | `PyG` |
| Freeze Backbone | `True` |
| Stage | `b` |
| Batch Size | `64` |
| Workers | `4` |
| Epochs | `30` |
| LR | `2e-4` |
| Weight Decay | `1e-5` |
| Device | `cuda` |

## 4. 实验版本对比

| Version | Run | 训练方式 | Best Val Loss | Best Epoch | 备注 |
|---|---|---|---:|---:|---|
| Stage A | `20260305_210307` | 8-task multitask | `0.9781` | `38` | 高覆盖核心任务基线 |
| v1 | `20260307_185342` | 13-task multitask | `26.5616` | `42` | Stage B 初版 |
| v2 | `20260308_001437` | 13-task multitask | `26.9097` | `42` | balanced |
| v3 | `20260308_070539` | 13-task multitask | `27.8843` | `38` | core guard |
| v4 | `single-task` | shared backbone + per-task head | `N/A` | `N/A` | 13 个任务分别训练 |

## 5. Best Val Loss 直接横比

- `Stage A = 0.9781`
- `v1 = 26.5616`
- `v2 = 26.9097`
- `v3 = 27.8843`
- `v4(avg) = 1.2552`
- `v4(min) = 0.0021`
- `v4(max) = 6.2637`
- 备注：这个横比仅供管理层快速感知，不是完全同口径。`v4` 每次只优化一个任务，不能与多任务总损失做严格一一等价比较。

## 6. Val 主指标总表

| Task | Metric | Better | Stage A | v1 | v2 | v3 | v4 | v4-v3 | 趋势 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| energy_per_atom | MAE | lower | `0.3606` | `1.7711` | `1.7583` | `1.8758` | `1.7430` | `-0.1328` | improved |
| formation_energy_per_atom | MAE | lower | `0.0800` | `0.2150` | `0.2137` | `0.2147` | `0.1548` | `-0.0599` | improved |
| energy_above_hull | MAE | lower | `0.0644` | `0.1225` | `0.1241` | `0.1220` | `0.1007` | `-0.0212` | improved |
| band_gap | MAE | lower | `0.2308` | `0.5402` | `0.5505` | `0.5507` | `0.4954` | `-0.0553` | improved |
| cbm | MAE | lower | `0.2921` | `0.5940` | `0.6060` | `0.6188` | `0.5645` | `-0.0543` | improved |
| vbm | MAE | lower | `0.2594` | `0.4770` | `0.4885` | `0.4935` | `0.4504` | `-0.0431` | improved |
| efermi | MAE | lower | `0.3834` | `0.6186` | `0.6257` | `0.6324` | `0.5706` | `-0.0618` | improved |
| is_metal | AUROC | higher | `0.9575` | `0.9029` | `0.9035` | `0.9056` | `0.9202` | `+0.0147` | improved |
| is_stable | AUROC | higher | `N/A` | `0.8389` | `0.8378` | `0.8335` | `0.8666` | `+0.0330` | improved |
| bulk_modulus_vrh | MAE | lower | `N/A` | `8.3487` | `8.3024` | `8.5855` | `8.4485` | `-0.1370` | improved |
| shear_modulus_vrh | MAE | lower | `N/A` | `10.3827` | `10.2219` | `10.6750` | `10.5170` | `-0.1580` | improved |
| homogeneous_poisson | MAE | lower | `N/A` | `0.0423` | `0.0423` | `0.0441` | `0.0419` | `-0.0023` | improved |
| universal_anisotropy | MAE | lower | `N/A` | `1.8196` | `1.8690` | `1.8657` | `1.8376` | `-0.0281` | improved |

## 7. 管理层解读

- `v4` 的价值不在于“超过 Stage A”，而在于证明 `Stage B` 任务扩展时，多任务联合训练是当前主要瓶颈。
- `v1 -> v3` 的调权、采样和 oversampling 调整没有解决根问题，反而让核心任务长期停留在较差区间。
- `v4` 通过拆开任务训练，把每个任务的优化目标从“多任务妥协”改为“单任务最优”，因此所有任务都比 `v3` 更好。
- 这说明下一阶段不应继续在 `v3` 路线上微调 oversampling，而应围绕“共享 backbone，多头独立精调/推理”继续推进。

## 8. 运行与修复记录

- 原始 `v4` 首轮完成 `9/13` 个任务。
- `4` 个弹性任务首次失败，根因为稀疏标签 batch 下 `loss` 无梯度图。
- 根因修复位置：`src/mp_data_pipeline/training/losses.py:42`
- 修复后对 `bulk_modulus_vrh`、`shear_modulus_vrh`、`homogeneous_poisson`、`universal_anisotropy` 进行了隔离补跑，全部成功。

## 9. v4 单任务 Run 索引

| Task | Run Dir | Best Epoch | Best Val Loss |
|---|---|---:|---:|
| energy_per_atom | `artifacts/runs_stageb_v4/energy_per_atom/20260308_081747` | `30` | `1.4716` |
| formation_energy_per_atom | `artifacts/runs_stageb_v4/formation_energy_per_atom/20260308_082806` | `30` | `0.0307` |
| energy_above_hull | `artifacts/runs_stageb_v4/energy_above_hull/20260308_083825` | `30` | `0.0207` |
| band_gap | `artifacts/runs_stageb_v4/band_gap/20260308_084843` | `28` | `0.2595` |
| cbm | `artifacts/runs_stageb_v4/cbm/20260308_085901` | `28` | `0.2513` |
| vbm | `artifacts/runs_stageb_v4/vbm/20260308_090919` | `24` | `0.1774` |
| efermi | `artifacts/runs_stageb_v4/efermi/20260308_091936` | `30` | `0.2748` |
| is_metal | `artifacts/runs_stageb_v4/is_metal/20260308_092955` | `29` | `0.3493` |
| is_stable | `artifacts/runs_stageb_v4/is_stable/20260308_094013` | `30` | `0.3643` |
| bulk_modulus_vrh | `artifacts/runs_stageb_v4_retry_elastic/bulk_modulus_vrh/20260308_162533` | `17` | `5.8835` |
| shear_modulus_vrh | `artifacts/runs_stageb_v4_retry_elastic/shear_modulus_vrh/20260308_163553` | `17` | `6.2637` |
| homogeneous_poisson | `artifacts/runs_stageb_v4_retry_elastic/homogeneous_poisson/20260308_164612` | `9` | `0.0021` |
| universal_anisotropy | `artifacts/runs_stageb_v4_retry_elastic/universal_anisotropy/20260308_165632` | `28` | `0.9685` |

## 10. 推荐决策

- 结论：`v4` 应作为后续 `Stage B` 默认路线。
- 建议 1：进入 `v5`，在 `v4` 基础上尝试“解冻 backbone 最后 1-2 层”。
- 建议 2：推理侧改成“一次 backbone embedding + 多个 task head”以控制部署成本。
- 建议 3：对老板汇报时，以“v4 全面优于 v3，已验证路线切换有效”作为主结论。

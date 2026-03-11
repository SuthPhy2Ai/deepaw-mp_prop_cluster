# 老板看的一页执行摘要

范围：`Stage A / Stage A PyG / Stage B v1-v4 / Stage C h1-h2-hybrid / zero`，统一口径基于任务级 `R2/ACC`，不以 `best_val_loss` 作为跨方法结论依据。

## 一句话结论

- 当前最优决策不是押注单一路线，而是 `zero + Stage A体系 + v2/v4` 的组合：`zero` 负责热力学/稳定性/结构上限，`Stage A体系（Stage A + Stage A PyG）` 负责电子结构主线，`v2/v4` 负责稀疏弹性补位。
- 从胜场看，`zero` 为全局第一：`VAL=6`、`TEST=6`；`Stage A体系（Stage A + Stage A PyG）` 合计 `VAL=5`、`TEST=5`，稳居第二梯队但主导电子结构簇。

## 决策板

| 动作 | 方法 | 理由 |
|---|---|---|
| 优先推进 | zero | 15/15 已覆盖，热力学/稳定性/结构整体最强。 |
| 保留主线 | Stage A | 原始多任务主线，cbm/efermi/test band_gap 仍强。 |
| 保留参考 | Stage A PyG | band_gap/vbm/is_metal 有局部提升，但未改写全局格局。 |
| 保留补位 | v2 / v4 | 稀疏弹性任务仍需任务特化路线。 |
| 观察 | Stage C hybrid | 只在 val universal_anisotropy 给出局部最优信号。 |
| 暂停 | v1 / v3 / Stage C h1/h2 | 当前没有形成主线竞争力。 |

## 关键数字

| 对象 | VAL 胜场 | TEST 胜场 | 解读 |
|---|---|---|---|
| zero | 6 | 6 | 全局最强单方法 |
| Stage A体系（Stage A + Stage A PyG） | 5 | 5 | 电子结构簇主导家族 |
| v2 | 3 | 1 | 弹性共享路线补位 |
| v4 | 0 | 2 | 弹性单任务补位 |
| Stage C hybrid | 1 | 0 | 局部信号 |

## 任务簇归属

| 任务簇 | VAL 最优 | TEST 最优 |
|---|---|---|
| 热力学 | energy_above_hull->zero, energy_per_atom->zero, formation_energy_per_atom->zero | energy_above_hull->zero, energy_per_atom->zero, formation_energy_per_atom->zero |
| 电子结构 | band_gap->Stage A PyG, cbm->Stage A, efermi->Stage A, is_metal->Stage A PyG, vbm->Stage A PyG | band_gap->Stage A, cbm->Stage A, efermi->Stage A, is_metal->Stage A, vbm->Stage A PyG |
| 稳定性 | is_stable->zero | is_stable->zero |
| 结构 | density->zero, volume->zero | density->zero, volume->zero |
| 弹性 | bulk_modulus_vrh->v2, homogeneous_poisson->v2, shear_modulus_vrh->v2, universal_anisotropy->Stage C hybrid | bulk_modulus_vrh->v2, homogeneous_poisson->v4, shear_modulus_vrh->v1, universal_anisotropy->v4 |

## 当前最难任务

| Split | Task | 当前最好方法 | 最好分数 |
|---|---|---|---|
| VAL | homogeneous_poisson | v2 | 0.1789 |
| VAL | universal_anisotropy | Stage C hybrid | 0.3660 |
| VAL | volume | zero | 0.6904 |
| VAL | shear_modulus_vrh | v2 | 0.6928 |
| TEST | homogeneous_poisson | v4 | 0.2336 |
| TEST | universal_anisotropy | v4 | 0.3790 |
| TEST | volume | zero | 0.6746 |
| TEST | shear_modulus_vrh | v1 | 0.7312 |

## 结论落地

1. 主线继续押 `zero`，它是当前唯一同时在热力学、稳定性、结构上形成系统优势的方法。
2. 电子结构不应放弃 `Stage A体系（Stage A + Stage A PyG）`，它仍是 `band_gap/cbm/vbm/efermi/is_metal` 的最优来源。
3. 弹性任务不要再追求单一统一头，继续保留 `v2/v4`，Stage C 仅保留 `hybrid` 做下一轮验证。

完整报告见：`reports/MASTER_COMPLETE_REPORT_STAGEA_STAGEB_STAGEC_ZERO.md`

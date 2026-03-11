# 全方法完整总报告

本报告是当前项目的完整总成品，统一覆盖 `Stage A / Stage A PyG / Stage B v1-v4 / Stage C h1-h2-hybrid / zero`。
它区别于之前的局部报告：前半部分用于路线决策，后半部分用于研发复盘，附录保留全量逐任务明细和索引。

## Part I 执行摘要

### 1. 一页结论

- 当前真正主导全局的方法家族仍然是 `Stage A体系（Stage A + Stage A PyG）` 与 `zero`。
- 胜场统计上，`zero` 目前在 `VAL=6`、`TEST=6` 个任务上取得全表最佳；`Stage A体系（Stage A + Stage A PyG）` 合计在 `VAL=5`、`TEST=5` 个任务上取得全表最佳。
- 电子结构高覆盖任务簇当前由 `Stage A体系（Stage A + Stage A PyG）` 主导：`band_gap / cbm / vbm / efermi / is_metal` 在 `Stage A` 与 `Stage A PyG` 之间分工领先。
- `zero` 主导热力学、稳定性和结构任务簇：`energy_per_atom / formation_energy_per_atom / energy_above_hull / is_stable / volume / density`。
- 稀疏弹性任务仍然不是 `Stage A体系（Stage A + Stage A PyG）` 或 `zero` 的绝对主场：`v2` 与 `v4` 依然保留局部最优。
- `Stage C` 三组 head 结构没有形成整体突破；但 `Stage C hybrid` 在 `VAL universal_anisotropy` 上拿到当前最好结果，说明“头上加头”并非完全无效，只是还不够强。
- 现阶段最合理的主线，不是统一押注单一方法，而是：`Stage A体系 + zero + v4/v2` 的分工组合。

### 2. 路线判断

| 路线 | 建议 | 理由 |
|---|---|---|
| Stage A | 保留 | 原始 grouped 多任务主线，仍在 cbm/efermi 等电子结构任务上保持强势。 |
| Stage A PyG | 保留参考 | 同口径 PyG 版本在 band_gap/vbm/is_metal 上有局部增益，但还没有改写全局路线。 |
| zero | 优先推进 | 热力学、稳定性、结构任务上限最强，且已覆盖 15/15 任务。 |
| v4 | 保留 | 稀疏弹性任务上仍有明显局部最优，适合作为单任务微调路线。 |
| v2 | 保留 | 共享多任务路线里对部分弹性任务仍最强，是 elastic fallback。 |
| Stage C hybrid | 观察推进 | 只在 universal_anisotropy 上给出局部信号，需下一轮解冻 backbone 再验证。 |
| Stage C h1 / h2 | 暂停独立扩展 | 冻结 backbone 条件下未显示出足够独立价值。 |
| v1 / v3 | 暂停 | 已被 v4、zero、Stage A体系或 v2 全面压制，不宜继续主线投入。 |

### 3. 总体胜场统计

![Method wins](figures/master_complete_report/method_win_counts_val_test.png)

| Method | VAL Wins | TEST Wins |
|---|---|---|
| Stage A | 2 | 4 |
| Stage A PyG | 3 | 1 |
| v1 | 0 | 1 |
| v2 | 3 | 1 |
| v3 | 0 | 0 |
| v4 | 0 | 2 |
| zero | 6 | 6 |
| Stage C h1 | 0 | 0 |
| Stage C h2 | 0 | 0 |
| Stage C hybrid | 1 | 0 |

注：胜场统计已纳入 `Stage A PyG`；路线判断仍把它视为同口径参考/补充分支，而非独立替代主线。

### 4. 当前最难任务

![Hardest tasks](figures/master_complete_report/hardest_tasks_val_test.png)

按“所有方法中的最佳主分数”排序，当前最难的第一梯队任务是：

| Split | Task | Type | Best Score | Best Method |
|---|---|---|---|---|
| VAL | homogeneous_poisson | regression | 0.1789 | v2 |
| VAL | universal_anisotropy | regression | 0.3660 | Stage C hybrid |
| VAL | volume | regression | 0.6904 | zero |
| VAL | shear_modulus_vrh | regression | 0.6928 | v2 |
| TEST | homogeneous_poisson | regression | 0.2336 | v4 |
| TEST | universal_anisotropy | regression | 0.3790 | v4 |
| TEST | volume | regression | 0.6746 | zero |
| TEST | shear_modulus_vrh | regression | 0.7312 | v1 |

核心判断：`homogeneous_poisson` 与 `universal_anisotropy` 仍然是全项目最难任务，`volume` 与 `shear_modulus_vrh` 处于第二梯队。

## Part II 完整技术正文

### 5. 方法谱系总览

- `Stage A`: 高覆盖 8 任务共享多任务。
- `Stage A PyG`: 与 Stage A 同口径的 PyG 基线，只作补充参考。
- `v1/v2/v3`: Stage B 13 任务共享多任务变体。
- `v4`: 共享预训练 backbone + 单任务 head 微调家族。
- `zero`: 完全独立单任务家族，每个性质单独训练。
- `Stage C h1`: 电子结构层级头。
- `Stage C h2`: 弹性派生层级头。
- `Stage C hybrid`: 电子层级 + 弹性派生层级的组合。

### 6. 方法设置与实验口径

| Method | Run | Branch | Backbone | Head Variant | Hidden | Layers | Batch | Epochs | LR | WD | PyG | Enabled Tasks | Freeze Backbone |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Stage A | 20260305_210307 | stage_a | graph | grouped | 256 | 6 | 64 | 50 | 0.0001 | 1e-05 | None | 8 | None |
| Stage A PyG | 20260308_182946 | stage_a | graph | grouped | 256 | 6 | 64 | 50 | 0.0001 | 1e-05 | True | 8 | False |
| v1 | 20260307_185342 | stage_b | graph | grouped | 256 | 6 | 64 | 50 | 0.0001 | 1e-05 | None | 13 | None |
| v2 | 20260308_001437 | stage_b | graph | grouped | 256 | 6 | 64 | 50 | 0.0001 | 1e-05 | True | 13 | None |
| v3 | 20260308_070539 | stage_b | graph | grouped | 256 | 6 | 64 | 50 | 0.0001 | 1e-05 | True | 13 | None |
| v4 | single-task family | stage_b | graph | grouped | 256 | 6 | 64 | 30 | 0.0002 | 1e-05 | True | 13 | True |
| zero | exp106_zero_single_task_family | zero | graph | per_task | 128-320 | 4-7 | 32-64 | 50-90 | 8e-5~2e-4 | 1e-05 | True | 15 | False |
| Stage C h1 | 20260310_003913 | stage_c | graph | stagec_h1 | 256 | 6 | 64 | 35 | 0.0002 | 1e-05 | True | 13 | True |
| Stage C h2 | 20260310_005519 | stage_c | graph | stagec_h2 | 256 | 6 | 64 | 35 | 0.0002 | 1e-05 | True | 13 | True |
| Stage C hybrid | 20260310_011108 | stage_c | graph | stagec_hybrid | 256 | 6 | 64 | 35 | 0.0002 | 1e-05 | True | 13 | True |

### 7. 核心结果总表

![Best method matrix](figures/master_complete_report/task_best_method_matrix_val_test.png)

#### VAL 主分数统一横比

| Task | Type | Best | Stage A | Stage A PyG | v1 | v2 | v3 | v4 | zero | Stage C h1 | Stage C h2 | Stage C hybrid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| energy_per_atom | regression | zero | 0.9373 | 0.9405 | 0.6876 | 0.7105 | 0.6756 | 0.6865 | 0.9414 | 0.6794 | 0.6776 | 0.6759 |
| formation_energy_per_atom | regression | zero | 0.9880 | 0.9876 | 0.9119 | 0.9136 | 0.9126 | 0.9472 | 0.9966 | 0.9254 | 0.9206 | 0.9204 |
| energy_above_hull | regression | zero | 0.9237 | 0.9241 | 0.5565 | 0.5580 | 0.5487 | 0.7302 | 0.9733 | 0.5797 | 0.5663 | 0.5652 |
| band_gap | regression | Stage A PyG | 0.9038 | 0.9085 | 0.6506 | 0.6480 | 0.6462 | 0.6957 | 0.8989 | 0.6691 | 0.6613 | 0.6519 |
| cbm | regression | Stage A | 0.9625 | 0.9618 | 0.8689 | 0.8655 | 0.8609 | 0.8795 | 0.9612 | 0.8729 | 0.8676 | 0.8620 |
| vbm | regression | Stage A PyG | 0.9722 | 0.9729 | 0.9137 | 0.9132 | 0.9119 | 0.9235 | 0.9705 | 0.9197 | 0.9163 | 0.9160 |
| efermi | regression | Stage A | 0.9345 | 0.9336 | 0.8837 | 0.8858 | 0.8833 | 0.8962 | 0.9290 | 0.8897 | 0.8867 | 0.8858 |
| is_metal | classification | Stage A PyG | 0.8902 | 0.8972 | 0.8217 | 0.8222 | 0.8232 | 0.8429 | 0.8767 | 0.8273 | 0.8302 | 0.8267 |
| is_stable | classification | zero | N/A | N/A | 0.8106 | 0.8138 | 0.8095 | 0.8248 | 0.8511 | 0.8171 | 0.8124 | 0.8127 |
| volume | regression | zero | N/A | N/A | N/A | N/A | N/A | N/A | 0.6904 | N/A | N/A | N/A |
| density | regression | zero | N/A | N/A | N/A | N/A | N/A | N/A | 0.9997 | N/A | N/A | N/A |
| bulk_modulus_vrh | regression | v2 | N/A | N/A | 0.9166 | 0.9245 | 0.9134 | 0.9136 | 0.9227 | 0.9140 | 0.9138 | 0.9137 |
| shear_modulus_vrh | regression | v2 | N/A | N/A | 0.6908 | 0.6928 | 0.6815 | 0.6862 | 0.6614 | 0.6840 | 0.6854 | 0.6864 |
| homogeneous_poisson | regression | v2 | N/A | N/A | 0.1653 | 0.1789 | 0.1418 | 0.1532 | 0.1557 | 0.0924 | 0.1295 | 0.0830 |
| universal_anisotropy | regression | Stage C hybrid | N/A | N/A | 0.3543 | 0.3495 | 0.3393 | 0.3636 | 0.3023 | 0.3475 | 0.3660 | 0.3660 |

#### TEST 主分数统一横比

| Task | Type | Best | Stage A | Stage A PyG | v1 | v2 | v3 | v4 | zero | Stage C h1 | Stage C h2 | Stage C hybrid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| energy_per_atom | regression | zero | 0.9461 | 0.9358 | 0.6895 | 0.7118 | 0.6863 | 0.6958 | 0.9597 | 0.6903 | 0.6880 | 0.6867 |
| formation_energy_per_atom | regression | zero | 0.9872 | 0.9869 | 0.9153 | 0.9160 | 0.9150 | 0.9481 | 0.9963 | 0.9280 | 0.9233 | 0.9228 |
| energy_above_hull | regression | zero | 0.9211 | 0.9211 | 0.5791 | 0.5839 | 0.5743 | 0.7379 | 0.9707 | 0.6057 | 0.5916 | 0.5911 |
| band_gap | regression | Stage A | 0.9039 | 0.9016 | 0.6463 | 0.6358 | 0.6387 | 0.6882 | 0.8923 | 0.6639 | 0.6550 | 0.6460 |
| cbm | regression | Stage A | 0.9607 | 0.9605 | 0.8608 | 0.8541 | 0.8488 | 0.8689 | 0.9568 | 0.8625 | 0.8560 | 0.8505 |
| vbm | regression | Stage A PyG | 0.9713 | 0.9722 | 0.9090 | 0.9066 | 0.9080 | 0.9208 | 0.9695 | 0.9168 | 0.9125 | 0.9124 |
| efermi | regression | Stage A | 0.9350 | 0.9344 | 0.8856 | 0.8874 | 0.8858 | 0.8973 | 0.9283 | 0.8915 | 0.8885 | 0.8882 |
| is_metal | classification | Stage A | 0.8935 | 0.8933 | 0.8214 | 0.8212 | 0.8251 | 0.8425 | 0.8782 | 0.8301 | 0.8268 | 0.8254 |
| is_stable | classification | zero | N/A | N/A | 0.8160 | 0.8113 | 0.8123 | 0.8263 | 0.8510 | 0.8197 | 0.8154 | 0.8152 |
| volume | regression | zero | N/A | N/A | N/A | N/A | N/A | N/A | 0.6746 | N/A | N/A | N/A |
| density | regression | zero | N/A | N/A | N/A | N/A | N/A | N/A | 0.9997 | N/A | N/A | N/A |
| bulk_modulus_vrh | regression | v2 | N/A | N/A | 0.9173 | 0.9180 | 0.9174 | 0.9167 | 0.9102 | 0.9169 | 0.9174 | 0.9170 |
| shear_modulus_vrh | regression | v1 | N/A | N/A | 0.7312 | 0.7209 | 0.7268 | 0.7282 | 0.7162 | 0.7271 | 0.7285 | 0.7284 |
| homogeneous_poisson | regression | v4 | N/A | N/A | 0.2318 | 0.2222 | 0.2068 | 0.2336 | 0.1735 | 0.1895 | 0.2282 | 0.1781 |
| universal_anisotropy | regression | v4 | N/A | N/A | 0.3524 | 0.3621 | 0.3697 | 0.3790 | 0.2528 | 0.3689 | 0.3660 | 0.3691 |

### 8. 方法胜负图谱

| Split | Family | Task -> Best Method |
|---|---|---|
| VAL | thermo | energy_per_atom->zero, formation_energy_per_atom->zero, energy_above_hull->zero |
| VAL | electronic | band_gap->Stage A PyG, cbm->Stage A, vbm->Stage A PyG, efermi->Stage A, is_metal->Stage A PyG |
| VAL | stability | is_stable->zero |
| VAL | structure | volume->zero, density->zero |
| VAL | elastic | bulk_modulus_vrh->v2, shear_modulus_vrh->v2, homogeneous_poisson->v2, universal_anisotropy->Stage C hybrid |
| TEST | thermo | energy_per_atom->zero, formation_energy_per_atom->zero, energy_above_hull->zero |
| TEST | electronic | band_gap->Stage A, cbm->Stage A, vbm->Stage A PyG, efermi->Stage A, is_metal->Stage A |
| TEST | stability | is_stable->zero |
| TEST | structure | volume->zero, density->zero |
| TEST | elastic | bulk_modulus_vrh->v2, shear_modulus_vrh->v1, homogeneous_poisson->v4, universal_anisotropy->v4 |

按任务簇看，当前主导区很清楚：

- `VAL`：电子簇由 `Stage A体系（Stage A + Stage A PyG）` 主导，热力学/稳定性/结构由 `zero` 主导，弹性簇由 `v2` 与 `Stage C hybrid` 局部主导。
- `TEST`：电子簇由 `Stage A体系（Stage A + Stage A PyG）` 主导，热力学/稳定性/结构由 `zero` 主导，弹性簇由 `v2/v4/v1` 分裂主导。

### 9. 任务分层结论

- 热力学任务簇：`zero` 全面最强，说明单任务精细回归上限更高。
- 电子结构任务簇：`Stage A体系（Stage A + Stage A PyG）` 全面最强，说明高覆盖共享多任务加上 PyG 数据路径，对电子结构表征更有利。
- 稳定性任务：`zero` 最强，`is_stable` 更像热力学边界判别，不适合硬塞进统一共享主线。
- 结构任务：目前只有 `zero` 有完整结果，`volume` 依旧偏难，`density` 几乎已解决。
- 弹性任务：没有单一王者，需要按任务单独选路线。

### 10. Stage A vs zero 专题

Stage A 与 zero 不是简单的谁全面碾压谁，而是分别统治不同任务簇。

#### VAL: Stage A 胜过 zero

| Task | Stage A | zero | Delta(StageA-zero) |
|---|---|---|---|
| is_metal | 0.8902 | 0.8767 | +0.0134 |
| efermi | 0.9345 | 0.9290 | +0.0055 |
| band_gap | 0.9038 | 0.8989 | +0.0050 |
| vbm | 0.9722 | 0.9705 | +0.0017 |
| cbm | 0.9625 | 0.9612 | +0.0013 |

#### VAL: zero 胜过 Stage A

| Task | zero | Stage A | Delta(zero-StageA) |
|---|---|---|---|
| energy_above_hull | 0.9733 | 0.9237 | +0.0496 |
| formation_energy_per_atom | 0.9966 | 0.9880 | +0.0086 |
| energy_per_atom | 0.9414 | 0.9373 | +0.0041 |

#### TEST: Stage A 胜过 zero

| Task | Stage A | zero | Delta(StageA-zero) |
|---|---|---|---|
| is_metal | 0.8935 | 0.8782 | +0.0153 |
| band_gap | 0.9039 | 0.8923 | +0.0116 |
| efermi | 0.9350 | 0.9283 | +0.0067 |
| cbm | 0.9607 | 0.9568 | +0.0039 |
| vbm | 0.9713 | 0.9695 | +0.0018 |

#### TEST: zero 胜过 Stage A

| Task | zero | Stage A | Delta(zero-StageA) |
|---|---|---|---|
| energy_above_hull | 0.9707 | 0.9211 | +0.0496 |
| energy_per_atom | 0.9597 | 0.9461 | +0.0136 |
| formation_energy_per_atom | 0.9963 | 0.9872 | +0.0091 |

物理上，`Stage A` 胜出的任务高度集中在电子结构簇，这和现有物理分析一致；`zero` 胜出的任务则高度集中于热力学与稳定性簇。

### 11. 稀疏弹性任务专题

| Split | Task | Best Method | Best Score |
|---|---|---|---|
| VAL | bulk_modulus_vrh | v2 | 0.9245 |
| VAL | shear_modulus_vrh | v2 | 0.6928 |
| VAL | homogeneous_poisson | v2 | 0.1789 |
| VAL | universal_anisotropy | Stage C hybrid | 0.3660 |
| TEST | bulk_modulus_vrh | v2 | 0.9180 |
| TEST | shear_modulus_vrh | v1 | 0.7312 |
| TEST | homogeneous_poisson | v4 | 0.2336 |
| TEST | universal_anisotropy | v4 | 0.3790 |

- `bulk_modulus_vrh`：当前最优仍是 `v2`。
- `shear_modulus_vrh`：`VAL` 最优是 `v2`，`TEST` 最优是 `v1`。
- `homogeneous_poisson`：`VAL` 最优是 `v2`，`TEST` 最优是 `v4`。
- `universal_anisotropy`：`VAL` 最优已被 `Stage C hybrid` 触达，但 `TEST` 仍由 `v4` 保持最优。
- 结论：弹性派生量仍然最依赖任务特化与结构设计，统一共享主线还没有解决这类任务。

### 12. 电子结构任务簇专题

| Split | Task | Best Method | Best Score |
|---|---|---|---|
| VAL | band_gap | Stage A PyG | 0.9085 |
| VAL | cbm | Stage A | 0.9625 |
| VAL | vbm | Stage A PyG | 0.9729 |
| VAL | efermi | Stage A | 0.9345 |
| VAL | is_metal | Stage A PyG | 0.8972 |
| TEST | band_gap | Stage A | 0.9039 |
| TEST | cbm | Stage A | 0.9607 |
| TEST | vbm | Stage A PyG | 0.9722 |
| TEST | efermi | Stage A | 0.9350 |
| TEST | is_metal | Stage A | 0.8935 |

电子簇在 `VAL/TEST` 上都由 `Stage A体系（Stage A + Stage A PyG）` 主导，但领导权在 `Stage A` 与 `Stage A PyG` 之间分摊。这支持“电子结构流形适合高覆盖共享多任务，且 PyG 数据路径能带来局部增益”的判断。
更完整的物理解释见 [PHYSICAL_ANALYSIS_STAGEA_VS_ZERO_TASK_RELATIONS.md](PHYSICAL_ANALYSIS_STAGEA_VS_ZERO_TASK_RELATIONS.md)。

### 13. Stage C 新架构专题

![Stage C delta](figures/master_complete_report/stagec_vs_v3_delta_val.png)

#### 13.1 设计意图

- `Stage C h1`：电子结构层级头，测试 `group head -> task heads` 是否优于直接并列输出。
- `Stage C h2`：弹性派生层级头，测试 `base elastic head -> derived-property heads` 是否更适合泊松比和各向异性。
- `Stage C hybrid`：把 h1 和 h2 合在同一共享 backbone 中。

#### 13.2 相对 v3 的同口径结果

| Method | Split | Improved vs v3 | Degraded vs v3 | Best Improved Task | Best Delta | Worst Task | Worst Delta |
|---|---|---|---|---|---|---|---|
| Stage C h1 | VAL | 12 | 1 | energy_above_hull | 0.0310 | homogeneous_poisson | -0.0494 |
| Stage C h1 | TEST | 10 | 3 | energy_above_hull | 0.0314 | homogeneous_poisson | -0.0174 |
| Stage C h2 | VAL | 12 | 1 | universal_anisotropy | 0.0267 | homogeneous_poisson | -0.0123 |
| Stage C h2 | TEST | 12 | 1 | homogeneous_poisson | 0.0213 | universal_anisotropy | -0.0037 |
| Stage C hybrid | VAL | 12 | 1 | universal_anisotropy | 0.0267 | homogeneous_poisson | -0.0588 |
| Stage C hybrid | TEST | 10 | 3 | energy_above_hull | 0.0168 | homogeneous_poisson | -0.0287 |

#### 13.3 最终判断

- `Stage C h1/h2/hybrid` 相对 `v3` 在多数任务上分数有改善，但改善幅度大多不足以跨过 `Stage A / zero / v2 / v4` 这些当前强者。
- `Stage C hybrid` 在 `VAL universal_anisotropy` 上给出了全表最佳，说明层级 head 对弹性派生量确实有信号。
- 但在当前“冻结 backbone + 35 epoch”的设定下，Stage C 还不足以成为主线。
- 最合理的下一步不是全面扩张 Stage C，而是只保留 `hybrid` 做一次允许 backbone 继续适配的 v2 版本。

## Part III 附录

### Appendix A. TRAIN/VAL/TEST 逐任务主表

#### TRAIN Regression: task loss + R2

| Task | Stage A loss | Stage A R2 | Stage A PyG loss | Stage A PyG R2 | v1 loss | v1 R2 | v2 loss | v2 R2 | v3 loss | v3 R2 | v4 loss | v4 R2 | zero loss | zero R2 | Stage C h1 loss | Stage C h1 R2 | Stage C h2 loss | Stage C h2 R2 | Stage C hybrid loss | Stage C hybrid R2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| energy_per_atom | 0.0872 | 0.9789 | 0.0987 | 0.9766 | 1.3926 | 0.7220 | 1.3809 | 0.7323 | 1.4773 | 0.7116 | 1.3500 | 0.7252 | 0.0668 | 0.9835 | 1.4138 | 0.7170 | 1.4365 | 0.7141 | 1.4367 | 0.7125 |
| formation_energy_per_atom | 0.0052 | 0.9925 | 0.0056 | 0.9919 | 0.0493 | 0.9172 | 0.0488 | 0.9184 | 0.0495 | 0.9175 | 0.0278 | 0.9526 | 0.0005 | 0.9992 | 0.0414 | 0.9302 | 0.0443 | 0.9256 | 0.0443 | 0.9254 |
| energy_above_hull | 0.0046 | 0.9536 | 0.0049 | 0.9500 | 0.0292 | 0.5906 | 0.0290 | 0.5946 | 0.0297 | 0.5843 | 0.0185 | 0.7540 | 0.0005 | 0.9948 | 0.0273 | 0.6160 | 0.0283 | 0.6027 | 0.0283 | 0.6016 |
| band_gap | 0.0350 | 0.9646 | 0.0377 | 0.9611 | 0.2898 | 0.6578 | 0.2955 | 0.6549 | 0.2988 | 0.6466 | 0.2573 | 0.7033 | 0.0264 | 0.9705 | 0.2782 | 0.6719 | 0.2889 | 0.6632 | 0.2885 | 0.6552 |
| cbm | 0.0349 | 0.9855 | 0.0373 | 0.9843 | 0.2766 | 0.8678 | 0.2833 | 0.8650 | 0.2916 | 0.8609 | 0.2476 | 0.8823 | 0.0155 | 0.9933 | 0.2641 | 0.8740 | 0.2773 | 0.8683 | 0.2841 | 0.8627 |
| vbm | 0.0296 | 0.9887 | 0.0322 | 0.9877 | 0.1980 | 0.9161 | 0.2015 | 0.9144 | 0.2066 | 0.9129 | 0.1743 | 0.9268 | 0.0107 | 0.9957 | 0.1870 | 0.9220 | 0.1964 | 0.9175 | 0.1964 | 0.9179 |
| efermi | 0.1265 | 0.9474 | 0.1297 | 0.9454 | 0.3099 | 0.8855 | 0.3118 | 0.8866 | 0.3201 | 0.8832 | 0.2748 | 0.8975 | 0.0766 | 0.9714 | 0.2983 | 0.8900 | 0.3085 | 0.8866 | 0.3078 | 0.8861 |
| volume | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 82.2914 | 0.8379 | N/A | N/A | N/A | N/A | N/A | N/A |
| density | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.0003 | 0.9999 | N/A | N/A | N/A | N/A | N/A | N/A |
| bulk_modulus_vrh | N/A | N/A | N/A | N/A | 2.8352 | 0.9660 | 3.0462 | 0.9613 | 3.4925 | 0.9602 | 3.1844 | 0.9614 | 3.7203 | 0.9582 | 3.1199 | 0.9615 | 3.1513 | 0.9618 | 3.1921 | 0.9620 |
| shear_modulus_vrh | N/A | N/A | N/A | N/A | 4.5051 | 0.7177 | 4.6394 | 0.7168 | 5.2798 | 0.7055 | 4.9824 | 0.7084 | 6.5180 | 0.6928 | 4.9694 | 0.7076 | 4.9487 | 0.7087 | 4.9654 | 0.7097 |
| homogeneous_poisson | N/A | N/A | N/A | N/A | 0.0039 | 0.3066 | 0.0040 | 0.2962 | 0.0041 | 0.2806 | 0.0037 | 0.3478 | 0.0039 | 0.3293 | 0.0040 | 0.2897 | 0.0038 | 0.3329 | 0.0042 | 0.2656 |
| universal_anisotropy | N/A | N/A | N/A | N/A | 1.1154 | 0.5518 | 1.0400 | 0.6026 | 1.2003 | 0.5389 | 1.0887 | 0.5898 | 1.1603 | 0.5347 | 1.1564 | 0.5555 | 1.1280 | 0.5829 | 1.1232 | 0.5846 |

#### TRAIN Classification: task loss + ACC

| Task | Stage A loss | Stage A ACC | Stage A PyG loss | Stage A PyG ACC | v1 loss | v1 ACC | v2 loss | v2 ACC | v3 loss | v3 ACC | v4 loss | v4 ACC | zero loss | zero ACC | Stage C h1 loss | Stage C h1 ACC | Stage C h2 loss | Stage C h2 ACC | Stage C hybrid loss | Stage C hybrid ACC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| is_metal | 0.0911 | 0.9640 | 0.1002 | 0.9608 | 0.3766 | 0.8276 | 0.3764 | 0.8265 | 0.3770 | 0.8270 | 0.3323 | 0.8514 | 0.2150 | 0.9112 | 0.3646 | 0.8322 | 0.3699 | 0.8320 | 0.3720 | 0.8299 |
| is_stable | N/A | N/A | N/A | N/A | 0.3838 | 0.8180 | 0.3843 | 0.8167 | 0.3874 | 0.8143 | 0.3439 | 0.8370 | 0.2513 | 0.8872 | 0.3671 | 0.8248 | 0.3776 | 0.8191 | 0.3756 | 0.8197 |

#### VAL Regression: task loss + R2

| Task | Stage A loss | Stage A R2 | Stage A PyG loss | Stage A PyG R2 | v1 loss | v1 R2 | v2 loss | v2 R2 | v3 loss | v3 R2 | v4 loss | v4 R2 | zero loss | zero R2 | Stage C h1 loss | Stage C h1 R2 | Stage C h2 loss | Stage C h2 R2 | Stage C hybrid loss | Stage C hybrid R2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| energy_per_atom | 0.2437 | 0.9373 | 0.2405 | 0.9405 | 1.4905 | 0.6876 | 1.4711 | 0.7105 | 1.5850 | 0.6756 | 1.4716 | 0.6865 | 0.1984 | 0.9414 | 1.5276 | 0.6794 | 1.5449 | 0.6776 | 1.5460 | 0.6759 |
| formation_energy_per_atom | 0.0084 | 0.9880 | 0.0086 | 0.9876 | 0.0516 | 0.9119 | 0.0506 | 0.9136 | 0.0515 | 0.9126 | 0.0307 | 0.9472 | 0.0024 | 0.9966 | 0.0435 | 0.9254 | 0.0465 | 0.9206 | 0.0465 | 0.9204 |
| energy_above_hull | 0.0076 | 0.9237 | 0.0074 | 0.9241 | 0.0313 | 0.5565 | 0.0315 | 0.5580 | 0.0320 | 0.5487 | 0.0207 | 0.7302 | 0.0027 | 0.9733 | 0.0298 | 0.5797 | 0.0307 | 0.5663 | 0.0307 | 0.5652 |
| band_gap | 0.0868 | 0.9038 | 0.0829 | 0.9085 | 0.2875 | 0.6506 | 0.2943 | 0.6480 | 0.2945 | 0.6462 | 0.2595 | 0.6957 | 0.0868 | 0.8989 | 0.2750 | 0.6691 | 0.2856 | 0.6613 | 0.2856 | 0.6519 |
| cbm | 0.0872 | 0.9625 | 0.0884 | 0.9618 | 0.2782 | 0.8689 | 0.2870 | 0.8655 | 0.2949 | 0.8609 | 0.2560 | 0.8795 | 0.0880 | 0.9612 | 0.2689 | 0.8729 | 0.2818 | 0.8676 | 0.2882 | 0.8620 |
| vbm | 0.0708 | 0.9722 | 0.0683 | 0.9729 | 0.1983 | 0.9137 | 0.2029 | 0.9132 | 0.2060 | 0.9119 | 0.1799 | 0.9235 | 0.0715 | 0.9705 | 0.1894 | 0.9197 | 0.1970 | 0.9163 | 0.1981 | 0.9160 |
| efermi | 0.1604 | 0.9345 | 0.1594 | 0.9336 | 0.3087 | 0.8837 | 0.3099 | 0.8858 | 0.3154 | 0.8833 | 0.2748 | 0.8962 | 0.1712 | 0.9290 | 0.2953 | 0.8897 | 0.3045 | 0.8867 | 0.3047 | 0.8858 |
| volume | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 145.0043 | 0.6904 | N/A | N/A | N/A | N/A | N/A | N/A |
| density | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.0011 | 0.9997 | N/A | N/A | N/A | N/A | N/A | N/A |
| bulk_modulus_vrh | N/A | N/A | N/A | N/A | 7.8839 | 0.9166 | 7.8362 | 0.9245 | 8.1216 | 0.9134 | 7.9840 | 0.9136 | 8.4433 | 0.9227 | 7.9600 | 0.9140 | 7.9303 | 0.9138 | 7.9382 | 0.9137 |
| shear_modulus_vrh | N/A | N/A | N/A | N/A | 9.9112 | 0.6908 | 9.7514 | 0.6928 | 10.2040 | 0.6815 | 10.0441 | 0.6862 | 11.2718 | 0.6614 | 10.0779 | 0.6840 | 10.0513 | 0.6854 | 10.0452 | 0.6864 |
| homogeneous_poisson | N/A | N/A | N/A | N/A | 0.0041 | 0.1653 | 0.0040 | 0.1789 | 0.0042 | 0.1418 | 0.0042 | 0.1532 | 0.0042 | 0.1557 | 0.0045 | 0.0924 | 0.0043 | 0.1295 | 0.0045 | 0.0830 |
| universal_anisotropy | N/A | N/A | N/A | N/A | 1.4907 | 0.3543 | 1.5469 | 0.3495 | 1.5425 | 0.3393 | 1.5106 | 0.3636 | 1.5448 | 0.3023 | 1.5211 | 0.3475 | 1.4995 | 0.3660 | 1.4961 | 0.3660 |

#### VAL Classification: task loss + ACC

| Task | Stage A loss | Stage A ACC | Stage A PyG loss | Stage A PyG ACC | v1 loss | v1 ACC | v2 loss | v2 ACC | v3 loss | v3 ACC | v4 loss | v4 ACC | zero loss | zero ACC | Stage C h1 loss | Stage C h1 ACC | Stage C h2 loss | Stage C h2 ACC | Stage C hybrid loss | Stage C hybrid ACC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| is_metal | 0.3369 | 0.8902 | 0.3004 | 0.8972 | 0.3837 | 0.8217 | 0.3822 | 0.8222 | 0.3798 | 0.8232 | 0.3493 | 0.8429 | 0.2845 | 0.8767 | 0.3700 | 0.8273 | 0.3739 | 0.8302 | 0.3764 | 0.8267 |
| is_stable | N/A | N/A | N/A | N/A | 0.3926 | 0.8106 | 0.3927 | 0.8138 | 0.3964 | 0.8095 | 0.3643 | 0.8248 | 0.3172 | 0.8511 | 0.3787 | 0.8171 | 0.3872 | 0.8124 | 0.3861 | 0.8127 |

#### TEST Regression: task loss + R2

| Task | Stage A loss | Stage A R2 | Stage A PyG loss | Stage A PyG R2 | v1 loss | v1 R2 | v2 loss | v2 R2 | v3 loss | v3 R2 | v4 loss | v4 R2 | zero loss | zero R2 | Stage C h1 loss | Stage C h1 R2 | Stage C h2 loss | Stage C h2 R2 | Stage C hybrid loss | Stage C hybrid R2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| energy_per_atom | 0.2198 | 0.9461 | 0.2449 | 0.9358 | 1.4779 | 0.6895 | 1.4406 | 0.7118 | 1.5453 | 0.6863 | 1.4317 | 0.6958 | 0.1710 | 0.9597 | 1.4849 | 0.6903 | 1.5072 | 0.6880 | 1.5057 | 0.6867 |
| formation_energy_per_atom | 0.0086 | 0.9872 | 0.0088 | 0.9869 | 0.0510 | 0.9153 | 0.0510 | 0.9160 | 0.0516 | 0.9150 | 0.0313 | 0.9481 | 0.0024 | 0.9963 | 0.0434 | 0.9280 | 0.0463 | 0.9233 | 0.0465 | 0.9228 |
| energy_above_hull | 0.0074 | 0.9211 | 0.0075 | 0.9211 | 0.0305 | 0.5791 | 0.0304 | 0.5839 | 0.0310 | 0.5743 | 0.0206 | 0.7379 | 0.0028 | 0.9707 | 0.0285 | 0.6057 | 0.0296 | 0.5916 | 0.0296 | 0.5911 |
| band_gap | 0.0903 | 0.9039 | 0.0911 | 0.9016 | 0.2990 | 0.6463 | 0.3078 | 0.6358 | 0.3066 | 0.6387 | 0.2719 | 0.6882 | 0.0951 | 0.8923 | 0.2865 | 0.6639 | 0.2966 | 0.6550 | 0.2975 | 0.6460 |
| cbm | 0.0892 | 0.9607 | 0.0906 | 0.9605 | 0.2876 | 0.8608 | 0.2983 | 0.8541 | 0.3096 | 0.8488 | 0.2702 | 0.8689 | 0.0944 | 0.9568 | 0.2814 | 0.8625 | 0.2966 | 0.8560 | 0.3021 | 0.8505 |
| vbm | 0.0732 | 0.9713 | 0.0710 | 0.9722 | 0.2080 | 0.9090 | 0.2100 | 0.9066 | 0.2116 | 0.9080 | 0.1833 | 0.9208 | 0.0742 | 0.9695 | 0.1930 | 0.9168 | 0.2022 | 0.9125 | 0.2025 | 0.9124 |
| efermi | 0.1674 | 0.9350 | 0.1664 | 0.9344 | 0.3160 | 0.8856 | 0.3158 | 0.8874 | 0.3200 | 0.8858 | 0.2817 | 0.8973 | 0.1807 | 0.9283 | 0.3009 | 0.8915 | 0.3102 | 0.8885 | 0.3093 | 0.8882 |
| volume | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 153.0870 | 0.6746 | N/A | N/A | N/A | N/A | N/A | N/A |
| density | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.0011 | 0.9997 | N/A | N/A | N/A | N/A | N/A | N/A |
| bulk_modulus_vrh | N/A | N/A | N/A | N/A | 7.1517 | 0.9173 | 7.0904 | 0.9180 | 7.3146 | 0.9174 | 7.2506 | 0.9167 | 7.7669 | 0.9102 | 7.2245 | 0.9169 | 7.1746 | 0.9174 | 7.2697 | 0.9170 |
| shear_modulus_vrh | N/A | N/A | N/A | N/A | 8.7552 | 0.7312 | 9.2176 | 0.7209 | 9.1389 | 0.7268 | 9.0616 | 0.7282 | 10.3661 | 0.7162 | 9.0895 | 0.7271 | 9.0575 | 0.7285 | 9.0730 | 0.7284 |
| homogeneous_poisson | N/A | N/A | N/A | N/A | 0.0040 | 0.2318 | 0.0041 | 0.2222 | 0.0041 | 0.2068 | 0.0040 | 0.2336 | 0.0043 | 0.1735 | 0.0042 | 0.1895 | 0.0040 | 0.2282 | 0.0043 | 0.1781 |
| universal_anisotropy | N/A | N/A | N/A | N/A | 1.8163 | 0.3524 | 1.8367 | 0.3621 | 1.8140 | 0.3697 | 1.7957 | 0.3790 | 1.9763 | 0.2528 | 1.7946 | 0.3689 | 1.8051 | 0.3660 | 1.7988 | 0.3691 |

#### TEST Classification: task loss + ACC

| Task | Stage A loss | Stage A ACC | Stage A PyG loss | Stage A PyG ACC | v1 loss | v1 ACC | v2 loss | v2 ACC | v3 loss | v3 ACC | v4 loss | v4 ACC | zero loss | zero ACC | Stage C h1 loss | Stage C h1 ACC | Stage C h2 loss | Stage C h2 ACC | Stage C hybrid loss | Stage C hybrid ACC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| is_metal | 0.3289 | 0.8935 | 0.3196 | 0.8933 | 0.3856 | 0.8214 | 0.3879 | 0.8212 | 0.3866 | 0.8251 | 0.3563 | 0.8425 | 0.2924 | 0.8782 | 0.3756 | 0.8301 | 0.3812 | 0.8268 | 0.3832 | 0.8254 |
| is_stable | N/A | N/A | N/A | N/A | 0.3881 | 0.8160 | 0.3895 | 0.8113 | 0.3925 | 0.8123 | 0.3564 | 0.8263 | 0.3169 | 0.8510 | 0.3722 | 0.8197 | 0.3819 | 0.8154 | 0.3801 | 0.8152 |

### Appendix B. 回归附加指标与分类 AUROC

#### TRAIN Regression: MAE + RMSE

| Task | Stage A MAE | Stage A RMSE | Stage A PyG MAE | Stage A PyG RMSE | v1 MAE | v1 RMSE | v2 MAE | v2 RMSE | v3 MAE | v3 RMSE | v4 MAE | v4 RMSE | zero MAE | zero RMSE | Stage C h1 MAE | Stage C h1 RMSE | Stage C h2 MAE | Stage C h2 RMSE | Stage C hybrid MAE | Stage C hybrid RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| energy_per_atom | 0.1848 | 1.1552 | 0.2040 | 1.2153 | 1.6699 | 4.1920 | 1.6646 | 4.1141 | 1.7658 | 4.2699 | 1.6161 | 4.1677 | 0.1275 | 1.0216 | 1.6834 | 4.2300 | 1.7110 | 4.2513 | 1.7104 | 4.2631 |
| formation_energy_per_atom | 0.0696 | 0.1042 | 0.0720 | 0.1082 | 0.2108 | 0.3460 | 0.2104 | 0.3435 | 0.2116 | 0.3455 | 0.1491 | 0.2619 | 0.0174 | 0.0343 | 0.1881 | 0.3177 | 0.1963 | 0.3280 | 0.1954 | 0.3284 |
| energy_above_hull | 0.0568 | 0.0975 | 0.0574 | 0.1012 | 0.1195 | 0.2897 | 0.1204 | 0.2882 | 0.1189 | 0.2919 | 0.0958 | 0.2246 | 0.0168 | 0.0327 | 0.1139 | 0.2805 | 0.1159 | 0.2854 | 0.1133 | 0.2857 |
| band_gap | 0.1433 | 0.2843 | 0.1494 | 0.2981 | 0.5431 | 0.8842 | 0.5514 | 0.8879 | 0.5555 | 0.8986 | 0.4919 | 0.8233 | 0.1186 | 0.2595 | 0.5220 | 0.8658 | 0.5415 | 0.8772 | 0.5341 | 0.8875 |
| cbm | 0.1861 | 0.2768 | 0.1929 | 0.2876 | 0.5911 | 0.8349 | 0.6003 | 0.8436 | 0.6126 | 0.8563 | 0.5500 | 0.7876 | 0.1120 | 0.1881 | 0.5732 | 0.8151 | 0.5929 | 0.8334 | 0.5986 | 0.8508 |
| vbm | 0.1701 | 0.2537 | 0.1780 | 0.2653 | 0.4790 | 0.6926 | 0.4861 | 0.6994 | 0.4926 | 0.7059 | 0.4406 | 0.6470 | 0.0902 | 0.1561 | 0.4639 | 0.6679 | 0.4774 | 0.6867 | 0.4790 | 0.6851 |
| efermi | 0.3167 | 0.6343 | 0.3175 | 0.6462 | 0.6196 | 0.9355 | 0.6254 | 0.9312 | 0.6364 | 0.9448 | 0.5670 | 0.8852 | 0.2400 | 0.4679 | 0.6033 | 0.9171 | 0.6182 | 0.9312 | 0.6167 | 0.9330 |
| volume | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 82.7826 | 231.5513 | N/A | N/A | N/A | N/A | N/A | N/A |
| density | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.0142 | 0.0236 | N/A | N/A | N/A | N/A | N/A | N/A |
| bulk_modulus_vrh | N/A | N/A | N/A | N/A | 3.2653 | 13.9206 | 3.4806 | 14.8618 | 3.9368 | 15.0603 | 3.6197 | 14.8391 | 4.1662 | 15.4461 | 3.5501 | 14.8182 | 3.5851 | 14.7625 | 3.6285 | 14.7266 |
| shear_modulus_vrh | N/A | N/A | N/A | N/A | 4.9327 | 27.6702 | 5.0658 | 27.7123 | 5.7265 | 28.2641 | 5.4190 | 28.1208 | 6.9751 | 28.8660 | 5.4045 | 28.1608 | 5.3849 | 28.1059 | 5.4030 | 28.0575 |
| homogeneous_poisson | N/A | N/A | N/A | N/A | 0.0308 | 0.0899 | 0.0321 | 0.0906 | 0.0340 | 0.0916 | 0.0293 | 0.0872 | 0.0447 | 0.0884 | 0.0341 | 0.0910 | 0.0265 | 0.0882 | 0.0378 | 0.0925 |
| universal_anisotropy | N/A | N/A | N/A | N/A | 1.4089 | 5.2124 | 1.3302 | 4.9081 | 1.4967 | 5.2870 | 1.3834 | 4.9864 | 1.4066 | 5.3110 | 1.4483 | 5.1908 | 1.4288 | 5.0283 | 1.4231 | 5.0184 |

#### TRAIN Classification: AUROC

| Task | Stage A AUROC | Stage A PyG AUROC | v1 AUROC | v2 AUROC | v3 AUROC | v4 AUROC | zero AUROC | Stage C h1 AUROC | Stage C h2 AUROC | Stage C hybrid AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| is_metal | 0.9950 | 0.9938 | 0.9056 | 0.9054 | 0.9059 | 0.9271 | 0.9712 | 0.9117 | 0.9093 | 0.9082 |
| is_stable | N/A | N/A | 0.8448 | 0.8437 | 0.8405 | 0.8815 | 0.9432 | 0.8606 | 0.8507 | 0.8527 |

#### VAL Regression: MAE + RMSE

| Task | Stage A MAE | Stage A RMSE | Stage A PyG MAE | Stage A PyG RMSE | v1 MAE | v1 RMSE | v2 MAE | v2 RMSE | v3 MAE | v3 RMSE | v4 MAE | v4 RMSE | zero MAE | zero RMSE | Stage C h1 MAE | Stage C h1 RMSE | Stage C h2 MAE | Stage C h2 RMSE | Stage C hybrid MAE | Stage C hybrid RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| energy_per_atom | 0.3606 | 1.9919 | 0.3617 | 1.9403 | 1.7711 | 4.4470 | 1.7583 | 4.2806 | 1.8758 | 4.5314 | 1.7430 | 4.4544 | 0.2762 | 1.9257 | 1.8003 | 4.5045 | 1.8219 | 4.5170 | 1.8225 | 4.5295 |
| formation_energy_per_atom | 0.0800 | 0.1315 | 0.0816 | 0.1338 | 0.2150 | 0.3560 | 0.2137 | 0.3526 | 0.2147 | 0.3546 | 0.1548 | 0.2758 | 0.0330 | 0.0699 | 0.1914 | 0.3276 | 0.1999 | 0.3382 | 0.1987 | 0.3386 |
| energy_above_hull | 0.0644 | 0.1261 | 0.0639 | 0.1258 | 0.1225 | 0.3040 | 0.1241 | 0.3035 | 0.1220 | 0.3067 | 0.1007 | 0.2371 | 0.0307 | 0.0746 | 0.1178 | 0.2959 | 0.1196 | 0.3006 | 0.1170 | 0.3010 |
| band_gap | 0.2308 | 0.4649 | 0.2258 | 0.4533 | 0.5402 | 0.8861 | 0.5505 | 0.8894 | 0.5507 | 0.8916 | 0.4954 | 0.8269 | 0.2252 | 0.4768 | 0.5188 | 0.8623 | 0.5381 | 0.8724 | 0.5313 | 0.8844 |
| cbm | 0.2921 | 0.4463 | 0.2932 | 0.4504 | 0.5940 | 0.8344 | 0.6060 | 0.8452 | 0.6188 | 0.8595 | 0.5645 | 0.8001 | 0.2836 | 0.4540 | 0.5821 | 0.8219 | 0.6012 | 0.8388 | 0.6061 | 0.8562 |
| vbm | 0.2594 | 0.3981 | 0.2539 | 0.3930 | 0.4770 | 0.7014 | 0.4885 | 0.7037 | 0.4935 | 0.7087 | 0.4504 | 0.6603 | 0.2482 | 0.4099 | 0.4679 | 0.6765 | 0.4798 | 0.6908 | 0.4818 | 0.6920 |
| efermi | 0.3834 | 0.7039 | 0.3756 | 0.7086 | 0.6186 | 0.9376 | 0.6257 | 0.9290 | 0.6324 | 0.9393 | 0.5706 | 0.8858 | 0.3848 | 0.7327 | 0.6025 | 0.9131 | 0.6158 | 0.9257 | 0.6153 | 0.9291 |
| volume | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 145.4975 | 317.0639 | N/A | N/A | N/A | N/A | N/A | N/A |
| density | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.0190 | 0.0480 | N/A | N/A | N/A | N/A | N/A | N/A |
| bulk_modulus_vrh | N/A | N/A | N/A | N/A | 8.3487 | 21.6581 | 8.3024 | 20.6097 | 8.5855 | 22.0659 | 8.4485 | 22.0418 | 8.9106 | 20.8483 | 8.4242 | 21.9890 | 8.3932 | 22.0152 | 8.3986 | 22.0342 |
| shear_modulus_vrh | N/A | N/A | N/A | N/A | 10.3827 | 28.6922 | 10.2219 | 28.6012 | 10.6750 | 29.1235 | 10.5170 | 28.9037 | 11.7471 | 30.0261 | 10.5496 | 29.0069 | 10.5251 | 28.9441 | 10.5177 | 28.8953 |
| homogeneous_poisson | N/A | N/A | N/A | N/A | 0.0423 | 0.0915 | 0.0423 | 0.0908 | 0.0441 | 0.0928 | 0.0419 | 0.0922 | 0.0498 | 0.0920 | 0.0456 | 0.0954 | 0.0419 | 0.0935 | 0.0482 | 0.0959 |
| universal_anisotropy | N/A | N/A | N/A | N/A | 1.8196 | 5.1129 | 1.8690 | 5.1320 | 1.8657 | 5.1719 | 1.8376 | 5.0760 | 1.8494 | 5.3148 | 1.8413 | 5.1396 | 1.8231 | 5.0665 | 1.8191 | 5.0665 |

#### VAL Classification: AUROC

| Task | Stage A AUROC | Stage A PyG AUROC | v1 AUROC | v2 AUROC | v3 AUROC | v4 AUROC | zero AUROC | Stage C h1 AUROC | Stage C h2 AUROC | Stage C hybrid AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| is_metal | 0.9575 | 0.9621 | 0.9029 | 0.9035 | 0.9056 | 0.9202 | 0.9498 | 0.9102 | 0.9085 | 0.9071 |
| is_stable | N/A | N/A | 0.8389 | 0.8378 | 0.8335 | 0.8666 | 0.9056 | 0.8517 | 0.8433 | 0.8450 |

#### TEST Regression: MAE + RMSE

| Task | Stage A MAE | Stage A RMSE | Stage A PyG MAE | Stage A PyG RMSE | v1 MAE | v1 RMSE | v2 MAE | v2 RMSE | v3 MAE | v3 RMSE | v4 MAE | v4 RMSE | zero MAE | zero RMSE | Stage C h1 MAE | Stage C h1 RMSE | Stage C h2 MAE | Stage C h2 RMSE | Stage C hybrid MAE | Stage C hybrid RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| energy_per_atom | 0.3355 | 1.8476 | 0.3665 | 2.0169 | 1.7580 | 4.4361 | 1.7272 | 4.2733 | 1.8377 | 4.4588 | 1.7022 | 4.3904 | 0.2480 | 1.5972 | 1.7581 | 4.4301 | 1.7855 | 4.4467 | 1.7828 | 4.4556 |
| formation_energy_per_atom | 0.0809 | 0.1363 | 0.0823 | 0.1375 | 0.2142 | 0.3503 | 0.2149 | 0.3488 | 0.2149 | 0.3510 | 0.1572 | 0.2742 | 0.0329 | 0.0733 | 0.1927 | 0.3229 | 0.2004 | 0.3334 | 0.2000 | 0.3344 |
| energy_above_hull | 0.0642 | 0.1276 | 0.0643 | 0.1276 | 0.1221 | 0.2946 | 0.1237 | 0.2929 | 0.1216 | 0.2963 | 0.1016 | 0.2325 | 0.0312 | 0.0778 | 0.1169 | 0.2851 | 0.1190 | 0.2902 | 0.1164 | 0.2904 |
| band_gap | 0.2379 | 0.4729 | 0.2386 | 0.4785 | 0.5530 | 0.9072 | 0.5658 | 0.9205 | 0.5642 | 0.9168 | 0.5098 | 0.8517 | 0.2378 | 0.5007 | 0.5312 | 0.8843 | 0.5500 | 0.8960 | 0.5439 | 0.9075 |
| cbm | 0.2959 | 0.4527 | 0.2978 | 0.4539 | 0.6057 | 0.8520 | 0.6174 | 0.8721 | 0.6339 | 0.8879 | 0.5777 | 0.8268 | 0.2919 | 0.4745 | 0.5938 | 0.8468 | 0.6166 | 0.8666 | 0.6203 | 0.8829 |
| vbm | 0.2643 | 0.4031 | 0.2603 | 0.3972 | 0.4933 | 0.7182 | 0.4957 | 0.7272 | 0.4996 | 0.7220 | 0.4536 | 0.6697 | 0.2519 | 0.4155 | 0.4723 | 0.6867 | 0.4848 | 0.7041 | 0.4870 | 0.7046 |
| efermi | 0.3930 | 0.7114 | 0.3859 | 0.7144 | 0.6283 | 0.9438 | 0.6319 | 0.9362 | 0.6366 | 0.9428 | 0.5774 | 0.8940 | 0.3997 | 0.7472 | 0.6071 | 0.9191 | 0.6209 | 0.9316 | 0.6187 | 0.9330 |
| volume | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 153.5800 | 340.6306 | N/A | N/A | N/A | N/A | N/A | N/A |
| density | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.0191 | 0.0475 | N/A | N/A | N/A | N/A | N/A | N/A |
| bulk_modulus_vrh | N/A | N/A | N/A | N/A | 7.6187 | 21.6550 | 7.5573 | 21.5679 | 7.7825 | 21.6465 | 7.7127 | 21.7421 | 8.2352 | 22.5729 | 7.6892 | 21.7151 | 7.6371 | 21.6416 | 7.7358 | 21.7033 |
| shear_modulus_vrh | N/A | N/A | N/A | N/A | 9.2220 | 25.5083 | 9.6910 | 25.9960 | 9.6139 | 25.7168 | 9.5324 | 25.6519 | 10.8423 | 26.2116 | 9.5588 | 25.7042 | 9.5285 | 25.6366 | 9.5410 | 25.6431 |
| homogeneous_poisson | N/A | N/A | N/A | N/A | 0.0408 | 0.0897 | 0.0419 | 0.0903 | 0.0422 | 0.0912 | 0.0408 | 0.0896 | 0.0483 | 0.0931 | 0.0438 | 0.0922 | 0.0398 | 0.0899 | 0.0457 | 0.0928 |
| universal_anisotropy | N/A | N/A | N/A | N/A | 2.1316 | 6.8763 | 2.1431 | 6.8245 | 2.1274 | 6.7836 | 2.1128 | 6.7332 | 2.2798 | 7.3862 | 2.1032 | 6.7878 | 2.1219 | 6.8033 | 2.1163 | 6.7868 |

#### TEST Classification: AUROC

| Task | Stage A AUROC | Stage A PyG AUROC | v1 AUROC | v2 AUROC | v3 AUROC | v4 AUROC | zero AUROC | Stage C h1 AUROC | Stage C h2 AUROC | Stage C hybrid AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| is_metal | 0.9581 | 0.9586 | 0.9011 | 0.8992 | 0.9010 | 0.9165 | 0.9467 | 0.9066 | 0.9037 | 0.9028 |
| is_stable | N/A | N/A | 0.8404 | 0.8378 | 0.8352 | 0.8717 | 0.9041 | 0.8558 | 0.8461 | 0.8480 |

### Appendix C. 方法运行索引

| Method | Run | Branch | Backbone | Head Variant | Enabled Tasks | PyG | Freeze Backbone | Config Path | Results Path | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| Stage A | 20260305_210307 | stage_a | graph | grouped | 8 | None | None | artifacts/runs/20260305_210307/config.json | reports/gpt_eval_20260305_210307/results.json | 8-task shared multitask baseline |
| Stage A PyG | 20260308_182946 | stage_a | graph | grouped | 8 | True | False | artifacts/runs_stagea_pyg/20260308_182946/config.json | reports/gpt_eval_20260308_182946/results.json | PyG baseline reference |
| v1 | 20260307_185342 | stage_b | graph | grouped | 13 | None | None | artifacts/runs/20260307_185342/config.json | reports/gpt_eval_20260307_185342/results.json | Stage B multitask baseline |
| v2 | 20260308_001437 | stage_b | graph | grouped | 13 | True | None | artifacts/runs_stageb_v2/20260308_001437/config.json | experiments/stage_b/phase3_enhancements/exp102_stageb_v2_balanced/analysis/results.json | balanced Stage B multitask |
| v3 | 20260308_070539 | stage_b | graph | grouped | 13 | True | None | artifacts/runs_stageb_v3/20260308_070539/config.json | reports/gpt_eval_20260308_070539/results.json | core-guard Stage B multitask |
| v4 | single-task family | stage_b | graph | grouped | 13 | True | True | configs/exp104_stageb_v4_single_task_heads.json | reports/gpt_eval_v4_single_task/*/results.json | shared pretrained backbone + per-task fine-tuning |
| zero | exp106_zero_single_task_family | zero | graph | per_task | 15 | True | False | experiments/zero_version/exp106_zero_single_task_family/metrics/zero_runs.csv | reports/gpt_eval_zero_single_task/*/results.json | fully isolated single-task family |
| Stage C h1 | 20260310_003913 | stage_c | graph | stagec_h1 | 13 | True | True | artifacts/runs_stagec_h1/20260310_003913/config.json | reports/gpt_eval_20260310_003913_stagec_h1/results.json | electronic hierarchical head |
| Stage C h2 | 20260310_005519 | stage_c | graph | stagec_h2 | 13 | True | True | artifacts/runs_stagec_h2/20260310_005519/config.json | reports/gpt_eval_20260310_005519_stagec_h2/results.json | elastic derived hierarchical head |
| Stage C hybrid | 20260310_011108 | stage_c | graph | stagec_hybrid | 13 | True | True | artifacts/runs_stagec_hybrid/20260310_011108/config.json | reports/gpt_eval_20260310_011108_stagec_hybrid/results.json | electronic hierarchy + elastic derived hierarchy |

### Appendix D. 来源文件与关联专题

- 主底表: `reports/full_method_comparison_stagea_stageb_branches.csv`
- 机器可读矩阵: `reports/master_complete_report_method_task_matrix.csv`
- 现有全方法主对比: [FULL_METHOD_COMPARISON_STAGEA_STAGEB_BRANCHES.md](FULL_METHOD_COMPARISON_STAGEA_STAGEB_BRANCHES.md)
- Stage A vs zero 物理专题: [PHYSICAL_ANALYSIS_STAGEA_VS_ZERO_TASK_RELATIONS.md](PHYSICAL_ANALYSIS_STAGEA_VS_ZERO_TASK_RELATIONS.md)
- v4 一页总结: [V4_EXEC_SUMMARY_ONE_PAGER.md](V4_EXEC_SUMMARY_ONE_PAGER.md)
- 图表目录: `reports/figures/master_complete_report/`


# Stage B v2 隔离实验报告 (exp102)

## Run 信息
- Stage B v2: `20260308_001437` (artifacts/runs_stageb_v2)
- Stage B v1: `20260307_185342`
- Stage A baseline: `20260305_210307`

## 总体损失 (val)
| Run | best_val_loss | best_epoch |
|---|---:|---:|
| Stage A | 0.9781 | 38 |
| Stage B v1 | 26.5616 | 42 |
| Stage B v2 | 26.9097 | 42 |

## 关键指标对比 (val)
| Metric | Better | Stage A | Stage B v1 | Stage B v2 | v2-v1 | Trend vs v1 | v2-StageA | Trend vs StageA |
|---|---|---:|---:|---:|---:|---|---:|---|
| energy_per_atom_mae | lower | 0.3606 | 1.7711 | 1.7583 | -0.0129 | improved | +1.3976 | degraded |
| formation_energy_per_atom_mae | lower | 0.0800 | 0.2150 | 0.2137 | -0.0013 | improved | +0.1338 | degraded |
| energy_above_hull_mae | lower | 0.0644 | 0.1225 | 0.1241 | +0.0016 | degraded | +0.0597 | degraded |
| band_gap_mae | lower | 0.2308 | 0.5402 | 0.5505 | +0.0103 | degraded | +0.3197 | degraded |
| cbm_mae | lower | 0.2921 | 0.5940 | 0.6060 | +0.0120 | degraded | +0.3139 | degraded |
| vbm_mae | lower | 0.2594 | 0.4770 | 0.4885 | +0.0115 | degraded | +0.2291 | degraded |
| efermi_mae | lower | 0.3834 | 0.6186 | 0.6257 | +0.0072 | degraded | +0.2424 | degraded |
| is_metal_auroc | higher | 0.9575 | 0.9029 | 0.9035 | +0.0006 | improved | -0.0540 | degraded |
| is_stable_auroc | higher | N/A | 0.8389 | 0.8378 | -0.0010 | degraded | N/A | N/A |
| bulk_modulus_vrh_mae | lower | N/A | 8.3487 | 8.3024 | -0.0463 | improved | N/A | N/A |
| shear_modulus_vrh_mae | lower | N/A | 10.3827 | 10.2219 | -0.1607 | improved | N/A | N/A |
| homogeneous_poisson_mae | lower | N/A | 0.0423 | 0.0423 | -0.0000 | improved | N/A | N/A |
| universal_anisotropy_mae | lower | N/A | 1.8196 | 1.8690 | +0.0494 | degraded | N/A | N/A |

## 结论
- 相比 Stage B v1：improved=6, degraded=7
- 相比 Stage A：improved=0, degraded=8 (仅两边都存在的指标)
- 该 v2 方案（oversample_elastic 2.0）未显著改善核心 Stage A 任务，整体与 v1 接近。

- 详细数值 CSV: `experiments/stage_b/phase3_enhancements/exp102_stageb_v2_balanced/metrics/comparison_v2_vs_v1_stagea.csv`
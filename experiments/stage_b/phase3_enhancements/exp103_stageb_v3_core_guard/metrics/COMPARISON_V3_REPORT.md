# Stage B v3 隔离实验报告 (exp103)

## Run 信息
- Stage B v3: `20260308_070539` (artifacts/runs_stageb_v3)
- Stage B v2: `20260308_001437`
- Stage B v1: `20260307_185342`
- Stage A baseline: `20260305_210307`

## 总体损失 (val)
| Run | best_val_loss | best_epoch |
|---|---:|---:|
| Stage A | 0.9781 | 38 |
| Stage B v1 | 26.5616 | 42 |
| Stage B v2 | 26.9097 | 42 |
| Stage B v3 | 27.8843 | 38 |

## 关键指标对比 (val)
| Metric | Better | Stage A | v1 | v2 | v3 | v3-v2 | Trend vs v2 | v3-v1 | Trend vs v1 | v3-StageA | Trend vs StageA |
|---|---|---:|---:|---:|---:|---:|---|---:|---|---:|---|
| energy_per_atom_mae | lower | 0.3606 | 1.7711 | 1.7583 | 1.8758 | +0.1176 | degraded | +0.1047 | degraded | +1.5152 | degraded |
| formation_energy_per_atom_mae | lower | 0.0800 | 0.2150 | 0.2137 | 0.2147 | +0.0010 | degraded | -0.0003 | improved | +0.1348 | degraded |
| energy_above_hull_mae | lower | 0.0644 | 0.1225 | 0.1241 | 0.1220 | -0.0021 | improved | -0.0006 | improved | +0.0576 | degraded |
| band_gap_mae | lower | 0.2308 | 0.5402 | 0.5505 | 0.5507 | +0.0003 | degraded | +0.0106 | degraded | +0.3199 | degraded |
| cbm_mae | lower | 0.2921 | 0.5940 | 0.6060 | 0.6188 | +0.0127 | degraded | +0.0248 | degraded | +0.3267 | degraded |
| vbm_mae | lower | 0.2594 | 0.4770 | 0.4885 | 0.4935 | +0.0050 | degraded | +0.0165 | degraded | +0.2341 | degraded |
| efermi_mae | lower | 0.3834 | 0.6186 | 0.6257 | 0.6324 | +0.0067 | degraded | +0.0139 | degraded | +0.2490 | degraded |
| is_metal_auroc | higher | 0.9575 | 0.9029 | 0.9035 | 0.9056 | +0.0021 | improved | +0.0027 | improved | -0.0519 | degraded |
| is_stable_auroc | higher | N/A | 0.8389 | 0.8378 | 0.8335 | -0.0043 | degraded | -0.0053 | degraded | N/A | N/A |
| bulk_modulus_vrh_mae | lower | N/A | 8.3487 | 8.3024 | 8.5855 | +0.2831 | degraded | +0.2368 | degraded | N/A | N/A |
| shear_modulus_vrh_mae | lower | N/A | 10.3827 | 10.2219 | 10.6750 | +0.4531 | degraded | +0.2923 | degraded | N/A | N/A |
| homogeneous_poisson_mae | lower | N/A | 0.0423 | 0.0423 | 0.0441 | +0.0018 | degraded | +0.0018 | degraded | N/A | N/A |
| universal_anisotropy_mae | lower | N/A | 1.8196 | 1.8690 | 1.8657 | -0.0033 | improved | +0.0461 | degraded | N/A | N/A |

## 结论
- 相比 v2：improved=3, degraded=10
- 相比 v1：improved=3, degraded=10
- 相比 Stage A：improved=0, degraded=8 (仅两边都存在的指标)
- v3（oversample_elastic=1.0）整体进一步恶化，未带来核心任务恢复。
- 详细数值 CSV: `experiments/stage_b/phase3_enhancements/exp103_stageb_v3_core_guard/metrics/comparison_v3_vs_v2_v1_stagea.csv`
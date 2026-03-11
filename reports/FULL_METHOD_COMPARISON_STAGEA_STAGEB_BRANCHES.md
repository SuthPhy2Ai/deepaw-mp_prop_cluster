# Stage A / Stage B / Stage C / zero 全方法全面对比报告

## 方法范围

- `Stage A` baseline: `20260305_210307`
- `Stage B v1`: `20260307_185342`
- `Stage B v2`: `20260308_001437`
- `Stage B v3`: `20260308_070539`
- `Stage B v4`: `exp104_stageb_v4_single_task_heads` + retry elastic runs
- `zero`: `exp106_zero_single_task_family` + retry sparse runs
- `Stage C h1`: `20260310_003913`
- `Stage C h2`: `20260310_005519`
- `Stage C hybrid`: `20260310_011108`
- `Stage A PyG`: 补充参考方法

## 训练设置汇总

| Method | Run | Stage | Backbone | Head Variant | Hidden | Layers | Batch | Epochs | LR | WD | PyG | Enabled Tasks | Freeze Backbone | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---:|---|---|
| Stage A | 20260305_210307 | a | graph | grouped | 256 | 6 | 64 | 50 | 0.0001 | 1e-05 | None | 8 | None |  |
| Stage A PyG | 20260308_182946 | a | graph | grouped | 256 | 6 | 64 | 50 | 0.0001 | 1e-05 | True | 8 | False |  |
| v1 | 20260307_185342 | b | graph | grouped | 256 | 6 | 64 | 50 | 0.0001 | 1e-05 | None | 13 | None |  |
| v2 | 20260308_001437 | b | graph | grouped | 256 | 6 | 64 | 50 | 0.0001 | 1e-05 | True | 13 | None |  |
| v3 | 20260308_070539 | b | graph | grouped | 256 | 6 | 64 | 50 | 0.0001 | 1e-05 | True | 13 | None |  |
| Stage C h1 | 20260310_003913 | b | graph | stagec_h1 | 256 | 6 | 64 | 35 | 0.0002 | 1e-05 | True | 13 | True |  |
| Stage C h2 | 20260310_005519 | b | graph | stagec_h2 | 256 | 6 | 64 | 35 | 0.0002 | 1e-05 | True | 13 | True |  |
| Stage C hybrid | 20260310_011108 | b | graph | stagec_hybrid | 256 | 6 | 64 | 35 | 0.0002 | 1e-05 | True | 13 | True |  |
| v4 | single-task family | b | graph | grouped | 256 | 6 | 64 | 30 | 0.0002 | 1e-05 | True | 13 | True | shared backbone + per-task head |
| zero | exp106_zero_single_task_family | full | graph | per_task | 128-320 | 4-7 | 32-64 | 50-90 | 8e-5~2e-4 | 1e-05 | True | 15 | False | fully isolated per-task model |

## 数据说明

- `loss` 为任务级 loss：回归用 `SmoothL1`，分类用 `BCE`。
- `score` 为主分数：回归用 `R2`，分类用 `ACC`。
- `v4` 与 `zero` 是单任务训练；`Stage C` 是共享 backbone 下的新 head 架构实验。
- `volume` 与 `density` 仍然只在 `zero` 中训练，其余方法按 `N/A` 处理。

## VAL 主分数统一横比

| Task | Type | Better | Stage A | v1 | v2 | v3 | v4 | zero | Stage C h1 | Stage C h2 | Stage C hybrid | Best |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| energy_per_atom | regression | higher | 0.9373 | 0.6876 | 0.7105 | 0.6756 | 0.6865 | 0.9414 | 0.6794 | 0.6776 | 0.6759 | zero |
| formation_energy_per_atom | regression | higher | 0.9880 | 0.9119 | 0.9136 | 0.9126 | 0.9472 | 0.9966 | 0.9254 | 0.9206 | 0.9204 | zero |
| energy_above_hull | regression | higher | 0.9237 | 0.5565 | 0.5580 | 0.5487 | 0.7302 | 0.9733 | 0.5797 | 0.5663 | 0.5652 | zero |
| band_gap | regression | higher | 0.9038 | 0.6506 | 0.6480 | 0.6462 | 0.6957 | 0.8989 | 0.6691 | 0.6613 | 0.6519 | Stage A |
| cbm | regression | higher | 0.9625 | 0.8689 | 0.8655 | 0.8609 | 0.8795 | 0.9612 | 0.8729 | 0.8676 | 0.8620 | Stage A |
| vbm | regression | higher | 0.9722 | 0.9137 | 0.9132 | 0.9119 | 0.9235 | 0.9705 | 0.9197 | 0.9163 | 0.9160 | Stage A |
| efermi | regression | higher | 0.9345 | 0.8837 | 0.8858 | 0.8833 | 0.8962 | 0.9290 | 0.8897 | 0.8867 | 0.8858 | Stage A |
| volume | regression | higher | N/A | N/A | N/A | N/A | N/A | 0.6904 | N/A | N/A | N/A | zero |
| density | regression | higher | N/A | N/A | N/A | N/A | N/A | 0.9997 | N/A | N/A | N/A | zero |
| bulk_modulus_vrh | regression | higher | N/A | 0.9166 | 0.9245 | 0.9134 | 0.9136 | 0.9227 | 0.9140 | 0.9138 | 0.9137 | v2 |
| shear_modulus_vrh | regression | higher | N/A | 0.6908 | 0.6928 | 0.6815 | 0.6862 | 0.6614 | 0.6840 | 0.6854 | 0.6864 | v2 |
| homogeneous_poisson | regression | higher | N/A | 0.1653 | 0.1789 | 0.1418 | 0.1532 | 0.1557 | 0.0924 | 0.1295 | 0.0830 | v2 |
| universal_anisotropy | regression | higher | N/A | 0.3543 | 0.3495 | 0.3393 | 0.3636 | 0.3023 | 0.3475 | 0.3660 | 0.3660 | Stage C hybrid |
| is_metal | classification | higher | 0.8902 | 0.8217 | 0.8222 | 0.8232 | 0.8429 | 0.8767 | 0.8273 | 0.8302 | 0.8267 | Stage A |
| is_stable | classification | higher | N/A | 0.8106 | 0.8138 | 0.8095 | 0.8248 | 0.8511 | 0.8171 | 0.8124 | 0.8127 | zero |

## TEST 主分数统一横比

| Task | Type | Better | Stage A | v1 | v2 | v3 | v4 | zero | Stage C h1 | Stage C h2 | Stage C hybrid | Best |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| energy_per_atom | regression | higher | 0.9461 | 0.6895 | 0.7118 | 0.6863 | 0.6958 | 0.9597 | 0.6903 | 0.6880 | 0.6867 | zero |
| formation_energy_per_atom | regression | higher | 0.9872 | 0.9153 | 0.9160 | 0.9150 | 0.9481 | 0.9963 | 0.9280 | 0.9233 | 0.9228 | zero |
| energy_above_hull | regression | higher | 0.9211 | 0.5791 | 0.5839 | 0.5743 | 0.7379 | 0.9707 | 0.6057 | 0.5916 | 0.5911 | zero |
| band_gap | regression | higher | 0.9039 | 0.6463 | 0.6358 | 0.6387 | 0.6882 | 0.8923 | 0.6639 | 0.6550 | 0.6460 | Stage A |
| cbm | regression | higher | 0.9607 | 0.8608 | 0.8541 | 0.8488 | 0.8689 | 0.9568 | 0.8625 | 0.8560 | 0.8505 | Stage A |
| vbm | regression | higher | 0.9713 | 0.9090 | 0.9066 | 0.9080 | 0.9208 | 0.9695 | 0.9168 | 0.9125 | 0.9124 | Stage A |
| efermi | regression | higher | 0.9350 | 0.8856 | 0.8874 | 0.8858 | 0.8973 | 0.9283 | 0.8915 | 0.8885 | 0.8882 | Stage A |
| volume | regression | higher | N/A | N/A | N/A | N/A | N/A | 0.6746 | N/A | N/A | N/A | zero |
| density | regression | higher | N/A | N/A | N/A | N/A | N/A | 0.9997 | N/A | N/A | N/A | zero |
| bulk_modulus_vrh | regression | higher | N/A | 0.9173 | 0.9180 | 0.9174 | 0.9167 | 0.9102 | 0.9169 | 0.9174 | 0.9170 | v2 |
| shear_modulus_vrh | regression | higher | N/A | 0.7312 | 0.7209 | 0.7268 | 0.7282 | 0.7162 | 0.7271 | 0.7285 | 0.7284 | v1 |
| homogeneous_poisson | regression | higher | N/A | 0.2318 | 0.2222 | 0.2068 | 0.2336 | 0.1735 | 0.1895 | 0.2282 | 0.1781 | v4 |
| universal_anisotropy | regression | higher | N/A | 0.3524 | 0.3621 | 0.3697 | 0.3790 | 0.2528 | 0.3689 | 0.3660 | 0.3691 | v4 |
| is_metal | classification | higher | 0.8935 | 0.8214 | 0.8212 | 0.8251 | 0.8425 | 0.8782 | 0.8301 | 0.8268 | 0.8254 | Stage A |
| is_stable | classification | higher | N/A | 0.8160 | 0.8113 | 0.8123 | 0.8263 | 0.8510 | 0.8197 | 0.8154 | 0.8152 | zero |

## VAL 胜场统计

| Method | Win Tasks |
|---|---:|
| Stage A | 5 |
| v1 | 0 |
| v2 | 3 |
| v3 | 0 |
| v4 | 0 |
| zero | 6 |
| Stage C h1 | 0 |
| Stage C h2 | 0 |
| Stage C hybrid | 1 |

## TEST 胜场统计

| Method | Win Tasks |
|---|---:|
| Stage A | 5 |
| v1 | 1 |
| v2 | 1 |
| v3 | 0 |
| v4 | 2 |
| zero | 6 |
| Stage C h1 | 0 |
| Stage C h2 | 0 |
| Stage C hybrid | 0 |

## 单独阅读单元：Stage A vs zero

### VAL: Stage A 胜过 zero

| Task | Stage A score | zero score | Delta(StageA-zero) |
|---|---:|---:|---:|
| is_metal | 0.8902 | 0.8767 | +0.0134 |
| efermi | 0.9345 | 0.9290 | +0.0055 |
| band_gap | 0.9038 | 0.8989 | +0.0050 |
| vbm | 0.9722 | 0.9705 | +0.0017 |
| cbm | 0.9625 | 0.9612 | +0.0013 |

### VAL: zero 胜过 Stage A

| Task | zero score | Stage A score | Delta(zero-StageA) |
|---|---:|---:|---:|
| energy_above_hull | 0.9733 | 0.9237 | +0.0496 |
| formation_energy_per_atom | 0.9966 | 0.9880 | +0.0086 |
| energy_per_atom | 0.9414 | 0.9373 | +0.0041 |

### TEST: Stage A 胜过 zero

| Task | Stage A score | zero score | Delta(StageA-zero) |
|---|---:|---:|---:|
| is_metal | 0.8935 | 0.8782 | +0.0153 |
| band_gap | 0.9039 | 0.8923 | +0.0116 |
| efermi | 0.9350 | 0.9283 | +0.0067 |
| cbm | 0.9607 | 0.9568 | +0.0039 |
| vbm | 0.9713 | 0.9695 | +0.0018 |

### TEST: zero 胜过 Stage A

| Task | zero score | Stage A score | Delta(zero-StageA) |
|---|---:|---:|---:|
| energy_above_hull | 0.9707 | 0.9211 | +0.0496 |
| energy_per_atom | 0.9597 | 0.9461 | +0.0136 |
| formation_energy_per_atom | 0.9963 | 0.9872 | +0.0091 |

## TRAIN 回归任务：task loss + R2

| Task | Stage A loss | Stage A R2 | v1 loss | v1 R2 | v2 loss | v2 R2 | v3 loss | v3 R2 | v4 loss | v4 R2 | zero loss | zero R2 | Stage C h1 loss | Stage C h1 R2 | Stage C h2 loss | Stage C h2 R2 | Stage C hybrid loss | Stage C hybrid R2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| energy_per_atom | 0.0872 | 0.9789 | 1.3926 | 0.7220 | 1.3809 | 0.7323 | 1.4773 | 0.7116 | 1.3500 | 0.7252 | 0.0668 | 0.9835 | 1.4138 | 0.7170 | 1.4365 | 0.7141 | 1.4367 | 0.7125 |
| formation_energy_per_atom | 0.0052 | 0.9925 | 0.0493 | 0.9172 | 0.0488 | 0.9184 | 0.0495 | 0.9175 | 0.0278 | 0.9526 | 0.0005 | 0.9992 | 0.0414 | 0.9302 | 0.0443 | 0.9256 | 0.0443 | 0.9254 |
| energy_above_hull | 0.0046 | 0.9536 | 0.0292 | 0.5906 | 0.0290 | 0.5946 | 0.0297 | 0.5843 | 0.0185 | 0.7540 | 0.0005 | 0.9948 | 0.0273 | 0.6160 | 0.0283 | 0.6027 | 0.0283 | 0.6016 |
| band_gap | 0.0350 | 0.9646 | 0.2898 | 0.6578 | 0.2955 | 0.6549 | 0.2988 | 0.6466 | 0.2573 | 0.7033 | 0.0264 | 0.9705 | 0.2782 | 0.6719 | 0.2889 | 0.6632 | 0.2885 | 0.6552 |
| cbm | 0.0349 | 0.9855 | 0.2766 | 0.8678 | 0.2833 | 0.8650 | 0.2916 | 0.8609 | 0.2476 | 0.8823 | 0.0155 | 0.9933 | 0.2641 | 0.8740 | 0.2773 | 0.8683 | 0.2841 | 0.8627 |
| vbm | 0.0296 | 0.9887 | 0.1980 | 0.9161 | 0.2015 | 0.9144 | 0.2066 | 0.9129 | 0.1743 | 0.9268 | 0.0107 | 0.9957 | 0.1870 | 0.9220 | 0.1964 | 0.9175 | 0.1964 | 0.9179 |
| efermi | 0.1265 | 0.9474 | 0.3099 | 0.8855 | 0.3118 | 0.8866 | 0.3201 | 0.8832 | 0.2748 | 0.8975 | 0.0766 | 0.9714 | 0.2983 | 0.8900 | 0.3085 | 0.8866 | 0.3078 | 0.8861 |
| volume | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 82.2914 | 0.8379 | N/A | N/A | N/A | N/A | N/A | N/A |
| density | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.0003 | 0.9999 | N/A | N/A | N/A | N/A | N/A | N/A |
| bulk_modulus_vrh | N/A | N/A | 2.8352 | 0.9660 | 3.0462 | 0.9613 | 3.4925 | 0.9602 | 3.1844 | 0.9614 | 3.7203 | 0.9582 | 3.1199 | 0.9615 | 3.1513 | 0.9618 | 3.1921 | 0.9620 |
| shear_modulus_vrh | N/A | N/A | 4.5051 | 0.7177 | 4.6394 | 0.7168 | 5.2798 | 0.7055 | 4.9824 | 0.7084 | 6.5180 | 0.6928 | 4.9694 | 0.7076 | 4.9487 | 0.7087 | 4.9654 | 0.7097 |
| homogeneous_poisson | N/A | N/A | 0.0039 | 0.3066 | 0.0040 | 0.2962 | 0.0041 | 0.2806 | 0.0037 | 0.3478 | 0.0039 | 0.3293 | 0.0040 | 0.2897 | 0.0038 | 0.3329 | 0.0042 | 0.2656 |
| universal_anisotropy | N/A | N/A | 1.1154 | 0.5518 | 1.0400 | 0.6026 | 1.2003 | 0.5389 | 1.0887 | 0.5898 | 1.1603 | 0.5347 | 1.1564 | 0.5555 | 1.1280 | 0.5829 | 1.1232 | 0.5846 |

## TRAIN 分类任务：task loss + ACC

| Task | Stage A loss | Stage A ACC | v1 loss | v1 ACC | v2 loss | v2 ACC | v3 loss | v3 ACC | v4 loss | v4 ACC | zero loss | zero ACC | Stage C h1 loss | Stage C h1 ACC | Stage C h2 loss | Stage C h2 ACC | Stage C hybrid loss | Stage C hybrid ACC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| is_metal | 0.0911 | 0.9640 | 0.3766 | 0.8276 | 0.3764 | 0.8265 | 0.3770 | 0.8270 | 0.3323 | 0.8514 | 0.2150 | 0.9112 | 0.3646 | 0.8322 | 0.3699 | 0.8320 | 0.3720 | 0.8299 |
| is_stable | N/A | N/A | 0.3838 | 0.8180 | 0.3843 | 0.8167 | 0.3874 | 0.8143 | 0.3439 | 0.8370 | 0.2513 | 0.8872 | 0.3671 | 0.8248 | 0.3776 | 0.8191 | 0.3756 | 0.8197 |

## VAL 回归任务：task loss + R2

| Task | Stage A loss | Stage A R2 | v1 loss | v1 R2 | v2 loss | v2 R2 | v3 loss | v3 R2 | v4 loss | v4 R2 | zero loss | zero R2 | Stage C h1 loss | Stage C h1 R2 | Stage C h2 loss | Stage C h2 R2 | Stage C hybrid loss | Stage C hybrid R2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| energy_per_atom | 0.2437 | 0.9373 | 1.4905 | 0.6876 | 1.4711 | 0.7105 | 1.5850 | 0.6756 | 1.4716 | 0.6865 | 0.1984 | 0.9414 | 1.5276 | 0.6794 | 1.5449 | 0.6776 | 1.5460 | 0.6759 |
| formation_energy_per_atom | 0.0084 | 0.9880 | 0.0516 | 0.9119 | 0.0506 | 0.9136 | 0.0515 | 0.9126 | 0.0307 | 0.9472 | 0.0024 | 0.9966 | 0.0435 | 0.9254 | 0.0465 | 0.9206 | 0.0465 | 0.9204 |
| energy_above_hull | 0.0076 | 0.9237 | 0.0313 | 0.5565 | 0.0315 | 0.5580 | 0.0320 | 0.5487 | 0.0207 | 0.7302 | 0.0027 | 0.9733 | 0.0298 | 0.5797 | 0.0307 | 0.5663 | 0.0307 | 0.5652 |
| band_gap | 0.0868 | 0.9038 | 0.2875 | 0.6506 | 0.2943 | 0.6480 | 0.2945 | 0.6462 | 0.2595 | 0.6957 | 0.0868 | 0.8989 | 0.2750 | 0.6691 | 0.2856 | 0.6613 | 0.2856 | 0.6519 |
| cbm | 0.0872 | 0.9625 | 0.2782 | 0.8689 | 0.2870 | 0.8655 | 0.2949 | 0.8609 | 0.2560 | 0.8795 | 0.0880 | 0.9612 | 0.2689 | 0.8729 | 0.2818 | 0.8676 | 0.2882 | 0.8620 |
| vbm | 0.0708 | 0.9722 | 0.1983 | 0.9137 | 0.2029 | 0.9132 | 0.2060 | 0.9119 | 0.1799 | 0.9235 | 0.0715 | 0.9705 | 0.1894 | 0.9197 | 0.1970 | 0.9163 | 0.1981 | 0.9160 |
| efermi | 0.1604 | 0.9345 | 0.3087 | 0.8837 | 0.3099 | 0.8858 | 0.3154 | 0.8833 | 0.2748 | 0.8962 | 0.1712 | 0.9290 | 0.2953 | 0.8897 | 0.3045 | 0.8867 | 0.3047 | 0.8858 |
| volume | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 145.0043 | 0.6904 | N/A | N/A | N/A | N/A | N/A | N/A |
| density | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.0011 | 0.9997 | N/A | N/A | N/A | N/A | N/A | N/A |
| bulk_modulus_vrh | N/A | N/A | 7.8839 | 0.9166 | 7.8362 | 0.9245 | 8.1216 | 0.9134 | 7.9840 | 0.9136 | 8.4433 | 0.9227 | 7.9600 | 0.9140 | 7.9303 | 0.9138 | 7.9382 | 0.9137 |
| shear_modulus_vrh | N/A | N/A | 9.9112 | 0.6908 | 9.7514 | 0.6928 | 10.2040 | 0.6815 | 10.0441 | 0.6862 | 11.2718 | 0.6614 | 10.0779 | 0.6840 | 10.0513 | 0.6854 | 10.0452 | 0.6864 |
| homogeneous_poisson | N/A | N/A | 0.0041 | 0.1653 | 0.0040 | 0.1789 | 0.0042 | 0.1418 | 0.0042 | 0.1532 | 0.0042 | 0.1557 | 0.0045 | 0.0924 | 0.0043 | 0.1295 | 0.0045 | 0.0830 |
| universal_anisotropy | N/A | N/A | 1.4907 | 0.3543 | 1.5469 | 0.3495 | 1.5425 | 0.3393 | 1.5106 | 0.3636 | 1.5448 | 0.3023 | 1.5211 | 0.3475 | 1.4995 | 0.3660 | 1.4961 | 0.3660 |

## VAL 分类任务：task loss + ACC

| Task | Stage A loss | Stage A ACC | v1 loss | v1 ACC | v2 loss | v2 ACC | v3 loss | v3 ACC | v4 loss | v4 ACC | zero loss | zero ACC | Stage C h1 loss | Stage C h1 ACC | Stage C h2 loss | Stage C h2 ACC | Stage C hybrid loss | Stage C hybrid ACC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| is_metal | 0.3369 | 0.8902 | 0.3837 | 0.8217 | 0.3822 | 0.8222 | 0.3798 | 0.8232 | 0.3493 | 0.8429 | 0.2845 | 0.8767 | 0.3700 | 0.8273 | 0.3739 | 0.8302 | 0.3764 | 0.8267 |
| is_stable | N/A | N/A | 0.3926 | 0.8106 | 0.3927 | 0.8138 | 0.3964 | 0.8095 | 0.3643 | 0.8248 | 0.3172 | 0.8511 | 0.3787 | 0.8171 | 0.3872 | 0.8124 | 0.3861 | 0.8127 |

## TEST 回归任务：task loss + R2

| Task | Stage A loss | Stage A R2 | v1 loss | v1 R2 | v2 loss | v2 R2 | v3 loss | v3 R2 | v4 loss | v4 R2 | zero loss | zero R2 | Stage C h1 loss | Stage C h1 R2 | Stage C h2 loss | Stage C h2 R2 | Stage C hybrid loss | Stage C hybrid R2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| energy_per_atom | 0.2198 | 0.9461 | 1.4779 | 0.6895 | 1.4406 | 0.7118 | 1.5453 | 0.6863 | 1.4317 | 0.6958 | 0.1710 | 0.9597 | 1.4849 | 0.6903 | 1.5072 | 0.6880 | 1.5057 | 0.6867 |
| formation_energy_per_atom | 0.0086 | 0.9872 | 0.0510 | 0.9153 | 0.0510 | 0.9160 | 0.0516 | 0.9150 | 0.0313 | 0.9481 | 0.0024 | 0.9963 | 0.0434 | 0.9280 | 0.0463 | 0.9233 | 0.0465 | 0.9228 |
| energy_above_hull | 0.0074 | 0.9211 | 0.0305 | 0.5791 | 0.0304 | 0.5839 | 0.0310 | 0.5743 | 0.0206 | 0.7379 | 0.0028 | 0.9707 | 0.0285 | 0.6057 | 0.0296 | 0.5916 | 0.0296 | 0.5911 |
| band_gap | 0.0903 | 0.9039 | 0.2990 | 0.6463 | 0.3078 | 0.6358 | 0.3066 | 0.6387 | 0.2719 | 0.6882 | 0.0951 | 0.8923 | 0.2865 | 0.6639 | 0.2966 | 0.6550 | 0.2975 | 0.6460 |
| cbm | 0.0892 | 0.9607 | 0.2876 | 0.8608 | 0.2983 | 0.8541 | 0.3096 | 0.8488 | 0.2702 | 0.8689 | 0.0944 | 0.9568 | 0.2814 | 0.8625 | 0.2966 | 0.8560 | 0.3021 | 0.8505 |
| vbm | 0.0732 | 0.9713 | 0.2080 | 0.9090 | 0.2100 | 0.9066 | 0.2116 | 0.9080 | 0.1833 | 0.9208 | 0.0742 | 0.9695 | 0.1930 | 0.9168 | 0.2022 | 0.9125 | 0.2025 | 0.9124 |
| efermi | 0.1674 | 0.9350 | 0.3160 | 0.8856 | 0.3158 | 0.8874 | 0.3200 | 0.8858 | 0.2817 | 0.8973 | 0.1807 | 0.9283 | 0.3009 | 0.8915 | 0.3102 | 0.8885 | 0.3093 | 0.8882 |
| volume | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 153.0870 | 0.6746 | N/A | N/A | N/A | N/A | N/A | N/A |
| density | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.0011 | 0.9997 | N/A | N/A | N/A | N/A | N/A | N/A |
| bulk_modulus_vrh | N/A | N/A | 7.1517 | 0.9173 | 7.0904 | 0.9180 | 7.3146 | 0.9174 | 7.2506 | 0.9167 | 7.7669 | 0.9102 | 7.2245 | 0.9169 | 7.1746 | 0.9174 | 7.2697 | 0.9170 |
| shear_modulus_vrh | N/A | N/A | 8.7552 | 0.7312 | 9.2176 | 0.7209 | 9.1389 | 0.7268 | 9.0616 | 0.7282 | 10.3661 | 0.7162 | 9.0895 | 0.7271 | 9.0575 | 0.7285 | 9.0730 | 0.7284 |
| homogeneous_poisson | N/A | N/A | 0.0040 | 0.2318 | 0.0041 | 0.2222 | 0.0041 | 0.2068 | 0.0040 | 0.2336 | 0.0043 | 0.1735 | 0.0042 | 0.1895 | 0.0040 | 0.2282 | 0.0043 | 0.1781 |
| universal_anisotropy | N/A | N/A | 1.8163 | 0.3524 | 1.8367 | 0.3621 | 1.8140 | 0.3697 | 1.7957 | 0.3790 | 1.9763 | 0.2528 | 1.7946 | 0.3689 | 1.8051 | 0.3660 | 1.7988 | 0.3691 |

## TEST 分类任务：task loss + ACC

| Task | Stage A loss | Stage A ACC | v1 loss | v1 ACC | v2 loss | v2 ACC | v3 loss | v3 ACC | v4 loss | v4 ACC | zero loss | zero ACC | Stage C h1 loss | Stage C h1 ACC | Stage C h2 loss | Stage C h2 ACC | Stage C hybrid loss | Stage C hybrid ACC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| is_metal | 0.3289 | 0.8935 | 0.3856 | 0.8214 | 0.3879 | 0.8212 | 0.3866 | 0.8251 | 0.3563 | 0.8425 | 0.2924 | 0.8782 | 0.3756 | 0.8301 | 0.3812 | 0.8268 | 0.3832 | 0.8254 |
| is_stable | N/A | N/A | 0.3881 | 0.8160 | 0.3895 | 0.8113 | 0.3925 | 0.8123 | 0.3564 | 0.8263 | 0.3169 | 0.8510 | 0.3722 | 0.8197 | 0.3819 | 0.8154 | 0.3801 | 0.8152 |

## 补充参考：Stage A PyG

| Split | Task | loss | score | score_name |
|---|---|---:|---:|---|
| train | energy_per_atom | 0.0987 | 0.9766 | r2 |
| train | formation_energy_per_atom | 0.0056 | 0.9919 | r2 |
| train | energy_above_hull | 0.0049 | 0.9500 | r2 |
| train | band_gap | 0.0377 | 0.9611 | r2 |
| train | cbm | 0.0373 | 0.9843 | r2 |
| train | vbm | 0.0322 | 0.9877 | r2 |
| train | efermi | 0.1297 | 0.9454 | r2 |
| train | is_metal | 0.1002 | 0.9608 | acc |
| val | energy_per_atom | 0.2405 | 0.9405 | r2 |
| val | formation_energy_per_atom | 0.0086 | 0.9876 | r2 |
| val | energy_above_hull | 0.0074 | 0.9241 | r2 |
| val | band_gap | 0.0829 | 0.9085 | r2 |
| val | cbm | 0.0884 | 0.9618 | r2 |
| val | vbm | 0.0683 | 0.9729 | r2 |
| val | efermi | 0.1594 | 0.9336 | r2 |
| val | is_metal | 0.3004 | 0.8972 | acc |
| test | energy_per_atom | 0.2449 | 0.9358 | r2 |
| test | formation_energy_per_atom | 0.0088 | 0.9869 | r2 |
| test | energy_above_hull | 0.0075 | 0.9211 | r2 |
| test | band_gap | 0.0911 | 0.9016 | r2 |
| test | cbm | 0.0906 | 0.9605 | r2 |
| test | vbm | 0.0710 | 0.9722 | r2 |
| test | efermi | 0.1664 | 0.9344 | r2 |
| test | is_metal | 0.3196 | 0.8933 | acc |

## 单任务家族 Run 索引

| Family | Task | Run Dir | Best Epoch | Best Val Loss |
|---|---|---|---:|---:|
| v4 | band_gap | `artifacts/runs_stageb_v4/band_gap/20260308_084843` | 28 | 0.2595 |
| v4 | bulk_modulus_vrh | `artifacts/runs_stageb_v4_retry_elastic/bulk_modulus_vrh/20260308_162533` | 17 | 5.8835 |
| v4 | cbm | `artifacts/runs_stageb_v4/cbm/20260308_085901` | 28 | 0.2513 |
| v4 | efermi | `artifacts/runs_stageb_v4/efermi/20260308_091936` | 30 | 0.2748 |
| v4 | energy_above_hull | `artifacts/runs_stageb_v4/energy_above_hull/20260308_083825` | 30 | 0.0207 |
| v4 | energy_per_atom | `artifacts/runs_stageb_v4/energy_per_atom/20260308_081747` | 30 | 1.4716 |
| v4 | formation_energy_per_atom | `artifacts/runs_stageb_v4/formation_energy_per_atom/20260308_082806` | 30 | 0.0307 |
| v4 | homogeneous_poisson | `artifacts/runs_stageb_v4_retry_elastic/homogeneous_poisson/20260308_164612` | 9 | 0.0021 |
| v4 | is_metal | `artifacts/runs_stageb_v4/is_metal/20260308_092955` | 29 | 0.3493 |
| v4 | is_stable | `artifacts/runs_stageb_v4/is_stable/20260308_094013` | 30 | 0.3643 |
| v4 | shear_modulus_vrh | `artifacts/runs_stageb_v4_retry_elastic/shear_modulus_vrh/20260308_163553` | 17 | 6.2637 |
| v4 | universal_anisotropy | `artifacts/runs_stageb_v4_retry_elastic/universal_anisotropy/20260308_165632` | 28 | 0.9685 |
| v4 | vbm | `artifacts/runs_stageb_v4/vbm/20260308_090919` | 24 | 0.1774 |
| zero | band_gap | `artifacts/runs_zero/band_gap/20260308_210954` | 37 | 0.0868 |
| zero | bulk_modulus_vrh | `artifacts/runs_zero/bulk_modulus_vrh/20260309_024714` | 65 | 4.5966 |
| zero | cbm | `artifacts/runs_zero/cbm/20260308_214108` | 47 | 0.0914 |
| zero | density | `artifacts/runs_zero/density/20260309_013312` | 70 | 0.0011 |
| zero | efermi | `artifacts/runs_zero/efermi/20260308_224533` | 33 | 0.1712 |
| zero | energy_above_hull | `artifacts/runs_zero/energy_above_hull/20260308_203846` | 45 | 0.0027 |
| zero | energy_per_atom | `artifacts/runs_zero/energy_per_atom/20260308_193626` | 47 | 0.1984 |
| zero | formation_energy_per_atom | `artifacts/runs_zero/formation_energy_per_atom/20260308_200735` | 50 | 0.0024 |
| zero | homogeneous_poisson | `artifacts/runs_zero_retry_sparse/homogeneous_poisson/20260309_132811` | 22 | 0.0016 |
| zero | is_metal | `artifacts/runs_zero/is_metal/20260308_231645` | 14 | 0.2845 |
| zero | is_stable | `artifacts/runs_zero/is_stable/20260308_234757` | 12 | 0.3172 |
| zero | shear_modulus_vrh | `artifacts/runs_zero/shear_modulus_vrh/20260309_032534` | 52 | 5.4769 |
| zero | universal_anisotropy | `artifacts/runs_zero_retry_sparse/universal_anisotropy/20260309_140536` | 30 | 0.7314 |
| zero | vbm | `artifacts/runs_zero/vbm/20260308_221320` | 48 | 0.0737 |
| zero | volume | `artifacts/runs_zero/volume/20260309_001909` | 65 | 144.9968 |

## 读法摘要

- `VAL` 胜场最多的方法：`zero`，共 `6` 个任务。
- `TEST` 胜场最多的方法：`zero`，共 `6` 个任务。
- Stage C 三组都已并入统一横比，可以直接和 Stage A / Stage B / zero 同口径比较。
- 如果要看 Stage C head 结构本身的价值，应重点比 `Stage C h1 / h2 / hybrid` 相对 `v3` 的变化。

## 关联文件

- 明细 CSV: `reports/full_method_comparison_stagea_stageb_branches.csv`
- zero 单任务评估目录: `reports/gpt_eval_zero_single_task/`
- Stage C h1 评估目录: `reports/gpt_eval_20260310_003913_stagec_h1/`
- Stage C h2 评估目录: `reports/gpt_eval_20260310_005519_stagec_h2/`
- Stage C hybrid 评估目录: `reports/gpt_eval_20260310_011108_stagec_hybrid/`

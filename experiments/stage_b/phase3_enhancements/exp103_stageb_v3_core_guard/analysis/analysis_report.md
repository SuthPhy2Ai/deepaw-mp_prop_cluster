# Model Performance Analysis Report

Generated: 2026-03-08 07:41:57

================================================================================

## Model Configuration

- **Backbone**: graph
- **Hidden Dimension**: 256
- **Layers**: 6
- **Learning Rate**: 0.0001
- **Batch Size**: 64
- **Training Epochs**: 50
- **Stage**: b
- **Enabled Tasks**: 13

## Overall Performance

| Split | Loss |
|-------|------|
| Train | 1.2266 |
| Val | 2.0035 |
| Test | 1.8798 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| energy_per_atom | train | 1.7658 | 4.2699 | 0.712 | 123902 |
| energy_per_atom | val | 1.8758 | 4.5314 | 0.676 | 15487 |
| energy_per_atom | test | 1.8377 | 4.4588 | 0.686 | 15489 |
| formation_energy_per_atom | train | 0.2116 | 0.3455 | 0.917 | 123902 |
| formation_energy_per_atom | val | 0.2147 | 0.3546 | 0.913 | 15487 |
| formation_energy_per_atom | test | 0.2149 | 0.3510 | 0.915 | 15489 |
| energy_above_hull | train | 0.1189 | 0.2919 | 0.584 | 123902 |
| energy_above_hull | val | 0.1220 | 0.3067 | 0.549 | 15487 |
| energy_above_hull | test | 0.1216 | 0.2963 | 0.574 | 15489 |
| band_gap | train | 0.5555 | 0.8986 | 0.647 | 123902 |
| band_gap | val | 0.5507 | 0.8916 | 0.646 | 15487 |
| band_gap | test | 0.5642 | 0.9168 | 0.639 | 15489 |
| cbm | train | 0.6126 | 0.8563 | 0.861 | 71574 |
| cbm | val | 0.6188 | 0.8595 | 0.861 | 8836 |
| cbm | test | 0.6339 | 0.8879 | 0.849 | 8928 |
| vbm | train | 0.4926 | 0.7059 | 0.913 | 71574 |
| vbm | val | 0.4935 | 0.7087 | 0.912 | 8836 |
| vbm | test | 0.4996 | 0.7220 | 0.908 | 8928 |
| efermi | train | 0.6364 | 0.9448 | 0.883 | 123856 |
| efermi | val | 0.6324 | 0.9393 | 0.883 | 15485 |
| efermi | test | 0.6366 | 0.9428 | 0.886 | 15483 |
| bulk_modulus_vrh | train | 3.9368 | 15.0603 | 0.960 | 10217 |
| bulk_modulus_vrh | val | 8.5855 | 22.0659 | 0.913 | 1348 |
| bulk_modulus_vrh | test | 7.7825 | 21.6465 | 0.917 | 1287 |
| shear_modulus_vrh | train | 5.7265 | 28.2641 | 0.705 | 9685 |
| shear_modulus_vrh | val | 10.6750 | 29.1235 | 0.681 | 1293 |
| shear_modulus_vrh | test | 9.6139 | 25.7168 | 0.727 | 1227 |
| homogeneous_poisson | train | 0.0340 | 0.0916 | 0.281 | 9752 |
| homogeneous_poisson | val | 0.0441 | 0.0928 | 0.142 | 1304 |
| homogeneous_poisson | test | 0.0422 | 0.0912 | 0.207 | 1236 |
| universal_anisotropy | train | 1.4967 | 5.2870 | 0.539 | 9057 |
| universal_anisotropy | val | 1.8657 | 5.1719 | 0.339 | 1213 |
| universal_anisotropy | test | 2.1274 | 6.7836 | 0.370 | 1148 |

### Classification Tasks

| Task | Split | Accuracy | AUROC | Samples |
|------|-------|----------|-------|---------|
| is_metal | train | 0.8270 | 0.9059 | 123902 |
| is_metal | val | 0.8232 | 0.9056 | 15487 |
| is_metal | test | 0.8251 | 0.9010 | 15489 |
| is_stable | train | 0.8143 | 0.8405 | 123902 |
| is_stable | val | 0.8095 | 0.8335 | 15487 |
| is_stable | test | 0.8123 | 0.8352 | 15489 |

## Key Findings

### Best Regression Tasks (by MAE):
- **homogeneous_poisson**: MAE = 0.0441
- **energy_above_hull**: MAE = 0.1220
- **formation_energy_per_atom**: MAE = 0.2147

### Worst Regression Tasks (by MAE):
- **energy_per_atom**: MAE = 1.8758
- **bulk_modulus_vrh**: MAE = 8.5855
- **shear_modulus_vrh**: MAE = 10.6750

### Classification Tasks Performance:
- **is_metal**: Accuracy = 0.8232, AUROC = 0.9056
- **is_stable**: Accuracy = 0.8095, AUROC = 0.8335

## Overfitting Analysis

- **bulk_modulus_vrh**: Train MAE=3.9368, Val MAE=8.5855 (Gap: 4.6486) ⚠️
- **shear_modulus_vrh**: Train MAE=5.7265, Val MAE=10.6750 (Gap: 4.9485) ⚠️
- **homogeneous_poisson**: Train MAE=0.0340, Val MAE=0.0441 (Gap: 0.0102) ⚠️
- **universal_anisotropy**: Train MAE=1.4967, Val MAE=1.8657 (Gap: 0.3689) ⚠️

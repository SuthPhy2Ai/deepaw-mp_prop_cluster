# Model Performance Analysis Report

Generated: 2026-03-08 00:51:04

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
| Train | 1.1172 |
| Val | 1.9359 |
| Test | 1.8598 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| energy_per_atom | train | 1.6646 | 4.1141 | 0.732 | 123902 |
| energy_per_atom | val | 1.7583 | 4.2806 | 0.710 | 15487 |
| energy_per_atom | test | 1.7272 | 4.2733 | 0.712 | 15489 |
| formation_energy_per_atom | train | 0.2104 | 0.3435 | 0.918 | 123902 |
| formation_energy_per_atom | val | 0.2137 | 0.3526 | 0.914 | 15487 |
| formation_energy_per_atom | test | 0.2149 | 0.3488 | 0.916 | 15489 |
| energy_above_hull | train | 0.1204 | 0.2882 | 0.595 | 123902 |
| energy_above_hull | val | 0.1241 | 0.3035 | 0.558 | 15487 |
| energy_above_hull | test | 0.1237 | 0.2929 | 0.584 | 15489 |
| band_gap | train | 0.5514 | 0.8879 | 0.655 | 123902 |
| band_gap | val | 0.5505 | 0.8894 | 0.648 | 15487 |
| band_gap | test | 0.5658 | 0.9205 | 0.636 | 15489 |
| cbm | train | 0.6003 | 0.8436 | 0.865 | 71574 |
| cbm | val | 0.6060 | 0.8452 | 0.866 | 8836 |
| cbm | test | 0.6174 | 0.8721 | 0.854 | 8928 |
| vbm | train | 0.4861 | 0.6994 | 0.914 | 71574 |
| vbm | val | 0.4885 | 0.7037 | 0.913 | 8836 |
| vbm | test | 0.4957 | 0.7272 | 0.907 | 8928 |
| efermi | train | 0.6254 | 0.9312 | 0.887 | 123856 |
| efermi | val | 0.6257 | 0.9290 | 0.886 | 15485 |
| efermi | test | 0.6319 | 0.9362 | 0.887 | 15483 |
| bulk_modulus_vrh | train | 3.4806 | 14.8618 | 0.961 | 10217 |
| bulk_modulus_vrh | val | 8.3024 | 20.6097 | 0.924 | 1348 |
| bulk_modulus_vrh | test | 7.5573 | 21.5679 | 0.918 | 1287 |
| shear_modulus_vrh | train | 5.0658 | 27.7123 | 0.717 | 9685 |
| shear_modulus_vrh | val | 10.2219 | 28.6012 | 0.693 | 1293 |
| shear_modulus_vrh | test | 9.6910 | 25.9960 | 0.721 | 1227 |
| homogeneous_poisson | train | 0.0321 | 0.0906 | 0.296 | 9752 |
| homogeneous_poisson | val | 0.0423 | 0.0908 | 0.179 | 1304 |
| homogeneous_poisson | test | 0.0419 | 0.0903 | 0.222 | 1236 |
| universal_anisotropy | train | 1.3302 | 4.9081 | 0.603 | 9057 |
| universal_anisotropy | val | 1.8690 | 5.1320 | 0.349 | 1213 |
| universal_anisotropy | test | 2.1431 | 6.8245 | 0.362 | 1148 |

### Classification Tasks

| Task | Split | Accuracy | AUROC | Samples |
|------|-------|----------|-------|---------|
| is_metal | train | 0.8265 | 0.9054 | 123902 |
| is_metal | val | 0.8222 | 0.9035 | 15487 |
| is_metal | test | 0.8212 | 0.8992 | 15489 |
| is_stable | train | 0.8167 | 0.8437 | 123902 |
| is_stable | val | 0.8138 | 0.8378 | 15487 |
| is_stable | test | 0.8113 | 0.8378 | 15489 |

## Key Findings

### Best Regression Tasks (by MAE):
- **homogeneous_poisson**: MAE = 0.0423
- **energy_above_hull**: MAE = 0.1241
- **formation_energy_per_atom**: MAE = 0.2137

### Worst Regression Tasks (by MAE):
- **universal_anisotropy**: MAE = 1.8690
- **bulk_modulus_vrh**: MAE = 8.3024
- **shear_modulus_vrh**: MAE = 10.2219

### Classification Tasks Performance:
- **is_metal**: Accuracy = 0.8222, AUROC = 0.9035
- **is_stable**: Accuracy = 0.8138, AUROC = 0.8378

## Overfitting Analysis

- **bulk_modulus_vrh**: Train MAE=3.4806, Val MAE=8.3024 (Gap: 4.8218) ⚠️
- **shear_modulus_vrh**: Train MAE=5.0658, Val MAE=10.2219 (Gap: 5.1561) ⚠️
- **homogeneous_poisson**: Train MAE=0.0321, Val MAE=0.0423 (Gap: 0.0102) ⚠️
- **universal_anisotropy**: Train MAE=1.3302, Val MAE=1.8690 (Gap: 0.5387) ⚠️

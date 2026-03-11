# Model Performance Analysis Report

Generated: 2026-03-07 22:30:55

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
| Train | 1.0942 |
| Val | 1.9461 |
| Test | 1.8269 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| energy_per_atom | train | 1.6699 | 4.1920 | 0.722 | 123902 |
| energy_per_atom | val | 1.7711 | 4.4470 | 0.688 | 15487 |
| energy_per_atom | test | 1.7580 | 4.4361 | 0.689 | 15489 |
| formation_energy_per_atom | train | 0.2108 | 0.3460 | 0.917 | 123902 |
| formation_energy_per_atom | val | 0.2150 | 0.3560 | 0.912 | 15487 |
| formation_energy_per_atom | test | 0.2142 | 0.3503 | 0.915 | 15489 |
| energy_above_hull | train | 0.1195 | 0.2897 | 0.591 | 123902 |
| energy_above_hull | val | 0.1225 | 0.3040 | 0.556 | 15487 |
| energy_above_hull | test | 0.1221 | 0.2946 | 0.579 | 15489 |
| band_gap | train | 0.5431 | 0.8842 | 0.658 | 123902 |
| band_gap | val | 0.5402 | 0.8861 | 0.651 | 15487 |
| band_gap | test | 0.5530 | 0.9072 | 0.646 | 15489 |
| cbm | train | 0.5911 | 0.8349 | 0.868 | 71574 |
| cbm | val | 0.5940 | 0.8344 | 0.869 | 8836 |
| cbm | test | 0.6057 | 0.8520 | 0.861 | 8928 |
| vbm | train | 0.4790 | 0.6926 | 0.916 | 71574 |
| vbm | val | 0.4770 | 0.7014 | 0.914 | 8836 |
| vbm | test | 0.4933 | 0.7182 | 0.909 | 8928 |
| efermi | train | 0.6196 | 0.9355 | 0.886 | 123856 |
| efermi | val | 0.6186 | 0.9376 | 0.884 | 15485 |
| efermi | test | 0.6283 | 0.9438 | 0.886 | 15483 |
| bulk_modulus_vrh | train | 3.2653 | 13.9206 | 0.966 | 10217 |
| bulk_modulus_vrh | val | 8.3487 | 21.6581 | 0.917 | 1348 |
| bulk_modulus_vrh | test | 7.6187 | 21.6550 | 0.917 | 1287 |
| shear_modulus_vrh | train | 4.9327 | 27.6702 | 0.718 | 9685 |
| shear_modulus_vrh | val | 10.3827 | 28.6922 | 0.691 | 1293 |
| shear_modulus_vrh | test | 9.2220 | 25.5083 | 0.731 | 1227 |
| homogeneous_poisson | train | 0.0308 | 0.0899 | 0.307 | 9752 |
| homogeneous_poisson | val | 0.0423 | 0.0915 | 0.165 | 1304 |
| homogeneous_poisson | test | 0.0408 | 0.0897 | 0.232 | 1236 |
| universal_anisotropy | train | 1.4089 | 5.2124 | 0.552 | 9057 |
| universal_anisotropy | val | 1.8196 | 5.1129 | 0.354 | 1213 |
| universal_anisotropy | test | 2.1316 | 6.8763 | 0.352 | 1148 |

### Classification Tasks

| Task | Split | Accuracy | AUROC | Samples |
|------|-------|----------|-------|---------|
| is_metal | train | 0.8276 | 0.9056 | 123902 |
| is_metal | val | 0.8217 | 0.9029 | 15487 |
| is_metal | test | 0.8214 | 0.9011 | 15489 |
| is_stable | train | 0.8180 | 0.8448 | 123902 |
| is_stable | val | 0.8106 | 0.8389 | 15487 |
| is_stable | test | 0.8160 | 0.8404 | 15489 |

## Key Findings

### Best Regression Tasks (by MAE):
- **homogeneous_poisson**: MAE = 0.0423
- **energy_above_hull**: MAE = 0.1225
- **formation_energy_per_atom**: MAE = 0.2150

### Worst Regression Tasks (by MAE):
- **universal_anisotropy**: MAE = 1.8196
- **bulk_modulus_vrh**: MAE = 8.3487
- **shear_modulus_vrh**: MAE = 10.3827

### Classification Tasks Performance:
- **is_metal**: Accuracy = 0.8217, AUROC = 0.9029
- **is_stable**: Accuracy = 0.8106, AUROC = 0.8389

## Overfitting Analysis

- **bulk_modulus_vrh**: Train MAE=3.2653, Val MAE=8.3487 (Gap: 5.0833) ⚠️
- **shear_modulus_vrh**: Train MAE=4.9327, Val MAE=10.3827 (Gap: 5.4499) ⚠️
- **homogeneous_poisson**: Train MAE=0.0308, Val MAE=0.0423 (Gap: 0.0116) ⚠️
- **universal_anisotropy**: Train MAE=1.4089, Val MAE=1.8196 (Gap: 0.4107) ⚠️

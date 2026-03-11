# Model Performance Analysis Report

Generated: 2026-03-10 01:32:06

================================================================================

## Model Configuration

- **Backbone**: graph
- **Hidden Dimension**: 256
- **Layers**: 6
- **Learning Rate**: 0.0002
- **Batch Size**: 64
- **Training Epochs**: 35
- **Stage**: b
- **Enabled Tasks**: 13

## Overall Performance

| Split | Loss |
|-------|------|
| Train | 1.1559 |
| Val | 1.9628 |
| Test | 1.8509 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| energy_per_atom | train | 1.7110 | 4.2513 | 0.714 | 123902 |
| energy_per_atom | val | 1.8219 | 4.5170 | 0.678 | 15487 |
| energy_per_atom | test | 1.7855 | 4.4467 | 0.688 | 15489 |
| formation_energy_per_atom | train | 0.1963 | 0.3280 | 0.926 | 123902 |
| formation_energy_per_atom | val | 0.1999 | 0.3382 | 0.921 | 15487 |
| formation_energy_per_atom | test | 0.2004 | 0.3334 | 0.923 | 15489 |
| energy_above_hull | train | 0.1159 | 0.2854 | 0.603 | 123902 |
| energy_above_hull | val | 0.1196 | 0.3006 | 0.566 | 15487 |
| energy_above_hull | test | 0.1190 | 0.2902 | 0.592 | 15489 |
| band_gap | train | 0.5415 | 0.8772 | 0.663 | 123902 |
| band_gap | val | 0.5381 | 0.8724 | 0.661 | 15487 |
| band_gap | test | 0.5500 | 0.8960 | 0.655 | 15489 |
| cbm | train | 0.5929 | 0.8334 | 0.868 | 71574 |
| cbm | val | 0.6012 | 0.8388 | 0.868 | 8836 |
| cbm | test | 0.6166 | 0.8666 | 0.856 | 8928 |
| vbm | train | 0.4774 | 0.6867 | 0.918 | 71574 |
| vbm | val | 0.4798 | 0.6908 | 0.916 | 8836 |
| vbm | test | 0.4848 | 0.7041 | 0.912 | 8928 |
| efermi | train | 0.6182 | 0.9312 | 0.887 | 123856 |
| efermi | val | 0.6158 | 0.9257 | 0.887 | 15485 |
| efermi | test | 0.6209 | 0.9316 | 0.889 | 15483 |
| bulk_modulus_vrh | train | 3.5851 | 14.7625 | 0.962 | 10217 |
| bulk_modulus_vrh | val | 8.3932 | 22.0152 | 0.914 | 1348 |
| bulk_modulus_vrh | test | 7.6371 | 21.6416 | 0.917 | 1287 |
| shear_modulus_vrh | train | 5.3849 | 28.1059 | 0.709 | 9685 |
| shear_modulus_vrh | val | 10.5251 | 28.9441 | 0.685 | 1293 |
| shear_modulus_vrh | test | 9.5285 | 25.6366 | 0.729 | 1227 |
| homogeneous_poisson | train | 0.0265 | 0.0882 | 0.333 | 9752 |
| homogeneous_poisson | val | 0.0419 | 0.0935 | 0.130 | 1304 |
| homogeneous_poisson | test | 0.0398 | 0.0899 | 0.228 | 1236 |
| universal_anisotropy | train | 1.4288 | 5.0283 | 0.583 | 9057 |
| universal_anisotropy | val | 1.8231 | 5.0665 | 0.366 | 1213 |
| universal_anisotropy | test | 2.1219 | 6.8033 | 0.366 | 1148 |

### Classification Tasks

| Task | Split | Accuracy | AUROC | Samples |
|------|-------|----------|-------|---------|
| is_metal | train | 0.8320 | 0.9093 | 123902 |
| is_metal | val | 0.8302 | 0.9085 | 15487 |
| is_metal | test | 0.8268 | 0.9037 | 15489 |
| is_stable | train | 0.8191 | 0.8507 | 123902 |
| is_stable | val | 0.8124 | 0.8433 | 15487 |
| is_stable | test | 0.8154 | 0.8461 | 15489 |

## Key Findings

### Best Regression Tasks (by MAE):
- **homogeneous_poisson**: MAE = 0.0419
- **energy_above_hull**: MAE = 0.1196
- **formation_energy_per_atom**: MAE = 0.1999

### Worst Regression Tasks (by MAE):
- **universal_anisotropy**: MAE = 1.8231
- **bulk_modulus_vrh**: MAE = 8.3932
- **shear_modulus_vrh**: MAE = 10.5251

### Classification Tasks Performance:
- **is_metal**: Accuracy = 0.8302, AUROC = 0.9085
- **is_stable**: Accuracy = 0.8124, AUROC = 0.8433

## Overfitting Analysis

- **bulk_modulus_vrh**: Train MAE=3.5851, Val MAE=8.3932 (Gap: 4.8081) ⚠️
- **shear_modulus_vrh**: Train MAE=5.3849, Val MAE=10.5251 (Gap: 5.1402) ⚠️
- **homogeneous_poisson**: Train MAE=0.0265, Val MAE=0.0419 (Gap: 0.0154) ⚠️
- **universal_anisotropy**: Train MAE=1.4288, Val MAE=1.8231 (Gap: 0.3943) ⚠️

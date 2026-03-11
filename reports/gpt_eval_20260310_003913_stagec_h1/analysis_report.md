# Model Performance Analysis Report

Generated: 2026-03-10 01:31:20

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
| Train | 1.1483 |
| Val | 1.9613 |
| Test | 1.8475 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| energy_per_atom | train | 1.6834 | 4.2300 | 0.717 | 123902 |
| energy_per_atom | val | 1.8003 | 4.5045 | 0.679 | 15487 |
| energy_per_atom | test | 1.7581 | 4.4301 | 0.690 | 15489 |
| formation_energy_per_atom | train | 0.1881 | 0.3177 | 0.930 | 123902 |
| formation_energy_per_atom | val | 0.1914 | 0.3276 | 0.925 | 15487 |
| formation_energy_per_atom | test | 0.1927 | 0.3229 | 0.928 | 15489 |
| energy_above_hull | train | 0.1139 | 0.2805 | 0.616 | 123902 |
| energy_above_hull | val | 0.1178 | 0.2959 | 0.580 | 15487 |
| energy_above_hull | test | 0.1169 | 0.2851 | 0.606 | 15489 |
| band_gap | train | 0.5220 | 0.8658 | 0.672 | 123902 |
| band_gap | val | 0.5188 | 0.8623 | 0.669 | 15487 |
| band_gap | test | 0.5312 | 0.8843 | 0.664 | 15489 |
| cbm | train | 0.5732 | 0.8151 | 0.874 | 71574 |
| cbm | val | 0.5821 | 0.8219 | 0.873 | 8836 |
| cbm | test | 0.5938 | 0.8468 | 0.862 | 8928 |
| vbm | train | 0.4639 | 0.6679 | 0.922 | 71574 |
| vbm | val | 0.4679 | 0.6765 | 0.920 | 8836 |
| vbm | test | 0.4723 | 0.6867 | 0.917 | 8928 |
| efermi | train | 0.6033 | 0.9171 | 0.890 | 123856 |
| efermi | val | 0.6025 | 0.9131 | 0.890 | 15485 |
| efermi | test | 0.6071 | 0.9191 | 0.891 | 15483 |
| bulk_modulus_vrh | train | 3.5501 | 14.8182 | 0.962 | 10217 |
| bulk_modulus_vrh | val | 8.4242 | 21.9890 | 0.914 | 1348 |
| bulk_modulus_vrh | test | 7.6892 | 21.7151 | 0.917 | 1287 |
| shear_modulus_vrh | train | 5.4045 | 28.1608 | 0.708 | 9685 |
| shear_modulus_vrh | val | 10.5496 | 29.0069 | 0.684 | 1293 |
| shear_modulus_vrh | test | 9.5588 | 25.7042 | 0.727 | 1227 |
| homogeneous_poisson | train | 0.0341 | 0.0910 | 0.290 | 9752 |
| homogeneous_poisson | val | 0.0456 | 0.0954 | 0.092 | 1304 |
| homogeneous_poisson | test | 0.0438 | 0.0922 | 0.189 | 1236 |
| universal_anisotropy | train | 1.4483 | 5.1908 | 0.556 | 9057 |
| universal_anisotropy | val | 1.8413 | 5.1396 | 0.348 | 1213 |
| universal_anisotropy | test | 2.1032 | 6.7878 | 0.369 | 1148 |

### Classification Tasks

| Task | Split | Accuracy | AUROC | Samples |
|------|-------|----------|-------|---------|
| is_metal | train | 0.8322 | 0.9117 | 123902 |
| is_metal | val | 0.8273 | 0.9102 | 15487 |
| is_metal | test | 0.8301 | 0.9066 | 15489 |
| is_stable | train | 0.8248 | 0.8606 | 123902 |
| is_stable | val | 0.8171 | 0.8517 | 15487 |
| is_stable | test | 0.8197 | 0.8558 | 15489 |

## Key Findings

### Best Regression Tasks (by MAE):
- **homogeneous_poisson**: MAE = 0.0456
- **energy_above_hull**: MAE = 0.1178
- **formation_energy_per_atom**: MAE = 0.1914

### Worst Regression Tasks (by MAE):
- **universal_anisotropy**: MAE = 1.8413
- **bulk_modulus_vrh**: MAE = 8.4242
- **shear_modulus_vrh**: MAE = 10.5496

### Classification Tasks Performance:
- **is_metal**: Accuracy = 0.8273, AUROC = 0.9102
- **is_stable**: Accuracy = 0.8171, AUROC = 0.8517

## Overfitting Analysis

- **bulk_modulus_vrh**: Train MAE=3.5501, Val MAE=8.4242 (Gap: 4.8740) ⚠️
- **shear_modulus_vrh**: Train MAE=5.4045, Val MAE=10.5496 (Gap: 5.1450) ⚠️
- **homogeneous_poisson**: Train MAE=0.0341, Val MAE=0.0456 (Gap: 0.0115) ⚠️
- **universal_anisotropy**: Train MAE=1.4483, Val MAE=1.8413 (Gap: 0.3930) ⚠️

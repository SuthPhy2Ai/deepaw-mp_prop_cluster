# Model Performance Analysis Report

Generated: 2026-03-10 01:32:52

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
| Train | 1.1608 |
| Val | 1.9628 |
| Test | 1.8590 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| energy_per_atom | train | 1.7104 | 4.2631 | 0.713 | 123902 |
| energy_per_atom | val | 1.8225 | 4.5295 | 0.676 | 15487 |
| energy_per_atom | test | 1.7828 | 4.4556 | 0.687 | 15489 |
| formation_energy_per_atom | train | 0.1954 | 0.3284 | 0.925 | 123902 |
| formation_energy_per_atom | val | 0.1987 | 0.3386 | 0.920 | 15487 |
| formation_energy_per_atom | test | 0.2000 | 0.3344 | 0.923 | 15489 |
| energy_above_hull | train | 0.1133 | 0.2857 | 0.602 | 123902 |
| energy_above_hull | val | 0.1170 | 0.3010 | 0.565 | 15487 |
| energy_above_hull | test | 0.1164 | 0.2904 | 0.591 | 15489 |
| band_gap | train | 0.5341 | 0.8875 | 0.655 | 123902 |
| band_gap | val | 0.5313 | 0.8844 | 0.652 | 15487 |
| band_gap | test | 0.5439 | 0.9075 | 0.646 | 15489 |
| cbm | train | 0.5986 | 0.8508 | 0.863 | 71574 |
| cbm | val | 0.6061 | 0.8562 | 0.862 | 8836 |
| cbm | test | 0.6203 | 0.8829 | 0.850 | 8928 |
| vbm | train | 0.4790 | 0.6851 | 0.918 | 71574 |
| vbm | val | 0.4818 | 0.6920 | 0.916 | 8836 |
| vbm | test | 0.4870 | 0.7046 | 0.912 | 8928 |
| efermi | train | 0.6167 | 0.9330 | 0.886 | 123856 |
| efermi | val | 0.6153 | 0.9291 | 0.886 | 15485 |
| efermi | test | 0.6187 | 0.9330 | 0.888 | 15483 |
| bulk_modulus_vrh | train | 3.6285 | 14.7266 | 0.962 | 10217 |
| bulk_modulus_vrh | val | 8.3986 | 22.0342 | 0.914 | 1348 |
| bulk_modulus_vrh | test | 7.7358 | 21.7033 | 0.917 | 1287 |
| shear_modulus_vrh | train | 5.4030 | 28.0575 | 0.710 | 9685 |
| shear_modulus_vrh | val | 10.5177 | 28.8953 | 0.686 | 1293 |
| shear_modulus_vrh | test | 9.5410 | 25.6431 | 0.728 | 1227 |
| homogeneous_poisson | train | 0.0378 | 0.0925 | 0.266 | 9752 |
| homogeneous_poisson | val | 0.0482 | 0.0959 | 0.083 | 1304 |
| homogeneous_poisson | test | 0.0457 | 0.0928 | 0.178 | 1236 |
| universal_anisotropy | train | 1.4231 | 5.0184 | 0.585 | 9057 |
| universal_anisotropy | val | 1.8191 | 5.0665 | 0.366 | 1213 |
| universal_anisotropy | test | 2.1163 | 6.7868 | 0.369 | 1148 |

### Classification Tasks

| Task | Split | Accuracy | AUROC | Samples |
|------|-------|----------|-------|---------|
| is_metal | train | 0.8299 | 0.9082 | 123902 |
| is_metal | val | 0.8267 | 0.9071 | 15487 |
| is_metal | test | 0.8254 | 0.9028 | 15489 |
| is_stable | train | 0.8197 | 0.8527 | 123902 |
| is_stable | val | 0.8127 | 0.8450 | 15487 |
| is_stable | test | 0.8152 | 0.8480 | 15489 |

## Key Findings

### Best Regression Tasks (by MAE):
- **homogeneous_poisson**: MAE = 0.0482
- **energy_above_hull**: MAE = 0.1170
- **formation_energy_per_atom**: MAE = 0.1987

### Worst Regression Tasks (by MAE):
- **energy_per_atom**: MAE = 1.8225
- **bulk_modulus_vrh**: MAE = 8.3986
- **shear_modulus_vrh**: MAE = 10.5177

### Classification Tasks Performance:
- **is_metal**: Accuracy = 0.8267, AUROC = 0.9071
- **is_stable**: Accuracy = 0.8127, AUROC = 0.8450

## Overfitting Analysis

- **bulk_modulus_vrh**: Train MAE=3.6285, Val MAE=8.3986 (Gap: 4.7701) ⚠️
- **shear_modulus_vrh**: Train MAE=5.4030, Val MAE=10.5177 (Gap: 5.1148) ⚠️
- **homogeneous_poisson**: Train MAE=0.0378, Val MAE=0.0482 (Gap: 0.0104) ⚠️
- **universal_anisotropy**: Train MAE=1.4231, Val MAE=1.8191 (Gap: 0.3960) ⚠️

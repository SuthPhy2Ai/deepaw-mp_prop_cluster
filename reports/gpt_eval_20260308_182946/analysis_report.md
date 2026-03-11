# Model Performance Analysis Report

Generated: 2026-03-08 19:23:10

================================================================================

## Model Configuration

- **Backbone**: graph
- **Hidden Dimension**: 256
- **Layers**: 6
- **Learning Rate**: 0.0001
- **Batch Size**: 64
- **Training Epochs**: 50
- **Stage**: a
- **Enabled Tasks**: 8

## Overall Performance

| Split | Loss |
|-------|------|
| Train | 0.1513 |
| Val | 0.2198 |
| Test | 0.2253 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| energy_per_atom | train | 0.2040 | 1.2153 | 0.977 | 123902 |
| energy_per_atom | val | 0.3617 | 1.9403 | 0.941 | 15487 |
| energy_per_atom | test | 0.3665 | 2.0169 | 0.936 | 15489 |
| formation_energy_per_atom | train | 0.0720 | 0.1082 | 0.992 | 123902 |
| formation_energy_per_atom | val | 0.0816 | 0.1338 | 0.988 | 15487 |
| formation_energy_per_atom | test | 0.0823 | 0.1375 | 0.987 | 15489 |
| energy_above_hull | train | 0.0574 | 0.1012 | 0.950 | 123902 |
| energy_above_hull | val | 0.0639 | 0.1258 | 0.924 | 15487 |
| energy_above_hull | test | 0.0643 | 0.1276 | 0.921 | 15489 |
| band_gap | train | 0.1494 | 0.2981 | 0.961 | 123902 |
| band_gap | val | 0.2258 | 0.4533 | 0.909 | 15487 |
| band_gap | test | 0.2386 | 0.4785 | 0.902 | 15489 |
| cbm | train | 0.1929 | 0.2876 | 0.984 | 71574 |
| cbm | val | 0.2932 | 0.4504 | 0.962 | 8836 |
| cbm | test | 0.2978 | 0.4539 | 0.960 | 8928 |
| vbm | train | 0.1780 | 0.2653 | 0.988 | 71574 |
| vbm | val | 0.2539 | 0.3930 | 0.973 | 8836 |
| vbm | test | 0.2603 | 0.3972 | 0.972 | 8928 |
| efermi | train | 0.3175 | 0.6462 | 0.945 | 123856 |
| efermi | val | 0.3756 | 0.7086 | 0.934 | 15485 |
| efermi | test | 0.3859 | 0.7144 | 0.934 | 15483 |

### Classification Tasks

| Task | Split | Accuracy | AUROC | Samples |
|------|-------|----------|-------|---------|
| is_metal | train | 0.9608 | 0.9938 | 123902 |
| is_metal | val | 0.8972 | 0.9621 | 15487 |
| is_metal | test | 0.8933 | 0.9586 | 15489 |

## Key Findings

### Best Regression Tasks (by MAE):
- **energy_above_hull**: MAE = 0.0639
- **formation_energy_per_atom**: MAE = 0.0816
- **band_gap**: MAE = 0.2258

### Worst Regression Tasks (by MAE):
- **cbm**: MAE = 0.2932
- **energy_per_atom**: MAE = 0.3617
- **efermi**: MAE = 0.3756

### Classification Tasks Performance:
- **is_metal**: Accuracy = 0.8972, AUROC = 0.9621

## Overfitting Analysis

- **energy_per_atom**: Train MAE=0.2040, Val MAE=0.3617 (Gap: 0.1578) ⚠️
- **band_gap**: Train MAE=0.1494, Val MAE=0.2258 (Gap: 0.0763) ⚠️
- **cbm**: Train MAE=0.1929, Val MAE=0.2932 (Gap: 0.1003) ⚠️
- **vbm**: Train MAE=0.1780, Val MAE=0.2539 (Gap: 0.0758) ⚠️

# Model Performance Analysis Report

Generated: 2026-03-06 12:26:01

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
| Train | 0.1454 |
| Val | 0.2226 |
| Test | 0.2223 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| energy_per_atom | train | 0.1848 | 1.1552 | 0.979 | 123903 |
| energy_per_atom | val | 0.3606 | 1.9919 | 0.937 | 15487 |
| energy_per_atom | test | 0.3355 | 1.8476 | 0.946 | 15489 |
| formation_energy_per_atom | train | 0.0696 | 0.1042 | 0.992 | 123903 |
| formation_energy_per_atom | val | 0.0800 | 0.1315 | 0.988 | 15487 |
| formation_energy_per_atom | test | 0.0809 | 0.1363 | 0.987 | 15489 |
| energy_above_hull | train | 0.0568 | 0.0975 | 0.954 | 123903 |
| energy_above_hull | val | 0.0644 | 0.1261 | 0.924 | 15487 |
| energy_above_hull | test | 0.0642 | 0.1276 | 0.921 | 15489 |
| band_gap | train | 0.1433 | 0.2843 | 0.965 | 123903 |
| band_gap | val | 0.2308 | 0.4649 | 0.904 | 15487 |
| band_gap | test | 0.2379 | 0.4729 | 0.904 | 15489 |
| cbm | train | 0.1861 | 0.2768 | 0.985 | 71575 |
| cbm | val | 0.2921 | 0.4463 | 0.963 | 8836 |
| cbm | test | 0.2959 | 0.4527 | 0.961 | 8928 |
| vbm | train | 0.1701 | 0.2537 | 0.989 | 71575 |
| vbm | val | 0.2594 | 0.3981 | 0.972 | 8836 |
| vbm | test | 0.2643 | 0.4031 | 0.971 | 8928 |
| efermi | train | 0.3167 | 0.6343 | 0.947 | 123857 |
| efermi | val | 0.3834 | 0.7039 | 0.934 | 15485 |
| efermi | test | 0.3930 | 0.7114 | 0.935 | 15483 |

### Classification Tasks

| Task | Split | Accuracy | AUROC | Samples |
|------|-------|----------|-------|---------|
| is_metal | train | 0.9640 | 0.9950 | 123903 |
| is_metal | val | 0.8902 | 0.9575 | 15487 |
| is_metal | test | 0.8935 | 0.9581 | 15489 |

## Key Findings

### Best Regression Tasks (by MAE):
- **energy_above_hull**: MAE = 0.0644
- **formation_energy_per_atom**: MAE = 0.0800
- **band_gap**: MAE = 0.2308

### Worst Regression Tasks (by MAE):
- **cbm**: MAE = 0.2921
- **energy_per_atom**: MAE = 0.3606
- **efermi**: MAE = 0.3834

### Classification Tasks Performance:
- **is_metal**: Accuracy = 0.8902, AUROC = 0.9575

## Overfitting Analysis

- **energy_per_atom**: Train MAE=0.1848, Val MAE=0.3606 (Gap: 0.1758) ⚠️
- **band_gap**: Train MAE=0.1433, Val MAE=0.2308 (Gap: 0.0875) ⚠️
- **cbm**: Train MAE=0.1861, Val MAE=0.2921 (Gap: 0.1060) ⚠️
- **vbm**: Train MAE=0.1701, Val MAE=0.2594 (Gap: 0.0893) ⚠️
- **efermi**: Train MAE=0.3167, Val MAE=0.3834 (Gap: 0.0666) ⚠️

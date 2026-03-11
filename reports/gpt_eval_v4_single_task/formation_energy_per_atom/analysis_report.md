# Model Performance Analysis Report

Generated: 2026-03-08 18:16:37

================================================================================

## Model Configuration

- **Backbone**: graph
- **Hidden Dimension**: 256
- **Layers**: 6
- **Learning Rate**: 0.0002
- **Batch Size**: 64
- **Training Epochs**: 30
- **Stage**: b
- **Enabled Tasks**: 1

## Overall Performance

| Split | Loss |
|-------|------|
| Train | 0.1491 |
| Val | 0.1548 |
| Test | 0.1572 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| formation_energy_per_atom | train | 0.1491 | 0.2619 | 0.953 | 123902 |
| formation_energy_per_atom | val | 0.1548 | 0.2758 | 0.947 | 15487 |
| formation_energy_per_atom | test | 0.1572 | 0.2742 | 0.948 | 15489 |

## Key Findings

### Best Regression Tasks (by MAE):
- **formation_energy_per_atom**: MAE = 0.1548

### Worst Regression Tasks (by MAE):
- **formation_energy_per_atom**: MAE = 0.1548

## Overfitting Analysis


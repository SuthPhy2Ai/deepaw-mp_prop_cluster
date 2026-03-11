# Model Performance Analysis Report

Generated: 2026-03-09 23:13:59

================================================================================

## Model Configuration

- **Backbone**: graph
- **Hidden Dimension**: 256
- **Layers**: 6
- **Learning Rate**: 0.0001
- **Batch Size**: 64
- **Training Epochs**: 50
- **Stage**: full
- **Enabled Tasks**: 1

## Overall Performance

| Split | Loss |
|-------|------|
| Train | 0.0174 |
| Val | 0.0330 |
| Test | 0.0329 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| formation_energy_per_atom | train | 0.0174 | 0.0343 | 0.999 | 123902 |
| formation_energy_per_atom | val | 0.0330 | 0.0699 | 0.997 | 15487 |
| formation_energy_per_atom | test | 0.0329 | 0.0733 | 0.996 | 15489 |

## Key Findings

### Best Regression Tasks (by MAE):
- **formation_energy_per_atom**: MAE = 0.0330

### Worst Regression Tasks (by MAE):
- **formation_energy_per_atom**: MAE = 0.0330

## Overfitting Analysis

- **formation_energy_per_atom**: Train MAE=0.0174, Val MAE=0.0330 (Gap: 0.0156) ⚠️

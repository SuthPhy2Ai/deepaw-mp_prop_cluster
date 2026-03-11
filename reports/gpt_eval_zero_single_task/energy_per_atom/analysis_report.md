# Model Performance Analysis Report

Generated: 2026-03-09 23:13:28

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
| Train | 0.1275 |
| Val | 0.2762 |
| Test | 0.2480 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| energy_per_atom | train | 0.1275 | 1.0216 | 0.983 | 123902 |
| energy_per_atom | val | 0.2762 | 1.9257 | 0.941 | 15487 |
| energy_per_atom | test | 0.2480 | 1.5972 | 0.960 | 15489 |

## Key Findings

### Best Regression Tasks (by MAE):
- **energy_per_atom**: MAE = 0.2762

### Worst Regression Tasks (by MAE):
- **energy_per_atom**: MAE = 0.2762

## Overfitting Analysis

- **energy_per_atom**: Train MAE=0.1275, Val MAE=0.2762 (Gap: 0.1486) ⚠️

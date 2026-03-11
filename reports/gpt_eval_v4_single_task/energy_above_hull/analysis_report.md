# Model Performance Analysis Report

Generated: 2026-03-08 18:17:08

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
| Train | 0.0958 |
| Val | 0.1007 |
| Test | 0.1016 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| energy_above_hull | train | 0.0958 | 0.2246 | 0.754 | 123902 |
| energy_above_hull | val | 0.1007 | 0.2371 | 0.730 | 15487 |
| energy_above_hull | test | 0.1016 | 0.2325 | 0.738 | 15489 |

## Key Findings

### Best Regression Tasks (by MAE):
- **energy_above_hull**: MAE = 0.1007

### Worst Regression Tasks (by MAE):
- **energy_above_hull**: MAE = 0.1007

## Overfitting Analysis


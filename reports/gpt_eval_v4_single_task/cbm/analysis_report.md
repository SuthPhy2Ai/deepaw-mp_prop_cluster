# Model Performance Analysis Report

Generated: 2026-03-08 18:18:09

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
| Train | 0.5500 |
| Val | 0.5645 |
| Test | 0.5777 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| cbm | train | 0.5500 | 0.7876 | 0.882 | 71574 |
| cbm | val | 0.5645 | 0.8001 | 0.880 | 8836 |
| cbm | test | 0.5777 | 0.8268 | 0.869 | 8928 |

## Key Findings

### Best Regression Tasks (by MAE):
- **cbm**: MAE = 0.5645

### Worst Regression Tasks (by MAE):
- **cbm**: MAE = 0.5645

## Overfitting Analysis


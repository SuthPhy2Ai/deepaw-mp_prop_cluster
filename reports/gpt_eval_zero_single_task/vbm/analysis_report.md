# Model Performance Analysis Report

Generated: 2026-03-09 23:16:03

================================================================================

## Model Configuration

- **Backbone**: graph
- **Hidden Dimension**: 224
- **Layers**: 6
- **Learning Rate**: 0.0001
- **Batch Size**: 64
- **Training Epochs**: 55
- **Stage**: full
- **Enabled Tasks**: 1

## Overall Performance

| Split | Loss |
|-------|------|
| Train | 0.0902 |
| Val | 0.2482 |
| Test | 0.2519 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| vbm | train | 0.0902 | 0.1561 | 0.996 | 71574 |
| vbm | val | 0.2482 | 0.4099 | 0.971 | 8836 |
| vbm | test | 0.2519 | 0.4155 | 0.970 | 8928 |

## Key Findings

### Best Regression Tasks (by MAE):
- **vbm**: MAE = 0.2482

### Worst Regression Tasks (by MAE):
- **vbm**: MAE = 0.2482

## Overfitting Analysis

- **vbm**: Train MAE=0.0902, Val MAE=0.2482 (Gap: 0.1579) ⚠️

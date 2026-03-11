# Model Performance Analysis Report

Generated: 2026-03-08 18:22:10

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
| Train | 1.3834 |
| Val | 1.8376 |
| Test | 2.1128 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| universal_anisotropy | train | 1.3834 | 4.9864 | 0.590 | 9057 |
| universal_anisotropy | val | 1.8376 | 5.0760 | 0.364 | 1213 |
| universal_anisotropy | test | 2.1128 | 6.7332 | 0.379 | 1148 |

## Key Findings

### Best Regression Tasks (by MAE):
- **universal_anisotropy**: MAE = 1.8376

### Worst Regression Tasks (by MAE):
- **universal_anisotropy**: MAE = 1.8376

## Overfitting Analysis

- **universal_anisotropy**: Train MAE=1.3834, Val MAE=1.8376 (Gap: 0.4541) ⚠️

# Model Performance Analysis Report

Generated: 2026-03-08 18:18:39

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
| Train | 0.4406 |
| Val | 0.4504 |
| Test | 0.4536 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| vbm | train | 0.4406 | 0.6470 | 0.927 | 71574 |
| vbm | val | 0.4504 | 0.6603 | 0.924 | 8836 |
| vbm | test | 0.4536 | 0.6697 | 0.921 | 8928 |

## Key Findings

### Best Regression Tasks (by MAE):
- **vbm**: MAE = 0.4504

### Worst Regression Tasks (by MAE):
- **vbm**: MAE = 0.4504

## Overfitting Analysis


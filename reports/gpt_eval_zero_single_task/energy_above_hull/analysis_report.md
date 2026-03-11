# Model Performance Analysis Report

Generated: 2026-03-09 23:14:30

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
| Train | 0.0168 |
| Val | 0.0307 |
| Test | 0.0312 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| energy_above_hull | train | 0.0168 | 0.0327 | 0.995 | 123902 |
| energy_above_hull | val | 0.0307 | 0.0746 | 0.973 | 15487 |
| energy_above_hull | test | 0.0312 | 0.0778 | 0.971 | 15489 |

## Key Findings

### Best Regression Tasks (by MAE):
- **energy_above_hull**: MAE = 0.0307

### Worst Regression Tasks (by MAE):
- **energy_above_hull**: MAE = 0.0307

## Overfitting Analysis

- **energy_above_hull**: Train MAE=0.0168, Val MAE=0.0307 (Gap: 0.0139) ⚠️

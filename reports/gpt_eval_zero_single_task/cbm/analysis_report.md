# Model Performance Analysis Report

Generated: 2026-03-09 23:15:32

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
| Train | 0.1120 |
| Val | 0.2836 |
| Test | 0.2919 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| cbm | train | 0.1120 | 0.1881 | 0.993 | 71574 |
| cbm | val | 0.2836 | 0.4540 | 0.961 | 8836 |
| cbm | test | 0.2919 | 0.4745 | 0.957 | 8928 |

## Key Findings

### Best Regression Tasks (by MAE):
- **cbm**: MAE = 0.2836

### Worst Regression Tasks (by MAE):
- **cbm**: MAE = 0.2836

## Overfitting Analysis

- **cbm**: Train MAE=0.1120, Val MAE=0.2836 (Gap: 0.1716) ⚠️

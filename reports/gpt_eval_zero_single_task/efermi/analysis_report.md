# Model Performance Analysis Report

Generated: 2026-03-09 23:16:34

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
| Train | 0.2400 |
| Val | 0.3848 |
| Test | 0.3997 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| efermi | train | 0.2400 | 0.4679 | 0.971 | 123856 |
| efermi | val | 0.3848 | 0.7327 | 0.929 | 15485 |
| efermi | test | 0.3997 | 0.7472 | 0.928 | 15483 |

## Key Findings

### Best Regression Tasks (by MAE):
- **efermi**: MAE = 0.3848

### Worst Regression Tasks (by MAE):
- **efermi**: MAE = 0.3848

## Overfitting Analysis

- **efermi**: Train MAE=0.2400, Val MAE=0.3848 (Gap: 0.1448) ⚠️

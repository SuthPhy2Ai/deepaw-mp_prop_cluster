# Model Performance Analysis Report

Generated: 2026-03-09 23:20:20

================================================================================

## Model Configuration

- **Backbone**: graph
- **Hidden Dimension**: 128
- **Layers**: 4
- **Learning Rate**: 8e-05
- **Batch Size**: 32
- **Training Epochs**: 90
- **Stage**: full
- **Enabled Tasks**: 1

## Overall Performance

| Split | Loss |
|-------|------|
| Train | 1.4066 |
| Val | 1.8494 |
| Test | 2.2798 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| universal_anisotropy | train | 1.4066 | 5.3110 | 0.535 | 9057 |
| universal_anisotropy | val | 1.8494 | 5.3148 | 0.302 | 1213 |
| universal_anisotropy | test | 2.2798 | 7.3862 | 0.253 | 1148 |

## Key Findings

### Best Regression Tasks (by MAE):
- **universal_anisotropy**: MAE = 1.8494

### Worst Regression Tasks (by MAE):
- **universal_anisotropy**: MAE = 1.8494

## Overfitting Analysis

- **universal_anisotropy**: Train MAE=1.4066, Val MAE=1.8494 (Gap: 0.4428) ⚠️

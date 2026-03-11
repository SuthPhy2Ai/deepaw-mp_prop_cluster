# Model Performance Analysis Report

Generated: 2026-03-09 23:15:01

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
| Train | 0.1186 |
| Val | 0.2252 |
| Test | 0.2378 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| band_gap | train | 0.1186 | 0.2595 | 0.971 | 123902 |
| band_gap | val | 0.2252 | 0.4768 | 0.899 | 15487 |
| band_gap | test | 0.2378 | 0.5007 | 0.892 | 15489 |

## Key Findings

### Best Regression Tasks (by MAE):
- **band_gap**: MAE = 0.2252

### Worst Regression Tasks (by MAE):
- **band_gap**: MAE = 0.2252

## Overfitting Analysis

- **band_gap**: Train MAE=0.1186, Val MAE=0.2252 (Gap: 0.1066) ⚠️

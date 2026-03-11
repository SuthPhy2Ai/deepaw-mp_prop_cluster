# Model Performance Analysis Report

Generated: 2026-03-08 18:17:38

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
| Train | 0.4919 |
| Val | 0.4954 |
| Test | 0.5098 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| band_gap | train | 0.4919 | 0.8233 | 0.703 | 123902 |
| band_gap | val | 0.4954 | 0.8269 | 0.696 | 15487 |
| band_gap | test | 0.5098 | 0.8517 | 0.688 | 15489 |

## Key Findings

### Best Regression Tasks (by MAE):
- **band_gap**: MAE = 0.4954

### Worst Regression Tasks (by MAE):
- **band_gap**: MAE = 0.4954

## Overfitting Analysis


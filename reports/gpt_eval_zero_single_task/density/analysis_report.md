# Model Performance Analysis Report

Generated: 2026-03-09 23:19:07

================================================================================

## Model Configuration

- **Backbone**: graph
- **Hidden Dimension**: 320
- **Layers**: 7
- **Learning Rate**: 8e-05
- **Batch Size**: 32
- **Training Epochs**: 70
- **Stage**: full
- **Enabled Tasks**: 1

## Overall Performance

| Split | Loss |
|-------|------|
| Train | 0.0142 |
| Val | 0.0190 |
| Test | 0.0191 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| density | train | 0.0142 | 0.0236 | 1.000 | 123902 |
| density | val | 0.0190 | 0.0480 | 1.000 | 15487 |
| density | test | 0.0191 | 0.0475 | 1.000 | 15489 |

## Key Findings

### Best Regression Tasks (by MAE):
- **density**: MAE = 0.0190

### Worst Regression Tasks (by MAE):
- **density**: MAE = 0.0190

## Overfitting Analysis

- **density**: Train MAE=0.0142, Val MAE=0.0190 (Gap: 0.0048) ⚠️

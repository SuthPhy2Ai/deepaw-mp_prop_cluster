# Model Performance Analysis Report

Generated: 2026-03-09 23:20:03

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
| Train | 0.0447 |
| Val | 0.0498 |
| Test | 0.0483 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| homogeneous_poisson | train | 0.0447 | 0.0884 | 0.329 | 9752 |
| homogeneous_poisson | val | 0.0498 | 0.0920 | 0.156 | 1304 |
| homogeneous_poisson | test | 0.0483 | 0.0931 | 0.173 | 1236 |

## Key Findings

### Best Regression Tasks (by MAE):
- **homogeneous_poisson**: MAE = 0.0498

### Worst Regression Tasks (by MAE):
- **homogeneous_poisson**: MAE = 0.0498

## Overfitting Analysis


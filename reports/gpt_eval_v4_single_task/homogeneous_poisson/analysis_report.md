# Model Performance Analysis Report

Generated: 2026-03-08 18:21:40

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
| Train | 0.0293 |
| Val | 0.0419 |
| Test | 0.0408 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| homogeneous_poisson | train | 0.0293 | 0.0872 | 0.348 | 9752 |
| homogeneous_poisson | val | 0.0419 | 0.0922 | 0.153 | 1304 |
| homogeneous_poisson | test | 0.0408 | 0.0896 | 0.234 | 1236 |

## Key Findings

### Best Regression Tasks (by MAE):
- **homogeneous_poisson**: MAE = 0.0419

### Worst Regression Tasks (by MAE):
- **homogeneous_poisson**: MAE = 0.0419

## Overfitting Analysis

- **homogeneous_poisson**: Train MAE=0.0293, Val MAE=0.0419 (Gap: 0.0125) ⚠️

# Model Performance Analysis Report

Generated: 2026-03-09 23:18:21

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
| Train | 82.7826 |
| Val | 145.4975 |
| Test | 153.5800 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| volume | train | 82.7826 | 231.5513 | 0.838 | 123890 |
| volume | val | 145.4975 | 317.0639 | 0.690 | 15487 |
| volume | test | 153.5800 | 340.6306 | 0.675 | 15488 |

## Key Findings

### Best Regression Tasks (by MAE):
- **volume**: MAE = 145.4975

### Worst Regression Tasks (by MAE):
- **volume**: MAE = 145.4975

## Overfitting Analysis

- **volume**: Train MAE=82.7826, Val MAE=145.4975 (Gap: 62.7148) ⚠️

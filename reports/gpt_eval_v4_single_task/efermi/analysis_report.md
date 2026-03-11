# Model Performance Analysis Report

Generated: 2026-03-08 18:19:10

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
| Train | 0.5670 |
| Val | 0.5706 |
| Test | 0.5774 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| efermi | train | 0.5670 | 0.8852 | 0.898 | 123856 |
| efermi | val | 0.5706 | 0.8858 | 0.896 | 15485 |
| efermi | test | 0.5774 | 0.8940 | 0.897 | 15483 |

## Key Findings

### Best Regression Tasks (by MAE):
- **efermi**: MAE = 0.5706

### Worst Regression Tasks (by MAE):
- **efermi**: MAE = 0.5706

## Overfitting Analysis


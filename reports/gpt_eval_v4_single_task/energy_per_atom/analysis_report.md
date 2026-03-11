# Model Performance Analysis Report

Generated: 2026-03-08 18:16:06

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
| Train | 1.6161 |
| Val | 1.7430 |
| Test | 1.7022 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| energy_per_atom | train | 1.6161 | 4.1677 | 0.725 | 123902 |
| energy_per_atom | val | 1.7430 | 4.4544 | 0.687 | 15487 |
| energy_per_atom | test | 1.7022 | 4.3904 | 0.696 | 15489 |

## Key Findings

### Best Regression Tasks (by MAE):
- **energy_per_atom**: MAE = 1.7430

### Worst Regression Tasks (by MAE):
- **energy_per_atom**: MAE = 1.7430

## Overfitting Analysis


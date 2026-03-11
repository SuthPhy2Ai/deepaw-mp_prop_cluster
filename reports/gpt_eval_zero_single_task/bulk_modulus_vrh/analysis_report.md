# Model Performance Analysis Report

Generated: 2026-03-09 23:19:27

================================================================================

## Model Configuration

- **Backbone**: graph
- **Hidden Dimension**: 160
- **Layers**: 4
- **Learning Rate**: 0.0002
- **Batch Size**: 32
- **Training Epochs**: 80
- **Stage**: full
- **Enabled Tasks**: 1

## Overall Performance

| Split | Loss |
|-------|------|
| Train | 4.1662 |
| Val | 8.9106 |
| Test | 8.2352 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| bulk_modulus_vrh | train | 4.1662 | 15.4461 | 0.958 | 10217 |
| bulk_modulus_vrh | val | 8.9106 | 20.8483 | 0.923 | 1348 |
| bulk_modulus_vrh | test | 8.2352 | 22.5729 | 0.910 | 1287 |

## Key Findings

### Best Regression Tasks (by MAE):
- **bulk_modulus_vrh**: MAE = 8.9106

### Worst Regression Tasks (by MAE):
- **bulk_modulus_vrh**: MAE = 8.9106

## Overfitting Analysis

- **bulk_modulus_vrh**: Train MAE=4.1662, Val MAE=8.9106 (Gap: 4.7443) ⚠️

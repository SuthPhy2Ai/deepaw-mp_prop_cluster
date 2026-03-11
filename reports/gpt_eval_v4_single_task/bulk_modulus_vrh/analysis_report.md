# Model Performance Analysis Report

Generated: 2026-03-08 18:20:40

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
| Train | 3.6197 |
| Val | 8.4485 |
| Test | 7.7127 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| bulk_modulus_vrh | train | 3.6197 | 14.8391 | 0.961 | 10217 |
| bulk_modulus_vrh | val | 8.4485 | 22.0418 | 0.914 | 1348 |
| bulk_modulus_vrh | test | 7.7127 | 21.7421 | 0.917 | 1287 |

## Key Findings

### Best Regression Tasks (by MAE):
- **bulk_modulus_vrh**: MAE = 8.4485

### Worst Regression Tasks (by MAE):
- **bulk_modulus_vrh**: MAE = 8.4485

## Overfitting Analysis

- **bulk_modulus_vrh**: Train MAE=3.6197, Val MAE=8.4485 (Gap: 4.8288) ⚠️

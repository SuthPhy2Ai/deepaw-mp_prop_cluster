# Model Performance Analysis Report

Generated: 2026-03-08 18:21:10

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
| Train | 5.4190 |
| Val | 10.5170 |
| Test | 9.5324 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| shear_modulus_vrh | train | 5.4190 | 28.1208 | 0.708 | 9685 |
| shear_modulus_vrh | val | 10.5170 | 28.9037 | 0.686 | 1293 |
| shear_modulus_vrh | test | 9.5324 | 25.6519 | 0.728 | 1227 |

## Key Findings

### Best Regression Tasks (by MAE):
- **shear_modulus_vrh**: MAE = 10.5170

### Worst Regression Tasks (by MAE):
- **shear_modulus_vrh**: MAE = 10.5170

## Overfitting Analysis

- **shear_modulus_vrh**: Train MAE=5.4190, Val MAE=10.5170 (Gap: 5.0980) ⚠️

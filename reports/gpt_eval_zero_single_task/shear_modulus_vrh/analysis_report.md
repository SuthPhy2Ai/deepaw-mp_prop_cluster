# Model Performance Analysis Report

Generated: 2026-03-09 23:19:46

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
| Train | 6.9751 |
| Val | 11.7471 |
| Test | 10.8423 |

## Per-Task Performance

### Regression Tasks

| Task | Split | MAE | RMSE | R² | Samples |
|------|-------|-----|------|----|---------| 
| shear_modulus_vrh | train | 6.9751 | 28.8660 | 0.693 | 9685 |
| shear_modulus_vrh | val | 11.7471 | 30.0261 | 0.661 | 1293 |
| shear_modulus_vrh | test | 10.8423 | 26.2116 | 0.716 | 1227 |

## Key Findings

### Best Regression Tasks (by MAE):
- **shear_modulus_vrh**: MAE = 11.7471

### Worst Regression Tasks (by MAE):
- **shear_modulus_vrh**: MAE = 11.7471

## Overfitting Analysis

- **shear_modulus_vrh**: Train MAE=6.9751, Val MAE=11.7471 (Gap: 4.7719) ⚠️

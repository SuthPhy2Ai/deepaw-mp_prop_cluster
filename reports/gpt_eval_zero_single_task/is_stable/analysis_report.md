# Model Performance Analysis Report

Generated: 2026-03-09 23:17:34

================================================================================

## Model Configuration

- **Backbone**: graph
- **Hidden Dimension**: 256
- **Layers**: 6
- **Learning Rate**: 0.0001
- **Batch Size**: 64
- **Training Epochs**: 50
- **Stage**: full
- **Enabled Tasks**: 1

## Overall Performance

| Split | Loss |
|-------|------|
| Train | 0.1128 |
| Val | 0.1489 |
| Test | 0.1490 |

## Per-Task Performance

### Classification Tasks

| Task | Split | Accuracy | AUROC | Samples |
|------|-------|----------|-------|---------|
| is_stable | train | 0.8872 | 0.9432 | 123902 |
| is_stable | val | 0.8511 | 0.9056 | 15487 |
| is_stable | test | 0.8510 | 0.9041 | 15489 |

## Key Findings

### Classification Tasks Performance:
- **is_stable**: Accuracy = 0.8511, AUROC = 0.9056

## Overfitting Analysis


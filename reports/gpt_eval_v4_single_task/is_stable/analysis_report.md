# Model Performance Analysis Report

Generated: 2026-03-08 18:20:10

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
| Train | 0.1630 |
| Val | 0.1752 |
| Test | 0.1737 |

## Per-Task Performance

### Classification Tasks

| Task | Split | Accuracy | AUROC | Samples |
|------|-------|----------|-------|---------|
| is_stable | train | 0.8370 | 0.8815 | 123902 |
| is_stable | val | 0.8248 | 0.8666 | 15487 |
| is_stable | test | 0.8263 | 0.8717 | 15489 |

## Key Findings

### Classification Tasks Performance:
- **is_stable**: Accuracy = 0.8248, AUROC = 0.8666

## Overfitting Analysis


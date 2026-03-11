# Model Performance Analysis Report

Generated: 2026-03-08 18:19:40

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
| Train | 0.1486 |
| Val | 0.1571 |
| Test | 0.1575 |

## Per-Task Performance

### Classification Tasks

| Task | Split | Accuracy | AUROC | Samples |
|------|-------|----------|-------|---------|
| is_metal | train | 0.8514 | 0.9271 | 123902 |
| is_metal | val | 0.8429 | 0.9202 | 15487 |
| is_metal | test | 0.8425 | 0.9165 | 15489 |

## Key Findings

### Classification Tasks Performance:
- **is_metal**: Accuracy = 0.8429, AUROC = 0.9202

## Overfitting Analysis


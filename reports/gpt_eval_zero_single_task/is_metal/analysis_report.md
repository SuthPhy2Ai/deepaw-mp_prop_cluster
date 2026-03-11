# Model Performance Analysis Report

Generated: 2026-03-09 23:17:04

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
| Train | 0.0888 |
| Val | 0.1233 |
| Test | 0.1218 |

## Per-Task Performance

### Classification Tasks

| Task | Split | Accuracy | AUROC | Samples |
|------|-------|----------|-------|---------|
| is_metal | train | 0.9112 | 0.9712 | 123902 |
| is_metal | val | 0.8767 | 0.9498 | 15487 |
| is_metal | test | 0.8782 | 0.9467 | 15489 |

## Key Findings

### Classification Tasks Performance:
- **is_metal**: Accuracy = 0.8767, AUROC = 0.9498

## Overfitting Analysis


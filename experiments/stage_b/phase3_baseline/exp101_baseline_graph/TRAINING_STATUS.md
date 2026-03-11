# Stage B Training Status

**Date**: 2026-03-06 15:32
**Experiment**: exp101_baseline_graph
**Status**: ✅ Training Running Successfully

---

## Current Status

### Training Process
- **PID**: 3814947
- **Started**: 2026-03-06 15:31:30 CST
- **CPU Usage**: 55.8%
- **Memory**: 9.4 GB
- **Progress**: Epoch 1, batch 183/1936 (9%)
- **Speed**: ~4-6 iterations/second
- **Status**: Running normally, no NaN/Inf errors

### What Happened

**Previous Issues (Resolved)**:
1. ❌ First attempt: NaN/Inf loss at batch 24
   - Cause: Extreme outlier values in elastic moduli (billions/trillions instead of 0-1000 GPa)
   - Also: Volume values up to 20,000+ causing gradient explosion

2. ✅ Fixed: Updated dataset validation
   - Elastic moduli: Filter to reasonable range (0-1000 GPa)
   - Universal anisotropy: Filter to 0-100
   - Volume: Filter to 0-10,000
   - Density: Filter to 0-50

3. ✅ Training relaunched successfully
   - Sampler cache loaded instantly (no 47-minute rebuild needed)
   - Passed batch 24 without errors
   - Currently at batch 183 and running smoothly

---

## Training Configuration

### Model
- **Backbone**: Graph (SchNet-style message passing)
- **Hidden dim**: 256
- **Layers**: 6
- **Cutoff**: 6.0 Å
- **Max neighbors**: 24

### Training
- **Stage**: B (18 tasks including elastic properties)
- **Batch size**: 64
- **Epochs**: 100
- **Learning rate**: 2e-4
- **Weight decay**: 1e-5
- **Gradient clipping**: 1.0
- **Warmup epochs**: 5
- **Elastic oversampling**: 4.0×

### Data
- **Training samples**: 123,903
- **Samples with elastic data**: 10,368 (8.37%)
- **Batches per epoch**: 1,936

---

## Expected Timeline

| Time | Event |
|------|-------|
| 15:31 | Training started |
| 15:31-15:38 | Epoch 1 (~7 minutes per epoch) |
| 15:38-16:30 | Epochs 2-10 |
| 16:30-17:30 | Epochs 11-20 |
| ... | ... |
| ~23:00 | Training completes (100 epochs, ~7.5 hours) |

**Estimated completion**: ~23:00 (7.5 hours total)

---

## Summary

✅ **Fixed Issues**:
- Dataset validation now filters extreme outlier values
- Elastic moduli limited to 0-1000 GPa (reasonable physical range)
- Volume limited to 0-10,000
- Density limited to 0-50

✅ **Training Status**:
- Running successfully without NaN/Inf errors
- Passed the problematic batch 24
- Currently at 9% of epoch 1
- Expected to complete in ~7.5 hours

🔄 **Next Steps**:
- Monitor convergence over 100 epochs
- Check validation metrics
- Analyze task-specific performance
- Compare to Stage A baseline results

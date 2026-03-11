# Phase 2 Quick Start Guide

**Date**: 2026-03-06
**Status**: 🚀 Ready to Launch

---

## 📋 Prerequisites

✅ Phase 1 completed and reorganized
✅ Experiment structure created
✅ Configuration files prepared
✅ Analysis tools ready

---

## 🚀 Quick Start

### 1. List Current Experiments

```bash
python scripts/experiment_manager.py list
```

### 2. Start Experiment 2 (Regularization)

```bash
# Create experiment directory
python scripts/experiment_manager.py create exp002 regularization

# Start training
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone graph \
  --batch-size 64 \
  --num-workers 8 \
  --exclude-tasks volume density is_stable \
  --epochs 100 \
  --lr 5e-5 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --warmup-epochs 5 \
  --dropout 0.1 \
  > experiments/stage_a/phase2_enhancements/exp002_regularization/training_log.txt 2>&1 &

# Monitor training
tail -f experiments/stage_a/phase2_enhancements/exp002_regularization/training_log.txt

# Or use TensorBoard
tensorboard --logdir experiments/stage_a/phase2_enhancements/exp002_regularization/tensorboard --port 6006 --bind_all
```

### 3. After Training Completes

```bash
# Copy best checkpoint
cp artifacts/runs/LATEST_RUN/checkpoints/best.pt \
   experiments/stage_a/phase2_enhancements/exp002_regularization/model_checkpoint.pt

# Run analysis
python scripts/experiment_manager.py analyze exp002 regularization

# View results
cat experiments/stage_a/phase2_enhancements/exp002_regularization/analysis/performance_report.md
```

---

## 📊 Experiment Workflow

### Standard Workflow for Each Experiment

1. **Create** experiment directory
2. **Configure** training parameters
3. **Launch** training in background
4. **Monitor** via logs or TensorBoard
5. **Analyze** results after completion
6. **Document** findings in README
7. **Compare** with baseline and other experiments

---

## 🔧 Useful Commands

### Monitor Training

```bash
# Watch training log
tail -f experiments/stage_a/phase2_enhancements/expXXX_name/training_log.txt

# Check GPU usage
nvidia-smi -l 1

# Check training process
ps aux | grep train_multitask
```

### TensorBoard

```bash
# Start TensorBoard for specific experiment
tensorboard --logdir experiments/stage_a/phase2_enhancements/expXXX_name/tensorboard --port 6006

# Compare multiple experiments
tensorboard --logdir experiments/stage_a/phase2_enhancements --port 6006
```

### Quick Analysis

```bash
# Check validation loss trend
grep "val_loss" experiments/stage_a/phase2_enhancements/expXXX_name/training_log.txt | tail -20

# Find best epoch
grep "best" experiments/stage_a/phase2_enhancements/expXXX_name/training_log.txt
```

---

## 📝 Experiment Checklist

### Before Training

- [ ] Review experiment configuration
- [ ] Verify data split exists
- [ ] Check GPU availability
- [ ] Ensure sufficient disk space
- [ ] Create experiment directory

### During Training

- [ ] Monitor training logs for errors
- [ ] Check TensorBoard curves
- [ ] Verify GPU utilization (>30%)
- [ ] Watch for NaN/Inf issues
- [ ] Monitor gradient norms

### After Training

- [ ] Copy best checkpoint to experiment dir
- [ ] Run comprehensive analysis
- [ ] Generate visualization
- [ ] Update experiment README
- [ ] Update EXPERIMENTS.md tracking file
- [ ] Compare with baseline

---

## 🎯 Phase 2 Experiment Sequence

### Week 1: Regularization & Architecture

**Experiment 2** (2-3 days):
- Focus: Reduce overfitting
- Expected: 1-2 days training + 1 day analysis

**Experiment 3** (2-3 days):
- Focus: Enhanced graph architecture
- Expected: 1-2 days training + 1 day analysis

### Week 2: Advanced Features

**Experiment 4** (3-4 days):
- Focus: Angle features
- Expected: 2-3 days training (slower) + 1 day analysis

**Comparison** (1 day):
- Compare Exp002-004
- Identify best approaches

### Week 3: Full Stack & Production

**Experiment 5** (3-4 days):
- Focus: Combined enhancements
- Expected: 2-3 days training + 1 day analysis

**Final Analysis** (2-3 days):
- Comprehensive comparison
- Production model selection
- Documentation

---

## 📈 Success Criteria

### Minimum Requirements (Must Achieve)

- ✅ Val loss < 0.20 (vs 0.2226 baseline)
- ✅ Train-val gap < 0.05 (vs 0.077 baseline)
- ✅ No NaN/Inf during training
- ✅ Training completes without crashes

### Target Goals (Should Achieve)

- 🎯 efermi MAE < 0.35 eV (vs 0.3834 baseline)
- 🎯 band_gap MAE < 0.20 eV (vs 0.2308 baseline)
- 🎯 cbm/vbm MAE < 0.25 eV (vs 0.29 baseline)
- 🎯 is_metal AUROC > 0.96 (vs 0.9575 baseline)

### Stretch Goals (Nice to Have)

- 🌟 Val loss < 0.18
- 🌟 All tasks improve by >15%
- 🌟 Train-val gap < 0.03

---

## 🔍 Troubleshooting

### Training Crashes

```bash
# Check for NaN/Inf
grep -i "nan\|inf" experiments/stage_a/phase2_enhancements/expXXX_name/training_log.txt

# Check GPU memory
nvidia-smi

# Reduce batch size if OOM
# Edit config: batch_size 64 → 32
```

### Slow Training

```bash
# Check GPU utilization
nvidia-smi -l 1

# Verify num_workers
# Should be 8 for good performance

# Check if graph cache is loaded
grep "cache" experiments/stage_a/phase2_enhancements/expXXX_name/training_log.txt
```

### Poor Performance

```bash
# Compare with baseline
python scripts/compare_experiments.py exp001 expXXX

# Check overfitting
# Look at train vs val loss gap

# Analyze per-task performance
cat experiments/stage_a/phase2_enhancements/expXXX_name/analysis/performance_report.md
```

---

## 📚 Key Files

### Configuration
- `configs/exp002_regularization.json` - Exp002 config
- `configs/exp003_enhanced_graph.json` - Exp003 config
- `configs/exp004_angle_features.json` - Exp004 config
- `configs/exp005_full_stack.json` - Exp005 config

### Documentation
- `experiments/EXPERIMENTS.md` - Experiment tracking
- `experiments/stage_a/summary.md` - Phase 2 summary
- `reports/PHASE2_TRAINING_PLAN.md` - Detailed plan

### Tools
- `scripts/experiment_manager.py` - Experiment management
- `scripts/train_multitask.py` - Training script
- `scripts/analyze_best_model.py` - Analysis script

---

## 🚀 Ready to Start?

**Recommended First Step**: Launch Experiment 2 (Regularization)

```bash
# Quick start command
cd /scratch/sutianhao/data/mp-data-pipeline

python scripts/experiment_manager.py create exp002 regularization

nohup python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone graph \
  --batch-size 64 \
  --num-workers 8 \
  --exclude-tasks volume density is_stable \
  --epochs 100 \
  --lr 5e-5 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --warmup-epochs 5 \
  --dropout 0.1 \
  > experiments/stage_a/phase2_enhancements/exp002_regularization/training_log.txt 2>&1 &

echo "Training started! Monitor with:"
echo "tail -f experiments/stage_a/phase2_enhancements/exp002_regularization/training_log.txt"
```

Good luck! 🎉

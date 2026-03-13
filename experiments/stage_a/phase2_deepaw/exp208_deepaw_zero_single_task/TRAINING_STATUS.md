# EXP-208 Training Status

**Experiment**: DeePAW Zero Single-Task Family
**Created**: 2026-03-14
**Status**: Ready for training

## Quick Start

```bash
# Verify setup (1-epoch test)
bash experiments/stage_a/phase2_deepaw/exp208_deepaw_zero_single_task/verify_setup.sh

# Start full training (all 15 tasks, ~10 hours)
bash experiments/stage_a/phase2_deepaw/exp208_deepaw_zero_single_task/training_cmd.sh

# Or train specific tasks
python scripts/run_exp208_deepaw_zero.py --tasks band_gap cbm vbm
```

## Training Progress

| Task | Status | Capacity | Samples | Epochs | Best Val Loss | Primary Metric | Notes |
|------|--------|----------|---------|--------|---------------|----------------|-------|
| energy_per_atom | ⏳ Pending | 256×6 | 123,902 | 50 | - | - | - |
| formation_energy_per_atom | ⏳ Pending | 256×6 | 123,902 | 50 | - | - | - |
| energy_above_hull | ⏳ Pending | 256×6 | 123,902 | 50 | - | - | - |
| band_gap | ⏳ Pending | 256×6 | 123,902 | 50 | - | - | - |
| cbm | ⏳ Pending | 224×6 | 71,574 | 55 | - | - | - |
| vbm | ⏳ Pending | 224×6 | 71,574 | 55 | - | - | - |
| efermi | ⏳ Pending | 256×6 | 123,856 | 50 | - | - | - |
| is_metal | ⏳ Pending | 256×6 | 123,902 | 50 | - | - | - |
| is_stable | ⏳ Pending | 256×6 | 123,902 | 50 | - | - | - |
| volume | ⏳ Pending | 320×7 | 123,890 | 70 | - | - | Special capacity |
| density | ⏳ Pending | 320×7 | 123,902 | 70 | - | - | Special capacity |
| bulk_modulus_vrh | ⏳ Pending | 160×4 | 10,217 | 80 | - | - | Low coverage |
| shear_modulus_vrh | ⏳ Pending | 160×4 | 9,685 | 80 | - | - | Low coverage |
| homogeneous_poisson | ⏳ Pending | 160×4 | 9,752 | 80 | - | - | Low coverage, --no-amp |
| universal_anisotropy | ⏳ Pending | 160×4 | 9,057 | 80 | - | - | Low coverage, --no-amp |

**Legend**:
- ⏳ Pending: Not started
- 🔄 Running: Training in progress
- ✅ Complete: Training finished successfully
- ❌ Failed: Training encountered errors

## Timeline

- **Setup completed**: 2026-03-14 04:13
- **Training started**: TBD
- **Training completed**: TBD
- **Estimated duration**: ~10 hours (sequential), ~5-6 hours (parallel with 2 GPUs)

## Results

Results will be saved to:
- **CSV**: `metrics/exp208_results.csv`
- **Logs**: `logs/<task>.log`
- **Checkpoints**: `artifacts/runs_exp208/<task>/<timestamp>/`

## Comparison with exp106

After training completes, compare performance:
- exp106: Pure Graph baseline (learnable embeddings)
- exp208: Enhanced Graph + DeePAW Replace (pretrained embeddings)

Expected improvements:
- Electronic properties: 10-20% MAE reduction
- Thermodynamic properties: 5-10% MAE reduction
- Structural/elastic properties: Minimal change

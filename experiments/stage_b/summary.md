# Stage B Summary

**Stage**: B (18 tasks including elastic properties)
**Status**: Not started yet (planned for Phase 3)

---

## Tasks Included (18)

### Stage A Tasks (8)
- energy_per_atom
- formation_energy_per_atom
- energy_above_hull
- band_gap
- cbm
- vbm
- efermi
- is_metal

### Additional Stage B Tasks (10)

**Structural (2)**:
- volume
- density

**Stability (1)**:
- is_stable

**Elastic/Mechanical (7)**:
- bulk_modulus_vrh
- shear_modulus_vrh
- youngs_modulus
- homogeneous_poisson
- universal_anisotropy
- (2 more elastic properties)

---

## Data Coverage Challenge

**Critical Issue**: Only ~7% of materials have elastic data

- Total materials: ~155k
- With elastic data: ~11k (7.1%)
- Without elastic data: ~144k (92.9%)

**Solution**: Use weighted sampling with `--oversample-elastic 4.0` to balance coverage

---

## Phase 3: Baseline (Planned)

### exp101_baseline_graph

**Goal**: Establish Stage B baseline
**Approach**: Same architecture as exp001, but with all 18 tasks
**Expected Challenges**:
- Lower coverage for elastic tasks
- Need careful loss weighting
- Longer training time

---

## Phase 3: Enhancements (Planned)

### exp102_regularization
Based on exp002 but with Stage B tasks

### exp103_enhanced_graph
Based on exp003 but with Stage B tasks

### exp104_full_stack
Based on exp005 but with Stage B tasks

---

## Prerequisites

Before starting Stage B:
1. ✅ Complete Stage A Phase 2 experiments
2. ✅ Identify best Stage A architecture
3. ✅ Understand elastic property prediction challenges
4. ✅ Validate weighted sampling strategy

---

## Expected Timeline

- **After Stage A Phase 2 completes** (~3-4 weeks)
- **Stage B Phase 3**: 4-6 weeks
  - exp101: 1 week (baseline)
  - exp102-104: 3-4 weeks (enhancements)
  - Analysis: 1 week

---

## Notes

- Stage B is significantly more challenging due to low elastic data coverage
- May need specialized heads for elastic properties
- Consider atomic baseline corrections for better energy predictions
- Elastic tensor predictions may require equivariant architectures

---

## Files

- Stage B experiments will be in: `stage_b/phase3_baseline/` and `stage_b/phase3_enhancements/`
- Experiment IDs: exp101-199 (reserved for Stage B)

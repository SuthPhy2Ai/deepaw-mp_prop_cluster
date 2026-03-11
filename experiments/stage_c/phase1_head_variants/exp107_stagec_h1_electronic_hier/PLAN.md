# Stage C h1 Plan

- experiment_id: `exp107`
- head_variant: `stagec_h1`
- description: Electronic hierarchical head: group head -> task heads for band tasks
- base_checkpoint: `artifacts/runs_stageb_v3/20260308_070539/checkpoints/best.pt`
- stage: `b`
- freeze_backbone: `True`
- use_pyg: `True`
- epochs: `35`
- lr: `0.0002`
- runs_root: `artifacts/runs_stagec_h1`

## Intent

- h1: test group-head to task-head hierarchy for electronic tasks.
- h2: test base-elastic to derived-elastic hierarchy for Poisson/anisotropy.
- hybrid: combine h1 and h2 in one shared-backbone experiment.

# Stage C hybrid Plan

- experiment_id: `exp109`
- head_variant: `stagec_hybrid`
- description: Hybrid hierarchical head: electronic hierarchy + elastic derived hierarchy
- base_checkpoint: `artifacts/runs_stageb_v3/20260308_070539/checkpoints/best.pt`
- stage: `b`
- freeze_backbone: `True`
- use_pyg: `True`
- epochs: `35`
- lr: `0.0002`
- runs_root: `artifacts/runs_stagec_hybrid`

## Intent

- h1: test group-head to task-head hierarchy for electronic tasks.
- h2: test base-elastic to derived-elastic hierarchy for Poisson/anisotropy.
- hybrid: combine h1 and h2 in one shared-backbone experiment.

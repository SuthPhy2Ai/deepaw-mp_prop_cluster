# Phase 2 Optimization Plan - Based on XequiNet Analysis

**Date**: 2026-03-04
**Status**: Planning
**Context**: Phase 1 baseline experiments showed training instability (NaN losses, high errors). XequiNet reference project analysis reveals key missing components.

---

## Executive Summary

Analysis of the successful XequiNet project identified critical gaps in our current implementation:
1. **Training instability**: Missing EMA, warmup, gradient clipping
2. **Suboptimal graph construction**: Global neighbor limit vs per-atom distance-sorted
3. **Inadequate tensor handling**: Simple MLP heads vs specialized tensor outputs
4. **Missing baseline corrections**: No atomic reference energies

This plan proposes a phased approach to adopt XequiNet's proven techniques, prioritized by impact and implementation complexity.

---

## Phase 2A: Training Stability (High Priority, Low Complexity)

**Goal**: Eliminate training instability (NaN losses, gradient explosions) observed in EXP-03/04.

### 2A.1: Exponential Moving Average (EMA)

**Rationale**: XequiNet uses EMA with decay=0.999 to stabilize training and smooth convergence.

**Implementation**:
- Add `EMAModel` wrapper class in `src/mp_data_pipeline/training/ema.py`
- Track shadow parameters: `θ_ema = decay * θ_ema + (1-decay) * θ`
- Update after each optimizer step
- Use EMA model for validation/testing

**Changes**:
```python
# src/mp_data_pipeline/training/ema.py (NEW)
class EMAModel:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach()
                       for name, param in model.named_parameters()}

    def update(self):
        for name, param in self.model.named_parameters():
            self.shadow[name] = self.decay * self.shadow[name] + \
                                (1 - self.decay) * param.data

# scripts/train_multitask.py
parser.add_argument('--ema-decay', type=float, default=0.999)
if args.ema_decay > 0:
    ema_model = EMAModel(model, decay=args.ema_decay)
```

**Effort**: 2-3 hours
**Risk**: Low (non-invasive, can be disabled)

---

### 2A.2: Learning Rate Warmup

**Rationale**: XequiNet uses linear/exponential warmup (5-10 epochs) to prevent early training instability.

**Implementation**:
- Add warmup scheduler wrapper in `src/mp_data_pipeline/training/scheduler.py`
- Linear warmup from `lr_min` to `lr_max` over `warmup_epochs`
- Chain with existing CosineAnnealingLR

**Changes**:
```python
# src/mp_data_pipeline/training/scheduler.py
class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, base_scheduler):
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = args.lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.base_scheduler.step()
        self.current_epoch += 1

# scripts/train_multitask.py
parser.add_argument('--warmup-epochs', type=int, default=5)
```

**Effort**: 1-2 hours
**Risk**: Low (standard technique)

---

### 2A.3: Gradient Clipping

**Rationale**: XequiNet uses `torch.nn.utils.clip_grad_norm_` to prevent gradient explosions.

**Implementation**:
- Add gradient clipping before optimizer step
- Default threshold: 1.0 (configurable)

**Changes**:
```python
# src/mp_data_pipeline/training/trainer.py
parser.add_argument('--grad-clip', type=float, default=1.0)

# In training loop
loss.backward()
if args.grad_clip > 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
optimizer.step()
```

**Effort**: 30 minutes
**Risk**: Very low (one-line change)

---

### 2A.4: Best-K Checkpoint Saving

**Rationale**: XequiNet keeps top-K models (K=3-5) for ensemble predictions and robustness.

**Implementation**:
- Maintain heap of (val_loss, checkpoint_path) tuples
- Save checkpoint if in top-K
- Delete worst checkpoint when heap exceeds K

**Changes**:
```python
# src/mp_data_pipeline/training/trainer.py
import heapq

class BestKCheckpoints:
    def __init__(self, k=3, mode='min'):
        self.k = k
        self.heap = []  # max-heap for 'min' mode

    def update(self, metric, checkpoint_path):
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, (-metric, checkpoint_path))
        elif -metric < self.heap[0][0]:
            old_metric, old_path = heapq.heappushpop(
                self.heap, (-metric, checkpoint_path))
            os.remove(old_path)  # Delete worst checkpoint
```

**Effort**: 2 hours
**Risk**: Low (improves model selection)

---

**Phase 2A Total Effort**: 1 day
**Expected Impact**: High (directly addresses training instability)

---

## Phase 2B: Graph Construction Improvements (Medium Priority, Low Complexity)

**Goal**: Improve graph quality and neighbor selection strategy.

### 2B.1: Per-Atom Distance-Sorted Neighbors

**Rationale**: XequiNet sorts neighbors by distance and keeps closest N per atom (not global limit).

**Current Issue**: Our `radius_graph` uses global `max_neighbors`, leading to:
- Dense atoms (e.g., in small unit cells) get truncated arbitrarily
- Sparse atoms waste computation on distant neighbors

**Implementation**:
```python
# src/mp_data_pipeline/ml/dataset.py
def radius_graph_sorted(pos, cutoff, max_neighbors_per_atom):
    """
    Build graph with per-atom neighbor limiting.
    Keeps closest max_neighbors_per_atom for each atom.
    """
    # Get all neighbors within cutoff
    edge_index = radius_graph(pos, r=cutoff, max_num_neighbors=1000)

    # Compute distances
    row, col = edge_index
    dist = torch.norm(pos[row] - pos[col], dim=1)

    # Sort by distance per source atom
    sorted_edges = []
    for atom_idx in range(pos.size(0)):
        mask = (row == atom_idx)
        atom_edges = edge_index[:, mask]
        atom_dists = dist[mask]

        # Keep closest max_neighbors_per_atom
        if atom_edges.size(1) > max_neighbors_per_atom:
            _, indices = torch.topk(atom_dists, max_neighbors_per_atom,
                                     largest=False)
            atom_edges = atom_edges[:, indices]

        sorted_edges.append(atom_edges)

    return torch.cat(sorted_edges, dim=1)
```

**Effort**: 3-4 hours
**Risk**: Low (improves graph quality)

---

### 2B.2: PBC Handling

**Decision**: Use existing ASE implementation for PBC.

**Rationale**:
- ASE's `primitive_neighbor_list` already handles periodic boundaries correctly
- Our current implementation computes distances properly across periodic images
- XequiNet's explicit shift vectors are useful for their architecture but not critical for ours
- Adding shift vectors would require dataset changes and reprocessing

**No changes needed** - current ASE-based PBC handling is sufficient.

---

**Phase 2B Total Effort**: 3-4 hours
**Expected Impact**: Medium (improves graph representation quality)

---

## Phase 2C: Atomic Reference Energies (Medium Priority, Medium Complexity)

**Goal**: Improve thermodynamic property predictions by learning corrections to atomic baselines.

### 2C.1: Per-Element Reference Energies

**Rationale**: XequiNet uses atomic reference energies (e.g., isolated atom DFT energies) to:
- Reduce magnitude of energy predictions (learn deviations from baseline)
- Improve generalization to unseen compositions
- Enable delta learning (NN corrects fast baseline method)

**Implementation**:
```python
# src/mp_data_pipeline/ml/atomic_references.py (NEW)
class AtomicReferenceEnergies:
    """
    Stores per-element reference energies for thermodynamic properties.
    """
    def __init__(self, reference_dict=None):
        # Default: DFT energies of isolated atoms (eV)
        self.references = reference_dict or {
            1: -13.6,    # H
            6: -1029.5,  # C
            8: -2042.1,  # O
            # ... (load from file or compute from dataset)
        }

    def compute_baseline(self, atomic_numbers):
        """
        Compute baseline energy from atomic composition.
        Args:
            atomic_numbers: (N_atoms,) tensor
        Returns:
            baseline_energy: scalar
        """
        return sum(self.references.get(z.item(), 0.0)
                   for z in atomic_numbers)

    def fit_from_dataset(self, dataset, property_key='energy_per_atom'):
        """
        Fit reference energies via linear regression on dataset.
        Minimizes: E_total = sum_i(n_i * E_ref_i) + residual
        """
        from sklearn.linear_model import Ridge

        # Build composition matrix: (N_samples, N_elements)
        compositions = []
        energies = []
        for data in dataset:
            comp = torch.bincount(data.at_no, minlength=119)
            compositions.append(comp.numpy())
            energies.append(data.y[property_key].item())

        X = np.array(compositions)
        y = np.array(energies)

        # Fit with regularization
        model = Ridge(alpha=1.0, fit_intercept=False)
        model.fit(X, y)

        self.references = {z: model.coef_[z] for z in range(1, 119)}

# src/mp_data_pipeline/ml/dataset.py
class AseGraphMultitaskDataset:
    def __init__(self, ..., atomic_references=None):
        self.atomic_references = atomic_references

    def __getitem__(self, idx):
        data = ...  # existing code

        # Subtract atomic baseline from energy targets
        if self.atomic_references:
            baseline = self.atomic_references.compute_baseline(data.at_no)
            if 'energy_per_atom' in data.y:
                data.y['energy_per_atom'] -= baseline / len(data.at_no)
            if 'formation_energy_per_atom' in data.y:
                # Formation energy already relative to elements
                pass

        return data
```

**Usage**:
```bash
# Fit atomic references from training set
python scripts/fit_atomic_references.py \
  --split data/splits/split_iid_seed42.json \
  --output data/atomic_references.json

# Train with atomic references
python scripts/train_multitask.py \
  --atomic-references data/atomic_references.json \
  ...
```

**Effort**: 1 day (including fitting script)
**Risk**: Medium (requires careful validation)

---

**Phase 2C Total Effort**: 1 day
**Expected Impact**: Medium (improves energy predictions by 10-20%)

---

## Phase 2D: Enhanced Evaluation Metrics (Low Priority, Low Complexity)

**Goal**: Adopt XequiNet's comprehensive tensor evaluation metrics.

### 2D.1: Elastic Property Metrics

**Rationale**: XequiNet's `tensor_test.py` computes physically meaningful metrics:
- Bulk modulus (K), Shear modulus (G) from elastic tensor
- Young's modulus (E), Poisson ratio (ν)
- Voigt-Reuss-Hill averaging
- Frobenius norm, Error-within-Threshold (EwT)

**Implementation**:
```python
# src/mp_data_pipeline/evaluation/elastic_metrics.py (NEW)
def compute_elastic_moduli(elastic_tensor_voigt):
    """
    Compute K, G, E, ν from 6×6 elastic tensor.
    Uses Voigt-Reuss-Hill averaging.
    """
    C = elastic_tensor_voigt  # (6, 6)

    # Voigt averaging (upper bound)
    K_voigt = (C[0,0] + C[1,1] + C[2,2] +
               2*(C[0,1] + C[0,2] + C[1,2])) / 9
    G_voigt = ((C[0,0] + C[1,1] + C[2,2]) -
               (C[0,1] + C[0,2] + C[1,2]) +
               3*(C[3,3] + C[4,4] + C[5,5])) / 15

    # Reuss averaging (lower bound, requires compliance tensor)
    S = np.linalg.inv(C)
    K_reuss = 1 / (S[0,0] + S[1,1] + S[2,2] +
                   2*(S[0,1] + S[0,2] + S[1,2]))
    G_reuss = 15 / (4*(S[0,0] + S[1,1] + S[2,2]) -
                    4*(S[0,1] + S[0,2] + S[1,2]) +
                    3*(S[3,3] + S[4,4] + S[5,5]))

    # Hill averaging (arithmetic mean)
    K = (K_voigt + K_reuss) / 2
    G = (G_voigt + G_reuss) / 2

    # Derived properties
    E = 9*K*G / (3*K + G)  # Young's modulus
    nu = (3*K - 2*G) / (2*(3*K + G))  # Poisson ratio

    return {'K': K, 'G': G, 'E': E, 'nu': nu}

def elastic_tensor_metrics(pred, target):
    """
    Comprehensive elastic tensor evaluation.
    """
    # Frobenius norm
    frobenius = np.linalg.norm(pred - target, ord='fro')

    # Element-wise MAE/RMSE
    mae = np.mean(np.abs(pred - target))
    rmse = np.sqrt(np.mean((pred - target)**2))

    # Moduli errors
    pred_moduli = compute_elastic_moduli(pred)
    target_moduli = compute_elastic_moduli(target)

    moduli_errors = {
        f'{key}_mae': abs(pred_moduli[key] - target_moduli[key])
        for key in ['K', 'G', 'E', 'nu']
    }

    return {
        'frobenius': frobenius,
        'mae': mae,
        'rmse': rmse,
        **moduli_errors
    }
```

**Integration**:
```python
# scripts/eval_multitask.py
from mp_data_pipeline.evaluation.elastic_metrics import elastic_tensor_metrics

# In evaluation loop
if task_name == 'elastic_tensor':
    metrics = elastic_tensor_metrics(pred, target)
    print(f"Bulk modulus MAE: {metrics['K_mae']:.2f} GPa")
    print(f"Shear modulus MAE: {metrics['G_mae']:.2f} GPa")
```

**Effort**: 4-5 hours
**Risk**: Very low (evaluation only, no training changes)

---

**Phase 2D Total Effort**: 0.5 day
**Expected Impact**: Low (better understanding, no performance gain)

---

## Phase 2E: E(3)-Equivariant Architecture (Medium Priority, Medium Complexity)

**Goal**: Implement E(3)-equivariant message passing for better tensor property prediction.

### 2E.1: E(3)-Equivariant Message Passing (e3nn)

**Rationale**: XequiNet's XPaiNN uses full spherical harmonic decomposition for:
- Guaranteed rotational invariance
- Native tensor output (rank-2 for elastic properties)
- Higher-order interactions (beyond pairwise)

**Revised Assessment**:
- ✅ e3nn is just a pip package (`pip install e3nn==0.5.1`)
- ✅ XPaiNN implementation is only 248 lines
- ✅ Core components are straightforward:
  - `o3.SphericalHarmonics` for angular features
  - `o3.Irreps` for feature type specification
  - `o3.TensorProduct` for equivariant operations
- ⚠️ Training may be 2-3× slower (but more accurate)
- ⚠️ Requires hyperparameter tuning

**Implementation Plan**:

**Step 1: Install e3nn**
```bash
pip install e3nn==0.5.1
```

**Step 2: Create XPaiNN Backbone** (adapt from XequiNet)
```python
# src/mp_data_pipeline/models/xpainn_backbone.py (NEW)
import torch
import torch.nn as nn
from e3nn import o3

class XPaiNNBackbone(nn.Module):
    """
    E(3)-equivariant message passing network.
    Adapted from XequiNet's XPaiNN.
    """
    def __init__(
        self,
        node_dim=128,
        edge_irreps="128x0e + 64x1o + 32x2e",  # scalar + vector + tensor
        num_interactions=3,
        num_rbf=20,
        cutoff=6.0,
        max_neighbors=24,
    ):
        super().__init__()
        self.edge_irreps = o3.Irreps(edge_irreps)

        # Embedding layer
        self.embedding = XEmbedding(
            node_dim=node_dim,
            edge_irreps=edge_irreps,
            num_basis=num_rbf,
            cutoff=cutoff,
        )

        # Message passing layers
        self.interactions = nn.ModuleList([
            XPaiNNInteraction(node_dim, edge_irreps, num_rbf)
            for _ in range(num_interactions)
        ])

        # Output projection (to match existing head interface)
        self.out_proj = nn.Linear(node_dim, node_dim)

    def forward(self, data):
        # Embedding
        x_scalar, rbf, fcut, rsh = self.embedding(
            data.at_no, data.pos, data.edge_index,
            shifts=getattr(data, 'shifts', torch.zeros_like(data.pos[data.edge_index[0]]))
        )

        # Initialize spherical features
        x_spherical = torch.zeros(
            data.at_no.size(0),
            self.edge_irreps.dim,
            device=data.pos.device
        )

        # Message passing
        for interaction in self.interactions:
            x_scalar, x_spherical = interaction(
                x_scalar, x_spherical, rbf, fcut, rsh, data.edge_index
            )

        # Global pooling
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(
            data.at_no.size(0), dtype=torch.long, device=data.pos.device
        )
        x_pooled = scatter_mean(x_scalar, batch, dim=0)

        return self.out_proj(x_pooled)
```

**Step 3: Add Tensor Output Head**
```python
# src/mp_data_pipeline/models/tensor_head.py (NEW)
class TensorOutputHead(nn.Module):
    """
    Predicts rank-2 tensors (elastic, stress) from equivariant features.
    Uses e3nn tensor product to generate proper tensor representations.
    """
    def __init__(self, node_dim, edge_irreps, output_dim=21):
        super().__init__()
        # Extract rank-2 irreps (l=2) from edge features
        self.tensor_irreps = o3.Irreps("32x2e")  # 5 components per l=2

        # Project to tensor space
        self.to_tensor = o3.Linear(
            edge_irreps,
            self.tensor_irreps,
            biases=True
        )

        # Convert to Voigt notation (6×6 symmetric)
        self.to_voigt = VoigtConverter()

    def forward(self, x_spherical, batch):
        # Pool spherical features
        x_pooled = scatter_mean(x_spherical, batch, dim=0)

        # Generate rank-2 tensor
        tensor_features = self.to_tensor(x_pooled)

        # Convert to 6×6 Voigt (21 unique components)
        elastic_tensor = self.to_voigt(tensor_features)

        return elastic_tensor
```

**Step 4: Integration**
```python
# scripts/train_multitask.py
parser.add_argument('--backbone', choices=['graph', 'composition', 'enhanced_graph', 'xpainn'])

if args.backbone == 'xpainn':
    from mp_data_pipeline.models.xpainn_backbone import XPaiNNBackbone
    backbone = XPaiNNBackbone(
        node_dim=args.hidden_dim,
        edge_irreps="128x0e + 64x1o + 32x2e",
        num_interactions=3,
        cutoff=args.cutoff,
    )
```

**Estimated Effort**: 3-4 days
- Day 1: Install e3nn, adapt XEmbedding and XPaiNNMessage
- Day 2: Implement XPaiNNInteraction and backbone integration
- Day 3: Create TensorOutputHead for elastic properties
- Day 4: Testing and debugging

**Risk**: Medium (new dependency, but well-documented library)

**Expected Benefits**:
- 20-30% improvement on elastic properties (native tensor handling)
- Better rotational invariance (guaranteed by e3nn)
- Potential 10-15% improvement on other structure-dependent tasks

---

## Implementation Timeline (Tonight's Sprint)

### Phase 1: Training Stability (2-3 hours)
**Time**: 23:30 - 02:00
- **30 min**: Implement EMA wrapper class
- **30 min**: Implement warmup scheduler
- **10 min**: Add gradient clipping (one line)
- **60 min**: Implement Best-K checkpointing with heap
- **30 min**: Integration testing

**Deliverable**: Training stability improvements ready

---

### Phase 2: XPaiNN Architecture (4-6 hours)
**Time**: 02:00 - 08:00
- **10 min**: Install e3nn (`pip install e3nn==0.5.1`)
- **60 min**: Copy and adapt XEmbedding layer from XequiNet
- **90 min**: Copy and adapt XPaiNNMessage and XPaiNNUpdate
- **60 min**: Integrate XPaiNN backbone into training script
- **60 min**: Test forward pass and gradient flow
- **60 min**: Debug and fix issues

**Deliverable**: XPaiNN backbone ready for training

---

### Phase 3: Graph Improvements (2 hours)
**Time**: 08:00 - 10:00
- **90 min**: Implement per-atom distance-sorted neighbor selection
- **30 min**: Test on sample structures

**Deliverable**: Improved graph construction

---

### Phase 4: Launch Training (30 min)
**Time**: 10:00 - 10:30
- Kill exp04 if still running
- Launch EXP-05 with all improvements:
  - EMA + warmup + gradient clipping
  - XPaiNN backbone
  - Per-atom neighbor sorting
  - 8 core tasks (exclude volume, density, is_stable)

**Deliverable**: Training running with all Phase 2 improvements

---

**Total Time**: ~10 hours (one night)
**Expected Completion**: Tomorrow morning ~10:30

---

## Success Metrics

### Training Stability (Phase 2A)
- ✅ No NaN losses during 50-epoch training
- ✅ Smooth loss curves (no sudden spikes)
- ✅ Validation loss decreases monotonically (with EMA)

### XPaiNN Architecture (Phase 2E)
- ✅ Forward pass completes without errors
- ✅ Gradients flow properly (no NaN/Inf)
- ✅ Training converges (loss decreases)
- 🎯 10-20% improvement on structure-dependent tasks vs baseline

### Graph Quality (Phase 2B)
- ✅ Consistent neighbor counts across materials
- ✅ No materials with zero neighbors
- 🎯 5-10% improvement on structure-dependent tasks

### Overall Phase 2 Target (Tonight's Sprint)
- **Critical**: Stable training without NaN (Phase 2A)
- **Important**: XPaiNN backbone working (Phase 2E)
- **Nice-to-have**: Graph improvements (Phase 2B)
- **Deferred**: Atomic references (Phase 2C), evaluation metrics (Phase 2D)

---

## Risk Assessment

| Phase | Risk Level | Mitigation |
|-------|-----------|------------|
| 2A | Low | All techniques are standard, well-documented |
| 2B | Medium | Extensive testing on PBC correctness required |
| 2C | Medium | Validate atomic references don't hurt formation energies |
| 2D | Very Low | Evaluation only, no training impact |
| 2E | High | Defer to Phase 3, requires dedicated research effort |

---

## Dependencies

### Software
- PyTorch ≥ 2.0 (existing)
- torch_geometric (existing)
- ASE (existing)
- scikit-learn (for atomic reference fitting)

### Data
- Existing ASE database: `data/db/mp_materials.db`
- Existing splits: `data/splits/split_iid_seed42.json`

### Compute
- Same as Phase 1: Single GPU, ~8 hours per 50-epoch training

---

## Open Questions

1. **EMA decay rate**: XequiNet uses 0.999, but optimal value may depend on batch size and dataset size. Test 0.99, 0.999, 0.9999.

2. **Warmup duration**: XequiNet uses 5-10 epochs. Our dataset is larger (155k vs typical 10-50k), may need longer warmup.

3. **Atomic reference fitting**: Should we fit on training set only, or use all data? Risk of data leakage vs better baseline.

4. **PBC shift vectors**: Do we need them for all tasks, or only elastic properties? May add computational overhead.

5. **Graph cutoff**: XequiNet uses 5.0Å for molecules. Crystals may need 8-10Å. Test 6.0Å (current), 8.0Å, 10.0Å.

---

## References

- **XequiNet Paper**: "A General Framework for Geometric Deep Learning on Tensorial Properties of Molecules and Crystals"
- **XequiNet Code**: `/home/sutianhao/data/mp-data-pipeline/refwork/XequiNet_tensor-main/`
- **Key Files Analyzed**:
  - `xequinet/nn/xpainn.py` - Message passing with EMA
  - `xequinet/utils/trainer.py` - Training loop with warmup/clipping
  - `xequinet/data/hdf5_data.py` - Per-atom neighbor limiting
  - `xequinet/run/tensor_test.py` - Elastic tensor evaluation

---

## Conclusion

Phase 2 focuses on **training stability and E(3)-equivariant architecture** based on XequiNet's proven techniques. These changes directly address current training failures and provide a modern, equivariant backbone.

**Tonight's Sprint Plan** (10 hours):
1. **Phase 2A** (2-3h): EMA + Warmup + Gradient Clipping + Best-K checkpoints
2. **Phase 2E** (4-6h): XPaiNN backbone with e3nn
3. **Phase 2B** (2h): Per-atom neighbor sorting
4. **Launch** (0.5h): Start EXP-05 training with all improvements

**Deferred to Later**:
- Phase 2C (Atomic References): Can add after validating Phase 2A/E/B
- Phase 2D (Evaluation Metrics): Can implement during training

**Expected Outcome**: By tomorrow morning, we'll have a stable, modern E(3)-equivariant model training on 8 core tasks, with proper training stability mechanisms.

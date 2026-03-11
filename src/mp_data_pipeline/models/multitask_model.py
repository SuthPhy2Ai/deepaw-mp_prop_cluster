"""End-to-end multitask model with selectable backbone."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..ml.tasks import TASK_NAME_LIST
from .backbones import CompositionBackbone, GraphBackbone
from .enhanced_backbones import EnhancedGraphBackbone
from .xpainn_backbone import XPaiNNBackbone
from .heads import (
    ElasticDerivedHeads,
    ElectronicHierarchicalHeads,
    GroupedTaskHeads,
    HybridHierarchicalHeads,
)


class MultitaskPropertyModel(nn.Module):
    """Shared backbone + grouped heads for property prediction."""

    def __init__(
        self,
        backbone_name: str = "graph",
        hidden_dim: int = 256,
        n_layers: int = 6,
        cutoff: float = 6.0,
        n_rbf: int = 64,
        dropout: float = 0.1,
        use_angles: bool = False,
        use_edge_update: bool = False,
        use_deepaw_features: bool = False,
        deepaw_checkpoint: str = None,
        deepaw_fusion: str = "add",
        head_variant: str = "grouped",
    ):
        super().__init__()

        if backbone_name == "graph":
            self.backbone = GraphBackbone(
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                cutoff=cutoff,
                n_rbf=n_rbf,
            )
        elif backbone_name == "enhanced_graph":
            # Use EnhancedGraphBackbone but with same interface as GraphBackbone
            # This allows us to use the fast data loading from original dataset
            self.backbone = EnhancedGraphBackbone(
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                cutoff=cutoff,
                n_rbf=n_rbf,
                use_angles=use_angles,
                use_edge_update=use_edge_update,
                use_deepaw_features=use_deepaw_features,
                deepaw_checkpoint=deepaw_checkpoint,
                deepaw_fusion=deepaw_fusion,
            )
        elif backbone_name == "graph_enhanced":
            # Alias: use enhanced backbone with graph data loading
            # This is the recommended way for Phase 2 experiments
            self.backbone = EnhancedGraphBackbone(
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                cutoff=cutoff,
                n_rbf=n_rbf,
                use_angles=False,  # No angles with standard dataset
                use_edge_update=use_edge_update,
            )
        elif backbone_name == "composition":
            self.backbone = CompositionBackbone(hidden_dim=hidden_dim)
        elif backbone_name == "xpainn":
            # Phase 2E: E(3)-equivariant backbone
            self.backbone = XPaiNNBackbone(
                node_dim=hidden_dim,
                edge_irreps="128x0e + 64x1o + 32x2e",
                num_interactions=n_layers // 2,  # Fewer interactions for XPaiNN
                num_rbf=n_rbf,
                cutoff=cutoff,
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        head_variant = head_variant.lower()
        head_cls = {
            "grouped": GroupedTaskHeads,
            "stagec_h1": ElectronicHierarchicalHeads,
            "stagec_h2": ElasticDerivedHeads,
            "stagec_hybrid": HybridHierarchicalHeads,
        }.get(head_variant)
        if head_cls is None:
            raise ValueError(f"Unsupported head variant: {head_variant}")

        self.head_variant = head_variant
        self.heads = head_cls(hidden_dim=hidden_dim, head_hidden_dim=hidden_dim, dropout=dropout)

        self.task_to_index: Dict[str, int] = {name: i for i, name in enumerate(TASK_NAME_LIST)}

    def forward(self, batch_dict: dict) -> torch.Tensor:
        graph_emb = self.backbone(batch_dict)
        head_out = self.heads(graph_emb)

        ordered_outputs = []
        for task_name in TASK_NAME_LIST:
            ordered_outputs.append(head_out[task_name])
        pred = torch.stack(ordered_outputs, dim=-1)

        # Note: band_gap constraint already applied in GroupedTaskHeads
        # No need for duplicate softplus here

        return pred

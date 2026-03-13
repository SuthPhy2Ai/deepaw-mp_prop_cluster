#!/usr/bin/env python3
"""Enhanced Graph Backbone with angle features and improved architecture.

This module provides an enhanced version of GraphBackbone that includes:
- Angle features (three-body interactions)
- Larger graph coverage (cutoff, max_neighbors)
- Enhanced edge features (more RBF basis functions)
- Optional edge update mechanism
- AMP compatibility fixes
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from .graph_features import AngleExpansion


class EnhancedRBFExpansion(nn.Module):
    """Enhanced Radial Basis Function expansion with more basis functions."""

    def __init__(self, n_rbf: int = 128, cutoff: float = 8.0, trainable: bool = False):
        """Initialize RBF expansion.

        Args:
            n_rbf: Number of radial basis functions (default: 128, up from 64)
            cutoff: Cutoff radius in Angstroms (default: 8.0, up from 6.0)
            trainable: Whether basis centers and widths are trainable
        """
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff

        # Centers uniformly distributed in [0, cutoff]
        centers = torch.linspace(0, cutoff, n_rbf)
        self.register_buffer("centers", centers)

        # Width parameter
        spacing = cutoff / (n_rbf - 1) if n_rbf > 1 else 1.0
        gamma = 1.0 / (2 * spacing ** 2)

        if trainable:
            self.centers = nn.Parameter(self.centers)
            self.gamma = nn.Parameter(torch.tensor(gamma))
        else:
            self.register_buffer("gamma", torch.tensor(gamma))

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Expand distances using Gaussian RBF.

        Args:
            distances: (num_edges,) tensor of edge distances

        Returns:
            (num_edges, n_rbf) tensor of expanded features
        """
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * diff * diff)


class EnhancedMessagePassingLayer(nn.Module):
    """Enhanced message passing layer with angle features and optional edge updates."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        angle_dim: int = 0,
        use_edge_update: bool = False,
        activation: str = "silu",
    ):
        """Initialize enhanced message passing layer.

        Args:
            hidden_dim: Dimension of node embeddings
            edge_dim: Dimension of edge features
            angle_dim: Dimension of angle features (0 = no angles)
            use_edge_update: Whether to update edge features
            activation: Activation function ('silu' or 'relu')
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.angle_dim = angle_dim
        self.use_edge_update = use_edge_update

        # Activation function
        if activation == "silu":
            self.act = nn.SiLU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Message MLP: [h_src, h_dst, edge_feat] → message
        msg_input_dim = hidden_dim * 2 + edge_dim
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_input_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Edge update MLP (optional)
        if use_edge_update:
            edge_update_input_dim = edge_dim + hidden_dim  # edge_feat + message
            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_update_input_dim, edge_dim),
                self.act,
                nn.Linear(edge_dim, edge_dim),
            )

        # Node update MLP: [h_node, aggregated_messages] → h_node'
        self.upd_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of message passing layer.

        Args:
            node_emb: (num_nodes, hidden_dim) node embeddings
            edge_index: (2, num_edges) edge indices [src, dst]
            edge_feat: (num_edges, edge_dim) edge features

        Returns:
            updated_node_emb: (num_nodes, hidden_dim) updated node embeddings
            updated_edge_feat: (num_edges, edge_dim) updated edge features
        """
        src, dst = edge_index[0], edge_index[1]

        # Compute messages
        msg_input = torch.cat([node_emb[src], node_emb[dst], edge_feat], dim=-1)
        messages = self.msg_mlp(msg_input)  # (num_edges, hidden_dim)

        # Edge update (optional)
        if self.use_edge_update:
            edge_update_input = torch.cat([edge_feat, messages], dim=-1)
            edge_feat = edge_feat + self.edge_mlp(edge_update_input)

        # Aggregate messages to destination nodes
        # CRITICAL: Use dtype=messages.dtype for AMP compatibility
        agg = torch.zeros_like(node_emb, dtype=messages.dtype)
        agg.index_add_(0, dst, messages)

        # Update node embeddings
        update = self.upd_mlp(torch.cat([node_emb, agg], dim=-1))
        node_emb = self.norm(node_emb + update)

        return node_emb, edge_feat


class EnhancedGraphBackbone(nn.Module):
    """Enhanced Graph Backbone with angle features and improved architecture.

    Key improvements over GraphBackbone:
    - Larger graph coverage (cutoff=8.0Å, max_neighbors=48)
    - More RBF basis functions (n_rbf=128)
    - Optional angle features (three-body interactions)
    - Optional edge update mechanism
    - AMP compatibility ensured
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 6,
        cutoff: float = 8.0,
        n_rbf: int = 128,
        use_angles: bool = False,
        n_angle_basis: int = 32,
        use_edge_update: bool = False,
        activation: str = "silu",
        use_deepaw_features: bool = False,
        deepaw_checkpoint: Optional[str] = None,
        deepaw_fusion: str = "add",
    ):
        """Initialize enhanced graph backbone.

        Args:
            hidden_dim: Dimension of node embeddings
            n_layers: Number of message passing layers
            cutoff: Cutoff radius for graph construction (Angstroms)
            n_rbf: Number of radial basis functions for edge features
            use_angles: Whether to use angle features
            n_angle_basis: Number of angle basis functions
            use_edge_update: Whether to update edge features in message passing
            activation: Activation function ('silu' or 'relu')
            use_deepaw_features: Whether to use DeePAW pretrained atom features
            deepaw_checkpoint: Path to DeePAW checkpoint (required if use_deepaw_features=True)
            deepaw_fusion: How to fuse DeePAW features ('add' or 'concat')
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.cutoff = cutoff
        self.n_rbf = n_rbf
        self.use_angles = use_angles
        self.use_edge_update = use_edge_update
        self.use_deepaw_features = use_deepaw_features
        self.deepaw_fusion = deepaw_fusion

        # Atom embedding (atomic number → hidden_dim)
        # Support up to atomic number 118
        self.atom_emb = nn.Embedding(119, hidden_dim)

        # DeePAW feature extractor (optional)
        if use_deepaw_features:
            if deepaw_checkpoint is None:
                raise ValueError("deepaw_checkpoint must be provided when use_deepaw_features=True")

            from .deepaw_extractor import DeePAWAtomFeatureExtractor

            self.deepaw_extractor = DeePAWAtomFeatureExtractor(
                checkpoint_path=deepaw_checkpoint,
                device="cuda" if torch.cuda.is_available() else "cpu",
                freeze=True,  # Freeze DeePAW weights by default
            )

            # Project DeePAW features (3200-dim) to hidden_dim
            self.deepaw_proj = nn.Linear(3200, hidden_dim)

            # If concat fusion, need additional projection
            if deepaw_fusion == "concat":
                self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Edge feature expansion
        self.rbf_expansion = EnhancedRBFExpansion(n_rbf=n_rbf, cutoff=cutoff)
        edge_dim = n_rbf

        # Angle feature expansion (optional)
        if use_angles:
            self.angle_expansion = AngleExpansion(n_angle_basis=n_angle_basis)
            # Angle features will be aggregated to edges
            # Add a projection layer to combine angle features with edge features
            self.angle_to_edge = nn.Linear(n_angle_basis, n_rbf)

        # Message passing layers
        self.layers = nn.ModuleList([
            EnhancedMessagePassingLayer(
                hidden_dim=hidden_dim,
                edge_dim=edge_dim,
                angle_dim=0,  # Angles aggregated to edges, not passed directly
                use_edge_update=use_edge_update,
                activation=activation,
            )
            for _ in range(n_layers)
        ])

        # Graph-level pooling
        self.pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU() if activation == "silu" else nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, batch_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of enhanced graph backbone.

        Args:
            batch_dict: Dictionary containing:
                - z: (num_nodes,) atomic numbers
                - edge_index: (2, num_edges) edge indices
                - edge_dist: (num_edges,) edge distances
                - batch: (num_nodes,) graph indices
                - pos (optional): (num_nodes, 3) atomic positions (required for DeePAW)
                - edge_angles (optional): (num_triplets,) angles
                - triplet_index (optional): (3, num_triplets) triplet indices

        Returns:
            (num_graphs, hidden_dim) graph-level embeddings
        """
        z = batch_dict["z"]
        edge_index = batch_dict["edge_index"]
        edge_dist = batch_dict["edge_dist"]
        batch = batch_dict["batch"]

        # Initialize node embeddings
        if self.use_deepaw_features and self.deepaw_fusion == "replace":
            # Replace mode: Use DeePAW features as the sole source of atom embeddings
            pos = batch_dict["pos"]

            # Extract DeePAW features (frozen)
            with torch.no_grad():
                deepaw_features = self.deepaw_extractor.extract_atom_features(
                    z, pos, edge_index
                )  # (num_nodes, 3200)

            # Project to hidden_dim (trainable)
            node_emb = self.deepaw_proj(deepaw_features)  # (num_nodes, hidden_dim)

        else:
            # Standard mode: Start with learnable atom embeddings
            node_emb = self.atom_emb(z)  # (num_nodes, hidden_dim)

            # Add/Concat DeePAW features if enabled
            if self.use_deepaw_features:
                # Extract DeePAW features using pre-built graph structure
                pos = batch_dict["pos"]

                # Extract features for all nodes at once (no need to split by graph)
                with torch.no_grad():
                    deepaw_features = self.deepaw_extractor.extract_atom_features(
                        z, pos, edge_index
                    )  # (num_nodes, 3200)

                # Project to hidden_dim
                deepaw_proj = self.deepaw_proj(deepaw_features)  # (num_nodes, hidden_dim)

                # Fuse with atom embeddings
                if self.deepaw_fusion == "add":
                    node_emb = node_emb + deepaw_proj
                elif self.deepaw_fusion == "concat":
                    node_emb = torch.cat([node_emb, deepaw_proj], dim=-1)
                    node_emb = self.fusion_proj(node_emb)  # (num_nodes, hidden_dim)

        # Expand edge distances to features
        edge_feat = self.rbf_expansion(edge_dist)  # (num_edges, n_rbf)

        # Aggregate angle features to edges if available
        if self.use_angles and "edge_angles" in batch_dict and "triplet_to_edge" in batch_dict:
            edge_angles = batch_dict["edge_angles"]  # (num_triplets,)
            triplet_to_edge = batch_dict["triplet_to_edge"]  # (num_triplets, 2) - indices of two edges per triplet

            # Expand angles to features
            angle_feat = self.angle_expansion(edge_angles)  # (num_triplets, n_angle_basis)

            # Project angle features to edge feature space
            angle_edge_feat = self.angle_to_edge(angle_feat)  # (num_triplets, n_rbf)

            # Aggregate angle features to edges
            # Each triplet (i-j-k) contributes to two edges: (i,j) and (k,j)
            edge_angle_agg = torch.zeros_like(edge_feat)  # (num_edges, n_rbf)
            edge_angle_count = torch.zeros(edge_feat.size(0), dtype=torch.float32, device=edge_feat.device)

            # Add angle contributions from both edges in each triplet
            for edge_idx_col in range(2):
                edge_indices = triplet_to_edge[:, edge_idx_col]  # (num_triplets,)
                valid_mask = edge_indices >= 0  # Filter out invalid indices
                if valid_mask.any():
                    edge_angle_agg.index_add_(0, edge_indices[valid_mask], angle_edge_feat[valid_mask])
                    edge_angle_count.index_add_(0, edge_indices[valid_mask],
                                               torch.ones(valid_mask.sum(), dtype=torch.float32, device=edge_feat.device))

            # Average angle features per edge (normalize by count)
            edge_angle_count = edge_angle_count.clamp_min(1.0).unsqueeze(-1)
            edge_angle_agg = edge_angle_agg / edge_angle_count

            # Combine distance and angle features
            edge_feat = edge_feat + edge_angle_agg

        # Message passing
        for layer in self.layers:
            node_emb, edge_feat = layer(node_emb, edge_index, edge_feat)

        # Graph-level pooling (mean pooling)
        num_graphs = batch.max().item() + 1
        graph_emb = torch.zeros(
            num_graphs, self.hidden_dim,
            dtype=node_emb.dtype, device=node_emb.device
        )

        # Sum pooling with index_add_
        graph_emb.index_add_(0, batch, node_emb)

        # Divide by number of nodes per graph for mean pooling
        counts = torch.zeros(num_graphs, dtype=torch.float32, device=batch.device)
        counts.index_add_(0, batch, torch.ones_like(batch, dtype=torch.float32))
        graph_emb = graph_emb / counts.unsqueeze(-1)

        # Final MLP
        graph_emb = self.pool(graph_emb)

        return graph_emb

"""Backbone encoders for multitask crystal-property prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class RBFExpansion(nn.Module):
    """Radial basis expansion for distances."""

    def __init__(self, n_rbf: int = 64, cutoff: float = 6.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, n_rbf)
        gamma = 10.0 / cutoff
        self.register_buffer("centers", centers)
        self.gamma = gamma

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        diff = dist.unsqueeze(-1) - self.centers.unsqueeze(0)
        return torch.exp(-self.gamma * diff * diff)


def graph_mean_pool(x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Mean pool node embeddings to graph embeddings."""
    pooled = torch.zeros(num_graphs, x.size(-1), device=x.device, dtype=x.dtype)
    pooled.index_add_(0, batch, x)
    counts = torch.bincount(batch, minlength=num_graphs).clamp_min(1).unsqueeze(-1)
    return pooled / counts


class CompositionBackbone(nn.Module):
    """Simple composition-only encoder."""

    def __init__(self, hidden_dim: int = 256, max_atomic_number: int = 100):
        super().__init__()
        self.atom_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, batch_dict: dict) -> torch.Tensor:
        z = batch_dict["z"]
        batch = batch_dict["batch"]
        num_graphs = int(batch.max().item()) + 1
        node_emb = self.atom_embedding(z)
        pooled = graph_mean_pool(node_emb, batch, num_graphs)
        return self.norm(pooled)


class MessagePassingLayer(nn.Module):
    """Distance-aware message passing block."""

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]

        src_h = node_emb[src]
        dst_h = node_emb[dst]
        msg_input = torch.cat([src_h, dst_h, edge_feat], dim=-1)
        messages = self.msg_mlp(msg_input)

        # Fix AMP type mismatch: ensure agg has same dtype as messages
        agg = torch.zeros_like(node_emb, dtype=messages.dtype)
        agg.index_add_(0, dst, messages)

        update = self.upd_mlp(torch.cat([node_emb, agg], dim=-1))
        return self.norm(node_emb + update)


class GraphBackbone(nn.Module):
    """Lightweight graph backbone with distance-aware message passing."""

    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 6,
        n_rbf: int = 64,
        cutoff: float = 6.0,
        max_atomic_number: int = 100,
    ):
        super().__init__()
        self.atom_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim)
        self.rbf = RBFExpansion(n_rbf=n_rbf, cutoff=cutoff)
        self.edge_proj = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [MessagePassingLayer(hidden_dim=hidden_dim, edge_dim=hidden_dim) for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, batch_dict: dict) -> torch.Tensor:
        z = batch_dict["z"]
        edge_index = batch_dict["edge_index"]
        edge_dist = batch_dict["edge_dist"]
        batch = batch_dict["batch"]
        num_graphs = int(batch.max().item()) + 1

        node_emb = self.atom_embedding(z)
        edge_feat = self.edge_proj(self.rbf(edge_dist))

        for layer in self.layers:
            node_emb = layer(node_emb, edge_index, edge_feat)

        pooled = graph_mean_pool(node_emb, batch, num_graphs)
        return self.out_norm(pooled)

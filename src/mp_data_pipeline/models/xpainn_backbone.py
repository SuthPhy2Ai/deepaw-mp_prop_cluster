"""
XPaiNN backbone adapted from XequiNet for Materials Project pipeline.
Implements E(3)-equivariant message passing with e3nn.
"""
import math
from typing import Union, Iterable, Tuple

import torch
import torch.nn as nn
from e3nn import o3
from e3nn import nn as e3nn_nn
from torch_geometric.utils import scatter


# ============================================================================
# RBF and Cutoff Functions
# ============================================================================

class CosineCutoff(nn.Module):
    """Cosine cutoff function."""
    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.cos(math.pi * dist / self.cutoff) + 1.0)


class SphericalBesselj0(nn.Module):
    """Spherical Bessel function of the first kind."""
    def __init__(self, num_basis: int, cutoff: float):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        freq = math.pi * torch.arange(1, num_basis + 1) / cutoff
        self.register_buffer('freq', freq.view(1, -1))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        coeff = math.sqrt(2 / self.cutoff)
        rbf = torch.where(
            dist == 0,
            torch.ones_like(dist),
            torch.sin(self.freq * dist) / dist
        )
        return coeff * rbf


# ============================================================================
# E3NN Helper Layers
# ============================================================================

class Invariant(nn.Module):
    """Extract invariant (norm) from equivariant features."""
    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps, Iterable],
        squared: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.squared = squared
        self.eps = eps
        self.invariant = o3.Norm(irreps_in, squared=squared)

    def forward(self, x):
        if self.squared:
            x = self.invariant(x)
        else:
            x = self.invariant(x + self.eps ** 2) - self.eps
        return x


class EquivariantDot(nn.Module):
    """Equivariant dot product (inner product preserving equivariance)."""
    def __init__(self, irreps_in: Union[str, o3.Irreps, Iterable]):
        super().__init__()
        irreps_in = o3.Irreps(irreps_in).simplify()
        irreps_out = o3.Irreps([(mul, "0e") for mul, _ in irreps_in])

        instr = [(i, i, i, "uuu", False, ir.dim) for i, (mul, ir) in enumerate(irreps_in)]

        self.tp = o3.TensorProduct(
            irreps_in, irreps_in, irreps_out, instr,
            irrep_normalization="component"
        )

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out.simplify()
        self.input_dim = self.irreps_in.dim

    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        return self.tp(features1, features2)


# ============================================================================
# XPaiNN Components
# ============================================================================

class XEmbedding(nn.Module):
    """Embedding layer with spherical harmonics."""
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1o + 32x2e",
        num_basis: int = 20,
        cutoff: float = 6.0,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.edge_num_irreps = self.edge_irreps.num_irreps

        # One-hot embedding (120 elements, padding_idx=0)
        self.node_embed = nn.Embedding(120, self.node_dim, padding_idx=0)

        # Spherical harmonics for angular features
        self.sph_harm = o3.SphericalHarmonics(
            self.edge_irreps, normalize=True, normalization="component"
        )

        # Radial basis functions
        self.rbf = SphericalBesselj0(num_basis, cutoff)
        self.cutoff_fn = CosineCutoff(cutoff)

    def forward(
        self,
        at_no: torch.LongTensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            at_no: Atomic numbers (N_atoms,)
            pos: Atomic positions (N_atoms, 3)
            edge_index: Edge indices (2, N_edges)

        Returns:
            x_scalar: Scalar node features (N_atoms, node_dim)
            rbf: Radial basis functions (N_edges, num_basis)
            fcut: Cutoff values (N_edges, 1)
            rsh: Spherical harmonics (N_edges, edge_irreps.dim)
        """
        # Node embedding
        x_scalar = self.node_embed(at_no)

        # Edge vectors and distances
        vec = pos[edge_index[0]] - pos[edge_index[1]]
        dist = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)

        # Radial basis and cutoff
        rbf = self.rbf(dist)
        fcut = self.cutoff_fn(dist)

        # Spherical harmonics (e3nn expects [y, z, x] order)
        rsh = self.sph_harm(vec[:, [1, 2, 0]])

        return x_scalar, rbf, fcut, rsh


class XPaiNNMessage(nn.Module):
    """Message passing for XPaiNN."""
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1o + 32x2e",
        num_basis: int = 20,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.edge_num_irreps = self.edge_irreps.num_irreps
        self.hidden_dim = self.node_dim + self.edge_num_irreps * 2
        self.num_basis = num_basis

        # Scalar MLP
        self.scalar_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.node_dim),
            nn.SiLU(),
            nn.Linear(self.node_dim, self.hidden_dim),
        )
        nn.init.zeros_(self.scalar_mlp[0].bias)
        nn.init.zeros_(self.scalar_mlp[2].bias)

        # RBF projection
        self.rbf_lin = nn.Linear(self.num_basis, self.hidden_dim, bias=True)
        nn.init.zeros_(self.rbf_lin.bias)

        # Elementwise tensor product for spherical features
        self.rsh_conv = o3.ElementwiseTensorProduct(
            self.edge_irreps, f"{self.edge_num_irreps}x0e"
        )

        # Layer normalization
        self.norm = nn.LayerNorm(self.node_dim)
        self.o3norm = e3nn_nn.BatchNorm(self.edge_irreps)

    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
        rbf: torch.Tensor,
        fcut: torch.Tensor,
        rsh: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_scalar: Scalar features (N_atoms, node_dim)
            x_spherical: Spherical features (N_atoms, edge_irreps.dim)
            rbf: Radial basis (N_edges, num_basis)
            fcut: Cutoff (N_edges, 1)
            rsh: Spherical harmonics (N_edges, edge_irreps.dim)
            edge_index: Edge indices (2, N_edges)

        Returns:
            new_scalar: Updated scalar features
            new_spherical: Updated spherical features
        """
        scalar_in = self.norm(x_scalar)
        spherical_in = self.o3norm(x_spherical)

        # Scalar message
        scalar_out = self.scalar_mlp(scalar_in)

        # Filter with RBF and cutoff
        filter_weight = self.rbf_lin(rbf) * fcut
        filter_out = scalar_out[edge_index[1]] * filter_weight

        # Split into gates and message
        gate_state_spherical, gate_edge_spherical, message_scalar = torch.split(
            filter_out,
            [self.edge_num_irreps, self.edge_num_irreps, self.node_dim],
            dim=-1,
        )

        # Spherical message
        message_spherical = self.rsh_conv(spherical_in[edge_index[1]], gate_state_spherical)
        edge_spherical = self.rsh_conv(rsh, gate_edge_spherical)
        message_spherical = message_spherical + edge_spherical

        # Aggregate messages (ensure dtype compatibility for AMP)
        new_scalar = x_scalar.index_add(0, edge_index[0], message_scalar.to(x_scalar.dtype))
        new_spherical = x_spherical.index_add(0, edge_index[0], message_spherical.to(x_spherical.dtype))

        return new_scalar, new_spherical


class XPaiNNUpdate(nn.Module):
    """Update function for XPaiNN."""
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1o + 32x2e",
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.edge_num_irreps = self.edge_irreps.num_irreps
        self.hidden_dim = self.node_dim * 2 + self.edge_num_irreps

        # Spherical transformations
        self.update_U = o3.Linear(self.edge_irreps, self.edge_irreps, biases=True)
        self.update_V = o3.Linear(self.edge_irreps, self.edge_irreps, biases=True)
        self.invariant = Invariant(self.edge_irreps)
        self.equidot = EquivariantDot(self.edge_irreps)
        self.dot_lin = nn.Linear(self.edge_num_irreps, self.node_dim, bias=False)
        self.rsh_conv = o3.ElementwiseTensorProduct(
            self.edge_irreps, f"{self.edge_num_irreps}x0e"
        )

        # Scalar MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(self.node_dim + self.edge_num_irreps, self.node_dim),
            nn.SiLU(),
            nn.Linear(self.node_dim, self.hidden_dim),
        )
        nn.init.zeros_(self.update_mlp[0].bias)
        nn.init.zeros_(self.update_mlp[2].bias)

        # Normalization
        self.norm = nn.LayerNorm(self.node_dim)
        self.o3norm = e3nn_nn.BatchNorm(self.edge_irreps)

    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_scalar: Scalar features (N_atoms, node_dim)
            x_spherical: Spherical features (N_atoms, edge_irreps.dim)

        Returns:
            new_scalar: Updated scalar features
            new_spherical: Updated spherical features
        """
        scalar_in = self.norm(x_scalar)
        spherical_in = self.o3norm(x_spherical)

        U_spherical = self.update_U(spherical_in)
        V_spherical = self.update_V(spherical_in)

        # Extract invariant from V
        V_invariant = self.invariant(V_spherical)
        mlp_in = torch.cat([scalar_in, V_invariant], dim=-1)
        mlp_out = self.update_mlp(mlp_in)

        # Split MLP output
        a_vv, a_sv, a_ss = torch.split(
            mlp_out,
            [self.edge_num_irreps, self.node_dim, self.node_dim],
            dim=-1
        )

        # Update spherical features
        d_spherical = self.rsh_conv(U_spherical, a_vv)

        # Update scalar features
        inner_prod = self.equidot(U_spherical, V_spherical)
        inner_prod = self.dot_lin(inner_prod)
        d_scalar = a_sv * inner_prod + a_ss

        return x_scalar + d_scalar, x_spherical + d_spherical


# ============================================================================
# XPaiNN Backbone
# ============================================================================

class XPaiNNBackbone(nn.Module):
    """
    E(3)-equivariant message passing network.
    Adapted from XequiNet's XPaiNN for Materials Project pipeline.

    Args:
        node_dim: Dimension of scalar node features
        edge_irreps: Irreducible representations for edge features
        num_interactions: Number of message passing layers
        num_rbf: Number of radial basis functions
        cutoff: Cutoff distance for neighbors
    """

    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: str = "128x0e + 64x1o + 32x2e",
        num_interactions: int = 3,
        num_rbf: int = 20,
        cutoff: float = 6.0,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.num_interactions = num_interactions

        # Embedding layer
        self.embedding = XEmbedding(
            node_dim=node_dim,
            edge_irreps=edge_irreps,
            num_basis=num_rbf,
            cutoff=cutoff,
        )

        # Message passing layers
        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            self.interactions.append(
                nn.ModuleDict({
                    'message': XPaiNNMessage(node_dim, edge_irreps, num_rbf),
                    'update_fn': XPaiNNUpdate(node_dim, edge_irreps),
                })
            )

        # Output projection (to match existing head interface)
        self.out_proj = nn.Linear(node_dim, node_dim)

    def forward(self, data):
        """
        Forward pass compatible with PyG Data objects or dict.

        Args:
            data: PyG Data object or dict with keys:
                - z or at_no: Atomic numbers (N_atoms,)
                - pos: Positions (N_atoms, 3)
                - edge_index: Edge indices (2, N_edges)
                - batch: Batch indices (N_atoms,) [optional]

        Returns:
            Pooled graph-level features (N_graphs, node_dim)
        """
        # Handle both dict and PyG Data object
        if isinstance(data, dict):
            at_no = data.get('z', data.get('at_no'))
            pos = data['pos']
            edge_index = data['edge_index']
            batch = data.get('batch', None)
        else:
            at_no = data.at_no if hasattr(data, 'at_no') else data.z
            pos = data.pos
            edge_index = data.edge_index
            batch = data.batch if hasattr(data, 'batch') else None

        # Embedding
        x_scalar, rbf, fcut, rsh = self.embedding(at_no, pos, edge_index)

        # Initialize spherical features
        x_spherical = torch.zeros(
            at_no.size(0),
            self.edge_irreps.dim,
            device=pos.device,
            dtype=pos.dtype
        )

        # Message passing
        for interaction in self.interactions:
            # Message
            x_scalar, x_spherical = interaction['message'](
                x_scalar, x_spherical, rbf, fcut, rsh, edge_index
            )
            # Update
            x_scalar, x_spherical = interaction['update_fn'](
                x_scalar, x_spherical
            )

        # Global pooling
        if batch is None:
            batch = torch.zeros(at_no.size(0), dtype=torch.long, device=pos.device)
        x_pooled = scatter(x_scalar, batch, dim=0, reduce='mean')

        return self.out_proj(x_pooled)

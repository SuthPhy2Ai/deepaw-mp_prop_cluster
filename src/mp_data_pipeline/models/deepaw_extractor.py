"""DeePAW Atom Feature Extractor.

This module provides functionality to extract atom-level features from the
pretrained DeePAW eSCN model for use in materials property prediction.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from ase import Atoms
from ase.neighborlist import neighbor_list

# Add DeePAW to path
DEEPAW_PATH = Path("/home/sutianhao/data/deepaw_test/DeePAW-main")
if str(DEEPAW_PATH) not in sys.path:
    sys.path.insert(0, str(DEEPAW_PATH))

# Workaround: Mock torch_cluster to avoid import errors
# We don't need torch_cluster since we use ASE for graph construction
import types
mock_torch_cluster = types.ModuleType('torch_cluster')
mock_torch_cluster.radius = lambda *args, **kwargs: None
sys.modules['torch_cluster'] = mock_torch_cluster

# Now import DeePAW modules
from deepaw.models.escn.f_nonlocal_escn import F_nonlocal_escn
from deepaw.config import get_model_config


class DeePAWAtomFeatureExtractor(nn.Module):
    """Extract atom-level features from pretrained DeePAW eSCN model.
    
    This extractor loads a pretrained F_nonlocal_escn model and uses only
    the atom_blocks (atom tower) to generate rich atom representations that
    encode electronic structure information learned from charge density prediction.
    
    Args:
        checkpoint_path: Path to DeePAW checkpoint (.pth file)
        device: Device to run on ("cuda" or "cpu")
        cutoff: Cutoff radius for graph construction (Angstroms)
        max_neighbors: Maximum number of neighbors per atom
        freeze: Whether to freeze model parameters
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        cutoff: float = 4.0,
        max_neighbors: int = 20,
        freeze: bool = True,
    ):
        super().__init__()
        self.device = device
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        
        # Load DeePAW model configuration
        config = get_model_config("f_nonlocal_escn")
        
        # Create model
        self.model = F_nonlocal_escn(**config).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint)
        
        # Set to eval mode
        self.model.eval()
        
        # Freeze parameters if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Extract key components
        self.atom_embedding = self.model.atom_embedding
        self.atom_blocks = self.model.atom_blocks
        self.distance_expansion = self.model.distance_expansion
        self.envelope = self.model.envelope
        self.mappingReduced = self.model.mappingReduced
        
        # Dimensions
        self.sphere_channels = self.model.sphere_channels  # 128
        self.num_sphere_basis = self.model.num_sphere_basis  # 25 = (lmax+1)^2
        self.lmax = self.model.lmax  # 4
        self.mmax = self.model.mmax  # 2
        
        # Output dimension: 25 * 128 = 3200
        self.output_dim = self.num_sphere_basis * self.sphere_channels
    
    def _build_atom_graph(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
    ) -> dict:
        """Build atom-atom graph using ASE neighbor_list.
        
        Args:
            atomic_numbers: (N,) atomic numbers
            positions: (N, 3) atomic positions in Cartesian coordinates
            cell: (3, 3) cell matrix
        
        Returns:
            Graph dictionary with atom edges and displacements
        """
        # Convert to numpy for ASE
        z_np = atomic_numbers.cpu().numpy()
        pos_np = positions.cpu().numpy()
        cell_np = cell.cpu().numpy()
        
        # Create ASE Atoms object
        atoms = Atoms(
            numbers=z_np,
            positions=pos_np,
            cell=cell_np,
            pbc=True,
        )
        
        # Build neighbor list
        src, dst, shifts = neighbor_list(
            "ijS",
            atoms,
            self.cutoff,
            self_interaction=False,
        )
        
        # Limit neighbors per atom (distance-sorted)
        if len(src) > 0:
            # Compute distances
            edge_vec = positions[dst] - (positions[src] + torch.tensor(
                shifts @ cell_np, dtype=positions.dtype, device=positions.device
            ))
            edge_dist = edge_vec.norm(dim=1)
            
            # Sort by distance and limit
            unique_dst = torch.unique(torch.tensor(dst, device=positions.device))
            keep_mask = torch.zeros(len(src), dtype=torch.bool, device=positions.device)
            
            for dst_idx in unique_dst:
                mask = torch.tensor(dst == dst_idx.item(), device=positions.device)
                indices = torch.where(mask)[0]
                distances = edge_dist[indices]
                
                # Keep closest max_neighbors
                if len(indices) > self.max_neighbors:
                    _, sorted_idx = torch.sort(distances)
                    keep_indices = indices[sorted_idx[:self.max_neighbors]]
                else:
                    keep_indices = indices
                
                keep_mask[keep_indices] = True
            
            # Filter edges
            src = src[keep_mask.cpu().numpy()]
            dst = dst[keep_mask.cpu().numpy()]
            shifts = shifts[keep_mask.cpu().numpy()]
        
        # Convert to tensors
        atom_edges = torch.tensor(
            [[s, d] for s, d in zip(src, dst)],
            dtype=torch.long,
            device=self.device,
        )
        
        atom_edges_displacement = torch.tensor(
            shifts,
            dtype=torch.float32,
            device=self.device,
        )
        
        return {
            "atom_edges": atom_edges,  # (num_edges, 2)
            "atom_edges_displacement": atom_edges_displacement,  # (num_edges, 3)
        }
    
    def _convert_to_deepaw_format(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
        atom_edges: torch.Tensor,
        atom_edges_displacement: torch.Tensor,
    ) -> dict:
        """Convert MP format to DeePAW padded batch format.
        
        Args:
            atomic_numbers: (N,) atomic numbers
            positions: (N, 3) positions
            cell: (3, 3) cell matrix
            atom_edges: (E, 2) edge indices
            atom_edges_displacement: (E, 3) edge displacements
        
        Returns:
            DeePAW input_dict with batch dimension
        """
        num_atoms = len(atomic_numbers)
        num_edges = len(atom_edges)
        
        # Add batch dimension (batch_size=1)
        input_dict = {
            "nodes": atomic_numbers.unsqueeze(0),  # (1, N)
            "atom_xyz": positions.unsqueeze(0),  # (1, N, 3)
            "atom_edges": atom_edges.unsqueeze(0),  # (1, E, 2)
            "atom_edges_displacement": atom_edges_displacement.unsqueeze(0),  # (1, E, 3)
            "cell": cell.unsqueeze(0),  # (1, 3, 3)
            "num_nodes": torch.tensor([num_atoms], device=self.device),
            "num_atom_edges": torch.tensor([num_edges], device=self.device),
        }
        
        return input_dict
    
    @torch.no_grad()
    def extract_atom_features(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Extract atom features from DeePAW model using pre-built graph.

        Args:
            atomic_numbers: (N,) atomic numbers (int64)
            positions: (N, 3) atomic positions in Cartesian coordinates (float32)
            edge_index: (2, E) edge indices [src, dst] from pre-built graph

        Returns:
            (N, 3200) atom features (flattened spherical harmonics representation)
        """
        # Ensure correct device
        atomic_numbers = atomic_numbers.to(self.device)
        positions = positions.to(self.device)
        edge_index = edge_index.to(self.device)

        # Use pre-built graph structure (no need to rebuild with cell)
        # Convert edge_index format: (2, E) -> (E, 2) for DeePAW
        atom_edges = edge_index.t()  # (E, 2)

        # No displacement needed since we're not using PBC
        atom_edges_displacement = torch.zeros(
            atom_edges.shape[0], 3,
            dtype=positions.dtype,
            device=self.device
        )

        # Dummy cell (not used in computation)
        cell = torch.eye(3, dtype=positions.dtype, device=self.device)

        # Convert to DeePAW format
        input_dict = self._convert_to_deepaw_format(
            atomic_numbers,
            positions,
            cell,
            atom_edges,
            atom_edges_displacement,
        )
        
        # Extract atom features (following F_nonlocal_escn forward logic)
        num_atoms = len(atomic_numbers)
        
        # 1. Atom embedding
        x = torch.zeros(
            num_atoms, self.num_sphere_basis, self.sphere_channels,
            dtype=positions.dtype, device=self.device,
        )
        x[:, 0, :] = self.atom_embedding(atomic_numbers)  # Place at L=0
        
        # 2. Compute edge data
        atom_edges = input_dict["atom_edges"].squeeze(0)  # (E, 2)
        atom_edges_disp = input_dict["atom_edges_displacement"].squeeze(0)  # (E, 3)
        
        # Compute edge vectors
        unitcell = cell
        displacement = torch.matmul(
            atom_edges_disp.unsqueeze(1), unitcell
        ).squeeze(1)  # (E, 3)
        
        neigh_pos = positions[atom_edges[:, 0]]
        neigh_abs_pos = neigh_pos + displacement
        this_pos = positions[atom_edges[:, 1]]
        edge_vec = this_pos - neigh_abs_pos  # (E, 3)
        
        edge_distance = edge_vec.norm(dim=1)
        
        # Distance embedding
        edge_embedding = self.distance_expansion(edge_distance)
        
        # Envelope
        edge_envelope = self.envelope(edge_distance / self.cutoff)
        edge_envelope = edge_envelope.unsqueeze(1).unsqueeze(2)  # (E, 1, 1)
        
        # Euler angles and Wigner-D matrices
        from deepaw.models.escn.rotation import init_edge_rot_euler_angles, eulers_to_wigner
        
        euler_angles = init_edge_rot_euler_angles(edge_vec)
        
        # Load Jd buffers
        Jd = [J.to(device=self.device, dtype=edge_vec.dtype) for J in self.model.Jd_buffers]
        
        wigner = eulers_to_wigner(euler_angles, 0, self.lmax, Jd)
        wigner_inv = wigner.transpose(1, 2).contiguous()
        
        # Select subset for mmax
        coeff_idx = self.mappingReduced.coefficient_idx(self.lmax, self.mmax)
        if self.mmax != self.lmax:
            wigner = wigner.index_select(1, coeff_idx)
            wigner_inv = wigner_inv.index_select(2, coeff_idx)
        
        # Combine with coefficient mapping
        to_m = self.mappingReduced.to_m.to(wigner.dtype)
        wigner_and_M_mapping = torch.einsum("mk,nkj->nmj", to_m, wigner)
        wigner_and_M_mapping_inv = torch.einsum("njk,mk->njm", wigner_inv, to_m)
        
        # Edge index
        edge_index = torch.stack([atom_edges[:, 0], atom_edges[:, 1]], dim=0)
        
        # 3. Message passing through atom_blocks
        for block in self.atom_blocks:
            x = block(
                x,
                edge_embedding,
                edge_distance,
                edge_index,
                wigner_and_M_mapping,
                wigner_and_M_mapping_inv,
                edge_envelope,
            )
        
        # 4. Flatten spherical harmonics representation
        # x: (N, 25, 128) → (N, 3200)
        atom_features = x.reshape(num_atoms, -1)
        
        return atom_features
    
    def forward(self, batch_dict: dict) -> torch.Tensor:
        """Forward pass for batch processing.
        
        Args:
            batch_dict: Dictionary containing:
                - z: (total_nodes,) atomic numbers
                - pos: (total_nodes, 3) positions
                - batch: (total_nodes,) graph indices
                - cell: (batch_size, 3, 3) cell matrices
        
        Returns:
            (total_nodes, 3200) atom features for all atoms in batch
        """
        z = batch_dict["z"]
        pos = batch_dict["pos"]
        batch = batch_dict["batch"]
        cells = batch_dict["cell"]
        
        # Process each graph separately
        features_list = []
        num_graphs = int(batch.max().item()) + 1
        
        for graph_idx in range(num_graphs):
            # Extract atoms for this graph
            mask = batch == graph_idx
            graph_z = z[mask]
            graph_pos = pos[mask]
            graph_cell = cells[graph_idx]
            
            # Extract features
            graph_features = self.extract_atom_features(
                graph_z, graph_pos, graph_cell
            )
            features_list.append(graph_features)
        
        # Concatenate all features
        all_features = torch.cat(features_list, dim=0)
        
        return all_features


def test_extractor():
    """Test the DeePAW feature extractor."""
    print("Testing DeePAW Atom Feature Extractor...")
    
    # Create extractor
    extractor = DeePAWAtomFeatureExtractor(
        checkpoint_path="/home/sutianhao/data/deepaw_test/DeePAW-main/checkpoints/f_nonlocal_escn_best.pth",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    print(f"✓ Extractor created")
    print(f"  Output dimension: {extractor.output_dim}")
    print(f"  Device: {extractor.device}")
    
    # Test with simple structure (Si2O2)
    z = torch.tensor([14, 14, 8, 8], dtype=torch.long)
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [2.5, 2.5, 0.0],
        [1.25, 1.25, 1.5],
        [3.75, 3.75, 1.5],
    ], dtype=torch.float32)
    cell = torch.eye(3, dtype=torch.float32) * 5.0
    
    # Extract features
    features = extractor.extract_atom_features(z, pos, cell)
    
    print(f"✓ Features extracted")
    print(f"  Shape: {features.shape}")
    print(f"  Mean: {features.mean().item():.4f}")
    print(f"  Std: {features.std().item():.4f}")
    
    assert features.shape == (4, 3200), f"Expected (4, 3200), got {features.shape}"
    print("✓ All tests passed!")


if __name__ == "__main__":
    test_extractor()

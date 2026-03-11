"""
Best-K checkpoint manager with heap-based tracking.
Adapted from XequiNet's training utilities.
"""
import heapq
import os
from pathlib import Path
from typing import Optional, Tuple, List
import torch


class BestKCheckpoints:
    """
    Manages top-K model checkpoints based on validation metric.

    Uses a max-heap to track best checkpoints (for minimization).
    Automatically deletes worst checkpoint when exceeding K.

    Args:
        save_dir: Directory to save checkpoints
        k: Number of best checkpoints to keep
        mode: 'min' for minimization, 'max' for maximization
        prefix: Checkpoint filename prefix (default: 'best')
    """

    def __init__(
        self,
        save_dir: str,
        k: int = 3,
        mode: str = 'min',
        prefix: str = 'best'
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.k = k
        self.mode = mode
        self.prefix = prefix

        # Max-heap for 'min' mode (store negative values)
        # Min-heap for 'max' mode (store positive values)
        self.heap: List[Tuple[float, str]] = []

    def should_save(self, metric: float) -> bool:
        """Check if current metric qualifies for top-K."""
        if len(self.heap) < self.k:
            return True

        # Compare with worst in heap
        worst_metric, _ = self.heap[0]
        if self.mode == 'min':
            return metric < -worst_metric  # heap stores negative values
        else:
            return metric > worst_metric

    def save(
        self,
        metric: float,
        checkpoint: dict,
        epoch: int,
        extra_info: Optional[str] = None
    ) -> Optional[str]:
        """
        Save checkpoint if it's in top-K.

        Args:
            metric: Validation metric value
            checkpoint: Checkpoint dict to save
            epoch: Current epoch
            extra_info: Optional info to append to filename

        Returns:
            Path to saved checkpoint, or None if not saved
        """
        if not self.should_save(metric):
            return None

        # Generate filename
        if extra_info:
            filename = f"{self.prefix}_epoch{epoch}_{extra_info}.pt"
        else:
            filename = f"{self.prefix}_epoch{epoch}_metric{metric:.4f}.pt"

        filepath = self.save_dir / filename

        # Save checkpoint
        torch.save(checkpoint, filepath)

        # Add to heap
        heap_metric = -metric if self.mode == 'min' else metric
        heapq.heappush(self.heap, (heap_metric, str(filepath)))

        # Remove worst if exceeding K
        if len(self.heap) > self.k:
            worst_metric, worst_path = heapq.heappop(self.heap)
            if os.path.exists(worst_path):
                os.remove(worst_path)
                print(f"Removed checkpoint: {worst_path}")

        return str(filepath)

    def get_best_checkpoint(self) -> Optional[str]:
        """Return path to best checkpoint."""
        if not self.heap:
            return None

        if self.mode == 'min':
            # Best is the one with most negative value (smallest actual metric)
            best_metric, best_path = min(self.heap, key=lambda x: x[0])
        else:
            # Best is the one with largest value
            best_metric, best_path = max(self.heap, key=lambda x: x[0])

        return best_path

    def get_all_checkpoints(self) -> List[Tuple[float, str]]:
        """Return all checkpoints sorted by metric (best first)."""
        if self.mode == 'min':
            # Convert back to actual metrics and sort ascending
            checkpoints = [(-metric, path) for metric, path in self.heap]
            checkpoints.sort(key=lambda x: x[0])
        else:
            # Sort descending
            checkpoints = sorted(self.heap, key=lambda x: x[0], reverse=True)

        return checkpoints

    def state_dict(self):
        """Return state for persistence."""
        return {
            'k': self.k,
            'mode': self.mode,
            'prefix': self.prefix,
            'heap': self.heap
        }

    def load_state_dict(self, state_dict):
        """Load state from dict."""
        self.k = state_dict['k']
        self.mode = state_dict['mode']
        self.prefix = state_dict['prefix']
        self.heap = state_dict['heap']

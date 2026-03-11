"""
Exponential Moving Average (EMA) for model parameters.
Adapted from XequiNet's training utilities.
"""
import torch
import torch.nn as nn
from typing import Optional
from copy import deepcopy


class EMAModel:
    """
    Exponential Moving Average of model parameters.

    Maintains shadow parameters that are updated as:
        θ_ema = decay * θ_ema + (1 - decay) * θ

    Args:
        model: PyTorch model to track
        decay: EMA decay rate (default: 0.999)
        device: Device to store shadow parameters (default: same as model)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.decay = decay
        self.device = device or next(model.parameters()).device

        # Create shadow parameters
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self):
        """Update EMA parameters after optimizer step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    self.decay * self.shadow[name] +
                    (1.0 - self.decay) * param.data
                )
                self.shadow[name].copy_(new_average)

    @torch.no_grad()
    def apply_shadow(self):
        """
        Apply EMA parameters to model (for validation/testing).
        Backs up current parameters first.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self):
        """Restore original parameters after validation/testing."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        """Return EMA state for checkpointing."""
        return {
            'decay': self.decay,
            'shadow': self.shadow
        }

    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']

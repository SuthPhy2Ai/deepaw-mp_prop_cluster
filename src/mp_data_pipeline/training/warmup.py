"""
Learning rate warmup schedulers.
Adapted from XequiNet's training utilities.
"""
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupScheduler(_LRScheduler):
    """
    Learning rate warmup wrapper for any base scheduler.

    Linearly increases learning rate from 0 to base_lr over warmup_epochs,
    then switches to the base scheduler.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs for warmup
        base_scheduler: Base scheduler to use after warmup (optional)
        last_epoch: Last epoch index (for resuming)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        base_scheduler: _LRScheduler = None,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.finished_warmup = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Use base scheduler
            if self.base_scheduler is not None:
                if not self.finished_warmup:
                    # Initialize base scheduler at end of warmup
                    self.base_scheduler.last_epoch = self.last_epoch - self.warmup_epochs
                    self.finished_warmup = True
                return self.base_scheduler.get_last_lr()
            else:
                return self.base_lrs

    def step(self, epoch=None):
        if self.last_epoch < self.warmup_epochs:
            # During warmup
            super().step(epoch)
        else:
            # After warmup, step base scheduler
            if self.base_scheduler is not None:
                self.base_scheduler.step(epoch)
                self.last_epoch = self.warmup_epochs + self.base_scheduler.last_epoch
            else:
                super().step(epoch)


class LinearWarmup:
    """
    Simple linear warmup for learning rate.

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of steps for warmup
        start_lr: Starting learning rate (default: 0)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        start_lr: float = 0.0
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.current_step = 0

        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        """Update learning rate for current step."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.current_step / self.warmup_steps
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = self.start_lr + (self.base_lrs[i] - self.start_lr) * warmup_factor
                param_group['lr'] = lr

        self.current_step += 1

    def state_dict(self):
        """Return state for checkpointing."""
        return {
            'warmup_steps': self.warmup_steps,
            'start_lr': self.start_lr,
            'current_step': self.current_step,
            'base_lrs': self.base_lrs
        }

    def load_state_dict(self, state_dict):
        """Load state from checkpoint."""
        self.warmup_steps = state_dict['warmup_steps']
        self.start_lr = state_dict['start_lr']
        self.current_step = state_dict['current_step']
        self.base_lrs = state_dict['base_lrs']

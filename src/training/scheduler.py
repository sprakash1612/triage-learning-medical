"""
Learning rate schedulers and warmup strategies.

Implements various learning rate scheduling strategies for training optimization.
"""

from typing import Dict, List, Optional, Union
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingLR, OneCycleLR
import math
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    num_epochs: int = 100,
    num_steps_per_epoch: int = None,
    **kwargs,
):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: One of 'step', 'cosine', 'onecycle', 'polynomial', 'linear'
        num_epochs: Total number of training epochs
        num_steps_per_epoch: Number of steps per epoch (for step-based schedulers)
        **kwargs: Scheduler-specific arguments
    
    Returns:
        Learning rate scheduler object
    
    Raises:
        ValueError: If scheduler_type is not supported
    
    Example:
        >>> scheduler = create_scheduler(
        ...     optimizer, 'cosine', num_epochs=100, num_steps_per_epoch=100
        ... )
    """
    logger.info(
        f"Creating scheduler: type={scheduler_type}, num_epochs={num_epochs}"
    )
    
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == "step":
        step_size = kwargs.get('step_size', num_epochs // 3)
        gamma = kwargs.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == "cosine":
        T_max = kwargs.get('T_max', num_epochs)
        eta_min = kwargs.get('eta_min', 0.0)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    elif scheduler_type == "onecycle":
        if num_steps_per_epoch is None:
            raise ValueError("num_steps_per_epoch required for onecycle scheduler")
        
        max_lr = kwargs.get('max_lr', 0.1)
        total_steps = num_epochs * num_steps_per_epoch
        
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=kwargs.get('pct_start', 0.3),
            anneal_strategy=kwargs.get('anneal_strategy', 'cos'),
            cycle_momentum=kwargs.get('cycle_momentum', True),
        )
    
    elif scheduler_type == "polynomial":
        power = kwargs.get('power', 1.0)
        
        def lr_lambda(epoch):
            return (1 - epoch / num_epochs) ** power
        
        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    elif scheduler_type == "linear":
        def lr_lambda(epoch):
            return 1 - epoch / num_epochs
        
        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    elif scheduler_type == "exp":
        decay_rate = kwargs.get('decay_rate', 0.95)
        
        def lr_lambda(epoch):
            return decay_rate ** epoch
        
        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    else:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. "
            f"Supported: ['step', 'cosine', 'onecycle', 'polynomial', 'linear', 'exp']"
        )


class WarmupScheduler:
    """
    Wrapper to add warmup to any learning rate scheduler.
    
    Gradually increases learning rate from 0 to initial_lr over warmup_epochs.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        base_scheduler,
        warmup_epochs: int = 5,
        warmup_type: str = "linear",
    ):
        """
        Initialize WarmupScheduler.
        
        Args:
            optimizer: PyTorch optimizer
            base_scheduler: Base scheduler to wrap
            warmup_epochs: Number of warmup epochs
            warmup_type: One of 'linear', 'cos'
        """
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_type = warmup_type
        self.current_epoch = 0
    
    def step(self, epoch: int = None):
        """
        Step the scheduler.
        
        Args:
            epoch: Current epoch (0-indexed)
        """
        if epoch is None:
            epoch = self.current_epoch
        
        if epoch < self.warmup_epochs:
            # Warmup phase
            if self.warmup_type == "linear":
                warmup_progress = epoch / self.warmup_epochs
            elif self.warmup_type == "cos":
                warmup_progress = 0.5 * (1 - math.cos(math.pi * epoch / self.warmup_epochs))
            else:
                warmup_progress = epoch / self.warmup_epochs
            
            # Get base learning rates
            base_lrs = self.base_scheduler.base_lrs if hasattr(self.base_scheduler, 'base_lrs') else [group['lr'] for group in self.optimizer.param_groups]
            
            # Set learning rates with warmup
            for param_group, base_lr in zip(self.optimizer.param_groups, base_lrs):
                param_group['lr'] = warmup_progress * base_lr
        else:
            # Post-warmup phase: use base scheduler
            if hasattr(self.base_scheduler, 'step'):
                self.base_scheduler.step(epoch - self.warmup_epochs)
        
        self.current_epoch = epoch + 1
    
    def get_last_lr(self):
        """Get last learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]


def plot_lr_schedule(
    optimizer: Optimizer,
    scheduler,
    num_epochs: int = 100,
    num_steps_per_epoch: int = 1,
    save_path: Optional[str] = None,
):
    """
    Plot learning rate schedule.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to plot
        num_steps_per_epoch: Number of steps per epoch
        save_path: Path to save figure (optional)
    
    Returns:
        Matplotlib figure object
    """
    lrs = []
    
    for epoch in range(num_epochs):
        for step in range(num_steps_per_epoch):
            lrs.append(optimizer.param_groups[0]['lr'])
            
            # Step the scheduler (handles both epoch-based and step-based)
            if hasattr(scheduler, 'step'):
                if isinstance(scheduler, OneCycleLR):
                    scheduler.step()
                elif step == num_steps_per_epoch - 1:
                    scheduler.step()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(lrs, linewidth=2)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"LR schedule plot saved to {save_path}")
    
    return fig


def get_lr_lambda(
    schedule_type: str = "cosine",
    num_epochs: int = 100,
    **kwargs,
):
    """
    Get learning rate lambda function for LambdaLR scheduler.
    
    Args:
        schedule_type: Type of schedule
        num_epochs: Total number of epochs
        **kwargs: Schedule-specific arguments
    
    Returns:
        Lambda function that takes epoch and returns lr multiplier
    """
    if schedule_type == "cosine":
        def lr_lambda(epoch):
            return 0.5 * (1 + math.cos(math.pi * epoch / num_epochs))
    
    elif schedule_type == "linear":
        def lr_lambda(epoch):
            return 1 - epoch / num_epochs
    
    elif schedule_type == "polynomial":
        power = kwargs.get('power', 1.0)
        def lr_lambda(epoch):
            return (1 - epoch / num_epochs) ** power
    
    elif schedule_type == "step":
        steps = kwargs.get('steps', [30, 60, 90])
        gamma = kwargs.get('gamma', 0.1)
        
        def lr_lambda(epoch):
            lr = 1.0
            for step in steps:
                if epoch >= step:
                    lr *= gamma
            return lr
    
    else:
        def lr_lambda(epoch):
            return 1.0
    
    return lr_lambda

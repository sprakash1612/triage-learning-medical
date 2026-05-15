"""
Optimizer configurations and factory functions.

Supports multiple optimizer types with layer-wise learning rates and special
techniques like Sharpness Aware Minimization (SAM).
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    nesterov: bool = True,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Create optimizer with specified type and parameters.
    
    Args:
        model: PyTorch model to optimize
        optimizer_type: One of 'adam', 'adamw', 'sgd', 'rmsprop', 'radam'
        learning_rate: Learning rate
        weight_decay: L2 regularization coefficient
        momentum: Momentum for SGD
        nesterov: Whether to use Nesterov momentum
        **kwargs: Additional optimizer-specific arguments
    
    Returns:
        Initialized optimizer
    
    Raises:
        ValueError: If optimizer_type is not supported
    
    Example:
        >>> optimizer = create_optimizer(model, 'adamw', learning_rate=1e-3)
    """
    logger.info(
        f"Creating optimizer: type={optimizer_type}, lr={learning_rate}, "
        f"weight_decay={weight_decay}"
    )
    
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )
    
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )
    
    elif optimizer_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            **kwargs,
        )
    
    elif optimizer_type == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            **kwargs,
        )
    
    elif optimizer_type == "radam":
        try:
            from torch.optim.lr_scheduler import LambdaLR
            from torch.optim import Optimizer
            # RAdam implementation or use torch's if available
            return torch.optim.RAdam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **kwargs,
            ) if hasattr(torch.optim, 'RAdam') else torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        except ImportError:
            logger.warning("RAdam not available, using AdamW instead")
            return torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
    
    else:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. "
            f"Supported: ['adam', 'adamw', 'sgd', 'rmsprop', 'radam']"
        )


def create_param_groups(
    model: nn.Module,
    learning_rate: float,
    layer_lr_decay: float = 1.0,
    separate_backbone: bool = False,
) -> List[Dict]:
    """
    Create parameter groups with different learning rates.
    
    Useful for transfer learning where backbone and head have different learning rates.
    Also supports layer-wise learning rate decay for transformers.
    
    Args:
        model: PyTorch model
        learning_rate: Base learning rate
        layer_lr_decay: Decay factor for earlier layers (1.0 = no decay)
        separate_backbone: Whether to use different LR for backbone and classifier
    
    Returns:
        List of parameter group dictionaries
    
    Example:
        >>> param_groups = create_param_groups(
        ...     model, 1e-3, layer_lr_decay=0.9, separate_backbone=True
        ... )
        >>> optimizer = torch.optim.AdamW(param_groups)
    """
    param_groups = []
    
    if separate_backbone:
        # Find backbone and classifier parameters
        backbone_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if 'classifier' in name or 'head' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': learning_rate * 0.1,  # Lower LR for backbone
                'weight_decay': 1e-4,
            })
        
        if classifier_params:
            param_groups.append({
                'params': classifier_params,
                'lr': learning_rate,
                'weight_decay': 1e-4,
            })
        
        logger.info(
            f"Created param groups with separate backbone and classifier LRs"
        )
    
    elif layer_lr_decay < 1.0:
        # Layer-wise learning rate decay
        named_parameters = list(model.named_parameters())
        
        # Group parameters by layer depth
        for depth_idx in range(len(named_parameters)):
            param_name, param = named_parameters[depth_idx]
            
            # Calculate decay based on layer position
            layer_depth = len(named_parameters) - depth_idx
            lr = learning_rate * (layer_lr_decay ** (layer_depth / 10))
            
            param_groups.append({
                'params': [param],
                'lr': lr,
            })
        
        logger.info(f"Created {len(param_groups)} param groups with layer-wise LR decay")
    
    else:
        # Single parameter group
        param_groups.append({
            'params': model.parameters(),
            'lr': learning_rate,
        })
    
    return param_groups


class SAM(torch.optim.Optimizer):
    """
    Sharpness Aware Minimization (SAM) optimizer wrapper.
    
    Seeks parameters in flat minima which tend to generalize better.
    
    Reference:
        Foret et al. (2020). Sharpness Aware Minimization for Improved 
        Training Generalization. NeurIPS.
    """
    
    def __init__(
        self,
        params,
        base_optimizer: torch.optim.Optimizer = torch.optim.SGD,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs,
    ):
        """
        Initialize SAM wrapper.
        
        Args:
            params: Model parameters
            base_optimizer: Base optimizer class (default: SGD)
            rho: Neighborhood radius
            adaptive: Whether to use adaptive rho per layer
            **kwargs: Arguments for base optimizer
        """
        self.rho = rho
        self.adaptive = adaptive
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        
        self.base_optimizer = base_optimizer
        self.param_groups = [
            {**group, 'optimizer': base_optimizer([group['params']], **kwargs)}
            for group in self.param_groups
        ]
    
    def step(self, closure=None):
        """
        Perform SAM step.
        
        Computes gradient at current position, makes a step in the direction
        of high loss, and then takes a step with the gradient at that position.
        """
        assert closure is not None, "SAM requires a closure"
        
        # First step: compute gradient at current position
        closure()
        loss = closure()
        
        # Compute neighborhood
        for group in self.param_groups:
            self._apply_neighborhood_step(group)
        
        # Second step: compute gradient in neighborhood
        closure()
        loss = closure()
        
        # Final step: return to center with accumulated gradients
        for group in self.param_groups:
            self._apply_return_step(group)
        
        return loss
    
    def _apply_neighborhood_step(self, group):
        """Move to high-loss region."""
        for p in group['params']:
            if p.grad is None:
                continue
            
            e_w = (torch.pow(p, 2).sum() ** 0.5).clamp(min=1e-12)
            rho = self.rho / e_w if self.adaptive else self.rho
            
            p.data.add_(p.grad, alpha=rho)
    
    def _apply_return_step(self, group):
        """Return to center."""
        for p in group['params']:
            if p.grad is None:
                continue
            
            e_w = (torch.pow(p, 2).sum() ** 0.5).clamp(min=1e-12)
            rho = self.rho / e_w if self.adaptive else self.rho
            
            p.data.add_(p.grad, alpha=-rho)

"""
Custom loss functions for improved training.

Implements various loss functions optimized for medical image classification
and handling imbalanced datasets.
"""

from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal loss reduces the loss for well-classified examples, focusing on hard negatives.
    
    Reference:
        Lin et al. (2017). Focal Loss for Dense Object Detection. ICCV.
    """
    
    def __init__(
        self,
        alpha: Union[float, torch.Tensor] = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Initialize FocalLoss.
        
        Args:
            alpha: Weighting factor in range (0,1) or weights for each class
            gamma: Exponent of the modulating factor (1 - p_t) to balance easy/hard examples
            reduction: Specifies the reduction to apply to the output
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Model logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
        
        Returns:
            Focal loss value
        """
        p = F.softmax(inputs, dim=1)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    
    Label smoothing prevents the model from becoming overconfident by
    assigning small probability to incorrect classes.
    
    Reference:
        Szegedy et al. (2016). Rethinking the Inception Architecture. CVPR.
    """
    
    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        """
        Initialize LabelSmoothingCrossEntropy.
        
        Args:
            num_classes: Number of classes
            smoothing: Label smoothing factor (0 to 1)
            reduction: Specifies the reduction to apply to the output
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute label smoothing cross-entropy loss.
        
        Args:
            inputs: Model logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
        
        Returns:
            Loss value
        """
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smoothed target distribution
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = -torch.sum(true_dist * log_probs, dim=1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class SymmetricCrossEntropy(nn.Module):
    """
    Symmetric Cross-Entropy Loss for robust training with noisy labels.
    
    Combines cross-entropy in both directions (forward and backward) to be
    robust to label noise.
    
    Reference:
        Wang et al. (2019). Symmetric Cross-Entropy for Robust Learning with Noisy Labels.
    """
    
    def __init__(
        self,
        num_classes: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        reduction: str = "mean",
    ):
        """
        Initialize SymmetricCrossEntropy.
        
        Args:
            num_classes: Number of classes
            alpha: Weight for cross-entropy loss
            beta: Weight for reverse cross-entropy loss
            reduction: Specifies the reduction to apply to the output
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute symmetric cross-entropy loss.
        
        Args:
            inputs: Model logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
        
        Returns:
            Loss value
        """
        # Forward cross-entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Reverse cross-entropy
        p = F.softmax(inputs, dim=1)
        rce_loss = -torch.sum(p * F.log_softmax(targets.float().unsqueeze(1) / self.num_classes, dim=1), dim=1)
        
        loss = self.alpha * ce_loss + self.beta * rce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for medical image segmentation-like problems.
    
    Dice loss measures the overlap between predicted and ground truth.
    Useful for imbalanced multi-class problems.
    
    Reference:
        Milletari et al. (2016). The Dice coefficient for measuring segmentation accuracy.
    """
    
    def __init__(
        self,
        num_classes: int,
        smooth: float = 1e-6,
        reduction: str = "mean",
    ):
        """
        Initialize DiceLoss.
        
        Args:
            num_classes: Number of classes
            smooth: Smoothing constant to avoid division by zero
            reduction: Specifies the reduction to apply to the output
        """
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            inputs: Model logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
        
        Returns:
            Loss value
        """
        p = F.softmax(inputs, dim=1)
        
        # One-hot encode targets
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # Compute Dice coefficient for each class
        intersection = torch.sum(p * targets_onehot, dim=0)
        union = torch.sum(p, dim=0) + torch.sum(targets_onehot, dim=0)
        
        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score.mean()
        
        return dice_loss


class CompositeLoss(nn.Module):
    """
    Composite loss function combining multiple losses.
    
    Allows flexible combination of multiple loss functions with different weights.
    """
    
    def __init__(
        self,
        losses: list,
        weights: Optional[list] = None,
    ):
        """
        Initialize CompositeLoss.
        
        Args:
            losses: List of loss modules
            weights: List of weights for each loss (default: equal weights)
        
        Example:
            >>> losses = [nn.CrossEntropyLoss(), FocalLoss()]
            >>> composite = CompositeLoss(losses, weights=[0.7, 0.3])
        """
        super().__init__()
        self.losses = nn.ModuleList(losses)
        
        if weights is None:
            weights = [1.0 / len(losses)] * len(losses)
        
        self.weights = weights
        
        if len(self.losses) != len(self.weights):
            raise ValueError("Number of losses and weights must match")
        
        logger.info(f"Created CompositeLoss with {len(losses)} losses")
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute composite loss.
        
        Args:
            inputs: Model logits
            targets: Ground truth labels
        
        Returns:
            Weighted sum of all losses
        """
        total_loss = 0.0
        
        for loss_fn, weight in zip(self.losses, self.weights):
            loss = loss_fn(inputs, targets)
            total_loss += weight * loss
        
        return total_loss


def create_loss_function(
    loss_type: str = "cross_entropy",
    num_classes: int = None,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create loss function based on type.
    
    Args:
        loss_type: One of 'cross_entropy', 'focal', 'smoothing', 'symmetric', 'dice'
        num_classes: Number of output classes
        **kwargs: Loss-specific arguments
    
    Returns:
        Loss module
    
    Raises:
        ValueError: If loss_type is not supported
    
    Example:
        >>> loss_fn = create_loss_function('focal', num_classes=10, gamma=2.0)
    """
    logger.info(f"Creating loss function: type={loss_type}")
    
    loss_type = loss_type.lower()
    
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(**kwargs)
    
    elif loss_type == "focal":
        return FocalLoss(**kwargs)
    
    elif loss_type == "smoothing":
        if num_classes is None:
            raise ValueError("num_classes required for smoothing loss")
        return LabelSmoothingCrossEntropy(num_classes=num_classes, **kwargs)
    
    elif loss_type == "symmetric":
        if num_classes is None:
            raise ValueError("num_classes required for symmetric loss")
        return SymmetricCrossEntropy(num_classes=num_classes, **kwargs)
    
    elif loss_type == "dice":
        if num_classes is None:
            raise ValueError("num_classes required for dice loss")
        return DiceLoss(num_classes=num_classes, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Supported: ['cross_entropy', 'focal', 'smoothing', 'symmetric', 'dice']"
        )

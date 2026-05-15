"""
EfficientNet-based classifiers for medical image classification.

Supports:
- EfficientNet variants (b0 through b7)
- Pretrained ImageNet and medical imaging weights
- Compound scaling
- Feature extraction and fine-tuning
- Uncertainty estimation capabilities
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import timm
import logging
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class EfficientNetClassifier(BaseModel):
    """
    EfficientNet-based image classifier for medical diagnosis.
    
    Implements EfficientNet architecture with support for:
    - Multiple depth variants (b0 through b7)
    - Compound scaling (depth, width, resolution)
    - Transfer learning from ImageNet
    - Feature extraction
    - Uncertainty estimation via MC Dropout
    
    Reference:
        Tan & Le (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
        https://arxiv.org/abs/1905.11946
    """
    
    SUPPORTED_VARIANTS = {
        "efficientnet_b0": {"scale": 1.0, "feature_dim": 1280},
        "efficientnet_b1": {"scale": 1.1, "feature_dim": 1280},
        "efficientnet_b2": {"scale": 1.2, "feature_dim": 1408},
        "efficientnet_b3": {"scale": 1.4, "feature_dim": 1536},
        "efficientnet_b4": {"scale": 1.8, "feature_dim": 1792},
        "efficientnet_b5": {"scale": 2.2, "feature_dim": 2048},
        "efficientnet_b6": {"scale": 2.6, "feature_dim": 2304},
        "efficientnet_b7": {"scale": 4.0, "feature_dim": 2560},
    }
    
    def __init__(
        self,
        num_classes: int,
        variant: str = "efficientnet_b0",
        pretrained: bool = True,
        num_channels: int = 3,
        dropout_rate: float = 0.2,
        dropout_p_mc: float = 0.5,
        feature_dim: Optional[int] = None,
        in_chans: int = 3,
    ):
        """
        Initialize EfficientNet classifier.
        
        Args:
            num_classes: Number of output classes
            variant: EfficientNet variant - "efficientnet_b0" through "efficientnet_b7"
            pretrained: Whether to use ImageNet pretrained weights
            num_channels: Number of input channels (1 for grayscale, 3 for RGB)
            dropout_rate: Dropout rate in classifier head
            dropout_p_mc: Dropout probability for MC Dropout uncertainty estimation
            feature_dim: Dimension of feature space. If None, uses model defaults.
            in_chans: Number of input channels for timm
        
        Raises:
            ValueError: If variant is not supported
        """
        super().__init__(num_classes=num_classes, dropout_rate=dropout_rate)
        self.dropout_p_mc = dropout_p_mc

        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported EfficientNet variant '{variant}'. "
                f"Supported: {list(self.SUPPORTED_VARIANTS.keys())}"
            )
        
        self.variant = variant
        self.pretrained = pretrained
        self.num_channels = num_channels
        self.feature_dim = feature_dim or self.SUPPORTED_VARIANTS[variant]["feature_dim"]
        
        logger.info(
            f"Initialized {variant} with pretrained={pretrained}, "
            f"num_classes={num_classes}, feature_dim={self.feature_dim}"
        )
        
        # Load model using timm
        model_name = variant.replace("_", "-")
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=num_channels,
            num_classes=0,  # Remove classification head to get features
            global_pool="avg",
        )
        
        # Add classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, num_classes),
        )
        
        # For MC Dropout
        self.mc_dropout = nn.Dropout(p=self.dropout_p_mc)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, num_channels, height, width)
        
        Returns:
            Tuple of (logits, features) where:
            - logits: Class predictions of shape (batch_size, num_classes)
            - features: Feature representations of shape (batch_size, feature_dim)
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Apply classifier
        logits = self.classifier(features)
        
        return logits, features
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature representations from input images.
        
        Args:
            x: Input tensor of shape (batch_size, num_channels, height, width)
        
        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        return self.backbone(x)
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 10,
        return_all_samples: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty estimates using MC Dropout.
        
        Args:
            x: Input tensor of shape (batch_size, num_channels, height, width)
            num_samples: Number of MC Dropout samples
            return_all_samples: If True, return all prediction samples
        
        Returns:
            Dictionary containing:
            - 'logits': Mean prediction logits (batch_size, num_classes)
            - 'probs': Mean prediction probabilities (batch_size, num_classes)
            - 'uncertainty': Predictive entropy (batch_size,)
            - 'samples': All prediction samples if return_all_samples=True
        """
        self.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                logits, _ = self.forward(x)
                probs = torch.softmax(logits, dim=1)
                predictions.append(probs)
        
        self.eval()  # Disable dropout
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # (num_samples, batch_size, num_classes)
        
        # Compute mean and uncertainty
        mean_probs = predictions.mean(dim=0)
        
        # Entropy-based uncertainty
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
        
        results = {
            "logits": torch.log(mean_probs + 1e-10),
            "probs": mean_probs,
            "uncertainty": entropy,
        }
        
        if return_all_samples:
            results["samples"] = predictions
        
        return results
    
    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze backbone parameters for transfer learning.
        
        Args:
            freeze: If True, freeze backbone. If False, unfreeze.
        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        
        logger.info(f"Backbone frozen: {freeze}")
    
    def set_dropout_rate(self, rate: float) -> None:
        """
        Update dropout rate in the classifier.
        
        Args:
            rate: New dropout rate
        """
        self.classifier[0].p = rate
    
    def get_model_summary(self) -> Dict:
        """
        Get a summary of the model architecture and parameters.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "variant": self.variant,
            "num_classes": self.num_classes,
            "pretrained": self.pretrained,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "feature_dimension": self.feature_dim,
        }
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"EfficientNetClassifier(\n"
            f"  variant={self.variant},\n"
            f"  num_classes={self.num_classes},\n"
            f"  pretrained={self.pretrained},\n"
            f")"
        )

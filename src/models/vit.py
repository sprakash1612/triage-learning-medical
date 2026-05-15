"""
Vision Transformer (ViT) based classifiers for medical image classification.

Supports:
- Vision Transformer variants (tiny, small, base)
- Patch embedding extraction
- Attention map visualization
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


class VisionTransformerClassifier(BaseModel):
    """
    Vision Transformer-based image classifier for medical diagnosis.
    
    Implements Vision Transformer architecture with support for:
    - Multiple size variants (tiny, small, base)
    - Patch-based image processing
    - Self-attention mechanisms
    - Transfer learning from ImageNet
    - Feature extraction and attention visualization
    - Uncertainty estimation via MC Dropout
    
    Reference:
        Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
        https://arxiv.org/abs/2010.11929
    """
    
    SUPPORTED_VARIANTS = {
        "vit_tiny_patch16_224": {"hidden_dim": 192, "num_heads": 3},
        "vit_small_patch16_224": {"hidden_dim": 384, "num_heads": 6},
        "vit_base_patch16_224": {"hidden_dim": 768, "num_heads": 12},
        "vit_large_patch16_224": {"hidden_dim": 1024, "num_heads": 16},
    }
    
    def __init__(
        self,
        num_classes: int,
        variant: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_channels: int = 3,
        dropout_rate: float = 0.1,
        dropout_p_mc: float = 0.5,
        image_size: int = 224,
        patch_size: int = 16,
        feature_dim: Optional[int] = None,
    ):
        """
        Initialize Vision Transformer classifier.
        
        Args:
            num_classes: Number of output classes
            variant: ViT variant - "vit_tiny_patch16_224", "vit_small_patch16_224", etc.
            pretrained: Whether to use ImageNet pretrained weights
            num_channels: Number of input channels (1 for grayscale, 3 for RGB)
            dropout_rate: Dropout rate in classifier head
            dropout_p_mc: Dropout probability for MC Dropout uncertainty estimation
            image_size: Expected input image size
            patch_size: Size of image patches
            feature_dim: Dimension of feature space. If None, uses model hidden dimension.
        
        Raises:
            ValueError: If variant is not supported
        """
        super().__init__(num_classes=num_classes, dropout_rate=dropout_rate)
        self.dropout_p_mc = dropout_p_mc

        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported ViT variant '{variant}'. "
                f"Supported: {list(self.SUPPORTED_VARIANTS.keys())}"
            )
        
        self.variant = variant
        self.pretrained = pretrained
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Get model configuration
        model_config = self.SUPPORTED_VARIANTS[variant]
        self.hidden_dim = model_config["hidden_dim"]
        self.num_heads = model_config["num_heads"]
        self.feature_dim = feature_dim or self.hidden_dim
        
        logger.info(
            f"Initialized {variant} with pretrained={pretrained}, "
            f"num_classes={num_classes}, feature_dim={self.feature_dim}"
        )
        
        # Load model using timm (img_size matches MedMNIST 28 or full-res training)
        self.backbone = timm.create_model(
            variant,
            pretrained=pretrained,
            in_chans=num_channels,
            num_classes=0,  # Remove classification head to get features
            global_pool="avg",
            img_size=image_size,
        )
        
        # Add classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, num_classes),
        )
        
        # For MC Dropout
        self.mc_dropout = nn.Dropout(p=self.dropout_p_mc)
        
        # Store original model for attention extraction
        self._original_model = timm.create_model(
            variant,
            pretrained=pretrained,
            in_chans=num_channels,
            num_classes=num_classes,
            img_size=image_size,
        )
    
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
    
    def get_attention_maps(
        self,
        x: torch.Tensor,
        layer_idx: int = -1,
        head_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Extract attention maps from the model.
        
        Args:
            x: Input tensor of shape (batch_size, num_channels, height, width)
            layer_idx: Which transformer layer to extract attention from (-1 for last)
            head_idx: Which attention head to extract (None for average across all heads)
        
        Returns:
            Attention maps of shape (batch_size, num_patches+1, num_patches+1)
            or (batch_size, num_patches+1, num_patches+1) if head_idx specified
        """
        # Register hook to capture attention maps
        attention_maps = []
        
        def hook_fn(module, input, output):
            attention_maps.append(output)
        
        # Get transformer blocks
        blocks = self._original_model.blocks
        
        # Register hook on specified layer
        if layer_idx == -1:
            layer_idx = len(blocks) - 1
        
        hook = blocks[layer_idx].attn.register_forward_hook(hook_fn)
        
        try:
            # Forward pass
            with torch.no_grad():
                _ = self._original_model(x)
            
            # Extract attention maps
            attn = attention_maps[0]  # Shape: (batch_size, num_heads, num_patches+1, num_patches+1)
            
            if head_idx is not None:
                attn = attn[:, head_idx, :, :]  # (batch_size, num_patches+1, num_patches+1)
            else:
                # Average across heads
                attn = attn.mean(dim=1)  # (batch_size, num_patches+1, num_patches+1)
            
            return attn
        
        finally:
            hook.remove()
    
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
            "num_patches": self.num_patches,
            "hidden_dimension": self.hidden_dim,
            "num_heads": self.num_heads,
        }
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"VisionTransformerClassifier(\n"
            f"  variant={self.variant},\n"
            f"  num_classes={self.num_classes},\n"
            f"  pretrained={self.pretrained},\n"
            f")"
        )

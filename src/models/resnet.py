"""
ResNet models for medical image classification
"""

import torch
import torch.nn as nn
import torchvision.models as models
from .base_model import BaseModel
from typing import Optional


class ResNetClassifier(BaseModel):
    """
    ResNet-based classifier with uncertainty support
    
    Args:
        variant: 'resnet18', 'resnet34', 'resnet50', 'resnet101'
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        dropout_rate: Dropout rate for classifier
        freeze_backbone: Freeze backbone weights
        freeze_layers: Number of layers to freeze
    """
    
    def __init__(
        self,
        variant: str = 'resnet18',
        num_classes: int = 9,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False,
        freeze_layers: int = 0
    ):
        super().__init__(num_classes, dropout_rate)
        
        # Load backbone
        if variant == 'resnet18':
            backbone_fn = models.resnet18
            W = getattr(models, "ResNet18_Weights", None)
        elif variant == 'resnet34':
            backbone_fn = models.resnet34
            W = getattr(models, "ResNet34_Weights", None)
        elif variant == 'resnet50':
            backbone_fn = models.resnet50
            W = getattr(models, "ResNet50_Weights", None)
        elif variant == 'resnet101':
            backbone_fn = models.resnet101
            W = getattr(models, "ResNet101_Weights", None)
        else:
            raise ValueError(f"Unknown ResNet variant: {variant}")
        weights = W.DEFAULT if (pretrained and W is not None and hasattr(W, "DEFAULT")) else None
        try:
            if weights is not None:
                self.backbone = backbone_fn(weights=weights)
            else:
                self.backbone = backbone_fn(pretrained=pretrained)
        except TypeError:
            self.backbone = backbone_fn(pretrained=pretrained)
        
        # Get feature dimension
        num_features = self.backbone.fc.in_features
        
        # Replace final layer with custom classifier
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Freeze specific layers
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
    
    def _freeze_layers(self, num_layers: int):
        """Freeze first n layers of backbone"""
        layers = [
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4
        ]
        
        for i, layer in enumerate(layers[:num_layers]):
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            logits: Output logits (B, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification layer"""
        return self.backbone(x)
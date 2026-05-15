"""
Model factory for creating models from configuration or explicit architecture API.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

from .resnet import ResNetClassifier
from .densenet import DenseNetClassifier
from .efficientnet import EfficientNetClassifier
from .vit import VisionTransformerClassifier
from .base_model import BaseModel

logger = logging.getLogger(__name__)


def _unpack_logits(outputs):
    """Support models that return (logits, features)."""
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


class ModelFactory:
    """Factory for medical imaging classifiers (PathMNIST / ChestMNIST / etc.)."""

    @staticmethod
    def create(
        architecture: str,
        variant: str,
        num_classes: int,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        num_channels: int = 3,
        image_size: int = 28,
        dropout_p_mc: float = 0.5,
    ) -> nn.Module:
        """
        Create a classifier by architecture name and variant string.

        Args:
            architecture: One of resnet, densenet, efficientnet, vit
            variant: Short variant (e.g. "18", "121", "b3", "vit_tiny") or full internal name
            num_classes: Number of output classes / labels
            pretrained: ImageNet (or timm) pretrained weights
            dropout_rate: Dropout in classification head (where applicable)
            num_channels: Input channels (1 or 3)
            image_size: Spatial size (ViT uses this as timm img_size)
            dropout_p_mc: Extra MC dropout strength for some architectures
        """
        architecture = architecture.lower().strip()
        variant = str(variant).lower().strip()

        if architecture == "resnet":
            name_map = {"18": "resnet18", "34": "resnet34", "50": "resnet50"}
            res_name = name_map.get(variant, variant if variant.startswith("resnet") else f"resnet{variant}")
            model = ResNetClassifier(
                variant=res_name,
                num_classes=num_classes,
                pretrained=pretrained,
                dropout_rate=dropout_rate,
            )
        elif architecture == "densenet":
            if variant in ("121", "169", "201"):
                full = f"densenet{variant}"
            elif not variant.startswith("densenet"):
                full = f"densenet{variant}"
            else:
                full = variant
            model = DenseNetClassifier(
                num_classes=num_classes,
                variant=full,
                pretrained=pretrained,
                num_channels=num_channels,
                dropout_rate=dropout_rate,
                dropout_p_mc=dropout_p_mc,
            )
        elif architecture == "efficientnet":
            if variant.startswith("efficientnet"):
                full = variant
            else:
                full = f"efficientnet_{variant.replace('-', '_')}"
            model = EfficientNetClassifier(
                num_classes=num_classes,
                variant=full,
                pretrained=pretrained,
                num_channels=num_channels,
                dropout_rate=dropout_rate,
                dropout_p_mc=dropout_p_mc,
            )
        elif architecture == "vit":
            if variant in ("vit_tiny", "tiny"):
                timm_variant = "vit_tiny_patch16_224"
            elif variant in ("vit_small", "small"):
                timm_variant = "vit_small_patch16_224"
            elif variant in ("vit_base", "base"):
                timm_variant = "vit_base_patch16_224"
            else:
                timm_variant = variant if "patch" in variant else "vit_tiny_patch16_224"
            model = VisionTransformerClassifier(
                num_classes=num_classes,
                variant=timm_variant,
                pretrained=pretrained,
                num_channels=num_channels,
                dropout_rate=dropout_rate,
                dropout_p_mc=dropout_p_mc,
                image_size=image_size,
                patch_size=16,
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"ModelFactory.create: {architecture}/{variant} -> "
            f"{model.__class__.__name__} ({total_params:,} params, {trainable:,} trainable)"
        )
        return model


def create_model(config: Dict[str, Any]) -> BaseModel:
    """
    Create model from configuration dictionary.

    If config['model'] contains 'architecture' and 'variant', delegates to ModelFactory.create.
    Otherwise uses legacy config['model']['name'] for ResNet only.
    """
    model_config = config["model"]
    num_classes = config["dataset"]["num_classes"]
    pretrained = model_config.get("pretrained", True)
    dropout_rate = model_config.get("dropout_rate", 0.5)
    num_channels = model_config.get("input_channels", config["dataset"].get("input_channels", 3))
    image_size = config["dataset"].get("size", 28)

    if model_config.get("architecture"):
        return ModelFactory.create(
            architecture=model_config["architecture"],
            variant=str(model_config.get("variant", "18")),
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            num_channels=int(num_channels),
            image_size=int(image_size),
            dropout_p_mc=float(model_config.get("dropout_p_mc", 0.5)),
        )

    model_name = model_config["name"].lower()
    freeze_backbone = model_config.get("freeze_backbone", False)
    freeze_layers = model_config.get("freeze_layers", 0)

    if "resnet" in model_name:
        model = ResNetClassifier(
            variant=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone,
            freeze_layers=freeze_layers,
        )
    else:
        raise ValueError(f"Unknown model name (set model.architecture for non-ResNet): {model_name}")

    total_params = model.get_num_parameters()
    trainable_params = model.get_num_trainable_parameters()
    logger.info(f"Created {model_name} with {total_params:,} parameters")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    return model

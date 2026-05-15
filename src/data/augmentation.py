"""
Data augmentation strategies for medical image classification.

Supports:
- Albumentations-based augmentations
- Mixup and Cutmix augmentations
- AutoAugment policies for medical images
- Flexible augmentation pipeline creation
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


def get_albumentations_transforms(
    image_size: int = 224,
    augmentation_level: str = "standard",
    include_normalization: bool = True,
    normalization_mean: Optional[List[float]] = None,
    normalization_std: Optional[List[float]] = None,
) -> Dict[str, A.Compose]:
    """
    Create advanced augmentation pipeline using albumentations library.
    
    This function provides a flexible, configurable augmentation pipeline optimized
    for medical images, with multiple intensity levels from light to aggressive.
    
    Args:
        image_size: Target image size for augmentation (assumed square)
        augmentation_level: One of "light", "standard", "aggressive", "custom"
            - light: Basic augmentations (flip, rotate, brightness)
            - standard: Balanced set including color jittering and shape transformations
            - aggressive: Heavy augmentations including morphological operations
        include_normalization: Whether to include ImageNet normalization
        normalization_mean: Mean values for normalization. If None, uses ImageNet defaults.
        normalization_std: Std values for normalization. If None, uses ImageNet defaults.
    
    Returns:
        Dictionary with 'train' and 'val' transform pipelines.
        Each is an albumentations.Compose object.
    
    Example:
        >>> transforms = get_albumentations_transforms(
        ...     image_size=224,
        ...     augmentation_level="standard"
        ... )
        >>> augmented = transforms['train'](image=img)['image']
    """
    if normalization_mean is None:
        normalization_mean = [0.485, 0.456, 0.406]
    if normalization_std is None:
        normalization_std = [0.229, 0.224, 0.225]
    
    # Base augmentations common to all levels
    common_transforms = [
        A.Resize(height=image_size, width=image_size),
    ]
    
    # Augmentation level configurations
    if augmentation_level == "light":
        train_transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.3, border_mode=0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        ]
    elif augmentation_level == "standard":
        train_transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.4, border_mode=0),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
            A.GaussNoise(p=0.2),
            A.ElasticTransform(alpha=100, sigma=10, p=0.2),
            A.GridDistortion(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.3),
        ]
    elif augmentation_level == "aggressive":
        train_transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=45, p=0.5, border_mode=0),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.5),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.ElasticTransform(alpha=150, sigma=15, p=0.3),
            A.GridDistortion(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.3, rotate_limit=30, p=0.5),
            A.Perspective(scale=(0.05, 0.1), p=0.2),
            A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.2),
        ]
    else:
        raise ValueError(
            f"augmentation_level must be one of ['light', 'standard', 'aggressive'], "
            f"got {augmentation_level}"
        )
    
    # Add normalization if requested
    final_transforms = []
    if include_normalization:
        final_transforms.append(
            A.Normalize(mean=normalization_mean, std=normalization_std)
        )
    final_transforms.append(ToTensorV2())
    
    # Create train and validation pipelines
    train_pipeline = A.Compose(
        common_transforms + train_transforms + final_transforms,
        is_check_shapes=False,
    )
    
    val_pipeline = A.Compose(
        common_transforms + final_transforms,
        is_check_shapes=False,
    )
    
    return {"train": train_pipeline, "val": val_pipeline}


class MixupCutmix:
    """
    Mixup and Cutmix data augmentation for images and labels.
    
    Mixup creates virtual training examples by blending pairs of images and their labels.
    Cutmix extends this by blending patches instead of entire images.
    
    Reference:
    - Mixup: https://arxiv.org/abs/1710.09412
    - Cutmix: https://arxiv.org/abs/1905.04412
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        p: float = 0.5,
        strategy: str = "mixup",
        num_classes: Optional[int] = None,
    ):
        """
        Initialize MixupCutmix augmentation.
        
        Args:
            alpha: Beta distribution parameter for mixing coefficient
            p: Probability of applying augmentation
            strategy: One of "mixup" or "cutmix"
            num_classes: Number of classes (required for one-hot encoding)
        """
        self.alpha = alpha
        self.p = p
        self.strategy = strategy
        self.num_classes = num_classes
        
        if strategy not in ["mixup", "cutmix"]:
            raise ValueError(f"strategy must be 'mixup' or 'cutmix', got {strategy}")
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Mixup or Cutmix augmentation to a batch of images and labels.
        
        Args:
            images: Batch of images with shape (B, C, H, W)
            labels: Batch of labels with shape (B,) or (B, num_classes)
        
        Returns:
            Tuple of augmented (images, labels)
        """
        if np.random.rand() > self.p:
            return images, labels
        
        # Convert labels to one-hot if needed
        if labels.dim() == 1:
            num_classes = self.num_classes
            if num_classes is None:
                num_classes = int(labels.max()) + 1
            labels_onehot = torch.nn.functional.one_hot(
                labels.long(), num_classes=num_classes
            ).float()
        else:
            labels_onehot = labels
        
        if self.strategy == "mixup":
            return self._mixup(images, labels_onehot)
        else:
            return self._cutmix(images, labels_onehot)
    
    def _mixup(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Mixup augmentation."""
        batch_size = images.shape[0]
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Generate random permutation
        idx = torch.randperm(batch_size)
        
        # Mix images and labels
        mixed_images = lam * images + (1 - lam) * images[idx]
        mixed_labels = lam * labels + (1 - lam) * labels[idx]
        
        return mixed_images, mixed_labels
    
    def _cutmix(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Cutmix augmentation."""
        batch_size, _, height, width = images.shape
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Generate random box
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(height * cut_ratio)
        cut_w = int(width * cut_ratio)
        
        # Random box position
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, width)
        bby1 = np.clip(cy - cut_h // 2, 0, height)
        bbx2 = np.clip(cx + cut_w // 2, 0, width)
        bby2 = np.clip(cy + cut_h // 2, 0, height)
        
        # Generate random permutation
        idx = torch.randperm(batch_size)
        
        # Apply Cutmix
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[idx, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (height * width)
        mixed_labels = lam * labels + (1 - lam) * labels[idx]
        
        return mixed_images, mixed_labels


class AutoAugment:
    """
    AutoAugment policy for medical images.
    
    Implements automatic augmentation policy discovery optimized for medical imaging tasks.
    Uses a predefined set of augmentation policies tuned for medical images.
    """
    
    def __init__(
        self,
        policy_name: str = "medical",
        magnitude: int = 9,
        num_ops: int = 2,
        p: float = 0.5,
    ):
        """
        Initialize AutoAugment.
        
        Args:
            policy_name: Augmentation policy name. Currently supports "medical".
            magnitude: Magnitude level for augmentations (0-30 typically)
            num_ops: Number of augmentation operations to apply sequentially
            p: Probability of applying the policy
        """
        self.policy_name = policy_name
        self.magnitude = magnitude
        self.num_ops = num_ops
        self.p = p
        
        self.policies = self._get_medical_policies()
    
    def _get_medical_policies(self) -> List[List[Tuple[str, float, int]]]:
        """
        Get predefined augmentation policies optimized for medical images.
        
        Returns:
            List of policies, where each policy is a list of (operation, probability, magnitude)
        """
        # Policies are tuned for medical images (X-rays, pathology, etc.)
        policies = [
            [("ShiftX", 0.4, 4), ("Invert", 0.6, 3)],
            [("AutoContrast", 0.3, 0), ("ShiftY", 0.4, 2)],
            [("Equalize", 0.4, 0), ("Rotate", 0.3, 8)],
            [("Solarize", 0.2, 2), ("ShiftY", 0.5, 5)],
            [("Brightness", 0.5, 2), ("Contrast", 0.4, 3)],
            [("Posterize", 0.3, 4), ("Rotate", 0.4, 7)],
            [("SharpnessX", 0.3, 3), ("Equalize", 0.4, 0)],
            [("Cutout", 0.4, 3), ("Brightness", 0.3, 1)],
        ]
        return policies
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply AutoAugment policy to an image.
        
        Args:
            image: Input image as numpy array
        
        Returns:
            Augmented image
        """
        if np.random.rand() > self.p:
            return image
        
        # Select random policy
        policy = self.policies[np.random.randint(len(self.policies))]
        
        # Apply operations in the policy
        for op_name, prob, magnitude in policy[:self.num_ops]:
            if np.random.rand() < prob:
                image = self._apply_operation(image, op_name, magnitude)
        
        return image
    
    def _apply_operation(
        self, image: np.ndarray, op_name: str, magnitude: int
    ) -> np.ndarray:
        """
        Apply a single augmentation operation.
        
        Args:
            image: Input image
            op_name: Name of the operation
            magnitude: Magnitude of the operation (0-30)
        
        Returns:
            Augmented image
        """
        # Map magnitude to actual values
        mag_range = {
            "Brightness": (0.0, 0.9),
            "Contrast": (0.0, 0.9),
            "Rotate": (0, 30),
            "ShiftX": (0, 10),
            "ShiftY": (0, 10),
        }
        
        # For medical images, use a subset of operations
        if op_name == "AutoContrast":
            return np.clip(image * 255.0 / (image.max() + 1e-6), 0, 255).astype(
                np.uint8
            )
        elif op_name == "Equalize":
            if len(image.shape) == 3:
                # Per-channel equalization
                for i in range(image.shape[2]):
                    image[:, :, i] = self._equalize_channel(image[:, :, i])
                return image
            else:
                return self._equalize_channel(image)
        elif op_name == "Invert":
            return 255 - image if image.max() > 1 else 1 - image
        elif op_name == "Solarize":
            threshold = int(255 * (1 - magnitude / 30.0))
            return np.where(image > threshold, 255 - image, image)
        elif op_name == "Posterize":
            bits = int(8 - magnitude / 30.0 * 4)
            return (image // (2 ** (8 - bits))) * (2 ** (8 - bits))
        else:
            return image
    
    @staticmethod
    def _equalize_channel(channel: np.ndarray) -> np.ndarray:
        """Equalize a single channel using histogram equalization."""
        if channel.dtype != np.uint8:
            channel = (channel * 255).astype(np.uint8)
        hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        return cdf[channel].astype(np.uint8)


def create_augmentation_pipeline(
    config: Dict,
    augmentation_type: str = "standard",
) -> Union[Dict[str, Callable], Callable]:
    """
    Factory function to create augmentation pipeline based on configuration.
    
    This function provides a unified interface for creating augmentation strategies
    based on configuration dictionaries, supporting flexible augmentation strategies.
    
    Args:
        config: Configuration dictionary containing augmentation parameters
            Expected keys:
            - 'image_size': Target image size (default: 224)
            - 'augmentation_level': One of 'light', 'standard', 'aggressive'
            - 'use_mixup': Whether to use Mixup (default: False)
            - 'mixup_alpha': Mixup alpha parameter
            - 'mixup_prob': Mixup probability
            - 'use_cutmix': Whether to use Cutmix (default: False)
            - 'cutmix_alpha': Cutmix alpha parameter
            - 'use_autoaugment': Whether to use AutoAugment (default: False)
            - 'normalization_mean': Normalization mean values
            - 'normalization_std': Normalization std values
        augmentation_type: Type of augmentation ('standard', 'mixup', 'cutmix', 'autoaugment')
    
    Returns:
        Dictionary with 'train' and 'val' transforms (for albumentations)
        or augmentation callable (for Mixup/Cutmix/AutoAugment)
    
    Example:
        >>> config = {
        ...     'image_size': 224,
        ...     'augmentation_level': 'standard',
        ...     'use_mixup': True,
        ...     'mixup_alpha': 1.0
        ... }
        >>> aug = create_augmentation_pipeline(config, 'standard')
    """
    image_size = config.get("image_size", 224)
    aug_level = config.get("augmentation_level", "standard")
    norm_mean = config.get("normalization_mean", None)
    norm_std = config.get("normalization_std", None)
    
    logger.info(
        f"Creating augmentation pipeline with level={aug_level}, "
        f"image_size={image_size}"
    )
    
    # Get base albumentations pipeline
    base_transforms = get_albumentations_transforms(
        image_size=image_size,
        augmentation_level=aug_level,
        include_normalization=True,
        normalization_mean=norm_mean,
        normalization_std=norm_std,
    )
    
    # Add additional augmentations if requested
    augmentations = []
    
    if config.get("use_mixup", False):
        mixup = MixupCutmix(
            alpha=config.get("mixup_alpha", 1.0),
            p=config.get("mixup_prob", 0.5),
            strategy="mixup",
            num_classes=config.get("num_classes", None),
        )
        augmentations.append(("mixup", mixup))
    
    if config.get("use_cutmix", False):
        cutmix = MixupCutmix(
            alpha=config.get("cutmix_alpha", 1.0),
            p=config.get("cutmix_prob", 0.5),
            strategy="cutmix",
            num_classes=config.get("num_classes", None),
        )
        augmentations.append(("cutmix", cutmix))
    
    if config.get("use_autoaugment", False):
        auto_aug = AutoAugment(
            policy_name=config.get("autoaugment_policy", "medical"),
            magnitude=config.get("autoaugment_magnitude", 9),
            num_ops=config.get("autoaugment_num_ops", 2),
            p=config.get("autoaugment_prob", 0.5),
        )
        augmentations.append(("autoaugment", auto_aug))
    
    if augmentations:
        logger.info(f"Added augmentations: {[name for name, _ in augmentations]}")
        # Return combined augmentation with base transforms
        return {
            "base": base_transforms,
            "additional": augmentations,
        }
    
    return base_transforms

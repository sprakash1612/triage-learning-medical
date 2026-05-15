"""
MedMNIST Dataset Loader
Handles loading and preprocessing of MedMNIST datasets
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MedMNISTDataset(Dataset):
    """
    PyTorch Dataset wrapper for MedMNIST datasets
    
    Args:
        data_flag: Dataset name (e.g., 'pathmnist', 'chestmnist')
        split: 'train', 'val', or 'test'
        transform: Optional torchvision transforms
        download: Whether to download dataset if not present
        as_rgb: Convert grayscale to RGB
        size: Image size (28 or 224)
    """
    
    def __init__(
        self,
        data_flag: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        download: bool = True,
        as_rgb: bool = True,
        size: int = 28
    ):
        self.data_flag = data_flag
        self.split = split
        self.transform = transform
        self.as_rgb = as_rgb
        
        # Get dataset info
        info = INFO[data_flag]
        self.num_classes = len(info['label'])
        self.task = info['task']
        
        # Load dataset
        DataClass = getattr(medmnist, info['python_class'])
        self.dataset = DataClass(
            split=split,
            transform=None,  # We'll apply transforms manually
            download=download,
            as_rgb=as_rgb,
            size=size
        )
        
        logger.info(f"Loaded {data_flag} {split} set: {len(self.dataset)} samples")
        logger.info(f"Task: {self.task}, Number of classes: {self.num_classes}")
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: Tensor of shape (C, H, W)
            label: Tensor of shape (1,) for binary/multi-class, (num_classes,) for multi-label
        """
        img, label = self.dataset[idx]
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        else:
            # Default: convert to tensor
            img = transforms.ToTensor()(img)
            
        # Process label based on task type
        if self.task == 'multi-label, binary-class':
            label = torch.FloatTensor(label.squeeze())
        else:
            label = torch.LongTensor(label).squeeze()
            
        return img, label
    
    def get_labels(self) -> np.ndarray:
        """Return all labels for computing class weights"""
        return self.dataset.labels
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Compute class distribution"""
        labels = self.get_labels()
        if self.task == 'multi-label, binary-class':
            # For multi-label, count each class separately
            distribution = {}
            for i in range(self.num_classes):
                distribution[i] = int(labels[:, i].sum())
        else:
            unique, counts = np.unique(labels, return_counts=True)
            distribution = dict(zip(unique.astype(int), counts.astype(int)))
        return distribution


def get_transforms(
    config: Dict[str, Any],
    split: str = 'train'
) -> transforms.Compose:
    """
    Create data transforms based on configuration
    
    Args:
        config: Configuration dictionary
        split: 'train', 'val', or 'test'
    
    Returns:
        Composed transforms
    """
    aug_config = config['augmentation']
    
    if split == 'train' and aug_config['train']['enable']:
        # MedMNIST already returns PIL images, so no need to convert
        transform_list = []
        
        # Add augmentations
        if 'random_horizontal_flip' in aug_config['train']:
            transform_list.append(
                transforms.RandomHorizontalFlip(
                    p=aug_config['train']['random_horizontal_flip']
                )
            )
        
        if 'random_vertical_flip' in aug_config['train']:
            transform_list.append(
                transforms.RandomVerticalFlip(
                    p=aug_config['train']['random_vertical_flip']
                )
            )
            
        if 'random_rotation' in aug_config['train']:
            transform_list.append(
                transforms.RandomRotation(
                    degrees=aug_config['train']['random_rotation']
                )
            )
            
        if 'color_jitter' in aug_config['train']:
            cj = aug_config['train']['color_jitter']
            transform_list.append(
                transforms.ColorJitter(
                    brightness=cj.get('brightness', 0),
                    contrast=cj.get('contrast', 0),
                    saturation=cj.get('saturation', 0),
                    hue=cj.get('hue', 0)
                )
            )
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=aug_config['train']['normalize']['mean'],
                std=aug_config['train']['normalize']['std']
            )
        ])
    else:
        # Test/validation transforms (no augmentation)
        # MedMNIST already returns PIL images, so no need for ToPILImage()
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=aug_config['test']['normalize']['mean'],
                std=aug_config['test']['normalize']['std']
            )
        ]
    
    return transforms.Compose(transform_list)


def create_dataloaders(
    config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        config: Configuration dictionary
    
    Returns:
        train_loader, val_loader, test_loader
    """
    dataset_config = config['dataset']
    training_config = config['training']
    
    # Create transforms
    train_transform = get_transforms(config, 'train')
    test_transform = get_transforms(config, 'test')
    
    # Create datasets
    train_dataset = MedMNISTDataset(
        data_flag=dataset_config['data_flag'],
        split='train',
        transform=train_transform,
        download=dataset_config['download'],
        as_rgb=dataset_config['as_rgb'],
        size=dataset_config['size']
    )
    
    val_dataset = MedMNISTDataset(
        data_flag=dataset_config['data_flag'],
        split='val',
        transform=test_transform,
        download=dataset_config['download'],
        as_rgb=dataset_config['as_rgb'],
        size=dataset_config['size']
    )
    
    test_dataset = MedMNISTDataset(
        data_flag=dataset_config['data_flag'],
        split='test',
        transform=test_transform,
        download=dataset_config['download'],
        as_rgb=dataset_config['as_rgb'],
        size=dataset_config['size']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=training_config['num_workers'],
        pin_memory=training_config['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=training_config['num_workers'],
        pin_memory=training_config['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=training_config['num_workers'],
        pin_memory=training_config['pin_memory']
    )
    
    # Log dataset info
    logger.info(f"Train set: {len(train_dataset)} samples")
    logger.info(f"Validation set: {len(val_dataset)} samples")
    logger.info(f"Test set: {len(test_dataset)} samples")
    logger.info(f"Class distribution (train): {train_dataset.get_class_distribution()}")
    
    return train_loader, val_loader, test_loader


def compute_class_weights(dataset: MedMNISTDataset) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets
    
    Args:
        dataset: MedMNIST dataset
    
    Returns:
        Tensor of class weights
    """
    labels = dataset.get_labels()
    
    if dataset.task == 'multi-label, binary-class':
        # For multi-label, compute weights per class
        weights = []
        for i in range(dataset.num_classes):
            pos_count = labels[:, i].sum()
            neg_count = len(labels) - pos_count
            weight = neg_count / (pos_count + 1e-6)
            weights.append(weight)
        weights = torch.FloatTensor(weights)
    else:
        # For single-label, use sklearn's compute_class_weight
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels.flatten()
        )
        weights = torch.FloatTensor(class_weights)
    
    logger.info(f"Computed class weights: {weights}")
    return weights
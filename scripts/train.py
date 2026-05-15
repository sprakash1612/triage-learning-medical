"""
Main training script
Usage: python scripts/train.py --config configs/base_config.yaml
"""

import argparse
import yaml
import torch
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.medmnist_loader import create_dataloaders, compute_class_weights
from src.models.model_factory import create_model
from src.training.trainer import Trainer
from src.utils.reproducibility import set_seed
from src.utils.logger import setup_logger
from src.utils.device_manager import get_device


def parse_args():
    parser = argparse.ArgumentParser(description='Train medical diagnosis model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Override experiment name'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU ID to use'
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override experiment name if provided
    if args.experiment_name:
        config['experiment']['name'] = args.experiment_name
    
    # Setup logging
    log_dir = Path(config['paths']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir / f"{config['experiment']['name']}.log")
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("TRAINING MEDICAL DIAGNOSIS MODEL")
    logger.info("=" * 80)
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Configuration: {args.config}")
    
    # Set reproducibility
    set_seed(config['reproducibility']['seed'])
    
    # Get device
    device = get_device(gpu_id=args.gpu)
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Compute class weights if needed
    if config['training']['loss']['class_weights'] is None:
        logger.info("Computing class weights...")
        class_weights = compute_class_weights(train_loader.dataset)
        config['training']['loss']['class_weights'] = class_weights
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    
    # Create trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(model, config, device)
    
    # Train model
    logger.info("Starting training...")
    trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_loss, test_metrics = trainer.validate(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()
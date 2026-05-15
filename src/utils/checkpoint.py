"""
Checkpoint management for model training and evaluation.

Features:
- Save and load model checkpoints
- Track best performance checkpoints
- Resume training from checkpoints
- Automatic checkpoint cleanup
- Checkpoint validation
"""

from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
import logging
import os
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manager for saving, loading, and tracking model checkpoints.
    
    Handles:
    - Saving model, optimizer, and scheduler states
    - Loading checkpoints with validation
    - Tracking best checkpoints by metric
    - Automatic cleanup of old checkpoints
    - Resuming training from checkpoints
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_keep: int = 3,
        monitor_metric: str = "val_loss",
        mode: str = "min",
    ):
        """
        Initialize CheckpointManager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_keep: Maximum number of checkpoints to keep
            monitor_metric: Metric to monitor for best checkpoint
            mode: One of 'min' (lower is better) or 'max' (higher is better)
        
        Raises:
            ValueError: If mode is invalid
        """
        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_keep = max_keep
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        self.best_metric_value = float("inf") if mode == "min" else float("-inf")
        self.best_checkpoint_path = None
        self.checkpoint_list = []
        
        logger.info(
            f"Initialized CheckpointManager at {self.checkpoint_dir}, "
            f"monitoring {monitor_metric} (mode={mode})"
        )
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        checkpoint_name: Optional[str] = None,
    ) -> Path:
        """
        Save model checkpoint with optimizer and scheduler states.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state to save
            scheduler: Learning rate scheduler state to save
            epoch: Current epoch number
            metrics: Dictionary of metrics (e.g., loss, accuracy)
            is_best: Whether this is the best checkpoint so far
            checkpoint_name: Custom checkpoint name. If None, uses epoch number.
        
        Returns:
            Path to saved checkpoint
        
        Example:
            >>> path = checkpoint_manager.save_checkpoint(
            ...     model, optimizer, scheduler,
            ...     epoch=10, metrics={'val_loss': 0.5},
            ...     is_best=True
            ... )
        """
        if checkpoint_name is None:
            if epoch is not None:
                checkpoint_name = f"checkpoint_epoch_{epoch:03d}.pt"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_name = f"checkpoint_{timestamp}.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint dict
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics or {},
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        # Save checkpoint
        try:
            torch.save(checkpoint, checkpoint_path)
            self.checkpoint_list.append(checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
        
        # Track best checkpoint
        if metrics and self.monitor_metric in metrics:
            metric_value = metrics[self.monitor_metric]
            is_new_best = False
            
            if self.mode == "min":
                is_new_best = metric_value < self.best_metric_value
            else:
                is_new_best = metric_value > self.best_metric_value
            
            if is_new_best:
                self.best_metric_value = metric_value
                self.best_checkpoint_path = checkpoint_path
                logger.info(
                    f"New best checkpoint: {self.monitor_metric}={metric_value:.6f}"
                )
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore model, optimizer, and scheduler states.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load checkpoint into
            optimizer: Optimizer to restore state into
            scheduler: Scheduler to restore state into
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            strict: Whether to strictly enforce matching keys
        
        Returns:
            Checkpoint dictionary containing metadata
        
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If loading fails
        
        Example:
            >>> checkpoint = checkpoint_manager.load_checkpoint(
            ...     'checkpoints/best.pt', model, optimizer, scheduler
            ... )
            >>> epoch = checkpoint['epoch']
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            logger.info(f"Loading checkpoint from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
        
        # Load model state
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
            logger.info("Model state loaded successfully")
        except RuntimeError as e:
            logger.error(f"Failed to load model state: {e}")
            raise
        
        # Load optimizer state
        if load_optimizer and optimizer is not None and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info("Optimizer state loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {e}")
        
        # Load scheduler state
        if load_scheduler and scheduler is not None and "scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.info("Scheduler state loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load scheduler state: {e}")
        
        return checkpoint
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """
        Get path to the best checkpoint.
        
        Returns:
            Path to best checkpoint, or None if no checkpoints saved
        """
        return self.best_checkpoint_path
    
    def load_best_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Load the best checkpoint.
        
        Args:
            model: Model to load checkpoint into
            optimizer: Optimizer to restore state into
            scheduler: Scheduler to restore state into
        
        Returns:
            Checkpoint dictionary
        
        Raises:
            ValueError: If no best checkpoint has been saved
        """
        if self.best_checkpoint_path is None:
            raise ValueError("No best checkpoint has been saved yet")
        
        return self.load_checkpoint(
            self.best_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    
    def _cleanup_old_checkpoints(self) -> None:
        """
        Remove old checkpoints to keep only max_keep most recent.
        """
        if len(self.checkpoint_list) > self.max_keep:
            # Remove oldest checkpoints
            num_to_remove = len(self.checkpoint_list) - self.max_keep
            
            for _ in range(num_to_remove):
                old_checkpoint = self.checkpoint_list.pop(0)
                
                # Don't delete best checkpoint
                if old_checkpoint != self.best_checkpoint_path:
                    try:
                        old_checkpoint.unlink()
                        logger.debug(f"Removed old checkpoint: {old_checkpoint}")
                    except Exception as e:
                        logger.warning(f"Failed to remove checkpoint {old_checkpoint}: {e}")
    
    def get_checkpoint_list(self) -> List[Path]:
        """
        Get list of saved checkpoints.
        
        Returns:
            List of checkpoint paths
        """
        return self.checkpoint_list.copy()
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Get path to the most recently saved checkpoint.
        
        Returns:
            Path to latest checkpoint, or None if no checkpoints saved
        """
        if not self.checkpoint_list:
            return None
        return self.checkpoint_list[-1]
    
    def cleanup_all(self) -> None:
        """Remove all checkpoints."""
        for checkpoint_path in self.checkpoint_list:
            try:
                checkpoint_path.unlink()
                logger.debug(f"Deleted checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint {checkpoint_path}: {e}")
        
        self.checkpoint_list.clear()
        self.best_checkpoint_path = None
        logger.info("All checkpoints cleaned up")
    
    def validate_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
    ) -> bool:
        """
        Validate checkpoint integrity.
        
        Args:
            checkpoint_path: Path to checkpoint to validate
        
        Returns:
            True if checkpoint is valid, False otherwise
        """
        checkpoint_path = Path(checkpoint_path)
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Check required keys
            required_keys = {"model_state_dict"}
            if not required_keys.issubset(checkpoint.keys()):
                logger.warning(f"Checkpoint missing required keys: {required_keys}")
                return False
            
            logger.info(f"Checkpoint {checkpoint_path} is valid")
            return True
        
        except Exception as e:
            logger.error(f"Checkpoint validation failed: {e}")
            return False
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CheckpointManager(\n"
            f"  checkpoint_dir={self.checkpoint_dir},\n"
            f"  max_keep={self.max_keep},\n"
            f"  monitor_metric={self.monitor_metric},\n"
            f"  best_metric_value={self.best_metric_value},\n"
            f")"
        )

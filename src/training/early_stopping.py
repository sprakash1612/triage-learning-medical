"""
Early stopping callback for training termination.

Monitors validation metrics and stops training when improvement plateaus.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping callback to prevent overfitting.
    
    Monitors a validation metric and stops training if it doesn't improve
    for a specified number of epochs.
    """
    
    def __init__(
        self,
        monitor_metric: str = "val_loss",
        mode: str = "min",
        patience: int = 10,
        min_delta: float = 1e-4,
        restore_best_weights: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize EarlyStopping.
        
        Args:
            monitor_metric: Name of the metric to monitor
            mode: One of 'min' (lower is better) or 'max' (higher is better)
            patience: Number of epochs with no improvement to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best model weights after stopping
            verbose: Whether to log messages
        
        Raises:
            ValueError: If mode is invalid
        
        Example:
            >>> early_stopping = EarlyStopping(
            ...     monitor_metric='val_loss',
            ...     mode='min',
            ...     patience=10
            ... )
        """
        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        # Tracking variables
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = 0
        self.best_weights = None
        self.wait_count = 0
        self.epoch = 0
        self.history = []
        self.best_metric_epoch = {}
    
    def check_improvement(
        self,
        current_metric: float,
    ) -> bool:
        """
        Check if current metric is an improvement.
        
        Args:
            current_metric: Current metric value
        
        Returns:
            True if improvement detected, False otherwise
        """
        if self.mode == "min":
            is_improvement = current_metric < self.best_metric - self.min_delta
        else:
            is_improvement = current_metric > self.best_metric + self.min_delta
        
        return is_improvement
    
    def step(
        self,
        metrics: Dict[str, float],
        model_weights: Optional[Dict] = None,
    ) -> bool:
        """
        Check stopping criteria and update state.
        
        Args:
            metrics: Dictionary of current metrics
            model_weights: Optional model weights to save as best
        
        Returns:
            True if training should stop, False otherwise
        
        Example:
            >>> should_stop = early_stopping.step(
            ...     {'val_loss': 0.5, 'val_acc': 0.95},
            ...     model.state_dict()
            ... )
            >>> if should_stop:
            ...     break  # Stop training
        """
        if self.monitor_metric not in metrics:
            raise ValueError(
                f"Metric '{self.monitor_metric}' not found in provided metrics. "
                f"Available metrics: {list(metrics.keys())}"
            )
        
        current_metric = metrics[self.monitor_metric]
        self.history.append(current_metric)
        
        if self.check_improvement(current_metric):
            # Improvement detected
            self.best_metric = current_metric
            self.best_epoch = self.epoch
            self.wait_count = 0
            
            if model_weights is not None:
                self.best_weights = {k: v.clone() if hasattr(v, 'clone') else v 
                                    for k, v in model_weights.items()}
            
            if self.verbose:
                logger.info(
                    f"Epoch {self.epoch}: {self.monitor_metric} improved to "
                    f"{current_metric:.6f}"
                )
        else:
            # No improvement
            self.wait_count += 1
            
            if self.verbose and self.wait_count % 5 == 0:
                logger.info(
                    f"Epoch {self.epoch}: {self.monitor_metric} did not improve "
                    f"for {self.wait_count}/{self.patience} epochs"
                )
        
        # Store metric by epoch
        self.best_metric_epoch[self.epoch] = current_metric
        self.epoch += 1
        
        # Check stopping criteria
        should_stop = self.wait_count >= self.patience
        
        if should_stop and self.verbose:
            logger.info(
                f"Early stopping triggered. Best {self.monitor_metric}: "
                f"{self.best_metric:.6f} at epoch {self.best_epoch}"
            )
        
        return should_stop
    
    def should_stop(self) -> bool:
        """
        Check if training should stop based on current state.
        
        Returns:
            True if patience exceeded, False otherwise
        """
        return self.wait_count >= self.patience
    
    def get_best_epoch(self) -> int:
        """
        Get the epoch with the best metric value.
        
        Returns:
            Best epoch number
        """
        return self.best_epoch
    
    def get_best_metric(self) -> float:
        """
        Get the best metric value seen.
        
        Returns:
            Best metric value
        """
        return self.best_metric
    
    def restore_best_weights(self) -> Dict:
        """
        Get the best model weights.
        
        Returns:
            Best weights dictionary, or None if not saved
        """
        if self.best_weights is None:
            logger.warning("No best weights available. Model weights were not saved.")
            return None
        return self.best_weights
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.best_metric = float("inf") if self.mode == "min" else float("-inf")
        self.best_epoch = 0
        self.best_weights = None
        self.wait_count = 0
        self.epoch = 0
        self.history = []
        self.best_metric_epoch = {}
        
        if self.verbose:
            logger.info("Early stopping reset")
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EarlyStopping(\n"
            f"  monitor_metric={self.monitor_metric},\n"
            f"  mode={self.mode},\n"
            f"  patience={self.patience},\n"
            f"  best_metric={self.best_metric:.6f},\n"
            f"  wait_count={self.wait_count}/{self.patience},\n"
            f")"
        )


class MultiMetricEarlyStopping:
    """
    Early stopping that monitors multiple metrics simultaneously.
    
    Useful when you want to optimize for multiple objectives.
    """
    
    def __init__(
        self,
        monitors: Dict[str, str],  # {metric_name: 'min' or 'max'}
        patience: int = 10,
        min_delta: Dict[str, float] = None,
        verbose: bool = True,
    ):
        """
        Initialize MultiMetricEarlyStopping.
        
        Args:
            monitors: Dictionary mapping metric names to optimization direction ('min' or 'max')
            patience: Patience for early stopping
            min_delta: Minimum improvement per metric
            verbose: Whether to log messages
        """
        self.monitors = monitors
        self.patience = patience
        self.verbose = verbose
        
        if min_delta is None:
            min_delta = {metric: 1e-4 for metric in monitors.keys()}
        self.min_delta = min_delta
        
        # Initialize stoppers for each metric
        self.stoppers = {
            metric: EarlyStopping(
                monitor_metric=metric,
                mode=direction,
                patience=patience,
                min_delta=min_delta.get(metric, 1e-4),
                verbose=False,
            )
            for metric, direction in monitors.items()
        }
    
    def step(
        self,
        metrics: Dict[str, float],
        model_weights: Optional[Dict] = None,
    ) -> bool:
        """
        Check stopping criteria for all monitored metrics.
        
        Returns:
            True if any metric suggests stopping
        """
        all_should_stop = True
        
        for metric_name, stopper in self.stoppers.items():
            if metric_name in metrics:
                should_stop = stopper.step({metric_name: metrics[metric_name]}, model_weights)
            else:
                should_stop = False
            
            all_should_stop = all_should_stop and should_stop
        
        return all_should_stop
    
    def get_best_weights(self) -> Optional[Dict]:
        """Get best weights from primary metric stopper."""
        primary_metric = list(self.stoppers.keys())[0]
        return self.stoppers[primary_metric].restore_best_weights()

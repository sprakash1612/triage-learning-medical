"""
Calibration metrics and utilities for measuring model confidence quality.

Implements:
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Adaptive Binning
- Reliability diagrams
- Per-class calibration metrics
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import logging
from pathlib import Path
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class CalibrationMetrics:
    """
    Compute calibration quality metrics.
    """
    
    @staticmethod
    def compute_ece(
        predictions: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 10,
        binning_strategy: str = "uniform",
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE measures the average difference between confidence and accuracy.
        Lower ECE indicates better calibration.
        
        Formula:
            ECE = Σ |accuracy_i - confidence_i| * |bin_i| / N
        
        Args:
            predictions: Model predictions of shape (N, num_classes)
            labels: Ground truth labels of shape (N,)
            num_bins: Number of confidence bins
            binning_strategy: One of 'uniform' or 'adaptive'
        
        Returns:
            Expected Calibration Error value (0 to 1)
        
        Example:
            >>> ece = CalibrationMetrics.compute_ece(probs, labels)
            >>> print(f"ECE: {ece:.4f}")
        """
        confidences = np.max(predictions, axis=1)
        pred_labels = np.argmax(predictions, axis=1)
        correctness = (pred_labels == labels).astype(np.float32)
        
        if binning_strategy == "uniform":
            bin_edges = np.linspace(0, 1, num_bins + 1)
        elif binning_strategy == "adaptive":
            bin_edges = CalibrationMetrics._get_adaptive_bin_edges(
                confidences, num_bins
            )
        else:
            raise ValueError(f"Unknown binning strategy: {binning_strategy}")
        
        ece = 0.0
        
        for i in range(num_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
            
            if mask.sum() == 0:
                continue
            
            bin_accuracy = correctness[mask].mean()
            bin_confidence = confidences[mask].mean()
            bin_weight = mask.sum() / len(labels)
            
            ece += np.abs(bin_accuracy - bin_confidence) * bin_weight
        
        return ece
    
    @staticmethod
    def compute_mce(
        predictions: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 10,
    ) -> float:
        """
        Compute Maximum Calibration Error (MCE).
        
        MCE is the maximum difference between accuracy and confidence across all bins.
        Provides a worst-case calibration metric.
        
        Args:
            predictions: Model predictions of shape (N, num_classes)
            labels: Ground truth labels of shape (N,)
            num_bins: Number of confidence bins
        
        Returns:
            Maximum Calibration Error value (0 to 1)
        """
        confidences = np.max(predictions, axis=1)
        pred_labels = np.argmax(predictions, axis=1)
        correctness = (pred_labels == labels).astype(np.float32)
        
        bin_edges = np.linspace(0, 1, num_bins + 1)
        
        mce = 0.0
        
        for i in range(num_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
            
            if mask.sum() == 0:
                continue
            
            bin_accuracy = correctness[mask].mean()
            bin_confidence = confidences[mask].mean()
            
            mce = max(mce, np.abs(bin_accuracy - bin_confidence))
        
        return mce
    
    @staticmethod
    def compute_brier_score(
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Multiclass Brier score: mean squared error between one-hot labels and probs.
        
        Args:
            predictions: (N, num_classes) probability distribution (rows sum to ~1)
            labels: (N,) integer class indices
        
        Returns:
            Scalar Brier score in [0, 2] (lower is better)
        """
        n = len(labels)
        num_classes = predictions.shape[1]
        y = np.zeros((n, num_classes), dtype=np.float64)
        y[np.arange(n), labels.astype(np.int64)] = 1.0
        p = np.clip(predictions.astype(np.float64), 0.0, 1.0)
        return float(np.mean(np.sum((p - y) ** 2, axis=1)))
    
    @staticmethod
    def compute_reliability_diagram(
        predictions: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute reliability diagram data (bin-wise confidence vs accuracy).
        
        Args:
            predictions: Model predictions of shape (N, num_classes)
            labels: Ground truth labels of shape (N,)
            num_bins: Number of confidence bins
        
        Returns:
            Tuple of (bin_centers, accuracies, confidences)
        """
        confidences = np.max(predictions, axis=1)
        pred_labels = np.argmax(predictions, axis=1)
        correctness = (pred_labels == labels).astype(np.float32)
        
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        accuracies = []
        conf_values = []
        
        for i in range(num_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
            
            if mask.sum() == 0:
                accuracies.append(0.0)
                conf_values.append(bin_centers[i])
            else:
                accuracies.append(correctness[mask].mean())
                conf_values.append(confidences[mask].mean())
        
        return bin_centers, np.array(accuracies), np.array(conf_values)
    
    @staticmethod
    def compute_classwise_ece(
        predictions: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 10,
    ) -> Dict[int, float]:
        """
        Compute per-class Expected Calibration Error.
        
        Useful for multi-class problems where calibration varies by class.
        
        Args:
            predictions: Model predictions of shape (N, num_classes)
            labels: Ground truth labels of shape (N,)
            num_bins: Number of confidence bins
        
        Returns:
            Dictionary mapping class index to ECE value
        """
        num_classes = predictions.shape[1]
        classwise_ece = {}
        
        for class_idx in range(num_classes):
            # Get one-hot encoded predictions and labels for this class
            class_preds = predictions[:, class_idx]
            class_labels = (labels == class_idx).astype(np.float32)
            
            # Compute ECE for this class (as binary classification)
            bin_edges = np.linspace(0, 1, num_bins + 1)
            ece = 0.0
            
            for i in range(num_bins):
                mask = (class_preds >= bin_edges[i]) & (class_preds < bin_edges[i+1])
                
                if mask.sum() == 0:
                    continue
                
                bin_accuracy = class_labels[mask].mean()
                bin_confidence = class_preds[mask].mean()
                bin_weight = mask.sum() / len(labels)
                
                ece += np.abs(bin_accuracy - bin_confidence) * bin_weight
            
            classwise_ece[class_idx] = ece
        
        return classwise_ece
    
    @staticmethod
    def _get_adaptive_bin_edges(
        confidences: np.ndarray,
        num_bins: int = 10,
    ) -> np.ndarray:
        """
        Get adaptive bin edges based on quantiles of confidence distribution.
        
        Args:
            confidences: Confidence values
            num_bins: Number of bins
        
        Returns:
            Array of bin edges
        """
        quantiles = np.linspace(0, 1, num_bins + 1)
        bin_edges = np.quantile(confidences, quantiles)
        
        # Ensure bin edges are strictly increasing
        bin_edges = np.unique(bin_edges)
        
        return bin_edges


class AdaptiveBinning:
    """
    Adaptive binning strategy for calibration analysis.
    
    Creates bins based on sample distribution rather than uniform intervals,
    which can provide better statistical coverage.
    """
    
    def __init__(
        self,
        num_bins: int = 10,
        min_samples_per_bin: int = 10,
    ):
        """
        Initialize AdaptiveBinning.
        
        Args:
            num_bins: Target number of bins
            min_samples_per_bin: Minimum samples per bin
        """
        self.num_bins = num_bins
        self.min_samples_per_bin = min_samples_per_bin
        self.bin_edges = None
    
    def fit(self, confidences: np.ndarray) -> None:
        """
        Fit binning strategy to confidence distribution.
        
        Args:
            confidences: Confidence values to use for fitting
        """
        quantiles = np.linspace(0, 1, self.num_bins + 1)
        bin_edges = np.quantile(confidences, quantiles)
        self.bin_edges = np.unique(bin_edges)
        
        logger.info(
            f"Adaptive binning fitted with {len(self.bin_edges)-1} bins "
            f"from {len(self.num_bins)} target bins"
        )
    
    def get_bin_indices(self, confidences: np.ndarray) -> np.ndarray:
        """
        Get bin indices for confidences.
        
        Args:
            confidences: Confidence values
        
        Returns:
            Array of bin indices
        """
        if self.bin_edges is None:
            raise ValueError("Binning strategy not fitted. Call fit() first.")
        
        return np.digitize(confidences, self.bin_edges) - 1
    
    def get_bin_stats(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get statistics for each bin.
        
        Args:
            confidences: Confidence values
            correctness: Binary correctness values
        
        Returns:
            Tuple of (bin_accuracies, bin_confidences, bin_sizes)
        """
        if self.bin_edges is None:
            raise ValueError("Binning strategy not fitted. Call fit() first.")
        
        num_bins = len(self.bin_edges) - 1
        bin_indices = self.get_bin_indices(confidences)
        
        accuracies = np.zeros(num_bins)
        confs = np.zeros(num_bins)
        sizes = np.zeros(num_bins, dtype=int)
        
        for i in range(num_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                accuracies[i] = correctness[mask].mean()
                confs[i] = confidences[mask].mean()
                sizes[i] = mask.sum()
        
        return accuracies, confs, sizes


def plot_calibration_curve(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot calibration curve (reliability diagram).
    
    Args:
        predictions: Model predictions of shape (N, num_classes)
        labels: Ground truth labels of shape (N,)
        num_bins: Number of confidence bins
        save_path: Path to save figure (optional)
    
    Returns:
        Matplotlib figure object
    """
    bin_centers, accuracies, confidences = CalibrationMetrics.compute_reliability_diagram(
        predictions, labels, num_bins
    )
    
    ece = CalibrationMetrics.compute_ece(predictions, labels, num_bins)
    mce = CalibrationMetrics.compute_mce(predictions, labels, num_bins)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    
    # Reliability curve
    ax.plot(confidences, accuracies, 'o-', color='steelblue', 
           markersize=8, linewidth=2, label='Model')
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Calibration Curve (ECE={ece:.4f}, MCE={mce:.4f})', 
                fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Calibration curve saved to {save_path}")
    
    return fig

"""
Evaluation metrics for classification and triage system
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from typing import Dict, Optional
import torch


def compute_classification_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics
    
    Args:
        predictions: Predicted classes (N,)
        labels: True labels (N,)
        probabilities: Predicted probabilities (N, num_classes)
        num_classes: Number of classes
    
    Returns:
        metrics: Dictionary of metric values
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = float(accuracy_score(labels, predictions))
    
    # Multi-class metrics
    avg_method = 'macro' if num_classes and num_classes > 2 else 'binary'
    
    metrics['precision'] = float(precision_score(
        labels, predictions, average=avg_method, zero_division=0
    ))
    metrics['recall'] = float(recall_score(
        labels, predictions, average=avg_method, zero_division=0
    ))
    metrics['f1_score'] = float(f1_score(
        labels, predictions, average=avg_method, zero_division=0
    ))
    
    # ROC-AUC and PR-AUC (if probabilities provided)
    if probabilities is not None:
        try:
            if num_classes == 2:
                metrics['auroc'] = float(roc_auc_score(labels, probabilities[:, 1]))
                metrics['auprc'] = float(average_precision_score(labels, probabilities[:, 1]))
            else:
                metrics['auroc'] = float(roc_auc_score(
                    labels, probabilities, multi_class='ovr', average='macro'
                ))
        except:
            pass
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def compute_calibration_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15
) -> Dict[str, float]:
    """
    Compute calibration metrics (ECE, MCE)
    
    Args:
        probabilities: Predicted probabilities (N, num_classes)
        labels: True labels (N,)
        num_bins: Number of bins for calibration
    
    Returns:
        metrics: Dictionary with ECE and MCE
    """
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    # Create bins
    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1
    
    ece = 0.0
    mce = 0.0
    
    for i in range(num_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_confidence = confidences[mask].mean()
            bin_accuracy = accuracies[mask].mean()
            bin_weight = mask.sum() / len(labels)
            
            ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
            mce = max(mce, np.abs(bin_accuracy - bin_confidence))
    
    return {
        'ece': float(ece),
        'mce': float(mce)
    }
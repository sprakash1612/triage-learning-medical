#!/usr/bin/env python3
"""
Evaluate trained model on test set.

Loads a checkpoint, runs inference, computes metrics,
generates confusion matrix, and creates visualizations.
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_checkpoint(
    checkpoint_path: str,
    device: str = 'cpu',
) -> Tuple[torch.nn.Module, Dict]:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to ('cpu' or 'cuda')
    
    Returns:
        (model, checkpoint_dict) tuple
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    logger.info(f"Checkpoint keys: {checkpoint.keys()}")
    
    return checkpoint


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    logits: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: (n_samples,) Predicted class indices
        targets: (n_samples,) True class indices
        logits: (n_samples, n_classes) Optional logits for cross-entropy
    
    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        balanced_accuracy_score, hamming_loss
    )
    
    correct = (predictions == targets).astype(int)
    accuracy = accuracy_score(targets, predictions)
    balanced_acc = balanced_accuracy_score(targets, predictions)
    
    # Per-class metrics
    precision = precision_score(targets, predictions, average='weighted', zero_division=0)
    recall = recall_score(targets, predictions, average='weighted', zero_division=0)
    f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
    
    # Cross-entropy loss
    ce_loss = np.nan
    if logits is not None:
        logits_torch = torch.from_numpy(logits).float()
        targets_torch = torch.from_numpy(targets).long()
        ce_loss = F.cross_entropy(logits_torch, targets_torch).item()
    
    metrics = {
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'cross_entropy_loss': float(ce_loss),
        'error_rate': float(1 - accuracy),
    }
    
    return metrics


def generate_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    normalize: str = 'true',
) -> np.ndarray:
    """
    Generate confusion matrix.
    
    Args:
        predictions: Predicted labels
        targets: True labels
        normalize: One of 'true', 'pred', 'all', None
    
    Returns:
        (n_classes, n_classes) confusion matrix
    """
    from sklearn.metrics import confusion_matrix as sklearn_cm
    
    cm = sklearn_cm(targets, predictions, normalize=normalize)
    return cm


def save_results(
    predictions: np.ndarray,
    targets: np.ndarray,
    metrics: Dict[str, float],
    confusion_matrix: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Save evaluation results to files.
    
    Args:
        predictions: Predicted labels
        targets: True labels
        metrics: Dictionary of metrics
        confusion_matrix: Confusion matrix
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics as JSON
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Save confusion matrix
    cm_path = output_dir / 'confusion_matrix.npy'
    np.save(cm_path, confusion_matrix)
    logger.info(f"Saved confusion matrix to {cm_path}")
    
    # Save predictions and targets as CSV
    import csv
    predictions_path = output_dir / 'predictions.csv'
    with open(predictions_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'predicted', 'target', 'correct'])
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            correct = int(pred == target)
            writer.writerow([i, int(pred), int(target), correct])
    logger.info(f"Saved predictions to {predictions_path}")
    
    # Save summary report
    report = f"""
EVALUATION REPORT
=================

Accuracy: {metrics['accuracy']:.4f}
Balanced Accuracy: {metrics['balanced_accuracy']:.4f}
Precision: {metrics['precision']:.4f}
Recall: {metrics['recall']:.4f}
F1 Score: {metrics['f1_score']:.4f}
Error Rate: {metrics['error_rate']:.4f}
Cross-Entropy Loss: {metrics['cross_entropy_loss']:.4f}

Confusion Matrix Shape: {confusion_matrix.shape}
Confusion Matrix (first 5x5):
{confusion_matrix[:5, :5]}

Total Samples: {len(targets)}
"""
    
    report_path = output_dir / 'evaluation_report.txt'
    report_path.write_text(report)
    logger.info(f"Saved report to {report_path}")


def create_visualizations(
    predictions: np.ndarray,
    targets: np.ndarray,
    confusion_matrix: np.ndarray,
    output_dir: Path,
    class_names: Optional[list] = None,
) -> None:
    """
    Create evaluation visualizations.
    
    Args:
        predictions: Predicted labels
        targets: True labels
        confusion_matrix: Confusion matrix
        output_dir: Directory to save plots
        class_names: Optional list of class names
    """
    try:
        import matplotlib.pyplot as plt
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix heatmap
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(confusion_matrix, cmap='Blues', aspect='auto')
        
        n_classes = len(confusion_matrix)
        if class_names is None:
            class_names = [str(i) for i in range(n_classes)]
        
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        
        # Add values
        for i in range(n_classes):
            for j in range(n_classes):
                text = ax.text(j, i, f'{confusion_matrix[i, j]:.2f}',
                             ha='center', va='center',
                             color='white' if confusion_matrix[i, j] > 0.5 else 'black')
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix (Normalized)')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        cm_path = output_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix plot to {cm_path}")
        plt.close()
        
        # Error distribution
        errors = (predictions != targets).astype(int)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(errors, bins=2, edgecolor='black')
        ax.set_xlabel('Error (0=correct, 1=incorrect)')
        ax.set_ylabel('Count')
        ax.set_title('Error Distribution')
        ax.grid(True, alpha=0.3)
        
        error_path = output_dir / 'error_distribution.png'
        plt.savefig(error_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved error distribution plot to {error_path}")
        plt.close()
        
    except ImportError:
        logger.warning("Matplotlib not available, skipping visualizations")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate model with predictions
  python evaluate.py --predictions preds.npy --targets targets.npy
  
  # Evaluate with logits
  python evaluate.py --logits logits.npy --targets targets.npy
  
  # Save to custom directory
  python evaluate.py --predictions preds.npy --targets targets.npy --output-dir results/eval
        """,
    )
    
    parser.add_argument(
        '--predictions',
        required=True,
        help='Path to predictions numpy file (n_samples,)',
    )
    parser.add_argument(
        '--targets',
        required=True,
        help='Path to target labels numpy file (n_samples,)',
    )
    parser.add_argument(
        '--logits',
        help='Path to logits numpy file (n_samples, n_classes) - optional',
    )
    parser.add_argument(
        '--class-names',
        help='Path to class names JSON file',
    )
    parser.add_argument(
        '--output-dir',
        default='results/evaluation',
        help='Directory to save results (default: results/evaluation)',
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualization plots',
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose logging',
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Load data
    logger.info("Loading predictions and targets...")
    predictions = np.load(args.predictions)
    targets = np.load(args.targets)
    
    logits = None
    if args.logits:
        logits = np.load(args.logits)
    
    class_names = None
    if args.class_names:
        with open(args.class_names) as f:
            class_names = json.load(f)
    
    logger.info(f"Predictions shape: {predictions.shape}")
    logger.info(f"Targets shape: {targets.shape}")
    
    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(predictions, targets, logits)
    
    logger.info("Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Generate confusion matrix
    logger.info("Generating confusion matrix...")
    cm = generate_confusion_matrix(predictions, targets, normalize='true')
    
    # Save results
    logger.info("Saving results...")
    save_results(predictions, targets, metrics, cm, args.output_dir)
    
    # Create visualizations
    if args.visualize:
        logger.info("Creating visualizations...")
        create_visualizations(predictions, targets, cm, args.output_dir, class_names)
    
    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()

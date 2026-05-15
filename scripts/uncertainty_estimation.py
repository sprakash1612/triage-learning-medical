#!/usr/bin/env python3
"""
Estimate uncertainty using MC Dropout and analyze predictions.

Runs inference with MC Dropout to estimate aleatoric and epistemic uncertainty,
analyzes uncertainty distribution, and correlates with prediction errors.
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def mc_dropout_inference(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    num_samples: int = 10,
    device: str = 'cpu',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run MC Dropout inference.
    
    Args:
        model: Neural network model with dropout
        data_loader: DataLoader for test data
        num_samples: Number of MC samples
        device: 'cpu' or 'cuda'
    
    Returns:
        (predictions, logits, uncertainties) tuple
        - predictions: (n_samples,) predicted classes
        - logits: (n_samples, n_classes) prediction logits
        - uncertainties: (n_samples,) entropy-based uncertainty
    """
    model.eval()
    
    # Enable dropout during inference
    def enable_dropout(m):
        if type(m) == torch.nn.Dropout:
            m.train()
    
    all_probs = []
    all_logits = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Assume batch is either tensor or (images, labels)
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            
            images = images.to(device)
            
            # MC samples
            sample_probs = []
            sample_logits = []
            
            for _ in range(num_samples):
                model.apply(enable_dropout)
                outputs = model(images)
                
                logits = outputs if isinstance(outputs, torch.Tensor) else outputs['logits']
                probs = torch.softmax(logits, dim=1)
                
                sample_probs.append(probs.cpu().numpy())
                sample_logits.append(logits.cpu().numpy())
            
            # Average across samples
            mean_probs = np.mean(sample_probs, axis=0)  # (batch_size, n_classes)
            mean_logits = np.mean(sample_logits, axis=0)
            
            all_probs.append(mean_probs)
            all_logits.append(mean_logits)
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1} batches")
    
    # Concatenate all batches
    all_probs = np.concatenate(all_probs, axis=0)  # (n_samples, n_classes)
    all_logits = np.concatenate(all_logits, axis=0)
    
    # Predictions from mean probabilities
    predictions = np.argmax(all_probs, axis=1)
    
    # Entropy-based uncertainty
    uncertainties = -np.sum(all_probs * np.log(all_probs + 1e-10), axis=1)
    
    return predictions, all_logits, uncertainties


def compute_uncertainty_metrics(
    uncertainties: np.ndarray,
    predictions: np.ndarray,
    targets: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute uncertainty quality metrics.
    
    Args:
        uncertainties: (n_samples,) Uncertainty values
        predictions: (n_samples,) Predicted classes
        targets: (n_samples,) Optional true labels
    
    Returns:
        Dictionary with uncertainty metrics
    """
    metrics = {
        'uncertainty_mean': float(np.mean(uncertainties)),
        'uncertainty_std': float(np.std(uncertainties)),
        'uncertainty_min': float(np.min(uncertainties)),
        'uncertainty_max': float(np.max(uncertainties)),
    }
    
    # Correlation with errors if targets provided
    if targets is not None:
        errors = (predictions != targets).astype(int)
        
        if len(np.unique(errors)) > 1:
            from scipy.stats import spearmanr, pointbiserialr
            
            # Spearman correlation
            spear_corr, spear_pval = spearmanr(uncertainties, errors)
            metrics['uncertainty_error_spearman'] = float(spear_corr)
            metrics['uncertainty_error_spearman_pval'] = float(spear_pval)
            
            # Point-biserial correlation (for binary errors)
            pbis_corr, pbis_pval = pointbiserialr(errors, uncertainties)
            metrics['uncertainty_error_pointbiserial'] = float(pbis_corr)
            metrics['uncertainty_error_pointbiserial_pval'] = float(pbis_pval)
    
    return metrics


def analyze_uncertainty_distribution(
    uncertainties: np.ndarray,
    errors: Optional[np.ndarray] = None,
) -> Dict:
    """
    Analyze uncertainty distribution.
    
    Args:
        uncertainties: (n_samples,) Uncertainty values
        errors: (n_samples,) Optional binary error indicators
    
    Returns:
        Dictionary with distribution analysis
    """
    analysis = {
        'mean': float(np.mean(uncertainties)),
        'std': float(np.std(uncertainties)),
        'median': float(np.median(uncertainties)),
        'min': float(np.min(uncertainties)),
        'max': float(np.max(uncertainties)),
        'q25': float(np.percentile(uncertainties, 25)),
        'q75': float(np.percentile(uncertainties, 75)),
        'histogram_bins': 10,
        'histogram': list(np.histogram(uncertainties, bins=10)[0]),
    }
    
    # Per-error-status distribution
    if errors is not None:
        errors = np.asarray(errors)
        correct_unc = uncertainties[errors == 0]
        error_unc = uncertainties[errors == 1]
        
        analysis['correct_uncertainty_mean'] = float(np.mean(correct_unc)) if len(correct_unc) > 0 else np.nan
        analysis['correct_uncertainty_std'] = float(np.std(correct_unc)) if len(correct_unc) > 0 else np.nan
        analysis['error_uncertainty_mean'] = float(np.mean(error_unc)) if len(error_unc) > 0 else np.nan
        analysis['error_uncertainty_std'] = float(np.std(error_unc)) if len(error_unc) > 0 else np.nan
    
    return analysis


def generate_uncertainty_report(
    uncertainties: np.ndarray,
    predictions: np.ndarray,
    targets: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate uncertainty analysis report.
    
    Args:
        uncertainties: Uncertainty values
        predictions: Predicted classes
        targets: Optional true labels
        output_path: Optional path to save report
    
    Returns:
        Report text
    """
    errors = None
    if targets is not None:
        errors = (predictions != targets).astype(int)
    
    metrics = compute_uncertainty_metrics(uncertainties, predictions, targets)
    analysis = analyze_uncertainty_distribution(uncertainties, errors)
    
    report = """
UNCERTAINTY ESTIMATION REPORT
=============================

UNCERTAINTY STATISTICS
----------------------
Mean: {:.4f}
Std: {:.4f}
Median: {:.4f}
Min: {:.4f}
Max: {:.4f}
Q25: {:.4f}
Q75: {:.4f}

""".format(
        analysis['mean'],
        analysis['std'],
        analysis['median'],
        analysis['min'],
        analysis['max'],
        analysis['q25'],
        analysis['q75'],
    )
    
    if errors is not None:
        report += """UNCERTAINTY BY PREDICTION STATUS
--------------------------------
Correct Predictions:
  Mean Uncertainty: {:.4f}
  Std Uncertainty: {:.4f}
  Count: {}

Incorrect Predictions:
  Mean Uncertainty: {:.4f}
  Std Uncertainty: {:.4f}
  Count: {}

CORRELATION WITH ERRORS
-----------------------
Spearman Correlation: {:.4f} (p={:.2e})
Point-Biserial Correlation: {:.4f} (p={:.2e})

""".format(
            analysis['correct_uncertainty_mean'],
            analysis['correct_uncertainty_std'],
            len(uncertainties[errors == 0]),
            analysis['error_uncertainty_mean'],
            analysis['error_uncertainty_std'],
            len(uncertainties[errors == 1]),
            metrics.get('uncertainty_error_spearman', np.nan),
            metrics.get('uncertainty_error_spearman_pval', np.nan),
            metrics.get('uncertainty_error_pointbiserial', np.nan),
            metrics.get('uncertainty_error_pointbiserial_pval', np.nan),
        )
    
    report += f"""
INTERPRETATION
--------------
If uncertainty is well-calibrated:
- Samples with HIGH uncertainty should have HIGHER error rate
- Spearman/Biserial correlation should be POSITIVE and significant
- This indicates uncertainty can be used for rejection/deferral

Total Samples: {len(uncertainties)}
"""
    
    if output_path:
        Path(output_path).write_text(report)
        logger.info(f"Report saved to {output_path}")
    
    return report


def save_uncertainties(
    uncertainties: np.ndarray,
    predictions: np.ndarray,
    targets: Optional[np.ndarray] = None,
    output_dir: Path = None,
) -> None:
    """
    Save uncertainty estimates and analysis.
    
    Args:
        uncertainties: Uncertainty values
        predictions: Predicted classes
        targets: Optional true labels
        output_dir: Directory to save files
    """
    output_dir = Path(output_dir) if output_dir else Path('results/uncertainty')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save numpy arrays
    np.save(output_dir / 'uncertainties.npy', uncertainties)
    np.save(output_dir / 'predictions.npy', predictions)
    
    if targets is not None:
        np.save(output_dir / 'targets.npy', targets)
        errors = (predictions != targets).astype(int)
        np.save(output_dir / 'errors.npy', errors)
    
    # Save metrics
    metrics = compute_uncertainty_metrics(uncertainties, predictions, targets)
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save analysis
    analysis = analyze_uncertainty_distribution(
        uncertainties,
        (predictions != targets) if targets is not None else None
    )
    with open(output_dir / 'analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"Saved uncertainties to {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Estimate uncertainty using MC Dropout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Estimate uncertainty with predictions
  python uncertainty_estimation.py --predictions preds.npy --uncertainties unc.npy
  
  # Analyze with targets
  python uncertainty_estimation.py --predictions preds.npy --targets targets.npy --output-dir results/unc
  
  # Generate report and visualizations
  python uncertainty_estimation.py --predictions preds.npy --targets targets.npy --report --visualize
        """,
    )
    
    parser.add_argument(
        '--predictions',
        required=True,
        help='Path to predictions numpy file (n_samples,)',
    )
    parser.add_argument(
        '--uncertainties',
        help='Path to uncertainties numpy file (n_samples,)',
    )
    parser.add_argument(
        '--targets',
        help='Path to target labels (optional)',
    )
    parser.add_argument(
        '--output-dir',
        default='results/uncertainty',
        help='Directory to save results',
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate analysis report',
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create uncertainty visualizations',
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose logging',
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Load data
    logger.info("Loading data...")
    predictions = np.load(args.predictions)
    
    if args.uncertainties:
        uncertainties = np.load(args.uncertainties)
    else:
        logger.error("--uncertainties required")
        return
    
    targets = None
    if args.targets:
        targets = np.load(args.targets)
        errors = (predictions != targets).astype(int)
        logger.info(f"Error rate: {errors.mean():.2%}")
    
    # Save results
    logger.info("Saving results...")
    save_uncertainties(uncertainties, predictions, targets, args.output_dir)
    
    # Generate report
    if args.report:
        logger.info("Generating report...")
        report = generate_uncertainty_report(
            uncertainties, predictions, targets,
            output_path=Path(args.output_dir) / 'report.txt'
        )
        print(report)
    
    # Create visualizations
    if args.visualize:
        logger.info("Creating visualizations...")
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            axes[0].hist(uncertainties, bins=50, edgecolor='black')
            axes[0].set_xlabel('Uncertainty')
            axes[0].set_ylabel('Count')
            axes[0].set_title('Uncertainty Distribution')
            axes[0].grid(True, alpha=0.3)
            
            if targets is not None:
                correct_unc = uncertainties[errors == 0]
                error_unc = uncertainties[errors == 1]
                
                axes[1].hist(correct_unc, bins=30, alpha=0.6, label='Correct')
                axes[1].hist(error_unc, bins=30, alpha=0.6, label='Errors')
                axes[1].set_xlabel('Uncertainty')
                axes[1].set_ylabel('Count')
                axes[1].set_title('Uncertainty by Prediction Status')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            viz_path = Path(args.output_dir) / 'uncertainty_distribution.png'
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {viz_path}")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not create visualizations: {e}")
    
    logger.info("Uncertainty estimation complete!")


if __name__ == '__main__':
    main()

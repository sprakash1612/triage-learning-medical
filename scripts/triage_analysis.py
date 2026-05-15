#!/usr/bin/env python3
"""
Analyze triage system performance and optimize deferral strategies.

Evaluates different deferral strategies, optimizes thresholds,
simulates various human accuracy levels, and generates comprehensive
triage performance reports.
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_data(
    predictions_path: str,
    uncertainties_path: str,
    targets_path: str,
) -> tuple:
    """Load predictions, uncertainties, and targets."""
    predictions = np.load(predictions_path)
    uncertainties = np.load(uncertainties_path)
    targets = np.load(targets_path)
    
    if not (len(predictions) == len(uncertainties) == len(targets)):
        raise ValueError("All arrays must have same length")
    
    return predictions, uncertainties, targets


def evaluate_threshold(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    human_accuracy: float = 0.95,
) -> Dict[str, float]:
    """
    Evaluate triage system at specific threshold.
    
    Args:
        predictions: (n_samples,) Model predictions
        uncertainties: (n_samples,) Uncertainty estimates
        targets: (n_samples,) Ground truth labels
        threshold: Uncertainty threshold for deferral
        human_accuracy: Human accuracy (0-1)
    
    Returns:
        Dictionary with performance metrics
    """
    defer_mask = uncertainties > threshold
    deferral_rate = defer_mask.mean()
    automation_rate = 1 - deferral_rate
    
    # AI accuracy on non-deferred
    ai_mask = ~defer_mask
    if ai_mask.sum() > 0:
        ai_accuracy = (predictions[ai_mask] == targets[ai_mask]).mean()
        ai_errors = (predictions[ai_mask] != targets[ai_mask]).sum()
    else:
        ai_accuracy = np.nan
        ai_errors = 0
    
    # System accuracy (AI on non-deferred + human on deferred)
    system_predictions = predictions.copy()
    if defer_mask.sum() > 0:
        # Simulate human predictions
        np.random.seed(42)  # For reproducibility
        human_correct = np.random.rand(defer_mask.sum()) < human_accuracy
        
        # For correct predictions, keep true labels; for errors, pick random
        human_predictions = np.zeros(defer_mask.sum(), dtype=int)
        for i, correct in enumerate(human_correct):
            if correct:
                # Find index of this deferred sample
                deferred_idx = np.where(defer_mask)[0][i]
                human_predictions[i] = targets[deferred_idx]
            else:
                # Random wrong prediction
                n_classes = len(np.unique(targets))
                human_predictions[i] = np.random.randint(0, n_classes)
        
        system_predictions[defer_mask] = human_predictions
    
    system_accuracy = (system_predictions == targets).mean()
    
    # Cost analysis
    ai_error_cost = ai_errors * 100  # Cost of AI error
    human_review_cost = defer_mask.sum() * 1  # Cost of human review
    total_cost = ai_error_cost + human_review_cost
    
    return {
        'threshold': float(threshold),
        'deferral_rate': float(deferral_rate),
        'automation_rate': float(automation_rate),
        'ai_accuracy': float(ai_accuracy),
        'system_accuracy': float(system_accuracy),
        'human_accuracy': float(human_accuracy),
        'ai_errors': int(ai_errors),
        'total_cost': float(total_cost),
        'cost_ai_errors': float(ai_error_cost),
        'cost_human_reviews': float(human_review_cost),
    }


def sweep_thresholds(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    targets: np.ndarray,
    num_points: int = 20,
    human_accuracy: float = 0.95,
) -> List[Dict[str, float]]:
    """
    Evaluate triage system across uncertainty thresholds.
    
    Args:
        predictions: Model predictions
        uncertainties: Uncertainty estimates
        targets: Ground truth labels
        num_points: Number of thresholds to evaluate
        human_accuracy: Human accuracy
    
    Returns:
        List of evaluation results
    """
    thresholds = np.linspace(
        uncertainties.min(),
        uncertainties.max(),
        num_points,
    )
    
    results = []
    for threshold in thresholds:
        result = evaluate_threshold(
            predictions, uncertainties, targets,
            threshold, human_accuracy
        )
        results.append(result)
    
    return results


def find_optimal_threshold(
    results: List[Dict[str, float]],
    objective: str = 'system_accuracy',
) -> Dict[str, float]:
    """
    Find optimal threshold for given objective.
    
    Args:
        results: List of evaluation results
        objective: One of 'system_accuracy', 'cost_total', 'automation_rate'
    
    Returns:
        Best result dictionary
    """
    if objective == 'system_accuracy':
        values = [r['system_accuracy'] for r in results]
        best_idx = np.nanargmax(values)
    elif objective == 'cost_total':
        values = [r['total_cost'] for r in results]
        best_idx = np.nanargmin(values)
    elif objective == 'automation_rate':
        values = [r['automation_rate'] for r in results]
        best_idx = np.nanargmax(values)
    else:
        raise ValueError(f"Unknown objective: {objective}")
    
    return results[best_idx]


def compare_human_accuracies(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    targets: np.ndarray,
    optimal_threshold: float,
    human_accuracies: List[float] = None,
) -> Dict[float, Dict[str, float]]:
    """
    Evaluate system at different human accuracy levels.
    
    Args:
        predictions: Model predictions
        uncertainties: Uncertainty estimates
        targets: Ground truth labels
        optimal_threshold: Optimal uncertainty threshold
        human_accuracies: List of human accuracy values (default: [0.8, 0.9, 0.95, 0.99])
    
    Returns:
        Dictionary mapping human_accuracy → results
    """
    if human_accuracies is None:
        human_accuracies = [0.8, 0.9, 0.95, 0.99]
    
    comparison = {}
    for human_acc in human_accuracies:
        result = evaluate_threshold(
            predictions, uncertainties, targets,
            optimal_threshold, human_acc
        )
        comparison[human_acc] = result
    
    return comparison


def generate_triage_report(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    targets: np.ndarray,
    results: List[Dict[str, float]],
    comparison: Dict[float, Dict[str, float]],
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate comprehensive triage analysis report.
    
    Args:
        predictions: Model predictions
        uncertainties: Uncertainty estimates
        targets: Ground truth labels
        results: Threshold sweep results
        comparison: Human accuracy comparison results
        output_path: Optional path to save report
    
    Returns:
        Report text
    """
    baseline_accuracy = (predictions == targets).mean()
    
    best_system_acc = max(results, key=lambda x: x['system_accuracy'])
    best_cost = min(results, key=lambda x: x['total_cost'])
    
    report = f"""
TRIAGE SYSTEM ANALYSIS REPORT
=============================

BASELINE PERFORMANCE
--------------------
Model Accuracy (no triage): {baseline_accuracy:.4f}
Total Samples: {len(targets)}

OPTIMAL THRESHOLD ANALYSIS
--------------------------
Best for System Accuracy:
  Threshold: {best_system_acc['threshold']:.4f}
  Deferral Rate: {best_system_acc['deferral_rate']:.1%}
  AI Accuracy: {best_system_acc['ai_accuracy']:.4f}
  System Accuracy: {best_system_acc['system_accuracy']:.4f}
  Improvement over baseline: {best_system_acc['system_accuracy'] - baseline_accuracy:.4f}

Best for Cost:
  Threshold: {best_cost['threshold']:.4f}
  Deferral Rate: {best_cost['deferral_rate']:.1%}
  Total Cost: ${best_cost['total_cost']:,.2f}
  Cost saved vs. full AI: ${len(targets) * 100 - best_cost['total_cost']:,.2f}

HUMAN ACCURACY SENSITIVITY ANALYSIS
------------------------------------
"""
    
    for human_acc in sorted(comparison.keys()):
        result = comparison[human_acc]
        report += f"\nHuman Accuracy: {human_acc:.0%}\n"
        report += f"  System Accuracy: {result['system_accuracy']:.4f}\n"
        report += f"  Cost: ${result['total_cost']:,.2f}\n"
        report += f"  Improvement: {result['system_accuracy'] - baseline_accuracy:.4f}\n"
    
    report += """

INTERPRETATION
--------------
1. If system_accuracy > baseline: Triage is beneficial
2. Higher human_accuracy → higher system_accuracy (expected)
3. Optimal threshold balances automation rate and accuracy
4. Cost analysis considers both AI errors and human review effort

Key Recommendations:
- Choose threshold balancing accuracy and automation rate
- Monitor actual human accuracy in deployment
- Periodically retrain model to improve AI performance
"""
    
    if output_path:
        Path(output_path).write_text(report)
        logger.info(f"Report saved to {output_path}")
    
    return report


def save_results(
    results: List[Dict[str, float]],
    comparison: Dict[float, Dict[str, float]],
    output_dir: Path,
) -> None:
    """
    Save triage analysis results.
    
    Args:
        results: Threshold sweep results
        comparison: Human accuracy comparison
        output_dir: Directory to save files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save threshold sweep
    with open(output_dir / 'threshold_sweep.json', 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved threshold sweep to {output_dir / 'threshold_sweep.json'}")
    
    # Save human accuracy comparison
    with open(output_dir / 'human_accuracy_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"Saved comparison to {output_dir / 'human_accuracy_comparison.json'}")


def create_visualizations(
    results: List[Dict[str, float]],
    comparison: Dict[float, Dict[str, float]],
    output_dir: Path,
) -> None:
    """
    Create triage performance visualizations.
    
    Args:
        results: Threshold sweep results
        comparison: Human accuracy comparison
        output_dir: Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract data
        thresholds = [r['threshold'] for r in results]
        system_accs = [r['system_accuracy'] for r in results]
        automation_rates = [r['automation_rate'] for r in results]
        costs = [r['total_cost'] for r in results]
        
        # Plot 1: Automation vs. Accuracy
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(automation_rates, system_accs, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Automation Rate', fontsize=11)
        ax.set_ylabel('System Accuracy', fontsize=11)
        ax.set_title('Triage Performance: Automation vs. Accuracy', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        
        fig.tight_layout()
        fig.savefig(output_dir / 'automation_vs_accuracy.png', dpi=300, bbox_inches='tight')
        logger.info("Saved automation vs. accuracy plot")
        plt.close()
        
        # Plot 2: Cost Analysis
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(thresholds, costs, 's-', linewidth=2, markersize=6)
        ax.set_xlabel('Uncertainty Threshold', fontsize=11)
        ax.set_ylabel('Total Cost ($)', fontsize=11)
        ax.set_title('Triage Cost Analysis', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(output_dir / 'cost_analysis.png', dpi=300, bbox_inches='tight')
        logger.info("Saved cost analysis plot")
        plt.close()
        
        # Plot 3: Human Accuracy Sensitivity
        fig, ax = plt.subplots(figsize=(10, 6))
        human_accs = sorted(comparison.keys())
        system_accs_by_human = [comparison[h]['system_accuracy'] for h in human_accs]
        
        ax.plot(human_accs, system_accs_by_human, 'D-', linewidth=2, markersize=8)
        ax.set_xlabel('Human Accuracy', fontsize=11)
        ax.set_ylabel('System Accuracy', fontsize=11)
        ax.set_title('Human Accuracy Sensitivity Analysis', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.7, 1.0])
        
        fig.tight_layout()
        fig.savefig(output_dir / 'human_accuracy_sensitivity.png', dpi=300, bbox_inches='tight')
        logger.info("Saved human accuracy sensitivity plot")
        plt.close()
        
    except ImportError:
        logger.warning("Matplotlib not available, skipping visualizations")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze triage system performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic triage analysis
  python triage_analysis.py --predictions preds.npy --uncertainties unc.npy --targets targets.npy
  
  # With visualizations and report
  python triage_analysis.py --predictions preds.npy --uncertainties unc.npy --targets targets.npy \\
    --report --visualize --output-dir results/triage
  
  # Test multiple human accuracy levels
  python triage_analysis.py --predictions preds.npy --uncertainties unc.npy --targets targets.npy \\
    --human-accuracies 0.8 0.9 0.95 0.99
        """,
    )
    
    parser.add_argument(
        '--predictions',
        required=True,
        help='Path to model predictions (n_samples,)',
    )
    parser.add_argument(
        '--uncertainties',
        required=True,
        help='Path to uncertainty estimates (n_samples,)',
    )
    parser.add_argument(
        '--targets',
        required=True,
        help='Path to ground truth labels (n_samples,)',
    )
    parser.add_argument(
        '--human-accuracy',
        type=float,
        default=0.95,
        help='Default human accuracy for threshold sweep (default: 0.95)',
    )
    parser.add_argument(
        '--human-accuracies',
        type=float,
        nargs='+',
        help='Multiple human accuracy levels to compare',
    )
    parser.add_argument(
        '--num-thresholds',
        type=int,
        default=20,
        help='Number of thresholds to evaluate',
    )
    parser.add_argument(
        '--output-dir',
        default='results/triage',
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
        help='Create visualizations',
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
    predictions, uncertainties, targets = load_data(
        args.predictions, args.uncertainties, args.targets
    )
    logger.info(f"Loaded {len(predictions)} samples")
    
    # Sweep thresholds
    logger.info("Sweeping uncertainty thresholds...")
    results = sweep_thresholds(
        predictions, uncertainties, targets,
        num_points=args.num_thresholds,
        human_accuracy=args.human_accuracy,
    )
    
    # Compare human accuracies
    optimal_result = max(results, key=lambda x: x['system_accuracy'])
    comparison = compare_human_accuracies(
        predictions, uncertainties, targets,
        optimal_result['threshold'],
        human_accuracies=args.human_accuracies,
    )
    
    # Save results
    logger.info("Saving results...")
    save_results(results, comparison, args.output_dir)
    
    # Generate report
    if args.report:
        logger.info("Generating report...")
        report = generate_triage_report(
            predictions, uncertainties, targets,
            results, comparison,
            output_path=Path(args.output_dir) / 'triage_report.txt'
        )
        print(report)
    
    # Create visualizations
    if args.visualize:
        logger.info("Creating visualizations...")
        create_visualizations(results, comparison, args.output_dir)
    
    logger.info("Triage analysis complete!")


if __name__ == '__main__':
    main()

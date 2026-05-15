"""
Uncertainty quantification evaluation metrics.

Metrics for assessing how well uncertainty estimates predict model errors
and their utility for rejection/selective prediction.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
import logging
from sklearn.metrics import auc, roc_curve, roc_auc_score

logger = logging.getLogger(__name__)


class UncertaintyMetrics:
    """Static methods for uncertainty quality evaluation."""
    
    @staticmethod
    def compute_uncertainty_quality(
        uncertainties: np.ndarray,
        errors: np.ndarray,
    ) -> Dict[str, float]:
        """
        Measure how well uncertainty estimates predict errors.
        
        Ranks samples by uncertainty and checks if high uncertainty correlates
        with incorrect predictions. Metrics include:
        - AUROC: Area under ROC curve (higher is better)
        - Spearman correlation: Rank correlation between uncertainty and error
        
        Args:
            uncertainties: (n_samples,) Uncertainty values (0-1 or 0-max)
            errors: (n_samples,) Binary error indicators (1=error, 0=correct)
        
        Returns:
            Dictionary with AUROC, Spearman correlation, and other metrics
        
        Example:
            >>> uncertainties = np.array([0.1, 0.5, 0.9])
            >>> errors = np.array([0, 0, 1])
            >>> metrics = UncertaintyMetrics.compute_uncertainty_quality(
            ...     uncertainties, errors
            ... )
            >>> print(metrics['auroc'])  # Should be > 0.5 if uncertainty is useful
        """
        from scipy.stats import spearmanr
        
        uncertainties = np.asarray(uncertainties, dtype=np.float32)
        errors = np.asarray(errors, dtype=np.int32)
        
        if len(uncertainties) != len(errors):
            raise ValueError("uncertainties and errors must have same length")
        
        if len(np.unique(errors)) == 1:
            logger.warning(
                "No errors in batch - cannot compute uncertainty quality metrics"
            )
            return {
                'auroc': np.nan,
                'spearman_corr': np.nan,
                'spearman_pval': np.nan,
            }
        
        # AUROC: Can uncertainty distinguish errors from correct predictions?
        try:
            auroc = roc_auc_score(errors, uncertainties)
        except ValueError:
            auroc = np.nan
        
        # Spearman rank correlation
        corr, pval = spearmanr(uncertainties, errors)
        
        return {
            'auroc': float(auroc),
            'spearman_corr': float(corr),
            'spearman_pval': float(pval),
        }
    
    @staticmethod
    def compute_rejection_curve(
        uncertainties: np.ndarray,
        errors: np.ndarray,
        num_points: int = 50,
    ) -> Dict[str, np.ndarray]:
        """
        Rejection curve: Accuracy vs. rejection rate.
        
        By rejecting (deferring) samples with highest uncertainty, measure
        how accuracy improves. This quantifies the value of uncertainty for
        selective prediction.
        
        Args:
            uncertainties: (n_samples,) Uncertainty values
            errors: (n_samples,) Binary error indicators
            num_points: Number of rejection thresholds to evaluate
        
        Returns:
            Dictionary with:
            - rejection_rates: (num_points,) Proportion of samples rejected
            - accuracies: (num_points,) System accuracy at each rejection rate
            - ideal_accuracies: Perfect rejection curve (oracle)
            - auc_rejection: Area under rejection curve
        
        Example:
            >>> rejection_results = UncertaintyMetrics.compute_rejection_curve(
            ...     uncertainties, errors, num_points=100
            ... )
            >>> plt.plot(
            ...     rejection_results['rejection_rates'],
            ...     rejection_results['accuracies']
            ... )
        """
        uncertainties = np.asarray(uncertainties, dtype=np.float32)
        errors = np.asarray(errors, dtype=np.int32)
        correct = 1 - errors
        
        # Thresholds from max uncertainty down (high uncertainty → rejection)
        thresholds = np.linspace(
            uncertainties.min(), uncertainties.max(), num_points
        )
        
        rejection_rates = []
        accuracies = []
        
        for threshold in thresholds:
            # Reject samples with uncertainty > threshold
            rejected_mask = uncertainties > threshold
            rejection_rate = rejected_mask.sum() / len(uncertainties)
            
            if rejection_rate == 1.0:
                # All rejected - no system accuracy to measure
                accuracy = np.nan
            elif rejection_rate == 0.0:
                # No rejections - baseline accuracy
                accuracy = correct.mean()
            else:
                # Accuracy on non-rejected samples
                kept_mask = ~rejected_mask
                if kept_mask.sum() > 0:
                    accuracy = correct[kept_mask].mean()
                else:
                    accuracy = np.nan
            
            rejection_rates.append(rejection_rate)
            accuracies.append(accuracy)
        
        rejection_rates = np.array(rejection_rates)
        accuracies = np.array(accuracies)
        
        # Ideal rejection curve: reject errors first
        sorted_indices = np.argsort(errors)[::-1]  # Errors first
        ideal_accuracies = np.cumsum(1 - errors[sorted_indices]) / np.arange(1, len(errors) + 1)
        ideal_accuracies[0] = 1.0  # All rejected = 100% accuracy
        
        # AUC of rejection curve
        valid_mask = ~np.isnan(accuracies)
        if valid_mask.sum() > 1:
            auc_rejection = auc(rejection_rates[valid_mask], accuracies[valid_mask])
        else:
            auc_rejection = np.nan
        
        return {
            'rejection_rates': rejection_rates,
            'accuracies': accuracies,
            'ideal_accuracies': ideal_accuracies,
            'auc_rejection': float(auc_rejection),
            'thresholds': thresholds,
        }
    
    @staticmethod
    def compute_risk_coverage_curve(
        uncertainties: np.ndarray,
        errors: np.ndarray,
        num_points: int = 50,
    ) -> Dict[str, np.ndarray]:
        """
        Risk-coverage curve: Selective prediction trade-off.
        
        Coverage = fraction of dataset the system processes
        Risk = error rate on processed samples
        
        As uncertainty threshold increases, fewer samples are processed (lower
        coverage) but errors on processed samples decrease (lower risk).
        
        Args:
            uncertainties: (n_samples,) Uncertainty values
            errors: (n_samples,) Binary error indicators
            num_points: Number of thresholds to evaluate
        
        Returns:
            Dictionary with:
            - coverage: Fraction of dataset processed at each threshold
            - risk: Error rate on processed samples
            - thresholds: Uncertainty thresholds used
            - optimal_coverage: Coverage at 50% risk reduction
        
        Example:
            >>> risk_results = UncertaintyMetrics.compute_risk_coverage_curve(
            ...     uncertainties, errors
            ... )
            >>> print(f"At 90% coverage: {risk_results['risk'][...]}% error")
        """
        uncertainties = np.asarray(uncertainties, dtype=np.float32)
        errors = np.asarray(errors, dtype=np.int32)
        
        # High uncertainty → low confidence → should be rejected
        thresholds = np.linspace(
            uncertainties.min(), uncertainties.max(), num_points
        )
        
        coverage = []
        risk = []
        
        for threshold in thresholds:
            # Accept samples with uncertainty <= threshold
            accepted_mask = uncertainties <= threshold
            cov = accepted_mask.sum() / len(uncertainties)
            
            if cov == 0:
                # No samples accepted
                r = np.nan
            else:
                # Risk (error rate) on accepted samples
                r = errors[accepted_mask].mean()
            
            coverage.append(cov)
            risk.append(r)
        
        coverage = np.array(coverage)
        risk = np.array(risk)
        
        # Optimal coverage: threshold where risk reduced by 50%
        baseline_risk = errors.mean()
        target_risk = baseline_risk * 0.5
        risk_diff = np.abs(risk - target_risk)
        optimal_idx = np.nanargmin(risk_diff)
        optimal_coverage = coverage[optimal_idx] if ~np.isnan(risk[optimal_idx]) else np.nan
        
        return {
            'coverage': coverage,
            'risk': risk,
            'thresholds': thresholds,
            'baseline_risk': float(baseline_risk),
            'optimal_coverage': float(optimal_coverage),
        }
    
    @staticmethod
    def compute_selective_prediction_metrics(
        uncertainties: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray,
        percentiles: List[int] = [90, 95, 99],
    ) -> Dict[str, Dict[str, float]]:
        """
        Selective prediction: Retain only confident predictions.
        
        Accepts only top-percentile most confident samples and measures
        accuracy on the accepted set.
        
        Args:
            uncertainties: (n_samples,) Uncertainty values
            predictions: (n_samples,) Model predictions
            targets: (n_samples,) Ground truth labels
            percentiles: List of percentiles to evaluate (e.g., [90, 95, 99])
        
        Returns:
            Dictionary mapping percentile → metrics dict with:
            - accuracy: Accuracy on selected samples
            - coverage: Fraction of samples selected
            - num_selected: Number of samples retained
        
        Example:
            >>> metrics = UncertaintyMetrics.compute_selective_prediction_metrics(
            ...     uncertainties, predictions, targets
            ... )
            >>> print(f"At 95th percentile: {metrics[95]['accuracy']:.3f} accuracy")
        """
        uncertainties = np.asarray(uncertainties, dtype=np.float32)
        predictions = np.asarray(predictions, dtype=np.int32)
        targets = np.asarray(targets, dtype=np.int32)
        
        if not (len(uncertainties) == len(predictions) == len(targets)):
            raise ValueError("All inputs must have same length")
        
        correct = (predictions == targets).astype(np.int32)
        results = {}
        
        for percentile in percentiles:
            # Threshold for keeping top (100-percentile)% confident
            threshold = np.percentile(uncertainties, percentile)
            
            # Keep samples with uncertainty <= threshold (more confident)
            keep_mask = uncertainties <= threshold
            num_selected = keep_mask.sum()
            coverage = num_selected / len(uncertainties)
            
            if num_selected > 0:
                accuracy = correct[keep_mask].mean()
            else:
                accuracy = np.nan
            
            results[percentile] = {
                'accuracy': float(accuracy),
                'coverage': float(coverage),
                'num_selected': int(num_selected),
                'threshold': float(threshold),
            }
        
        return results
    
    @staticmethod
    def compute_confidence_based_metrics(
        confidences: np.ndarray,
        errors: np.ndarray,
        num_bins: int = 10,
    ) -> Dict[str, np.ndarray]:
        """
        Metrics based on model confidence (softmax max).
        
        Confidence (max softmax) is inverse of uncertainty. Measure how
        well confidence predicts correctness.
        
        Args:
            confidences: (n_samples,) Softmax max values (0-1)
            errors: (n_samples,) Binary error indicators
            num_bins: Number of confidence bins
        
        Returns:
            Dictionary with bin-wise statistics
        """
        confidences = np.asarray(confidences, dtype=np.float32)
        errors = np.asarray(errors, dtype=np.int32)
        
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        accuracies = []
        counts = []
        
        for i in range(num_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if mask.sum() > 0:
                accuracy = 1 - errors[mask].mean()  # Correct rate
                accuracies.append(accuracy)
                counts.append(mask.sum())
            else:
                accuracies.append(np.nan)
                counts.append(0)
        
        return {
            'bin_centers': bin_centers,
            'accuracies': np.array(accuracies),
            'counts': np.array(counts),
            'ece': float(np.nanmean(np.abs(np.array(accuracies) - bin_centers))),
        }

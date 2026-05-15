"""
Metrics for evaluating human-AI collaboration and triage systems.

Implements various performance metrics specific to triage and collaboration scenarios.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class CollaborationMetrics:
    """
    Compute metrics for human-AI collaboration evaluation.
    """
    
    @staticmethod
    def compute_system_accuracy(
        ai_predictions: np.ndarray,
        human_predictions: np.ndarray,
        defer_decisions: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Compute overall triage system accuracy.
        
        Uses AI predictions for non-deferred samples and human predictions for deferred.
        
        Args:
            ai_predictions: AI model predictions (N,)
            human_predictions: Human expert predictions (N,)
            defer_decisions: Binary deferral decisions (1=defer, 0=use AI) (N,)
            labels: Ground truth labels (N,)
        
        Returns:
            System accuracy (0 to 1)
        """
        system_predictions = np.where(
            defer_decisions == 1,
            human_predictions,
            ai_predictions
        )
        
        accuracy = (system_predictions == labels).mean()
        return accuracy
    
    @staticmethod
    def compute_automation_efficiency(
        ai_predictions: np.ndarray,
        human_predictions: np.ndarray,
        defer_decisions: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute automation efficiency metrics.
        
        Measures how much the system automates while maintaining accuracy.
        
        Args:
            ai_predictions: AI model predictions (N,)
            human_predictions: Human expert predictions (N,)
            defer_decisions: Binary deferral decisions (N,)
            labels: Ground truth labels (N,)
        
        Returns:
            Dictionary with efficiency metrics:
            - 'automation_rate': Fraction of samples handled by AI
            - 'ai_accuracy': Accuracy on automated samples
            - 'human_accuracy': Accuracy on deferred samples
            - 'system_accuracy': Overall system accuracy
        """
        ai_mask = defer_decisions == 0
        human_mask = defer_decisions == 1
        
        automation_rate = ai_mask.mean()
        ai_accuracy = (ai_predictions[ai_mask] == labels[ai_mask]).mean() if ai_mask.sum() > 0 else 0
        human_accuracy = (human_predictions[human_mask] == labels[human_mask]).mean() if human_mask.sum() > 0 else 0
        
        system_accuracy = CollaborationMetrics.compute_system_accuracy(
            ai_predictions, human_predictions, defer_decisions, labels
        )
        
        return {
            "automation_rate": automation_rate,
            "ai_accuracy": ai_accuracy,
            "human_accuracy": human_accuracy,
            "system_accuracy": system_accuracy,
        }
    
    @staticmethod
    def compute_workload_reduction(
        defer_decisions: np.ndarray,
        baseline_workload: int = 100,
    ) -> float:
        """
        Compute human workload reduction.
        
        Args:
            defer_decisions: Binary deferral decisions (N,)
            baseline_workload: Total number of samples to review (if all deferred)
        
        Returns:
            Workload reduction percentage (0 to 100)
        """
        deferred_count = (defer_decisions == 1).sum()
        reduction = (1 - deferred_count / len(defer_decisions)) * 100
        return reduction
    
    @staticmethod
    def compute_safety_metrics(
        ai_predictions: np.ndarray,
        defer_decisions: np.ndarray,
        labels: np.ndarray,
        critical_classes: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Compute safety-related metrics for critical healthcare scenarios.
        
        Args:
            ai_predictions: AI model predictions (N,)
            defer_decisions: Binary deferral decisions (N,)
            labels: Ground truth labels (N~,)
            critical_classes: List of critical class indices that must be detected
        
        Returns:
            Dictionary with safety metrics:
            - 'critical_deferral_rate': Fraction of critical cases deferred
            - 'false_negative_rate': Critical cases misclassified by AI
            - 'false_positive_rate': Non-critical classified as critical by AI
        """
        if critical_classes is None:
            critical_classes = []
        
        is_critical = np.isin(labels, critical_classes)
        
        # Deferral rate for critical cases
        critical_deferred = (defer_decisions[is_critical] == 1).sum() if is_critical.sum() > 0 else 0
        critical_deferral_rate = critical_deferred / is_critical.sum() if is_critical.sum() > 0 else 0
        
        # False negatives (critical not detected by AI when not deferred)
        ai_handles = defer_decisions == 0
        ai_critical_errors = (
            (ai_predictions[ai_handles & is_critical] != labels[ai_handles & is_critical]).sum()
            / is_critical.sum() if is_critical.sum() > 0 else 0
        )
        
        # False positives
        non_critical = ~is_critical
        ai_non_critical_errors = (
            (ai_predictions[ai_handles & non_critical] != labels[ai_handles & non_critical]).sum()
            / non_critical.sum() if non_critical.sum() > 0 else 0
        )
        
        return {
            "critical_deferral_rate": critical_deferral_rate,
            "critical_misclassification_rate": ai_critical_errors,
            "non_critical_misclassification_rate": ai_non_critical_errors,
        }
    
    @staticmethod
    def compute_cost_benefit_analysis(
        ai_predictions: np.ndarray,
        human_predictions: np.ndarray,
        defer_decisions: np.ndarray,
        labels: np.ndarray,
        ai_error_cost: float = 100.0,
        human_review_cost: float = 10.0,
    ) -> Dict[str, float]:
        """
        Compute cost-benefit analysis of triage system.
        
        Args:
            ai_predictions: AI model predictions (N,)
            human_predictions: Human expert predictions (N,)
            defer_decisions: Binary deferral decisions (N,)
            labels: Ground truth labels (N,)
            ai_error_cost: Cost of AI error
            human_review_cost: Cost of human review
        
        Returns:
            Dictionary with cost metrics:
            - 'ai_error_cost': Total cost of AI errors
            - 'human_review_cost': Total cost of human reviews
            - 'total_cost': Total system cost
            - 'cost_per_sample': Average cost per sample
            - 'cost_savings': Cost saved vs. reviewing everything
        """
        ai_mask = defer_decisions == 0
        human_mask = defer_decisions == 1
        
        # AI errors on automated samples
        ai_errors = (ai_predictions[ai_mask] != labels[ai_mask]).sum() if ai_mask.sum() > 0 else 0
        
        # Human reviews
        human_reviews = human_mask.sum()
        
        ai_error_total = ai_errors * ai_error_cost
        human_review_total = human_reviews * human_review_cost
        total_cost = ai_error_total + human_review_total
        
        # Cost if everything was reviewed by human
        baseline_cost = len(labels) * human_review_cost
        cost_savings = baseline_cost - total_cost
        
        return {
            "ai_error_cost": ai_error_total,
            "human_review_cost": human_review_total,
            "total_cost": total_cost,
            "cost_per_sample": total_cost / len(labels),
            "cost_savings": cost_savings,
            "cost_savings_percentage": (cost_savings / baseline_cost) * 100,
        }
    
    @staticmethod
    def compute_trade_off_metrics(
        ai_predictions: np.ndarray,
        uncertainties: np.ndarray,
        labels: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute trade-off metrics across different uncertainty thresholds.
        
        Useful for understanding automation vs. accuracy trade-offs.
        
        Args:
            ai_predictions: AI model predictions (N,)
            uncertainties: Uncertainty estimates (N,)
            labels: Ground truth labels (N,)
            thresholds: Uncertainty thresholds to evaluate. 
                       If None, uses percentiles.
        
        Returns:
            Dictionary with arrays for each metric across thresholds
        """
        if thresholds is None:
            thresholds = np.percentile(uncertainties, np.arange(0, 101, 5))
        
        automation_rates = []
        accuracies = []
        
        for threshold in thresholds:
            defer_mask = uncertainties > threshold
            ai_mask = ~defer_mask
            
            automation_rate = ai_mask.mean()
            automation_rates.append(automation_rate)
            
            if ai_mask.sum() > 0:
                accuracy = (ai_predictions[ai_mask] == labels[ai_mask]).mean()
            else:
                accuracy = 1.0  # All deferred, perfect accuracy
            accuracies.append(accuracy)
        
        return {
            "thresholds": thresholds,
            "automation_rates": np.array(automation_rates),
            "accuracies": np.array(accuracies),
        }


def plot_performance_curves(
    metrics_dict: Dict[str, np.ndarray],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot automation rate vs. accuracy curves.
    
    Args:
        metrics_dict: Dictionary from compute_trade_off_metrics
        save_path: Path to save figure (optional)
    
    Returns:
        Matplotlib figure object
    """
    automation_rates = metrics_dict["automation_rates"]
    accuracies = metrics_dict["accuracies"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(automation_rates * 100, accuracies * 100, 'o-', 
           color='steelblue', markersize=8, linewidth=2)
    
    ax.set_xlabel('Automation Rate (%)', fontsize=12)
    ax.set_ylabel('System Accuracy (%)', fontsize=12)
    ax.set_title('Automation Rate vs. System Accuracy Trade-off', 
                fontsize=12, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Performance curves saved to {save_path}")
    
    return fig

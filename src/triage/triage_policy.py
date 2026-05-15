"""
Triage policy for deciding when to defer to human experts
"""

import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.metrics import roc_curve, auc
import logging

logger = logging.getLogger(__name__)


class TriagePolicy:
    """
    Triage policy for AI-Human collaboration
    
    Args:
        threshold: Uncertainty threshold for deferral
        uncertainty_metric: Type of uncertainty ('entropy', 'confidence', 'margin')
        optimize: Whether to optimize threshold on validation set
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        uncertainty_metric: str = 'entropy',
        optimize: bool = True
    ):
        self.threshold = threshold
        self.uncertainty_metric = uncertainty_metric
        self.optimize = optimize
        self.optimal_threshold = threshold
        
    def make_decisions(
        self,
        uncertainties: np.ndarray,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Make deferral decisions based on uncertainty
        
        Args:
            uncertainties: Uncertainty scores (N,)
            threshold: Override default threshold
        
        Returns:
            decisions: Binary decisions (0=AI, 1=Human) (N,)
        """
        thresh = threshold if threshold is not None else self.optimal_threshold
        decisions = (uncertainties > thresh).astype(int)
        return decisions
    
    def optimize_threshold(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        uncertainties: np.ndarray,
        human_accuracy: float = 0.95,
        target_automation_rate: Optional[float] = None
    ) -> float:
        """
        Optimize threshold to maximize system accuracy
        
        Args:
            predictions: Model predictions (N,)
            true_labels: Ground truth labels (N,)
            uncertainties: Uncertainty scores (N,)
            human_accuracy: Simulated human expert accuracy
            target_automation_rate: Target fraction of cases handled by AI
        
        Returns:
            optimal_threshold: Optimized threshold value
        """
        # Generate candidate thresholds
        thresholds = np.percentile(uncertainties, np.linspace(0, 100, 101))
        
        best_accuracy = 0
        best_threshold = self.threshold
        
        for thresh in thresholds:
            decisions = self.make_decisions(uncertainties, thresh)
            
            # Compute system accuracy
            ai_correct = (predictions == true_labels) & (decisions == 0)
            
            # Simulate human predictions on deferred cases
            human_cases = decisions == 1
            human_correct = np.random.rand(len(true_labels)) < human_accuracy
            human_correct = human_correct & human_cases
            
            total_correct = ai_correct.sum() + human_correct.sum()
            accuracy = total_correct / len(true_labels)
            
            # Check automation rate constraint
            automation_rate = 1 - decisions.mean()
            if target_automation_rate is not None:
                if automation_rate < target_automation_rate:
                    continue
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = thresh
        
        self.optimal_threshold = best_threshold
        logger.info(f"Optimized threshold: {best_threshold:.4f}")
        logger.info(f"Expected system accuracy: {best_accuracy:.4f}")
        
        return best_threshold
    
    def compute_metrics(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        decisions: np.ndarray,
        human_accuracy: float = 0.95
    ) -> Dict[str, float]:
        """
        Compute triage system metrics
        
        Args:
            predictions: Model predictions (N,)
            true_labels: Ground truth labels (N,)
            decisions: Deferral decisions (N,)
            human_accuracy: Simulated human accuracy
        
        Returns:
            metrics: Dictionary of metric values
        """
        # AI performance
        ai_cases = decisions == 0
        ai_accuracy = (predictions[ai_cases] == true_labels[ai_cases]).mean() if ai_cases.sum() > 0 else 0
        
        # Human performance (simulated)
        human_cases = decisions == 1
        human_predictions = true_labels[human_cases].copy()
        # Simulate errors
        error_mask = np.random.rand(human_cases.sum()) > human_accuracy
        if error_mask.sum() > 0:
            # Random incorrect predictions
            num_classes = len(np.unique(true_labels))
            wrong_labels = np.random.randint(0, num_classes, error_mask.sum())
            human_predictions[error_mask] = wrong_labels
        
        human_acc = (human_predictions == true_labels[human_cases]).mean() if human_cases.sum() > 0 else 0
        
        # System performance
        all_predictions = predictions.copy()
        all_predictions[human_cases] = human_predictions
        system_accuracy = (all_predictions == true_labels).mean()
        
        # Automation rate
        automation_rate = ai_cases.mean()
        
        # Performance gain over full AI
        full_ai_accuracy = (predictions == true_labels).mean()
        performance_gain = system_accuracy - full_ai_accuracy
        
        metrics = {
            'ai_accuracy': float(ai_accuracy),
            'human_accuracy': float(human_acc),
            'system_accuracy': float(system_accuracy),
            'automation_rate': float(automation_rate),
            'deferral_rate': float(1 - automation_rate),
            'performance_gain': float(performance_gain),
            'ai_cases': int(ai_cases.sum()),
            'human_cases': int(human_cases.sum())
        }
        
        return metrics
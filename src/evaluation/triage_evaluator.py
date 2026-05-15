"""
Comprehensive triage system evaluator.

End-to-end evaluation of human-AI collaboration including strategy comparison,
performance analysis, and report generation.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TriageEvaluationResult:
    """Container for triage evaluation results."""
    
    strategy_name: str
    threshold: float
    deferral_rate: float
    automation_rate: float
    ai_accuracy: float
    human_accuracy: float
    system_accuracy: float
    system_reliability: float
    cost_total: float
    cost_ai_errors: float
    cost_human_reviews: float
    critical_deferral_rate: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'strategy': self.strategy_name,
            'threshold': self.threshold,
            'deferral_rate': self.deferral_rate,
            'automation_rate': self.automation_rate,
            'ai_accuracy': self.ai_accuracy,
            'human_accuracy': self.human_accuracy,
            'system_accuracy': self.system_accuracy,
            'system_reliability': self.system_reliability,
            'cost_total': self.cost_total,
            'cost_ai_errors': self.cost_ai_errors,
            'cost_human_reviews': self.cost_human_reviews,
            'critical_deferral_rate': self.critical_deferral_rate,
        }


class TriageEvaluator:
    """Comprehensive triage system evaluator."""
    
    def __init__(
        self,
        model_predictions: np.ndarray,
        model_uncertainties: np.ndarray,
        human_predictions: Optional[np.ndarray] = None,
        human_accuracy: float = 0.95,
        targets: Optional[np.ndarray] = None,
        critical_classes: Optional[List[int]] = None,
        ai_error_cost: float = 100.0,
        human_review_cost: float = 1.0,
        verbose: bool = True,
    ):
        """
        Initialize TriageEvaluator.
        
        Args:
            model_predictions: (n_samples,) Model class predictions
            model_uncertainties: (n_samples,) Uncertainty estimates
            human_predictions: (n_samples,) Human expert predictions (optional)
            human_accuracy: Overall human accuracy (0-1)
            targets: (n_samples,) Ground truth labels
            critical_classes: List of high-risk class indices (e.g., cancer)
            ai_error_cost: Cost of AI error (default 100)
            human_review_cost: Cost of human review (default 1)
            verbose: Whether to log messages
        """
        self.model_predictions = np.asarray(model_predictions)
        self.model_uncertainties = np.asarray(model_uncertainties)
        self.targets = np.asarray(targets) if targets is not None else None
        self.critical_classes = critical_classes or []
        self.ai_error_cost = ai_error_cost
        self.human_review_cost = human_review_cost
        self.verbose = verbose
        
        # Human predictions
        if human_predictions is not None:
            self.human_predictions = np.asarray(human_predictions)
        else:
            self.human_predictions = self.model_predictions.copy()
        
        self.human_accuracy = human_accuracy
        
        if len(self.model_predictions) != len(self.model_uncertainties):
            raise ValueError("predictions and uncertainties must have same length")
    
    def evaluate_threshold(
        self,
        threshold: float,
        strategy_name: str = "threshold",
    ) -> TriageEvaluationResult:
        """
        Evaluate system at specific uncertainty threshold.
        
        Samples with uncertainty > threshold are deferred to human.
        
        Args:
            threshold: Uncertainty threshold for deferral
            strategy_name: Name of deferral strategy
        
        Returns:
            TriageEvaluationResult with performance metrics
        """
        defer_mask = self.model_uncertainties > threshold
        deferral_rate = defer_mask.mean()
        automation_rate = 1 - deferral_rate
        
        # AI predictions on non-deferred samples
        ai_predictions = self.model_predictions.copy()
        final_predictions = ai_predictions.copy()
        
        # Human handles deferred samples
        final_predictions[defer_mask] = self.human_predictions[defer_mask]
        
        # Compute accuracies if targets available
        if self.targets is not None:
            # Accuracy of AI on samples it handles
            ai_mask = ~defer_mask
            if ai_mask.sum() > 0:
                ai_accuracy = (ai_predictions[ai_mask] == self.targets[ai_mask]).mean()
            else:
                ai_accuracy = np.nan
            
            # Accuracy of human on deferred samples
            if defer_mask.sum() > 0:
                human_accuracy = (self.human_predictions[defer_mask] == 
                                self.targets[defer_mask]).mean()
            else:
                human_accuracy = np.nan
            
            # System accuracy (hybrid)
            system_accuracy = (final_predictions == self.targets).mean()
            
            # System reliability: does it defer errors?
            errors = (ai_predictions != self.targets).astype(int)
            deferred_errors = errors[defer_mask].sum()
            total_errors = errors.sum()
            system_reliability = 1.0 - (deferred_errors / total_errors) if total_errors > 0 else np.nan
        else:
            ai_accuracy = np.nan
            human_accuracy = np.nan
            system_accuracy = np.nan
            system_reliability = np.nan
        
        # Cost analysis
        if self.targets is not None:
            ai_errors = (ai_predictions[~defer_mask] != self.targets[~defer_mask]).sum()
        else:
            ai_errors = 0
        
        cost_ai_errors = ai_errors * self.ai_error_cost
        cost_human_reviews = defer_mask.sum() * self.human_review_cost
        cost_total = cost_ai_errors + cost_human_reviews
        
        # Critical case deferral rate
        if len(self.critical_classes) > 0 and self.targets is not None:
            critical_mask = np.isin(self.targets, self.critical_classes)
            if critical_mask.sum() > 0:
                critical_deferral_rate = defer_mask[critical_mask].mean()
            else:
                critical_deferral_rate = np.nan
        else:
            critical_deferral_rate = np.nan
        
        return TriageEvaluationResult(
            strategy_name=strategy_name,
            threshold=threshold,
            deferral_rate=float(deferral_rate),
            automation_rate=float(automation_rate),
            ai_accuracy=float(ai_accuracy),
            human_accuracy=float(human_accuracy),
            system_accuracy=float(system_accuracy),
            system_reliability=float(system_reliability),
            cost_total=float(cost_total),
            cost_ai_errors=float(cost_ai_errors),
            cost_human_reviews=float(cost_human_reviews),
            critical_deferral_rate=float(critical_deferral_rate),
        )
    
    def sweep_thresholds(
        self,
        num_thresholds: int = 20,
        strategy_name: str = "threshold",
    ) -> List[TriageEvaluationResult]:
        """
        Evaluate system across range of uncertainty thresholds.
        
        Args:
            num_thresholds: Number of thresholds to evaluate
            strategy_name: Name of strategy
        
        Returns:
            List of evaluation results for each threshold
        """
        thresholds = np.linspace(
            self.model_uncertainties.min(),
            self.model_uncertainties.max(),
            num_thresholds,
        )
        
        results = []
        for threshold in thresholds:
            result = self.evaluate_threshold(threshold, strategy_name)
            results.append(result)
        
        if self.verbose:
            logger.info(f"Evaluated {len(results)} thresholds")
        
        return results
    
    def compare_strategies(
        self,
        strategies: Dict[str, Tuple[np.ndarray, str]],
        num_points: int = 15,
    ) -> Dict[str, List[TriageEvaluationResult]]:
        """
        Compare multiple deferral strategies.
        
        Args:
            strategies: Dict mapping strategy name → (uncertainty_estimates, strategy_type)
            num_points: Points per strategy
        
        Returns:
            Dict mapping strategy name → list of evaluation results
        
        Example:
            >>> comparison = evaluator.compare_strategies({
            ...     'entropy': (entropy_uncertainties, 'threshold'),
            ...     'confidence': (1 - confidence_probs, 'threshold'),
            ...     'ensemble': (ensemble_std, 'budget'),
            ... })
        """
        all_results = {}
        
        for strategy_name, (uncertainties, strategy_type) in strategies.items():
            # Temporarily replace uncertainties
            original_unc = self.model_uncertainties.copy()
            self.model_uncertainties = np.asarray(uncertainties)
            
            # Evaluate this strategy
            results = self.sweep_thresholds(num_points, strategy_name)
            all_results[strategy_name] = results
            
            # Restore original
            self.model_uncertainties = original_unc
            
            if self.verbose:
                logger.info(f"Evaluated strategy: {strategy_name}")
        
        return all_results
    
    def find_optimal_threshold(
        self,
        objective: str = "system_accuracy",
    ) -> Tuple[float, float]:
        """
        Find uncertainty threshold that optimizes objective.
        
        Args:
            objective: One of 'system_accuracy', 'cost_total', 'critical_deferral_rate',
                      'automation_rate', 'f1_score'
        
        Returns:
            (optimal_threshold, optimal_value) tuple
        """
        results = self.sweep_thresholds()
        
        if objective == "system_accuracy":
            values = [r.system_accuracy for r in results]
            best_idx = np.nanargmax(values)
        elif objective == "cost_total":
            values = [r.cost_total for r in results]
            best_idx = np.nanargmin(values)
        elif objective == "critical_deferral_rate":
            values = [r.critical_deferral_rate for r in results]
            best_idx = np.nanargmax(values)
        elif objective == "automation_rate":
            values = [r.automation_rate for r in results]
            best_idx = np.nanargmax(values)
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        best_result = results[best_idx]
        
        if self.verbose:
            logger.info(
                f"Optimal {objective}: {values[best_idx]:.4f} "
                f"at threshold {best_result.threshold:.4f}"
            )
        
        return best_result.threshold, values[best_idx]
    
    def create_confusion_matrices(
        self,
        threshold: float,
    ) -> Dict[str, np.ndarray]:
        """
        Create confusion matrices for AI, human, and system.
        
        Args:
            threshold: Uncertainty threshold
        
        Returns:
            Dictionary with 'ai', 'human', 'system' confusion matrices
        """
        if self.targets is None:
            raise ValueError("targets required for confusion matrices")
        
        from sklearn.metrics import confusion_matrix
        
        defer_mask = self.model_uncertainties > threshold
        
        # AI confusion matrix (on non-deferred samples)
        ai_mask = ~defer_mask
        if ai_mask.sum() > 0:
            ai_cm = confusion_matrix(
                self.targets[ai_mask],
                self.model_predictions[ai_mask],
            )
        else:
            ai_cm = np.array([[]])
        
        # Human confusion matrix (on deferred samples)
        if defer_mask.sum() > 0:
            human_cm = confusion_matrix(
                self.targets[defer_mask],
                self.human_predictions[defer_mask],
            )
        else:
            human_cm = np.array([[]])
        
        # System confusion matrix (overall)
        final_predictions = self.model_predictions.copy()
        final_predictions[defer_mask] = self.human_predictions[defer_mask]
        system_cm = confusion_matrix(self.targets, final_predictions)
        
        return {
            'ai': ai_cm,
            'human': human_cm,
            'system': system_cm,
        }
    
    def generate_performance_report(
        self,
        threshold: float,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate text report of system performance.
        
        Args:
            threshold: Uncertainty threshold to report
            output_path: Optional path to save report
        
        Returns:
            Report text
        """
        result = self.evaluate_threshold(threshold)
        
        report = f"""
TRIAGE SYSTEM EVALUATION REPORT
================================

Strategy: {result.strategy_name}
Uncertainty Threshold: {result.threshold:.4f}

DEFERRAL METRICS
----------------
Deferral Rate: {result.deferral_rate:.1%}
Automation Rate: {result.automation_rate:.1%}

ACCURACY METRICS
----------------
AI Accuracy (non-deferred): {result.ai_accuracy:.3f}
Human Accuracy (deferred): {result.human_accuracy:.3f}
System Accuracy (hybrid): {result.system_accuracy:.3f}
System Reliability: {result.system_reliability:.3f}

COST ANALYSIS
-------------
AI Error Cost: ${result.cost_ai_errors:,.2f}
Human Review Cost: ${result.cost_human_reviews:,.2f}
Total Cost: ${result.cost_total:,.2f}

CRITICAL CASES
--------------
Critical Deferral Rate: {result.critical_deferral_rate:.1%}
"""
        
        if output_path is not None:
            Path(output_path).write_text(report)
            if self.verbose:
                logger.info(f"Report saved to {output_path}")
        
        return report

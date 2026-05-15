"""
Deferral strategies for human-AI collaboration.

Implements different strategies for deciding which samples to defer to human experts
based on model uncertainty and performance characteristics.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DeferralStrategy(ABC):
    """Abstract base class for deferral strategies."""
    
    @abstractmethod
    def make_decisions(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
    ) -> np.ndarray:
        """
        Make deferral decisions for samples.
        
        Args:
            predictions: Model predictions (N, num_classes)
            uncertainties: Uncertainty estimates (N,)
        
        Returns:
            Binary array (N,) indicating defer decisions (1=defer, 0=use AI prediction)
        """
        pass
    
    @abstractmethod
    def optimize(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        labels: np.ndarray,
        objective: str = "maximize_accuracy",
    ) -> None:
        """
        Optimize strategy parameters on validation set.
        
        Args:
            predictions: Model predictions (N, num_classes)
            uncertainties: Uncertainty estimates (N,)
            labels: Ground truth labels (N,)
            objective: Optimization objective
        """
        pass


class ThresholdStrategy(DeferralStrategy):
    """
    Simple uncertainty threshold-based deferral.
    
    Defers samples with uncertainty above a threshold to human experts.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        uncertainty_type: str = "entropy",
    ):
        """
        Initialize ThresholdStrategy.
        
        Args:
            threshold: Uncertainty threshold for deferral
            uncertainty_type: Type of uncertainty metric
        """
        self.threshold = threshold
        self.uncertainty_type = uncertainty_type
    
    def make_decisions(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
    ) -> np.ndarray:
        """Make deferral decisions based on uncertainty threshold."""
        return (uncertainties > self.threshold).astype(np.int32)
    
    def optimize(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        labels: np.ndarray,
        objective: str = "maximize_accuracy",
    ) -> None:
        """
        Find optimal threshold that maximizes accuracy on validation set.
        
        Args:
            predictions: Model predictions (N, num_classes)
            uncertainties: Uncertainty estimates (N,)
            labels: Ground truth labels (N,)
            objective: Optimization objective
        """
        pred_labels = np.argmax(predictions, axis=1)
        correctness = (pred_labels == labels).astype(np.float32)
        
        # Try all percentiles as thresholds
        thresholds = np.percentile(uncertainties, np.arange(0, 101, 1))
        best_threshold = self.threshold
        best_metric = float("-inf")
        
        for threshold in thresholds:
            defer_mask = (uncertainties > threshold).astype(np.int32)
            
            if objective == "maximize_accuracy":
                # Assume human has perfect accuracy
                metric = correctness[defer_mask == 0].mean()
            elif objective == "coverage":
                # Maximize coverage while maintaining high accuracy
                coverage = (defer_mask == 0).mean()
                accuracy = correctness[defer_mask == 0].mean() if (defer_mask == 0).sum() > 0 else 0
                metric = coverage * accuracy
            else:
                raise ValueError(f"Unknown objective: {objective}")
            
            if metric > best_metric:
                best_metric = metric
                best_threshold = threshold
        
        self.threshold = best_threshold
        logger.info(f"Optimized threshold: {self.threshold:.4f}")


class BudgetConstrainedStrategy(DeferralStrategy):
    """
    Defer top-k most uncertain samples within a budget.
    
    Useful when human review capacity is limited.
    """
    
    def __init__(self, defer_budget: float = 0.1):
        """
        Initialize BudgetConstrainedStrategy.
        
        Args:
            defer_budget: Fraction of samples to defer (0.0 to 1.0)
        """
        self.defer_budget = defer_budget
    
    def make_decisions(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
    ) -> np.ndarray:
        """Defer top-k most uncertain samples."""
        num_defer = max(1, int(len(uncertainties) * self.defer_budget))
        threshold = np.partition(uncertainties, -num_defer)[-num_defer]
        return (uncertainties >= threshold).astype(np.int32)
    
    def optimize(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        labels: np.ndarray,
        objective: str = "maximize_accuracy",
    ) -> None:
        """
        Find optimal budget that maximizes system accuracy.
        
        Args:
            predictions: Model predictions (N, num_classes)
            uncertainties: Uncertainty estimates (N,)
            labels: Ground truth labels (N,)
            objective: Optimization objective
        """
        pred_labels = np.argmax(predictions, axis=1)
        correctness = (pred_labels == labels).astype(np.float32)
        
        best_budget = self.defer_budget
        best_metric = float("-inf")
        
        for budget in np.arange(0.0, 1.01, 0.05):
            self.defer_budget = budget
            defer_mask = self.make_decisions(predictions, uncertainties)
            
            if (defer_mask == 0).sum() == 0:
                continue
            
            if objective == "maximize_accuracy":
                # Assume human has perfect accuracy
                metric = correctness[defer_mask == 0].mean()
            else:
                raise ValueError(f"Unknown objective: {objective}")
            
            if metric > best_metric:
                best_metric = metric
                best_budget = budget
        
        self.defer_budget = best_budget
        logger.info(f"Optimized deferral budget: {self.defer_budget:.4f}")


class CostSensitiveStrategy(DeferralStrategy):
    """
    Defer based on cost matrix considering misclassification costs.
    
    Accounts for different costs of AI errors vs human review costs.
    """
    
    def __init__(
        self,
        cost_matrix: Optional[np.ndarray] = None,
        human_cost: float = 1.0,
        ai_threshold: float = 0.5,
    ):
        """
        Initialize CostSensitiveStrategy.
        
        Args:
            cost_matrix: Misclassification cost matrix (num_classes, num_classes)
            human_cost: Cost of human review
            ai_threshold: Confidence threshold for AI decisions
        """
        self.cost_matrix = cost_matrix
        self.human_cost = human_cost
        self.ai_threshold = ai_threshold
    
    def make_decisions(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
    ) -> np.ndarray:
        """Make deferral decisions considering expected costs."""
        confidences = np.max(predictions, axis=1)
        
        # Defer if expected AI cost is higher than human cost
        expected_ai_cost = (1 - confidences) * np.mean(self.cost_matrix or 1.0)
        should_defer = expected_ai_cost > self.human_cost
        
        return should_defer.astype(np.int32)
    
    def optimize(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        labels: np.ndarray,
        objective: str = "minimize_cost",
    ) -> None:
        """
        Optimize cost-sensitive parameters.
        
        Args:
            predictions: Model predictions (N, num_classes)
            uncertainties: Uncertainty estimates (N,)
            labels: Ground truth labels (N,)
            objective: Optimization objective
        """
        pred_labels = np.argmax(predictions, axis=1)
        
        # Try different thresholds
        best_threshold = self.ai_threshold
        best_cost = float("inf")
        
        for threshold in np.arange(0.1, 1.0, 0.1):
            self.ai_threshold = threshold
            defer_mask = self.make_decisions(predictions, uncertainties)
            
            # Compute total cost
            ai_errors = (pred_labels[defer_mask == 0] != labels[defer_mask == 0]).sum()
            defer_cost = (defer_mask == 1).sum() * self.human_cost
            
            total_cost = ai_errors * np.mean(self.cost_matrix or 1.0) + defer_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_threshold = threshold
        
        self.ai_threshold = best_threshold
        logger.info(f"Optimized AI confidence threshold: {self.ai_threshold:.4f}")


class ConfidenceBasedStrategy(DeferralStrategy):
    """
    Defer samples with low model confidence.
    
    Simple strategy based on max softmax probability.
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize ConfidenceBasedStrategy.
        
        Args:
            confidence_threshold: Confidence threshold for deferral
        """
        self.confidence_threshold = confidence_threshold
    
    def make_decisions(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
    ) -> np.ndarray:
        """Defer low-confidence predictions."""
        confidences = np.max(predictions, axis=1)
        return (confidences < self.confidence_threshold).astype(np.int32)
    
    def optimize(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        labels: np.ndarray,
        objective: str = "maximize_accuracy",
    ) -> None:
        """Find optimal confidence threshold."""
        pred_labels = np.argmax(predictions, axis=1)
        correctness = (pred_labels == labels).astype(np.float32)
        confidences = np.max(predictions, axis=1)
        
        best_threshold = self.confidence_threshold
        best_metric = float("-inf")
        
        for threshold in np.arange(0.1, 1.0, 0.05):
            defer_mask = (confidences < threshold).astype(np.int32)
            
            if (defer_mask == 0).sum() == 0:
                continue
            
            if objective == "maximize_accuracy":
                metric = correctness[defer_mask == 0].mean()
            else:
                raise ValueError(f"Unknown objective: {objective}")
            
            if metric > best_metric:
                best_metric = metric
                best_threshold = threshold
        
        self.confidence_threshold = best_threshold
        logger.info(f"Optimized confidence threshold: {self.confidence_threshold:.4f}")


class AdaptiveStrategy(DeferralStrategy):
    """
    Dynamically adjust deferral threshold based on performance feedback.
    
    Adapts threshold as system receives feedback on human decisions.
    """
    
    def __init__(
        self,
        initial_threshold: float = 0.5,
        learning_rate: float = 0.01,
    ):
        """
        Initialize AdaptiveStrategy.
        
        Args:
            initial_threshold: Initial deferral threshold
            learning_rate: Learning rate for threshold adaptation
        """
        self.threshold = initial_threshold
        self.learning_rate = learning_rate
        self.performance_history = []
    
    def make_decisions(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
    ) -> np.ndarray:
        """Make deferral decisions with current threshold."""
        return (uncertainties > self.threshold).astype(np.int32)
    
    def optimize(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        labels: np.ndarray,
        objective: str = "maximize_accuracy",
    ) -> None:
        """Initialize threshold optimization."""
        strategy = ThresholdStrategy(threshold=self.threshold)
        strategy.optimize(predictions, uncertainties, labels, objective)
        self.threshold = strategy.threshold
    
    def update_with_feedback(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        labels: np.ndarray,
        human_decisions: np.ndarray,
        human_accuracy: float,
    ) -> None:
        """
        Update threshold based on feedback.
        
        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            labels: Ground truth labels
            human_decisions: Human's decisions on deferred samples
            human_accuracy: Accuracy of human decisions
        """
        pred_labels = np.argmax(predictions, axis=1)
        defer_mask = self.make_decisions(predictions, uncertainties)
        
        # Compare AI and human performance
        ai_correct = (pred_labels[defer_mask == 0] == labels[defer_mask == 0]).mean()
        
        # Adjust threshold based on relative performance
        if human_accuracy > ai_correct:
            # Human is more accurate, increase deferral
            self.threshold *= (1 - self.learning_rate)
        else:
            # AI is more accurate, decrease deferral
            self.threshold *= (1 + self.learning_rate)
        
        self.threshold = np.clip(self.threshold, 0.01, 0.99)
        
        logger.info(
            f"Updated threshold to {self.threshold:.4f} "
            f"(AI acc: {ai_correct:.4f}, Human acc: {human_accuracy:.4f})"
        )

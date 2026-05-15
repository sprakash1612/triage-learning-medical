"""
Simulated human expert for triage system evaluation.

Allows evaluation of triage systems without requiring actual human annotators.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class HumanExpert:
    """
    Simulated human expert for triage system evaluation.
    
    Simulates human decision-making with configurable accuracy, confidence,
    and inter-rater variability characteristics.
    """
    
    def __init__(
        self,
        overall_accuracy: float = 0.95,
        class_specific_accuracy: Optional[Dict[int, float]] = None,
        confidence_calibration: float = 1.0,
    ):
        """
        Initialize HumanExpert simulator.
        
        Args:
            overall_accuracy: Overall human accuracy (0 to 1)
            class_specific_accuracy: Per-class accuracy (e.g., {0: 0.90, 1: 0.98})
            confidence_calibration: How well calibrated human confidence is (1.0 = perfect)
        """
        self.overall_accuracy = overall_accuracy
        self.class_specific_accuracy = class_specific_accuracy or {}
        self.confidence_calibration = confidence_calibration
        
        logger.info(
            f"Initialized HumanExpert with overall_accuracy={overall_accuracy}, "
            f"num_class_specific={len(self.class_specific_accuracy)}"
        )
    
    def generate_human_predictions(
        self,
        labels: np.ndarray,
        num_classes: int,
        use_class_specific_accuracy: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate simulated human predictions.
        
        Args:
            labels: Ground truth labels (N,)
            num_classes: Number of classes
            use_class_specific_accuracy: Whether to use per-class accuracy
        
        Returns:
            Tuple of (predictions, confidence) where:
            - predictions: Predicted labels (N,)
            - confidence: Prediction confidence (N,)
        """
        N = len(labels)
        predictions = labels.copy()
        confidence = np.ones(N) * (self.overall_accuracy ** 0.5)
        
        for i in range(N):
            if use_class_specific_accuracy and labels[i] in self.class_specific_accuracy:
                accuracy = self.class_specific_accuracy[labels[i]]
            else:
                accuracy = self.overall_accuracy
            
            # Make error with probability (1 - accuracy)
            if np.random.rand() > accuracy:
                # Choose random incorrect prediction
                wrong_classes = [c for c in range(num_classes) if c != labels[i]]
                predictions[i] = np.random.choice(wrong_classes)
                confidence[i] = accuracy * self.confidence_calibration
            else:
                confidence[i] = min(1.0, accuracy * self.confidence_calibration)
        
        return predictions, confidence
    
    def confidence_based_errors(
        self,
        labels: np.ndarray,
        num_classes: int,
        sample_confidence: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make errors that correlate with sample difficulty (inverse of sample confidence).
        
        More difficult samples have higher human error rates.
        
        Args:
            labels: Ground truth labels (N,)
            num_classes: Number of classes
            sample_confidence: Model confidence for each sample (N,). 
                              If None, uses uniform confidence.
        
        Returns:
            Tuple of (predictions, confidence)
        """
        N = len(labels)
        predictions = labels.copy()
        confidence = np.ones(N)
        
        if sample_confidence is None:
            sample_confidence = np.ones(N) * 0.5
        
        for i in range(N):
            # Human finds difficult samples harder
            difficulty = 1 - sample_confidence[i]
            human_error_rate = (1 - self.overall_accuracy) + difficulty * 0.3
            human_error_rate = np.clip(human_error_rate, 0, 1)
            
            if np.random.rand() < human_error_rate:
                # Make error
                wrong_classes = [c for c in range(num_classes) if c != labels[i]]
                predictions[i] = np.random.choice(wrong_classes)
                confidence[i] = 1 - human_error_rate
            else:
                confidence[i] = 1 - human_error_rate
        
        return predictions, confidence
    
    def class_specific_accuracy_errors(
        self,
        labels: np.ndarray,
        num_classes: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make errors with class-specific accuracy rates.
        
        Different disease classes may have different human accuracy.
        
        Args:
            labels: Ground truth labels (N,)
            num_classes: Number of classes
        
        Returns:
            Tuple of (predictions, confidence)
        """
        N = len(labels)
        predictions = labels.copy()
        confidence = np.ones(N)
        
        for i in range(N):
            class_idx = labels[i]
            accuracy = self.class_specific_accuracy.get(class_idx, self.overall_accuracy)
            
            if np.random.rand() > accuracy:
                wrong_classes = [c for c in range(num_classes) if c != class_idx]
                predictions[i] = np.random.choice(wrong_classes)
                confidence[i] = accuracy * self.confidence_calibration
            else:
                confidence[i] = min(1.0, accuracy * self.confidence_calibration)
        
        return predictions, confidence
    
    def simulate_inter_rater_variability(
        self,
        labels: np.ndarray,
        num_classes: int,
        num_raters: int = 3,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Simulate multiple human raters with slight variations.
        
        Useful for analyzing consensus and disagreement patterns.
        
        Args:
            labels: Ground truth labels (N,)
            num_classes: Number of classes
            num_raters: Number of simulated raters
        
        Returns:
            List of (predictions, confidence) for each rater
        """
        rater_results = []
        
        for rater_idx in range(num_raters):
            # Vary accuracy slightly for each rater
            rater_accuracy = self.overall_accuracy * np.random.uniform(0.9, 1.1)
            rater_accuracy = np.clip(rater_accuracy, 0.6, 1.0)
            
            # Create rater with adjusted accuracy
            rater = HumanExpert(
                overall_accuracy=rater_accuracy,
                class_specific_accuracy=self.class_specific_accuracy,
                confidence_calibration=self.confidence_calibration,
            )
            
            predictions, confidence = rater.generate_human_predictions(
                labels, num_classes
            )
            rater_results.append((predictions, confidence))
        
        logger.info(f"Simulated {num_raters} human raters")
        
        return rater_results
    
    def get_human_confidence(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """
        Get simulated human confidence levels.
        
        Confidence correlates with correctness (higher for correct predictions).
        
        Args:
            predictions: Human predictions (N,)
            labels: Ground truth labels (N,)
        
        Returns:
            Confidence levels (N,)
        """
        correctness = (predictions == labels).astype(np.float32)
        
        # Confidence is higher for correct predictions
        base_confidence = np.full_like(correctness, self.overall_accuracy)
        confidence = base_confidence * correctness + (1 - base_confidence) * (1 - correctness)
        
        # Add some noise
        noise = np.random.normal(0, 0.05, len(labels))
        confidence = np.clip(confidence + noise, 0, 1)
        
        return confidence
    
    def get_confusion_matrix(
        self,
        labels: np.ndarray,
        num_classes: int,
        num_samples: int = 1000,
    ) -> np.ndarray:
        """
        Generate a simulated confusion matrix for the human expert.
        
        Args:
            labels: Ground truth labels (N,)
            num_classes: Number of classes
            num_samples: Number of samples to use for estimation
        
        Returns:
            Confusion matrix of shape (num_classes, num_classes)
        """
        # Sample from provided labels
        if len(labels) >= num_samples:
            sample_indices = np.random.choice(len(labels), num_samples, replace=False)
            sample_labels = labels[sample_indices]
        else:
            sample_labels = labels
        
        predictions, _ = self.generate_human_predictions(
            sample_labels, num_classes, use_class_specific_accuracy=True
        )
        
        # Compute confusion matrix
        confusion = np.zeros((num_classes, num_classes))
        for true_label, pred_label in zip(sample_labels, predictions):
            confusion[true_label, pred_label] += 1
        
        # Normalize by class
        confusion = confusion / confusion.sum(axis=1, keepdims=True)
        
        return confusion
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HumanExpert(\n"
            f"  overall_accuracy={self.overall_accuracy:.4f},\n"
            f"  num_class_specific={len(self.class_specific_accuracy)},\n"
            f"  confidence_calibration={self.confidence_calibration:.2f},\n"
            f")"
        )

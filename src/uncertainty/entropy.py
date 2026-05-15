"""
Entropy-based uncertainty metrics
"""

import numpy as np
import torch
from typing import Tuple


def compute_predictive_entropy(probabilities: np.ndarray) -> np.ndarray:
    """
    Compute predictive entropy
    
    Args:
        probabilities: Predicted probabilities (N, num_classes)
    
    Returns:
        entropy: Entropy values (N,)
    """
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
    return entropy


def compute_confidence(probabilities: np.ndarray) -> np.ndarray:
    """
    Compute prediction confidence (max probability)
    
    Args:
        probabilities: Predicted probabilities (N, num_classes)
    
    Returns:
        confidence: Confidence scores (N,)
    """
    return np.max(probabilities, axis=1)


def compute_margin(probabilities: np.ndarray) -> np.ndarray:
    """
    Compute margin (difference between top-2 probabilities)
    
    Args:
        probabilities: Predicted probabilities (N, num_classes)
    
    Returns:
        margin: Margin values (N,)
    """
    sorted_probs = np.sort(probabilities, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    return margin


def compute_variation_ratio(probabilities: np.ndarray) -> np.ndarray:
    """
    Compute variation ratio (1 - max probability)
    
    Args:
        probabilities: Predicted probabilities (N, num_classes)
    
    Returns:
        variation_ratio: Variation ratio values (N,)
    """
    max_prob = np.max(probabilities, axis=1)
    return 1 - max_prob
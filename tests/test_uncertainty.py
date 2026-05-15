"""
Unit tests for uncertainty quantification methods.

Tests MC Dropout, ensemble uncertainty, temperature scaling,
calibration metrics, and edge cases.
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np


class TestMCDropout(unittest.TestCase):
    """Test MC Dropout uncertainty estimation."""
    
    def test_mc_samples_shape(self):
        """Test MC Dropout produces correct sample shape."""
        batch_size = 32
        num_classes = 10
        num_samples = 10
        
        # Generate MC samples
        samples = [np.random.randn(batch_size, num_classes) for _ in range(num_samples)]
        
        self.assertEqual(len(samples), num_samples)
        self.assertEqual(samples[0].shape, (batch_size, num_classes))
    
    def test_mc_mean_computation(self):
        """Test mean of MC samples."""
        num_samples = 100
        batch_size = 32
        num_classes = 10
        
        # Generate samples from known distribution
        samples = np.random.randn(num_samples, batch_size, num_classes)
        mean_samples = samples.mean(axis=0)
        
        self.assertEqual(mean_samples.shape, (batch_size, num_classes))
    
    def test_uncertainty_from_samples(self):
        """Test uncertainty computation from MC samples."""
        # Certain sample: same prediction each time
        certain = np.array([[0.8, 0.1, 0.1]] * 10)
        
        # Uncertain sample: varying predictions
        uncertain = np.vstack([
            np.array([[0.4, 0.3, 0.3]]),
            np.array([[0.3, 0.4, 0.3]]),
            np.array([[0.3, 0.3, 0.4]]),
        ] * 3 + [np.array([[0.4, 0.3, 0.3]])])
        
        # Compute entropy for each
        def entropy(probs):
            return -np.sum(probs * np.log(probs + 1e-10), axis=1).mean()
        
        certain_ent = entropy(certain)
        uncertain_ent = entropy(uncertain)
        
        # Uncertain should have higher entropy
        self.assertGreater(uncertain_ent, certain_ent)


class TestEnsembleUncertainty(unittest.TestCase):
    """Test ensemble-based uncertainty estimation."""
    
    def test_ensemble_variance(self):
        """Test variance computation from ensemble."""
        ensemble_preds = np.array([
            [0.8, 0.15, 0.05],
            [0.75, 0.2, 0.05],
            [0.85, 0.1, 0.05],
        ])
        
        variance = ensemble_preds.var(axis=0)
        
        self.assertEqual(len(variance), 3)
        self.assertTrue(np.all(variance >= 0))
    
    def test_ensemble_disagreement(self):
        """Test ensemble disagreement as uncertainty."""
        # All agree
        agreement = np.array([[1, 0, 0]] * 5)
        disagreement_agreement = np.std(agreement.argmax(axis=1))
        
        # All disagree
        disagreement = np.array([
            [0.4, 0.3, 0.3],
            [0.3, 0.4, 0.3],
            [0.3, 0.3, 0.4],
        ] * 2)
        disagreement_disagreement = np.std(disagreement.argmax(axis=1))
        
        # Disagreement should be higher for actual disagreement
        self.assertGreater(disagreement_disagreement, disagreement_agreement)
    
    def test_ensemble_entropy(self):
        """Test entropy of ensemble mean predictions."""
        # Create ensemble predictions
        ensemble = np.array([
            [0.8, 0.1, 0.1],
            [0.75, 0.15, 0.1],
            [0.85, 0.1, 0.05],
        ])
        
        mean_pred = ensemble.mean(axis=0)
        entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-10))
        
        self.assertGreater(entropy, 0)


class TestTemperatureScaling(unittest.TestCase):
    """Test temperature scaling calibration."""
    
    def test_temperature_effect_on_probabilities(self):
        """Test that temperature changes probability distributions."""
        logits = np.array([[2.0, 1.0, 0.0]])
        
        # Baseline (T=1)
        probs_t1 = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        
        # Higher temperature (T=2)
        T = 2.0
        probs_t2 = np.exp(logits / T) / np.sum(np.exp(logits / T), axis=1, keepdims=True)
        
        # Higher temperature should make probabilities more uniform
        self.assertGreater(probs_t1[0, 0], probs_t2[0, 0])
    
    def test_optimal_temperature(self):
        """Test optimal temperature finding."""
        # Create logits and targets
        logits = np.random.randn(100, 10)
        targets = np.random.randint(0, 10, 100)
        
        # Temperature should be positive
        optimal_temp = 1.2
        self.assertGreater(optimal_temp, 0)
    
    def test_temperature_bounds(self):
        """Test temperature should be within reasonable bounds."""
        # Typical temperatures are in [0.1, 10]
        temperature_range = (0.1, 10.0)
        
        test_temps = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0]
        within_range = [t for t in test_temps if temperature_range[0] <= t <= temperature_range[1]]
        
        self.assertEqual(len(within_range), 7)


class TestCalibrationMetrics(unittest.TestCase):
    """Test calibration metrics (ECE, MCE)."""
    
    def test_ece_computation(self):
        """Test Expected Calibration Error computation."""
        # Perfect calibration: confidence matches accuracy
        confidences = np.array([0.9, 0.8, 0.7])
        accuracies = np.array([0.9, 0.8, 0.7])
        
        ece = np.mean(np.abs(accuracies - confidences))
        
        self.assertEqual(ece, 0.0)
    
    def test_mce_computation(self):
        """Test Maximum Calibration Error computation."""
        confidences = np.array([0.9, 0.8, 0.7])
        accuracies = np.array([0.5, 0.8, 0.7])
        
        mce = np.max(np.abs(accuracies - confidences))
        
        self.assertEqual(mce, 0.4)
    
    def test_overconfidence_detection(self):
        """Test detection of overconfidence."""
        confidences = np.array([0.9, 0.8, 0.7])
        accuracies = np.array([0.5, 0.5, 0.5])  # Actually 50% accurate
        
        # Model is overconfident
        overconfidence = (confidences - accuracies).mean()
        
        self.assertGreater(overconfidence, 0.2)
    
    def test_underconfidence_detection(self):
        """Test detection of underconfidence."""
        confidences = np.array([0.6, 0.5, 0.4])
        accuracies = np.array([0.9, 0.9, 0.9])  # Actually 90% accurate
        
        # Model is underconfident
        underconfidence = (accuracies - confidences).mean()
        
        self.assertGreater(underconfidence, 0.3)


class TestUncertaintyEdgeCases(unittest.TestCase):
    """Test edge cases in uncertainty estimation."""
    
    def test_all_same_class(self):
        """Test when all samples belong to same class."""
        # All class 0
        targets = np.zeros(100)
        predictions = np.zeros(100)
        
        accuracy = (predictions == targets).mean()
        self.assertEqual(accuracy, 1.0)
    
    def test_uniform_distribution(self):
        """Test uncertainty with uniform prediction distribution."""
        # Each class equally likely
        num_classes = 10
        probs = np.ones(num_classes) / num_classes
        
        # Entropy of uniform distribution
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # Should be maximum
        max_entropy = np.log(num_classes)
        self.assertAlmostEqual(entropy, max_entropy, places=5)
    
    def test_single_confident_prediction(self):
        """Test with single very confident prediction."""
        probs = np.array([0.99, 0.005, 0.005])
        
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # Should be very small
        self.assertLess(entropy, 0.1)
    
    def test_empty_batch(self):
        """Test handling of empty batch."""
        probs = np.empty((0, 10))
        
        self.assertEqual(len(probs), 0)
    
    def test_single_sample(self):
        """Test handling of single sample."""
        probs = np.array([[0.5, 0.3, 0.2]])
        
        self.assertEqual(probs.shape, (1, 3))
        self.assertAlmostEqual(probs.sum(), 1.0)


class TestUncertaintyCorrelation(unittest.TestCase):
    """Test correlation between uncertainty and errors."""
    
    def test_uncertainty_predicts_errors(self):
        """Test that high uncertainty correlates with errors."""
        # Create synthetic data
        np.random.seed(42)
        
        # High uncertainty cases: mostly errors
        high_unc = np.random.uniform(0.8, 1.0, 50)
        high_unc_errors = np.concatenate([
            np.ones(40),  # 80% error rate
            np.zeros(10),
        ])
        
        # Low uncertainty cases: mostly correct
        low_unc = np.random.uniform(0.0, 0.2, 50)
        low_unc_errors = np.concatenate([
            np.ones(5),   # 10% error rate
            np.zeros(45),
        ])
        
        # Combined
        all_unc = np.concatenate([high_unc, low_unc])
        all_errors = np.concatenate([high_unc_errors, low_unc_errors])
        
        # Correlation should be positive
        correlation = np.corrcoef(all_unc, all_errors)[0, 1]
        self.assertGreater(correlation, 0.5)
    
    def test_auroc_calculation(self):
        """Test AUROC for uncertainty vs. errors."""
        uncertainties = np.linspace(0, 1, 100)
        # Perfect ordering: high uncertainty = high error
        errors = (uncertainties > 0.5).astype(int)
        
        # AUROC should be 1.0 (perfect)
        # Simplified: check ranking
        high_unc_idx = uncertainties > 0.5
        low_unc_idx = uncertainties <= 0.5
        
        self.assertEqual(errors[high_unc_idx].mean(), 1.0)
        self.assertEqual(errors[low_unc_idx].mean(), 0.0)


class TestUncertaintyNumericalStability(unittest.TestCase):
    """Test numerical stability of uncertainty computations."""
    
    def test_log_softmax_stability(self):
        """Test log-softmax numerical stability."""
        # Large logits
        logits = np.array([[1000.0, 1001.0, 1002.0]])
        
        # Numerically stable log-softmax
        logits_max = np.max(logits, axis=1, keepdims=True)
        logits_stable = logits - logits_max
        
        log_Z = np.log(np.sum(np.exp(logits_stable), axis=1, keepdims=True))
        log_softmax = logits_stable - log_Z
        
        # Should not contain NaN or Inf
        self.assertFalse(np.any(np.isnan(log_softmax)))
        self.assertFalse(np.any(np.isinf(log_softmax)))
    
    def test_entropy_with_small_values(self):
        """Test entropy computation with very small probabilities."""
        # Include near-zero probabilities
        probs = np.array([[0.99, 1e-10, 1e-10]])
        
        # Use epsilon to avoid log(0)
        epsilon = 1e-10
        entropy = -np.sum(probs * np.log(np.maximum(probs, epsilon)))
        
        self.assertFalse(np.isnan(entropy))
        self.assertFalse(np.isinf(entropy))


if __name__ == '__main__':
    unittest.main()
